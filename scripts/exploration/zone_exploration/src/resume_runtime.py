"""Mission resume adapters; the exploration node owns all action/safety calls.

The factory hooks wrap only startup, checkpointing and reporting. No old Nav2
Goal, TF, costmap, ROS timestamp, or worker Future is reinstated.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import copy
import fcntl
import hashlib
import json
import os
import time
import numpy as np
from checkpoint import (VERSION, SCHEMA, plain, encode, sha, atomic_bytes,
    load_checkpoint, save_checkpoint, project_context, assert_context,
    coverage_payload, validate_coverage, restore_coverage, mask_from_runs,
    compatible_live_coverage, enforce_area_policy)
from exploration_results import BlockageRegistry
from home_state import HomeCheckpoint
from blocked_policy import normalize_points
from resume_selection import (recover_terminal_history,
    apply_terminal_history, current_pose_decision)
from zone_lifecycle import (ZoneLifecycle, completion_facts, legacy_upgrade,
    live_review_verdict, review_terminal, unresolved_zones)

NODE_COUNTERS = ('sent_goals', 'plan_attempts', 'succeeded_goals', 'aborted_goals', 'coverage_cancels')
VISIT_FIELDS = ('entered', 'entry_goals', 'observation_goals', 'low_gain', 'retries', 'entry_arrival_retries')
BOOK_DICTS = ('deferral_history', 'outcomes', 'nav_abort_counts')
CONNECTION_DICTS = ('failed_entries', 'second_zone_preflights', 'second_zone_nav',
                    'second_zone_work', 'region_ledgers', 'local_reviews', 'local_review_history')


def root_path():
    return Path(os.environ.get('ZB_PROJECT_ROOT') or Path(__file__).resolve().parents[4]).expanduser().resolve()


def extend_parser(parser):
    parser.add_argument('--resume-from', type=Path, help='v1.8 checkpoint/LATEST.json; never implicitly choose latest session')
    parser.add_argument('--resume-confirm-scene', action='store_true',
                        help='Operator confirms unchanged live scene and no active old navigation Goal')
    return parser


def mission_policy(args):
    # Execution mode/output paths differ on restart; geometry/selection policy may not.
    ignored = {'execute','supervised','resume_from','resume_confirm_scene','output_dir',
               'max_goals','max_plan_attempts','zone_ids','zones','reference','entrance_overlay_file'}
    return plain({k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
                  if k not in ignored})


def prepare_mission_args(args, bundle):
    if os.environ.get('DN012_V18_ENABLE') != '1' and not getattr(args, 'resume_from', None):
        return
    enforce_area_policy()
    source = getattr(args, 'resume_from', None)
    if source is None:
        return
    if not getattr(args, 'resume_confirm_scene', False):
        raise ValueError('RESUME_REQUIRES_EXPLICIT --resume-confirm-scene')
    payload, _, _ = load_checkpoint(source)
    mission = payload.get('mission')
    if not mission:
        raise ValueError('MISSION_RESUME_REQUIRES_MISSION_CHECKPOINT')
    scope = mission.get('scope')
    if not scope or not set(scope) <= {z['id'] for z in bundle[0]['zones']}:
        raise ValueError('RESUME_INVALID_MISSION_SCOPE')
    if args.zone_ids is not None and set(args.zone_ids) != set(scope):
        raise ValueError('RESUME_SCOPE_CHANGED: use the saved scope or start an explicit new mission')
    args.zone_ids = list(scope)
    if abs(float(mission['threshold']) - float(args.threshold)) > 1e-9:
        raise ValueError('RESUME_MISSION_THRESHOLD_CHANGED')


def capture_mission(node, report):
    book = node.book
    records = {zid: {key: getattr(rec, key) for key in
               ('state', 'measured', 'views', 'low_gain_count', 'deferred_reason')}
               for zid, rec in book.records.items()}
    visit_states = {zid: {key: getattr(v, key) for key in VISIT_FIELDS}
                    for zid, v in book.visit_state.items()}
    out = dict(scope=list(node.zones_by_id), threshold=float(book.threshold),
               counters={key: int(getattr(node, key)) for key in NODE_COUNTERS},
               records=records, visit_states=visit_states,
               visits=copy.deepcopy(book.visits), rejected=copy.deepcopy(book.rejected),
               historical_rejected=copy.deepcopy(getattr(node,'_v18_historical_rejected',[])),
               failed_entries_history=copy.deepcopy(getattr(node,'_v18_failed_entries_history',{})),
               progress=int(book.progress),
               deferred_this_mission=sorted(getattr(book, 'deferred_this_mission', set())),
               source_state=report.get('state'), source_session=report.get('source_session_v18'),
               interrupted_goal=copy.deepcopy(report.get('current')),
               interrupted_goal_id=report.get('owned_goal_id'),
               original_summary=copy.deepcopy({k: report[k] for k in
                   ('zones','region_review','path_blockage_cancellations','operational_cancellations','reroute_memory_v1_4') if k in report}))
    for key in BOOK_DICTS:
        out[key] = copy.deepcopy(getattr(book, key, {}))
    runtime = node.connectivity
    memory = getattr(runtime, 'blockage_memory', None)
    reroute = getattr(getattr(node, 'path_blockage', None), 'reroute', None)
    out['detector_memories'] = dict(
        blockage=copy.deepcopy(memory.records) if memory is not None else [],
        reroute=copy.deepcopy(reroute.store.data['records']) if reroute is not None else [])
    if runtime is not None:
        out['connectivity'] = {key: copy.deepcopy(getattr(runtime, key, {})) for key in CONNECTION_DICTS}
        out['connectivity'].update(
            primary_zone=getattr(runtime, 'primary_zone', None),
            local_second_zone=getattr(runtime, 'local_second_zone', None),
            stage=getattr(runtime, 'exploration_stage', 'PRIMARY'),
            local_reviewed=sorted(getattr(runtime, 'local_reviewed', set())),
            total_probes=int(getattr(runtime, 'total_probes', 0)))
        ledger = getattr(runtime, 'tasks', None)
        if ledger is not None:
            out['task_ledger'] = dict(tasks=ledger.public(getattr(runtime, 'region_last_inp', None)),
                inventoried_zones=sorted(ledger.inventoried_zones), small_by_zone=copy.deepcopy(ledger.small_by_zone),
                order=ledger.order, initialized=ledger.initialized)
    guard = node.mission_guard
    if guard is not None:
        out['guard'] = dict(settings=asdict(guard.s), records=copy.deepcopy(guard.records),
            zone_cancels=copy.deepcopy(guard.zone_cancels),
            ledgers={z: v.snapshot() for z, v in guard.ledgers.items()})
    out['pose_snapshot_v183'] = copy.deepcopy(getattr(node, '_v183_last_pose', None))
    out['source_execute'] = bool(report.get('execute', False))
    # One explicit state table is the only future completion authority.
    lifecycle = getattr(node, '_v187_lifecycle', None)
    if lifecycle is None:
        raise ValueError('V187_LIFECYCLE_NOT_INSTALLED')
    out['zone_lifecycle_v187'] = lifecycle.snapshot(node)
    facts = completion_facts(out)
    facts.update({z: 'PRESERVED_FINISHED_ZONE' for z in getattr(node, '_v183_finished_zones', ())})
    out['completion_v183'] = dict(finished_zones=sorted(facts), reasons=facts,
        lineage_reconciled=True, counters_preserved=True)
    out['completion_v184'] = dict(policy='COVERAGE_96_OR_LOCAL_REVIEW_FINISHED',
        completed_zones=sorted(facts), unfinished_zones=sorted(set(out['scope']) - set(facts)),
        unfinished_local_review_is_resumable=True, local_review_terminal_is_complete=True)
    out['phase_v185'] = dict(version='DN012_PHASE_CONTINUITY_V1_8_5',
        active_second_zone=getattr(runtime,'local_second_zone',None) if runtime else None,
        pending_second_zones=sorted(getattr(runtime,'_v185_pending_second',{})) if runtime else [],
        processed_second_zones=sorted(getattr(runtime,'_v185_processed_second',set())) if runtime else [],
        checkpoint_history_preserved=True)
    return plain(out)


def restore_mission(node, mission, inp, bundle):
    """Called once AFTER fresh inputs, stopped odometry and local-start checks."""
    if set(mission['scope']) != set(node.zones_by_id):
        raise ValueError('RESUME_SCOPE_MISMATCH')
    if any(getattr(node, key, None) is not None for key in ('nav_uuid', 'nav_handle', 'nav_send', 'plan_handle', 'plan_send', 'plan_result')):
        raise ValueError('RESUME_ACTION_ALREADY_ACTIVE')
    # Validate counters before mutating the live book.
    for key in NODE_COUNTERS:
        value = mission.get('counters', {}).get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError('RESUME_BAD_COUNTER:' + key)
    book = node.book
    by_zone = {z['id']: z for z in inp['status']['zones']}
    # Fail closed before touching the live mission book. Saved Zone outcomes are
    # scheduling history, not old TF/costmaps or a renewed physical cut proof.
    for zid in book.records:
        old = mission['records'].get(zid, {})
        now = by_zone[zid]
        fraction = now['observed_cells'] / now['reference_free_cells']
        if old.get('state') == 'OBSERVATION_SUFFICIENT' and fraction + 1e-9 < book.threshold:
            raise ValueError('RESUME_SAVED_COMPLETE_ZONE_LOST_COVERAGE:' + zid)
        if old.get('state') == 'PARTIAL_DEFERRED' and not (
                isinstance(old.get('deferred_reason'), str) and old['deferred_reason']):
            raise ValueError('RESUME_DEFERRED_REASON_MISSING:' + zid)
    book.session = inp['status']['session_id']
    for zid, rec in book.records.items():
        old = mission['records'].get(zid, {})
        # Preserve mission scheduling outcomes. Fresh tracker evidence, not the
        # saved timestamp, must reconfirm previously sufficient Coverage.
        now = by_zone[zid]
        rec.measured = now['observed_cells'] / now['reference_free_cells']
        rec.views = int(old.get('views', 0))
        rec.low_gain_count = int(old.get('low_gain_count', 0))
        old_state = old.get('state', 'UNOBSERVED')
        if old_state == 'OBSERVATION_SUFFICIENT':
            rec.state = 'THRESHOLD_PENDING'  # no Goal; fresh scans re-confirm
            rec.deferred_reason = None
        elif old_state == 'PARTIAL_DEFERRED':
            rec.state = 'PARTIAL_DEFERRED'
            rec.deferred_reason = old['deferred_reason']
        elif old_state == 'THRESHOLD_PENDING' and rec.measured + 1e-9 >= book.threshold:
            rec.state = 'THRESHOLD_PENDING'
            rec.deferred_reason = None
        elif old_state == 'OBSERVATION_LIMITED':
            # The old connectivity certificate may remain valid ONLY against the
            # same freshly received costmap fingerprint and recorded outcome.
            outcome = (mission.get('outcomes') or {}).get(zid) or {}
            evidence = outcome.get('evidence') or {}
            if (outcome.get('normal_termination') is True and
                    evidence.get('cost_key') == inp['cost']['key']):
                rec.state = 'OBSERVATION_LIMITED'
            else:
                rec.state = 'PARTIAL' if rec.measured else 'UNOBSERVED'
            rec.deferred_reason = None
        else:
            rec.state = 'PARTIAL' if rec.measured else 'UNOBSERVED'
            rec.deferred_reason = None
        for key, value in mission.get('visit_states', {}).get(zid, {}).items():
            if key in VISIT_FIELDS:
                setattr(book.visit_state[zid], key, value)
        # The historic "entered" flag is history, not proof of present location.
        book.visit_state[zid].deferred_key = None
        book.visit_state[zid].deferred_at = 0.
    book.visits = copy.deepcopy(mission.get('visits', []))
    node._v18_historical_rejected = copy.deepcopy(mission.get('historical_rejected',[])+mission.get('rejected',[]))
    book.rejected = []  # preserved as history, never revived as stale holds
    book.progress = int(mission.get('progress', 0))
    book.active_zone = book.locked_zone = None
    if hasattr(book, 'deferred_this_mission'):
        locked = set(mission.get('deferred_this_mission', []))
        locked.update(zid for zid, saved in mission['records'].items()
                      if saved.get('state') == 'PARTIAL_DEFERRED')
        if not locked <= set(book.records):
            raise ValueError('RESUME_DEFERRED_UNKNOWN_ZONE')
        # A restart is NOT a new mission. Reopen only through a separate,
        # explicitly requested follow-up mission (not this checkpoint resume).
        book.deferred_this_mission.clear()
        book.deferred_this_mission.update(locked)
    for key in BOOK_DICTS:
        if hasattr(book, key):
            defaults = {z: ([] if key == 'deferral_history' else 0) for z in node.zones_by_id} if key != 'outcomes' else {}
            defaults.update(copy.deepcopy(mission.get(key, {})))
            setattr(book, key, defaults)
    # History survives; old access outcomes are not asserted as fresh observations.
    for result in getattr(book, 'outcomes', {}).values():
        result['evidence_is_historical'] = True
    for key in NODE_COUNTERS:
        value = mission.get('counters', {}).get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError('RESUME_BAD_COUNTER:' + key)
        setattr(node, key, value)
    runtime = node.connectivity
    saved = mission.get('connectivity', {})
    if runtime is not None:
        for key in CONNECTION_DICTS:
            if hasattr(runtime, key):
                setattr(runtime, key, copy.deepcopy(saved.get(key, {})))
        runtime.total_probes = int(saved.get('total_probes', 0))
        runtime.local_reviewed = set(saved.get('local_reviewed', []))
        if not runtime.local_reviewed <= set(book.records):
            raise ValueError('RESUME_REVIEWED_UNKNOWN_ZONE')
        # A completed Zone-local Second Pass stays reviewed even when its
        # measured Coverage is <96%. Preserve this separately from completion.
        ledger = getattr(runtime, 'tasks', None)
        task_data = mission.get('task_ledger') or {}
        if ledger is not None:
            ledger.tasks = {}
            for record in task_data.get('tasks', []):
                item = copy.deepcopy(record)
                mask = mask_from_runs(item.pop('cell_runs_ros', []), bundle[1].height, bundle[1].width)
                if np.any(mask & (bundle[2] != node.zones_by_id[item['zone_id']]['mask_id'])):
                    raise ValueError('RESUME_TASK_OUTSIDE_ZONE')
                item['indices'] = np.flatnonzero(mask).astype(np.int32)
                # Replan unfinished work; never replay an active/accepted Goal.
                if item.get('status') in ('ACTIVE', 'NAVIGATING', 'PLANNING', 'INSPECTING', 'APPROACH_CHECKING', 'UNVERIFIED'):
                    item['status'] = 'PENDING'
                ledger.tasks[item['task_id']] = item
            ledger.inventoried_zones = set(task_data.get('inventoried_zones', []))
            ledger.small_by_zone = copy.deepcopy(task_data.get('small_by_zone', {}))
            ledger.order = int(task_data.get('order', 0))
            ledger.initialized = bool(task_data.get('initialized', False))
            ledger.session = inp['status']['session_id']
            ledger.refresh(inp)
        # Resume the interrupted local-second-pass, otherwise let existing primary select.
        zid = saved.get('local_second_zone')
        if saved.get('stage') == 'SECOND' and zid in node.zones_by_id:
            runtime.local_reviewed.discard(zid)
            runtime.local_second_zone = zid
            runtime.exploration_stage = 'SECOND'
        else:
            runtime.primary_zone = None
            runtime.local_second_zone = None
            runtime.exploration_stage = 'PRIMARY'
        runtime.second_task = runtime.second_catalog = runtime.second_context = None
        # Old failures are evidence, not current-costmap exclusion zones.
        node._v18_failed_entries_history = copy.deepcopy(mission.get('failed_entries_history', {}))
        for z, entries in getattr(runtime, 'failed_entries', {}).items():
            node._v18_failed_entries_history.setdefault(z, []).extend(copy.deepcopy(entries))
        if hasattr(runtime, 'failed_entries'):runtime.failed_entries = {}
        runtime.second_last_account = None
    guard = node.mission_guard
    old_guard = mission.get('guard')
    if guard is not None and old_guard:
        from mission_budget import ZoneWorkLedger
        if plain(asdict(guard.s)) != old_guard['settings']:
            raise ValueError('RESUME_GUARD_SETTINGS_CHANGED')
        guard.records = copy.deepcopy(old_guard.get('records', []))
        guard.zone_cancels = copy.deepcopy(old_guard.get('zone_cancels', {}))
        for zid, rec in old_guard.get('ledgers', {}).items():
            if zid not in node.zones_by_id:
                continue
            ledger = ZoneWorkLedger(guard.s)
            ledger.elapsed = float(rec['work_time_ros_s'])
            ledger.allowance = float(rec['allowance_ros_s'])
            ledger.extended = bool(rec.get('extended'))
            ledger.expired = bool(rec.get('expired'))
            ledger.pause()  # discard stale timer window, retain consumed budget
            guard.ledgers[zid] = ledger
    node.options = []
    node.current = None
    node.current_path = []
    node.current_path_length = None
    node.batch_zone = None
    node.options_context = None
    node.validated_plan_mode = None
    node.phase = 'SELECT_ZONE'
    node.state = 'SELECTING'


def _restore_legacy_detector_memories(node, mission):
    """Preserve cancellation histories; never restore active watch/pending state."""
    summaries = mission.get('original_summary') or {}
    saved = mission.get('detector_memories')
    if saved is None:
        # Older checkpoints contain summary snapshots, not full memory stores.
        # They may include older witnesses; never read the mutable global files.
        points = (summaries.get('region_review') or {}).get('blockage_memory', {}).get('points', [])
        saved = dict(blockage=[dict(p, goal_ids=[]) for p in points],
                     reroute=(summaries.get('reroute_memory_v1_4') or {}).get('points', []))
    memory = getattr(getattr(node, 'connectivity', None), 'blockage_memory', None)
    if memory is not None:
        memory.restore_records(saved.get('blockage', []))
    reroute = getattr(getattr(node, 'path_blockage', None), 'reroute', None)
    if reroute is not None:
        reroute.store.restore_records(saved.get('reroute', []))
    monitor = node.path_blockage
    if monitor is not None:
        records = summaries.get('path_blockage_cancellations', [])
        monitor.records = copy.deepcopy(records)
        monitor.zone_counts = {}
        monitor.latest_by_zone = {}
        for r in records:
            zid = (r.get('evidence') or {}).get('zone')
            if zid:
                monitor.zone_counts[zid] = monitor.zone_counts.get(zid, 0) + 1
                monitor.latest_by_zone[zid] = r


class MissionCheckpoint:
    def __init__(self, node, args, bundle, cfg, output):
        self.n, self.args, self.bundle, self.cfg = node, args, bundle, cfg
        self.output = Path(output)
        self.context = project_context(root_path(), bundle, args.namespace)
        self.policy = mission_policy(args)
        self.parent = None
        self.parent_digest = None
        self.parent_file = None
        self.restored = False
        self.next_save = 0.
        self.last_saved = None
        self.dirty = True
        self.write_failed = False
        self.access_event_keys = {}
        self.resume_mission = None
        self.pose_decision = None
        self.resume_started_wall = time.monotonic()
        self.registry = BlockageRegistry(bundle)
        if getattr(args, 'resume_from', None):
            if not getattr(args, 'resume_confirm_scene', False):
                raise ValueError('RESUME_REQUIRES --resume-confirm-scene')
            self.parent, self.parent_digest, self.parent_file = load_checkpoint(args.resume_from)
            assert_context(self.parent, self.context)
            if self.parent.get('mission_policy') is not None and self.parent['mission_policy'] != self.policy:
                raise ValueError('RESUME_MISSION_POLICY_CHANGED')
            self.saved_observed = validate_coverage(self.parent['coverage'], bundle)
            if not self.parent.get('mission'):
                raise ValueError('RESUME_MISSION_STATE_MISSING')
            recovered = recover_terminal_history(root_path(), self.parent, bundle, self.context)
            # In-memory only: selected legacy payload and its hash NEVER change.
            self.resume_mission = legacy_upgrade(recovered,
                source_coverage=self.parent['coverage']['source_session'],
                source_digest=self.parent_digest)
            self.registry = BlockageRegistry(bundle, [dict(r,historical=True) for r in self.parent.get('blocked_areas', []) if r.get('geometry_kind') == 'POINT'])
        self.session_ids = list((self.parent or {}).get('mission_session_ids', []))
        if self.output.name not in self.session_ids:
            self.session_ids.append(self.output.name)
        self.home = HomeCheckpoint(root_path(), self.context, self.session_ids, self.parent)

    def before_poll(self):
        n = self.n
        if not self.home.before_initial_poll(self):
            return False
        if self.parent is None or self.restored or n.stop_requested:
            return True
        # Parent poll normally advances these clocks. During the resume gate
        # no parent selection runs, so maintain the SAME clock-health fields.
        now, wall = n.ros_s(), time.monotonic()
        if n.last_clock is None or now > n.last_clock:n.clock_advanced_wall = wall
        if n.last_clock is not None and now < n.last_clock - .1:n.fault = 'CLOCK_RESET_RESTART_REQUIRED'
        n.last_clock = now
        if time.monotonic() - getattr(self, 'resume_started_wall', time.monotonic()) > 60.:
            raise ValueError('RESUME_CURRENT_INPUT_TIMEOUT; no new Goal sent')
        error, inp = n.inputs()
        if error or inp is None:
            n.info('resume_wait', '[RESUME_WAIT] ' + str(error), 3.)
            if n.fault:
                n.request_stop(n.fault)
                return True
            return False
        if not getattr(n, '_v183_action_seen', True):
            n.info('resume_status', '[RESUME_WAIT] receiving current Nav2 action status', 3.)
            return False
        if n.foreign_active():
            n.info('resume_foreign', '[RESUME_WAIT] Cancel the previous Goal and verify stopped robot.', 3.)
            return False
        if not n.stop_evidence.ready(n.ros_s(), time.monotonic()) or not n.local_start_clear(inp):
            n.info('resume_stop', '[RESUME_WAIT] fresh stopped odometry and clear current footprint required', 3.)
            return False
        saved = self.parent['coverage']
        status = inp['status']
        if status['session_id'] != saved['source_session']:
            resume = status.get('resume_v18') or {}
            if resume.get('parent_payload_sha256') != self.parent_digest:
                raise ValueError('RESUME_T5_NOT_RESTORED_FROM_SELECTED_CHECKPOINT')
        compatible_live_coverage(self.saved_observed, inp['observed'], status, self.parent_digest)
        n._v187_lifecycle.load(self.resume_mission)
        restore_mission(n, self.resume_mission, inp, self.bundle)
        apply_terminal_history(n, self.resume_mission)
        # Phase continuity: a reviewed/entered Zone with unfinished SECOND
        # tasks MUST NOT replay PRIMARY ENTRY on resume.
        from phase_resume import install_second_phase_continuation
        install_second_phase_continuation(n, self.resume_mission)
        # v1.8 originally discarded reviewed <96% zones during restore.
        # A restart is a continuation, not a second attempt at finished rooms.
        prior_reviews = set((self.resume_mission.get('connectivity') or {}).get('local_reviewed', []))
        if prior_reviews - set(n.book.records):
            raise ValueError('RESUME_UNKNOWN_REVIEWED_ZONE')
        if n.connectivity is not None and hasattr(n.connectivity, 'local_reviewed'):
            # Keep past review events as history, but only terminal Zones as
            # scheduling locks. CORRIDOR_E's old false review must not re-lock.
            n.connectivity.local_reviewed.update(prior_reviews & set(completion_facts(self.resume_mission)))
        if hasattr(n, '_dn012_finalize_reviewed'):
            n._dn012_finalize_reviewed()
        _restore_legacy_detector_memories(n, self.resume_mission)
        from zone_planner import is_inside
        self.pose_decision = current_pose_decision(n, self.resume_mission, inp, self.bundle, self.cfg, is_inside)
        selection_path = os.environ.get('DN012_RESUME_SELECTION_V183')
        manual_note = None
        if selection_path:
            from checkpoint import read_json
            selected = read_json(selection_path)
            if selected.get('payload_sha256') != self.parent_digest:
                raise ValueError('AUTO_RESUME_SELECTION_DIGEST_MISMATCH')
            manual_note = selected.get('manual_note')
        self.pose_decision['manual_intervention'] = 'DECLARED' if manual_note else 'NOT_DECLARED'
        self.pose_decision['manual_note'] = manual_note
        n.event('RESUME_CURRENT_POSE_V183', **self.pose_decision)
        n.info('resume_pose', '[RESUME_CURRENT_POSE] map=' + str(self.pose_decision['pose_map']) +
               ' zone=' + str(self.pose_decision['spatial_zone']) +
               ' completed=' + str(len(self.pose_decision['finished_zones'])) +
               ' remaining=' + str(self.pose_decision['remaining_zones']), 0)
        self.restored = True
        self.home.resumed(inp['pose'], self.output.name, n.ros_s())
        n.event('RESUME_RESTORED', parent_payload_sha256=self.parent_digest,
                observed_cells=int(inp['observed'].sum()), cumulative_goals=n.sent_goals,
                active_goal_replayed=False, historical_thresholds_require_fresh_reconfirmation=True)
        return True

    def event(self, kind, fields):
        self.registry.event(kind, fields)
        if kind in {'NAV_REQUESTED','NAV_RESULT','ZONE_COMPLETE','ZONE_DEFERRED','ZONE_ASSESSMENT',
                    'ENTRY_CONFIRMED','LOCAL_SECOND_PASS_START','LOCAL_ZONE_REVIEW_FINISHED',
                    'PATH_BLOCKED_EVIDENCE','REROUTE_POINT_RECORDED','SECOND_PASS_REGION_RESULT',
                    'ZONE_EXPLORATION_COMPLETE','SUMMARY'}:
            self.dirty = True

    def enrich(self, report):
        n = self.n
        if hasattr(n, '_dn012_finalize_reviewed'):
            n._dn012_finalize_reviewed()
        finished = {z: rec.state for z, rec in n.book.records.items()}
        done = {'OBSERVATION_SUFFICIENT', EXPLORATION_COMPLETE, 'OBSERVATION_LIMITED'}
        done_ids = set(getattr(n, '_v183_finished_zones', ())) | {z for z,s in finished.items() if s in done}
        if getattr(n, 'connectivity', None) is not None:
            rt = n.connectivity
            report['phase_resume_v185'] = dict(version='DN012_PHASE_CONTINUITY_V1_8_5',
                active_second_zone=getattr(rt,'local_second_zone',None),
                pending_second_zones=sorted(getattr(rt,'_v185_pending_second',{})),
                processed_second_zones=sorted(getattr(rt,'_v185_processed_second',set())),
                original_primary_not_replayed=True)
        report['exploration_completion_v182'] = dict(
            status='탐사 완료' if set(finished) <= done_ids else '탐사 진행 중',
            completed_zones=len(done_ids),
            total_zones=len(finished),
            coverage_96_percent_zones=sum(s == 'OBSERVATION_SUFFICIENT' for s in finished.values()),
            locally_completed_zones=sum(s == EXPLORATION_COMPLETE for s in finished.values()),
            remaining_zones=sorted(set(finished) - done_ids),
            coverage_unchanged=True,
            inaccessible_or_unexplorable_areas_reported_separately=True)
        for zid, state in finished.items():
            if zid in (report.get('zones') or {}):
                report['zones'][zid]['state'] = state
                if zid in done_ids:
                    report['zones'][zid]['exploration_label'] = '탐사 완료'
                    report['zones'][zid]['exploration_status'] = 'COMPLETE'
                    met = n.book.records[zid].measured + 1e-12 >= n.book.threshold
                    confirmed = state == 'OBSERVATION_SUFFICIENT' and met
                    report['zones'][zid]['coverage_target_met'] = bool(met)
                    report['zones'][zid]['coverage_target_reached'] = bool(met)
                    report['zones'][zid]['coverage_confirmed'] = bool(confirmed)
                    report['zones'][zid]['coverage_status'] = ('TARGET_CONFIRMED' if confirmed else
                        'TARGET_REACHED_UNCONFIRMED' if met else 'BELOW_TARGET')
                    report['zones'][zid]['deferred_reason'] = None
        lifecycle = getattr(n, '_v187_lifecycle', None)
        if lifecycle is not None:
            table = lifecycle.snapshot(n)
            report['zone_lifecycle_v187'] = table
            for zid, item in table['zones'].items():
                if zid in report.get('zones', {}):
                    report['zones'][zid]['exploration_status'] = item['exploration_status']
                    report['zones'][zid]['access_status_v187'] = item['access_status']
                    report['zones'][zid]['completion_reason_v187'] = item['completion_reason']
            complete_count = sum(x['exploration_status']=='COMPLETE' for x in table['zones'].values())
            report['exploration_completion_v182']['completed_zones'] = complete_count
            report['exploration_completion_v182']['remaining_zones'] = [
                z for z,x in table['zones'].items() if x['exploration_status']!='COMPLETE']
            report['exploration_completion_v182']['status'] = (
                '탐사 완료' if complete_count == len(table['zones']) else '탐사 진행 중')
        report['current_pose_resume_v183'] = copy.deepcopy(self.pose_decision)
        report['completion_v184'] = dict(
            policy='COVERAGE_96_OR_LOCAL_REVIEW_FINISHED',
            unfinished_local_review_is_resumable=True,
            resumed_false_completions=sorted(getattr(n, '_v184_review_reopened', set())),
            unfinished_zones=sorted(set(n.book.records)-done_ids),
            completed_zones=sorted(done_ids),
            task_results_and_coverage_unchanged=True)
        if lifecycle is not None:
            complete_ids = {z for z,x in table['zones'].items() if x['exploration_status']=='COMPLETE'}
            report['completion_v184']['completed_zones'] = sorted(complete_ids)
            report['completion_v184']['unfinished_zones'] = sorted(set(table['zones'])-complete_ids)
        for zid in report['completion_v184']['unfinished_zones']:
            if zid in report.get('zones', {}):
                report['zones'][zid]['exploration_status']='INCOMPLETE'
                report['zones'][zid]['coverage_target_met']=False
                report['zones'][zid]['requires_followup_review']=True
        report['source_session_v18'] = self.output.name
        report['resume_v18'] = dict(version=VERSION, requested=self.parent is not None,
            restored=self.restored, parent_payload_sha256=self.parent_digest,
            lifetime_goals=int(n.sent_goals), checkpoint_interval_s=5.,
            checkpoint_directory=str(self.output / 'checkpoints_v18'),
            discarded_active_action=True, temporal_safety_revalidated=self.restored)
        if self.last_saved is not None:
            report['resume_v18']['last_checkpoint'] = str(self.last_saved)
        # Collect historical observer records without merging them into navigation masks.
        self.registry.absorb_summary(report)
        unified = self.registry.public()
        region = report.get('region_review') or {}
        projected = region.get('final_results') or {}
        # Persist regions and Zone findings as well as points across restarts.
        historical_regions = [dict(copy.deepcopy(r), historical=True) for r in
            (self.parent or {}).get('blocked_areas', []) if r.get('geometry_kind') != 'POINT']
        current_regions = [copy.deepcopy(r) for r in projected.get('blocked_areas', [])
                           if r.get('geometry_kind') != 'POINT']
        evaluated = set((region.get('zones') or {}).keys())
        unified.extend(r for r in historical_regions if r.get('zone_id') not in evaluated)
        unified.extend(current_regions)
        raw_points = [r for r in unified if r.get('geometry_kind') == 'POINT']
        decided_points, diagnostics = normalize_points(raw_points)
        unified = [r for r in unified if r.get('geometry_kind') != 'POINT'] + decided_points
        report['blocked_areas'] = unified
        report['blocked_point_diagnostics_v189'] = diagnostics
        report['navigation_v189'] = self.home.snapshot(n)
        report['unified_blockage_label'] = '차단지역'
        for zid, item in (region.get('zones') or {}).items():
            access = item.get('access') or {}
            if zid in report['zones'] and access.get('zone_access_status'):
                key=(access.get('snapshot_cost_key'),access['zone_access_status'])
                if not item.get('evidence_is_historical') and self.access_event_keys.get(zid)!=key:
                    self.access_event_keys[zid]=key
                    verdict=access['zone_access_status']
                    label={'INACCESSIBLE':'접근불가','PARTIALLY_INACCESSIBLE':'일부 영역 접근불가',
                        'LOCAL_BLOCKED_GLOBAL_CONNECTION':'국소 차단·전역 연결',
                        'NO_DISCONNECTION_IN_CURRENT_MODEL':'현재 연결성 모델에서 단절 없음',
                        'BELOW_SIGNIFICANT_AREA':'최소 면적 미만'}.get(verdict,verdict)
                    n.event('ACCESSIBILITY_DECIDED_V18',zone=zid,classification=verdict,
                            display_label=label,cost_key=access.get('snapshot_cost_key'))
                    n.info('access_v18_'+zid,'[ACCESSIBILITY_V18] '+zid+' '+label,0)
                report['zones'][zid]['current_accessibility'] = dict(
                    status=access['zone_access_status'], reason=access.get('reason'),
                    snapshot_cost_key=access.get('snapshot_cost_key'),
                    evidence_is_historical=bool(item.get('evidence_is_historical')))
                if access['zone_access_status'] == 'INACCESSIBLE':
                    if not item.get('evidence_is_historical') and hasattr(n.book, 'outcomes'):
                        n.book.outcomes[zid] = dict(classification='INACCESSIBLE',
                            display_label='접근불가', reason=access.get('reason'),
                            evidence=dict(copy.deepcopy(access), current_map_unreachable=True),
                            normal_termination=False, coverage_completion_inferred=False)
                    report['zones'][zid]['legacy_assessment'] = report['zones'][zid].get('assessment')
                    report['zones'][zid]['assessment'] = dict(classification='INACCESSIBLE',
                        display_label='접근불가', reason=access.get('reason'),
                        evidence=copy.deepcopy(access),
                        evidence_is_historical=bool(item.get('evidence_is_historical')))
        atomic_bytes(self.output / 'unified_blockage_v18.json', encode(dict(
            schema='dn012_unified_blockage_v18', display_label='차단지역',
            context=self.context, records=unified)))
        # No blank startup checkpoint can overwrite the selected parent snapshot.
        if n.matched is None or (self.parent is not None and not self.restored):
            return report
        if time.monotonic() < self.next_save and not self.dirty and not n.done:
            return report
        cov = coverage_payload(self.bundle, n.matched['observed'], n.matched['status'])
        payload = dict(schema=SCHEMA, kind='MISSION', created_utc=datetime.now(timezone.utc).isoformat(),
            context=self.context, mission_policy=self.policy, coverage=cov, mission=capture_mission(n, report),
            blocked_areas=unified,
            blocked_point_diagnostics_v189=report['blocked_point_diagnostics_v189'],
            navigation_v189=report['navigation_v189'],
            historical_reports=(self.parent or {}).get('historical_reports', []),
            mission_session_ids=self.session_ids, parent_payload_sha256=self.parent_digest,
            revalidation_required=['fresh TF','fresh costmaps','fresh LiDAR','stopped odometry','local footprint'],
            active_action_is_historical_only=True)
        target, digest = save_checkpoint(self.output / 'checkpoints_v18', payload)
        self.last_saved = target
        self.next_save = time.monotonic() + 5.
        self.dirty = False
        report['resume_v18']['last_checkpoint'] = str(target)
        report['resume_v18']['last_checkpoint_payload_sha256'] = digest
        return report


def enrich_mission_report(node, report):
    controller = getattr(node, '_dn012_v18', None)
    if controller is None:
        return report
    try:
        return controller.enrich(report)
    except Exception as exc:
        # Checkpoint failure cannot let a robot continue without the requested state persistence.
        node.warn('v18_write', '[CHECKPOINT_WRITE_FAILED] ' + repr(exc), 3.)
        report['resume_v18_error'] = repr(exc)
        if not controller.write_failed:
            controller.write_failed = True
            node.request_stop('CHECKPOINT_WRITE_FAILED')
        return report




# DN012 v1.8.2: Zone survey is terminal once its local procedure has ended.
# "Survey complete" is an operational result, NOT a claim of 96% coverage.
EXPLORATION_COMPLETE = 'EXPLORATION_COMPLETE'


def install_simple_completion_policy(node):
    """Keep navigation safety untouched; make the Zone lifecycle terminal.

    * Coverage >=96% remains OBSERVATION_SUFFICIENT after fresh tracker checks.
    * A Zone whose local PRIMARY+SECOND procedure finished is EXPLORATION_COMPLETE,
      regardless of remaining unobserved cells. Those cells and blocked areas are
      recorded as separate per-Zone information.
    * A restart never automatically revisits an already finished Zone.
    * Preflight BUDGET or UNVERIFIED is only a subregion diagnosis after _end_local.
      It must not reopen a Zone whose local review has finished.
    """
    book = node.book
    if getattr(book, '_dn012_simple_completion_installed', False):
        return
    if not all(hasattr(book, key) for key in ('update', 'eligible', 'all_resolved', 'is_complete')):
        raise ValueError('SIMPLE_ZONE_COMPLETION_UNSUPPORTED_BOOK')
    saved_update = book.update
    saved_eligible = book.eligible
    saved_resolved = book.all_resolved
    saved_complete = book.is_complete

    def finalize_reviewed():
        runtime = getattr(node, 'connectivity', None)
        if runtime is None:
            return []
        reviewed = set(getattr(runtime, 'local_reviewed', ())) | set(getattr(node, '_v183_finished_zones', ()))
        threshold = float(getattr(book, 'threshold', .96))
        reviewed.update(z for z,r in book.records.items() if r.measured + 1e-12 >= threshold)
        if not reviewed <= set(book.records):
            raise ValueError('SIMPLE_ZONE_COMPLETION_UNKNOWN_ZONE')
        just_finished = []
        for zid in sorted(reviewed):
            # v1.8.9.2: _end_local ends the procedure regardless of entry or
            # residual verdict. Completion does not assert physical blockage.
            if not review_terminal(node, zid):
                continue
            # Don't close a local second pass that is still physically in progress.
            if (getattr(runtime, 'exploration_stage', None) == 'SECOND' and
                    getattr(runtime, 'local_second_zone', None) == zid and
                    book.records[zid].measured + 1e-12 < float(getattr(book, 'threshold', .96))):
                continue
            rec = book.records[zid]
            # local_reviewed is set only when _end_local has finished.
            # Zone COMPLETION is irreversible within the same mission, even if
            # an internal task ended with UNVERIFIED or preflight BUDGET.
            # Keep those task results as diagnostic subregions, NOT as a reason
            # to reopen the Zone and resend ENTRY/OBSERVE goals.
            if not hasattr(node, '_v183_finished_zones'):node._v183_finished_zones = set()
            node._v183_finished_zones.add(zid)
            if rec.state in ('OBSERVATION_SUFFICIENT', 'THRESHOLD_PENDING'):
                continue
            if rec.state == EXPLORATION_COMPLETE:
                continue
            # Previous primary access/deferral reasons remain in history; they
            # are not mission-level unfinished tasks after the local review.
            old_reason = rec.deferred_reason or (getattr(book, 'outcomes', {}).get(zid) or {}).get('reason')
            diagnosis = (getattr(book, 'outcomes', {}).get(zid) or {})
            physical_access = (diagnosis.get('classification') == 'INACCESSIBLE' or
                (diagnosis.get('historical_diagnosis') or {}).get('classification') == 'INACCESSIBLE')
            operational_access_label = ('접근 불가' if physical_access
                                        else '로봇 탐사 불가' if old_reason else None)
            if hasattr(book, 'outcomes'):
                previous = copy.deepcopy(book.outcomes.get(zid, {}))
                if previous.get('classification') != EXPLORATION_COMPLETE:
                    book.outcomes[zid] = dict(
                    classification=EXPLORATION_COMPLETE,
                    display_label='탐사 완료',
                    normal_termination=True,
                    coverage_target_met=False,
                    reason='ZONE_LOCAL_PROCEDURE_FINISHED',
                    access_or_sensing_limit=old_reason,
                    operational_access_label=operational_access_label,
                    historical_diagnosis=previous,
                    evidence=dict(local_review_finished=True,
                        remaining_coverage_percent=round(100.0 * (1.0 - rec.measured), 4),
                        accessibility_and_unexplorable_areas_are_separate=True),
                    )
            rec.state = EXPLORATION_COMPLETE
            rec.deferred_reason = None
            if hasattr(book, 'audit_requested'):
                book.audit_requested.pop(zid, None)
            just_finished.append(zid)
        return just_finished

    def update_with_completion(status):
        newly = saved_update(status)
        finalize_reviewed()
        return newly

    def eligible_without_finished_zones():
        return [z for z in saved_eligible()
                if book.records[z['id']].state != EXPLORATION_COMPLETE
                and z['id'] not in getattr(node, '_v183_finished_zones', set())]

    def all_resolved_with_finished_zones():
        if saved_resolved():
            return True
        return all(rec.state in ('OBSERVATION_SUFFICIENT',
                                 'OBSERVATION_LIMITED', EXPLORATION_COMPLETE)
                   for rec in book.records.values())

    def is_complete_without_repeat(zid):
        return (book.records[zid].state == EXPLORATION_COMPLETE
                or saved_complete(zid))

    book.update = update_with_completion
    book.eligible = eligible_without_finished_zones
    book.all_resolved = all_resolved_with_finished_zones
    book.is_complete = is_complete_without_repeat
    book._dn012_simple_completion_installed = True
    node._dn012_finalize_reviewed = finalize_reviewed

    # A completed Zone can be crossed for transit, but gets no new local review.
    rt = getattr(node, 'connectivity', None)
    if rt is not None and hasattr(rt, '_maybe_begin_local'):
        saved_begin_local = rt._maybe_begin_local
        def begin_local_only_if_unfinished(inp):
            finalize_reviewed()
            if getattr(rt, 'primary_zone', None) in getattr(node, '_v183_finished_zones', set()):
                rt.primary_zone = None
                return False
            return saved_begin_local(inp)
        rt._maybe_begin_local = begin_local_only_if_unfinished

    if rt is not None and hasattr(rt, 'selection_hook'):
        saved_selection_hook = rt.selection_hook
        def select_only_unfinished(inp):
            finalize_reviewed()
            zid = getattr(rt, 'local_second_zone', None)
            if (getattr(rt, 'exploration_stage', None) == 'SECOND' and
                    zid in getattr(node, '_v183_finished_zones', set())):
                if any(getattr(node, k, None) is not None for k in
                       ('nav_uuid','nav_send','nav_handle','plan_send','plan_handle','plan_result')):
                    return True
                # Drain, never overwrite a live worker or a navigation action.
                for obj, key in ((rt, 'second_future'), (rt, 'future'), (node, 'worker')):
                    future = getattr(obj, key, None)
                    if future is not None:
                        if not future.done():return True
                        try:future.result()
                        except TimeoutError:pass
                        setattr(obj, key, None)
                rt._end_local(inp)
                return True
            return saved_selection_hook(inp)
        rt.selection_hook = select_only_unfinished

    # Mark a Zone complete immediately after _end_local, before selecting another.
    rt = getattr(node, 'connectivity', None)
    if rt is not None and hasattr(rt, '_end_local'):
        original_end_local = rt._end_local
        def end_local_then_finish(inp):
            ended_zid = getattr(rt, 'local_second_zone', None)
            result = original_end_local(inp)
            just_finished = finalize_reviewed()
            for zid in just_finished:
                node.event('ZONE_EXPLORATION_COMPLETE', zone=zid,
                           measured_percent=round(100*book.records[zid].measured, 4),
                           success_kind='VERIFIED_LOCAL_PROCEDURE_FINISHED',
                           not_a_claim_of_96_percent=True)
            if ended_zid and ended_zid not in node._v183_finished_zones:
                valid,reason = live_review_verdict(rt,ended_zid)
                if not valid and ended_zid in getattr(rt,'local_reviews',{}):
                    node.event('ZONE_REVIEW_INCOMPLETE_V184',zone=ended_zid,
                        reason=reason,terminal=False,remaining_candidates_possible=True,
                        completion_claim=False,task_results=copy.deepcopy(
                            rt.local_reviews[ended_zid].get('results', [])))
            return result
        rt._end_local = end_local_then_finish

    finalize_reviewed()


def wrap_mission(base, args, bundle, cfg, output):
    if os.environ.get('DN012_V18_ENABLE') != '1' and not getattr(args, 'resume_from', None):
        return base
    class ResumeSequence(base):
        def __init__(self):
            self._dn012_v18 = None
            self._v183_action_seen = False
            self._v183_last_pose = None
            self._v183_finished_zones = set()
            super().__init__()
            if getattr(self, 'connectivity', None) is not None:
                self.connectivity.local_review_history = {}
            self._v187_lifecycle = ZoneLifecycle(list(self.zones_by_id))
            self._dn012_v18 = MissionCheckpoint(self, args, bundle, cfg, output)
            install_simple_completion_policy(self)
            self.info('v18', '[DN012_V18_9_2] ended local procedure=COMPLETE; four result categories; existing HOME return', 0)

        def inputs(self):
            error, inp = super().inputs()
            if error is None and inp is not None:
                self._v183_last_pose = dict(pose=list(map(float, inp['pose'])),
                    ros_time_s=float(self.ros_s()), frame_id='map')
            return error, inp

        def on_action_status(self, msg):
            result = super().on_action_status(msg)
            self._v183_action_seen = True
            return result

        def nav_request(self, inp):
            c = self._dn012_v18
            if c.parent is None and args.execute and not c.home.capture_persisted:
                self.request_stop('ORIGINAL_START_NOT_PERSISTED_NO_NAV_SENT')
                return
            # Recheck lifecycle at the real send boundary, including second-pass
            # candidates planned just before newly measured Coverage reached 96%.
            self._dn012_finalize_reviewed()
            zid = (self.current or {}).get('zone_id')
            if zid in self._v183_finished_zones:
                self.event('COMPLETED_ZONE_GOAL_NOT_SENT_V183', zone=zid,
                           reason='ZONE_LIFECYCLE_FINISHED', navigation_sent=False)
                self.current = None;self.current_path = [];self.current_path_length = None
                self.options = [];self.batch_zone = None;self.options_context = None
                self.validated_plan_mode = None;self.plan_snapshot = None
                self.transition('SELECTING')
                return
            return super().nav_request(inp)

        def finish(self, state, code=0):
            # A terminal Zone lifecycle is not undone by residual task diagnostics.
            # User stop / faults / budget exhaustion are never renamed success.
            if state in ('PARTIAL_DEFERRED_REMAIN', 'ALL_ZONES_ASSESSED_WITH_LIMITS',
                         'ALL_REQUESTED_ZONES_OBSERVED') and code == 0:
                self._dn012_finalize_reviewed()
                if self.book.all_resolved():
                    state = 'ALL_ZONES_EXPLORATION_COMPLETE'
            return super().finish(state, code)

        def event(self, event, **fields):
            # Original event/cancel bookkeeping runs FIRST. Never suppress an owned cancel.
            super().event(event, **fields)
            controller = getattr(self, '_dn012_v18', None)
            if controller is not None:
                try:
                    self._v187_lifecycle.event(event, fields)
                    if event == 'REROUTE_POINT_RECORDED' and self.current:
                        fields = dict(fields, destination_zone=self.current.get('zone_id'))
                    controller.event(event, fields)
                except Exception as exc:
                    self.warn('v18_event', '[UNIFIED_EVENT_FAILED] ' + repr(exc), 3.)
                    self.fault = 'UNIFIED_EVENT_PERSISTENCE_FAILED'

        def poll(self):
            try:
                if not self._dn012_v18.before_poll():
                    self.publish_report()
                    return
            except Exception as exc:
                self.warn('v18_restore', '[RESUME_REFUSED] ' + repr(exc), 0)
                self.request_stop('RESUME_VALIDATION_FAILED')
            return super().poll()

    ResumeSequence.__name__ = 'HospitalSequenceV18'
    return ResumeSequence


def wrap_coverage(base, args, bundle, output_dir):
    if os.environ.get('DN012_V18_ENABLE') != '1' and not getattr(args, 'resume_from', None):
        return base
    class ResumeCoverage(base):
        def __init__(self):
            namespace = getattr(args, 'namespace', '/j100_0001')
            self._v18_context = project_context(root_path(), bundle, namespace)
            self._v18_parent = None
            self._v18_digest = None
            self._v18_restored_cells = 0
            self._v18_ready = False
            key = sha((os.environ.get('ROS_DOMAIN_ID','0') + '|' + namespace).encode())[:16]
            self._v18_lock = open('/tmp/dn012_coverage_v18_' + key + '.lock', 'a')
            try:
                fcntl.flock(self._v18_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._v18_lock.close()
                raise ValueError('ANOTHER_V18_COVERAGE_NODE_IS_RUNNING')
            if getattr(args, 'resume_from', None):
                if not getattr(args, 'resume_confirm_scene', False):
                    raise ValueError('RESUME_REQUIRES --resume-confirm-scene')
                self._v18_parent, self._v18_digest, _ = load_checkpoint(args.resume_from)
                assert_context(self._v18_parent, self._v18_context)
                validate_coverage(self._v18_parent['coverage'], bundle)
            super().__init__()
            if self._v18_parent is not None:
                restore_coverage(self.coverage, self._v18_parent['coverage'], bundle)
                self._v18_restored_cells = self.coverage.observed_total
                self.get_logger().info('[COVERAGE_RESTORED] cells=' + str(self._v18_restored_cells) +
                    '; awaiting fresh map/TF/scan; old scan timestamps discarded')
            self._v18_ready = True

        def make_report(self):
            report = super().make_report()
            report['resume_v18'] = dict(version=VERSION, parent_payload_sha256=self._v18_digest,
                restored_cells=self._v18_restored_cells, fresh_scans_since_restart=self.coverage.evidence_scan_count,
                historical_status_not_live=True)
            return report

        def save(self):
            super().save()
            if not self._v18_ready:
                return
            report = self.make_report()
            cov = coverage_payload(bundle, self.coverage.observed, report)
            payload = dict(schema=SCHEMA, kind='COVERAGE', context=self._v18_context,
                coverage=cov, mission=None, parent_payload_sha256=self._v18_digest,
                created_utc=datetime.now(timezone.utc).isoformat())
            save_checkpoint(Path(output_dir) / 'checkpoints_v18', payload)

    ResumeCoverage.__name__ = 'HospitalCoverageNodeV18'
    return ResumeCoverage
