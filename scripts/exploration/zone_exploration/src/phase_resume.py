"""Checkpoint phase continuity without PRIMARY replay.

For a previously entered Zone with a saved, budget-interrupted local review,
resume its existing SECOND-pass tasks. A Zone with no confirmed entry is left to
ordinary PRIMARY/ENTRY. The caller determines unfinished Zones using the shared
zone_lifecycle module before requesting phase continuation.

No ROS imports, navigation commands or mutation of saved checkpoint files.
"""
from __future__ import annotations
import copy
import math

VERSION = 'DN012_PHASE_CONTINUITY_V1_8_5'


def second_phase_candidates(mission, reopened):
    """Pure classification from *saved* phase/entry evidence, not Coverage alone.

    An old local review may exist even if the robot never entered (e.g.
    CORRIDOR_E). Such Zones must first attempt actual PRIMARY entry, not skip to
    SECOND. Do not infer physical entry from a successful global plan.
    """
    if not isinstance(mission, dict):
        raise ValueError('V185_MISSION_REQUIRED')
    records = mission.get('records') or {}
    visits = mission.get('visit_states') or {}
    conn = mission.get('connectivity') or {}
    reviews = conn.get('local_reviews') or {}
    tasks = (mission.get('task_ledger') or {}).get('tasks') or []
    result = {}
    for zid in sorted(set(reopened)):
        if zid not in records:
            raise ValueError('V185_UNKNOWN_REOPENED_ZONE:' + zid)
        rec = records[zid]
        review = reviews.get(zid)
        seen = visits.get(zid) or {}
        entered = seen.get('entered') is True
        # The live second pass itself is an explicit phase witness. Never
        # assume the other unrelated Zone's work completed PRIMARY.
        saved_active = (conn.get('stage') == 'SECOND' and conn.get('local_second_zone') == zid)
        # Saved local review with actual successful goal visits corroborates
        # previous primary work. An empty-review/no-goal case is NOT entry.
        visited = type(rec.get('views')) is int and rec['views'] > 0
        has_review = isinstance(review, dict) and isinstance(review.get('results'), list)
        historical_second = has_review and (entered or visited)
        if not (saved_active or historical_second):
            continue
        zone_tasks = [t for t in tasks if isinstance(t, dict) and t.get('zone_id') == zid]
        if not zone_tasks:
            # PRIMARY is known to have been attempted; replaying it as a fallback
            # would fabricate progress. Refuse until ledger evidence is recovered.
            raise ValueError('V185_RESUME_SECOND_TASK_HISTORY_MISSING:' + zid)
        if not any(t.get('status') in ('UNVERIFIED', 'INSPECTING', 'PENDING',
                    'APPROACH_CHECKING', 'NAVIGATING', 'PLANNING') for t in zone_tasks):
            # An explicit reopened Zone with prior PRIMARY history and no
            # unfinished SECOND work is internally contradictory. Fail closed.
            raise ValueError('V185_SECOND_PHASE_TASK_ALREADY_TERMINAL:' + zid)
        result[zid] = dict(phase='SECOND', previous_review_present=has_review,
            confirmed_entered=entered, historical_goal_visits=rec.get('views', 0),
            saved_phase_active=saved_active,
            task_ids=[str(t.get('task_id')) for t in zone_tasks
                      if isinstance(t.get('task_id'), str)])
    return result


def install_second_phase_continuation(node, mission):
    """Wrap the existing selection hook; NEVER issue a Goal or bypass its gates.

    Called only after the original restore and strict v1.8.4 invalid-terminal
    reconciliation. The original SECOND selector retains all action, costmap,
    stop, footprint, preflight, deadline and budget checks.
    """
    rt = getattr(node, 'connectivity', None)
    if rt is None or not hasattr(rt, 'selection_hook'):
        return {}
    if getattr(rt, '_v185_phase_installed', False):
        raise ValueError('V185_DUPLICATE_PHASE_ADAPTER')
    reopened = set(getattr(node, '_v184_review_reopened', ()))
    phases = second_phase_candidates(mission, reopened)
    rt._v185_pending_second = dict(phases)
    rt._v185_processed_second = set()
    rt._v185_current_second = None
    rt._v185_phase_installed = True
    if not hasattr(rt, 'local_review_history'):
        rt.local_review_history = {}
    old_hook = rt.selection_hook
    old_end = rt._end_local

    def finish_local(inp):
        zid = getattr(rt, 'local_second_zone', None)
        result = old_end(inp)
        if zid in rt._v185_pending_second:
            rt._v185_processed_second.add(zid)
            rt._v185_pending_second.pop(zid, None)
            rt._v185_current_second = None
            # Do not restart PRIMARY merely because this round was budgeted.
            # Original _end_local freezes/defers this Zone for this mission.
            node.event('RESUME_SECOND_EPISODE_FINISHED_V185', zone=zid,
                completed=zid in getattr(node, '_v183_finished_zones', set()),
                no_primary_replay=True)
        return result

    def select_with_saved_phase(inp):
        if (getattr(rt, 'exploration_stage', None) != 'PRIMARY' or
                getattr(node, 'done', False) or getattr(node, 'stop_requested', None)):
            return old_hook(inp)
        if any(getattr(node, key, None) is not None for key in
               ('nav_uuid', 'nav_send', 'nav_handle', 'plan_send', 'plan_handle', 'plan_result')):
            return old_hook(inp)
        if getattr(node, 'current', None) is not None:
            return old_hook(inp)
        if getattr(rt, 'primary_zone', None) is not None:
            # The active PRIMARY is authoritative; don't interrupt it.
            return old_hook(inp)
        if getattr(node.book, 'active_zone', None) is not None:
            return old_hook(inp)
        for zid in list(rt._v185_pending_second):
            if zid in getattr(node, '_v183_finished_zones', set()) or node.book.is_complete(zid):
                rt._v185_pending_second.pop(zid, None)
        if not rt._v185_pending_second:
            return old_hook(inp)
        if not node.stop_evidence.ready(node.ros_s(), __import__('time').monotonic()):
            return True  # wait; no unsafe goal/candidate selection
        if not node.local_start_clear(inp):
            node.request_stop('RESUME_SECOND_LOCAL_START_NOT_CLEAR')
            return True
        if node.foreign_active():
            node.request_stop('RESUME_SECOND_OTHER_GOAL_ACTIVE')
            return True
        if not rt._drain_workers():
            return True
        ledger = getattr(rt, 'tasks', None)
        if ledger is None:
            node.request_stop('RESUME_SECOND_LEDGER_MISSING')
            return True
        ledger.refresh(inp)
        actionable = []
        for zid in list(rt._v185_pending_second):
            if any(t.get('zone_id') == zid and t.get('status') not in
                   ('OBSERVED', 'SMALL_REMAINDER', 'ROBOT_EXPLORATION_UNAVAILABLE', 'UNVERIFIED', 'SPLIT')
                   for t in ledger.tasks.values()):
                actionable.append(zid)
            else:
                # No active task: require the existing local review to be made
                # consistent by the ordinary logic; never invent completion.
                node.event('RESUME_SECOND_NO_PENDING_TASK_V185', zone=zid,
                    automatic_completion=False)
                rt._v185_pending_second.pop(zid, None)
        if not actionable:
            # Never silently fall back to PRIMARY if the saved SECOND task
            # couldn't be resumed. The operator must review the checkpoint.
            node.request_stop('RESUME_SECOND_NO_RUNNABLE_TASK')
            return True
        # Ordering is a *proposal priority* only. Nav2 retains route validation.
        def rank(zid):
            candidates = [t for t in ledger.tasks.values() if t.get('zone_id') == zid and
                          t.get('status') not in ('OBSERVED', 'SMALL_REMAINDER',
                                               'ROBOT_EXPLORATION_UNAVAILABLE', 'UNVERIFIED', 'SPLIT')]
            dist = math.inf
            for t in candidates:
                xy = t.get('centroid_map_m')
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    try:
                        value = math.dist(inp['pose'][:2], [float(xy[0]), float(xy[1])])
                        if math.isfinite(value):dist = min(dist, value)
                    except (TypeError, ValueError):pass
            return (dist, zid)
        zid = min(actionable, key=rank)
        previous = copy.deepcopy(rt.local_reviews.get(zid))
        if previous is not None:
            rt.local_review_history.setdefault(zid, []).append(previous)
            rt.local_reviews.pop(zid, None)
        rt.local_reviewed.discard(zid)
        rt._v185_current_second = zid
        # Re-enter only the already existing SECOND state machine, not PRIMARY.
        rt.local_second_zone = zid
        rt.primary_zone = None
        rt.exploration_stage = 'SECOND'
        rt.second_task = rt.second_catalog = rt.second_context = None
        rt.second_future = rt.second_future_task = None
        rt.second_attempted = {}
        rt.second_last_account = None
        if hasattr(rt, 'clear_round'):rt.clear_round()
        rt.zid = None
        node.options = []
        node.batch_zone = None
        node.options_context = None
        node.book.active_zone = None
        node.book.locked_zone = None
        node.event('RESUME_SECOND_PHASE_V185', zone=zid,
            original_primary_not_replayed=True, saved_phase_evidence=phases[zid],
            preserved_coverage=True, task_count=len(actionable))
        return old_hook(inp)

    rt._end_local = finish_local
    rt.selection_hook = select_with_saved_phase
    node.event('RESUME_PHASE_SCHEDULE_V185', second_only_zones=sorted(phases),
        entry_or_primary_zones=sorted(reopened - set(phases)),
        historically_complete_zones=sorted(getattr(node, '_v183_finished_zones', set())),
        original_history_retained=True)
    return phases
