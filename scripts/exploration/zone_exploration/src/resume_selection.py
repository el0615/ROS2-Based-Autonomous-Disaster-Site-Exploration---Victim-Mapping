"""Current-session checkpoint selection and terminal Zone continuity.

Pure data functions. No ROS imports, movement, reset, or modification of old logs.
An invalid newer executing session is NOT silently replaced by an older one.
"""
from __future__ import annotations
import copy
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any
import numpy as np
from checkpoint import load_checkpoint, read_json, assert_context, validate_coverage, encode, sha
from zone_lifecycle import completion_facts, unresolved_zones

VERSION = 'DN012_CURRENT_POSE_RESUME_V1_8_3'
SESSION = re.compile(r'^session_\d{8}_\d{6}_\d{6}$')
COUNTERS = ('sent_goals', 'plan_attempts', 'succeeded_goals', 'aborted_goals', 'coverage_cancels')
FINISHED_STATES = {'OBSERVATION_SUFFICIENT', 'OBSERVATION_LIMITED', 'EXPLORATION_COMPLETE'}


def checked_counters(mission):
    c = mission.get('counters')
    if not isinstance(c, dict):
        raise ValueError('RESUME_COUNTERS_MISSING')
    for key in COUNTERS:
        if type(c.get(key)) is not int or c[key] < 0:
            raise ValueError('RESUME_BAD_COUNTER:' + key)
    if c['succeeded_goals'] + c['aborted_goals'] > c['sent_goals']:
        raise ValueError('RESUME_INCONSISTENT_GOAL_COUNTS')
    return dict(c)


def safe_session(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not SESSION.fullmatch(name):
        raise ValueError('INVALID_MISSION_SESSION_NAME')
    base = (Path(root)/'logs/hospital_zone_mission').resolve()
    directory = base/name
    if directory.is_symlink() or not directory.resolve().is_relative_to(base):
        raise ValueError('MISSION_SESSION_PATH_ESCAPE')
    return directory


def validated_checkpoint(path, bundle, context):
    path=Path(path)
    if path.is_symlink():
        raise ValueError('CHECKPOINT_SYMLINK_REFUSED')
    p, digest, actual = load_checkpoint(path)
    if p.get('kind') != 'MISSION' or not isinstance(p.get('mission'),dict):
        raise ValueError('MISSION_CHECKPOINT_REQUIRED')
    assert_context(p, context)
    observed=validate_coverage(p['coverage'],bundle)
    completion_facts(p['mission'])
    checked_counters(p['mission'])
    if not set(p['mission']['scope']) <= {z['id'] for z in bundle[0]['zones']}:
        raise ValueError('RESUME_SCOPE_NOT_IN_BUNDLE')
    return p,digest,actual,observed


def _session_execute(directory):
    manifest=read_json(directory/'run_manifest.json')
    value=(manifest.get('options') or {}).get('execute')
    if type(value) is not bool:
        raise ValueError('MISSION_MANIFEST_EXECUTE_MISSING:' + directory.name)
    return value


def select_checkpoint(root, bundle, context, live_status):
    """Newest real T6 execution of this live T5, never a newer dry-run.

    Sessions are ordered by their documented session timestamp, not directory
    mtime (which changes when inspecting/copying a file). Once a matching newer
    execute is found, any missing/broken checkpoint aborts selection.
    """
    root=Path(root).resolve()
    sid=live_status.get('session_id')
    if not isinstance(sid,str) or not sid:
        raise ValueError('LIVE_COVERAGE_SESSION_MISSING')
    if live_status.get('zone_bundle_signature')!=bundle[4]:
        raise ValueError('LIVE_COVERAGE_BUNDLE_MISMATCH')
    if live_status.get('data_health')!='LIVE' or not live_status.get('map_validated') or live_status.get('fault'):
        raise ValueError('T5_NOT_LIVE')
    base=root/'logs/hospital_zone_mission'
    ignored=[]
    directories=sorted((p for p in base.glob('session_*') if p.is_dir() and SESSION.fullmatch(p.name)),
                       key=lambda p:p.name,reverse=True)
    for directory in directories:
        directory=safe_session(root,directory.name)
        try:
            execute=_session_execute(directory)
        except (OSError,ValueError,KeyError) as exc:
            # A session without enough metadata is not safely attributable.
            raise ValueError('NEWER_SESSION_METADATA_UNREADABLE:'+directory.name+':'+str(exc)) from exc
        if not execute:
            ignored.append(dict(session=directory.name,reason='DRY_RUN_NOT_MISSION_PROGRESS'))
            continue
        status_path=directory/'latest_status.json'
        status=read_json(status_path) if status_path.is_file() else None
        if status is not None:
            resume=status.get('resume_v18') or {}
            if resume.get('requested') is True and resume.get('restored') is False and status.get('sent',0)==0:
                ignored.append(dict(session=directory.name,reason='FAILED_RESUME_BEFORE_RESTORE'))
                continue
            saved_sid=status.get('coverage_session')
            if saved_sid is not None and saved_sid != sid:
                ignored.append(dict(session=directory.name,reason='DIFFERENT_T5_SESSION'))
                continue
            if saved_sid is None and status.get('sent',0)==0:
                ignored.append(dict(session=directory.name,reason='NO_COVERAGE_OR_GOAL_AT_STARTUP'))
                continue
        pointer=directory/'checkpoints_v18/LATEST.json'
        if not pointer.is_file():
            raise ValueError('LATEST_EXECUTE_HAS_NO_CHECKPOINT:'+directory.name+
                             '; refusing to roll progress back to an older session')
        try:
            # Read once for session attribution before strict context checks.
            p,digest,actual=load_checkpoint(pointer)
            if (p.get('coverage') or {}).get('source_session')!=sid:
                if status is not None and status.get('coverage_session')==sid:
                    raise ValueError('STATUS_CHECKPOINT_T5_DISAGREEMENT')
                ignored.append(dict(session=directory.name,reason='DIFFERENT_T5_SESSION'))
                continue
            if pointer.parent.is_symlink():raise ValueError('CHECKPOINT_DIRECTORY_SYMLINK')
            p,digest,actual,observed=validated_checkpoint(pointer,bundle,context)
            if p['mission'].get('source_session') not in (None,directory.name):
                raise ValueError('CHECKPOINT_SOURCE_MISSION_MISMATCH')
            if p['mission'].get('source_execute') is False:
                raise ValueError('DRY_RUN_CHECKPOINT_IN_EXECUTE_SESSION')
            if status is not None:
                counts=checked_counters(p['mission'])
                for stat,key in [('sent','sent_goals'),('plan_attempts','plan_attempts'),
                                 ('nav_succeeded','succeeded_goals'),('nav_aborted','aborted_goals')]:
                    if type(status.get(stat)) is int and status[stat]>counts[key]:
                        raise ValueError('CHECKPOINT_BEHIND_SESSION_STATUS:'+key)
            past=float(p['coverage']['status_snapshot'].get('ros_time_s',0.))
            now=float(live_status.get('ros_time_s',float('nan')))
            if not math.isfinite(now) or now+.1<past:
                raise ValueError('LIVE_CLOCK_BEHIND_CHECKPOINT')
            if live_status.get('observed_cells',-1)<p['coverage']['observed_cells']:
                raise ValueError('LIVE_COVERAGE_COUNT_REGRESSED')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise ValueError('LATEST_EXECUTE_CHECKPOINT_REFUSED:'+directory.name+':'+str(exc)) from exc
        return dict(session=directory.name,path=str(actual),pointer=str(pointer),
                    payload_sha256=digest,payload=p,ignored=ignored)
    raise ValueError('NO_EXECUTE_CHECKPOINT_FOR_LIVE_T5:'+sid+
                     '; T5 is left untouched; no legacy import or older-session fallback')


def recover_terminal_history(root, payload, bundle, context):
    """Repair old v1.8 completion loss along THIS mission lineage only.

    Read terminal facts from named ancestors. Counters, budgets, current target
    and latest Coverage are NEVER replaced with older values or summed again.
    """
    current=copy.deepcopy(payload['mission'])
    facts=completion_facts(current)
    current_counts=checked_counters(current)
    now_mask=validate_coverage(payload['coverage'],bundle)
    sources=[];warnings=[]
    if (current.get('completion_v183') or {}).get('lineage_reconciled') is True:
        current['completion_v183'].update(finished_zones=sorted(facts), reasons=facts)
        return current
    source_name=current.get('source_session')
    for name in dict.fromkeys(payload.get('mission_session_ids', [])):
        if name==source_name:continue
        directory=safe_session(root,name)
        pointer=directory/'checkpoints_v18/LATEST.json'
        if not pointer.is_file():
            # Never silently lose lifecycle facts when original data was removed.
            warnings.append('ANCESTOR_CHECKPOINT_MISSING:'+name)
            continue
        prior,ph,_,pm=validated_checkpoint(pointer,bundle,context)
        if set(prior['mission']['scope'])!=set(current['scope']):
            raise ValueError('ANCESTOR_SCOPE_MISMATCH:'+name)
        prior_counts=checked_counters(prior['mission'])
        if any(prior_counts[k]>current_counts[k] for k in COUNTERS):
            raise ValueError('ANCESTOR_COUNTER_EXCEEDS_SELECTED:'+name)
        if np.any(pm & ~now_mask):
            raise ValueError('ANCESTOR_COVERAGE_EXCEEDS_SELECTED:'+name)
        old_facts=completion_facts(prior['mission'])
        additions=set(old_facts)-set(facts)
        facts.update({z:old_facts[z] for z in additions})
        if additions:sources.append(dict(session=name,payload_sha256=ph,recovered_zones=sorted(additions)))
    if warnings:
        # Old v1.8 reset completed <96% rooms; missing ancestors could hide them.
        # Do not repeat them by pretending history is complete.
        raise ValueError(';'.join(warnings))
    current['completion_v183']=dict(finished_zones=sorted(facts),reasons=facts,
                                    recovered_ancestors=sources,counters_preserved=True)
    return current


def apply_terminal_history(node, mission):
    """Called after normal restore; lifecycle exclusion is independent of sensing."""
    facts=completion_facts(mission)
    reopened=set(unresolved_zones(mission))
    # v1.8.7: an unfinished *explicit* Zone remains resumable across any
    # number of subsequent checkpoints. Its PARTIAL_DEFERRED scheduling lock
    # must not survive a new T6 process when the lifecycle is IN_PROGRESS.
    # Only a completed lifecycle is immune to reopening.
    lifecycle=(mission.get('zone_lifecycle_v187') or {}).get('zones') or {}
    reopened.update(z for z,item in lifecycle.items()
        if item.get('exploration_status')!='COMPLETE' and (
            z in getattr(node.book,'deferred_this_mission',set()) or
            getattr(node.book.records.get(z), 'state', None)=='PARTIAL_DEFERRED'))
    reopened=sorted(reopened - set(facts))
    node._v183_finished_zones=set(facts)
    node._v184_review_reopened=set(reopened)
    book=node.book
    for zid in reopened:
        rec=book.records[zid]
        rec.state='PARTIAL' if rec.measured else 'UNOBSERVED'
        rec.deferred_reason=None
        if hasattr(book, 'deferred_this_mission'):
            book.deferred_this_mission.discard(zid)
        if hasattr(book, 'audit_requested'):
            book.audit_requested.pop(zid,None)
        old=copy.deepcopy(getattr(book,'outcomes',{}).get(zid,{}))
        if hasattr(book,'outcomes'):
            book.outcomes[zid]=dict(classification='EXPLORATION_INCOMPLETE',
                reason='UNTESTED_CANDIDATES_NEED_CONTINUATION',normal_termination=False,
                historic_assessment=old,coverage_target_met=False,
                historical_complete_label_invalidated=True)
    runtime=getattr(node,'connectivity',None)
    if runtime is not None:
        # local_reviewed means reviewed in a historic episode. It is NOT a
        # current scheduling lock when that review left untested candidates.
        runtime.local_reviewed.difference_update(reopened)
        runtime.local_reviewed.update(facts)
        runtime._v184_previous_local_reviews={z:copy.deepcopy(getattr(runtime,'local_reviews',{}).get(z)) for z in reopened}
        # Per-resume **preflight** allowance, not lifetime nav/Goal budget.
        # Preserve the old counts and attempt records before resetting a new
        # inspection episode; never reset sent_goals or mission safety limits.
        runtime._v184_review_episodes=[]
        ledger=getattr(runtime,'tasks',None)
        for zid in reopened:
            previous=int(getattr(runtime,'second_zone_preflights',{}).get(zid,0))
            affected=[]
            # v1.8.7: In the real v1.8.6 legacy checkpoint, an earlier
            # terminal-lock pass changed the task's reason to
            # ZONE_LIFECYCLE_ALREADY_FINISHED and masked the original
            # PREFLIGHT_BUDGET_UNTESTED_CANDIDATES. Recover that meaning only
            # when the immutable saved local review still contains the original
            # unfinished-candidate reason. Keep all previous attempt counts
            # in prior_preflight_episodes; reset only this new review allowance.
            review=(((mission.get('connectivity') or {}).get('local_reviews') or {}).get(zid) or {}).get('results') or []
            unfinished_review=any(isinstance(item,dict) and
                'PREFLIGHT_BUDGET_UNTESTED_CANDIDATES' in str(item.get('reason',''))
                for item in review)
            if ledger is not None:
                for task in ledger.tasks.values():
                    if task.get('zone_id') != zid:continue
                    reason=str(task.get('reason',''))
                    if ('PREFLIGHT_BUDGET_UNTESTED_CANDIDATES' in reason or
                            (reason == 'ZONE_LIFECYCLE_ALREADY_FINISHED' and unfinished_review)):
                        count=int(task.get('checks',0))
                        task.setdefault('prior_preflight_episodes',[]).append(dict(
                            previous_checks=count,reason=reason,cost_key=task.get('source_cost_key'),
                            attempts=copy.deepcopy(task.get('attempts',[]))))
                        # Historical failures are kept for audit, NOT replayed as
                        # present-costmap failures when evaluating a new catalog.
                        task['attempts']=[]
                        task['checks']=0
                        task['restarts']=0
                        task['status']='PENDING'
                        affected.append(task.get('task_id'))
            if affected:
                if hasattr(runtime,'second_zone_preflights'):
                    runtime.second_zone_preflights[zid]=0
            runtime._v184_review_episodes.append(dict(zone=zid,
                previous_zone_preflights=previous,reset_for_new_session=bool(affected),
                resumed_task_ids=affected))
        if hasattr(node,'event') and reopened:
            node.event('ZONE_FALSE_COMPLETION_REOPENED_V184', zones=reopened,
                reason='UNTESTED_CANDIDATES_AFTER_PREFLIGHT_BUDGET',
                past_goal_counts_unchanged=True,old_coverage_unchanged=True)
        if getattr(runtime,'local_second_zone',None) in facts:
            runtime.local_second_zone=None
            runtime.second_task=runtime.second_catalog=runtime.second_context=None
            runtime.exploration_stage='PRIMARY'
            runtime.primary_zone=None
        saved = mission.get('connectivity') or {}
        pending_local = saved.get('primary_zone')
        if (saved.get('stage') == 'PRIMARY' and pending_local in node.zones_by_id
                and pending_local not in facts and pending_local not in reopened
                and mission['records'][pending_local].get('state') == 'PARTIAL_DEFERRED'):
            runtime.primary_zone = pending_local  # finish its local review, not PRIMARY again
        # Keep history but do not leave terminal tasks runnable.
        ledger=getattr(runtime,'tasks',None)
        if ledger is not None:
            for t in ledger.tasks.values():
                if t.get('zone_id') in facts and t.get('status') not in {
                    'OBSERVED','SMALL_REMAINDER','ROBOT_EXPLORATION_UNAVAILABLE','UNVERIFIED','SPLIT'}:
                    t.update(status='UNVERIFIED',reason='ZONE_LIFECYCLE_ALREADY_FINISHED',
                             completion_scope='retained_historical_task_not_scheduled')
    # The completion adapter will restore operational COMPLETE for <96% zones.
    if hasattr(node,'_dn012_finalize_reviewed'):node._dn012_finalize_reviewed()
    return sorted(facts)


def spatial_zone(bundle, pose):
    g=bundle[1]
    if not isinstance(pose,(list,tuple,np.ndarray)) or len(pose)<3 or not np.isfinite(pose[:3]).all():
        raise ValueError('INVALID_RESUME_CURRENT_POSE')
    i,j=map(math.floor,g.world_to_grid_float(float(pose[0]),float(pose[1])))
    if not g.contains(i,j):return None
    mask_id=int(bundle[2][j,i])
    return next((z['id'] for z in bundle[0]['zones'] if z['mask_id']==mask_id),None)


def current_pose_decision(node,mission,inp,bundle,cfg,inside_check):
    pose=list(map(float,inp['pose'][:3]));zid=spatial_zone(bundle,pose)
    facts=completion_facts(mission)
    facts.update({z:'CURRENT_COMPLETION' for z in getattr(node,'_v183_finished_zones',())})
    eligible={z['id'] for z in node.book.eligible()}-set(facts)
    runtime=getattr(node,'connectivity',None)
    unfinished_second=(getattr(runtime,'local_second_zone',None)
                       if getattr(runtime,'exploration_stage',None)=='SECOND' else None)
    choice=None
    if unfinished_second in eligible:
        choice=unfinished_second
        reason='CONTINUE_UNFINISHED_LOCAL_PROCEDURE_FROM_CURRENT_POSE'
    elif unfinished_second is not None:
        # A second-pass Zone may be excluded by the normal primary scheduler.
        choice=unfinished_second if unfinished_second not in facts else None
        reason='PRESERVE_UNFINISHED_SECOND_PASS'
    elif (getattr(runtime, 'primary_zone', None) not in facts
          and getattr(runtime, 'primary_zone', None) in node.zones_by_id
          and node.book.records[runtime.primary_zone].state == 'PARTIAL_DEFERRED'):
        choice=runtime.primary_zone;reason='CONTINUE_PENDING_ZONE_LOCAL_REVIEW_FROM_CURRENT_POSE'
    elif zid in eligible and inside_check(node.zones_by_id[zid],tuple(pose),bundle,cfg):
        node.book.activate(zid);node.book.mark_entered(zid)
        if runtime is not None:runtime.primary_zone=zid
        choice=zid;reason='CURRENT_UNFINISHED_ZONE_FIRST'
    else:
        reason='EXISTING_SELECTOR_FROM_CURRENT_POSE_NO_RETURN_TO_OLD_POSE'
    old=(mission.get('pose_snapshot_v183') or {}).get('pose')
    displacement=None
    if isinstance(old,list) and len(old)>=2 and np.isfinite(old[:2]).all():
        displacement=float(math.dist(old[:2],pose[:2]))
    return dict(version=VERSION,pose_map=pose,spatial_zone=zid,
                preferred_zone=choice,selection_reason=reason,
                finished_zones=sorted(facts),remaining_zones=sorted(set(node.zones_by_id)-set(facts)),
                displacement_from_checkpoint_m=displacement,
                displacement_is_not_proof_of_manual_intervention=True,
                old_goal_replayed=False,old_pose_targeted=False)


class LiveReadiness:
    """Testable passive preflight; runtime still owns the final safety checks."""
    def __init__(self,now=0.):
        self.started=now;self.clock=None;self.clock_wall=None;self.clock_first=None
        self.clock_reset=False;self.status=None;self.status_wall=None;self.first_evidence=None
        self.evidence_advanced=False;self.action=None;self.action_wall=None;self.sessions=set()
    def on_clock(self,t,wall):
        if not math.isfinite(t) or t<=0: return
        if self.clock is not None and t<self.clock-.1:self.clock_reset=True
        if self.clock_first is None:self.clock_first=t
        if self.clock is None or t>self.clock:self.clock_wall=wall
        self.clock=t
    def on_coverage(self,s,wall):
        if not isinstance(s,dict) or s.get('schema')!='hospital_room_coverage_v1':return
        sid=s.get('session_id')
        if not isinstance(sid,str) or not sid:return
        self.sessions.add(sid)
        t=s.get('last_evidence_scan_end_s')
        seq=s.get('evidence_scans')
        if type(seq) is not int or not isinstance(t,(int,float)) or not math.isfinite(t):return
        if self.first_evidence is None:self.first_evidence=(seq,t)
        if seq>self.first_evidence[0] and t>self.first_evidence[1]:self.evidence_advanced=True
        self.status=copy.deepcopy(s);self.status_wall=wall
    def on_actions(self,states,wall):
        self.action=list(states);self.action_wall=wall
    def check(self,wall,coverage_publishers=1,action_publishers=1):
        if coverage_publishers>1:raise ValueError('MULTIPLE_T5_COVERAGE_PUBLISHERS')
        if action_publishers>1:raise ValueError('MULTIPLE_NAVIGATION_STATUS_PUBLISHERS')
        if len(self.sessions)>1:raise ValueError('T5_SESSION_CHANGED_DURING_RESUME')
        if self.clock_reset:raise ValueError('ROS_CLOCK_RESET_DURING_RESUME')
        if coverage_publishers!=1:return False,'WAIT_T5_COVERAGE_PUBLISHER'
        if action_publishers!=1:return False,'WAIT_NAV2_STATUS_PUBLISHER'
        if self.action is None:return False,'WAIT_NAV2_GOAL_STATUS'
        if any(type(s) is not int or s not in range(1,7) for s in self.action):
            raise ValueError('NAVIGATION_STATUS_UNKNOWN')
        if any(s in (1,2,3) for s in self.action):
            raise ValueError('NAVIGATION_GOAL_STILL_ACTIVE; cancel old Goal and stop before resume')
        if self.clock is None or self.clock_wall is None or self.clock<=self.clock_first or wall-self.clock_wall>2.:
            return False,'WAIT_ADVANCING_SIMULATION_CLOCK'
        s=self.status
        if s is None or wall-self.status_wall>5.:return False,'WAIT_FRESH_T5_STATUS'
        if s.get('data_health')!='LIVE' or not s.get('map_validated') or s.get('fault'):
            return False,'WAIT_T5_LIVE_MAP_SCAN'
        stamp=s.get('ros_time_s')
        if not isinstance(stamp,(float,int)) or not math.isfinite(stamp) or not -.1<=self.clock-stamp<=5.:
            return False,'WAIT_CURRENT_T5_TIMESTAMP'
        if not self.evidence_advanced:return False,'WAIT_NEW_T5_SCAN_NOT_LATCHED_HISTORY'
        if wall-self.started<2.:return False,'WAIT_DISCOVERY_WINDOW'
        return True,'LIVE_T5_AND_NAVIGATION_IDLE'
