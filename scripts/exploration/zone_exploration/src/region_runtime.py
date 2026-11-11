"""DN-012 v1.6: per-Zone PRIMARY then immediate residual INTERIOR ENTRY.

No new ROS interfaces. The original action owner, local start, fresh data,
post-result stop, coverage confirmation, cancellation and global budgets remain
in zone_node. Hooks in the guarded installer are required; never copy this file
alone into another version of the project.
"""
from __future__ import annotations
from dataclasses import asdict
import math
import os
import time
import numpy as np
from region_legacy import make_runtime as make_legacy
from regions import row_runs
from two_phase import (VERSION, TERMINAL, STALE_REASONS, ACCESS_REJECTIONS,
    TwoPassSettings, TaskLedger, finite_json,
    path_guard, endpoint_guard, remaining_path, refresh_second_pass)
from planning_core import oriented_footprint_clear
from exploration_io import atomic_json
from region_entry import build_entry_catalog, validate_entry_endpoint, entry_membership
from path_blockage import enabled_for_candidate
from exploration_results import project_results


def make_runtime(base):
    legacy = make_legacy(base)
    class TwoPhaseConnectivityRuntime(legacy):
        def __init__(self,node,bundle,cfg,args,output):
            if os.getenv('DN012_REGION_TWO_PASS') != '1':
                raise ValueError('V1_2_REQUIRES_SOURCE_NEW_ENABLE_SCRIPT: DN012_REGION_TWO_PASS=1')
            super().__init__(node,bundle,cfg,args,output)
            self.second_settings=TwoPassSettings.from_environment()
            self.exploration_stage='DIAGNOSTIC' if self.region_mode=='diagnostic' else 'PRIMARY'
            self.tasks=TaskLedger(bundle,self.second_settings)
            self.primary_reservations={}
            self.primary_zone=None;self.local_second_zone=None
            self.local_reviewed=set();self.local_reviews={}
            self.second_task=None;self.second_catalog=None;self.second_context=None
            self.second_future=None;self.second_future_task=None
            self.second_attempted={};self.second_zone_preflights={};self.second_zone_nav={}
            self.second_zone_work={};self.second_last_account=None
            # DN012_GLOBAL_BLOCKAGE_V1_3: optional memory; no new Nav2 interfaces.
            self.blockage_memory=None
            if os.environ.get('DN012_BLOCKAGE_MEMORY')=='1':
                from pathlib import Path as _Path
                from blockage_memory import BlockageMemory
                zone_names={int(z['mask_id']):z['id'] for z in self.n.zones_by_id.values()}
                self.blockage_memory=BlockageMemory(_Path(output)/'memory/blockage_points.json',
                    signature=self.bundle[4],geometry=self.bundle[1],labels=self.bundle[2],
                    zone_mask_to_name=zone_names,
                    hold_radius_m=float(os.environ.get('DN012_BLOCKAGE_MEMORY_RADIUS_M','.40')))
                self.blockage_memory._save()
                self.n.event('BLOCKAGE_MEMORY_POLICY',version='DN012_GLOBAL_BLOCKAGE_MEMORY_V1_3',
                    saved_path=str(self.blockage_memory.path),
                    loaded_points=self.blockage_memory.loaded_count,
                    historical_requires_current_costmap=True,
                    physical_blockage_not_inferred=True)
                self.n.info('blockage_memory','[BLOCKAGE_MEMORY_POLICY] session point memory enabled; '+
                    str(self.blockage_memory.loaded_count)+' previous witnesses loaded',0)
            self.failed_positions=[];self.failed_routes=[];self.region_cancel=None
            self.second_monitor_wall=-math.inf;self.last_live_signature=None
            self.last_inventory_error=None
            self.n.event('TWO_PHASE_POLICY',version=VERSION,stage=self.exploration_stage,
                settings=asdict(self.second_settings),first_pass_rescue_disabled=False,
                schedule="ZONE_LOCAL_IMMEDIATE",global_second_pass=False,direct_approach=False,residual_entry=True,
                coverage_confirmation_unchanged=True,second_pass_includes_confirmed_zones=True,
                no_auto_revisit_bypassed_only_for_bounded_second_pass=True,
                safety_rejection_is_not_physical_blockage=True)
            self.n.info('two_phase_policy',f'[TWO_PHASE_POLICY] {VERSION}; '
                f'stage={self.exploration_stage}; Zone PRIMARY -> immediate INTERIOR ENTRY; no global second sweep',0)

        # DN012_GLOBAL_BLOCKAGE_V1_3: shared between primary and second pass.
        def on_blockage_memory_event(self,kind,fields):
            mem=self.blockage_memory
            if mem is None:return
            if kind=='PATH_BLOCKED_EVIDENCE':
                e=fields.get('evidence',{})
                ident,added=mem.detect(e,goal_id=fields.get('goal_id'),
                    ros_time_s=self.n.ros_s())
                self.n.event('BLOCKAGE_MEMORY_RECORDED',record_id=ident,new_record=added,
                    xy=list(e['xy']),destination_zone=e.get('zone'),
                    physical_passage_cut_proven=False)
            elif kind=='PATH_BLOCKAGE_STOP_CONFIRMED':
                ident=mem.confirm_stop(fields.get('record',{}))
                if ident:self.n.event('BLOCKAGE_MEMORY_STOP_CONFIRMED',record_id=ident,
                    zone=fields.get('zone'),physical_passage_cut_proven=False)

        def blockage_memory_guard(self,c,inp,points,stage):
            mem=self.blockage_memory
            if getattr(self,'blockage_memory_fault',None):
                raise ValueError('BLOCKAGE_MEMORY_PERSISTENCE_OR_VALIDATION_FAULT')
            if mem is None or not c or c.get('connectivity_probe'):return False
            if not enabled_for_candidate(c,self.n,self.bundle,self.cfg,inp):return False
            # inp is already validated by the existing original state machine.
            hit=mem.review_path(points,inp['cost']['values'],stage=stage)
            if hit is None:return False
            self.n.event('BLOCKAGE_MEMORY_ROUTE_HELD',zone=c.get('zone_id'),
                candidate_id=c.get('id'),evidence=hit,
                navigation_sent=False,not_an_area_cut_certificate=True)
            return True

        def _is_second(self,c):
            return bool(c and c.get('second_pass') and self.exploration_stage=='SECOND'
                and c.get('second_task_id') in self.tasks.tasks
                and self.tasks.tasks[c['second_task_id']]['zone_id']==c.get('zone_id'))

        def on_local_zone_event(self,event,fields):
            # Passive hook from zone_node.event; no motion, no recursive events.
            if self.exploration_stage!='PRIMARY':return
            c=fields.get('candidate') or {}
            zid=fields.get('zone') or c.get('zone_id')
            if event in ('ZONE_SELECTED','NAV_REQUESTED') and not c.get('second_pass'):
                if zid in self.n.zones_by_id and zid not in self.local_reviewed:
                    self.primary_zone=zid

        def _track_primary(self):
            if self.exploration_stage!='PRIMARY':return
            c=self.n.current or {}
            zid=(c.get('zone_id') if not c.get('second_pass') else None) or self.n.book.active_zone
            if zid in self.n.zones_by_id and zid not in self.local_reviewed:
                self.primary_zone=zid

        def finish_partial(self,reason,evidence=None):
            if self.exploration_stage=='PRIMARY' and self.zid:
                self.primary_zone=self.zid
                self.primary_reservations[self.zid]=dict(reason=reason,ros_time_s=self.n.ros_s())
                self.n.event('LOCAL_SECOND_PASS_RESERVED',zone=self.zid,reason=reason,
                    decision='REVIEW_THIS_ZONE_BEFORE_NEXT_ZONE')
            return base.finish_partial(self,reason,evidence)

        def _drain_workers(self):
            # Discard completed primary catalogs, never execute stale entry goals.
            for obj,name in ((self.n,'worker'),(self,'future')):
                f=getattr(obj,name,None)
                if f is None:continue
                if not f.done():return False
                try:f.result()
                except TimeoutError:pass  # A budget exception is not a cut certificate.
                setattr(obj,name,None)
            self.n.job_context=None
            return True

        def _maybe_begin_local(self,inp):
            self._track_primary()
            zid=self.primary_zone
            if zid is None or zid in self.local_reviewed:return False
            r=self.n.book.records[zid]
            if r.state=='THRESHOLD_PENDING':return True
            if r.state not in ('OBSERVATION_SUFFICIENT','OBSERVATION_LIMITED','PARTIAL_DEFERRED'):
                return False
            if not self._drain_workers():return True
            self._begin_second(inp,zid)
            return True

        def step(self,inp):
            if self.exploration_stage=='DIAGNOSTIC':return base.step(self,inp)
            if self.exploration_stage=='SECOND':return self.selection_hook(inp)
            if self._maybe_begin_local(inp):return True
            active=self.n.book.active_zone
            pending=active if active in self.n.book.audit_requested else next(iter(self.n.book.audit_requested),None)
            if pending and self.n.book.visit_state[pending].entered:
                reason=self.n.book.audit_requested[pending]
                if not reason.startswith('ENTRY_'):
                    self.zid=pending
                    self.finish_partial('PRIMARY_DONE_LOCAL_APPROACH_PENDING',dict(
                        trigger=reason,normal_termination=False,connectivity_verified=False))
                    self.n.options=[];self.n.batch_zone=None;self.n.options_context=None
                    return True
            return base.step(self,inp)

        def residual_step(self,inp):return None

        def consider_low_gain_plateau(self,record,coverage_session,confirmed_inside):
            if self.exploration_stage in ('PRIMARY','DIAGNOSTIC'):
                self.primary_zone=record['zone_id']
                return base.consider_low_gain_plateau(self,record,coverage_session,confirmed_inside)
            return False

        def _inventory(self,inp):
            # Report only: enumerate unreviewed Zones on shutdown, never navigate.
            for zone in self.n.zones_by_id.values():self.tasks.inventory_zone(inp,zone)

        def _begin_second(self,inp,zid):
            if zid in self.local_reviewed:raise RuntimeError('LOCAL_ZONE_ALREADY_REVIEWED')
            self.tasks.inventory_zone(inp,self.n.zones_by_id[zid])
            self.local_second_zone=zid;self.exploration_stage='SECOND'
            self.n.options=[];self.n.batch_zone=None;self.n.options_context=None
            self.n.book.active_zone=None;self.n.book.locked_zone=None
            self.zid=None;self.clear_round()
            self.n.event('LOCAL_SECOND_PASS_START',zone=zid,version=VERSION,
                meaningful_regions=sum(t['zone_id']==zid for t in self.tasks.pending()),
                global_second_pass=False,coverage_not_reset=True)
            self.n.info('second_start',f'[LOCAL_SECOND_PASS_START] {zid}; interior ENTRY; no global revisit',0)

        def _end_local(self,inp):
            zid=self.local_second_zone
            if zid is None:raise RuntimeError('MISSING_LOCAL_SECOND_ZONE')
            self.local_reviews[zid]=dict(ended_ros_s=self.n.ros_s(),
                measured_percent=100*self.n.book.records[zid].measured,
                results=[dict(task_id=t['task_id'],status=t['status'],reason=t['reason'])
                         for t in self.tasks.tasks.values() if t['zone_id']==zid])
            self.local_reviewed.add(zid)
            if not self.n.book.is_complete(zid) and self.n.book.records[zid].state!='OBSERVATION_LIMITED':
                self.n.book.finish_partial(zid,'LOCAL_APPROACH_REVIEW_FINISHED',evidence=self.local_reviews[zid])
            self.n.book.audit_requested.pop(zid,None)
            self._drop_current_task()
            if self.n.book.active_zone==zid:self.n.book.active_zone=self.n.book.locked_zone=None
            self.zid=None;self.clear_round()
            self.primary_zone=None;self.local_second_zone=None;self.exploration_stage='PRIMARY'
            self.n.current=None;self.n.options=[];self.n.batch_zone=None;self.n.options_context=None
            self.n.phase='SELECT_ZONE'
            self.n.event('LOCAL_ZONE_REVIEW_FINISHED',zone=zid,next_action='NEXT_PRIMARY_ZONE',
                global_second_pass=False)
            self.n.info('local_finished',f'[LOCAL_ZONE_REVIEW_FINISHED] {zid}; next primary Zone; no global second sweep',0)
            self.n.transition('SELECTING')

        def _drop_current_task(self):
            zid=None if self.second_task is None else self.tasks.tasks[self.second_task]['zone_id']
            if zid and self.n.book.active_zone==zid:
                self.n.book.active_zone=None;self.n.book.locked_zone=None
            if zid:self.n.book.audit_requested.pop(zid,None)
            self.second_task=None;self.second_catalog=None;self.second_context=None

        def _finish_task(self,status,reason,inp=None):
            if self.second_task is None:return
            tid=self.second_task;t=self.tasks.tasks[tid]
            self.tasks.finish(tid,status,reason,inp)
            t['last_update_ros_s']=float(self.n.ros_s())
            self.n.event('SECOND_PASS_REGION_RESULT',task_id=tid,zone=t['zone_id'],
                status=status,reason=reason,remaining_area_m2=t['area_m2'],
                checks=t['checks'],nav_sent=t['nav_sent'],
                physical_blockage_proven=False,scope='this_robot_policy_sampled_candidates')
            self._drop_current_task()

        def budget_deferred(self,inp):
            if self.exploration_stage=='SECOND' and self.second_task:
                self._finish_task('UNVERIFIED','EXISTING_MISSION_GUARD_LIMIT',inp)
                self.n.current=None;self.n.options=[];self.n.current_path=[]
                self.n.current_path_length=None;self.n.validated_plan_mode=None
                self.n.batch_zone=None;self.n.options_context=None
                self.n.transition('SELECTING')
                return True
            return False

        def _global_limit(self):
            if self.n.sent_goals>=self.args.max_goals:return 'GOAL_LIMIT_REACHED'
            if self.n.plan_attempts>=self.args.max_plan_attempts:return 'PLAN_ATTEMPT_LIMIT_PARTIAL'
            return None

        def _finish_all(self,reason=None):
            if reason is not None:
                for t in self.tasks.pending():self.tasks.finish(t['task_id'],'UNVERIFIED',reason,self.region_last_inp)
            # Remaining never-visited/passively covered Zones are report-only.
            inp=self.region_last_inp
            if inp is not None:
                self._inventory(inp);self.tasks.refresh(inp)
                for t in self.tasks.pending():
                    self.tasks.finish(t['task_id'],'UNVERIFIED',reason or 'NO_GLOBAL_SECOND_PASS_NOT_LOCALLY_REVIEWED',inp)
            self._drop_current_task();self.exploration_stage='DONE'
            self.n.event('EXPLORATION_STAGE',stage='DONE',version=VERSION,
                region_counts=self._counts(),all_regions_reviewed=not any(
                    t['status']=='UNVERIFIED' for t in self.tasks.tasks.values()))
            if reason:self.n.finish(reason);return
            unresolved=any(t['status'] in ('UNVERIFIED','ROBOT_EXPLORATION_UNAVAILABLE') for t in self.tasks.tasks.values())
            if unresolved:self.n.finish('PARTIAL_DEFERRED_REMAIN')
            elif self.n.book.all_complete():self.n.finish('ALL_REQUESTED_ZONES_OBSERVED')
            elif self.n.book.all_resolved():self.n.finish('ALL_ZONES_ASSESSED_WITH_LIMITS')
            else:self.n.finish('PARTIAL_DEFERRED_REMAIN')

        def on_mission_finish(self,state):
            # Called before original summary write. Enumerate anything not reached
            # because of global budget/stop; never hide it as inaccessible/completed.
            if self.exploration_stage in ('DIAGNOSTIC','DONE'):return
            err,inp=self.n.inputs()
            if err is None and inp is not None:
                self._inventory(inp);self.tasks.refresh(inp)
            else:self.last_inventory_error=str(err or 'NO_FRESH_INPUT')
            for t in self.tasks.pending():
                self.tasks.finish(t['task_id'],'UNVERIFIED','MISSION_ENDED_'+state,inp)
            self.exploration_stage='STOPPED'

        def _counts(self):
            counts={}
            for t in self.tasks.tasks.values():counts[t['status']]=counts.get(t['status'],0)+1
            return counts

        def _context(self,inp):
            from connectivity_audit import sensor_signature
            return dict(cost_key=inp['cost']['key'],epoch=inp['epoch'],
                session=inp['status']['session_id'],pose=list(inp['pose']),
                sensor_signature=sensor_signature(inp['sensor']))

        def _matches(self,ctx,inp):
            from connectivity_audit import sensor_signature
            return bool(ctx and ctx['cost_key']==inp['cost']['key'] and ctx['epoch']==inp['epoch']
                and ctx['session']==inp['status']['session_id']
                and ctx['sensor_signature']==sensor_signature(inp['sensor'])
                and math.dist(ctx['pose'][:2],inp['pose'][:2])<=.15
                and abs(math.atan2(math.sin(ctx['pose'][2]-inp['pose'][2]),
                                  math.cos(ctx['pose'][2]-inp['pose'][2])))<=.15)

        def selection_hook(self,inp):
            """Installed before early all_complete exit; no budget/safety bypass."""
            if getattr(self,'blockage_memory_fault',None):
                self.n.request_stop('BLOCKAGE_MEMORY_PERSISTENCE_OR_VALIDATION_FAULT')
                return True
            if self.exploration_stage in ('DIAGNOSTIC','DONE','STOPPED'):return False
            n=self.n
            self.region_last_inp=inp
            if n.stop_requested or n.done:return True
            if (n.nav_uuid is not None or n.plan_send is not None or n.plan_handle is not None
                    or n.plan_result is not None or getattr(n,'state',None)=='READY_TO_SEND'):
                return True
            if self.exploration_stage=='PRIMARY':
                if self._maybe_begin_local(inp):return True
                if n.book.eligible() or n.book.pending() or n.book.audit_requested:return False
                if not self._drain_workers():return True
                self._finish_all();return True
            self.tasks.refresh(inp)
            limit=self._global_limit()
            if limit:self._finish_all(limit);return True
            if not n.plan_client.server_is_ready() or (self.args.execute and not n.nav_client.server_is_ready()):
                n.info('second_servers','[WAIT_NAV2_ACTION_SERVERS] second pass');return True
            if time.monotonic()-n.boot_wall<2.:return True
            if not n.stop_evidence.ready(n.ros_s(),time.monotonic()):
                n.info('second_stop','[WAIT_STOP] before second-pass checks');return True
            if n.worker is not None:
                if not n.worker.done():return True
                n.collect_job(inp)
                if n.stop_requested:return True
            if self.second_task and self.tasks.tasks[self.second_task]['status'] in TERMINAL:
                self._drop_current_task()
            if self.second_future is not None:
                if not self.second_future.done():return True
                future=self.second_future;tid=self.second_future_task
                self.second_future=None;self.second_future_task=None
                if tid!=self.second_task:
                    # Consume any error, but never install another task's catalog.
                    try:future.result()
                    except Exception:pass
                    return True
                try:built=future.result()
                except TimeoutError as exc:
                    self._finish_task('UNVERIFIED',str(exc),inp);return True
                except Exception:
                    raise  # Expose genuine implementation faults; do not certify anything.
                if not self._matches(self.second_context,inp):
                    t=self.tasks.tasks[tid];t['restarts']+=1;self.second_catalog=None
                    n.event('SECOND_PASS_CATALOG_STALE',task_id=tid,restarts=t['restarts'])
                    if t['restarts']>=self.second_settings.max_catalog_restarts:
                        self._finish_task('UNVERIFIED','SECOND_PASS_CONTEXT_RESTART_BUDGET',inp)
                    return True
                self.second_catalog=built
                self.tasks.tasks[tid]['catalog_stats']=built['stats']
                n.event('SECOND_PASS_SAFE_CATALOG',task_id=tid,**finite_json(built['stats']))
            if self.second_task is None:
                t=self.tasks.next_task(inp['pose'],self.local_second_zone)
                if t is None:self._end_local(inp);return True
                self.second_task=t['task_id'];t['status']='INSPECTING'
                t['assessment_subject']='REGION_ENTRY'
                t['last_update_ros_s']=float(n.ros_s());t['source_cost_key']=inp['cost']['key']
                n.event('SECOND_PASS_REGION_CHECK',task_id=t['task_id'],zone=t['zone_id'],
                    area_m2=t['area_m2'],checks=t['checks'])
            t=self.tasks.tasks[self.second_task];zid=t['zone_id']
            n.book.activate(zid)  # Work accounting only; does not reopen original eligibility.
            if n.mission_guard and n.mission_guard.before_selection(inp):
                self.budget_deferred(inp);return True
            original_count=n.book.records[zid].views+n.book.nav_abort_counts.get(zid,0)
            if original_count>=self.args.zone_hard_goals:
                self._finish_task('UNVERIFIED','EXISTING_ZONE_HARD_GOAL_LIMIT',inp);return True
            if self.second_zone_nav.get(zid,0)>=self.second_settings.max_nav_per_zone:
                self._finish_task('UNVERIFIED','SECOND_PASS_ZONE_NAV_BUDGET',inp);return True
            if t['nav_sent']>=self.second_settings.max_nav_per_region:
                self._finish_task('UNVERIFIED','SECOND_PASS_REGION_NAV_BUDGET',inp);return True
            if self.second_zone_work.get(zid,0.)>=min(self.settings.zone_budget_s,
                    float(getattr(self.args,'guard_zone_work_max_s',self.settings.zone_budget_s))):
                self._finish_task('UNVERIFIED','SECOND_PASS_AUDIT_WORK_BUDGET',inp);return True
            # Build even with no motion budget left only when early global limit
            # has not fired. Uninspected tasks are always marked UNVERIFIED.
            if self.second_catalog is not None and not self._matches(self.second_context,inp):
                t['restarts']+=1;self.second_catalog=None
                if t['restarts']>=self.second_settings.max_catalog_restarts:
                    self._finish_task('UNVERIFIED','SECOND_PASS_CONTEXT_RESTART_BUDGET',inp);return True
            if self.second_catalog is None:
                snap=n.snapshot(inp);self.second_context=self._context(inp)
                region=dict(t,indices=t['indices'].copy())
                held=list(self.failed_positions)
                held.extend(self._ledger(zid)['low_yield_positions'])
                self.second_future=n.pool.submit(build_entry_catalog,self.bundle,self.cfg,self.settings,
                    self.second_settings,snap,n.zones_by_id[zid],region,n.zone_planner.visibility,
                    oriented_footprint_clear,n.zone_planner.cancel,held,
                    entry_inset=getattr(self.args,'entry_inset_m',.45),
                    arrival_buffer=getattr(self.args,'entry_arrival_buffer_m',.20))
                self.second_future_task=self.second_task
                n.transition('AUDITING_ZONE');return True
            attempted=self.second_attempted.setdefault(self.second_task,set())
            available=[]
            for c in self.second_catalog['views']:
                if c['id'] in attempted:continue
                fresh=refresh_second_pass(c,inp['observed'],self.bundle[1])
                if fresh is not None:available.append(fresh)
            if not available:
                verdict,reason=self.tasks.exhaustion_verdict(self.second_task,self.second_catalog['stats'])
                self._inspect(zid,inp)  # independent sensor-supported cut evidence, never from failed planning
                self._finish_task(verdict,reason,inp);return True
            if (t['checks']>=self.second_settings.max_preflights_per_region or
                    self.second_zone_preflights.get(zid,0)>=self.args.audit_max_view_preflights):
                self._finish_task('UNVERIFIED','SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES',inp);return True
            if not n.local_start_clear(inp):
                n.request_stop('LOCAL_START_NOT_CLEAR_OR_STALE');return True
            c=available[0];t['checks']+=1;t['status']='APPROACH_CHECKING'
            self.second_zone_preflights[zid]=self.second_zone_preflights.get(zid,0)+1
            attempted.add(c['id'])
            c=dict(c,second_preflight_index=t['checks'])
            n.event('SECOND_PASS_ENTRY_SELECTED',task_id=self.second_task,zone=zid,
                goal_kind='ENTRY',residual_entry=True,xy=[c['x'],c['y']],
                entry_target_xy=c['entry_target_xy'],goal_inside_residual=True,
                boundary_depth_m=c['entry_boundary_depth_m'],
                requires_original_nav2_preflight=True,waypoint_used=False)
            n.info('entry_pick',f'[SECOND_PASS_ENTRY_SELECTED] {zid}; '
                f'internal_goal=({c["x"]:.3f},{c["y"]:.3f}); task={self.second_task}; no waypoint',0)
            n.plan_request(c,inp)
            return True

        def validate_second_plan(self,c,inp,points):
            if not self._is_second(c):raise ValueError('SECOND_PASS_CANDIDATE_NOT_OWNED')
            if not self._matches(self.second_context,inp):raise ValueError('SECOND_PASS_CONTEXT_CHANGED')
            if c.get('residual_entry'):
                c['second_path_approved']=False
                # zone_node has already checked the actual Nav2 route and padded endpoint.
                # Use the SAME entry standards; do not apply viewpoint-only rotation disks.
                e=validate_entry_endpoint(c,self.bundle,inp,self.cfg,oriented_footprint_clear)
                # Repeat the SAME baseline route check at the actual send boundary.
                # No new sweep, rotation envelope, or distance limit is introduced.
                from sequence_core import validate_nav_path
                e['path_length_m']=validate_nav_path(points,'map',c,inp['pose'],
                    self.bundle[1],self.bundle[3],inp['cost']['values'])
                c['second_path_check']=finite_json(e);c['second_path_approved']=True
                self.n.event('SECOND_PASS_ENTRY_PATH_CHECK',task_id=c['second_task_id'],
                    candidate_id=c['id'],passed=True,evidence=finite_json(e))
                return True
            ok,reason,e=path_guard(points,inp['pose'],c,self.bundle,inp,self.cfg,
                self.second_settings,oriented_footprint_clear,self.failed_routes)
            outside=False
            for x,y in points:
                ii,jj=map(math.floor,self.bundle[1].world_to_grid_float(x,y))
                if not self.bundle[1].contains(ii,jj) or self.bundle[2][jj,ii]!=self.n.zones_by_id[c['zone_id']]['mask_id']:
                    outside=True;break
            self.n.event('SECOND_PASS_PATH_CHECK',task_id=c['second_task_id'],candidate_id=c['id'],
                path_left_target_zone=bool(outside),planned_not_yet_executed=True,
                passed=bool(ok),reason=reason,evidence=finite_json(e),physical_blockage_proven=False)
            c['second_path_check']=finite_json(e)
            if not ok:
                if 'xy' in e:c['second_failed_xy']=list(e['xy'])
                raise ValueError(reason)
            c['second_path_approved']=True
            return True

        def before_second_send(self,c,inp,points):
            # Fresh endpoint/path check after the original map/pose consistency gate.
            if not self._is_second(c):raise ValueError('SECOND_PASS_CANDIDATE_NOT_OWNED')
            fresh=refresh_second_pass(c,inp['observed'],self.bundle[1])
            if fresh is None:
                raise ValueError('GOAL_NO_LONGER_NEEDED')
            c.update(fresh)
            return self.validate_second_plan(c,inp,points)

        def on_plan_rejected(self,c,reason):
            if not self._is_second(c):return
            t=self.tasks.tasks[c['second_task_id']]
            record=dict(candidate_id=c['id'],reason=str(reason),xy=[float(c['x']),float(c['y'])],
                        stage='PREFLIGHT',physical_blockage_proven=False,
                         cost_key=None if self.region_last_inp is None else self.region_last_inp['cost']['key'],
                         start_xy=None if self.region_last_inp is None else list(self.region_last_inp['pose'][:2]))
            t['attempts'].append(record)
            if reason in STALE_REASONS:
                self.second_catalog=None
                self.second_attempted.get(t['task_id'],set()).discard(c['id'])
                t['restarts']+=1
                if t['restarts']>=self.second_settings.max_catalog_restarts:
                    self._finish_task('UNVERIFIED','SECOND_PASS_CONTEXT_RESTART_BUDGET',self.region_last_inp)
            elif reason in ACCESS_REJECTIONS:
                xy=list(c.get('second_failed_xy',record['xy']))
                if math.dist(xy,self.region_last_inp['pose'][:2])>.6:
                    self.failed_routes.append(dict(xy=xy,radius_m=.30,reason=str(reason)))
            self.n.event('SECOND_PASS_APPROACH_REJECTED',task_id=t['task_id'],**record)

        def audit_observation_goal_sent(self,c):
            if not self._is_second(c):return base.audit_observation_goal_sent(self,c)
            if c.get('second_path_approved') is not True:
                raise RuntimeError('SECOND_PASS_NAV_WITHOUT_SAFETY_APPROVAL')
            t=self.tasks.tasks[c['second_task_id']];t['nav_sent']+=1;t['status']='NAVIGATING'
            zid=c['zone_id'];self.second_zone_nav[zid]=self.second_zone_nav.get(zid,0)+1
            self._ledger(zid)['navigation_goals']+=1
            self._ledger(zid)['selected_regions'].append(c['residual_region_id'])
            self.second_catalog=None;self.second_context=None
            self.n.event('SECOND_PASS_NAV_SENT',task_id=t['task_id'],zone=zid,
                candidate_id=c['id'],navigation_goals=t['nav_sent'],safety_approved=True)

        def note_path_blockage(self,c,record,inp):
            if self._is_second(c):
                self.n.event('SECOND_PASS_PATH_OBSTRUCTION_RECORDED',zone=c['zone_id'],
                    task_id=c['second_task_id'],not_whole_zone_blockage=True)
                return  # confirmed record remains in original path monitor; no unsafe automatic same-goal retry
            return base.note_path_blockage(self,c,record,inp)

        def authorizes_region_cancel(self):
            n=self.n;r=self.region_cancel
            return bool(r and r['goal_id']==n.nav_uuid and n.cancel_reason==r['reason'])

        def after_second_result(self,inp):
            """Called ONLY after original exact result + fresh stop + NEW scan gates."""
            n=self.n;c=n.current
            if not self._is_second(c):return False
            tid=c['second_task_id'];t=self.tasks.tasks[tid];zid=c['zone_id']
            status={4:'SUCCEEDED',5:'CANCELED',6:'ABORTED'}.get(n.nav_terminal,'UNKNOWN')
            if n.nav_terminal==5:
                if self.authorizes_region_cancel():pass
                elif n.mission_guard and n.mission_guard.authorizes_cancel():
                    if not n.mission_guard.after_confirmed_stop(inp):return True
                elif n.path_blockage and n.path_blockage.authorizes_cancel():
                    if not n.path_blockage.after_confirmed_stop(inp):return True
                else:
                    n.request_stop('EXTERNAL_OR_SAFETY_CANCEL');return True
            elif n.nav_terminal==6:
                if not self.args.continue_on_nav_abort or not n.local_start_clear(inp):
                    n.finish('NAV_ABORTED_INSPECT_BEFORE_RETRY',2);return True
                n.aborted_goals+=1;n.book.nav_abort_counts[zid]+=1
            elif n.nav_terminal!=4:
                n.request_stop('UNEXPECTED_NAV_RESULT');return True
            before_count=len(c['residual_target_indices'])
            remaining=int((~inp['observed'].ravel()[c['residual_target_indices']]).sum())
            gain=max(0,before_count-remaining)*self.bundle[1].resolution**2
            reason='NAV_SUCCEEDED' if status=='SUCCEEDED' else str(n.cancel_reason or 'NAV_ABORTED')
            outcome=dict(candidate_id=c['id'],xy=[float(c['x']),float(c['y'])],
                stage='NAVIGATION',nav_status=status,reason=reason,target_gain_m2=float(gain),
                predicted_target_gain_m2=float(c['new_area_m2']),physical_blockage_proven=False)
            if c.get('residual_entry'):
                arrival=entry_membership(c,inp['pose'],self.bundle[1],self.cfg,oriented_footprint_clear)
                outcome['entry_arrival']=finite_json(arrival)
                t['entry_confirmed']=bool(t.get('entry_confirmed') or
                    (status=='SUCCEEDED' and arrival.get('entry_confirmed')))
                self.n.event('SECOND_PASS_ENTRY_ARRIVAL',task_id=tid,zone=zid,
                    nav_status=status,**finite_json(arrival))
                # A secondary goal can also be the first actual entry into its
                # parent Zone. Use the unchanged measured padded-footprint test.
                from zone_planner import is_inside
                if status=='SUCCEEDED' and is_inside(n.zones_by_id[zid],inp['pose'],self.bundle,self.cfg):
                    n.book.mark_entered(zid)
                    n.event('ENTRY_CONFIRMED',zone=zid,pose=list(inp['pose']),
                        goal_kind='ENTRY',second_pass=True,source='RESIDUAL_ENTRY')
            t['attempts'].append(outcome)
            if n.nav_terminal==4:
                n.succeeded_goals+=1
                record=n.book.record_visit(c,n.before_fraction,n.book.records[zid].measured,status,n.cancel_reason)
                n.event('COVERAGE_GAIN',**record)
            if status!='SUCCEEDED':
                self.failed_positions.append([float(c['x']),float(c['y'])])
                failure=(self.region_cancel or {}).get('evidence',{}).get('xy')
                if failure and math.dist(failure,inp['pose'][:2])>.6:
                    self.failed_routes.append(dict(xy=list(failure),radius_m=.3,reason=reason))
            elif not (c.get('direct_approach') or c.get('residual_entry')) and gain<self.second_settings.low_yield_area_m2:
                self._ledger(zid)['low_yield_positions'].append([float(c['x']),float(c['y'])])
            self.n.event('SECOND_PASS_VIEW_OUTCOME',zone=zid,task_id=tid,**outcome)
            self._ledger(zid)['outcomes'].append(outcome)
            self._inspect(zid,inp)
            if n.mission_guard:n.mission_guard.finish_terminal()
            n.nav_send=None;n.nav_handle=None;n.nav_result=None;n.nav_uuid=None
            n.nav_terminal=None;n.cancel_reason=None;n.cancel_future=None;n.post_result_ros=None
            n.current=None;n.options=[];n.current_path=[];n.current_path_length=None;n.batch_zone=None
            n.options_context=None;n.validated_plan_mode=None;n.phase='SELECT_ZONE'
            self.region_cancel=None;self.last_live_signature=None
            n.book.audit_requested.pop(zid,None)
            if status=='SUCCEEDED':t['status']='PENDING'
            elif 'BUDGET' in reason or 'WORK' in reason or 'RECOVERY_LIMIT' in reason:
                self.tasks.finish(tid,'UNVERIFIED',reason,inp)
            else:t['status']='PENDING'
            self.tasks.refresh(inp);self._drop_current_task()
            n.transition('SELECTING')
            return True

        def region_tick(self,inp):
            self._track_primary()
            monitor=getattr(self.n,'path_blockage',None)
            if monitor is not None:monitor.refresh_scan()  # sensing only, also while stationary
            super().region_tick(inp)
            now_wall=time.monotonic()
            if self.exploration_stage=='SECOND' and self.second_task:
                zid=self.tasks.tasks[self.second_task]['zone_id']
                if self.second_last_account is not None:
                    self.second_zone_work[zid]=self.second_zone_work.get(zid,0.)+min(1.,max(0.,now_wall-self.second_last_account))
            self.second_last_account=now_wall
            n=self.n;c=n.current
            if (not self._is_second(c) or n.nav_uuid is None or n.nav_terminal is not None
                or n.cancel_reason is not None or n.stop_requested
                or now_wall-self.second_monitor_wall<self.second_settings.monitor_period_s):return
            self.second_monitor_wall=now_wall
            # Internal ENTRY delegates live trajectories to the unchanged Nav2 +
            # PathBlockage + freshness/mission guards, as ordinary Zone ENTRY does.
            if c.get('residual_entry'):return
            path=getattr(monitor,'path',None)
            if path is None:return  # baseline runtime monitors stale/missing inputs independently
            if not -.1<=n.ros_s()-path['stamp']<=3. or now_wall-path['wall']>5.:return
            points=remaining_path(path['points'],inp['pose'],self.second_settings.live_horizon_m)
            if len(points)<2:return
            ok,reason,e=path_guard(points,inp['pose'],c,self.bundle,inp,self.cfg,
                self.second_settings,oriented_footprint_clear,(),check_endpoint=False,budget_s=.35)
            if ok:return
            # A bounded check timeout is not geometry evidence; stop conservatively
            # and retain UNVERIFIED rather than retrying into an unchecked route.
            cancel_reason='SECOND_PASS_SAFETY_'+reason
            self.region_cancel=dict(goal_id=n.nav_uuid,reason=cancel_reason,evidence=finite_json(e))
            n.event('SECOND_PASS_RUNTIME_SAFETY_CANCEL',goal_id=n.nav_uuid,
                    task_id=c['second_task_id'],reason=cancel_reason,evidence=finite_json(e))
            n.request_cancel(cancel_reason)

        def publish_region_report(self):
            # Original sidecar source retained; final layers below are separate.
            super().publish_region_report()
            if self.tasks.initialized:
                folder=self.output/'region_review';folder.mkdir(exist_ok=True)
                summary=self.region_summary()
                atomic_json(folder/'two_phase_latest.json',summary)
                atomic_json(folder/'final_results_v16.json',summary['final_results'])

        def region_summary(self):
            data=super().region_summary()
            if not hasattr(self,'tasks'):return data
            g=self.bundle[1]
            tasks=self.tasks.public(self.region_last_inp)
            inaccessible=[dict(task_id=t['task_id'],zone_id=t['zone_id'],reason=t['reason'],
                cell_runs_ros=t['cell_runs_ros'],area_m2=t['area_m2'],
                scope=t['completion_scope'],display_label='로봇 접근 불가 지역',
                access_scope='이번 로봇·설정·제한된 접근 시도 기준; 다른 경로 가능성은 배제하지 않음',
                evidence_is_historical=t.get('evidence_is_historical',False),
                physical_blockage_proven=False) for t in tasks if t['status']=='ROBOT_EXPLORATION_UNAVAILABLE']
            cuts=[]
            for zid,report in data.get('zones',{}).items():
                for scope in ('global','local'):
                    for region in report.get('access',{}).get(scope+'_regions',[]):
                        if not region['classification'].startswith('SENSOR_SUPPORTED'):continue
                        cuts.append(dict(region,zone_id=zid,scope=scope,
                            geometry_semantics='robot_center_subset_NOT_full_inaccessible_floor',
                            evidence_is_historical=report.get('evidence_is_historical',True)))
            data.update(version=VERSION,stage=self.exploration_stage,
                two_phase_settings=asdict(self.second_settings),primary_reservations=self.primary_reservations,
                schedule='ZONE_LOCAL_IMMEDIATE',global_second_pass=False,direct_approach=False,residual_entry=True,
                local_second_zone=self.local_second_zone,locally_reviewed_zones=sorted(self.local_reviewed),
                local_reviews=self.local_reviews,
                unreviewed_zones=sorted(set(self.n.zones_by_id)-self.local_reviewed),
                inventory_complete=set(self.n.zones_by_id)<=self.tasks.inventoried_zones,inventory_error=self.last_inventory_error,
                tasks=tasks,counts=self._counts(),small_cells_by_zone=self.tasks.small_by_zone,
                second_pass_safety_cancels=sum(a.get('nav_status')=='CANCELED' and a.get('reason','').startswith('SECOND_PASS_SAFETY_') for t in tasks for a in t['attempts']),
                geometry=dict(frame_id='map',width=g.width,height=g.height,resolution=g.resolution,
                              origin=list(g.origin),row_order='ros_bottom_up'),
                map_layers=dict(blocked_passages=[],blocked_regions=cuts,victims=[],
                    robot_exploration_unavailable=inaccessible),
                map_layer_status=dict(blocked_passages='NO_FABRICATED_LINE_FROM_PLANNER_FAILURE; exact barrier line extraction not implemented',
                    blocked_regions='SENSOR_SUPPORTED_GRID_CONDITIONED_CENTER_SUBSETS_ONLY',
                    victims='NOT_IMPLEMENTED_IN_THIS_PATCH',
                    robot_exploration_unavailable='OPERATIONAL_SAMPLED_POLICY_RESULT'),
                final_map_renderer_implemented=False,
                nav2_configuration_modified=False,coverage_denominator_changed=False)
            # DN012_GLOBAL_BLOCKAGE_V1_3: display remembered POINTS, never fake cut lines.
            if self.blockage_memory is not None:
                memory=self.blockage_memory.summary()
                data['blockage_memory']=memory
                data['map_layers']['blockage_memory_points']=memory['points']
                data['map_layer_status']['blockage_memory_points']=(
                    'PATH_OBSTRUCTION_WITNESSES_ONLY; coordinates not passage-cut lines; '+
                    'live costmap revalidation required before holding a path')
            monitor=getattr(self.n,'path_blockage',None)
            reroute=getattr(monitor,'reroute',None)
            records=self.n.book.summary() if hasattr(self.n.book,'summary') else {}
            data['final_results']=project_results(data,records,
                path_records=getattr(monitor,'records',()),
                reroute_summary=reroute.summary() if reroute is not None else None,
                owned_goal_ids=getattr(self.n,'owned_ids',()))
            return finite_json(data)

        def close(self):
            if self.second_future:self.second_future.cancel()
            return base.close(self)

    TwoPhaseConnectivityRuntime.__name__='TwoPhaseConnectivityRuntime'
    return TwoPhaseConnectivityRuntime
