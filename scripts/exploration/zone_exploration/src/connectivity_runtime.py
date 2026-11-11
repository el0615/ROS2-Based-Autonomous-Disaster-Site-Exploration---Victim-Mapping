"""Opt-in adapter to the existing async Nav2 state machine.

No independent motion publisher. Connectivity probes use existing preflight
ActionClient; every real observation still passes the original safety gates.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
import math
import time
from collections import deque
import numpy as np
from connectivity_audit import (AuditSettings, build_catalog, useful_untried,
                                assess_catalog, sensor_signature, EntryAccessSettings, entry_probe_sites)
from exploration_io import atomic_json


STALE_REASONS={'POSE_OR_SESSION_CHANGED','COSTMAP_CHANGED_REPLAN','INPUTS_NOT_FRESH',
               'GOAL_NO_LONGER_NEEDED'}


class ConnectivityRuntime:
    def __init__(self,node,bundle,cfg,args,output):
        self.n=node; self.bundle=bundle; self.cfg=cfg; self.args=args; self.output=output
        self.settings=AuditSettings(spacing_m=args.audit_spacing_m,headings=args.audit_headings,
            max_positions=args.audit_max_positions,max_probes=args.audit_max_probes,
            build_budget_s=args.audit_build_budget_s,zone_budget_s=args.audit_zone_budget_s,
            max_restarts=args.audit_max_restarts)
        self.settings.validate()
        self.zid=None; self.future=None; self.catalog=None; self.probes={}
        self.started=0.; self.restarts=0; self.probe_count=0
        # Preflight attempts of audit observation views (not Nav2 navigation Goals).
        # A limit defers the Zone as UNVERIFIED; it can never certify completion.
        self.view_preflight_attempts=0
        # A failed planning request is NOT a blocked-space certificate.
        # Group only negative Nav2 preflights by the exact sampled position
        # during ONE unchanged audit catalog.  Two distinct headings maximum.
        self.position_view_failures={}
        # Across successful, distinct OBSERVE navigation Goals only; no path probes
        # or ENTRY moves count. This is an operational stop, NEVER a certificate.
        self.low_gain_history={}
        self.total_probes=0; self.sequence=0; self.pending_context=None
        self.entry_settings=EntryAccessSettings(
            max_probes=args.entry_access_max_probes,
            per_component=args.entry_access_per_component,
            min_separation_m=args.entry_access_min_separation_m,
            failed_radius_m=args.entry_access_failed_radius_m)
        self.entry_settings.validate()
        self.blockage_replans={};self.blockage_results={};self.blockage_context=None
        self.blockage_verified=None
        self.failed_entries={}  # real confirmed-stop NavigateToPose ABORTED endpoints
        self.entry_mode=False;self.entry_sites=None;self.entry_index=0
        self.entry_verified_goal=None;self.entry_attempts=0
        self.n.event('CONNECTIVITY_POLICY',threshold=cfg.threshold,zone_lock=True,
            nav2_global_detour=True,normal_limit_requires_connectivity=True,
            lattice_spacing_m=self.settings.spacing_m,heading_samples=self.settings.headings)

    def clear_round(self):
        self.catalog=None; self.probes={}; self.pending_context=None
        self.position_view_failures={}
        self.entry_sites=None;self.entry_index=0;self.entry_verified_goal=None
        self.blockage_verified=None

    def note_audit_view_preflight_failure(self,c,reason):
        """Bound same-position yaw retries on a fresh, *stationary* catalog.

        Neither planner failure nor a position hold means physically blocked.
        A new catalog after fresh motion or changed context lifts every hold.
        Do not group start-pose, costmap or communication stale errors.
        """
        if (not c or not c.get('audit_observation') or c.get('connectivity_probe')
                or self.zid!=c.get('zone_id') or self.catalog is None
                or reason not in ('NAVFN_ABORTED','PATH_END_MISMATCH')):
            return
        pid=c.get('audit_position_id')
        if not isinstance(pid,str) or not pid:
            return
        heading=round(float(c['yaw']),6)
        entries=self.position_view_failures.setdefault(pid,{})
        if heading in entries:
            return
        entries[heading]=reason
        if len(entries)==2:
            self.n.event('AUDIT_POSITION_PREFLIGHT_HELD',zone=self.zid,
                position_id=pid,position=[float(c['x']),float(c['y'])],
                attempted_headings=sorted(entries),reasons=list(entries.values()),
                status='ACCESS_UNVERIFIED_NOT_BLOCKED',
                catalog_scope='CURRENT_COSTMAP_POSE_COVERAGE_SNAPSHOT')
            self.n.warn('audit_position_hold',
                f'[AUDIT_POSITION_HELD] {self.zid} {pid}; two heading preflights failed; '
                'not blocked proof; try different position',0)

    def audit_observation_goal_sent(self, c):
        """Invalidate audit evidence ONLY once an observation NAV Goal was sent.

        Nav2 preflight rejection has no robot motion, so it must not throw away
        a still-fresh audit lattice. The next post-motion audit must rebuild it.
        This does not change the original path, footprint or stop safety gates.
        """
        if self.zid == c.get('zone_id') and c.get('audit_observation'):
            self.n.event('ZONE_AUDIT_NAV_INVALIDATES_CATALOG', zone=self.zid,
                         reason='OBSERVATION_NAV_GOAL_SENT_REBUILD_AFTER_MOTION',
                         audit_view_preflights=self.view_preflight_attempts)
            self.clear_round()

    def reset_low_gain_history(self,zone_id):
        self.low_gain_history.pop(zone_id,None)

    def consider_low_gain_plateau(self,record,coverage_session,confirmed_inside):
        """Defer a low-yield Zone after 3 distinct *successful* observation Goals.

        Evidence is fresh post-Goal Coverage and confirmed stopped robot. A plateau
        only ends further motion *in this Zone*, not the full mission and never
        proves the Zone is observed, blocked, disconnected, or victim-free.
        The net change across the three Goals includes intervening observations.
        """
        zid=record['zone_id']
        if (not confirmed_inside or record.get('goal_kind')!='OBSERVE'
                or record.get('nav_status')!='SUCCEEDED'):
            self.reset_low_gain_history(zid)
            return False
        rec=self.n.book.records[zid]
        if (self.n.book.is_complete(zid) or rec.measured>=self.cfg.threshold
                or rec.state=='PARTIAL_DEFERRED'):
            self.reset_low_gain_history(zid)
            return False
        window=int(getattr(self.args,'low_gain_stop_window',3))
        if window==0:return False
        max_gain=float(getattr(self.args,'low_gain_stop_total_pp',1.0))
        history=self.low_gain_history.get(zid)
        if history is None or any(x['coverage_session']!=coverage_session for x in history):
            history=deque(maxlen=window)
            self.low_gain_history[zid]=history
        goal_id=str(record.get('id',''))
        before=float(record['before_percent']);after=float(record['after_percent'])
        if not (goal_id and math.isfinite(before) and math.isfinite(after)
                and -0.001<=before<=100.001 and -0.001<=after<=100.001):
            self.reset_low_gain_history(zid)
            return False
        if (after<before-1e-5 or
                (history and (before<history[-1]['after']-1e-5 or
                              any(x['goal_id']==goal_id for x in history)))):
            self.reset_low_gain_history(zid)
            return False
        history.append(dict(goal_id=goal_id,coverage_session=coverage_session,
                            before=before,after=after))
        if len(history)<window:
            self.n.event('ZONE_LOW_GAIN_PROGRESS',zone=zid,observations=len(history),
                         required_observations=window,total_gain_pp=round(after-history[0]['before'],6),
                         threshold_total_pp=max_gain,decision='CONTINUE')
            return False
        total=after-history[0]['before']
        if total>=max_gain-1e-8:return False
        evidence=dict(schema='low_gain_plateau_v1',normal_termination=False,
            connectivity_verified=False,all_potential_information_accounted=False,
            possible_unexamined_routes=True,scope='operational stop of further Zone exploration, not an accessibility/completion certificate',
            successful_observation_goals=window,
            distinct_goal_ids=[x['goal_id'] for x in history],
            coverage_session=coverage_session,first_before_percent=history[0]['before'],
            last_after_percent=after,total_gain_pp=round(total,6),
            cutoff_total_gain_pp=max_gain,
            guidance='Do not mark as observed/blocked; review after mission with current-map connectivity evidence')
        self.n.event('ZONE_LOW_GAIN_PLATEAU',zone=zid,**evidence)
        self.n.warn('low_gain_plateau',f'[ZONE_LOW_GAIN_PLATEAU] {zid}; '
            f'{window} successful observations total +{total:.3f}pp < {max_gain:.3f}pp; '
            'defer UNVERIFIED, do not send more Zone Goals',0)
        # finish_partial() owns the evidence archive and clears Zone lock.
        # Do not reassign to a different Zone during an active NavigateToPose.
        self.zid=zid
        self.finish_partial('LOW_GAIN_PLATEAU_UNVERIFIED',evidence)
        self.reset_low_gain_history(zid)
        return True

    def note_path_blockage(self,c,record,inp):
        # Called only after exact owned CANCELED + fresh stop + post-result coverage.
        zid=c['zone_id']
        self.blockage_replans[zid]=dict(c)
        self.blockage_results[zid]=[]
        self.n.book.activate(zid)
        self.n.book.audit_requested[zid]=('ENTRY_PATH_BLOCKAGE_ACCESS_RECHECK'
            if c['goal_kind']=='ENTRY' else 'PATH_BLOCKAGE_RECHECK')
        if c['goal_kind']=='ENTRY':
            self.failed_entries.setdefault(zid,[]).append(dict(x=float(c['x']),y=float(c['y']),
                yaw=float(c['yaw']),id=c.get('id'),kind='SENSOR_SUPPORTED_PATH_CANCELED'))
        self.zid=None;self.clear_round()

    def blockage_partial_evidence(self,inp,evidence):
        # No topology claim from one planner failure or a limit. Unknown cost cells
        # are allowed by the optimistic cut test to avoid inventing a blocked route.
        monitor=getattr(self.n,'path_blockage',None)
        record=monitor.latest_by_zone.get(self.zid) if monitor else None
        if not record:return evidence
        e=dict(evidence,path_blockage=record,normal_termination=False)
        prior_failures=self.blockage_results.get(self.zid,[])
        e['same_goal_replan_results']=prior_failures
        if inp is None:return e
        witness=record['evidence'];i,j=witness['cell'];geom=self.bundle[1]
        monitor.refresh_scan()
        h=monitor.hits;now=self.n.ros_s();wall=time.monotonic()
        live=(geom.contains(i,j) and inp['cost']['values'][j,i]==100 and h is not None
              and -.1<=now-h['stamp']<=1. and wall-h['wall']<=3.
              and len(h['points'])>0 and
              float(np.min(np.linalg.norm(
                  h['points']-np.asarray(witness['xy']),axis=1)))<=monitor.settings.hit_tolerance_m)
        e['blockage_still_sensor_supported']=bool(live)
        if not live or not any(r.get('reason')=='NAVFN_ABORTED' for r in prior_failures):return e
        from path_blockage import current_map_cut
        cut=current_map_cut(geom,self.bundle[3],inp['cost']['values'],inp['pose'],
            self.bundle[2],self.n.zones_by_id[self.zid]['mask_id'],
            fixed_prior_topology=bool(getattr(self.args,'prior_topology_fixed',False)))
        e['current_map_cut']=cut
        e['current_map_unreachable']=(cut.get('zone_cut') is True)
        return e

    def note_entry_abort(self,c):
        """Only called after owned NAV ABORTED, fresh odometry stop and coverage.

        The target coordinate is an unsuccessful entry goal; no blocked geometry
        or physical unreachable statement is inferred from Nav2 ABORTED.
        """
        if c and c.get('goal_kind')=='ENTRY':
            zid=c['zone_id']
            self.failed_entries.setdefault(zid,[]).append(
                dict(x=float(c['x']),y=float(c['y']),yaw=float(c['yaw']),
                     id=c.get('id'),kind='NAV_ABORTED_ENTRY'))
            self.n.event('ENTRY_NAV_ABORT_RECORDED',zone=zid,
                goal=[float(c['x']),float(c['y']),float(c['yaw'])],
                failed_entry_goals=len(self.failed_entries[zid]),
                not_proof_of_blocked_zone=True)

    def entry_reached(self,zid):
        """Only after fresh stopped pose+footprint confirm entry into the Zone."""
        if self.zid==zid and self.entry_mode:
            self.n.event('ENTRY_ACCESS_RECOVERED',zone=zid)
            self.n.book.audit_requested.pop(zid,None)
            self.zid=None;self.entry_mode=False;self.clear_round()

    def start_zone(self,zid):
        self.zid=zid; self.started=time.monotonic(); self.restarts=0; self.probe_count=0
        self.view_preflight_attempts=0
        reason=self.n.book.audit_requested[zid]
        self.entry_mode=(not self.n.book.visit_state[zid].entered and
                         (reason.startswith('ENTRY_') or reason.startswith('NAV_ABORTED_')))
        self.entry_attempts=0
        self.clear_round()
        self.n.book.activate(zid)
        self.n.options=[]; self.n.batch_zone=None; self.n.options_context=None
        self.n.event('ZONE_AUDIT_STARTED',zone=zid,reason=self.n.book.audit_requested[zid],
                     entry_access_mode=self.entry_mode)
        self.n.info('audit_start',f'[ZONE_AUDIT_STARTED] {zid}; keep zone lock; Nav2 global paths only',0)

    def context_matches(self,inp):
        c=self.catalog or self.pending_context
        return bool(c and c['cost_key']==inp['cost']['key'] and c['epoch']==inp['epoch']
            and c['sensor_signature']==sensor_signature(inp['sensor'])
            and math.dist(c['start_pose'][:2],inp['pose'][:2])<=.15
            and abs(math.atan2(math.sin(c['start_pose'][2]-inp['pose'][2]),
                              math.cos(c['start_pose'][2]-inp['pose'][2])))<=.15)

    def invalidate(self,reason):
        self.restarts+=1
        self.n.event('CONNECTIVITY_EVIDENCE_INVALIDATED',zone=self.zid,reason=reason,restarts=self.restarts)
        self.clear_round()
        if self.restarts>self.settings.max_restarts:
            self.finish_partial('AUDIT_CONTEXT_UNSTABLE_UNVERIFIED')

    def finish_partial(self,reason,evidence=None):
        if self.zid is None:return
        zid=self.zid
        e=evidence or {'scope':'incomplete assessment','connectivity_verified':False}
        if reason=='ENTRY_ACCESS_UNVERIFIED_ALTERNATES_EXHAUSTED':
            err,inp=self.n.inputs()
            e=self.blockage_partial_evidence(inp if err is None else None,e)
            if e.get('current_map_unreachable') is True:
                reason='CURRENT_MAP_UNREACHABLE_FIXED_PRIOR_TOPOLOGY'
                self.n.warn('map_unreachable',f'[CURRENT_MAP_UNREACHABLE] {zid}; occupied grid cut + failed replan; no new-wall-opening assumption; NOT complete',0)
            elif e.get('path_blockage'):
                reason='PATH_BLOCKED_NO_CONFIRMED_DETOUR'
                self.n.warn('path_no_detour',f'[PATH_BLOCKED_NO_CONFIRMED_DETOUR] {zid}; keep unresolved',0)
        self.save_evidence(zid,reason,e)
        self.n.book.finish_partial(zid,reason,evidence=e)
        self.n.event('ZONE_ASSESSMENT',zone=zid,classification=reason,normal_termination=False,
                     measured_percent=100*self.n.book.records[zid].measured)
        self.n.warn('audit_partial',f'[ZONE_UNVERIFIED] {zid}: {reason}; NOT normal completion',0)
        self.zid=None;self.entry_mode=False;self.clear_round()

    def save_evidence(self,zid,verdict,evidence):
        self.sequence+=1
        folder=self.output/'connectivity_evidence';folder.mkdir(exist_ok=True)
        record=dict(zone=zid,verdict=verdict,evidence=evidence,
                    probes=self.probes,ros_time_s=self.n.ros_s(),
                    measured_percent=self.n.book.records[zid].measured*100)
        atomic_json(folder/f'assessment_{self.sequence:04d}.json',record)

    def probe_success(self,c,inp,points,length):
        """Called only AFTER baseline path+goal-footprint preflight succeeded."""
        if c.get('path_blockage_probe'):
            context=self.blockage_context
            if (not context or self.zid!=c['zone_id'] or context['cost_key']!=inp['cost']['key']
                    or context['epoch']!=inp['epoch']):
                self.finish_partial('PATH_BLOCKAGE_REPLAN_CONTEXT_CHANGED');return
            monitor=getattr(self.n,'path_blockage',None)
            h=monitor.hits if monitor else None
            from path_blockage import path_obstacle_witness
            if monitor and h and -.1<=self.n.ros_s()-h['stamp']<=1. and time.monotonic()-h['wall']<=3.:
                w=path_obstacle_witness(points,inp['pose'],self.bundle[1],self.bundle[3],
                    inp['cost']['values'],h['points'],self.cfg,monitor.settings)
                if w is not None:
                    self.blockage_results.setdefault(self.zid,[]).append(dict(reason='REPLANNED_PATH_STILL_SENSOR_BLOCKED'))
                    self.n.event('PATH_BLOCKAGE_REPLAN_STILL_BLOCKED',zone=self.zid,witness=w)
                    return
            self.blockage_verified=dict(c)
            for key in ('path_blockage_probe','connectivity_probe'):
                self.blockage_verified.pop(key,None)
            self.blockage_verified['entry_access_recovery']=True
            self.n.event('PATH_BLOCKAGE_REPLAN_OK',zone=self.zid,length_m=float(length),fresh_preflight_required=True)
            self.n.info('path_replan_ok',f'[PATH_BLOCKAGE_REPLAN_OK] {self.zid}; fresh normal preflight required before motion',0)
            return
        if self.zid!=c['zone_id'] or not self.context_matches(inp):
            self.invalidate('PROBE_CONTEXT_CHANGED');return
        _,geom,labels,_,_=self.bundle
        mask_id=self.n.zones_by_id[self.zid]['mask_id']
        outside=False
        for x,y in points:
            i,j=map(math.floor,geom.world_to_grid_float(x,y))
            if not geom.contains(i,j) or int(labels[j,i])!=mask_id:
                outside=True;break
        pid=c['audit_position_id']
        if c.get('entry_access_probe'):
            self.entry_verified_goal=dict(c)
            self.entry_verified_goal.pop('connectivity_probe',None)
            self.entry_verified_goal.pop('entry_access_probe',None)
            self.entry_verified_goal['entry_access_recovery']=True
            self.n.event('ENTRY_ALTERNATE_PATH_VERIFIED',zone=self.zid,
                         position_id=pid,path_length_m=float(length),
                         path_left_target_zone=outside)
        self.probes[pid]=dict(success=True,path_length_m=float(length),
            path_left_target_zone=outside,path_sha256=hashlib.sha256(json.dumps(points).encode()).hexdigest(),
            goal=[c['x'],c['y'],c['yaw']],cost_key=inp['cost']['key'],ros_time_s=self.n.ros_s())
        self.n.event('CONNECTIVITY_PROBE_OK',zone=self.zid,position_id=pid,
                     path_left_target_zone=outside,path_length_m=float(length))
        self.n.info('audit_probe',f'[CONNECTIVITY_PROBE_OK] {self.zid} {len(self.probes)}/{len(self.catalog["positions"])}',2.)

    def probe_failure(self,c,reason):
        if c.get('path_blockage_probe'):
            self.blockage_results.setdefault(c['zone_id'],[]).append(dict(reason=reason))
            self.n.event('PATH_BLOCKAGE_REPLAN_FAILED',zone=c['zone_id'],reason=reason)
            self.n.warn('path_replan_fail',f'[PATH_BLOCKAGE_REPLAN_FAILED] {c["zone_id"]}: {reason}; checking alternatives',0)
            if reason in STALE_REASONS:self.finish_partial('PATH_BLOCKAGE_REPLAN_CONTEXT_UNVERIFIED')
            return
        if reason in STALE_REASONS:
            self.invalidate(reason);return
        if self.zid!=c['zone_id']:return
        self.probes[c['audit_position_id']]=dict(success=False,reason=reason,
            cost_key=self.catalog['cost_key'] if self.catalog else None,
            ros_time_s=self.n.ros_s())
        self.n.event('CONNECTIVITY_PROBE_FAILED',zone=self.zid,
                     position_id=c['audit_position_id'],reason=reason)

    def step(self,inp):
        """True: this adapter owns selection. False: retain original selection."""
        n=self.n
        if self.future is not None and not self.future.done():
            n.transition('AUDITING_ZONE');return True
        if self.future is not None:
            f=self.future;self.future=None
            try:built=f.result()
            except TimeoutError as exc:
                self.finish_partial(str(exc));return True
            except Exception:
                raise  # original runtime error + cancellation path remains authoritative
            if self.zid is None or n.book.is_complete(self.zid):
                self.clear_round();self.zid=None;return False
            if not self.context_matches(inp):
                self.invalidate('AUDIT_BUILD_INPUT_CHANGED');return True
            self.catalog=built
            n.event('ZONE_AUDIT_CATALOG',zone=self.zid,positions=len(built['positions']),
                    views=len(built['views']),components=built['components'],
                    unrepresented_components=built['unrepresented_components'],
                    lattice_spacing_m=built['lattice_spacing_m'],elapsed_s=built['elapsed_s'])
        if self.zid and (n.book.is_complete(self.zid) or self.zid not in n.book.audit_requested):
            self.zid=None;self.clear_round()
        if self.zid is None:
            active=n.book.active_zone
            if active and active not in n.book.audit_requested:
                return False  # never steal the lock to audit an unrelated zone
            pending=active or next(iter(n.book.audit_requested),None)
            if pending is None:return False
            self.start_zone(pending)
        if time.monotonic()-self.started>self.settings.zone_budget_s:
            self.finish_partial('AUDIT_TIME_BUDGET_UNVERIFIED');return True
        if self.blockage_verified is not None:
            c=self.blockage_verified;self.blockage_verified=None
            if not n.local_start_clear(inp):
                n.request_stop('LOCAL_START_NOT_CLEAR_OR_STALE');return True
            n.plan_request(c,inp);return True
        if self.zid in self.blockage_replans:
            if not n.local_start_clear(inp):
                n.request_stop('LOCAL_START_NOT_CLEAR_OR_STALE');return True
            c=self.blockage_replans.pop(self.zid)
            c=dict(c,connectivity_probe=True,path_blockage_probe=True,audit_position_id='same_goal_replan')
            self.blockage_context=dict(cost_key=inp['cost']['key'],epoch=inp['epoch'])
            self.probe_count+=1;self.total_probes+=1
            n.info('blocked_replan',f'[PATH_BLOCKAGE_REPLAN_REQUESTED] {self.zid}; same goal, entire global map',0)
            n.plan_request(c,inp);return True
        if self.catalog is not None and not self.context_matches(inp):
            self.invalidate('AUDIT_POSE_MAP_OR_SENSOR_CHANGED');return True
        if self.catalog is None:
            snap=n.snapshot(inp)
            self.pending_context=dict(cost_key=inp['cost']['key'],epoch=inp['epoch'],
                sensor_signature=sensor_signature(inp['sensor']),start_pose=list(inp['pose']))
            self.future=n.pool.submit(build_catalog,self.bundle,self.cfg,self.settings,snap,
                n.zones_by_id[self.zid],n.zone_planner.visibility,n.zone_planner.cancel)
            n.transition('AUDITING_ZONE');return True
        if self.entry_mode:
            if not n.local_start_clear(inp):
                n.request_stop('LOCAL_START_NOT_CLEAR_OR_STALE');return True
            if self.entry_verified_goal is not None:
                candidate=self.entry_verified_goal
                self.entry_verified_goal=None
                # Re-issue the ordinary preflight using CURRENT inputs. A probe
                # result can never authorize navigation by itself.
                n.event('ENTRY_ALTERNATE_GOAL_SELECTED',zone=self.zid,
                        xy=[candidate['x'],candidate['y']],
                        requires_fresh_nav2_preflight=True)
                n.plan_request(candidate,inp);return True
            if self.entry_sites is None:
                self.entry_sites=entry_probe_sites(
                    self.catalog,self.failed_entries.get(self.zid,[]),self.entry_settings)
                n.event('ENTRY_ACCESS_CANDIDATES',zone=self.zid,
                        candidates=len(self.entry_sites),
                        components=len({p['component'] for p in self.entry_sites}),
                        failed_navigation_entries=len(self.failed_entries.get(self.zid,[])),
                        max_probes=self.entry_settings.max_probes)
            if self.entry_index < len(self.entry_sites):
                if self.probe_count>=self.settings.max_probes:
                    self.finish_partial('ENTRY_ACCESS_PROBE_BUDGET_UNVERIFIED');return True
                pos=self.entry_sites[self.entry_index];self.entry_index+=1
                c=dict(pos['probe'],goal_kind='ENTRY',connectivity_probe=True,
                    entry_access_probe=True,audit_position_id=pos['position_id'])
                self.probe_count+=1;self.total_probes+=1;self.entry_attempts+=1
                n.event('ENTRY_ACCESS_PROBE',zone=self.zid,
                        component=pos['component'],position_id=pos['position_id'],
                        attempt=self.entry_attempts,max_attempts=len(self.entry_sites))
                n.plan_request(c,inp);return True
            evidence=dict(schema='entry_access_assessment_v2',
                entry_mode=True,route_probes_attempted=self.entry_attempts,
                route_probes_succeeded=sum(p.get('success') is True for p in self.probes.values()),
                route_probes_failed=sum(p.get('success') is False for p in self.probes.values()),
                failed_nav_entries=len(self.failed_entries.get(self.zid,[])),
                components_sampled=len({p['component'] for p in self.entry_sites}),
                possible_unexamined_routes=True,
                connectivity_verified=False,
                scope='bounded distinct interior target probes, Nav2 global path; NOT unreachable proof')
            n.event('ENTRY_ACCESS_EXHAUSTED',zone=self.zid,**evidence)
            self.finish_partial('ENTRY_ACCESS_UNVERIFIED_ALTERNATES_EXHAUSTED',evidence)
            n.transition('SELECTING');return True
        # DN012_REGION_RESCUE_V1: ENTRY + async context machinery above is retained.
        if hasattr(self, 'residual_step'):
            handled = self.residual_step(inp)
            if handled is not None:
                return handled
        history=list(n.book.visits)+list(n.book.rejected)
        available,_=useful_untried(self.catalog,inp['observed'],history,self.bundle[1],self.cfg)
        # Route feasibility belongs to a sampled XY; yaw controls sensing and
        # final-pose safety. Avoid requesting 8 doomed headings at the same XY.
        # Never treat skipped yaws as positive connectivity/coverage evidence.
        held_views=[c for c in available if len(self.position_view_failures.get(c.get('audit_position_id'),{}))>=2]
        available=[c for c in available if len(self.position_view_failures.get(c.get('audit_position_id'),{}))<2]
        if not available and held_views:
            held={c['audit_position_id'] for c in held_views}
            self.finish_partial('AUDIT_POSITION_ACCESS_UNVERIFIED',dict(
                scope='same-position planner preflight retry limit; other possible routes not excluded',
                connectivity_verified=False,normal_termination=False,
                held_positions=len(held),held_candidate_views=len(held_views),
                attempted_view_preflights=self.view_preflight_attempts,
                max_view_preflights=self.args.audit_max_view_preflights,
                planner_failures_by_position={pid:dict(entries)
                    for pid,entries in self.position_view_failures.items() if pid in held},
                possible_unexamined_routes=True))
            n.transition('SELECTING');return True
        if available:
            # Existing global max_plan_attempts is not a per-Zone bound. Avoid
            # exhausting the entire hospital mission on blocked R03 viewpoints.
            # Hitting this cap means incomplete evidence, NEVER a normal finish.
            limit=self.args.audit_max_view_preflights
            if self.view_preflight_attempts>=limit:
                self.finish_partial('AUDIT_VIEW_PREFLIGHT_BUDGET_UNVERIFIED',dict(
                    scope='bounded audit observation Nav2 path preflights; other safe views may remain',
                    connectivity_verified=False,normal_termination=False,
                    view_preflights=self.view_preflight_attempts,
                    max_view_preflights=limit,remaining_candidate_views=len(available)))
                n.transition('SELECTING');return True
            c=available[0]
            self.view_preflight_attempts+=1
            n.event('ZONE_AUDIT_NEW_VIEW',zone=self.zid,candidate_id=c['id'],
                    new_cells=c['new_cells'],view_preflight=self.view_preflight_attempts,
                    max_view_preflights=limit)
            # Preflight does NOT move the robot. Preserve the catalog and any
            # route-probe evidence across rejected paths at the same snapshot.
            # audit_observation_goal_sent() rebuilds only after a real NAV send.
            n.plan_request(c,inp);return True
        if not n.local_start_clear(inp):
            n.request_stop('LOCAL_START_NOT_CLEAR_OR_STALE');return True
        for position in self.catalog['positions']:
            pid=position['position_id']
            if pid in self.probes:continue
            if self.probe_count>=self.settings.max_probes:
                self.finish_partial('AUDIT_PROBE_BUDGET_UNVERIFIED');return True
            c=dict(position['probe'],connectivity_probe=True)
            self.probe_count+=1;self.total_probes+=1
            n.plan_request(c,inp);return True
        verdict,e=assess_catalog(self.catalog,self.probes,inp,self.bundle,n.zones_by_id[self.zid])
        if verdict=='NORMAL_LIMITED':
            zid=self.zid
            self.save_evidence(zid,verdict,e)
            n.book.finish_limited(zid,e)
            n.event('ZONE_OBSERVATION_LIMITED',zone=zid,measured_percent=n.book.records[zid].measured*100,
                    connectivity_verified=True,coverage_target_met=False)
            n.info('audit_limited',f'[ZONE_OBSERVATION_LIMITED] {zid}; connectivity verified on current-map lattice; raw Coverage unchanged',0)
            self.zid=None;self.clear_round()
        else:
            self.finish_partial(verdict+'_PARTIAL',evidence=e)
        n.transition('SELECTING');return True

    def close(self):
        if self.future:self.future.cancel()


# DN012_REGION_RESCUE_V1: disabled unless explicitly selected in terminal 6.
import os as _dn012_region_os
if _dn012_region_os.environ.get('DN012_REGION_RESCUE') == '1':
    from region_runtime import make_runtime as _dn012_make_runtime
    ConnectivityRuntime = _dn012_make_runtime(ConnectivityRuntime)
