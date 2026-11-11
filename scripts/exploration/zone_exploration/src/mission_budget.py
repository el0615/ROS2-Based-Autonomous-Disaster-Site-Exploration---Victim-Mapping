"""Operational budgets and their action-owner adapter. No reachability/completion claims.

Simulation time for navigation/Zone work, monotonic wall time for transport
freshness in the adapter. Budget exhaustion is UNVERIFIED, never completion.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
import math
import time

STAGNATION = 'NAVIGATION_STAGNATION_UNVERIFIED'
RECOVERY_LIMIT = 'NAV_RECOVERY_BUDGET_UNVERIFIED'
ZONE_LIMIT = 'ZONE_WORK_BUDGET_UNVERIFIED'
CANCEL_REASONS = frozenset((STAGNATION, RECOVERY_LIMIT, ZONE_LIMIT))

@dataclass(frozen=True)
class GuardSettings:
    stall_window_s: float = 20.0
    stall_extent_m: float = 0.20
    stall_recoveries: int = 5
    # Independent operational ceiling: an oscillating robot can move >0.20m.
    # This does NOT say the route is blocked. Zero disables this ceiling only.
    goal_recovery_limit: int = 20
    zone_cancel_limit: int = 2
    work_base_s: float = 180.0
    work_max_s: float = 300.0
    extension_window_s: float = 60.0
    extension_gain_pp: float = 1.0

    def validate(self):
        fs = (self.stall_window_s, self.stall_extent_m, self.work_base_s,
              self.work_max_s, self.extension_window_s, self.extension_gain_pp)
        if any(isinstance(v, bool) or not math.isfinite(v) for v in fs):
            raise ValueError('NONFINITE_GUARD_SETTING')
        if not (5 <= self.stall_window_s <= 120 and .02 <= self.stall_extent_m <= 1.):
            raise ValueError('INVALID_STAGNATION_SETTINGS')
        if not (30 <= self.work_base_s <= self.work_max_s <= 1800
                and 5 <= self.extension_window_s <= self.work_base_s
                and .01 <= self.extension_gain_pp <= 10):
            raise ValueError('INVALID_ZONE_BUDGET_SETTINGS')
        for v,lo,hi in ((self.stall_recoveries,1,30),(self.goal_recovery_limit,0,100),
                        (self.zone_cancel_limit,1,3)):
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError('INVALID_GUARD_INTEGER')
        return self

    @classmethod
    def from_args(cls,a):
        return cls(a.guard_stall_window_s,a.guard_stall_extent_m,a.guard_stall_recoveries,
                   a.guard_goal_recovery_limit,a.guard_zone_cancel_limit,
                   a.guard_zone_work_base_s,a.guard_zone_work_max_s,
                   a.guard_extension_window_s,a.guard_extension_gain_pp).validate()

class StagnationWindow:
    """Continuous fresh pose samples; a loop's endpoint is not motion evidence.

    Uses extent of the entire position window, not endpoint displacement,
    distance_remaining, yaw or wheel velocity. Exact Goal identity is external.
    """
    def __init__(self,settings):
        self.s=settings.validate(); self.samples=deque(); self.last=None
    def reset(self):
        self.samples.clear();self.last=None
    def sample(self,t,pose,recoveries):
        if (not math.isfinite(t) or len(pose)<2 or
                not all(math.isfinite(float(x)) for x in pose[:2]) or
                type(recoveries) is not int or recoveries<0):
            self.reset();return None
        if self.last is not None:
            if t<=self.last:
                if t<self.last:self.reset()
                return None
            if t-self.last>2.0:self.reset()
        self.last=t
        x,y=map(float,pose[:2]);self.samples.append((t,x,y,recoveries))
        edge=t-self.s.stall_window_s
        while len(self.samples)>1 and self.samples[1][0]<=edge:
            self.samples.popleft()
        if len(self.samples)<3 or t-self.samples[0][0]<self.s.stall_window_s:return None
        if recoveries<self.samples[0][3]:self.reset();return None
        xs=[r[1] for r in self.samples];ys=[r[2] for r in self.samples]
        extent=math.hypot(max(xs)-min(xs),max(ys)-min(ys))
        delta=recoveries-self.samples[0][3]
        if extent>=self.s.stall_extent_m or delta<self.s.stall_recoveries:return None
        return dict(reason=STAGNATION,window_s=t-self.samples[0][0],
                    extent_m=extent,recovery_increase=delta,recoveries=recoveries,
                    sample_count=len(self.samples),bounds=[min(xs),min(ys),max(xs),max(ys)],
                    limits=dict(window_s=self.s.stall_window_s,extent_m=self.s.stall_extent_m,
                                recoveries=self.s.stall_recoveries),
                    physical_blockage_proven=False,normal_termination=False)

class ZoneWorkLedger:
    """Cumulative work seconds; transit may pause, but can never reset a budget.

    Extension requires >=1pp within the last 60 ROS seconds at expiry.
    Coverage while excluded transit must not manufacture an extension.
    """
    def __init__(self,settings):
        self.s=settings.validate();self.elapsed=0.;self.allowance=self.s.work_base_s
        self.last=None;self.was_charge=False;self.history=deque();self.extended=False
        self.expired=False;self.recent_gain=0.
    def pause(self):
        self.last=None;self.was_charge=False;self.history.clear()
    def tick(self,t,coverage_percent,charge):
        if not math.isfinite(t) or not math.isfinite(coverage_percent) or not 0<=coverage_percent<=100:
            self.pause();return
        if self.last is not None:
            dt=t-self.last
            if dt<0 or dt>2.0:self.pause()
            elif charge and self.was_charge:self.elapsed+=dt
        self.last=t;self.was_charge=bool(charge)
        if not charge:self.history.clear();return
        if self.history and coverage_percent<self.history[-1][1]-1e-6:
            self.history.clear()
        self.history.append((t,coverage_percent))
        while self.history and t-self.history[0][0]>self.s.extension_window_s:
            self.history.popleft()
        self.recent_gain=coverage_percent-self.history[0][1] if self.history else 0.
        if self.elapsed>=self.allowance and not self.expired:
            if (not self.extended and self.s.work_max_s>self.allowance
                    and self.recent_gain>=self.s.extension_gain_pp-1e-8):
                self.extended=True;self.allowance=self.s.work_max_s
            if self.elapsed>=self.allowance:self.expired=True
    def snapshot(self):
        return dict(work_time_ros_s=round(self.elapsed,6),allowance_ros_s=self.allowance,
                    extended=self.extended,expired=self.expired,
                    recent_gain_pp=round(self.recent_gain,6),
                    extension_window_ros_s=self.s.extension_window_s,
                    time_basis='ROS_SIMULATION_WORK_EXCLUDES_ENTRY_AND_OUTSIDE_ZONE_TRANSIT')


class MissionBudgetRuntime:
    def __init__(self,node,bundle,cfg,args,output):
        self.n=node;self.bundle=bundle;self.cfg=cfg;self.args=args;self.output=output
        self.s=GuardSettings.from_args(args);self.watch=StagnationWindow(self.s)
        self.goal_id=None;self.recoveries=None;self.feedback_wall=None;self.feedback_start=0.
        self.pending=None;self.records=[];self.zone_cancels={};self.ledgers={}
        self.work_zone=None;self.last_sample=-math.inf;self.epoch=None
        node.info('mission_guard',f'[MISSION_BUDGET_POLICY] work={self.s.work_base_s:.0f}/{self.s.work_max_s:.0f}s ROS; '
                  f'stall={self.s.stall_window_s:.0f}s extent<{self.s.stall_extent_m:.2f}m recoveries+{self.s.stall_recoveries}; '
                  f'goal_recovery_limit={self.s.goal_recovery_limit}; no deferred-zone auto-revisit; '
                  'operational stop is NOT blockage/completion',0)
        node.event('MISSION_BUDGET_POLICY',settings=asdict(self.s),no_auto_revisit=True,
                   time_basis='ROS_SIMULATION',normal_termination_from_budget=False)

    def begin(self,inp):
        self.goal_id=self.n.nav_uuid;self.recoveries=None;self.feedback_wall=None
        self.feedback_start=time.monotonic();self.pending=None;self.watch.reset()
        self.last_sample=-math.inf;self.epoch=inp['epoch']

    def note_feedback(self,msg):
        n=self.n
        if (self.goal_id is None or self.goal_id!=n.nav_uuid or
                bytes(msg.goal_id.uuid).hex()!=self.goal_id):return
        try:v=int(msg.feedback.number_of_recoveries)
        except (AttributeError,TypeError,ValueError,OverflowError):return
        if v<0:return
        if self.recoveries is not None and v<self.recoveries:self.watch.reset()
        self.recoveries=v;self.feedback_wall=time.monotonic()

    def account(self,inp):
        """Called each poll, without requesting motion or changing any outcome."""
        from zone_planner import is_inside
        n=self.n
        if inp is None or n.stop_requested:
            if self.work_zone in self.ledgers:self.ledgers[self.work_zone].pause()
            self.work_zone=None;self.watch.reset();return
        current=n.current
        zid=(current.get('zone_id') if current else None) or (
            n.connectivity.zid if n.connectivity else None) or n.book.active_zone
        if zid not in n.book.records or n.book.is_complete(zid) or n.book.records[zid].state=='PARTIAL_DEFERRED':
            if self.work_zone in self.ledgers:self.ledgers[self.work_zone].pause()
            self.work_zone=None;return
        if self.work_zone and self.work_zone!=zid:self.ledgers[self.work_zone].pause()
        self.work_zone=zid
        ledger=self.ledgers.setdefault(zid,ZoneWorkLedger(self.s))
        nav_active=n.nav_uuid is not None and n.nav_terminal is None
        charge=True
        if nav_active:
            charge=(bool(current) and current.get('goal_kind')=='OBSERVE' and
                    is_inside(n.zones_by_id[zid],inp['pose'],self.bundle,self.cfg))
        # No work charged while canceling or waiting for a terminal stop/coverage.
        if n.cancel_reason is not None or n.nav_terminal is not None:charge=False
        old=ledger.extended
        ledger.tick(n.ros_s(),100*n.book.records[zid].measured,charge)
        if ledger.extended and not old:
            n.event('ZONE_WORK_BUDGET_EXTENDED',zone=zid,**ledger.snapshot())
            n.info('budget_extend',f'[ZONE_WORK_BUDGET_EXTENDED] {zid}: {ledger.allowance:.0f} simulation work seconds',0)

    def step(self,inp):
        n=self.n;wall=time.monotonic();now=n.ros_s()
        if (inp is None or n.stop_requested or n.cancel_reason is not None or
                n.nav_terminal is not None or n.nav_handle is None or
                n.nav_uuid!=self.goal_id or not n.current):
            self.watch.reset();return
        if inp['epoch']!=self.epoch:
            n.request_stop('DATA_LOSS_GUARD_SESSION_CHANGED');return
        zid=n.current['zone_id'];ledger=self.ledgers.get(zid)
        if ledger and ledger.expired:
            self.request_cancel(ZONE_LIMIT,dict(reason=ZONE_LIMIT,**ledger.snapshot()),inp);return
        age=wall-(self.feedback_wall if self.feedback_wall is not None else self.feedback_start)
        if age>10.:
            n.request_stop('DATA_LOSS_GUARD_FEEDBACK');return
        if self.recoveries is None or age>3.:
            self.watch.reset();return
        if self.s.goal_recovery_limit and self.recoveries>=self.s.goal_recovery_limit:
            self.request_cancel(RECOVERY_LIMIT,dict(reason=RECOVERY_LIMIT,
                recoveries=self.recoveries,recovery_limit=self.s.goal_recovery_limit),inp);return
        if now-self.last_sample<.20:return
        self.last_sample=now
        evidence=self.watch.sample(now,inp['pose'],self.recoveries)
        if evidence:self.request_cancel(STAGNATION,evidence,inp)

    def request_cancel(self,reason,evidence,inp):
        n=self.n
        if (n.stop_requested or n.cancel_reason or n.nav_uuid!=self.goal_id or
                n.nav_terminal is not None or n.nav_handle is None):return
        zid=n.current['zone_id']
        self.pending=dict(schema='dn012_operational_cancel_v1',goal_id=n.nav_uuid,
            zone=zid,goal_kind=n.current['goal_kind'],reason=reason,
            bundle_signature=self.bundle[4],coverage_session=n.session_id,epoch=inp['epoch'],
            requested_ros_s=n.ros_s(),requested_pose=list(inp['pose']),
            evidence=dict(evidence,physical_blockage_proven=False,normal_termination=False),
            settings=asdict(self.s))
        n.event('MISSION_GUARD_CANCEL_REQUESTED',record=self.pending)
        n.warn('guard_cancel',f'[MISSION_GUARD_CANCEL_REQUESTED] {zid}: {reason}; '
               'operational limit, NOT unreachable proof',0)
        n.request_cancel(reason)

    def authorizes_cancel(self):
        n=self.n;r=self.pending
        return bool(r and not n.stop_requested and n.nav_uuid==r['goal_id'] and
                    n.current and n.current['zone_id']==r['zone'] and
                    n.cancel_reason==r['reason'] and r['reason'] in CANCEL_REASONS)

    def after_confirmed_stop(self,inp):
        from exploration_io import atomic_json
        from sequence_core import json_candidate
        n=self.n
        # Repeat the evidence gates here; a cancellation ACK alone is insufficient.
        if not self.authorizes_cancel() or n.nav_terminal!=5:
            raise ValueError('UNAPPROVED_OPERATIONAL_CANCEL')
        if inp is None or inp['epoch']!=self.pending['epoch']:
            n.finish('MISSION_GUARD_SESSION_UNVERIFIED',2);return False
        if not n.stop_evidence.ready(n.ros_s(),time.monotonic()):return False
        status=inp['status']
        if (int(status['evidence_scans'])<=n.post_result_scans or
                float(status['last_evidence_scan_end_s'])<=n.post_result_ros):return False
        if not n.local_start_clear(inp):
            n.finish('MISSION_GUARD_LOCAL_START_UNSAFE',2);return False
        zid=n.current['zone_id'];c=dict(n.current)
        record=dict(self.pending,terminal='CANCELED',stop_confirmed=True,new_coverage=True,
                    local_start_clear=True,confirmed_ros_s=n.ros_s(),
                    measured_percent=n.book.records[zid].measured*100.)
        self.records.append(record);self.zone_cancels[zid]=self.zone_cancels.get(zid,0)+1
        folder=self.output/'mission_guard';folder.mkdir(exist_ok=True)
        atomic_json(folder/f'cancel_{len(self.records):04d}.json',record)
        n.event('MISSION_GUARD_STOP_CONFIRMED',record=record)
        n.info('guard_stopped',f'[MISSION_GUARD_STOP_CONFIRMED] {zid}; terminal cancel + fresh stop + new Coverage',0)
        self.pending=None;self.watch.reset()
        if n.book.is_complete(zid):return True
        n.book.rejected.append(json_candidate(c))
        rt=n.connectivity
        # No speculative planner requests before this point.
        if record['reason']==ZONE_LIMIT or self.zone_cancels[zid]>=self.s.zone_cancel_limit:
            reason=ZONE_LIMIT if record['reason']==ZONE_LIMIT else 'NAVIGATION_RETRY_BUDGET_UNVERIFIED'
            self.defer(zid,reason,dict(operational_cancel=record,
                       cancellations_in_zone=self.zone_cancels[zid]));return True
        rt.zid=None;rt.entry_mode=False;rt.clear_round()
        rt.blockage_replans.pop(zid,None)
        if c['goal_kind']=='ENTRY':
            rt.failed_entries.setdefault(zid,[]).append(dict(x=float(c['x']),y=float(c['y']),
                yaw=float(c['yaw']),id=c.get('id'),kind='OPERATIONAL_CANCEL_NOT_BLOCKED_PROOF'))
            reason='ENTRY_OPERATIONAL_CANCEL_ACCESS_RECHECK'
        else:reason='OPERATIONAL_CANCEL_OBSERVATION_RECHECK'
        n.book.activate(zid);n.book.audit_requested[zid]=reason
        n.event('MISSION_GUARD_ALTERNATIVE_REQUESTED',zone=zid,attempt=self.zone_cancels[zid],
                max_cancellations=self.s.zone_cancel_limit,reason=reason)
        return True

    def defer(self,zid,reason,evidence):
        n=self.n;rt=n.connectivity
        evidence=dict(evidence,normal_termination=False,connectivity_verified=False,
                      possible_unexamined_routes=True,no_auto_revisit_this_mission=True)
        # Caller must drain all workers first or be in post-navigation handling.
        rt.save_evidence(zid,reason,evidence)
        n.book.finish_partial(zid,reason,evidence=evidence)
        rt.blockage_replans.pop(zid,None)
        if rt.zid==zid:rt.zid=None;rt.entry_mode=False;rt.clear_round()
        n.options=[];n.batch_zone=None;n.options_context=None;n.current=None
        n.validated_plan_mode=None;n.current_path=[];n.current_path_length=None
        n.event('ZONE_ASSESSMENT',zone=zid,classification=reason,normal_termination=False,
                measured_percent=100*n.book.records[zid].measured)
        n.warn('guard_defer',f'[ZONE_UNVERIFIED] {zid}: {reason}; NOT normal completion',0)
        n.transition('SELECTING')

    def before_selection(self,inp):
        """At a stationary decision point only; never interrupts a live planner."""
        n=self.n
        if inp is None or n.stop_requested:return False
        if n.nav_uuid is not None:return False
        if n.plan_send is not None or n.plan_handle is not None or n.plan_result is not None:return False
        zid=(n.current.get('zone_id') if n.current else None) or (
            n.connectivity.zid if n.connectivity else None) or n.book.active_zone
        ledger=self.ledgers.get(zid)
        if not ledger or not ledger.expired or n.book.is_complete(zid):return False
        if not n.stop_evidence.ready(n.ros_s(),time.monotonic()):return True
        # Finish outstanding CPU work before another Zone can use the worker.
        for attr,obj in (('worker',n),('future',n.connectivity)):
            f=getattr(obj,attr,None)
            if f is not None:
                if not f.done():
                    n.info('budget_wait','[ZONE_BUDGET_WAIT_WORKER] bounded worker draining; no new Goal',2.)
                    return True
                try:f.result()
                except Exception as exc:
                    n.request_stop('MISSION_GUARD_WORKER_ERROR')
                    n.warn('budget_worker',str(exc),0);return True
                setattr(obj,attr,None)
        n.job_context=None
        if not n.local_start_clear(inp):
            n.request_stop('LOCAL_START_NOT_CLEAR_OR_STALE');return True
        self.defer(zid,ZONE_LIMIT,ledger.snapshot());return True

    def finish_terminal(self):
        # Success/abort may race a cancel; never invent a confirmed cancel record.
        self.goal_id=None;self.pending=None;self.watch.reset()
        self.recoveries=None;self.feedback_wall=None

    def summary(self):
        return dict(enabled=True,settings=asdict(self.s),no_auto_revisit=True,
                    zone_work={z:v.snapshot() for z,v in self.ledgers.items()},
                    operational_canceled=len(self.records),cancellations=self.records)
