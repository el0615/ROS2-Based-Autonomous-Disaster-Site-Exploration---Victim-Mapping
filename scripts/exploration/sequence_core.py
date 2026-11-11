#!/usr/bin/env python3
"""Pure policies for bounded hospital observation missions; no ROS calls.

Uses the shared planning_core for geometry/visibility. Coverage
is always measured by hospital_zone_coverage, never synthesized from predictions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any
import numpy as np
from planning_core import (ThresholdTracker, build_plan, rerank_plan,
    oriented_footprint_clear, conservative_safe)
from entrance_navigation_overlay import apply_entrance_overlay


def angle_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


def fraction(status: dict, zone: dict) -> float:
    rec = next(s for s in status['zones'] if s['id'] == zone['id'])
    total = int(rec['reference_free_cells']); seen = int(rec['observed_cells'])
    if total != zone['known_free_cells'] or total <= 0 or not 0 <= seen <= total:
        raise ValueError('Invalid zone denominator/count')
    return seen / total


@dataclass
class ZoneRecord:
    tracker: Any
    state: str = 'UNOBSERVED'
    measured: float = 0.0
    views: int = 0
    low_gain_count: int = 0
    deferred_reason: str | None = None


class MissionBook:
    def __init__(self, zones: list[dict], threshold=.90, max_views=3, min_gain_pp=.20):
        if not 0 < threshold <= 1 or max_views < 1 or min_gain_pp < 0:
            raise ValueError('Invalid mission settings')
        self.zones = zones; self.threshold = threshold
        self.max_views = max_views; self.min_gain_pp = min_gain_pp
        self.records = {z['id']: ZoneRecord(ThresholdTracker(threshold)) for z in zones}
        self.session: str | None = None
        self.visits: list[dict] = []
        self.rejected: list[dict] = []
        self.locked_zone: str | None = None

    def update(self, status: dict) -> list[str]:
        sid = status['session_id']
        if self.session is not None and sid != self.session:
            raise ValueError('COVERAGE_SESSION_CHANGED_RESTART_REQUIRED')
        self.session = sid
        newly = []
        for z in self.zones:
            rec = self.records[z['id']]; old = rec.state
            f = fraction(status,z)
            if f + 1e-12 < rec.measured:
                raise ValueError('COVERAGE_REGRESSION_RESTART_REQUIRED')
            rec.measured = f
            state = rec.tracker.update(status,f)
            if state in ('OBSERVATION_SUFFICIENT','THRESHOLD_PENDING'):
                rec.state = state
            elif rec.deferred_reason:
                rec.state = 'PARTIAL_DEFERRED'
            else:
                rec.state = 'PARTIAL' if f > 0 else 'UNOBSERVED'
            if rec.state == 'OBSERVATION_SUFFICIENT' and old != rec.state:
                newly.append(z['id'])
        return newly

    def eligible(self) -> list[dict]:
        return [z for z in self.zones if self.records[z['id']].state not in
                ('OBSERVATION_SUFFICIENT','THRESHOLD_PENDING','PARTIAL_DEFERRED')]

    def is_complete(self, zone_id: str) -> bool:
        return self.records[zone_id].state == 'OBSERVATION_SUFFICIENT'

    def all_complete(self) -> bool:
        return all(r.state == 'OBSERVATION_SUFFICIENT' for r in self.records.values())

    def pending(self) -> bool:
        return any(r.state == 'THRESHOLD_PENDING' for r in self.records.values())

    def defer(self, zone_id: str, reason: str) -> None:
        r=self.records[zone_id];r.deferred_reason=reason
        if not self.is_complete(zone_id):r.state='PARTIAL_DEFERRED'
        if self.locked_zone==zone_id:self.locked_zone=None

    def record_visit(self, c: dict, before: float, after: float, nav_status: str,
                     reason: str | None) -> dict:
        r=self.records[c['zone_id']];r.views+=1
        gain=100*(after-before)
        record={k:c[k] for k in ('zone_id','id','x','y','yaw')}
        record.update(before_percent=100*before,after_percent=100*after,
                      gain_pp=gain,nav_status=nav_status,cancel_reason=reason)
        self.visits.append(record)
        if not self.is_complete(c['zone_id']):
            r.low_gain_count = r.low_gain_count+1 if gain<self.min_gain_pp else 0
            if r.views>=self.max_views:self.defer(c['zone_id'],'VIEWPOINT_LIMIT_THIS_RUN')
            elif r.low_gain_count>=2:self.defer(c['zone_id'],'LOW_MEASURED_GAIN_TWICE')
            else:self.locked_zone=c['zone_id']
        elif self.locked_zone==c['zone_id']:self.locked_zone=None
        return record

    def summary(self) -> dict:
        return {k:dict(state=r.state,measured_percent=100*r.measured,views=r.views,
                       deferred_reason=r.deferred_reason) for k,r in self.records.items()}


def pose_repeated(c: dict, previous: list[dict], radius=.40, yaw_tolerance=.44) -> bool:
    # A genuinely different view direction is allowed for a partial-FoV sensor.
    return any(p['zone_id']==c['zone_id'] and math.hypot(p['x']-c['x'],p['y']-c['y'])<=radius
               and angle_delta(p['yaw'],c['yaw'])<=yaw_tolerance for p in previous)


def zone_order(zones, geom, labels, observed, robot_pose, locked=None):
    """Euclidean lower-bound ordering ONLY; final poses use the grid planner.

    Not an exact globally optimal room ordering. Keeps an unfinished active room
    unless it becomes completed/deferred; searches other rooms if it has no view.
    """
    distances=[]
    for z in zones:
        js,is_=np.nonzero((labels==z['mask_id']) & ~observed)
        if not len(is_):continue
        gx,gy=geom.world_to_grid_float(robot_pose[0],robot_pose[1])
        lower=float(np.sqrt(np.min((is_+.5-gx)**2+(js+.5-gy)**2))*geom.resolution)
        distances.append((0 if z['id']==locked else 1,lower,z['id'],z))
    return [v[-1] for v in sorted(distances,key=lambda v:v[:3])]


def build_batch(bundle, snapshot, zones, cfg, previous, locked=None, shortlist=3,
                entrance_overlay=None, navfn_fallback=False, *, plan_builder=None):
    if plan_builder is None:
        plan_builder = build_plan
    data,geom,labels,free,signature=bundle
    entrance_info=None
    if entrance_overlay is not None:
        free,entrance_info=apply_entrance_overlay(
            free,labels,geom,snapshot['cost']['values'],snapshot['pose'],entrance_overlay)
    ordered=zone_order(zones,geom,labels,snapshot['observed'],snapshot['pose'],locked)
    results=[];failures={};diagnostics={}
    for z in ordered[:shortlist]:
        plan=plan_builder(geom,labels,free,z,snapshot['observed'],snapshot['pose'],
                        snapshot['sensor'],cfg,snapshot['cost']['values'],
                        allow_navfn_fallback=navfn_fallback)
        diagnostics[z['id']]=dict(
            status=plan['status'],sampled=plan.get('sampled_candidates',0),
            strict_reachable=plan.get('reachable_sampled_candidates',0),
            fallback_status=plan.get('fallback_status','NOT_REQUESTED'),
            fallback_raw_connected=plan.get('fallback_raw_connected',0),
            fallback_evaluated=plan.get('fallback_evaluated',0),
            fallback_proposed=plan.get('fallback_proposed',0))
        if plan['status']=='START_CLEARANCE_BLOCKED':
            return dict(candidates=[],failures={},fatal=plan['status'],detail=plan['reason'],
                        entrance=entrance_info,diagnostics=diagnostics)
        good=[]
        for c0 in plan['candidates']:
            c=dict(c0,zone_id=z['id'])
            if c['new_area_m2']>=cfg.min_gain_m2 and c['new_cells']>0 and not pose_repeated(c,previous):
                good.append(c)
        if not good and navfn_fallback and 'fallback_status' not in plan:
            # A strict-grid view may be valid in isolation but already rejected
            # by Navfn or repeated. Offer disconnected strict-safe endpoints on
            # the next pass rather than incorrectly deferring the whole room.
            plan=plan_builder(geom,labels,free,z,snapshot['observed'],snapshot['pose'],
                            snapshot['sensor'],cfg,snapshot['cost']['values'],
                            allow_navfn_fallback=True,force_navfn_fallback=True)
            good=[]
            for c0 in plan['candidates']:
                c=dict(c0,zone_id=z['id'])
                if c['new_area_m2']>=cfg.min_gain_m2 and c['new_cells']>0 and not pose_repeated(c,previous):
                    good.append(c)
            diagnostics[z['id']].update(
                fallback_status=plan.get('fallback_status','NOT_REQUESTED'),
                fallback_raw_connected=plan.get('fallback_raw_connected',0),
                fallback_evaluated=plan.get('fallback_evaluated',0),
                fallback_proposed=plan.get('fallback_proposed',0))
        diagnostics[z['id']]['gain_and_unique']=len(good)
        if not good:
            if navfn_fallback and plan.get('fallback_status'):
                failures[z['id']]='NO_FEASIBLE_VIEW_AFTER_'+plan['fallback_status']
            else:
                failures[z['id']]='NO_NEW_FEASIBLE_SAMPLED_VIEW_THIS_RUN'
        else:
            # Top 3 distinct proposals per room; same-room ranking remains the existing scorer.
            good.sort(key=lambda c:(-c['score'],c['path_est_m'],c['id']))
            results.extend(good[:3])
            if locked==z['id']:break
    if locked and any(c['zone_id']==locked for c in results):
        results=[c for c in results if c['zone_id']==locked]
    # Choose among nearby-room proposals using conservative estimated travel.
    # Navfn then validates each proposed pose; no global shortest-tour claim.
    results.sort(key=lambda c:(0 if c['zone_id']==locked else 1,
                              c['path_est_m'] if c['zone_id']!=locked else 0,
                              -c['score'],c['zone_id'],c['id']))
    return dict(candidates=results,failures=failures,fatal=None,
                entrance=entrance_info,diagnostics=diagnostics)


def refresh_candidate(c, observed, geom, cfg):
    d=c.copy();ids=np.asarray(d['visible_indices'],dtype=np.int64)
    d['new_indices']=ids[~observed.ravel()[ids]];d['new_cells']=int(len(d['new_indices']))
    d['new_area_m2']=d['new_cells']*geom.resolution**2
    d['score']=d['new_area_m2']/(d['path_est_m']+cfg.distance_regularizer_m)
    return d if d['new_area_m2']>=cfg.min_gain_m2 else None


def validate_nav_path(xys, frame, goal, robot_pose, geom, known_free, costmap,
                      end_tolerance=.12,start_tolerance=.60, *, diagnostics=True):
    if frame.lstrip('/')!='map' or len(xys)<2:raise ValueError('EMPTY_OR_WRONG_FRAME_PATH')
    a=np.asarray(xys,dtype=float)
    if a.shape[1]!=2 or not np.isfinite(a).all():raise ValueError('INVALID_PATH_POINTS')
    if math.dist(a[0],robot_pose[:2])>start_tolerance:raise ValueError('PATH_START_MISMATCH')
    if math.dist(a[-1],(goal['x'],goal['y']))>end_tolerance:raise ValueError('PATH_END_MISMATCH')
    total=0.
    for p,q in zip(a[:-1],a[1:]):
        d=float(np.linalg.norm(q-p));total+=d
        # Dense centreline check only: NOT swept footprint / dynamic-safety proof.
        n=max(1,math.ceil(d/(geom.resolution/3)))
        if n>10000:raise ValueError('DISCONTINUOUS_PATH')
        for k in range(n+1):
            x,y=p+(q-p)*k/n;i,j=map(math.floor,geom.world_to_grid_float(float(x),float(y)))
            if not geom.contains(i,j) or not known_free[j,i] or not 0<=costmap[j,i]<99:
                if diagnostics:
                    inside = geom.contains(i, j)
                    cost = int(costmap[j, i]) if inside else None
                    prior_free = bool(known_free[j, i]) if inside else False
                    reason = (
                        'OUT_OF_MAP' if not inside else
                        'PRIOR_NOT_FREE' if not prior_free else
                        'COSTMAP_BLOCKED' if cost >= 99 else
                        'COSTMAP_UNKNOWN'
                    )
                    print(
                        f'[PATH_DIAG] reason={reason} '
                        f'x={float(x):.3f} y={float(y):.3f} '
                        f'cost={cost} known_free={prior_free} '
                        f'from_start={math.dist((float(x),float(y)),robot_pose[:2]):.3f}m',
                        flush=True
                    )
                raise ValueError('PATH_CENTRELINE_NOT_FREE')
    return total


def validate_oriented_swept_path(xys, robot_pose, goal, geom, known_free,
                                 costmap, cfg, *, max_samples=50000, inflation_aware=False):
    """Fail-closed sampled J100 rectangle check of Navfn's polyline.

    STRICT mode checks blocked squares (unknown, cost >=99, prior occupied)
    for the oriented J100 rectangle. Inflation-aware mode keeps 99 forbidden
    to the center, but uses cost >=100 for full-rectangle occupied-cell checks.
    Both use the original 0.04m default margin. Neither proves motion safety. Includes
    robot->first waypoint, last waypoint->goal, stationary start/turn/goal
    rotations, and translations at segment headings. The path orientation here
    is an intentionally conservative turn-in-place model; Nav2's RPP dynamics
    and changing local costmap are NOT proved collision-free by this check.
    """
    pts=np.asarray(xys,dtype=float)
    if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<2 or len(pts)>20000 or not np.isfinite(pts).all():
        raise ValueError('FOOTPRINT_SWEEP_INVALID_PATH')
    if len(robot_pose)<3 or not all(math.isfinite(float(v)) for v in robot_pose[:3]):
        raise ValueError('FOOTPRINT_SWEEP_INVALID_START')
    if not all(math.isfinite(float(goal[k])) for k in ('x','y','yaw')):
        raise ValueError('FOOTPRINT_SWEEP_INVALID_GOAL')
    if costmap.shape!=known_free.shape or known_free.shape!=(geom.height,geom.width):
        raise ValueError('FOOTPRINT_SWEEP_MAP_SHAPE')
    allowed=known_free & (costmap>=0) & (costmap < (100 if inflation_aware else 99))
    # Distance/rotation sampling bounded by the already configured margin;
    # even so this is sampled static geometry, NOT a continuous/dynamic proof.
    step_m=min(.012,geom.resolution/5,max(.002,cfg.safety_margin_m/4))
    step_yaw=min(.04,max(.005,cfg.safety_margin_m/(4*cfg.robot_radius_m)))
    samples=0

    def check(x,y,yaw):
        nonlocal samples
        samples+=1
        if samples>max_samples:
            raise ValueError('FOOTPRINT_SWEEP_SAMPLE_LIMIT')
        if inflation_aware:
            # The inscribed inflated band (99) is a forbidden robot-CENTER
            # cost.  The rectangular body still checks all prior occupied,
            # unknown, and lethal (100) cells, with the original safety margin.
            ci,cj=map(math.floor,geom.world_to_grid_float(float(x),float(y)))
            if (not geom.contains(ci,cj) or not known_free[cj,ci]
                    or not 0<=costmap[cj,ci]<99):
                raise ValueError('INFLATION_AWARE_CENTER_BLOCKED')
        if not oriented_footprint_clear(allowed,geom,(float(x),float(y),float(yaw)),cfg):
            raise ValueError('FOOTPRINT_SWEEP_COLLISION')

    def turn(x,y,from_yaw,to_yaw):
        delta=math.atan2(math.sin(to_yaw-from_yaw),math.cos(to_yaw-from_yaw))
        n=max(1,math.ceil(abs(delta)/step_yaw))
        for k in range(1,n+1):check(x,y,from_yaw+delta*k/n)

    start_x,start_y,start_yaw=map(float,robot_pose[:3])
    goal_x,goal_y,goal_yaw=(float(goal[k]) for k in ('x','y','yaw'))
    check(start_x,start_y,start_yaw)
    # Include the possible <=0.60m Navfn start mismatch and <=0.12m goal
    # tolerance so these short omitted connections never escape the sweep.
    route=np.vstack((np.array([[start_x,start_y]]),pts,
                     np.array([[goal_x,goal_y]])))
    previous_yaw=start_yaw
    for p,q in zip(route[:-1],route[1:]):
        dx,dy=(float(q[0]-p[0]),float(q[1]-p[1]))
        length=math.hypot(dx,dy)
        if length<1e-9:continue
        yaw=math.atan2(dy,dx)
        turn(p[0],p[1],previous_yaw,yaw)
        n=max(1,math.ceil(length/step_m))
        for k in range(1,n+1):
            u=k/n
            check(p[0]+u*dx,p[1]+u*dy,yaw)
        previous_yaw=yaw
    turn(goal_x,goal_y,previous_yaw,goal_yaw)
    return samples


def ensure_execution_preflight(validation_mode: str | None,
                               route_mode: str | None,
                               experimental_fallback_execute: bool) -> None:
    """Fail closed: an alternate/static-only diagnostic never authorizes motion.

    Fallback strict execution also requires an explicit independent opt-in.
    A strict static pass is necessary but still not a dynamic safety guarantee.
    """
    if validation_mode != 'STRICT':
        raise ValueError('EXECUTION_REQUIRES_STRICT_STATIC_PREFLIGHT')
    if (route_mode == 'NAVFN_FALLBACK_UNVERIFIED'
            and not experimental_fallback_execute):
        raise ValueError('FALLBACK_EXECUTION_REQUIRES_EXPERIMENTAL_OPT_IN')


class StopEvidence:
    """Require fresh low-speed odometry, not just a CANCELED action status."""
    def __init__(self, duration=1.0,linear=.03,angular=.05,min_samples=5):
        self.duration=duration;self.linear=linear;self.angular=angular;self.min_samples=min_samples
        self.reset()
    def reset(self, after=-math.inf):
        self.after=after;self.first=None;self.last=None;self.wall=None;self.n=0
    def update(self, stamp, wall, vx,vy,vz,wz):
        if not all(math.isfinite(v) for v in (stamp,wall,vx,vy,vz,wz)):
            self.first=None;self.last=None;self.n=0;return
        if stamp<=self.after:return
        if self.last is not None and stamp<=self.last:return
        gap=self.last is not None and stamp-self.last>1.0
        self.last=stamp;self.wall=wall
        if gap or math.sqrt(vx*vx+vy*vy+vz*vz)>self.linear or abs(wz)>self.angular:
            self.first=None;self.n=0;return
        if self.first is None:self.first=stamp
        self.n+=1
    def ready(self,now,wall):
        return (self.first is not None and self.last is not None and self.wall is not None
                and -.1<=now-self.last<=1.0 and 0<=wall-self.wall<=2.0
                and self.n>=self.min_samples and self.last-self.first>=self.duration)


def safe_early_stop(free, cost, geom, pose, cfg):
    safe,_=conservative_safe(free & (cost>=0)&(cost<100),geom,cfg)
    i,j=map(math.floor,geom.world_to_grid_float(*pose[:2]))
    return bool(geom.contains(i,j) and safe[j,i] and cost[j,i]<99)


def json_candidate(c):
    if c is None:return None
    return {k:v for k,v in c.items() if k not in ('visible_indices','new_indices','path_cells')}
