"""DN-012 v1.2 pure second-pass policy and bounded geometry checks.

No ROS imports, publishers, motion commands or parameter changes.
Operational rejection is NOT a physical blockage certificate. All coordinates
and masks use the original Geometry / ROS bottom-up grid convention.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, is_dataclass
from types import SimpleNamespace
import math
import os
import time
import numpy as np
from scipy import ndimage
from regions import components, public_region, row_runs, validate_grids

VERSION = 'DN012_REGION_ENTRY_V1_6'
TERMINAL = {'OBSERVED', 'SMALL_REMAINDER', 'ROBOT_EXPLORATION_UNAVAILABLE',
            'UNVERIFIED', 'SPLIT'}
STALE_REASONS = {'POSE_OR_SESSION_CHANGED', 'COSTMAP_CHANGED_REPLAN',
                 'INPUTS_NOT_FRESH', 'GOAL_NO_LONGER_NEEDED',
                 'SECOND_PASS_CONTEXT_CHANGED'}
ACCESS_REJECTIONS = {'NAVFN_ABORTED', 'PATH_END_MISMATCH', 'GOAL_FOOTPRINT_BLOCKED',
                     'SECOND_PASS_ENDPOINT_MARGIN', 'SECOND_PASS_PATH_CLEARANCE',
                     'SECOND_PASS_ROTATION_CLEARANCE', 'SECOND_PASS_FAILED_ROUTE_HOLD'}


@dataclass(frozen=True)
class TwoPassSettings:
    min_area_m2: float = 1.0
    priority_area_m2: float = 1.
    goal_extra_margin_m: float = .10
    path_extra_margin_m: float = .02
    position_separation_m: float = .60
    low_yield_area_m2: float = .05
    min_view_gain_m2: float = .01
    max_nav_per_region: int = 3
    max_nav_per_zone: int = 8
    max_preflights_per_region: int = 12
    max_catalog_restarts: int = 3
    path_check_budget_s: float = 1.5
    max_path_samples: int = 8000
    live_horizon_m: float = 1.5
    monitor_period_s: float = .75

    def validate(self):
        for name in ('min_area_m2','priority_area_m2','goal_extra_margin_m',
                     'path_extra_margin_m','position_separation_m','low_yield_area_m2',
                     'min_view_gain_m2','path_check_budget_s','live_horizon_m','monitor_period_s'):
            x = getattr(self, name)
            if not math.isfinite(x) or x <= 0:
                raise ValueError('INVALID_TWO_PASS_SETTING:' + name)
        if not .05 <= self.min_area_m2 <= self.priority_area_m2 <= 20:
            raise ValueError('INVALID_TWO_PASS_AREAS')
        if not .01 <= self.path_extra_margin_m <= self.goal_extra_margin_m <= .30:
            raise ValueError('INVALID_TWO_PASS_MARGIN')
        if not .2 <= self.position_separation_m <= 2.:
            raise ValueError('INVALID_TWO_PASS_SEPARATION')
        for name, upper in (('max_nav_per_region',10),('max_nav_per_zone',20),
                            ('max_preflights_per_region',48),('max_catalog_restarts',10),
                            ('max_path_samples',20000)):
            n = getattr(self,name)
            if type(n) is not int or not 1 <= n <= upper:
                raise ValueError('INVALID_TWO_PASS_COUNT:' + name)
        return self

    @classmethod
    def from_environment(cls):
        env = os.environ
        return cls(
            min_area_m2=float(env.get('DN012_SECOND_ENTRY_MIN_AREA_M2','1.0')),
            priority_area_m2=float(env.get('DN012_REGION_PRIORITY_AREA_M2','1.0')),
            goal_extra_margin_m=float(env.get('DN012_SECOND_GOAL_MARGIN_M','.10')),
            path_extra_margin_m=float(env.get('DN012_SECOND_PATH_MARGIN_M','.02')),
            max_nav_per_zone=int(env.get('DN012_REGION_MAX_GOALS','8')),
            position_separation_m=float(env.get('DN012_REGION_SEPARATION_M','.6')),
        ).validate()


def finite_json(value):
    """Strict JSON boundary, not default=str. Reject NaN, infinity and unknown types."""
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('NONFINITE_REPORT_VALUE')
        return value
    if isinstance(value, (tuple, list)):
        return [finite_json(x) for x in value]
    if isinstance(value, dict):
        if not all(isinstance(k,str) for k in value):
            raise TypeError('REPORT_KEYS_MUST_BE_STRINGS')
        return {k:finite_json(v) for k,v in value.items()}
    raise TypeError('UNSUPPORTED_REPORT_TYPE:' + type(value).__name__)


def safe_candidate(candidate):
    """Remove computational arrays; callers then use the original json_candidate."""
    if candidate is None:
        return None
    excluded = {'visible_indices','new_indices','residual_target_indices','region_entry_indices','path_cells'}
    return finite_json({k:v for k,v in candidate.items() if k not in excluded and not k.startswith('_')})


def cfg_with_margin(cfg, extra):
    margin = float(cfg.safety_margin_m) + extra
    if is_dataclass(cfg):
        return replace(cfg, safety_margin_m=margin)
    # Tests and legacy Settings implementations may not be dataclasses.
    d = dict(vars(cfg)); d['safety_margin_m'] = margin
    return SimpleNamespace(**d)


def clearance_grid(free, resolution):
    """Lower bound at cell CENTERS to occupied cell square; exterior is unsafe."""
    d = ndimage.distance_transform_edt(np.pad(free,1,constant_values=False))[1:-1,1:-1]
    return np.maximum(0., d * resolution - resolution / math.sqrt(2.))


def strict_free(bundle, inp):
    _,geom,labels,known,_ = bundle
    validate_grids(labels,known,inp['cost']['values'],inp['observed'])
    cost=inp['cost']['values']
    return known & (cost>=0) & (cost<100)


def endpoint_guard(candidate, bundle, inp, cfg, settings, footprint_check):
    """Second-pass ONLY: footprint + all-heading disk for arrival/rotation.

    The circular envelope is deliberately conservative at VIEWPOINTS only. It
    does not enlarge Nav2 footprints, or certify that a rejected passage is cut.
    """
    _,g,labels,known,_=bundle
    free=strict_free(bundle,inp)
    x,y,yaw=(float(candidate[k]) for k in ('x','y','yaw'))
    if not all(math.isfinite(v) for v in (x,y,yaw)):
        return False, 'SECOND_PASS_INVALID_ENDPOINT', {}
    i,j=map(math.floor,g.world_to_grid_float(x,y))
    if not g.contains(i,j) or labels[j,i]<=0 or not free[j,i]:
        return False,'SECOND_PASS_UNKNOWN_OR_UNAPPROVED_ENDPOINT',{}
    margin_cfg=cfg_with_margin(cfg,settings.goal_extra_margin_m)
    if not footprint_check(free,g,(x,y,yaw),margin_cfg):
        return False,'SECOND_PASS_ENDPOINT_MARGIN',{'xy':[x,y]}
    # Sub-cell offset is subtracted, not ignored.
    cx,cy=g.cell_to_world(i,j)
    lower=float(clearance_grid(free,g.resolution)[j,i])-math.hypot(cx-x,cy-y)
    required=float(cfg.robot_radius_m)+float(cfg.safety_margin_m)+settings.goal_extra_margin_m
    if lower+1e-9<required:
        return False,'SECOND_PASS_ROTATION_CLEARANCE',{'xy':[x,y],
            'center_clearance_lower_m':lower,'required_envelope_radius_m':required}
    return True,'ENDPOINT_SAFE_IN_SNAPSHOT',{'xy':[x,y],
        'center_clearance_lower_m':lower,'required_envelope_radius_m':required,
        'viewpoint_zone_mask_id':int(labels[j,i])}


def observed_bundle(bundle, inp):
    """Navigation authorization only: NEVER edits observed/labels/denominator."""
    from regions import validate_grids
    data, g, labels, known, signature = bundle
    validate_grids(labels, known, inp['cost']['values'], inp['observed'])
    return data, g, labels, known & inp['observed'], signature

def remaining_distance(indices, geom, xy):
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return math.inf
    # Inverse rotation is handled by Geometry; distance is in the grid plane.
    gx, gy = geom.world_to_grid_float(float(xy[0]), float(xy[1]))
    jj, ii = np.divmod(indices, geom.width)
    return float(np.min(np.hypot(ii + .5 - gx, jj + .5 - gy)) * geom.resolution)

def path_guard(points, start_pose, goal, bundle, inp, cfg, settings,
               footprint_check, holds=(), *, check_endpoint=True, budget_s=None):
    """Bounded sampled swept-footprint + turn checks, never a physical-cut proof.

    This rejects an unsafe Nav2 preflight BEFORE a NavigateToPose is sent.
    ds and dtheta bound inter-sample body displacement; their error is included
    in extra padding. The original Nav2/RPP runtime collision checks remain.
    """
    started=time.monotonic();budget=settings.path_check_budget_s if budget_s is None else budget_s
    # v1.5: direct approach executes only short, measured-free path segments.
    if goal.get('direct_approach'):
        arr=np.asarray(points,dtype=float)
        if arr.ndim!=2 or arr.shape[1]!=2 or not len(arr) or not np.isfinite(arr).all():
            return False,'SECOND_PASS_INVALID_PATH',{'samples':0}
        distance=float(np.linalg.norm(np.diff(arr,axis=0),axis=1).sum())
        distance+=math.dist(start_pose[:2],arr[0])
        if check_endpoint and distance>float(goal['approach_max_route_m'])+1e-9:
            return False,'DIRECT_APPROACH_DETOUR_LIMIT',{'path_length_m':distance,
                'limit_m':float(goal['approach_max_route_m']),'not_access_failure':True}
        bundle=observed_bundle(bundle,inp)
    _,g,_,_,_=bundle
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<1 or not np.isfinite(p).all():
        return False,'SECOND_PASS_INVALID_PATH',{'samples':0}
    if len(p)==1:
        p=np.vstack((np.asarray(start_pose[:2],dtype=float),p))
    if not np.isfinite(np.asarray(start_pose,dtype=float)).all():
        return False,'SECOND_PASS_INVALID_START',{'samples':0}
    free=strict_free(bundle,inp)
    dmap=clearance_grid(free,g.resolution)
    ds=min(g.resolution/2.,.025)
    radius=float(cfg.robot_radius_m)+float(cfg.safety_margin_m)
    da=min(math.radians(5.),ds/max(radius,ds))
    # Translational and rotational interpolation uncertainty is explicitly padded.
    checked_cfg=cfg_with_margin(cfg,settings.path_extra_margin_m+ds/2.+radius*da/2.)
    count=0;min_clear=None
    def pose_ok(x,y,a,kind):
        nonlocal count,min_clear
        count+=1
        if count>settings.max_path_samples or time.monotonic()-started>budget:
            return False,'SECOND_PASS_PATH_CHECK_BUDGET',{'samples':count}
        i,j=map(math.floor,g.world_to_grid_float(x,y))
        if not g.contains(i,j) or not free[j,i]:
            return False,'SECOND_PASS_UNKNOWN_OR_UNAPPROVED_PATH',{'samples':count,'xy':[x,y]}
        for h in holds:
            if math.hypot(x-h['xy'][0],y-h['xy'][1])<h.get('radius_m',settings.position_separation_m):
                return False,'SECOND_PASS_FAILED_ROUTE_HOLD',{'samples':count,'xy':[x,y]}
        if not footprint_check(free,g,(x,y,a),checked_cfg):
            return False,kind,{'samples':count,'xy':[x,y],'yaw':a}
        cx,cy=g.cell_to_world(i,j)
        lower=float(dmap[j,i])-math.hypot(x-cx,y-cy)
        min_clear=lower if min_clear is None else min(min_clear,lower)
        return True,'OK',{}
    previous=float(start_pose[2])
    current=np.asarray(start_pose[:2],dtype=float)
    if math.dist(current,p[0])>.5:
        return False,'SECOND_PASS_PATH_START_MISMATCH',{'samples':0}
    # Include the start-to-first-point bridge (not only the published polyline).
    route=np.vstack((current,p))
    for a,b in zip(route[:-1],route[1:]):
        dx,dy=b-a;length=float(math.hypot(dx,dy))
        if length<1e-8:continue
        heading=math.atan2(dy,dx)
        delta=math.atan2(math.sin(heading-previous),math.cos(heading-previous))
        turns=max(1,math.ceil(abs(delta)/da))
        for k in range(turns+1):
            ok,reason,e=pose_ok(float(a[0]),float(a[1]),previous+delta*k/turns,
                               'SECOND_PASS_ROTATION_CLEARANCE')
            if not ok:return ok,reason,e
        steps=max(1,math.ceil(length/ds))
        for k in range(1,steps+1):
            xy=a+(b-a)*k/steps
            ok,reason,e=pose_ok(float(xy[0]),float(xy[1]),heading,'SECOND_PASS_PATH_CLEARANCE')
            if not ok:return ok,reason,e
        previous=heading
    if check_endpoint:
        ok,reason,e=endpoint_guard(goal,bundle,inp,cfg,settings,footprint_check)
        if not ok:return ok,reason,dict(e,samples=count)
        if math.dist(p[-1],(goal['x'],goal['y']))>.5:
            return False,'SECOND_PASS_PATH_END_MISMATCH',{'samples':count}
        # Last plan pose can be inside the goal tolerance; check its terminal yaw too.
        delta=math.atan2(math.sin(goal['yaw']-previous),math.cos(goal['yaw']-previous))
        turns=max(1,math.ceil(abs(delta)/da))
        for k in range(turns+1):
            ok,reason,e=pose_ok(float(p[-1,0]),float(p[-1,1]),previous+delta*k/turns,
                               'SECOND_PASS_ROTATION_CLEARANCE')
            if not ok:return ok,reason,e
    return True,'PATH_SAFE_IN_SNAPSHOT',dict(samples=count,
        minimum_center_clearance_lower_m=min_clear,extra_margin_m=settings.path_extra_margin_m,
        interpolation_padding_m=ds/2.+radius*da/2.,elapsed_s=time.monotonic()-started,
        not_a_continuous_dynamic_safety_guarantee=True)


def remaining_path(points, pose, horizon):
    """Remaining XY horizon; no checked passage behind robot is re-evaluated."""
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():return []
    here=np.asarray(pose[:2],dtype=float);best=None
    for k,(a,b) in enumerate(zip(p[:-1],p[1:])):
        v=b-a;den=float(v@v)
        t=0. if den==0 else min(1.,max(0.,float((here-a)@v)/den))
        q=a+t*v;d=float(np.linalg.norm(q-here))
        if best is None or d<best[0]:best=(d,k,q)
    if best[0]>.5:return []
    route=[here,best[2]];distance=float(np.linalg.norm(here-best[2]))
    for q in p[best[1]+1:]:
        a=route[-1];step=float(np.linalg.norm(q-a))
        if distance+step>horizon and step>0:
            route.append(a+(q-a)*(horizon-distance)/step);break
        route.append(q);distance+=step
        if distance>=horizon:break
    return [list(map(float,q)) for q in route]


def refresh_second_pass(candidate, observed, geom):
    if candidate.get('residual_entry'):
        from region_entry import refresh_entry
        return refresh_entry(candidate, observed, geom)
    c=dict(candidate)
    target=np.asarray(c['residual_target_indices'],dtype=np.int64)
    obs=np.asarray(observed,dtype=bool).ravel()
    remaining=target[~obs[target]]
    if not len(remaining):return None
    # Do not drive to remnants split into only sub-threshold fragments.
    mask=np.zeros(observed.shape,dtype=bool);mask.ravel()[remaining]=True
    parts=components(mask,geom,float(c['residual_min_area_m2']))
    if not parts:return None
    remaining=np.concatenate([r['indices'] for r in parts])
    visible=np.asarray(c['visible_indices'],dtype=np.int64)
    novel=visible[~obs[visible]]
    overlap=np.intersect1d(novel,remaining,assume_unique=False)
    if c.get('direct_approach'):
        gain=remaining_distance(remaining,geom,c['approach_start_xy'])-remaining_distance(remaining,geom,(c['x'],c['y']))
        if gain+1e-9<float(c['approach_min_progress_m']):return None
        c['approach_progress_m']=gain
    elif len(overlap)*geom.resolution**2+1e-10<float(c.get('second_min_view_gain_m2',.01)):
        return None
    c.update(residual_target_indices=remaining.astype(np.int32),residual_target_cells=int(len(remaining)),
             new_indices=novel.astype(np.int32),new_cells=int(len(novel)),
             new_area_m2=float(len(novel)*geom.resolution**2),
             residual_overlap_cells=int(len(overlap)))
    return c




class TaskLedger:
    """Independent exploration ledger. NEVER alters ThresholdTracker or Coverage."""
    def __init__(self,bundle,settings):
        self.bundle=bundle;self.settings=settings;self.tasks={};self.order=0
        self.initialized=False;self.session=None;self.small_by_zone={}
        self.inventoried_zones=set()
    def inventory(self,inp,zones):
        if self.initialized:raise ValueError('SECOND_PASS_INVENTORY_ALREADY_CREATED')
        self.session=inp['status']['session_id'];_,g,labels,known,_=self.bundle
        validate_grids(labels,known,inp['cost']['values'],inp['observed'])
        for z in zones:
            self.inventoried_zones.add(z['id'])
            mask=(labels==z['mask_id'])&known&~inp['observed']
            all_parts=components(mask,g)
            self.small_by_zone[z['id']]=sum(r['cells'] for r in all_parts
                if r['area_m2']+1e-9<self.settings.min_area_m2)
            for r in all_parts:
                if r['area_m2']+1e-9>=self.settings.min_area_m2:self.add(z['id'],r)
        self.initialized=True
    def inventory_zone(self,inp,zone):
        """Append this Zone only; completed Zones are never reopened for motion."""
        if self.session is not None and inp['status']['session_id']!=self.session:
            raise ValueError('COVERAGE_SESSION_CHANGED')
        if zone['id'] in self.inventoried_zones:return
        self.session=inp['status']['session_id']
        _,g,labels,known,_=self.bundle
        validate_grids(labels,known,inp['cost']['values'],inp['observed'])
        parts=components((labels==zone['mask_id'])&known&~inp['observed'],g)
        self.small_by_zone[zone['id']]=sum(r['cells'] for r in parts if r['area_m2']+1e-9<self.settings.min_area_m2)
        for r in parts:
            if r['area_m2']+1e-9>=self.settings.min_area_m2:self.add(zone['id'],r)
        self.inventoried_zones.add(zone['id']);self.initialized=True

    def add(self,zid,region,parent=None):
        tid=f'{zid}:{region["region_id"]}'
        if tid in self.tasks:return tid
        self.order+=1
        t=dict(region,task_id=tid,zone_id=zid,parent_task_id=parent,status='PENDING',
               reason='SIGNIFICANT_RESIDUAL_MUST_BE_REVIEWED',sequence=self.order,
               checks=0,nav_sent=0,restarts=0,attempts=[],checked_positions=[],
               catalog_stats=None,source_cost_key=None,last_update_ros_s=None,
               completion_scope='not_assessed',original_cells=region['cells'])
        self.tasks[tid]=t;return tid
    def refresh(self,inp):
        if not self.initialized:return
        if inp['status']['session_id']!=self.session:raise ValueError('COVERAGE_SESSION_CHANGED')
        _,g,_,_,_=self.bundle
        for tid,t in list(self.tasks.items()):
            if t['status']=='SPLIT':continue
            remaining=t['indices'][~inp['observed'].ravel()[t['indices']]]
            # Preserve old finding, but do not leave an unavailable overlay on observed cells.
            t['indices']=remaining.astype(np.int32);t['cells']=int(len(remaining))
            t['area_m2']=float(len(remaining)*g.resolution**2)
            if not len(remaining):
                t.update(status='OBSERVED',reason='FRESH_COVERAGE_RESOLVED',completion_scope='measured_cells')
                continue
            mask=np.zeros(inp['observed'].shape,dtype=bool);mask.ravel()[remaining]=True
            parts=components(mask,g,self.settings.min_area_m2)
            if not parts:
                t.update(status='SMALL_REMAINDER',reason='NO_CONNECTED_SIGNIFICANT_REMAINDER',
                         completion_scope='raw_coverage_unchanged_small_cells_retained')
            elif len(parts)>1 and t['status'] not in TERMINAL:
                children=[self.add(t['zone_id'],r,tid) for r in parts]
                t.update(status='SPLIT',reason='OBSERVATION_SPLIT_RESIDUAL',children=children)
    def pending(self):return [t for t in self.tasks.values() if t['status'] not in TERMINAL]
    def next_task(self,pose,zone_id=None):
        pending=[t for t in self.pending() if zone_id is None or t['zone_id']==zone_id]
        if not pending:return None
        return min(pending,key=lambda t:(t['area_m2']<self.settings.priority_area_m2,
                   t['checks']//2,math.dist(pose[:2],t['centroid_map_m']),t['sequence']))
    def finish(self,tid,status,reason,inp=None):
        if status not in TERMINAL:raise ValueError('INVALID_REGION_TERMINAL_STATUS')
        t=self.tasks[tid];t.update(status=status,reason=reason,
            completion_scope='bounded_robot_policy_not_physical_proof')
        if inp is not None:t['source_cost_key']=inp['cost']['key']
    def exhaustion_verdict(self,tid,stats):
        """Only exhausted APPROACH checks can become robot-unavailable."""
        t=self.tasks[tid]
        if stats.get('direct_approach') or stats.get('residual_entry'):
            # This is an operational label for THIS bounded robot approach,
            # never a proof of no route through other entrances.
            if not stats.get('catalog_complete'):return 'UNVERIFIED','APPROACH_CATALOG_INCOMPLETE'
            reasons=[a.get('reason','') for a in t['attempts']]
            if any('BUDGET' in r or 'TIMEOUT' in r or r in STALE_REASONS or r=='DIRECT_APPROACH_DETOUR_LIMIT' for r in reasons):
                return 'UNVERIFIED','APPROACH_CHECK_INCOMPLETE_OR_DETOUR_LIMIT'
            if any(a.get('nav_status')=='SUCCEEDED' for a in t['attempts']):
                return 'UNVERIFIED','APPROACH_SUCCEEDED_RESIDUAL_NOT_YET_OBSERVED'
            count=int(stats.get('approach_candidates',0))
            attempts=t['attempts']
            if stats.get('cost_key') is not None:
                attempts=[a for a in attempts if a.get('cost_key')==stats['cost_key']
                    and a.get('start_xy') is not None
                    and math.dist(a['start_xy'],stats['start_xy'])<=.15]
            failures={tuple(a['xy']) for a in attempts if a.get('reason') in ACCESS_REJECTIONS or a.get('nav_status')=='ABORTED'}
            if count>=2 and len(failures)>=count:
                return 'ROBOT_EXPLORATION_UNAVAILABLE','ALL_BOUNDED_ROBOT_APPROACHES_FAILED'
            return 'UNVERIFIED','NO_VERIFIED_LOCAL_APPROACH_REMAINING'
        if not stats.get('catalog_complete',False):return 'UNVERIFIED','CATALOG_BUDGET_OR_UNTESTED_TILES'
        if stats.get('unknown_cells_near_target',0)>0 and not stats.get('positive_views',0):
            return 'UNVERIFIED','INSUFFICIENT_KNOWN_VIEWPOINTS'
        reasons=[a['reason'] for a in t['attempts']]
        if any('BUDGET' in r or 'TIMEOUT' in r or r in STALE_REASONS for r in reasons):
            return 'UNVERIFIED','INCOMPLETE_OR_STALE_ASSESSMENT'
        if any(a.get('nav_status')=='SUCCEEDED' for a in t['attempts']):
            return 'UNVERIFIED','ACCESS_SUCCEEDED_BUT_SENSING_REMAINED_INSUFFICIENT'
        # All generated safe, useful candidates actually had path/approach failures.
        failed_positions={tuple(a['xy']) for a in t['attempts'] if a['reason'] in ACCESS_REJECTIONS or a.get('nav_status')=='ABORTED'}
        count=int(stats.get('positive_views',0))
        if count>0 and len(failed_positions)>=count:
            return 'ROBOT_EXPLORATION_UNAVAILABLE','ALL_SAMPLED_APPROACHES_FAILED'
        if (stats.get('positions_considered',0)>0 and stats.get('endpoint_rejected',0)==stats['positions_considered']):
            return 'ROBOT_EXPLORATION_UNAVAILABLE','NO_SAFE_VIEWPOINT_IN_COMPLETE_SAMPLED_CATALOG'
        return 'UNVERIFIED','NO_VERIFIED_SENSING_OR_ACCESS_SOLUTION'
    def public(self,inp=None):
        _,g,_,_,_=self.bundle
        out=[]
        for t in self.tasks.values():
            d={k:v for k,v in t.items() if k!='indices'}
            d['cell_runs_ros']=row_runs(t['indices'],g.width)
            if len(t['indices']):
                yy,xx=np.divmod(t['indices'],g.width)
                d['centroid_map_m']=list(map(float,g.cell_to_world(float(xx.mean()),float(yy.mean()))))
                d['bbox_cells_ros']=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
            d['not_physical_blockage_proof']=True
            if inp is not None:
                d['evidence_is_historical']=d.get('source_cost_key') not in (None,inp['cost']['key'])
            out.append(finite_json(d))
        return out
