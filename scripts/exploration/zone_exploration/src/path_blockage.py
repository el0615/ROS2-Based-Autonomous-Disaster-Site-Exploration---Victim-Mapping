"""Read-only, finite-grid path blockage evidence. No ROS or motion commands.

A path/footprint intersection is NOT proof that every route is blocked. Unknown
and inflation are never new-obstacle witnesses. Current-map cut diagnosis is
optimistic (8-neighbour point connectivity, unknown cost cells allowed) and is
only meaningful under the explicitly selected fixed-prior-topology assumption.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import hashlib
import math
import numpy as np
from scipy import ndimage

CANCEL_REASON = 'PATH_BLOCKAGE_REPLAN_REQUIRED'

@dataclass(frozen=True)
class BlockageSettings:
    lookahead_m: float = 3.0
    sample_step_m: float = .05
    max_lateral_m: float = .50
    hit_tolerance_m: float = .12
    stationary_window_s: float = 5.0
    progress_m: float = .20
    persistence_s: float = 2.0
    minimum_samples: int = 3
    max_reference_age_s: float = 60.0
    max_reference_wall_s: float = 120.0
    max_cancel_per_zone: int = 2
    def validate(self):
        for key in ('lookahead_m','sample_step_m','max_lateral_m','hit_tolerance_m',
                    'stationary_window_s','progress_m','persistence_s',
                    'max_reference_age_s','max_reference_wall_s'):
            v=getattr(self,key)
            if not math.isfinite(v) or v<=0:raise ValueError('INVALID_BLOCKAGE_'+key)
        if not 2<=self.minimum_samples<=10 or not 1<=self.max_cancel_per_zone<=3:
            raise ValueError('INVALID_BLOCKAGE_COUNTS')
        if self.sample_step_m>.10 or self.lookahead_m>6 or self.hit_tolerance_m>.2:
            raise ValueError('UNSUPPORTED_BLOCKAGE_GEOMETRY')
        if self.persistence_s>self.stationary_window_s:
            raise ValueError('BLOCKAGE_PERSISTENCE_EXCEEDS_PROGRESS_WINDOW')


def project_scan_hits(ranges, angle_min, angle_increment, range_min, range_max,
                      first_pose, last_pose=None, range_cap=8.):
    """Finite returns only, linear SE(2) scan-end interpolation, not SLAM mapping."""
    r=np.asarray(ranges,dtype=float)
    if r.ndim!=1 or not 2<=len(r)<=4096:raise ValueError('INVALID_SCAN_SIZE')
    vals=[angle_min,angle_increment,range_min,range_max,range_cap,*first_pose]
    if last_pose is not None:vals.extend(last_pose)
    if not np.isfinite(vals).all() or range_min<0 or range_max<=range_min:
        raise ValueError('INVALID_SCAN_GEOMETRY')
    ids=np.flatnonzero(np.isfinite(r)&(r>=range_min)&(r>0)&(r<min(range_max,range_cap)))
    if not len(ids):return np.empty((0,2))
    end=first_pose if last_pose is None else last_pose
    t=ids/max(1,len(r)-1)
    dyaw=math.atan2(math.sin(end[2]-first_pose[2]),math.cos(end[2]-first_pose[2]))
    yaw=first_pose[2]+t*dyaw+angle_min+ids*angle_increment
    return np.column_stack((first_pose[0]+t*(end[0]-first_pose[0])+r[ids]*np.cos(yaw),
                            first_pose[1]+t*(end[1]-first_pose[1])+r[ids]*np.sin(yaw)))


def _remaining_samples(points,pose,step,lookahead,max_lateral):
    a=np.asarray(points,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or not 2<=len(a)<=20000 or not np.isfinite(a).all():
        return None
    d=np.diff(a,axis=0);lens=np.linalg.norm(d,axis=1)
    usable=lens>1e-9
    if not usable.any():return None
    t=np.zeros(len(d));np.divide(np.sum((np.asarray(pose[:2])-a[:-1])*d,axis=1),
                                 lens*lens,out=t,where=usable)
    t=np.clip(t,0.,1.);proj=a[:-1]+t[:,None]*d
    dist=np.linalg.norm(proj-np.asarray(pose[:2]),axis=1);dist[~usable]=np.inf
    i=int(np.argmin(dist))
    if dist[i]>max_lateral:return None
    cumulative=np.r_[0.,np.cumsum(lens)]
    start=cumulative[i]+t[i]*lens[i];stop=min(cumulative[-1],start+lookahead)
    samples=[]
    count=math.ceil((stop-start)/step)
    if count>500:return None
    for s in np.linspace(start,stop,max(2,count+1)):
        k=min(len(d)-1,int(np.searchsorted(cumulative,s,side='right')-1))
        while k<len(d)-1 and lens[k]<=1e-9:k+=1
        if lens[k]<=1e-9:continue
        f=np.clip((s-cumulative[k])/lens[k],0,1);p=a[k]+f*d[k]
        samples.append((float(p[0]),float(p[1]),math.atan2(d[k,1],d[k,0]),float(s-start)))
    return samples


def path_obstacle_witness(points,pose,geom,prior_free,cost,hits,cfg,settings=None):
    """Find a LiDAR-supported *new lethal* grid square in a sampled rectangle.

    Rectangle is the project's validated J100 .458 x .416 m plus existing margin.
    Segment tangents approximate the route corridor, not all RPP trajectories or
    rotation sweeps. No match means 'no witness', NEVER a safety authorization.
    """
    s=settings or BlockageSettings();s.validate()
    cost=np.asarray(cost);known=np.asarray(prior_free,dtype=bool);hits=np.asarray(hits,dtype=float)
    if cost.shape!=known.shape or cost.shape!=(geom.height,geom.width):return None
    if hits.ndim!=2 or hits.shape[1]!=2 or not len(hits) or not np.isfinite(hits).all():return None
    if not np.isfinite(pose).all():return None
    samples=_remaining_samples(points,pose,min(s.sample_step_m,geom.resolution/2),
                               s.lookahead_m,s.max_lateral_m)
    if samples is None:return None
    scale=cfg.robot_radius_m/math.hypot(.229,.208)
    half_x=(.229*scale+cfg.safety_margin_m)/geom.resolution
    half_y=(.208*scale+cfg.safety_margin_m)/geom.resolution
    for x,y,yaw,ahead in samples:
        gx,gy=geom.world_to_grid_float(x,y);theta=yaw-geom.origin[2]
        c=math.cos(theta);sn=math.sin(theta)
        ex=abs(c)*half_x+abs(sn)*half_y;ey=abs(sn)*half_x+abs(c)*half_y
        i0=max(0,math.floor(gx-ex));i1=min(geom.width-1,math.floor(gx+ex))
        j0=max(0,math.floor(gy-ey));j1=min(geom.height-1,math.floor(gy+ey))
        if i0>i1 or j0>j1:continue
        jj,ii=np.nonzero((cost[j0:j1+1,i0:i1+1]==100)&known[j0:j1+1,i0:i1+1])
        for j,i in zip(jj+j0,ii+i0):
            dx=(i+.5)-gx;dy=(j+.5)-gy
            if abs(c*dx+sn*dy)>half_x+.5*(abs(c)+abs(sn)):continue
            if abs(-sn*dx+c*dy)>half_y+.5*(abs(c)+abs(sn)):continue
            wx,wy=geom.cell_to_world(int(i),int(j))
            if np.min(np.hypot(hits[:,0]-wx,hits[:,1]-wy))>s.hit_tolerance_m:continue
            return dict(cell=[int(i),int(j)],xy=[float(wx),float(wy)],
                route_pose=[x,y,yaw],distance_ahead_m=ahead,cost=100,
                prior_free=True,scan_supported=True,
                scope='sampled remaining path corridor, not all routes')
    return None


class PersistentBlockage:
    """Translation stall AND repeated independent costmap+scan observations."""
    def __init__(self,settings=None):
        self.s=settings or BlockageSettings();self.s.validate();self.reset()
    def reset(self):
        self.history=deque(maxlen=1000);self.first=None;self.last=None;self.samples=0
        self.anchor=None;self.last_pair=None;self.triggered=False
    def reset_evidence(self):
        self.first=None;self.last=None;self.samples=0;self.anchor=None;self.last_pair=None
    def update(self,now,pose,witness,cost_stamp,scan_stamp):
        if not all(math.isfinite(v) for v in (now,cost_stamp,scan_stamp,*pose)):
            self.reset();return None
        if self.history and now<self.history[-1][0]:self.reset()
        if self.history and now-self.history[-1][0]>3.:self.reset()
        self.history.append((now,tuple(pose[:2])))
        while len(self.history)>1 and self.history[1][0]<=now-self.s.stationary_window_s:
            self.history.popleft()
        if witness is None:
            self.reset_evidence();return None
        if self.anchor is not None and math.dist(self.anchor,witness['xy'])>.30:
            self.reset_evidence()
        if self.first is None:
            self.first=now;self.anchor=list(witness['xy'])
        pair=(cost_stamp,scan_stamp)
        if self.last_pair is None or (cost_stamp>self.last_pair[0] and scan_stamp>self.last_pair[1]):
            self.samples+=1;self.last_pair=pair;self.last=now
        # Do not let timers alone convert one cached message into repeated evidence.
        if self.triggered or self.samples<self.s.minimum_samples or self.last-self.first<self.s.persistence_s:
            return None
        if now-self.history[0][0]<self.s.stationary_window_s:return None
        anchor=self.history[0][1]
        max_move=max(math.dist(anchor,p) for _,p in self.history)
        if max_move>self.s.progress_m:return None
        self.triggered=True
        return dict(witness,independent_samples=self.samples,persistence_ros_s=self.last-self.first,
                    stalled_window_ros_s=now-self.history[0][0],max_translation_m=max_move,
                    evidence_ros_time=now,global_stamp=cost_stamp,scan_stamp=scan_stamp,
                    persistent=True,translation_stalled=True)


def current_map_cut(geom,prior_free,cost,pose,labels,mask_id,*,fixed_prior_topology=False):
    """Optimistic component diagnosis, NOT a new path planner or motion gate.

    Unknown cost cells and diagonal corner connections are allowed. Therefore a
    negative result is conservative with respect to a point path IN the approved
    prior-free domain. It makes no claims about a new opening through old walls.
    """
    out=dict(fixed_prior_topology=bool(fixed_prior_topology),zone_cut=False,
             point_connectivity='optimistic_8_connected_unknown_cost_allowed',
             no_new_openings_assumption=bool(fixed_prior_topology),
             scope='all approved prior-free target cells; current grid only')
    if not fixed_prior_topology:return dict(out,reason='TOPOLOGY_ASSUMPTION_NOT_ENABLED')
    known=np.asarray(prior_free,dtype=bool);v=np.asarray(cost)
    if v.shape!=known.shape or v.shape!=np.asarray(labels).shape:return dict(out,reason='GEOMETRY_MISMATCH')
    i,j=map(math.floor,geom.world_to_grid_float(*pose[:2]))
    if not geom.contains(i,j) or not known[j,i] or v[j,i]<0 or v[j,i]>=99:
        return dict(out,reason='START_NOT_CONFIRMED_PRIOR_FREE')
    target=(labels==mask_id)&known&(v!=100)
    if not target.any():return dict(out,reason='NO_NONOCCUPIED_TARGET_CELLS')
    struct=np.ones((3,3),dtype=np.uint8)
    before,_=ndimage.label(known,structure=struct)
    if np.any(target & (before!=before[j,i])):return dict(out,reason='PRIOR_CONNECTIVITY_INCOMPLETE')
    opened=known&(v!=100)
    after,_=ndimage.label(opened,structure=struct)
    reachable=int(np.count_nonzero(target&(after==after[j,i])))
    count=int(np.count_nonzero(target))
    return dict(out,zone_cut=reachable==0,reachable_target_cells=reachable,
                disconnected_target_cells=count-reachable,target_cells=count,
                cost_key=hashlib.sha256(np.ascontiguousarray(v==100).tobytes()).hexdigest(),
                reason='OCCUPIED_CUT' if reachable==0 else 'TARGET_HAS_REACHABLE_CELLS')


def validate_cancel_record(record):
    if not isinstance(record,dict):return False
    e=record.get('evidence',{})
    gid=record.get('goal_id')
    return (record.get('reason')==CANCEL_REASON and record.get('terminal')=='CANCELED'
        and record.get('stop_confirmed') is True and record.get('new_coverage') is True
        and isinstance(gid,str) and len(gid)==32 and all(x in '0123456789abcdef' for x in gid)
        and isinstance(e,dict) and e.get('persistent') is True
        and e.get('translation_stalled') is True and e.get('scan_supported') is True
        and e.get('cost')==100 and e.get('prior_free') is True
        and type(e.get('independent_samples')) is int and e['independent_samples']>=3)


def blockage_enabled(candidate, *, inside_target=None):
    """Gate the bespoke watcher; Nav2, freshness and mission guard stay active.

    ENTRY always means transit/entry. A secondary goal remains enabled.
    A primary OBSERVE outside its target is transit until the padded body enters.
    Unknown purpose/geometry does not synthesize an ENTRY classification.
    """
    if not isinstance(candidate,dict):return False
    if candidate.get('second_pass'):return True
    if candidate.get('goal_kind')=='ENTRY':return True
    return candidate.get('goal_kind')=='OBSERVE' and inside_target is False


def enabled_for_node(node, bundle, cfg, inp=None):
    candidate=getattr(node,'current',None)
    if blockage_enabled(candidate):return True
    if not candidate or inp is None or candidate.get('goal_kind')!='OBSERVE':return False
    zone=getattr(node,'zones_by_id',{}).get(candidate.get('zone_id'))
    if zone is None:return False
    from zone_planner import is_inside
    return blockage_enabled(candidate,inside_target=bool(is_inside(zone,inp['pose'],bundle,cfg)))


def enabled_for_candidate(candidate, node, bundle, cfg, inp):
    if blockage_enabled(candidate):return True
    if not candidate or inp is None or candidate.get('goal_kind')!='OBSERVE':return False
    zone=node.zones_by_id.get(candidate.get('zone_id'))
    if zone is None:return False
    from zone_planner import is_inside
    return blockage_enabled(candidate,inside_target=bool(is_inside(zone,inp['pose'],bundle,cfg)))
