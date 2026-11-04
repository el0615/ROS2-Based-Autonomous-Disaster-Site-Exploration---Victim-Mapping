#!/usr/bin/env python3
"""Read-only single-room observation-point planning. New project implementation.

Concept references: IPA sensor placement (candidate/visibility evaluation) and
NBV (re-evaluate remaining observation gain). NOT a port of convexSPP, an exact
set-cover/TSP solver, or an implementation of the ETH 3-D RRT planner.
Requires existing hospital_coverage_core.Geometry; never mutates actual coverage.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import math
from typing import Any
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from hospital_coverage_core import Geometry


@dataclass(frozen=True)
class Settings:
    spacing_m: float = 0.60
    outside_room_m: float = 3.0
    robot_radius_m: float = math.hypot(0.229, 0.208)
    safety_margin_m: float = 0.04
    max_candidates: int = 100
    max_range_m: float = 8.0
    min_gain_m2: float = 0.10
    distance_regularizer_m: float = 2.0
    threshold: float = 0.90
    switch_ratio: float = 1.20
    switch_absolute: float = 0.05
    heading_samples: int = 8

    def validate(self) -> None:
        positive = ('spacing_m', 'outside_room_m', 'robot_radius_m', 'max_range_m',
                    'distance_regularizer_m', 'threshold')
        for name in positive:
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.robot_radius_m < math.hypot(.229, .208)-1e-9:
            raise ValueError('robot_radius_m must enclose the validated J100 footprint')
        if not 0 < self.threshold <= 1 or not 1 <= self.max_candidates <= 300:
            raise ValueError('Invalid threshold or candidate limit')
        if not 1 <= self.heading_samples <= 16 or self.switch_ratio < 1:
            raise ValueError('Invalid headings/hysteresis')
        for name in ('safety_margin_m','min_gain_m2','switch_absolute'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name) < 0:
                raise ValueError(f'{name} must be nonnegative and finite')


@dataclass(frozen=True)
class SensorModel:
    angle_min: float
    angle_increment: float
    beams: int
    range_min: float
    range_max: float
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_yaw: float = 0.0

    def validate(self) -> None:
        if not all(math.isfinite(v) for k,v in asdict(self).items()):
            raise ValueError('Nonfinite scan model')
        if not 2 <= self.beams <= 4096 or abs(self.angle_increment) < 1e-9:
            raise ValueError('Unsupported scan beam count/increment')
        if (self.beams-1)*abs(self.angle_increment) > 2*math.pi+.02:
            raise ValueError('Scan field of view exceeds one revolution')
        if self.range_min < 0 or self.range_max <= self.range_min:
            raise ValueError('Invalid scan ranges')
        if math.hypot(self.offset_x,self.offset_y)>2.0:
            raise ValueError('Unexpected base-to-sensor displacement')

    @property
    def full_circle(self) -> bool:
        return self.beams*abs(self.angle_increment) >= 2*math.pi-.02


def array_fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def conservative_safe(free: np.ndarray, geom: Geometry, cfg: Settings) -> tuple[np.ndarray,np.ndarray]:
    """Circular envelope, obstacle-cell extent and edge-interpolation margin.

    Boundary is blocked. One half diagonal for obstacle square extent + one
    half diagonal for centre-to-centre traversal. Conservative, not Nav2 proof.
    """
    clearance = ndimage.distance_transform_edt(np.pad(free,1,constant_values=False))[1:-1,1:-1]*geom.resolution
    required=cfg.robot_radius_m+cfg.safety_margin_m+math.sqrt(2)*geom.resolution
    return free & (clearance >= required), clearance


def oriented_footprint_clear(free: np.ndarray, geom: Geometry, pose: tuple[float,float,float], cfg: Settings) -> bool:
    """Check J100 oriented rectangle + margin against *full blocked grid squares*.

    Unlike conservative_safe(), this has no center-of-grid traversal allowance;
    it is used ONLY for an offline/display-only start connector, not Nav2 goal authorization.
    Unknown and out-of-map squares are obstacles. No collision-costmap inflation override.
    """
    x,y,yaw=pose
    gx,gy=geom.world_to_grid_float(x,y)
    angle=yaw-geom.origin[2]
    c=math.cos(angle);s=math.sin(angle)
    # Keep the originally validated robot radius lower bound in larger custom configurations.
    scale=cfg.robot_radius_m/math.hypot(.229,.208)
    a=(.229*scale+cfg.safety_margin_m)/geom.resolution
    b=(.208*scale+cfg.safety_margin_m)/geom.resolution
    ex=abs(c)*a+abs(s)*b;ey=abs(s)*a+abs(c)*b
    imin=math.floor(gx-ex);imax=math.floor(gx+ex)
    jmin=math.floor(gy-ey);jmax=math.floor(gy+ey)
    if imin<0 or jmin<0 or imax>=geom.width or jmax>=geom.height:
        return False
    for j in range(jmin,jmax+1):
        for i in range(imin,imax+1):
            if free[j,i]:continue
            dx=(i+.5)-gx;dy=(j+.5)-gy
            # Rectangle vs axis-aligned grid square using four separating axes.
            if abs(dx) > ex+.5 or abs(dy)>ey+.5:continue
            if abs(c*dx+s*dy)>a+.5*(abs(c)+abs(s)):continue
            if abs(-s*dx+c*dy)>b+.5*(abs(c)+abs(s)):continue
            return False
    return True


def conservative_start_bridge(free: np.ndarray, safe: np.ndarray,
                              geom: Geometry, robot_pose: tuple[float,float,float],
                              cfg: Settings, limit_m: float=.35) -> tuple[int,int,float] | None:
    """Find a short collision-checked straight connection to the strict safe graph.

    The bridge uses a fixed robot heading and never crosses unknown/occupied cells.
    This is ONLY an estimated visualization connection; it does not authorize Nav2 travel.
    """
    if not oriented_footprint_clear(free,geom,robot_pose,cfg):return None
    x,y,yaw=robot_pose
    gx,gy=geom.world_to_grid_float(x,y)
    r=math.ceil((limit_m+geom.resolution)/geom.resolution)
    ci,cj=math.floor(gx),math.floor(gy)
    nearby=[]
    for j in range(max(0,cj-r),min(geom.height,cj+r+1)):
        for i in range(max(0,ci-r),min(geom.width,ci+r+1)):
            if not safe[j,i]:continue
            ax,ay=geom.cell_to_world(i,j)
            length=math.hypot(ax-x,ay-y)
            if length<=limit_m:nearby.append((length,i,j,ax,ay))
    for length,i,j,ax,ay in sorted(nearby):
        # Sample the *swept, oriented footprint*, not only a point/center ray.
        count=max(1,math.ceil(length/min(.015,geom.resolution/4)))
        if all(oriented_footprint_clear(free,geom,(x+(ax-x)*k/count,y+(ay-y)*k/count,yaw),cfg)
               for k in range(1,count+1)):
            return i,j,float(length)
    return None


def path_graph(safe: np.ndarray, resolution: float):
    """8-connected undirected graph; a diagonal may not cut a blocked corner."""
    h,w=safe.shape
    ids=np.full((h,w),-1,dtype=np.int32)
    flat=np.flatnonzero(safe)
    ids.ravel()[flat]=np.arange(len(flat),dtype=np.int32)
    rows=[];cols=[];weights=[]
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        ya=slice(0,h-dy); yb=slice(dy,h)
        if dx >= 0: xa=slice(0,w-dx); xb=slice(dx,w)
        else: xa=slice(-dx,w); xb=slice(0,w+dx)
        a=ids[ya,xa];b=ids[yb,xb];valid=(a>=0)&(b>=0)
        if dx and dy: valid &= safe[ya,xb] & safe[yb,xa]
        av=a[valid];bv=b[valid];cost=resolution*math.hypot(dx,dy)
        rows.extend((av,bv));cols.extend((bv,av))
        weights.extend((np.full(av.size,cost),np.full(av.size,cost)))
    if not len(flat): return ids,flat,coo_matrix((0,0)).tocsr()
    graph=coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(len(flat),len(flat))).tocsr()
    graph.indices=graph.indices.astype(np.int32); graph.indptr=graph.indptr.astype(np.int32)
    return ids,flat,graph


def visibility_cells(geom: Geometry, free: np.ndarray, target: np.ndarray,
                     x: float, y: float, yaw: float, sensor: SensorModel,
                     range_cap: float) -> np.ndarray:
    """Vectorized exact grid DDA for hypothetical 2-D laser beams.

    This predicts potential visibility using prior/latest costmap geometry.
    All hypothetical rays assumed available; real finite/+inf return behavior
    and deskew are evaluated ONLY by the user's real Coverage node.
    No angular sector filling. Unknown/walls and diagonal corners stop rays.
    """
    sx=x+math.cos(yaw)*sensor.offset_x-math.sin(yaw)*sensor.offset_y
    sy=y+math.sin(yaw)*sensor.offset_x+math.cos(yaw)*sensor.offset_y
    gx,gy=geom.world_to_grid_float(sx,sy)
    ci,cj=math.floor(gx),math.floor(gy)
    if not geom.contains(ci,cj) or not free[cj,ci]:
        return np.empty(0,dtype=np.int32)
    angles=(yaw+sensor.offset_yaw+sensor.angle_min+np.arange(sensor.beams)*sensor.angle_increment-geom.origin[2])
    dx=np.cos(angles);dy=np.sin(angles)
    px=gx+sensor.range_min/geom.resolution*dx
    py=gy+sensor.range_min/geom.resolution*dy
    ix=np.floor(px).astype(np.int32);iy=np.floor(py).astype(np.int32)
    stepx=np.where(dx>0,1,-1);stepy=np.where(dy>0,1,-1)
    txdelta=np.full(sensor.beams,np.inf);tydelta=np.full(sensor.beams,np.inf)
    np.divide(1.,np.abs(dx),out=txdelta,where=np.abs(dx)>1e-12)
    np.divide(1.,np.abs(dy),out=tydelta,where=np.abs(dy)>1e-12)
    tx=np.full(sensor.beams,np.inf);ty=np.full(sensor.beams,np.inf)
    fx=np.where(dx>0,ix+1-px,px-ix);fy=np.where(dy>0,iy+1-py,py-iy)
    np.multiply(fx,txdelta,out=tx,where=np.isfinite(txdelta))
    np.multiply(fy,tydelta,out=ty,where=np.isfinite(tydelta))
    limit=(min(range_cap,sensor.range_max)-sensor.range_min)/geom.resolution
    if limit<=0: return np.empty(0,dtype=np.int32)
    travelled=np.zeros(sensor.beams);active=np.ones(sensor.beams,dtype=bool)
    seen=np.zeros(free.size,dtype=bool)
    def isfree(a,b):
        valid=(a>=0)&(b>=0)&(a<geom.width)&(b<geom.height)
        ans=np.zeros(a.shape,dtype=bool)
        ans[valid]=free[b[valid],a[valid]]
        return ans
    for _ in range(geom.width+geom.height+4):
        k=np.flatnonzero(active)
        if not k.size: break
        valid=isfree(ix[k],iy[k]);active[k[~valid]]=False;k=k[valid]
        if not k.size: break
        end=np.minimum(np.minimum(tx[k],ty[k]),limit)
        mark=end>travelled[k]+1e-10
        cells=iy[k[mark]]*geom.width+ix[k[mark]]
        seen[cells]=True
        complete=end>=limit-1e-10
        active[k[complete]]=False;k=k[~complete]
        if not k.size: break
        with np.errstate(invalid='ignore'):
            diagonal=np.abs(tx[k]-ty[k])<1e-10
        dk=k[diagonal]
        if dk.size:
            ok=isfree(ix[dk]+stepx[dk],iy[dk]) & isfree(ix[dk],iy[dk]+stepy[dk])
            active[dk[~ok]]=False;dk=dk[ok]
            travelled[dk]=tx[dk]; ix[dk]+=stepx[dk];iy[dk]+=stepy[dk]
            tx[dk]+=txdelta[dk];ty[dk]+=tydelta[dk]
        nk=k[~diagonal];xx=nk[tx[nk]<ty[nk]];yy=nk[tx[nk]>=ty[nk]]
        travelled[xx]=tx[xx];ix[xx]+=stepx[xx];tx[xx]+=txdelta[xx]
        travelled[yy]=ty[yy];iy[yy]+=stepy[yy];ty[yy]+=tydelta[yy]
    seen &= target.ravel()
    return np.flatnonzero(seen).astype(np.int32)


def validate_coverage(status: dict, grid: np.ndarray, labels: np.ndarray,
                      signature: str, zones: list[dict]) -> None:
    if status.get('schema')!='hospital_room_coverage_v1':
        raise ValueError('Unsupported coverage schema')
    if status.get('zone_bundle_signature')!=signature:
        raise ValueError('Coverage/zone bundle signature mismatch')
    if not status.get('session_id') or not status.get('map_validated') or status.get('fault'):
        raise ValueError('Coverage session/map is invalid')
    if grid.shape != labels.shape or not np.all(np.isin(grid,[-1,0,100])):
        raise ValueError('Invalid coverage grid shape/encoding')
    if np.any(grid[labels==0]!=-1) or np.any(grid[labels>0]<0):
        raise ValueError('Coverage mask differs from approved zone mask')
    observed=grid==100;counts=np.bincount(labels[observed],minlength=int(labels.max())+1)
    if int(observed.sum())!=int(status['observed_cells']):
        raise ValueError('Status/grid snapshots are not synchronized (total)')
    byid={z['id']:z for z in status['zones']}
    if len(byid)!=len(zones): raise ValueError('Coverage zone count differs')
    for zone in zones:
        s=byid.get(zone['id']);rid=zone['mask_id']
        if s is None or int(s['reference_free_cells'])!=zone['known_free_cells']:
            raise ValueError('Coverage denominator mismatch')
        if int(s['observed_cells'])!=int(counts[rid]):
            raise ValueError('Status/grid snapshots are not synchronized (zone)')


def current_input_error(status: dict, now_s: float, status_wall_age: float,
                        max_age_s: float=8.0, wall_timeout: float=15.0) -> str | None:
    if status.get('data_health')!='LIVE': return 'WAIT_COVERAGE_LIVE'
    try:
        evidence=float(status['last_evidence_scan_end_s']);stamp=float(status['ros_time_s'])
    except (TypeError,ValueError,KeyError): return 'WAIT_COVERAGE_EVIDENCE'
    if not all(math.isfinite(v) for v in [now_s,evidence,stamp]): return 'INVALID_COVERAGE_TIME'
    if now_s-stamp < -0.1 or now_s-evidence < -0.1: return 'CLOCK_MISMATCH'
    if now_s-evidence>max_age_s or now_s-stamp>max_age_s or status_wall_age>wall_timeout:
        return 'STALE_COVERAGE'
    if int(status.get('evidence_scans',0))<1: return 'WAIT_COVERAGE_EVIDENCE'
    return None


class ThresholdTracker:
    """Actual measured count threshold, not predicted gain or publisher's 75% star."""
    def __init__(self, threshold: float):
        self.threshold=threshold;self.session=None;self.first=None;self.last_scan=None;self.confirmed=False
    def update(self,status: dict,fraction: float) -> str:
        if status['session_id']!=self.session:
            self.session=status['session_id'];self.first=None;self.last_scan=None;self.confirmed=False
        if fraction<self.threshold:
            self.first=None;self.confirmed=False;self.last_scan=None;return 'NEEDS_OBSERVATION'
        if self.confirmed:return 'OBSERVATION_SUFFICIENT'
        t=float(status['last_evidence_scan_end_s']);seq=int(status['evidence_scans'])
        if self.first is None:self.first=(t,seq)
        elif seq>self.first[1] and t-self.first[0]>=1.0:self.confirmed=True
        self.last_scan=seq
        return 'OBSERVATION_SUFFICIENT' if self.confirmed else 'THRESHOLD_PENDING'


def generate_candidates(safe: np.ndarray,clearance: np.ndarray,target: np.ndarray,
                        labels: np.ndarray,geom: Geometry,cfg: Settings) -> list[tuple[int,int]]:
    within=ndimage.distance_transform_edt(~target)*geom.resolution<=cfg.outside_room_m
    eligible=safe & within & (labels>0)
    step=max(1,round(cfg.spacing_m/geom.resolution))
    ys,xs=np.nonzero(eligible)
    if not len(xs): return []
    # One maximum-clearance point per spatial tile, not one room centroid.
    tile=(ys//step)*math.ceil(geom.width/step)+xs//step
    order=np.lexsort((xs,ys,-clearance[ys,xs],tile))
    st=tile[order];take=order[np.r_[True,st[1:]!=st[:-1]]]
    points=list(zip(xs[take].tolist(),ys[take].tolist()))
    if len(points)>cfg.max_candidates:
        # Uniform row-major subsampling avoids selecting only near-robot tiles.
        select=np.linspace(0,len(points)-1,cfg.max_candidates,dtype=int)
        points=[points[k] for k in select]
    return points


def build_plan(geom: Geometry, labels: np.ndarray, known_free: np.ndarray,
               zone: dict, observed: np.ndarray, robot_pose: tuple[float,float,float],
               sensor: SensorModel, cfg: Settings, costmap: np.ndarray | None=None,
               *, allow_navfn_fallback: bool=False,
               force_navfn_fallback: bool=False) -> dict:
    """Expensive snapshot computation; no ROS calls and no input mutations."""
    cfg.validate();sensor.validate()
    if not all(math.isfinite(v) for v in robot_pose):raise ValueError('Invalid robot pose')
    target=labels==zone['mask_id']
    if not target.any(): raise ValueError('Target zone has no approved free cells')
    if observed.shape!=labels.shape or known_free.shape!=labels.shape: raise ValueError('Input shape mismatch')
    if np.any(observed & (labels==0)): raise ValueError('Observed cells outside approved mask')
    free=known_free.copy()
    mode='PRIOR_ONLY_OFFLINE'
    if costmap is not None:
        if costmap.shape!=free.shape or not np.all((costmap>=-1)&(costmap<=100)):
            raise ValueError('Costmap shape/value mismatch')
        # Costmap 100=lethal; unknown is blocked. Lower inflation costs are not walls.
        free &= (costmap>=0)&(costmap<100)
        mode='PRIOR_PLUS_GLOBAL_COSTMAP'
    safe,clearance=conservative_safe(free,geom,cfg)
    if costmap is not None: safe &= costmap<99
    i,j=map(math.floor,geom.world_to_grid_float(robot_pose[0],robot_pose[1]))
    base=dict(zone_id=zone['id'],zone_mask_id=zone['mask_id'],reference_cells=int(target.sum()),
              candidate_mode=mode,robot_pose=list(robot_pose),candidates=[],selected=None,
              nav2_plan_checked=False,navigation_commands_sent=False,
              status='NO_CANDIDATE',reason='',safe_cells=int(safe.sum()))
    base['start_reference_free']=bool(geom.contains(i,j) and known_free[j,i])
    base['start_zone_mask_id']=int(labels[j,i]) if geom.contains(i,j) else None
    base['start_clearance_m']=float(clearance[j,i]) if geom.contains(i,j) else None
    base['start_conservative_required_m']=cfg.robot_radius_m+cfg.safety_margin_m+math.sqrt(2)*geom.resolution
    bridge_m=0.0
    if not geom.contains(i,j) or not free[j,i]:
        base.update(status='START_CLEARANCE_BLOCKED',reason='Robot is outside accessible current free map; pose/Costmap must be checked')
        return base
    if not safe[j,i]:
        bridge=conservative_start_bridge(free,safe,geom,robot_pose,cfg)
        if bridge is None:
            base.update(status='START_CLEARANCE_BLOCKED',reason='No footprint-checked short bridge to strict conservative free graph; NOT safe to snap')
            return base
        i,j,bridge_m=bridge
        base.update(start_mode='FOOTPRINT_LOCAL_BRIDGE_DISPLAY_ONLY',
                    start_bridge_m=bridge_m,start_bridge_anchor_world=list(geom.cell_to_world(i,j)))
    else:
        base.update(start_mode='STRICT_CONSERVATIVE_GRID',start_bridge_m=0.0,
                    start_bridge_anchor_world=list(geom.cell_to_world(i,j)))
    ids,flat,graph=path_graph(safe,geom.resolution)
    seed_id=int(ids[j,i])
    distance,predecessor=dijkstra(graph,directed=True,indices=seed_id,return_predecessors=True)
    points=generate_candidates(safe,clearance,target,labels,geom,cfg)
    target_y,target_x=np.nonzero(target & ~observed)
    if not len(target_x):
        base.update(status='NO_UNOBSERVED_CELLS');return base
    tx,ty=geom.cell_to_world(float(np.mean(target_x)),float(np.mean(target_y)))
    npoints=len(points);reachable=0
    for i,j in points:
        ident=int(ids[j,i]);dist=float(distance[ident])+bridge_m
        if not math.isfinite(dist):continue
        reachable+=1;x,y=geom.cell_to_world(i,j)
        direction=math.atan2(ty-y,tx-x)-sensor.offset_yaw
        headings=[direction] if sensor.full_circle else [direction+k*2*math.pi/cfg.heading_samples for k in range(cfg.heading_samples)]
        best=None
        for yaw in headings:
            yaw=math.atan2(math.sin(yaw),math.cos(yaw))
            visible=visibility_cells(geom,free,target,x,y,yaw,sensor,cfg.max_range_m)
            new=int(np.count_nonzero(~observed.ravel()[visible]))
            if best is None or new>best[0]: best=(new,yaw,visible)
        count,yaw,visible=best
        if not len(visible):continue
        # Store grid path only for drawing; it is NOT a commanded Nav2 path.
        chain=[];node=ident
        for _ in range(len(flat)+1):
            cell=int(flat[node]);cy,cx=divmod(cell,geom.width);chain.append((cx,cy))
            if node==seed_id:break
            node=int(predecessor[node])
            if node<0: chain=[];break
        if not chain:continue
        chain.reverse()
        if bridge_m>0:
            ri,rj=map(math.floor,geom.world_to_grid_float(robot_pose[0],robot_pose[1]))
            chain.insert(0,(ri,rj))
        base['candidates'].append(dict(id=f'P_{i}_{j}',cell=[i,j],x=x,y=y,yaw=yaw,
                  path_est_m=dist,clearance_lower_bound_m=float(clearance[j,i]-geom.resolution/math.sqrt(2)),
                  viewpoint_zone_mask_id=int(labels[j,i]),visible_indices=visible,path_cells=chain))
    base.update(sampled_candidates=npoints,reachable_sampled_candidates=reachable,
                status='CANDIDATES_READY' if base['candidates'] else 'NO_CANDIDATE',
                reason='' if base['candidates'] else 'No visible reachable sampled pose; not proof of unreachable room')
    ranked=rerank_plan(base,observed,geom,cfg)
    if not allow_navfn_fallback or (ranked['selected'] is not None and not force_navfn_fallback):
        return ranked
    return append_navfn_fallback_views(
        ranked, geom, labels, free, safe, clearance, target, observed,
        robot_pose, sensor, cfg, points, costmap
    )


def append_navfn_fallback_views(plan: dict, geom: Geometry, labels: np.ndarray,
                              free: np.ndarray, safe: np.ndarray,
                              clearance: np.ndarray, target: np.ndarray,
                              observed: np.ndarray, robot_pose: tuple[float,float,float],
                              sensor: SensorModel, cfg: Settings,
                              points: list[tuple[int,int]],
                              costmap: np.ndarray | None=None,
                              max_evaluations: int=24) -> dict:
    """Offer STRICT-SAFE viewpoints disconnected in the strict *route* graph.

    The fallback never lowers the footprint/clearance requirement for GOAL poses.
    A raw-free 4-connected component is only a permissive proposal filter.
    Euclidean travel is a lower-bound RANKING estimate, never a path proof.
    Any proposed candidate must undergo Navfn AND the oriented swept-footprint
    preflight in the sequence node before it can be sent to NavigateToPose.
    This function is pure; it sends no ROS actions and changes no occupancy map.
    """
    if max_evaluations < 1:
        raise ValueError('max_evaluations must be positive')
    start_i,start_j=map(math.floor,geom.world_to_grid_float(*robot_pose[:2]))
    allowed=free.copy()
    if costmap is not None:
        allowed &= (costmap>=0)&(costmap<99)
    if not geom.contains(start_i,start_j) or not allowed[start_j,start_i]:
        return dict(plan,fallback_status='RAW_FREE_START_BLOCKED',fallback_evaluated=0,
                    fallback_proposed=0)
    # Four-way connectivity prevents corner-cutting and is only a *necessary*
    # connectivity screen, NOT a swept body path check.
    cc,_=ndimage.label(allowed,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8))
    start_component=int(cc[start_j,start_i])
    existing={c['id'] for c in plan['candidates']}
    usable=[(i,j) for i,j in points if safe[j,i] and int(cc[j,i])==start_component
            and f'P_{i}_{j}' not in existing]
    if not usable:
        return dict(plan,fallback_status='NO_RAW_CONNECTED_STRICT_SAFE_VIEW',
                    fallback_raw_connected=0,fallback_evaluated=0,fallback_proposed=0)
    # Existing generate_candidates has already spatially tiled the strict-safe
    # poses. Uniform selection retains geographic spread with bounded work.
    count=min(max_evaluations,len(usable))
    selected=[usable[int(k)] for k in np.linspace(0,len(usable)-1,count,dtype=int)]
    ty,tx=np.nonzero(target & ~observed)
    if len(tx)==0:
        return dict(plan,fallback_status='NO_UNOBSERVED_CELLS',
                    fallback_raw_connected=len(usable),fallback_evaluated=0,fallback_proposed=0)
    target_x,target_y=geom.cell_to_world(float(np.mean(tx)),float(np.mean(ty)))
    additions=[]
    for i,j in selected:
        x,y=geom.cell_to_world(i,j)
        direction=math.atan2(target_y-y,target_x-x)-sensor.offset_yaw
        headings=[direction] if sensor.full_circle else [
            direction+k*2*math.pi/cfg.heading_samples for k in range(cfg.heading_samples)]
        best=None
        for yaw in headings:
            yaw=math.atan2(math.sin(yaw),math.cos(yaw))
            if not oriented_footprint_clear(allowed,geom,(x,y,yaw),cfg):
                continue
            visible=visibility_cells(geom,allowed,target,x,y,yaw,sensor,cfg.max_range_m)
            new=int(np.count_nonzero(~observed.ravel()[visible]))
            if best is None or new>best[0]:best=(new,yaw,visible)
        if best is None or best[0]==0:
            continue
        new,yaw,visible=best
        # STRICT-SAFE goal pose, no claimed grid route. Only Navfn can produce
        # an actual route; no fallback grid chain is treated as a valid path.
        additions.append(dict(id=f'P_{i}_{j}',cell=[i,j],x=x,y=y,yaw=yaw,
            path_est_m=float(math.hypot(x-robot_pose[0],y-robot_pose[1])),
            clearance_lower_bound_m=float(clearance[j,i]-geom.resolution/math.sqrt(2)),
            viewpoint_zone_mask_id=int(labels[j,i]),visible_indices=visible,
            path_cells=[],route_mode='NAVFN_FALLBACK_UNVERIFIED'))
    expanded=dict(plan)
    expanded['candidates']=list(plan['candidates'])+additions
    expanded['fallback_status']='PROPOSED_FOR_NAVFN' if additions else 'NO_VISIBLE_FOOTPRINT_CLEAR_VIEW'
    expanded['fallback_raw_connected']=len(usable)
    expanded['fallback_evaluated']=len(selected)
    expanded['fallback_proposed']=len(additions)
    return rerank_plan(expanded,observed,geom,cfg)


def rerank_plan(plan: dict, observed: np.ndarray,geom: Geometry,cfg: Settings,
                previous_id: str | None=None) -> dict:
    result={k:v for k,v in plan.items() if k not in ('candidates','selected')}
    candidates=[]
    for c in plan['candidates']:
        d=c.copy();idx=c['visible_indices'];new=idx[~observed.ravel()[idx]]
        d['new_indices']=new;d['new_cells']=len(new);d['new_area_m2']=len(new)*geom.resolution**2
        d['potential_gain_pp']=100*len(new)/plan['reference_cells']
        d['score']=d['new_area_m2']/(d['path_est_m']+cfg.distance_regularizer_m)
        candidates.append(d)
    candidates.sort(key=lambda c:(-c['score'],-c['new_cells'],c['path_est_m'],c['id']))
    useful=[c for c in candidates if c['new_area_m2']>=cfg.min_gain_m2 and c['new_cells']>0]
    chosen=useful[0] if useful else None
    previous=next((c for c in useful if c['id']==previous_id),None)
    if chosen and previous and chosen['id']!=previous['id']:
        if chosen['score'] < previous['score']*cfg.switch_ratio+cfg.switch_absolute:
            chosen=previous
    result['candidates']=candidates;result['selected']=chosen
    if chosen: result.update(status='SELECTED',reason='Prior/costmap grid feasibility only; NOT a navigation authorization')
    elif candidates: result.update(status='NO_USEFUL_GAIN',reason='Sampled candidates add too little predicted coverage; not room completion')
    return result


def reportable(plan: dict) -> dict:
    """Small JSON: never ship raw numpy arrays as a live command."""
    def candidate(c):
        if c is None:return None
        return {k:v for k,v in c.items() if k not in ('visible_indices','new_indices','path_cells')}
    out={k:v for k,v in plan.items() if k not in ('candidates','selected') and not k.startswith('_')}
    out['candidates']=[candidate(c) for c in plan['candidates']]
    out['selected']=candidate(plan.get('selected'))
    return out
