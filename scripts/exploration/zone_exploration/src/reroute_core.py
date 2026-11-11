"""DN012 v1.4: ROS-free reroute measurements and static POINT memory.

Not an obstacle-volume model or a physical passage-cut certificate.
All lengths are compared from the SAME current pose to the SAME goal.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import tempfile
import fcntl

VERSION = 'DN012_REROUTE_POINT_MEMORY_V1_4'
SCHEMA = 'dn012_reroute_points_v1'


def finite(x):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError('NONFINITE_NUMBER')
    return x


def xy(p):
    if len(p) < 2:
        raise ValueError('BAD_POINT')
    return (finite(p[0]), finite(p[1]))


def polyline(points):
    if not 2 <= len(points) <= 20000:
        raise ValueError('INVALID_PATH_SIZE')
    out = [xy(p) for p in points]
    if length(out) <= 1e-8:
        raise ValueError('DEGENERATE_PATH')
    return out


def length(points):
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('SYMLINK_REFUSED')
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass(frozen=True)
class RerouteSettings:
    # PILOT VALUES: not empirically calibrated thresholds.
    min_extra_m: float = 3.0
    min_fraction: float = 0.50
    min_remaining_m: float = 1.0
    max_projection_m: float = 0.60
    new_start_tolerance_m: float = 0.50
    goal_tolerance_m: float = 0.30
    forward_offset_m: float = 0.70
    sensor_lookahead_m: float = 3.0
    confirmation_s: float = 0.50
    confirmation_samples: int = 2
    pending_timeout_s: float = 8.0
    # A fixed rasterization footprint around a POINT, NOT obstacle size inference.
    point_radius_m: float = 0.10
    robot_envelope_m: float = math.hypot(.229, .208)
    safety_margin_m: float = .04
    pose_guard_extra_m: float = .15
    max_records: int = 256

    def validate(self):
        limits = {
            'min_extra_m':(.1,100.), 'min_fraction':(.01,10.),
            'min_remaining_m':(.1,20.), 'max_projection_m':(.05,1.),
            'new_start_tolerance_m':(.05,1.), 'goal_tolerance_m':(.05,1.),
            'forward_offset_m':(.1,2.), 'sensor_lookahead_m':(.2,6.),
            'confirmation_s':(.1,5.), 'pending_timeout_s':(1.,30.),
            'point_radius_m':(.025,.30), 'robot_envelope_m':(.20,.60),
            'safety_margin_m':(.01,.20), 'pose_guard_extra_m':(.05,.5)}
        for name, (lo, hi) in limits.items():
            if not lo <= finite(getattr(self, name)) <= hi:
                raise ValueError('BAD_SETTING:'+name)
        if type(self.confirmation_samples) is not int or not 2 <= self.confirmation_samples <= 5:
            raise ValueError('BAD_CONFIRMATION_SAMPLES')
        if type(self.max_records) is not int or not 1 <= self.max_records <= 1024:
            raise ValueError('BAD_MAX_RECORDS')
        if self.pending_timeout_s <= self.confirmation_s:
            raise ValueError('BAD_PENDING_TIMEOUT')
        return self

    @classmethod
    def from_file(cls, path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(**d['detector']).validate()

    def mask_radius(self, resolution):
        return self.point_radius_m+resolution/math.sqrt(2.)


def project_remaining(points, pose, *, tolerance=.60, min_arc=0.):
    """Project onto a polyline, refusing off-route or ambiguous loop projections.

    min_arc is a monotonically tracked progress lower bound. The returned path
    includes the current-pose -> projection segment for length normalization.
    """
    p = polyline(points); here = xy(pose)
    candidates = []; arc = 0.
    for k, (a, b) in enumerate(zip(p[:-1], p[1:])):
        dx, dy = b[0]-a[0], b[1]-a[1]
        seg = math.hypot(dx, dy)
        if seg <= 1e-10:
            continue
        t = max(0., min(1., ((here[0]-a[0])*dx+(here[1]-a[1])*dy)/(seg*seg)))
        q = (a[0]+t*dx, a[1]+t*dy); where = arc+t*seg
        if where+0.10 >= min_arc:
            candidates.append((math.dist(here, q), where, k, q))
        arc += seg
    if not candidates:
        raise ValueError('NO_FORWARD_PROJECTION')
    best = min(candidates, key=lambda q:(q[0], q[1]))
    if best[0] > tolerance:
        raise ValueError('BASELINE_TOO_FAR_FROM_POSE')
    if any(c[0] <= best[0]+.025 and abs(c[1]-best[1]) > 1.0 for c in candidates):
        raise ValueError('AMBIGUOUS_LOOP_PROJECTION')
    route = [here, best[3], *p[best[2]+1:]]
    return route, best[1], best[0]


def point_at_distance(points, distance):
    p = polyline(points); rest = max(0., finite(distance))
    for a, b in zip(p[:-1], p[1:]):
        d = math.dist(a, b)
        if d <= 1e-10:
            continue
        if rest <= d:
            t = rest/d
            return [a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])]
        rest -= d
    return list(p[-1])


def point_path_distance(points, point):
    p = [xy(a) for a in points]; q = xy(point)
    if not p:
        raise ValueError('EMPTY_PATH')
    best = math.dist(q, p[0])
    for a, b in zip(p[:-1], p[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]; den = dx*dx+dy*dy
        t = 0. if den == 0. else max(0., min(1., ((q[0]-a[0])*dx+(q[1]-a[1])*dy)/den))
        best = min(best, math.hypot(q[0]-a[0]-t*dx, q[1]-a[1]-t*dy))
    return best


def directed_deviation(source,target,max_samples=100):
    """Bounded samples against ALL target segments; vectorized per sample.
    Used only for repeated-plan consistency, never to shorten measured paths.
    """
    import numpy as np
    p=np.asarray(target,dtype=float)
    a=p[:-1];v=np.diff(p,axis=0);den=np.sum(v*v,axis=1)
    if len(a)==0:raise ValueError('EMPTY_DEVIATION_PATH')
    samples=list(source[::max(1,len(source)//max_samples)])
    if samples[-1]!=source[-1]:samples.append(source[-1])
    worst=0.
    for q in samples:
        q=np.asarray(q,dtype=float);t=np.zeros(len(a),dtype=float)
        np.divide(np.sum((q-a)*v,axis=1),den,out=t,where=den>1e-12)
        t=np.clip(t,0.,1.);res=q-(a+t[:,None]*v)
        worst=max(worst,math.sqrt(float(np.min(np.sum(res*res,axis=1)))))
    return worst


def compare_paths(old, new, pose, goal, settings, *, min_arc=0.):
    settings.validate(); new = polyline(new)
    old_remaining, arc, lateral = project_remaining(old, pose,
        tolerance=settings.max_projection_m, min_arc=min_arc)
    if math.dist(xy(pose), new[0]) > settings.new_start_tolerance_m:
        raise ValueError('NEW_PATH_START_MISMATCH')
    if (math.dist(new[-1], xy(goal)) > settings.goal_tolerance_m or
            math.dist(old_remaining[-1], xy(goal)) > settings.goal_tolerance_m):
        raise ValueError('NOT_SAME_GOAL_ENDPOINT')
    old_len = length(old_remaining)
    new_route = [xy(pose), *new]
    new_len = length(new_route)
    if old_len < settings.min_remaining_m:
        raise ValueError('NEAR_GOAL_RATIO_DISABLED')
    delta = new_len-old_len; fraction = delta/old_len
    return dict(old_remaining_m=old_len, new_remaining_m=new_len,
        extra_m=delta, increase_fraction=fraction, increase_percent=100*fraction,
        triggered=bool(delta+1e-9 >= settings.min_extra_m and fraction+1e-9 >= settings.min_fraction),
        baseline_arc_m=arc, projection_distance_m=lateral,
        old_remaining=[list(p) for p in old_remaining], new_remaining=[list(p) for p in new_route])


class RerouteDetector:
    """Per-owned-Goal baseline. A new Goal, pose jump or mask revision resets it.

    First actual Nav2 path is the baseline (not the earlier planning probe).
    The initial baseline is kept through normal replans until a confirmed large
    reroute, then rebased. This catches cumulative small changes without
    repeatedly emitting the same large detour. 'confirmed' means stable plan
    publications, NOT proof that the controller executed the path.
    """
    def __init__(self, settings=None):
        self.s = (settings or RerouteSettings()).validate()
        self.reset()

    def reset(self, goal_id=None, goal=None, revision=None):
        self.goal_id=goal_id; self.goal=None if goal is None else xy(goal)
        self.revision=revision; self.base=None; self.previous=None
        self.progress=0.; self.pending=None; self.last_stamp=None
        self.last_pose=None; self.sequence=0; self.last_metric=None

    def feed(self, points, pose, stamp, goal_id, goal, *, revision=None, epoch=0):
        stamp=finite(stamp); p=polyline(points); pose=xy(pose); goal=xy(goal)
        if not goal_id:
            return None, 'NO_OWNED_GOAL'
        if self.goal_id != goal_id or self.goal != goal or getattr(self,'epoch',None)!=epoch:
            self.reset(goal_id,goal,revision); self.epoch=epoch
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return None,'DUPLICATE_OR_OLD_STAMP'
        reset_reason=None
        if revision != self.revision:
            reset_reason='MEMORY_REVISION_REBASE'
        if self.last_stamp is not None:
            dt=stamp-self.last_stamp
            if dt > 5.:
                reset_reason='PATH_GAP_REBASE'
            elif math.dist(pose,self.last_pose) > .50+1.5*dt:
                reset_reason='POSE_JUMP_REBASE'
        self.last_stamp=stamp; self.last_pose=pose; self.revision=revision
        if self.base is None or reset_reason:
            self.base=p;self.previous=p;self.progress=0.;self.pending=None
            return None,reset_reason or 'BASELINE_CAPTURED'
        try:
            m=compare_paths(self.base,p,pose,goal,self.s,min_arc=self.progress)
        except ValueError as exc:
            self.base=p;self.previous=p;self.progress=0.;self.pending=None
            return None,str(exc)+'_REBASE'
        self.progress=max(self.progress,m['baseline_arc_m'])
        m['previous_path_extra_m']=None
        try:
            pm=compare_paths(self.previous,p,pose,goal,self.s)
            m['previous_path_extra_m']=pm['extra_m']
        except ValueError:
            pass
        self.last_metric={k:v for k,v in m.items() if k not in ('old_remaining','new_remaining')}
        self.previous=p
        if not m['triggered']:
            self.pending=None
            return None,'BELOW_BOTH_THRESHOLDS'
        if self.pending is None or stamp-self.pending['stamp'] > self.s.pending_timeout_s:
            self.pending=dict(stamp=stamp,pose=list(pose),metric=m,samples=1,path=p)
            return self._event(self.pending,stamp,False,m), 'REROUTE_PENDING'
        # A different second route must not confirm the first route accidentally.
        old_new=self.pending['path']
        shape=directed_deviation(old_new,p)
        if shape > 1.0:
            self.pending=dict(stamp=stamp,pose=list(pose),metric=m,samples=1,path=p)
            return self._event(self.pending,stamp,False,m), 'REROUTE_PENDING_REPLACED'
        self.pending['samples']+=1
        if (self.pending['samples'] < self.s.confirmation_samples or
                stamp-self.pending['stamp'] < self.s.confirmation_s):
            return None,'REROUTE_PENDING'
        start=self.pending;self.sequence+=1
        event=self._event(start,stamp,True,m)
        self.base=p;self.progress=0.;self.pending=None
        return event,'REROUTE_CONFIRMED'


    def _event(self,start,stamp,confirmed,metric):
        return dict(goal_id=str(self.goal_id),goal_xy=list(self.goal),anchor_xy=start['pose'],
            anchor_ros_s=start['stamp'],confirmed_ros_s=stamp if confirmed else None,
            confirmation_samples=start['samples'],sequence=self.sequence,
            plan_change_confirmed=bool(confirmed),metric=start['metric'],confirm_metric=metric,
            physical_blockage_proven=False,path_execution_confirmed=False,
            semantics='large_detour_operational_candidate_NOT_passage_cut_proof')


def geometry_dict(g):
    return dict(width=int(g.width),height=int(g.height),resolution=finite(g.resolution),
        origin=[finite(v) for v in g.origin],frame_id='map')


def context_for(project, scene_id, namespace='/j100_0001'):
    p=Path(project).resolve()
    if not scene_id or len(scene_id)>120:
        raise ValueError('SCENE_ID_REQUIRED')
    paths=('config/exploration/navigation/hospital_zones.json',
           'config/exploration/navigation/hospital_coverage_reference.json.gz',
           'gazebo/worlds/disaster/disaster_hospital.sdf')
    hashes={}
    for rel in paths:
        q=p/rel
        if not q.is_file():raise ValueError('CONTEXT_FILE_MISSING:'+rel)
        hashes[rel]=hashlib.sha256(q.read_bytes()).hexdigest()
    return dict(scene_id=scene_id,namespace=namespace,ros_domain_id=os.environ.get('ROS_DOMAIN_ID','0'),
        inputs_sha256=hashes,assumption='obstacles_static_in_this_scene_revision')


class PointStore:
    def __init__(self,path,context,geometry,max_records=256):
        self.path=Path(path);self.context=context;self.geometry=geometry
        self.max_records=max_records
        self.context_hash=json_hash(dict(context=context,geometry=geometry))
        self.data=self.empty()
        self.load()

    def empty(self):
        return dict(schema=SCHEMA,version=VERSION,context_hash=self.context_hash,
            context=self.context,geometry=self.geometry,records=[])

    def contains(self,p):
        x,y=xy(p);o=self.geometry['origin'];a=o[2] if len(o)>2 else 0.
        dx=x-o[0];dy=y-o[1];r=self.geometry['resolution']
        i=(math.cos(a)*dx+math.sin(a)*dy)/r
        j=(-math.sin(a)*dx+math.cos(a)*dy)/r
        return 0<=i<self.geometry['width'] and 0<=j<self.geometry['height']

    def load(self):
        if not self.path.exists():return self.data
        if self.path.is_symlink() or self.path.parent.is_symlink():raise ValueError('SYMLINK_REFUSED')
        if self.path.stat().st_size>50*1024*1024:raise ValueError('MEMORY_TOO_LARGE')
        d=json.loads(self.path.read_text(encoding='utf-8'))
        return self._accept(d)

    def _accept(self,d):
        if d.get('schema')!=SCHEMA or d.get('context_hash')!=self.context_hash:
            raise ValueError('STATIC_SCENE_CONTEXT_MISMATCH')
        records=d.get('records')
        if not isinstance(records,list) or len(records)>self.max_records:raise ValueError('MEMORY_CAPACITY')
        ids=set()
        for r in records:
            if r['id'] in ids or not self.contains(r['point_xy']):raise ValueError('INVALID_MEMORY_POINT')
            if r.get('physical_blockage_proven') is not False:raise ValueError('INVALID_PROOF_FLAG')
            if type(r.get('mask_eligible')) is not bool or type(r.get('enabled')) is not bool:
                raise ValueError('INVALID_MEMORY_FLAGS')
            if r.get('mask_eligible') and r.get('plan_change_confirmed') is not True:
                raise ValueError('UNCONFIRMED_MEMORY_CANNOT_MASK')
            if r.get('correction_source') not in ('LIDAR_COSTMAP_OLD_PATH','OLD_PATH_FORWARD_OFFSET_ESTIMATE'):
                raise ValueError('INVALID_CORRECTION_SOURCE')
            if not .025<=finite(r.get('mask_radius_m'))<=.60:raise ValueError('INVALID_POINT_RADIUS')
            ids.add(r['id'])
        json.dumps(d,allow_nan=False)
        self.data=d
        return d

    def restore_records(self,records):
        """Seed a new session from an explicit checkpoint, never shared history."""
        import copy
        with self.locked():
            if self.data['records']:raise ValueError('REROUTE_RESTORE_REQUIRES_EMPTY_SESSION')
            data=self.empty();data['records']=copy.deepcopy(records)
            self._accept(data)
            atomic_json(self.path,self.data)

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        lock=self.path.with_suffix('.lock')
        if lock.is_symlink():raise ValueError('SYMLINK_REFUSED')
        with lock.open('a') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX)
            try:
                self.load();yield
            finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

    def append(self,event,point,*,settings,source,eligible,zone=None,spatial_zone=None,
               sensor_evidence=None,hold_reason=None):
        point=list(xy(point))
        if not self.contains(point):raise ValueError('POINT_OUTSIDE_MAP')
        ident=json_hash([self.context_hash,event['goal_id'],event['anchor_ros_s']])[:20]
        with self.locked():
            existing=next((r for r in self.data['records'] if r['id']==ident),None)
            if existing and (existing.get('plan_change_confirmed') or not event.get('plan_change_confirmed')):
                return ident,False
            if existing is None and len(self.data['records'])>=self.max_records:raise ValueError('MEMORY_CAPACITY')
            r=dict(id=ident,point_xy=point,anchor_xy=event['anchor_xy'],
                goal_id=event['goal_id'],goal_xy=event['goal_xy'],destination_zone=zone,
                spatial_zone=spatial_zone,mask_hold_reason=hold_reason,
                first_seen_utc=datetime.now(timezone.utc).isoformat(),
                anchor_ros_s=event['anchor_ros_s'],confirmed_ros_s=event['confirmed_ros_s'],
                extra_m=event['metric']['extra_m'],increase_fraction=event['metric']['increase_fraction'],
                old_remaining_m=event['metric']['old_remaining_m'],new_remaining_m=event['metric']['new_remaining_m'],
                correction_source=source,sensor_evidence=sensor_evidence,
                plan_change_confirmed=bool(event.get('plan_change_confirmed')),
                mask_eligible=bool(eligible and event.get('plan_change_confirmed')),enabled=True,physical_blockage_proven=False,
                mask_radius_m=settings.mask_radius(self.geometry['resolution']),
                path_execution_confirmed=False,thresholds=asdict(settings),
                old_path_sha256=json_hash(event['metric']['old_remaining']),
                new_path_sha256=json_hash(event['metric']['new_remaining']))
            from blocked_policy import annotate_reroute
            r=annotate_reroute(r)
            if existing is None:self.data['records'].append(r)
            else:
                r['first_seen_utc']=existing['first_seen_utc']
                existing.update(r)
            atomic_json(self.path,self.data)
        return ident,existing is None


def choose_point(event,settings,*,sensor_xy=None):
    """Use current sensor evidence on old route, otherwise advance along OLD arc."""
    old=event['metric']['old_remaining']
    if sensor_xy is not None:
        q=list(xy(sensor_xy))
        if (math.dist(q,event['anchor_xy'])<=settings.sensor_lookahead_m and
                point_path_distance(old,q)<=settings.robot_envelope_m+settings.safety_margin_m):
            return q,'LIDAR_COSTMAP_OLD_PATH'
    return point_at_distance(old,settings.forward_offset_m),'OLD_PATH_FORWARD_OFFSET_ESTIMATE'


def rasterize_points(geometry,records,pose,settings,previous_active=()):
    """Binary point kernels. -1 outside preserves unknown/free costs.
    Nav2 adds its own footprint-aware InflationLayer AFTER KeepoutFilter.

    Do NOT carve a moving hole in an existing keepout. Newly encountered points
    too close to the current footprint wait; overlap with an already applied
    point is a fault requiring stop/inspection.
    """
    import numpy as np
    w=geometry['width'];h=geometry['height'];r=geometry['resolution'];o=geometry['origin']
    if w<=0 or h<=0 or r<=0 or w*h>10000000:raise ValueError('BAD_MASK_GEOMETRY')
    if abs(o[2] if len(o)>2 else 0.)>1e-8:raise ValueError('ROTATED_MASK_UNSUPPORTED')
    mask=np.full((h,w),-1,dtype=np.int8)
    active=[];active_specs=[];pending=[];overlap=[];prior=set(previous_active)
    here=None if pose is None else xy(pose)
    for q in records:
        if not q.get('enabled') or not q.get('mask_eligible'):continue
        p=xy(q['point_xy']);rad=finite(q['mask_radius_m'])
        if here is None or math.dist(p,here)<=rad+settings.robot_envelope_m+settings.safety_margin_m+settings.pose_guard_extra_m:
            if q['id'] in prior:overlap.append(q['id'])
            else:pending.append(q['id'])
            # Previously active masks remain, never silently forget a barrier.
            if q['id'] not in prior:continue
        i0=max(0,math.floor((p[0]-rad-o[0])/r));i1=min(w-1,math.floor((p[0]+rad-o[0])/r))
        j0=max(0,math.floor((p[1]-rad-o[1])/r));j1=min(h-1,math.floor((p[1]+rad-o[1])/r))
        if i0>i1 or j0>j1:continue
        yy,xx=np.mgrid[j0:j1+1,i0:i1+1]
        inside=(o[0]+(xx+.5)*r-p[0])**2+(o[1]+(yy+.5)*r-p[1])**2<=rad**2
        mask[j0:j1+1,i0:i1+1][inside]=100
        active.append(q['id']);active_specs.append([q['id'],list(p),rad])
    return mask,dict(active_ids=sorted(active),pending_near_robot=sorted(pending),
        active_overlap=sorted(overlap),revision=json_hash(sorted(active_specs,key=lambda v:v[0])),
        occupied_cells=int((mask==100).sum()))
