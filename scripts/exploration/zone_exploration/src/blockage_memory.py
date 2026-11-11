"""DN-012 v1.3: persistent, global *path-obstruction* memory (not cut proof).

Pure Python, no ROS/API/motion.  A LiDAR+costmap+stall witness can be kept
immediately, even if a later stop confirmation fails.  A remembered point is
used to HOLD a NEW route only when a fresh, valid current-map costmap STILL
contains lethal occupancy at that point.  Otherwise memory is historical only.

One point does NOT prove passage width, a cut line, a blocked Zone, nor a
victim.  The original local-start/footprint/Nav2 gates remain authoritative.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SCHEMA = 'dn012_global_blockage_memory_v1'
VERSION = 'DN012_GLOBAL_BLOCKAGE_MEMORY_V1_3'
MAX_RECORDS = 1024


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('BLOCKAGE_MEMORY_BAD_NUMBER') from exc
    if not math.isfinite(v):
        raise ValueError('BLOCKAGE_MEMORY_NONFINITE')
    return v


def _point(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError('BLOCKAGE_MEMORY_BAD_XY')
    return [_finite(value[0]), _finite(value[1])]


def _safe_str(value, size=160):
    return str(value or '')[:size]


def _distance_to_segments(points, xy):
    """Minimum 2-D distance from point to a polyline, endpoints included."""
    if not points:
        raise ValueError('BLOCKAGE_MEMORY_EMPTY_PATH')
    p = [_point(q) for q in points]
    if len(p) == 1:
        return math.dist(p[0], xy)
    best = math.inf
    for a, b in zip(p[:-1], p[1:]):
        vx, vy = b[0]-a[0], b[1]-a[1]
        den = vx*vx + vy*vy
        t = 0.0 if den == 0.0 else max(0.0, min(1.0,
            ((xy[0]-a[0])*vx + (xy[1]-a[1])*vy)/den))
        best = min(best, math.hypot(xy[0]-a[0]-t*vx, xy[1]-a[1]-t*vy))
    return best


class BlockageMemory:
    def __init__(self, path, *, signature, geometry, labels=None,
                 zone_mask_to_name=None, hold_radius_m=.40):
        self.path = Path(path)
        self.signature = str(signature)
        self.geometry = geometry
        self.labels = labels
        self.zone_mask_to_name = zone_mask_to_name or {}
        self.hold_radius_m = _finite(hold_radius_m)
        if not .25 <= self.hold_radius_m <= .75:
            raise ValueError('BLOCKAGE_MEMORY_BAD_RADIUS')
        self.geometry_info = dict(width=int(geometry.width),height=int(geometry.height),
            resolution=_finite(geometry.resolution), origin=[_finite(x) for x in geometry.origin[:2]],
            frame_id='map')
        self.records = []
        self.loaded_count = 0
        self.holds_this_run = 0
        self._load()

    def _load(self):
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError('BLOCKAGE_MEMORY_SYMLINK_REFUSED')
        if not self.path.is_file():
            return
        obj = json.loads(self.path.read_text(encoding='utf-8'))
        if (obj.get('schema') != SCHEMA or obj.get('bundle_signature') != self.signature
                or obj.get('geometry') != self.geometry_info):
            raise ValueError('BLOCKAGE_MEMORY_CONTEXT_MISMATCH_NEW_MAP_REQUIRES_NEW_FILE')
        self._set_records(obj.get('records'))

    def _set_records(self, raw):
        if not isinstance(raw,list) or len(raw)>MAX_RECORDS:
            raise ValueError('BLOCKAGE_MEMORY_CORRUPT_RECORDS')
        found=set()
        for entry in raw:
            if not isinstance(entry,dict) or not entry.get('id') or entry['id'] in found:
                raise ValueError('BLOCKAGE_MEMORY_BAD_RECORD')
            _point(entry.get('xy'))
            if entry.get('observation_status') not in ('PATH_WITNESS_RECORDED','NAV_CANCELED_STOP_CONFIRMED'):
                raise ValueError('BLOCKAGE_MEMORY_BAD_STATUS')
            radius=_finite(entry.get('hold_radius_m'))
            if not .25<=radius<=.75:
                raise ValueError('BLOCKAGE_MEMORY_BAD_STORED_RADIUS')
            if not self._contains(entry['xy']):
                raise ValueError('BLOCKAGE_MEMORY_POINT_OUTSIDE_MAP')
            found.add(entry['id'])
        self.records = raw
        self.loaded_count = len(raw)

    def restore_records(self, records):
        """Seed this run only from the explicitly selected checkpoint."""
        import copy
        if self.records:
            raise ValueError('BLOCKAGE_MEMORY_RESTORE_REQUIRES_EMPTY_SESSION')
        self._set_records(copy.deepcopy(records))
        self._save()

    def _save(self):
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError('BLOCKAGE_MEMORY_SYMLINK_REFUSED')
        if len(self.records)>MAX_RECORDS:
            raise ValueError('BLOCKAGE_MEMORY_RECORD_CAPACITY_EXCEEDED')
        obj=dict(schema=SCHEMA,version=VERSION,bundle_signature=self.signature,
            geometry=self.geometry_info,records=self.records,
            semantics='historical path obstacle points; not verified passage cut or blocked region')
        data=json.dumps(obj,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2).encode('utf-8')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.blockage.',suffix='.tmp',dir=str(self.path.parent))
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.chmod(tmp,0o600)
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _contains(self, xy):
        i,j = self._ij(xy)
        return self.geometry.contains(i,j)

    def _ij(self, xy):
        return tuple(map(math.floor,self.geometry.world_to_grid_float(*xy)))

    def _zone_at(self, xy):
        if self.labels is None:return None
        i,j=self._ij(xy)
        if not self.geometry.contains(i,j):return None
        return self.zone_mask_to_name.get(int(self.labels[j,i]))

    def detect(self, evidence, *, goal_id, ros_time_s=None):
        """Register a detector witness before cancellation; never claim physical cut."""
        if not isinstance(evidence,dict):
            raise ValueError('BLOCKAGE_MEMORY_MISSING_EVIDENCE')
        xy = _point(evidence.get('xy'))
        if not self._contains(xy):
            raise ValueError('BLOCKAGE_MEMORY_POINT_OUTSIDE_MAP')
        goal_id=_safe_str(goal_id,80)
        if not goal_id:raise ValueError('BLOCKAGE_MEMORY_GOAL_ID_REQUIRED')
        for r in self.records:
            if goal_id in r.get('goal_ids',[]) and math.dist(r['xy'],xy)<.1:
                return r['id'],False
        if len(self.records)>=MAX_RECORDS:
            raise ValueError('BLOCKAGE_MEMORY_RECORD_CAPACITY_EXCEEDED')
        stamp=_finite(ros_time_s) if ros_time_s is not None else None
        identifier=hashlib.sha256(json.dumps([self.signature,goal_id,xy,stamp],
            separators=(',',':')).encode()).hexdigest()[:20]
        # The event is a path witness, even if LiDAR support is absent/stale.
        # The live costmap check below is always independent of this evidence.
        r=dict(id=identifier,xy=xy,goal_ids=[goal_id],
            destination_zone=_safe_str(evidence.get('zone'),80),
            spatial_zone=self._zone_at(xy),
            observation_status='PATH_WITNESS_RECORDED',
            initial_ros_time_s=stamp,first_seen_utc=_utc(),
            detected_cost=int(evidence['cost']) if evidence.get('cost') is not None else None,
            cost_context=_safe_str(evidence.get('cost_context'),128),
            scan_supported=bool(evidence.get('scan_supported',False)),
            persistent=bool(evidence.get('persistent',False)),
            translation_stalled=bool(evidence.get('translation_stalled',False)),
            independent_samples=int(evidence.get('independent_samples',0)),
            persistence_ros_s=_finite(evidence.get('persistence_ros_s',0.)),
            path_sha256=_safe_str(evidence.get('path_sha256'),128),
            prior_free=bool(evidence.get('prior_free',False)),
            stop_confirmed=False,physical_passage_cut_proven=False,
            physical_region_blocked_proven=False,
            hold_radius_m=self.hold_radius_m)
        self.records.append(r)
        self._save()
        return identifier,True

    def confirm_stop(self, record):
        """Promote an observed path witness to stopped-navigation witness only."""
        if not isinstance(record,dict) or not record.get('stop_confirmed'):
            return None
        goal_id=_safe_str(record.get('goal_id'),80)
        for r in reversed(self.records):
            if goal_id in r['goal_ids']:
                r['observation_status']='NAV_CANCELED_STOP_CONFIRMED'
                r['stop_confirmed']=True
                r['stop_confirmed_utc']=_utc()
                self._save()
                return r['id']
        return None

    def live_evidence(self, r, current_cost):
        """Current lethal cell + current verified map; old evidence alone is insufficient."""
        i,j=self._ij(r['xy'])
        if not self.geometry.contains(i,j):return False,None
        if getattr(current_cost,'shape',None)!=(self.geometry.height,self.geometry.width):
            raise ValueError('BLOCKAGE_MEMORY_COSTMAP_GEOMETRY_MISMATCH')
        value=int(current_cost[j,i])
        return value >= 100,value

    def review_path(self, points, current_cost, *, stage):
        """Return confirmed-live route intersection or None. No Nav2 actions sent."""
        if stage not in ('PLAN_PREFLIGHT','PRE_SEND'):
            raise ValueError('BLOCKAGE_MEMORY_BAD_STAGE')
        if not self.records:
            return None
        # Validate even if no hit, rather than silently trusting a corrupt path.
        path=[_point(q) for q in points]
        if not path:raise ValueError('BLOCKAGE_MEMORY_EMPTY_PATH')
        for r in self.records:
            distance=_distance_to_segments(path,r['xy'])
            if distance > r['hold_radius_m']:
                continue
            live,cell_cost=self.live_evidence(r,current_cost)
            if not live:
                continue
            self.holds_this_run+=1
            return dict(record_id=r['id'],xy=r['xy'],distance_to_path_m=distance,
                path_hold_radius_m=r['hold_radius_m'],current_cell_cost=cell_cost,
                current_live_costmap_support=True,observation_status=r['observation_status'],
                destination_zone=r['destination_zone'],spatial_zone=r['spatial_zone'],
                stage=stage,physical_passage_cut_proven=False,physical_region_blocked_proven=False)
        return None

    def summary(self, current_cost=None):
        points=[]
        for r in self.records:
            e={k:v for k,v in r.items() if k not in ('goal_ids',)}
            e['goal_count']=len(r['goal_ids'])
            e['current_live_costmap_support']=None
            if current_cost is not None:
                live,value=self.live_evidence(r,current_cost)
                e['current_live_costmap_support']=bool(live)
                e['current_cell_cost']=value
            e['geometry_semantics']='point_observation_only_NO_INFERRED_LINE_OR_REGION'
            points.append(e)
        return dict(schema=SCHEMA,version=VERSION,points=points,
            count=len(points),loaded_on_start=self.loaded_count,
            route_holds_this_run=self.holds_this_run,
            independent_current_costmap_required_for_route_hold=True,
            not_physical_cut_proof=True)
