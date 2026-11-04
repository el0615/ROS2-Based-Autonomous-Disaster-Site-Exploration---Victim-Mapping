#!/usr/bin/env python3
"""Per-room LiDAR ray coverage, no ROS imports and no navigation commands.

New project code. Approved polygon masks and map verification are reused from
hospital_zone_visualizer.py without altering that module or the original map.
"""
from __future__ import annotations
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    resolution: float
    origin: tuple[float, float, float]

    def world_to_grid_float(self, x: float, y: float) -> tuple[float, float]:
        ox, oy, yaw = self.origin
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = x-ox, y-oy
        return ((c*dx+s*dy)/self.resolution, (-s*dx+c*dy)/self.resolution)

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        ox, oy, yaw = self.origin
        x, y = (i+.5)*self.resolution, (j+.5)*self.resolution
        c, s = math.cos(yaw), math.sin(yaw)
        return ox+c*x-s*y, oy+s*x+c*y

    def contains(self, i: int, j: int) -> bool:
        return 0 <= i < self.width and 0 <= j < self.height


def decode_runs(width: int, height: int, runs: list, labeled: bool) -> np.ndarray:
    """Top-left PGM row runs -> bottom-up ROS row-major array."""
    grid = np.zeros((height, width), dtype=np.uint16 if labeled else bool)
    previous = (-1, 0)
    for run in runs:
        if len(run) != (4 if labeled else 3):
            raise ValueError('Incorrect RLE tuple size')
        row, left, right = run[:3]
        if not all(isinstance(v, int) for v in run):
            raise ValueError('RLE values must be integers')
        if not (0 <= row < height and 0 <= left < right <= width):
            raise ValueError('RLE outside map')
        if row < previous[0] or (row == previous[0] and left < previous[1]):
            raise ValueError('RLE overlap or unsorted rows')
        previous = (row, right)
        value = run[3] if labeled else True
        if labeled and not (1 <= value <= 65535):
            raise ValueError('Invalid zone mask ID')
        grid[height-1-row, left:right] = value
    return grid


def load_bundle(zones_path: Path, reference_path: Path) -> tuple[dict, Geometry, np.ndarray, np.ndarray, str]:
    from hospital_zone_visualizer import load_zones
    data, runs = load_zones(zones_path)
    info = data['map']
    geometry = Geometry(int(info['width']), int(info['height']),
                        float(info['resolution']), tuple(map(float, info['origin'])))
    labels = decode_runs(geometry.width, geometry.height, runs, True)
    with gzip.open(reference_path, 'rt', encoding='utf-8') as f:
        ref = json.load(f)
    for key in ('width', 'height', 'reference_pgm_sha256', 'occupancy_ros_order_sha256'):
        if ref[key] != info[key]:
            raise ValueError(f'Coverage reference mismatch: {key}')
    known_free = decode_runs(geometry.width, geometry.height, ref['known_free_runs_top_left'], False)
    if int(known_free.sum()) != ref['known_free_cells']:
        raise ValueError('Reference free-cell count mismatch')
    if hashlib.sha256(known_free.astype(np.uint8).tobytes()).hexdigest() != ref['free_ros_order_sha256']:
        raise ValueError('Reference free-cell fingerprint mismatch')
    if np.any((labels > 0) & ~known_free):
        raise ValueError('A survey cell is not source-PGM known free')
    ids = [z['mask_id'] for z in data['zones']]
    if set(np.unique(labels)) - {0, *ids}:
        raise ValueError('Unexpected region in mask')
    for z in data['zones']:
        if np.count_nonzero(labels == z['mask_id']) != z['known_free_cells']:
            raise ValueError(f"Wrong denominator for {z['id']}")
    signature = hashlib.sha256(
        zones_path.read_bytes() + labels.tobytes() + known_free.tobytes()).hexdigest()
    return data, geometry, labels, known_free, signature


def planar_pose(transform: Any, max_tilt_rad: float) -> tuple[float, float, float]:
    """Transform.translation/rotation -> x,y,yaw. Refuse tilted 2-D scans."""
    t, q = transform.translation, transform.rotation
    values = [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Nonfinite TF')
    norm = math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
    if norm < 1e-9:
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = [float(v)/norm for v in (q.x,q.y,q.z,q.w)]
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    if abs(roll) > max_tilt_rad or abs(pitch) > max_tilt_rad:
        raise ValueError('LiDAR plane tilt exceeds configured 2-D coverage limit')
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return float(t.x), float(t.y), yaw


class Coverage:
    """Cumulative *ray-observed reference free space*, not victim-search success.

    Source PGM walls/gray cells block rays. Measured obstacle endpoints and
    cells beyond the endpoint are never counted. This intentionally cannot
    recognize a demolished prior wall. No angular wedge filling is performed.
    """
    def __init__(self, geom: Geometry, labels: np.ndarray, known_free: np.ndarray,
                 zones: list[dict], threshold: float = .75):
        if labels.shape != (geom.height,geom.width) or labels.shape != known_free.shape:
            raise ValueError('Coverage shape mismatch')
        if not 0 < threshold <= 1:
            raise ValueError('Threshold must be in (0, 1]')
        self.geom, self.labels, self.known_free = geom, labels, known_free
        self.zones, self.threshold = zones, float(threshold)
        self.observed = np.zeros(labels.shape, dtype=bool)
        self.denominators = np.bincount(labels.ravel(), minlength=int(labels.max())+1)
        self.observed_counts = np.zeros_like(self.denominators)
        self.total_cells = int(np.count_nonzero(labels))
        self.observed_total = 0
        self.scan_count = 0
        self.evidence_scan_count = 0
        self.last_scan_end_s: float | None = None
        self.threshold_since: dict[int,float] = {}
        self.threshold_scan_count: dict[int,int] = {}
        self.threshold_confirmed: set[int] = set()

    def _is_free(self, i: int, j: int) -> bool:
        return self.geom.contains(i,j) and bool(self.known_free[j,i])

    def trace(self, sx: float, sy: float, theta: float, length_m: float,
              endpoint_cell: tuple[int,int] | None = None) -> int:
        """Continuous-grid DDA; diagonal corner blockers are checked, not filled."""
        g = self.geom
        x, y = g.world_to_grid_float(sx,sy)
        i, j = math.floor(x), math.floor(y)
        if length_m <= 0 or not self._is_free(i,j):
            return 0
        angle = theta-g.origin[2]
        dx, dy = math.cos(angle), math.sin(angle)
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        tx_delta = 1/abs(dx) if abs(dx) > 1e-12 else math.inf
        ty_delta = 1/abs(dy) if abs(dy) > 1e-12 else math.inf
        tx = ((i+1-x) if dx>0 else (x-i))*tx_delta if math.isfinite(tx_delta) else math.inf
        ty = ((j+1-y) if dy>0 else (y-j))*ty_delta if math.isfinite(ty_delta) else math.inf
        limit = length_m/g.resolution
        travelled = 0.0
        new = 0
        # The bound prevents malformed data from causing an infinite traversal.
        for _ in range(g.width+g.height+4):
            if not self._is_free(i,j) or (i,j) == endpoint_cell:
                break
            exit_t = min(tx,ty,limit)
            if exit_t > travelled+1e-10:
                region = int(self.labels[j,i])
                if region and not self.observed[j,i]:
                    self.observed[j,i] = True
                    self.observed_counts[region] += 1
                    self.observed_total += 1
                    new += 1
            if exit_t >= limit-1e-10:
                break
            if abs(tx-ty) < 1e-10:
                # A zero-width diagonal gap is not evidence of free visibility.
                if not self._is_free(i+step_x,j) or not self._is_free(i,j+step_y):
                    break
                i += step_x; j += step_y
                travelled = tx
                tx += tx_delta; ty += ty_delta
            elif tx < ty:
                i += step_x; travelled = tx; tx += tx_delta
            else:
                j += step_y; travelled = ty; ty += ty_delta
        return new

    def integrate(self, ranges, angle_min: float, angle_increment: float,
                  range_min: float, range_max: float,
                  start_pose: tuple[float,float,float], end_pose: tuple[float,float,float],
                  start_s: float, time_increment: float = 0.0,
                  max_range: float = 8.0, stride: int = 1,
                  use_infinite: bool = False) -> dict:
        metadata = [angle_min,angle_increment,range_min,range_max,start_s,time_increment,max_range,
                    *start_pose,*end_pose]
        if not all(math.isfinite(float(v)) for v in metadata):
            raise ValueError('Nonfinite scan metadata/pose')
        if not ranges or start_s <= 0 or time_increment < 0 or range_min < 0 or range_max <= range_min or max_range <= 0:
            raise ValueError('Invalid scan metadata')
        if stride < 1:
            raise ValueError('Stride must be positive')
        last_s = start_s+(len(ranges)-1)*time_increment
        if self.last_scan_end_s is not None and start_s <= self.last_scan_end_s+1e-9:
            return {'processed':False,'reason':'duplicate_or_overlapping_scan','new_cells':0}
        cap = min(max_range, range_max)
        result = dict(processed=True, new_cells=0, finite_beams=0, no_return_beams=0,
                      invalid_beams=0, skipped_infinite=0, sensor_blocked_beams=0, used_beams=0)
        delta_yaw = math.atan2(math.sin(end_pose[2]-start_pose[2]),math.cos(end_pose[2]-start_pose[2]))
        for k in range(0,len(ranges),stride):
            measured = float(ranges[k])
            infinite = math.isinf(measured) and measured > 0
            if infinite:
                if not use_infinite:
                    result['skipped_infinite'] += 1
                    continue
                length = cap
                result['no_return_beams'] += 1
            elif not math.isfinite(measured) or measured < range_min or measured > range_max:
                result['invalid_beams'] += 1
                continue
            else:
                length = min(measured,cap)
                result['finite_beams'] += 1
            if length <= range_min:
                continue
            fraction = k/max(1,len(ranges)-1) if time_increment > 0 else 0.0
            sx = start_pose[0]+fraction*(end_pose[0]-start_pose[0])
            sy = start_pose[1]+fraction*(end_pose[1]-start_pose[1])
            yaw = start_pose[2]+fraction*delta_yaw
            sensor_cell = tuple(map(math.floor,self.geom.world_to_grid_float(sx,sy)))
            if not self._is_free(*sensor_cell):
                result['sensor_blocked_beams'] += 1
                continue
            result['used_beams'] += 1
            theta = yaw+angle_min+k*angle_increment
            hit = None
            if not infinite and measured < range_max and measured <= cap:
                hit = tuple(map(math.floor,self.geom.world_to_grid_float(
                    sx+measured*math.cos(theta),sy+measured*math.sin(theta))))
            # Start at range_min: never fill the lidar blind disk by assumption.
            rx, ry = sx+range_min*math.cos(theta), sy+range_min*math.sin(theta)
            result['new_cells'] += self.trace(rx,ry,theta,length-range_min,hit)
        self.scan_count += 1
        self.last_scan_end_s = last_s
        if result['used_beams'] > 0:
            self.evidence_scan_count += 1
        for zone in self.zones:
            rid = zone['mask_id']
            fraction = self.observed_counts[rid]/self.denominators[rid] if self.denominators[rid] else 0.0
            if fraction >= self.threshold and result['used_beams'] > 0:
                self.threshold_since.setdefault(rid,last_s)
                self.threshold_scan_count[rid] = self.threshold_scan_count.get(rid,0)+1
                if self.threshold_scan_count[rid] >= 2 and last_s-self.threshold_since[rid] >= 1.0-1e-9:
                    self.threshold_confirmed.add(rid)
        return result

    def statistics(self) -> list[dict]:
        stats=[]
        for z in self.zones:
            rid=z['mask_id']; n=int(self.denominators[rid]); observed=int(self.observed_counts[rid])
            fraction=observed/n if n else None
            state=('NO_VALID_SCAN' if not self.evidence_scan_count else 'UNOBSERVED' if observed==0
                   else 'THRESHOLD_REACHED' if rid in self.threshold_confirmed else 'PARTIAL')
            if n==0:
                state='NO_REFERENCE_FREE_CELLS'
            stats.append(dict(id=z['id'],kind=z['kind'],mask_id=rid,
                              reference_free_cells=n,observed_cells=observed,
                              unobserved_cells=n-observed,
                              reference_free_area_m2=n*self.geom.resolution**2,
                              observed_area_m2=observed*self.geom.resolution**2,
                              coverage_fraction=fraction,
                              coverage_percent=None if fraction is None else round(fraction*100,4),
                              observation_state=state,
                              victim_search_complete=False))
        return stats

    def grid_values(self) -> np.ndarray:
        grid=np.full(self.labels.shape,-1,dtype=np.int8)
        grid[self.labels>0]=0
        grid[self.observed]=100
        return grid

    def observed_runs(self) -> list[list[int]]:
        runs=[]
        for j,row in enumerate(self.observed):
            changes=np.diff(np.r_[False,row,False].astype(np.int8))
            starts=np.flatnonzero(changes==1); ends=np.flatnonzero(changes==-1)
            runs.extend([[int(j),int(a),int(b)] for a,b in zip(starts,ends)])
        return runs
