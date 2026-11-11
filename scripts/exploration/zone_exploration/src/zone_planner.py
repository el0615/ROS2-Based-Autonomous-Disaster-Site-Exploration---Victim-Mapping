"""Zone-first entry generation, local-view catalog caching and fresh reranking.

No ROS calls. No mutation of map, masks, Coverage or Nav2 parameters. A returned
candidate is only a proposal; the executor must request Navfn and run preflight.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import asdict, replace
import hashlib
import math
import threading
import time
import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra
from planning_core import (
    conservative_safe, conservative_start_bridge, path_graph,
    generate_candidates, oriented_footprint_clear)
from zone_policy import room_priority, repeated
from visibility_backend import VisibilityBackend


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def arrival_diagnostic(zone, goal, pose, bundle, cfg):
    """Read-only measured arrival evidence. Never weakens is_inside().

    Signed depth is sampled from approved Zone mask, not Costmap; it is
    informational and NOT a clearance or collision certificate.
    """
    _, geom, labels, _, _ = bundle
    target = labels == zone['mask_id']
    inner = ndimage.distance_transform_edt(np.pad(target, 1))[1:-1, 1:-1] * geom.resolution
    outer = ndimage.distance_transform_edt(np.pad(~target, 1))[1:-1, 1:-1] * geom.resolution
    def detail(p):
        i,j = map(math.floor, geom.world_to_grid_float(*p[:2]))
        valid=geom.contains(i,j)
        center=bool(valid and target[j,i])
        signed_depth=(float(inner[j,i]) if center else -float(outer[j,i])) if valid else None
        return dict(center_inside=center,footprint_inside=bool(is_inside(zone,p,bundle,cfg)),
                    signed_mask_depth_m=signed_depth,cell=[i,j] if valid else None)
    delta_yaw = math.atan2(math.sin(pose[2]-goal[2]), math.cos(pose[2]-goal[2]))
    return dict(zone=zone['id'],goal_pose=list(goal),actual_pose=list(pose),
                xy_error_m=math.hypot(pose[0]-goal[0],pose[1]-goal[1]),
                yaw_error_rad=delta_yaw,goal=detail(goal),actual=detail(pose))


def is_inside(zone, pose, bundle, cfg):
    """Measured pose AND padded rectangle must lie in the approved zone free mask."""
    _, geom, labels, _, _ = bundle
    i, j = map(math.floor, geom.world_to_grid_float(*pose[:2]))
    return (geom.contains(i, j) and labels[j, i] == zone['mask_id']
            and oriented_footprint_clear(labels == zone['mask_id'], geom, pose, cfg))


def refresh(candidate, observed, geom, cfg):
    c = candidate.copy()
    if c.get('goal_kind') == 'ENTRY':
        # Entry is deliberately NOT filtered by information-gain threshold.
        return c
    ids = np.asarray(c['visible_indices'], dtype=np.int64)
    new = ids[~observed.ravel()[ids]]
    c.update(new_indices=new, new_cells=len(new), new_area_m2=len(new)*geom.resolution**2)
    c['score'] = c['new_area_m2']/(c['path_est_m']+cfg.distance_regularizer_m)
    return c if c['new_cells'] > 0 and c['new_area_m2'] >= cfg.min_gain_m2 else None


class ZonePlanner:
    def __init__(self, bundle, cfg, *, entry_candidates=8, entry_inset=.45,
                 policy='rooms-first', local_candidates=15, heading_samples=4,
                 budget_s=15., backend='auto', entrance_overlay=None,
                 entry_arrival_buffer=.20):
        self.bundle = bundle
        self.cfg = cfg
        self.entry_candidates = entry_candidates
        self.entry_inset = entry_inset
        self.entry_arrival_buffer = entry_arrival_buffer
        self.policy = policy
        self.local_candidates = local_candidates
        self.heading_samples = heading_samples
        self.budget_s = budget_s
        self.entrance_overlay = entrance_overlay
        self.visibility = VisibilityBackend(backend)
        self.cancel = threading.Event()
        self.catalogs = OrderedDict()
        self.map_cache = None
        self.jobs = self.catalog_hits = 0

    def stop(self):
        self.cancel.set()

    def checkpoint(self, deadline):
        if self.cancel.is_set():
            raise RuntimeError('PLANNER_CANCELLED')
        if time.perf_counter() >= deadline:
            raise TimeoutError('PLANNING_BUDGET_PARTIAL')

    def maps(self, snapshot):
        _, geom, labels, known, _ = self.bundle
        cost = snapshot['cost']['values']
        if cost.shape != labels.shape or not np.all((cost >= -1) & (cost <= 100)):
            raise ValueError('Invalid costmap')
        free = known
        overlay_info = None
        if self.entrance_overlay is not None:
            from entrance_navigation_overlay import apply_entrance_overlay
            free, overlay_info = apply_entrance_overlay(
                known, labels, geom, cost, snapshot['pose'], self.entrance_overlay)
        free = free & (cost >= 0) & (cost < 100)
        key = digest(free) + ':' + digest(cost == 99)
        if self.map_cache is None or self.map_cache['key'] != key:
            safe, clearance = conservative_safe(free, geom, self.cfg)
            safe &= cost < 99
            ids, flat, graph = path_graph(safe, geom.resolution)
            self.map_cache = dict(key=key, free=free, safe=safe, clearance=clearance,
                                  ids=ids, flat=flat, graph=graph)
        result = dict(self.map_cache)
        result['entrance'] = overlay_info
        return result

    def distances(self, maps, pose):
        """Strict graph for ranking ONLY; Navfn still authorizes every real route.

        A short conservative bridge is a DISPLAY/DISTANCE ESTIMATOR, not a
        collision proof or a prerequisite imposed by Nav2. If it cannot reach
        the strict circular graph, propose strict-safe endpoints with a
        Euclidean ranking; every proposal still needs fresh Navfn validation,
        target footprint and zone-mask checks, and local-start clearance.
        """
        _, geom, _, _, _ = self.bundle
        i, j = map(math.floor, geom.world_to_grid_float(*pose[:2]))
        if not geom.contains(i, j) or not maps['free'][j, i]:
            raise ValueError('START_NOT_CURRENT_FREE')
        bridge_m = 0.
        mode = 'STRICT_CONSERVATIVE_GRID'
        if not maps['safe'][j, i]:
            bridge = conservative_start_bridge(maps['free'], maps['safe'], geom, pose, self.cfg)
            if bridge is None:
                # Do not invent a connected path / snap robot onto a graph.
                # Distances are deliberately infinite and only endpoint probes
                # are proposed. Node-side Navfn and safety gates remain mandatory.
                distances = np.full(maps['graph'].shape[0], np.inf, dtype=float)
                return distances, 0., 'NAVFN_START_PROBE_NO_CONSERVATIVE_BRIDGE'
            i, j, bridge_m = bridge
            mode = 'FOOTPRINT_LOCAL_BRIDGE_DISPLAY_ONLY'
        distances = dijkstra(maps['graph'], directed=True, indices=int(maps['ids'][j, i]))
        return distances, bridge_m, mode

    def internal_mask(self, zone, maps, *, with_depth=False):
        _, geom, labels, _, _ = self.bundle
        target = labels == zone['mask_id']
        depth = ndimage.distance_transform_edt(np.pad(target, 1))[1:-1, 1:-1]*geom.resolution
        result = target & maps['safe'] & (depth >= self.entry_inset)
        return (result, depth) if with_depth else result

    def base_candidate(self, zone, i, j, yaw, distance, maps, kind, connected):
        _, geom, labels, _, _ = self.bundle
        x, y = geom.cell_to_world(i, j)
        return dict(id=f'{kind}_{zone["id"]}_{i}_{j}_{yaw:.6f}', zone_id=zone['id'],
                    cell=[int(i), int(j)], x=float(x), y=float(y), yaw=float(yaw),
                    goal_kind=kind, path_est_m=float(distance),
                    clearance_lower_bound_m=float(maps['clearance'][j, i]-geom.resolution/math.sqrt(2)),
                    viewpoint_zone_mask_id=int(labels[j, i]),
                    route_mode='STRICT_GRID' if connected else 'NAVFN_ENTRY_PROBE',
                    visible_indices=np.empty(0, dtype=np.int32), new_indices=np.empty(0, dtype=np.int32),
                    new_cells=0, new_area_m2=0., score=0., path_cells=[])

    def entry_batch(self, snapshot, zones, history):
        start = time.perf_counter()
        maps = self.maps(snapshot)
        distances, bridge, start_mode = self.distances(maps, snapshot['pose'])
        _, geom, labels, _, _ = self.bundle
        proposals, failures = [], {}
        for zone in zones:
            self.checkpoint(start + self.budget_s)
            target = labels == zone['mask_id']
            internal, depth = self.internal_mask(zone, maps, with_depth=True)
            ys, xs = np.nonzero(internal)
            if not len(xs):
                failures[zone['id']] = 'NO_SAFE_INTERIOR_POSE_NOT_UNREACHABLE_PROOF'
                continue
            # A coarse tile catalog is cheap, and endpoints remain strictly internal.
            cfg = replace(self.cfg, max_candidates=min(100, max(30, self.entry_candidates*6)), outside_room_m=0.)
            points = generate_candidates(internal, maps['clearance'], target, labels, geom, cfg)
            ty, tx = np.nonzero(target)
            cx, cy = geom.cell_to_world(float(np.mean(tx)), float(np.mean(ty)))
            options = []
            for i, j in points:
                ident = int(maps['ids'][j, i]); d = float(distances[ident]) + bridge
                connected = math.isfinite(d)
                x, y = geom.cell_to_world(i, j)
                # A disconnected conservative graph is not an unreachable-room verdict.
                # Only propose the endpoint; Navfn and goal validity checks decide later.
                if not connected:
                    d = math.hypot(x-snapshot['pose'][0], y-snapshot['pose'][1])
                yaw = math.atan2(cy-y, cx-x)
                c = self.base_candidate(zone, i, j, yaw, d, maps, 'ENTRY', connected)
                if not is_inside(zone, (x, y, yaw), self.bundle, self.cfg) or repeated(c, history):
                    continue
                # A failed ENTRY should not be retried at an almost identical
                # position merely by changing yaw. Keep distinct spatial goals.
                if any((p.get('zone_id') == zone['id'] and p.get('goal_kind') == 'ENTRY'
                        and math.hypot(x-p['x'], y-p['y']) < .60) for p in history):
                    continue
                c['entry_boundary_depth_m'] = float(depth[j, i])
                options.append(c)
            arrival_depth_target = self.entry_inset + self.entry_arrival_buffer
            options.sort(key=lambda c: (c['route_mode'] != 'STRICT_GRID',
                                        c['entry_boundary_depth_m'] < arrival_depth_target,
                                        c['path_est_m'], -c['entry_boundary_depth_m'],
                                        -c['clearance_lower_bound_m'], c['id']))
            distinct = []
            for c in options:
                if all(math.hypot(c['x']-p['x'], c['y']-p['y']) >= .4 for p in distinct):
                    distinct.append(c)
                if len(distinct) >= self.entry_candidates:
                    break
            if not distinct:
                failures[zone['id']] = 'NO_DISTINCT_INTERIOR_ENTRY_THIS_REVISION'
                continue
            proposals.append((room_priority(zone, self.policy),
                              distinct[0]['route_mode'] != 'STRICT_GRID',
                              distinct[0]['path_est_m'], zone['id'], distinct))
        proposals.sort(key=lambda row: row[:4])
        self.jobs += 1
        return dict(kind='ENTRY', candidates=proposals[0][4] if proposals else [],
                    zone_id=proposals[0][3] if proposals else None,
                    failures=failures, entrance=maps['entrance'],start_route_mode=start_mode,
                    elapsed_s=time.perf_counter()-start,
                    ranked_zones=[dict(zone_id=r[3], estimated_distance=r[2],
                                       strict_connected=not r[1], kind_priority=r[0]) for r in proposals])

    def catalog_key(self, zone, maps, sensor):
        return (maps['key'], zone['id'], tuple(asdict(sensor).values()),
                self.local_candidates, self.heading_samples, self.cfg.max_range_m)

    def catalog(self, snapshot, zone):
        start = time.perf_counter()
        maps = self.maps(snapshot)
        sensor = snapshot['sensor']; key = self.catalog_key(zone, maps, sensor)
        if key in self.catalogs:
            self.catalog_hits += 1
            self.catalogs.move_to_end(key)
            return self.catalogs[key], maps, True
        _, geom, labels, _, _ = self.bundle
        target = labels == zone['mask_id']
        internal = self.internal_mask(zone, maps)
        cfg = replace(self.cfg, max_candidates=self.local_candidates, outside_room_m=0.)
        points = generate_candidates(internal, maps['clearance'], target, labels, geom, cfg)
        ys, xs = np.nonzero(target)
        if not len(xs):
            raise ValueError('Empty zone')
        cx, cy = geom.cell_to_world(float(np.mean(xs)), float(np.mean(ys)))
        catalog = []
        for i, j in points:
            self.checkpoint(start+self.budget_s)
            x, y = geom.cell_to_world(i, j)
            base_yaw = math.atan2(cy-y, cx-x)-sensor.offset_yaw
            headings = 1 if sensor.full_circle else self.heading_samples
            for k in range(headings):
                self.checkpoint(start+self.budget_s)
                yaw = base_yaw + k*2*math.pi/headings
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                if not is_inside(zone, (x,y,yaw), self.bundle, self.cfg):
                    continue
                visible = self.visibility(geom, maps['free'], target, x, y, yaw, sensor, self.cfg.max_range_m)
                c = self.base_candidate(zone, i, j, yaw, 0., maps, 'OBSERVE', True)
                c['visible_indices'] = visible
                catalog.append(c)
        # Store only a COMPLETE catalog. A timeout never caches a partial result as complete.
        self.catalogs[key] = catalog
        while len(self.catalogs) > 4:
            self.catalogs.popitem(last=False)
        return catalog, maps, False

    def local_batch(self, snapshot, zone, history):
        start = time.perf_counter()
        catalog, maps, hit = self.catalog(snapshot, zone)
        distances, bridge, start_mode = self.distances(maps, snapshot['pose'])
        _, geom, _, _, _ = self.bundle
        result = []
        for original in catalog:
            i,j = original['cell']; d = float(distances[int(maps['ids'][j,i])])+bridge
            connected = math.isfinite(d)
            if not connected:
                d=math.hypot(original['x']-snapshot['pose'][0], original['y']-snapshot['pose'][1])
            c = dict(original, path_est_m=d,
                     route_mode='STRICT_GRID' if connected else 'NAVFN_ENTRY_PROBE')
            c = refresh(c, snapshot['observed'], geom, self.cfg)
            if c is not None and not repeated(c, history):
                result.append(c)
        result.sort(key=lambda c: (-c['score'], -c['new_area_m2'], c['path_est_m'], c['id']))
        return dict(kind='OBSERVE', zone_id=zone['id'], candidates=result,
                    failures={}, entrance=maps['entrance'],start_route_mode=start_mode,cache_hit=hit,
                    elapsed_s=time.perf_counter()-start, catalog_views=len(catalog),
                    best_gain_m2=result[0]['new_area_m2'] if result else 0.)

    def prefetch(self, snapshot, zone):
        start = time.perf_counter()
        catalog, _, hit = self.catalog(snapshot, zone)
        return dict(kind='PREFETCH', zone_id=zone['id'], catalog_views=len(catalog),
                    cache_hit=hit, elapsed_s=time.perf_counter()-start)

    def run(self, kind, snapshot, zones, history):
        if kind == 'ENTRY':
            return self.entry_batch(snapshot, zones, history)
        if len(zones) != 1:
            raise ValueError('Local observation must target one zone')
        if kind == 'PREFETCH':
            return self.prefetch(snapshot, zones[0])
        return self.local_batch(snapshot, zones[0], history)
