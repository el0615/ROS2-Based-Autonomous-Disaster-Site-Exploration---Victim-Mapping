"""DN-012 residual-region analysis, without ROS or motion commands.

Arrays use ROS bottom-up row order. Coverage and accessibility are independent.
A positive cut is scoped to the supplied finite grid and fixed-prior assumption;
it is NOT a permanent physical-world certificate. No inflation costs are
interpreted as measured obstacles. The configuration graph is deliberately an
OVER-approximation: connection is inconclusive, disconnection can be evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import os
import numpy as np
from scipy import ndimage

FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
VERSION = 'DN012_REGION_RESCUE_V1'


@dataclass(frozen=True)
class RegionSettings:
    min_area_m2: float = 1.0
    priority_area_m2: float = 1.0
    spatial_separation_m: float = .6
    low_yield_area_m2: float = .05
    max_rescue_goals: int = 8
    max_regions_per_round: int = 16
    max_access_regions: int = 64
    min_progress_m: float = .30
    sensor_tolerance_m: float = .12
    persistence_s: float = 2.0
    minimum_samples: int = 3
    support_max_age_s: float = 1.5

    def validate(self):
        for name in ('min_area_m2', 'priority_area_m2', 'spatial_separation_m',
                     'low_yield_area_m2', 'min_progress_m', 'sensor_tolerance_m',
                     'persistence_s', 'support_max_age_s'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('INVALID_REGION_SETTING:' + name)
        if not .05 <= self.min_area_m2 <= self.priority_area_m2 <= 20:
            raise ValueError('INVALID_REGION_AREAS')
        if not .2 <= self.spatial_separation_m <= 3:
            raise ValueError('INVALID_SPATIAL_SEPARATION')
        if not .01 <= self.sensor_tolerance_m <= .15:
            raise ValueError('INVALID_SENSOR_TOLERANCE')
        for name, high in (('max_rescue_goals', 20), ('max_regions_per_round', 64),
                           ('max_access_regions', 128), ('minimum_samples', 10)):
            v = getattr(self, name)
            if type(v) is not int or not 1 <= v <= high:
                raise ValueError('INVALID_REGION_COUNT:' + name)
        if self.minimum_samples < 3:
            raise ValueError('MINIMUM_THREE_INDEPENDENT_SAMPLES')
        return self

    @classmethod
    def from_environment(cls):
        # Only these four proposed experimental controls are exposed initially.
        return cls(min_area_m2=float(os.getenv('DN012_REGION_MIN_AREA_M2', '1.0')),
                   priority_area_m2=float(os.getenv('DN012_REGION_PRIORITY_AREA_M2', '1.0')),
                   max_rescue_goals=int(os.getenv('DN012_REGION_MAX_GOALS', '8')),
                   spatial_separation_m=float(os.getenv('DN012_REGION_SEPARATION_M', '.6'))).validate()


def validate_grids(labels, known, cost, observed):
    shape = np.asarray(labels).shape
    if len(shape) != 2 or any(np.asarray(x).shape != shape for x in (known, cost, observed)):
        raise ValueError('REGION_GRID_GEOMETRY_MISMATCH')
    if not np.all(np.isfinite(cost)) or not np.all((cost >= -1) & (cost <= 100)):
        raise ValueError('REGION_INVALID_COST_VALUES')
    if np.asarray(known).dtype != bool or np.asarray(observed).dtype != bool:
        raise ValueError('REGION_MASKS_MUST_BE_BOOLEAN')


def components(mask, geom, min_area=0.):
    """Exact 4-neighbour connected unobserved/floor-cell sets; no hole filling."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (geom.height, geom.width):
        raise ValueError('REGION_GEOMETRY_MISMATCH')
    lab, count = ndimage.label(mask, structure=FOUR)
    sizes = np.bincount(lab.ravel(), minlength=count + 1)
    min_cells = max(1, math.ceil(min_area / geom.resolution**2 - 1e-8))
    result = []
    boxes = ndimage.find_objects(lab)
    for ident in np.flatnonzero(sizes >= min_cells):
        if ident == 0:
            continue
        sy, sx = boxes[ident - 1]
        yy, xx = np.nonzero(lab[sy, sx] == ident)
        yy, xx = yy + sy.start, xx + sx.start
        indices = (yy * geom.width + xx).astype(np.int32)
        x, y = geom.cell_to_world(float(np.mean(xx)), float(np.mean(yy)))
        result.append(dict(region_id=hashlib.sha256(indices.tobytes()).hexdigest()[:16],
                           cells=int(len(indices)), area_m2=float(len(indices) * geom.resolution**2),
                           centroid_map_m=[float(x), float(y)],
                           bbox_cells_ros=[int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
                           indices=indices))
    result.sort(key=lambda c: (-c['cells'], c['region_id']))
    return result


def row_runs(indices, width):
    values = np.sort(np.asarray(indices, dtype=np.int64))
    runs = []
    for value in values.tolist():
        row, col = divmod(value, width)
        if runs and runs[-1][0] == row and runs[-1][2] == col:
            runs[-1][2] = col + 1
        else:
            runs.append([row, col, col + 1])
    return runs


def public_region(region):
    return {k: v for k, v in region.items() if k != 'indices'}


def residual_regions(bundle, snapshot, zone, settings):
    _, geom, labels, known, _ = bundle
    observed, cost = snapshot['observed'], snapshot['cost']['values']
    validate_grids(labels, known, cost, observed)
    remaining = (labels == zone['mask_id']) & known & ~observed
    all_regions = components(remaining, geom)
    meaningful = [r for r in all_regions if r['area_m2'] + 1e-9 >= settings.min_area_m2]
    return meaningful, dict(reference_cells=int(((labels == zone['mask_id']) & known).sum()),
                            remaining_cells=int(remaining.sum()),
                            remaining_area_m2=float(remaining.sum() * geom.resolution**2),
                            significant_regions=len(meaningful),
                            small_residual_cells=int(remaining.sum()) - sum(r['cells'] for r in meaningful),
                            significance_threshold_m2=settings.min_area_m2,
                            regions=[public_region(r) for r in meaningful],
                            observation_is_not_accessibility=True)


def same_view(candidate, history, radius=.08, yaw_tolerance=.10):
    for old in history:
        if old.get('zone_id') != candidate['zone_id']:
            continue
        delta = float(old.get('yaw', 0.)) - candidate['yaw']
        if (math.hypot(old['x'] - candidate['x'], old['y'] - candidate['y']) <= radius and
                abs(math.atan2(math.sin(delta), math.cos(delta))) <= yaw_tolerance):
            return True
    return False


def rank_region_views(catalog, regions, snapshot, geom, history, low_yield_positions,
                      settings, held_positions=()):
    """One best heading per XY per region; spatial progress before gain/distance.

    A zero-predicted-gain investigation is permitted only if it makes geometric
    progress towards a still-significant region. Existing executor preflights
    remain mandatory. This is proposal generation, NEVER a motion authorization.
    """
    observed = snapshot['observed'].ravel()
    pi, pj = map(math.floor, geom.world_to_grid_float(*snapshot['pose'][:2]))
    if not geom.contains(pi, pj):
        raise ValueError('REGION_START_OUTSIDE_GRID')
    options = []
    selected_regions = regions[:settings.max_regions_per_round]
    for region in selected_regions:
        target = np.zeros((geom.height, geom.width), dtype=bool)
        target.ravel()[region['indices']] = True
        distances = ndimage.distance_transform_edt(~target) * geom.resolution
        current_distance = float(distances[pj, pi])
        by_position = {}
        for original in catalog['views']:
            if original.get('audit_position_id') in held_positions or same_view(original, history):
                continue
            if any(math.hypot(original['x'] - p[0], original['y'] - p[1]) < settings.spatial_separation_m
                   for p in low_yield_positions):
                continue
            i, j = original['cell']
            goal_distance = float(distances[j, i])
            movement = math.hypot(original['x'] - snapshot['pose'][0], original['y'] - snapshot['pose'][1])
            ids = np.asarray(original['visible_indices'], dtype=np.int64)
            novel = ids[~observed[ids]]
            target_novel = int(target.ravel()[novel].sum())
            progress = current_distance - goal_distance
            # Distant novel targets must not keep selecting the same nearby XY.
            advancing = progress >= settings.min_progress_m or goal_distance <= geom.resolution
            if movement < settings.spatial_separation_m or not advancing:
                continue
            c = dict(original, new_indices=novel, new_cells=int(len(novel)),
                     new_area_m2=float(len(novel) * geom.resolution**2),
                     region_investigation=True, residual_region_id=region['region_id'],
                     residual_target_indices=region['indices'].copy(),
                     residual_min_area_m2=settings.min_area_m2,
                     residual_target_cells=region['cells'],
                     residual_overlap_cells=target_novel,
                     residual_distance_m=goal_distance,
                     residual_progress_m=progress)
            c['path_est_m'] = movement
            c['score'] = c['new_area_m2'] / (movement + 2.)
            # Try nearest feasible positions IN or immediately beside this region,
            # not repeated rotation at the current, already unproductive location.
            key = (goal_distance, -target_novel, movement, c['id'])
            position = tuple(c['cell'])
            if position not in by_position or key < by_position[position][0]:
                by_position[position] = (key, c)
        for key, c in by_position.values():
            tier = 0 if region['area_m2'] >= settings.priority_area_m2 else 1
            options.append(((tier, key[0], -region['area_m2'], key[2], key[1], key[3]), c))
    options.sort(key=lambda pair: pair[0])
    unique = []
    used = set()
    for _, c in options:
        if c['id'] not in used:
            unique.append(c)
            used.add(c['id'])
    return unique, dict(regions_considered=len(selected_regions),
                        regions_not_considered_this_round=max(0, len(regions) - len(selected_regions)),
                        distinct_view_proposals=len(unique))


def refresh_investigation(candidate, observed, geom):
    """Fresh residual-presence gate only; never bypass path/footprint/stop gates."""
    ids = np.asarray(candidate.get('residual_target_indices', []), dtype=np.int64)
    flat = np.asarray(observed, dtype=bool).ravel()
    if not len(ids) or np.any(ids < 0) or np.any(ids >= flat.size):
        return None
    remain = int((~flat[ids]).sum())
    minimum = float(candidate.get('residual_min_area_m2', math.inf))
    if remain * geom.resolution**2 + 1e-9 < minimum:
        return None
    visible = np.asarray(candidate['visible_indices'], dtype=np.int64)
    if np.any(visible < 0) or np.any(visible >= flat.size):
        return None
    new = visible[~flat[visible]]
    c = dict(candidate, new_indices=new, new_cells=int(len(new)),
             new_area_m2=float(len(new) * geom.resolution**2))
    c['score'] = c['new_area_m2'] / (float(c['path_est_m']) + 2.)
    return c


class PersistentCells:
    """Per-cell independent scan + costmap samples, finite freshness and epoch reset."""
    def __init__(self, shape, settings):
        self.shape = shape
        self.s = settings
        self.epoch = None
        self.last_pair = None
        self.last_now = None
        self.first = np.full(shape, np.nan)
        self.last = np.full(shape, -np.inf)
        self.count = np.zeros(shape, dtype=np.uint16)

    def reset(self):
        self.first.fill(np.nan); self.last.fill(-np.inf); self.count.fill(0)
        self.last_pair = None

    def update(self, supported, now, scan_stamp, cost_stamp, epoch):
        if np.asarray(supported).shape != self.shape:
            raise ValueError('WITNESS_GEOMETRY_MISMATCH')
        if not all(math.isfinite(v) for v in (now, scan_stamp, cost_stamp)):
            self.reset(); return np.zeros(self.shape, dtype=bool)
        if epoch != self.epoch or (self.last_now is not None and now < self.last_now):
            self.reset(); self.epoch = epoch
        self.last_now = now
        if not -.1 <= now - scan_stamp <= 1. or not -.1 <= now - cost_stamp <= 5.:
            self.reset(); return np.zeros(self.shape, dtype=bool)
        expired = now - self.last > self.s.support_max_age_s
        self.count[expired] = 0; self.first[expired] = np.nan
        independent = self.last_pair is None or (scan_stamp > self.last_pair[0] and cost_stamp > self.last_pair[1])
        if independent:
            fresh = supported & (self.count == 0)
            self.first[fresh] = now
            self.count[supported] = np.minimum(self.count[supported].astype(np.uint32) + 1, 60000)
            self.last[supported] = now
            self.last_pair = (scan_stamp, cost_stamp)
        return (supported & (self.count >= self.s.minimum_samples) &
                (self.last - self.first >= self.s.persistence_s) &
                (now - self.last <= self.s.support_max_age_s))


def scan_supported_cells(hits, geom, prior_free, cost, tolerance):
    """Only currently lethal cells near real finite LiDAR endpoints qualify."""
    hits = np.asarray(hits, dtype=float)
    result = np.zeros_like(prior_free, dtype=bool)
    if hits.ndim != 2 or hits.shape[1] != 2 or not np.isfinite(hits).all():
        return result
    if not len(hits):
        return result
    ox, oy, yaw = geom.origin
    dx, dy = hits[:, 0] - ox, hits[:, 1] - oy
    gx = (math.cos(yaw) * dx + math.sin(yaw) * dy) / geom.resolution
    gy = (-math.sin(yaw) * dx + math.cos(yaw) * dy) / geom.resolution
    base_i, base_j = np.floor(gx).astype(int), np.floor(gy).astype(int)
    reach = int(math.ceil(tolerance / geom.resolution))
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            ii, jj = base_i + di, base_j + dj
            valid = ((ii >= 0) & (ii < geom.width) & (jj >= 0) & (jj < geom.height) &
                     ((ii + .5 - gx)**2 + (jj + .5 - gy)**2 <= (tolerance / geom.resolution)**2))
            result[jj[valid], ii[valid]] = True
    return result & prior_free & (cost == 100)


def zone_polygon_domain(zone, geom, fallback):
    """Semantic zone extent, independent of the white-only Coverage mask."""
    polygon = zone.get('polygon_map_m')
    if polygon is None:
        return np.asarray(fallback, dtype=bool), 'LABEL_MASK_FALLBACK'
    vertices = np.asarray([geom.world_to_grid_float(*p) for p in polygon], dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError('INVALID_ZONE_POLYGON')
    result = np.zeros((geom.height, geom.width), dtype=bool)
    x0 = max(0, int(math.floor(vertices[:, 0].min())))
    x1 = min(geom.width, int(math.ceil(vertices[:, 0].max())))
    y0 = max(0, int(math.floor(vertices[:, 1].min())))
    y1 = min(geom.height, int(math.ceil(vertices[:, 1].max())))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    xx += .5; yy += .5
    inside = np.zeros(xx.shape, dtype=bool)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        if abs(b[1] - a[1]) < 1e-12:
            continue
        intersects = ((a[1] > yy) != (b[1] > yy)) & (xx < (b[0] - a[0]) * (yy - a[1]) / (b[1] - a[1]) + a[0])
        inside ^= intersects
    result[y0:y1, x0:x1] = inside
    return result, 'SEMANTIC_ZONE_POLYGON'


def access_analysis(bundle, snapshot, zone, persistent_blockers, cfg, settings,
                    fixed_prior_topology, endpoint_checker=None):
    from exploration_results import assess_access
    return assess_access(bundle, snapshot, zone, persistent_blockers, cfg, settings,
                         fixed_prior_topology, endpoint_checker)
