"""DN012 v1.6 residual-region ENTRY proposals, not motion authorization.

Uses the existing Zone ENTRY endpoint standards: prior-free/live-cost free,
conservative endpoint clearance, inward depth, padded oriented footprint.
No measured-observed-only restriction, short-waypoint cap, or visibility gate.
Original node-side Nav2 plan validation, local-start and live collision handling
remain authoritative. No ROS calls; no mutation of maps/coverage/parameters.
"""
from __future__ import annotations
import math
import time
import numpy as np
from scipy import ndimage
from regions import components, validate_grids


def _indices(value, size):
    a = np.asarray(value)
    if a.ndim != 1 or (a.size and (a.dtype.kind not in 'iu' or
                                  a.min() < 0 or a.max() >= size)):
        raise ValueError('REGION_ENTRY_INVALID_INDICES')
    return np.unique(a.astype(np.int32))


def entry_membership(candidate, pose, geom, cfg, footprint_check):
    """Measured entry into the FROZEN target, not today's observed-mask holes."""
    ids = _indices(candidate['region_entry_indices'], geom.width * geom.height)
    mask = np.zeros((geom.height, geom.width), dtype=bool)
    mask.ravel()[ids] = True
    if len(pose) < 3 or not np.isfinite(pose[:3]).all():
        return dict(center_inside=False, footprint_inside=False, pose_valid=False)
    i, j = map(math.floor, geom.world_to_grid_float(*pose[:2]))
    center = bool(geom.contains(i, j) and mask[j, i])
    fit = bool(center and footprint_check(mask, geom, tuple(pose[:3]), cfg))
    return dict(center_inside=center, footprint_inside=fit, pose_valid=True,
                entry_confirmed=fit, actual_pose=list(map(float, pose[:3])),
                target_identity='frozen_candidate_region_not_live_unobserved_mask')


def build_entry_catalog(bundle, cfg, audit_settings, settings, snapshot, zone,
                        region, visibility, footprint_check, cancel=None,
                        held_positions=(), *, entry_inset=.45, arrival_buffer=.20):
    """Sample distinct INTERNAL poses; Nav2 decides global route feasibility.

    A disconnected endpoint may still be proposed, exactly as Zone ENTRY probes:
    neither a straight-line collision nor a private graph vetoes all approaches.
    Unknown/lethal endpoint occupancy is never authorized for navigation.
    """
    started = time.monotonic()
    _, g, labels, known, _ = bundle
    validate_grids(labels, known, snapshot['cost']['values'], snapshot['observed'])
    if (not math.isfinite(entry_inset) or entry_inset <= 0 or
            not math.isfinite(arrival_buffer) or arrival_buffer < 0):
        raise ValueError('REGION_ENTRY_INVALID_INSET')
    pose = tuple(snapshot['pose'])
    if len(pose) < 3 or not np.isfinite(pose).all():
        raise ValueError('REGION_ENTRY_INVALID_START')
    cost = snapshot['cost']['values']
    ids = _indices(region['indices'], known.size)
    ids = ids[~snapshot['observed'].ravel()[ids]]
    target = np.zeros(known.shape, dtype=bool)
    target.ravel()[ids] = True
    if np.any(target & (~known | (labels != zone['mask_id']))):
        raise ValueError('REGION_ENTRY_TARGET_OUTSIDE_APPROVED_ZONE')
    stats = dict(residual_entry=True, direct_approach=False, catalog_complete=True,
                 positions_considered=0, endpoint_rejected=0, held_positions=0,
                 safe_positions=0, approach_candidates=0, entry_candidates=0,
                 zero_gain_positions=0, positive_views=0, unknown_cells_near_target=0,
                 zero_gain_motion_enabled=True, minimum_area_m2=settings.min_area_m2,
                 scope='bounded_sampled_interior_entry_positions_not_all_routes',
                 cost_key=snapshot['cost']['key'], start_xy=list(map(float, pose[:2])),
                 endpoint_policy='EXISTING_ZONE_ENTRY', navigation_authorized=False,
                 entry_inset_m=float(entry_inset), arrival_buffer_m=float(arrival_buffer),
                 observed_only_path_required=False, waypoint_distance_limit_m=None)
    def checkpoint():
        if cancel is not None and cancel.is_set():
            raise TimeoutError('CATALOG_CANCELED')
        if time.monotonic() - started > audit_settings.build_budget_s:
            raise TimeoutError('SECOND_PASS_CATALOG_BUILD_BUDGET')
    checkpoint()
    if not len(ids) or not components(target, g, settings.min_area_m2):
        return dict(views=[], stats=stats)
    # Unchanged ZonePlanner.maps()/internal_mask() endpoint geometry standards.
    free = known & (cost >= 0) & (cost < 100)
    clearance = ndimage.distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1,1:-1] * g.resolution
    required = float(cfg.robot_radius_m) + float(cfg.safety_margin_m) + math.sqrt(2) * g.resolution
    safe = free & (clearance >= required) & (cost < 99)
    depth = ndimage.distance_transform_edt(np.pad(target, 1, constant_values=False))[1:-1,1:-1] * g.resolution
    possible = target & safe & (depth >= entry_inset)
    stats['target_cells'] = int(target.sum())
    stats['unknown_cells_near_target'] = int((target & (cost < 0)).sum())
    stats['safe_interior_cells'] = int(possible.sum())
    stats['target_lethal_cells'] = int((target & (cost == 100)).sum())
    stats['target_inscribed_cells'] = int((target & (cost == 99)).sum())
    # Spatial tiles are ONLY proposal sampling, not connectivity or cut inference.
    tile = max(1, int(round(audit_settings.spacing_m / g.resolution)))
    wanted_depth = entry_inset + arrival_buffer
    yy, xx = np.nonzero(target)
    cx, cy = g.cell_to_world(float(np.mean(xx)), float(np.mean(yy)))
    groups = {}
    for j, i in zip(*np.nonzero(possible)):
        checkpoint()
        x, y = g.cell_to_world(int(i), int(j))
        rank = (depth[j,i] < wanted_depth, math.hypot(x-pose[0], y-pose[1]),
                -float(depth[j,i]), -float(clearance[j,i]), int(j), int(i))
        key = (int(j)//tile, int(i)//tile)
        if key not in groups or rank < groups[key][0]:
            groups[key] = (rank, int(i), int(j))
    sites = sorted(groups.values(), key=lambda p:p[0])
    if len(sites) > audit_settings.max_positions:
        stats['catalog_complete'] = False
        stats['omitted_positions'] = len(sites) - audit_settings.max_positions
        sites = sites[:audit_settings.max_positions]
    views = []
    for _, i, j in sites:
        checkpoint(); stats['positions_considered'] += 1
        x,y = g.cell_to_world(i,j)
        if any(math.dist((x,y), h) < settings.position_separation_m for h in held_positions):
            stats['held_positions'] += 1
            continue
        yaw = math.atan2(cy-y, cx-x)
        if not (footprint_check(free,g,(x,y,yaw),cfg) and
                footprint_check(target,g,(x,y,yaw),cfg)):
            stats['endpoint_rejected'] += 1
            continue
        # Keep separated poses (same spatial diversity as existing ENTRY).
        if any(math.hypot(x-v['x'], y-v['y']) < .4 for v in views):
            continue
        stats['safe_positions'] += 1
        # Zero information gain cannot veto entry. No visibility call is needed.
        candidate = dict(id=f'REGION_ENTRY_{zone["id"]}_{region["region_id"]}_{i}_{j}',
            zone_id=zone['id'], cell=[i,j], x=float(x), y=float(y), yaw=float(yaw),
            goal_kind='ENTRY', residual_entry=True, second_pass=True,
            region_investigation=True, audit_observation=True,
            audit_position_id=f'{region["task_id"]}:{i}:{j}',
            second_task_id=region['task_id'], residual_region_id=region['region_id'],
            residual_target_indices=ids.copy(), region_entry_indices=ids.copy(),
            residual_min_area_m2=float(settings.min_area_m2),
            residual_target_cells=len(ids), residual_overlap_cells=0,
            visible_indices=np.empty(0,dtype=np.int32), new_indices=np.empty(0,dtype=np.int32),
            new_cells=0, new_area_m2=0., score=0., path_cells=[],
            path_est_m=float(math.hypot(x-pose[0],y-pose[1])),
            clearance_lower_bound_m=float(clearance[j,i]-g.resolution/math.sqrt(2)),
            entry_boundary_depth_m=float(depth[j,i]),
            viewpoint_zone_mask_id=int(labels[j,i]), second_external_viewpoint=False,
            route_mode='REGION_INTERIOR_ENTRY_NAV2_PREFLIGHT',
            entry_target_xy=[float(x),float(y)],
            second_goal_margin_m=0., second_path_approved=False)
        views.append(candidate)
    stats['approach_candidates'] = stats['entry_candidates'] = len(views)
    stats['zero_gain_positions'] = len(views)
    stats['elapsed_s'] = time.monotonic()-started
    return dict(views=views, stats=stats)


def refresh_entry(candidate, observed, geom):
    """Keep an internal goal only while its target has a significant remainder.

    Never introduce a predicted-view-gain or measured-free transit requirement.
    The frozen entry mask is retained to evaluate actual arrival independently
    from observation acquired while moving toward the goal.
    """
    c = dict(candidate)
    if c.get('goal_kind') != 'ENTRY' or not c.get('residual_entry'):
        raise ValueError('REGION_ENTRY_BAD_CANDIDATE_KIND')
    if observed.shape != (geom.height,geom.width) or observed.dtype != bool:
        raise ValueError('REGION_ENTRY_BAD_OBSERVED_GRID')
    ids = _indices(c['residual_target_indices'], observed.size)
    remaining = ids[~observed.ravel()[ids]]
    mask = np.zeros(observed.shape,dtype=bool);mask.ravel()[remaining] = True
    parts = components(mask,geom,float(c['residual_min_area_m2']))
    if not parts:return None
    i,j = map(math.floor,geom.world_to_grid_float(float(c['x']),float(c['y'])))
    if not geom.contains(i,j):return None
    index=j*geom.width+i
    selected = next((r for r in parts if index in r['indices']),None)
    if selected is None:return None
    # Preserve frozen entry indices; parent coverage masks are never modified.
    c.update(residual_target_indices=selected['indices'].copy(),
             residual_target_cells=int(selected['cells']),
             visible_indices=np.empty(0,dtype=np.int32),new_indices=np.empty(0,dtype=np.int32),
             new_cells=0,new_area_m2=0.,residual_overlap_cells=0)
    return c


def validate_entry_endpoint(candidate,bundle,inp,cfg,footprint_check):
    """Additional target-membership check AFTER original Nav2 path validation.

    Uses the baseline padded rectangle, not second-pass all-heading extra disk.
    Does not authorize motion by itself or test an invented straight path.
    """
    _,g,labels,known,_ = bundle
    validate_grids(labels,known,inp['cost']['values'],inp['observed'])
    pose=tuple(float(candidate[k]) for k in ('x','y','yaw'))
    membership=entry_membership(candidate,pose,g,cfg,footprint_check)
    if not membership.get('entry_confirmed'):
        raise ValueError('REGION_ENTRY_GOAL_NOT_INSIDE_TARGET')
    i,j=map(math.floor,g.world_to_grid_float(*pose[:2]))
    cost=inp['cost']['values']
    if not known[j,i] or not 0<=cost[j,i]<99:
        raise ValueError('GOAL_FOOTPRINT_BLOCKED')
    if not footprint_check(known & (cost>=0) & (cost<100),g,pose,cfg):
        raise ValueError('GOAL_FOOTPRINT_BLOCKED')
    return dict(goal_xy=list(pose[:2]),goal_kind='ENTRY',residual_entry=True,
                goal_inside_frozen_target=True,baseline_footprint_checked=True,
                full_nav2_path_check_owned_by_zone_node=True,
                nav2_runtime_collision_checks_unchanged=True,
                extra_second_pass_rotation_sweep=False)
