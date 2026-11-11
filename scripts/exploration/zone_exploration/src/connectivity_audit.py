"""Bounded observation lattice and connectivity evidence. No ROS/no route planner.

Connected-component labelling only stratifies target samples and diagnoses a
known occupied cut. Every executable route/probe is requested from Nav2.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import time
import numpy as np
from scipy import ndimage
from planning_core import oriented_footprint_clear


@dataclass(frozen=True)
class AuditSettings:
    spacing_m: float = .60
    headings: int = 8
    max_positions: int = 1200
    max_probes: int = 1000
    build_budget_s: float = 30.
    zone_budget_s: float = 600.
    max_restarts: int = 2

    def validate(self):
        for v in (self.spacing_m, self.build_budget_s, self.zone_budget_s):
            if not math.isfinite(v) or v <= 0:
                raise ValueError('Invalid audit geometry/time limit')
        if not .1 <= self.spacing_m <= 1.0 or not 1 <= self.headings <= 16:
            raise ValueError('Invalid audit lattice')
        if not 1 <= self.max_positions <= 4000 or not 1 <= self.max_probes <= 4000:
            raise ValueError('Invalid audit bound')
        if not 0 <= self.max_restarts <= 10:
            raise ValueError('Invalid audit restart limit')


@dataclass(frozen=True)
class EntryAccessSettings:
    max_probes: int = 6
    per_component: int = 2
    min_separation_m: float = .80
    failed_radius_m: float = .50

    def validate(self):
        if (type(self.max_probes) is not int or not 1 <= self.max_probes <= 24
                or type(self.per_component) is not int or not 1 <= self.per_component <= 8):
            raise ValueError('INVALID_ENTRY_ACCESS_PROBE_BOUNDS')
        if (not all(math.isfinite(float(x)) for x in (self.min_separation_m,self.failed_radius_m))
                or not .15 <= self.min_separation_m <= 3.
                or not .10 <= self.failed_radius_m <= 2.):
            raise ValueError('INVALID_ENTRY_ACCESS_DISTANCES')

def sensor_signature(sensor):
    return tuple(sorted(vars(sensor).items()))


def try_position_history(candidate, history):
    # Smaller than the baseline coarse duplicate radius: an actual new view is
    # not dropped merely because it is close to a previously unhelpful view.
    return any(p.get('zone_id') == candidate['zone_id'] and
               math.hypot(float(p['x'])-candidate['x'], float(p['y'])-candidate['y']) <= .08 and
               abs(math.atan2(math.sin(float(p['yaw'])-candidate['yaw']),
                              math.cos(float(p['yaw'])-candidate['yaw']))) <= .10
               for p in history)


def residual_counts(labels, known, cost, observed, zone):
    target = (labels == zone['mask_id']) & known
    remaining = target & ~observed
    return {
        'reference_cells': int(target.sum()),
        'remaining_cells': int(remaining.sum()),
        # Unknown anywhere in the assessed zone prevents a normal limit verdict.
        'unknown_cells': int((target & (cost < 0)).sum()),
        'unobserved_free_cells': int((remaining & (cost >= 0) & (cost < 100)).sum()),
        'unobserved_occupied_cells': int((remaining & (cost == 100)).sum()),
    }


def optimistic_reachability(cost, geom, start, positions):
    """Diagnostic only. Unknown is allowed; only lethal cells block this graph.

    If this optimistic 8-neighbour point graph is disconnected, the snapshot
    contains an occupied cut. It does NOT authorize motion or prove permanent
    physical inaccessibility. This is not a detour/path planner.
    """
    optimistic = cost < 100
    components, _ = ndimage.label(optimistic, structure=np.ones((3, 3), dtype=np.uint8))
    i, j = map(math.floor, geom.world_to_grid_float(*start[:2]))
    source = int(components[j, i]) if geom.contains(i, j) else 0
    return {p['position_id']: bool(source and components[p['cell'][1], p['cell'][0]] == source)
            for p in positions}


def build_catalog(bundle, cfg, settings, snapshot, zone, visibility, cancel):
    """Enumerate every occupied lattice tile of EACH local free component.

    No arbitrary shortlist is declared exhaustive. Budget exhaustion raises and
    cannot produce a limited-completion certificate. A tile without a feasible
    padded footprint is recorded after all its cells/headings are checked.
    Components with no feasible robot pose are outside the navigation domain,
    not erased from Coverage: their unobserved free cells still block completion.
    """
    settings.validate()
    started = time.monotonic()
    def checkpoint():
        if cancel.is_set():
            raise RuntimeError('PLANNER_CANCELLED')
        if time.monotonic()-started > settings.build_budget_s:
            raise TimeoutError('AUDIT_BUILD_BUDGET_UNVERIFIED')
    _, geom, labels, known, signature = bundle
    cost = snapshot['cost']['values']
    target = (labels == zone['mask_id']) & known
    centre = target & (cost >= 0) & (cost < 99)
    components, ncomp = ndimage.label(centre)  # 4-neighbour, conservative segmentation
    free = known & (cost >= 0) & (cost < 100)
    clearance = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]*geom.resolution
    tile = max(1, int(math.floor(settings.spacing_m/geom.resolution+1e-9)))
    groups = {}
    ys, xs = np.nonzero(centre)
    for j, i in zip(ys.tolist(), xs.tolist()):
        groups.setdefault((int(components[j,i]), j//tile, i//tile), []).append((i,j))
    if len(groups) > settings.max_positions:
        raise TimeoutError('AUDIT_POSITION_BUDGET_UNVERIFIED')
    positions = []
    views = []
    represented = set()
    empty_tiles = 0
    sensor = snapshot['sensor']
    for (component, _, _), cells in sorted(groups.items()):
        checkpoint()
        cells.sort(key=lambda ij: (-float(clearance[ij[1],ij[0]]), ij[1], ij[0]))
        selected = None
        valid_yaws = []
        for i,j in cells:
            checkpoint()
            x,y = geom.cell_to_world(i,j)
            yaws = []
            for k in range(settings.headings):
                yaw = math.atan2(math.sin(2*math.pi*k/settings.headings),
                                 math.cos(2*math.pi*k/settings.headings))
                if (oriented_footprint_clear(free, geom, (x,y,yaw), cfg)
                        and oriented_footprint_clear(target, geom, (x,y,yaw), cfg)):
                    yaws.append(yaw)
            if yaws:
                selected=(i,j,x,y); valid_yaws=yaws; break
        if selected is None:
            empty_tiles += 1
            continue
        i,j,x,y=selected
        pid=f'{zone["id"]}:{i}:{j}'
        represented.add(component)
        local_views=[]
        for yaw in valid_yaws:
            checkpoint()
            visible = visibility(geom, free, target, x, y, yaw, sensor, cfg.max_range_m)
            novel = visible[~snapshot['observed'].ravel()[visible]]
            c=dict(id=f'AUDIT_{zone["id"]}_{i}_{j}_{yaw:.6f}',
                   zone_id=zone['id'], cell=[int(i),int(j)], x=float(x), y=float(y), yaw=float(yaw),
                   goal_kind='OBSERVE', path_est_m=float(math.hypot(x-snapshot['pose'][0],y-snapshot['pose'][1])),
                   clearance_lower_bound_m=float(clearance[j,i]-geom.resolution/math.sqrt(2)),
                   viewpoint_zone_mask_id=int(zone['mask_id']), route_mode='NAV2_GLOBAL_PROBE',
                   visible_indices=np.asarray(visible,dtype=np.int32), new_indices=np.asarray(novel,dtype=np.int32),
                   new_cells=int(len(novel)), new_area_m2=float(len(novel)*geom.resolution**2),
                   score=0., path_cells=[], audit_observation=True,
                   audit_position_id=pid, audit_component=int(component))
            local_views.append(c)
        local_views.sort(key=lambda c:(-c['new_cells'],c['yaw']))
        positions.append(dict(position_id=pid, cell=[i,j], component=int(component), probe=local_views[0]))
        views.extend(local_views)
    checkpoint()
    return dict(zone_id=zone['id'], catalog_complete=True, positions=positions, views=views,
                components=int(len(represented)), unrepresented_components=0,
                geometric_free_components=int(ncomp), no_robot_pose_components=int(ncomp-len(represented)),
                no_valid_pose_tiles=empty_tiles, tile_cells=tile,
                lattice_spacing_m=tile*geom.resolution, heading_samples=settings.headings,
                cost_key=snapshot['cost']['key'], bundle_signature=signature,
                sensor_signature=sensor_signature(sensor), epoch=snapshot['epoch'],
                start_pose=list(snapshot['pose']),
                optimistic_connected=optimistic_reachability(cost,geom,snapshot['pose'],positions),
                elapsed_s=time.monotonic()-started)


def entry_probe_sites(catalog: dict, failed: list[dict], settings: EntryAccessSettings):
    """Pick diverse goal *positions* from each local zone-free component.

    Nav2 decides actual global connectivity, including paths outside the Zone.
    If fewer samples are selected than exist, exhaustion is still UNVERIFIED.
    The selection does not treat local connected components as a route planner.
    """
    settings.validate()
    if not catalog.get('catalog_complete'):
        raise ValueError('INCOMPLETE_CATALOG_CANNOT_SELECT_ENTRY_SITES')
    failed_xy = [(float(p['x']),float(p['y'])) for p in failed
                 if isinstance(p,dict) and 'x' in p and 'y' in p
                 and math.isfinite(float(p['x'])) and math.isfinite(float(p['y']))]
    groups = {}
    for pos in catalog['positions']:
        c = pos['probe']
        x,y = float(c['x']),float(c['y'])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if any(math.hypot(x-a,y-b) < settings.failed_radius_m for a,b in failed_xy):
            continue
        groups.setdefault(int(pos['component']),[]).append(pos)
    selected=[]; counts={key:0 for key in groups}
    # Round robin: distinct components get a chance before repeated samples.
    while len(selected) < settings.max_probes:
        made=False
        for comp, poses in sorted(groups.items()):
            if counts[comp] >= settings.per_component:
                continue
            alternatives=[]
            for p in poses:
                c=p['probe']; x,y=float(c['x']),float(c['y'])
                if any(math.hypot(x-f['probe']['x'],y-f['probe']['y']) < settings.min_separation_m
                       for f in selected):
                    continue
                # Prefer samples far from known failed endpoints and previous probes.
                # This aims at distinct possible entry approaches, not the same doorway.
                references = failed_xy + [(f['probe']['x'],f['probe']['y']) for f in selected]
                distance = min((math.hypot(x-rx,y-ry) for rx,ry in references),default=1e6)
                alternatives.append((distance,p))
            if not alternatives:
                continue
            alternatives.sort(key=lambda it:(-it[0],it[1]['position_id']))
            best=alternatives[0][1]
            selected.append(best)
            counts[comp]+=1
            made=True
            if len(selected)>=settings.max_probes:break
        if not made:break
    return selected

def useful_untried(catalog, observed, history, geom, cfg):
    available=[]
    remaining=0
    for original in catalog['views']:
        ids=original['visible_indices']
        novel=ids[~observed.ravel()[ids]]
        if not len(novel):
            continue
        remaining += 1
        if try_position_history(original, history):
            continue
        c=dict(original, new_indices=novel, new_cells=int(len(novel)),
               new_area_m2=float(len(novel)*geom.resolution**2))
        c['score']=c['new_area_m2']/(c['path_est_m']+cfg.distance_regularizer_m)
        available.append(c)
    available.sort(key=lambda c:(-c['score'],-c['new_cells'],c['id']))
    return available, remaining


def assess_catalog(catalog, probes, snapshot, bundle, zone):
    """Only a complete/fresh/all-reachable lattice may produce NORMAL_LIMITED."""
    _, geom, labels, known, signature=bundle
    candidates, remaining=useful_untried(catalog,snapshot['observed'],[],geom,
                                      type('Rank',(),{'distance_regularizer_m':2.})())
    counts=residual_counts(labels,known,snapshot['cost']['values'],snapshot['observed'],zone)
    ids=[p['position_id'] for p in catalog['positions']]
    successes=sum(probes.get(i,{}).get('success') is True for i in ids)
    failed=sum(probes.get(i,{}).get('success') is False for i in ids)
    match=(catalog['cost_key']==snapshot['cost']['key'] and catalog['epoch']==snapshot['epoch']
           and catalog['sensor_signature']==sensor_signature(snapshot['sensor']))
    e=dict(schema='zone_connectivity_evidence_v1',catalog_complete=catalog['catalog_complete'],
           inputs_fresh=True,context_matches=bool(match),
           connectivity_verified=bool(ids) and successes==len(ids),
           all_potential_information_accounted=(remaining==0 and counts['unobserved_free_cells']==0
                                               and counts['unknown_cells']==0),
           positions=len(ids),probes_succeeded=successes,probes_failed=failed,
           unrepresented_components=catalog['unrepresented_components'],
           components=catalog['components'],remaining_gain_views=remaining,
           no_robot_pose_components=catalog.get('no_robot_pose_components',0),
           cost_key=snapshot['cost']['key'],bundle_signature=signature,
           lattice_spacing_m=catalog['lattice_spacing_m'],heading_samples=catalog['heading_samples'],
           scope='finite sampled J100 poses; current approved map and live costmap; Nav2 global routes',
           **counts)
    from zone_policy import limited_certificate_ok
    if limited_certificate_ok(e):
        return 'NORMAL_LIMITED', e
    blocked=[p for p in catalog['positions']
             if probes.get(p['position_id'],{}).get('success') is False
             and probes[p['position_id']].get('reason')=='NAVFN_ABORTED'
             and not catalog['optimistic_connected'].get(p['position_id'],True)]
    if blocked:
        return 'MAP_BLOCKED_CANDIDATE', e
    if failed or successes != len(ids) or not match or e['unrepresented_components']:
        return 'CONNECTIVITY_UNVERIFIED', e
    return 'OBSERVATION_UNRESOLVED', e
