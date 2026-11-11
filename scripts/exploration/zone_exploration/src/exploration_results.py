"""DN012 unified blockage results and robot-scoped current-map accessibility.

Reroute and obstruction points are ONE public category. A point never fabricates
an inaccessible polygon. Global and local connectivity are evaluated separately.
No Nav2 parameters, maps, goals, velocity commands, or keepout filters are changed.
"""
from __future__ import annotations
import copy
import hashlib
import math
import numpy as np
from scipy import ndimage
from checkpoint import MIN_AREA_M2, plain, encode

FOUR = ndimage.generate_binary_structure(2, 1)
EIGHT = np.ones((3, 3), dtype=np.uint8)


def _xy(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        result = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(v) for v in result) else None


def locate(bundle, point):
    if point is None:
        return None
    _, g, labels, _, _ = bundle
    i, j = map(math.floor, g.world_to_grid_float(*point))
    if not g.contains(i, j):
        return None
    ident = int(labels[j, i])
    return next((z['id'] for z in bundle[0]['zones'] if z['mask_id'] == ident), None)


class BlockageRegistry:
    """One collection; source types are provenance, not separate public layers.

    Cluster radius is only for displaying coincident POINT observations, never
    an obstacle radius. No geometry is inferred from closeness between points.
    """
    def __init__(self, bundle, records=(), merge_radius_m=.20):
        self.bundle = bundle
        self.radius = float(merge_radius_m)
        self.records = copy.deepcopy(list(records))

    def observe(self, kind, key, point, evidence, destination=None, stage='RECORDED', historical=False):
        point = _xy(point)
        if point is None:
            return
        zone = locate(self.bundle, point)
        token = str(kind) + ':' + str(key)
        record = next((r for r in self.records if token in r.get('evidence_keys', [])), None)
        if record is None:
            record = next((r for r in self.records if r.get('geometry_kind') == 'POINT'
                           and zone is not None and r.get('spatial_zone') == zone
                           and math.dist(r['xy'], point) <= self.radius), None)
        if record is None:
            record = dict(id='B-' + hashlib.sha256(token.encode()).hexdigest()[:16],
                          display_label='차단지역', geometry_kind='POINT', xy=point,
                          spatial_zone=zone, evidence_keys=[], observations=[], sources=[],
                          destination_zones=[], historical=bool(historical))
            self.records.append(record)
        if token not in record['evidence_keys']:
            record['evidence_keys'].append(token)
        if kind not in record['sources']:
            record['sources'].append(kind)
        if destination and destination not in record['destination_zones']:
            record['destination_zones'].append(destination)
        item = plain(dict(key=token, source=kind, stage=stage, xy=point,
                          destination_zone=destination, evidence=evidence,
                          historical=bool(historical)))
        # Same first-detection/confirmation is not duplicated on every report.
        ident = hashlib.sha256(encode(item)).hexdigest()
        if not any(x.get('snapshot_sha256') == ident for x in record['observations']):
            record['observations'].append(dict(item, snapshot_sha256=ident))
        record['historical'] = record['historical'] and bool(historical)
        record['latest_xy'] = point
        record['geometry_semantics'] = 'POINT_ONLY; no area or passage line inferred'

    def event(self, event, fields):
        if event == 'PATH_BLOCKED_EVIDENCE':
            e = fields.get('evidence') or {}
            self.observe('PATH_BLOCKAGE', str(fields.get('goal_id')) + ':' + str(e.get('evidence_ros_time')),
                         e.get('xy'), e, e.get('zone'), 'DETECTED')
        elif event == 'REROUTE_POINT_RECORDED':
            self.observe('REROUTE', fields.get('record_id'), fields.get('point_xy'), fields,
                         fields.get('destination_zone') or fields.get('zone'),
                         'CONFIRMED' if fields.get('plan_change_confirmed') else 'PENDING')

    def absorb_summary(self, summary):
        region = summary.get('region_review') or {}
        memory = region.get('blockage_memory') or {}
        points = memory.get('points') or (region.get('map_layers') or {}).get('blockage_memory_points') or []
        for record in points:
            self.observe('PATH_BLOCKAGE', record.get('id'), record.get('xy'), record,
                         record.get('destination_zone'), 'HISTORICAL_WITNESS', True)
        for record in summary.get('path_blockage_cancellations') or []:
            e = record.get('evidence') or {}
            self.observe('PATH_BLOCKAGE', str(record.get('goal_id')) + ':' + str(e.get('evidence_ros_time')),
                         e.get('xy'), record, e.get('zone'), 'STOP_CONFIRMED', True)
        reroute = summary.get('reroute_memory_v1_4') or {}
        for record in reroute.get('points') or []:
            self.observe('REROUTE', record.get('id'), record.get('point_xy'), record,
                         record.get('destination_zone') or record.get('zone'),
                         'CONFIRMED' if record.get('plan_change_confirmed') else 'PENDING', True)
        evidence = reroute.get('evidence_v1_7') or {}
        for record in evidence.get('records') or []:
            for phase, key in (('PENDING', 'first_detection'), ('CONFIRMED', 'confirmation')):
                snap = record.get(key)
                if snap:
                    self.observe('REROUTE', record.get('record_id'), snap.get('point_xy'), snap,
                                 snap.get('destination_zone'), phase, True)

    def public(self):
        return copy.deepcopy(self.records)


def _body_domain(free, resolution, radius):
    # An optimistic lower-bound collision test across each whole center cell.
    # Connectivity in this graph is not motion authorization. Disconnection is
    # decisive within the declared fixed navigation domain and footprint margin.
    d = ndimage.distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]
    return free & (d * resolution + resolution / math.sqrt(2.) >= radius - 1e-10)


def assess_access(bundle, snapshot, zone, persistent_blockers, cfg, settings,
                  fixed_prior_topology, endpoint_checker=None):
    from regions import components, public_region, row_runs, zone_polygon_domain
    data, g, labels, known, signature = bundle
    cost = np.asarray(snapshot['cost']['values'])
    out = dict(zone_id=zone['id'], scope='FIXED_REFERENCE_MAP_INTERIOR_WITH_FOOTPRINT_MARGIN',
               basis='CURRENT_VALIDATED_COSTMAP_AND_FIXED_PRIOR',
               fixed_prior_topology=bool(fixed_prior_topology),
               area_semantics='reference_free_robot_center_subset_m2',
               snapshot_cost_key=snapshot['cost']['key'], bundle_signature=signature,
               global_regions=[], local_regions=[], global_omitted_regions=0,
               local_omitted_regions=0, baseline_unverified_cells=0,
               full_accessibility_verified=False, min_area_m2=MIN_AREA_M2,
               unmapped_exterior_detours_allowed=False,
               unknown_cells_are_not_obstacles=True,
               safety_margin_used_for_robot_operational_access=True,
               assessment_is_motion_authorization=False)
    if not fixed_prior_topology:
        return dict(out, status='ACCESS_NOT_ASSESSED', reason='FIXED_PRIOR_NOT_ENABLED')
    if cost.shape != known.shape or not np.isfinite(cost).all() or not ((cost >= -1) & (cost <= 100)).all():
        return dict(out, status='ACCESS_NOT_ASSESSED', reason='INVALID_CURRENT_COSTMAP')
    pose = snapshot.get('pose')
    if pose is None or not np.isfinite(pose).all():
        return dict(out, status='ACCESS_NOT_ASSESSED', reason='INVALID_START_POSE')
    i, j = map(math.floor, g.world_to_grid_float(*pose[:2]))
    if not g.contains(i, j):
        return dict(out, status='ACCESS_NOT_ASSESSED', reason='START_OUTSIDE_DOMAIN')
    # J100 rectangle matches the existing oriented_footprint_clear scaling.
    scale = cfg.robot_radius_m / math.hypot(.229, .208)
    radius = min(.229, .208) * scale + cfg.safety_margin_m
    target = (labels == zone['mask_id']) & known
    reference = snapshot.get('reference_occupancy')
    if reference is not None:
        reference = np.asarray(reference)
        if reference.shape != known.shape or not np.isin(reference, [-1, 0, 100]).all():
            return dict(out, status='ACCESS_NOT_ASSESSED', reason='INVALID_VERIFIED_REFERENCE')
        # Do not confuse the white-only Coverage denominator with Nav2's domain.
        # Unknown INSIDE the map stays optimistically open; there is no virtual
        # outside halo. A positive connection still needs a real Nav2 path.
        prior_domain = reference != 100
        out['prior_domain_source'] = 'HASH_VERIFIED_OCCUPANCY'
    else:
        prior_domain = known
        out['prior_domain_source'] = 'APPROVED_FREE_FALLBACK'
    free = prior_domain & (cost != 100)  # inflation 1..99 is NOT a measured obstacle
    domain = _body_domain(free, g.resolution, radius)
    out.update(robot_half_extents_m=[.229 * scale, .208 * scale],
               safety_margin_m=float(cfg.safety_margin_m), inscribed_body_radius_m=float(radius),
               unknown_current_cells=int((target & (cost < 0)).sum()))
    if not domain[j, i] or (endpoint_checker is not None and not endpoint_checker(free, g, pose, cfg)):
        return dict(out, status='ACCESS_NOT_ASSESSED', reason='START_FOOTPRINT_UNSAFE')
    lab, _ = ndimage.label(domain, EIGHT)
    reachable = lab == lab[j, i]
    global_cut = target & domain & ~reachable
    polygon, local_scope = zone_polygon_domain(zone, g, target)
    out['local_scope'] = local_scope
    local_cut = np.zeros_like(known)
    if polygon[j, i]:
        local_lab, _ = ndimage.label(domain & polygon, EIGHT)
        if local_lab[j, i]:
            local_cut = target & domain & (local_lab != local_lab[j, i]) & reachable
    for scope, mask in (('global', global_cut), ('local', local_cut)):
        for region in components(mask, g, MIN_AREA_M2):
            r = public_region(region)
            r.update(cell_runs_ros=row_runs(region['indices'], g.width), scope=scope,
                     access_status='INACCESSIBLE' if scope == 'global' else 'LOCAL_BLOCKED_GLOBAL_DETOUR_POSSIBLE',
                     classification='ROBOT_GLOBAL_INACCESSIBLE' if scope == 'global' else 'ROBOT_LOCAL_BLOCKED',
                     display_label='접근불가 영역' if scope == 'global' else '차단지역',
                     global_route_status='NO_ROUTE_IN_COMPLETE_BODY_LOWER_BOUND_GRAPH' if scope == 'global'
                                         else 'GLOBAL_CONNECTION_REQUIRES_NAV2_ROUTE_VALIDATION',
                     persistent_sensor_supported_obstacle_cells=int(np.asarray(persistent_blockers).sum()),
                     costmap_lethal_cells=int((known & (cost == 100)).sum()),
                     snapshot_cost_key=snapshot['cost']['key'], bundle_signature=signature)
            out[scope + '_regions'].append(r)
    fitting_possible = target & domain
    if target.sum() * g.resolution**2 < MIN_AREA_M2:
        verdict = 'BELOW_SIGNIFICANT_AREA'
    elif not fitting_possible.any():
        verdict = 'INACCESSIBLE'
        out['reason'] = 'NO_POSSIBLE_ROBOT_CENTER_IN_ZONE'
    elif not (fitting_possible & reachable).any():
        verdict = 'INACCESSIBLE'
        out['reason'] = 'NO_GLOBAL_ROBOT_ROUTE'
    elif out['global_regions']:
        verdict = 'PARTIALLY_INACCESSIBLE'
    elif out['local_regions']:
        verdict = 'LOCAL_BLOCKED_GLOBAL_CONNECTION'
    else:
        verdict = 'NO_DISCONNECTION_IN_CURRENT_MODEL'
    out.update(status=verdict, zone_access_status=verdict,
               connectivity_evaluation_complete=True,
               disclaimer='Current fixed-map robot assessment; connected graph alone never authorizes motion')
    return out


CATEGORY_LABELS = {
    'inaccessible_zones': '접근 불가 Zone',
    'inaccessible_regions': '접근 불가 영역',
    'blocked_points': '차단 지점들',
    'robot_unexplorable_regions': '로봇 탐사 불가 영역',
}


def _point(value):
    if not isinstance(value,(list,tuple)) or len(value)!=2:return None
    try:p=[float(x) for x in value]
    except (TypeError,ValueError,OverflowError):return None
    return p if all(math.isfinite(x) for x in p) else None


def _legacy_project_results_v16(region_report, zone_records, *, path_records=(),
                    reroute_summary=None, owned_goal_ids=()):
    """Map EXISTING affirmative outputs. Unfinished work is metadata, not class 5."""
    groups={key:[] for key in CATEGORY_LABELS}
    owned=set(owned_goal_ids)
    zone_records=zone_records or {}
    for zid,r in sorted(zone_records.items()):
        a=r.get('assessment') or {};e=a.get('evidence') or {}
        # Exact affirmative field produced by existing connectivity machinery.
        if e.get('current_map_unreachable') is True and not r.get('entered'):
            groups['inaccessible_zones'].append(dict(zone_id=zid,
                reason=a.get('classification') or r.get('deferred_reason'),
                basis='existing_current_map_unreachable',assessment=copy.deepcopy(a),
                scope='existing_fixed_prior_current_grid_robot_assessment',
                coverage_percent=r.get('measured_percent')))
    layers=region_report.get('map_layers') or {}
    for r in layers.get('blocked_regions') or []:
        # A purely local cut can have a global detour. Keep it in raw diagnostics;
        # do not relabel it as global area inaccessibility or invent a barrier XY.
        if (r.get('scope')=='global' and
                r.get('classification')=='SENSOR_SUPPORTED_GLOBAL_CUT'):
            out=copy.deepcopy(r)
            out.update(basis='existing_sensor_supported_global_region',
                       display_label=CATEGORY_LABELS['inaccessible_regions'])
            groups['inaccessible_regions'].append(out)
    pending=[]
    for t in region_report.get('tasks') or []:
        if t.get('status')=='ROBOT_EXPLORATION_UNAVAILABLE':
            entry=t.get('assessment_subject')=='REGION_ENTRY'
            key=('inaccessible_regions' if entry and not t.get('entry_confirmed')
                 else 'robot_unexplorable_regions')
            groups[key].append(dict(task_id=t['task_id'],zone_id=t['zone_id'],
                area_m2=t['area_m2'],cell_runs_ros=copy.deepcopy(t.get('cell_runs_ros',[])),
                centroid_map_m=copy.deepcopy(t.get('centroid_map_m')),
                bbox_cells_ros=copy.deepcopy(t.get('bbox_cells_ros')),
                reason=t['reason'],source_status=t['status'],checks=t.get('checks',0),
                nav_sent=t.get('nav_sent',0),entry_confirmed=bool(t.get('entry_confirmed')),
                basis='existing_bounded_robot_policy_result',
                scope=t.get('completion_scope','bounded_sampled_policy'),
                physical_blockage_proven=False,
                display_label=CATEGORY_LABELS[key],
                evidence_is_historical=bool(t.get('evidence_is_historical',False))))
        elif t.get('status') not in ('OBSERVED','SMALL_REMAINDER','SPLIT'):
            pending.append(dict(task_id=t.get('task_id'),zone_id=t.get('zone_id'),
                                status=t.get('status'),reason=t.get('reason')))
    current_witnesses={}
    for record in path_records:
        e=record.get('evidence') or {};p=_point(e.get('xy'))
        if p is not None:
            current_witnesses[(tuple(p),e.get('path_sha256'))]=record
    seen=set()
    for r in layers.get('blockage_memory_points') or []:
        p=_point(r.get('xy'))
        if p is None:continue
        # Reuse affirmative legacy witness fields, not cost/scan guesses.
        if not (r.get('scan_supported') and r.get('persistent') and
                r.get('translation_stalled') and r.get('detected_cost')==100):continue
        ident=('path',r.get('id') or tuple(p))
        if ident in seen:continue
        seen.add(ident)
        current=(tuple(p),r.get('path_sha256')) in current_witnesses
        out=copy.deepcopy(r)
        out.update(xy=p,source='PATH_BLOCKAGE',historical=not current,
                   location_basis='sensor_supported_obstacle_cell',
                   display_label=CATEGORY_LABELS['blocked_points'])
        groups['blocked_points'].append(out)
    # Current confirmed monitor records remain visible when optional memory is off.
    for (xy,sha),record in current_witnesses.items():
        e=record['evidence']
        if not (e.get('scan_supported') and e.get('persistent') and
                e.get('translation_stalled') and e.get('cost')==100):continue
        if any(p.get('source')=='PATH_BLOCKAGE' and p.get('xy')==list(xy)
               and p.get('path_sha256')==sha for p in groups['blocked_points']):continue
        groups['blocked_points'].append(dict(id=record.get('goal_id'),xy=list(xy),
            source='PATH_BLOCKAGE',destination_zone=e.get('zone'),path_sha256=sha,
            historical=False,location_basis='sensor_supported_obstacle_cell',
            evidence=copy.deepcopy(e),display_label=CATEGORY_LABELS['blocked_points']))
    # Preserve the established confirmed reroute records, including estimation
    # provenance. PENDING is not promoted; mask_eligible is NOT used as a verdict.
    for r in (reroute_summary or {}).get('points') or []:
        if r.get('plan_change_confirmed') is not True:continue
        p=_point(r.get('point_xy'))
        if p is None:continue
        ident=('reroute',r.get('id') or tuple(p))
        if ident in seen:continue
        seen.add(ident)
        out=copy.deepcopy(r)
        out.update(xy=p,source='REROUTE',historical=r.get('goal_id') not in owned,
                   location_basis=r.get('correction_source'),
                   basis='existing_confirmed_plan_change_record',
                   physical_blockage_proven=False,
                   display_label=CATEGORY_LABELS['blocked_points'])
        groups['blocked_points'].append(out)
    return dict(schema='dn012_four_results_v1_6',version='DN012_REGION_ENTRY_V1_6',
        categories=groups,category_labels=dict(CATEGORY_LABELS),
        counts={k:len(v) for k,v in groups.items()},
        geometry=copy.deepcopy(region_report.get('geometry')),
        evaluation=dict(incomplete_tasks=pending,not_a_fifth_result_category=True,
                        small_cells_by_zone=copy.deepcopy(region_report.get('small_cells_by_zone',{})),
                        raw_coverage_unchanged=True,detector_thresholds_changed=False,
                        historical_points_count=sum(bool(p.get('historical')) for p in groups['blocked_points']),
                        point_records_are_not_barrier_shapes=True),
        classification_note='Names project existing robot-scoped assessments; no new physical-cut inference.')

def _project_results_with_baseline(legacy, region_report, zone_records, *, path_records=(), reroute_summary=None, owned_goal_ids=()):
    result = legacy(region_report, zone_records, path_records=path_records,
                    reroute_summary=reroute_summary, owned_goal_ids=owned_goal_ids)
    result['version'] = 'DN012_UNIFIED_RESUME_V1_8'
    groups = result['categories']
    result['category_labels']['blocked_points'] = '차단지역'
    # Existing point sources retain raw provenance, but no separate reroute category.
    for p in groups['blocked_points']:
        p['display_label'] = '차단지역'
    known_points = {(p.get('source'), p.get('id')) for p in groups['blocked_points']}
    for r in (reroute_summary or {}).get('points') or []:
        if ('REROUTE', r.get('id')) in known_points:
            continue
        point = _xy(r.get('point_xy'))
        if point is not None:
            groups['blocked_points'].append(dict(r, xy=point, source='REROUTE',
                display_label='차단지역', detection_stage='CONFIRMED' if r.get('plan_change_confirmed') else 'PENDING',
                geometry_kind='POINT', geometry_semantics='point_only_no_region_inference'))
    evidence = (reroute_summary or {}).get('evidence_v1_7') or {}
    known_reroutes = {p.get('id') for p in groups['blocked_points'] if p.get('source') == 'REROUTE'}
    for record in evidence.get('records') or []:
        if record.get('record_id') in known_reroutes:continue
        snap = record.get('confirmation') or record.get('first_detection') or {}
        point = _xy(snap.get('point_xy'))
        if point is None:continue
        groups['blocked_points'].append(dict(id=record.get('record_id'), xy=point,
            source='REROUTE', display_label='차단지역', geometry_kind='POINT',
            detection_stage=record.get('status'), evidence=copy.deepcopy(record),
            geometry_semantics='point_only_no_region_inference'))
    # A new complete assessment supersedes old Zone/region verdicts, without
    # deleting the original history retained by the mission checkpoint/logs.
    fresh_zones = {zid for zid, rec in (region_report.get('zones') or {}).items()
        if not rec.get('evidence_is_historical') and
        (rec.get('access') or {}).get('connectivity_evaluation_complete')}
    result['superseded_historical_access'] = {
        key:[r for r in groups[key] if r.get('zone_id') in fresh_zones]
        for key in ('inaccessible_zones','inaccessible_regions')}
    for key in ('inaccessible_zones','inaccessible_regions'):
        groups[key] = [dict(r, historical=True) for r in groups[key]
                       if r.get('zone_id') not in fresh_zones]
    areas = []
    known_region = {(r.get('zone_id'), r.get('region_id')) for r in groups['inaccessible_regions']}
    known_zone = {r.get('zone_id') for r in groups['inaccessible_zones']}
    for zid, record in (region_report.get('zones') or {}).items():
        access = record.get('access') or {}
        historical = bool(record.get('evidence_is_historical'))
        for scope in ('global', 'local'):
            for r in access.get(scope + '_regions') or []:
                if r.get('classification') not in ('ROBOT_GLOBAL_INACCESSIBLE', 'ROBOT_LOCAL_BLOCKED'):
                    continue
                item = dict(r, zone_id=zid, display_label='차단지역', geometry_kind='REGION',
                            historical=historical, source='ZONE_ACCESSIBILITY')
                areas.append(item)
                if scope == 'global' and (zid, r.get('region_id')) not in known_region:
                    groups['inaccessible_regions'].append(dict(item, display_label='접근불가 영역'))
        if access.get('zone_access_status') == 'INACCESSIBLE' and zid not in known_zone:
            item = dict(zone_id=zid, access_status='INACCESSIBLE', display_label='접근불가 Zone',
                        geometry_kind='ZONE', reason=access.get('reason'), assessment=copy.deepcopy(access),
                        historical=historical, source='ZONE_ACCESSIBILITY')
            groups['inaccessible_zones'].append(item)
            areas.append(dict(item, display_label='차단지역'))
    # v1.8.9.2: a finished review with an unresolved physical verdict is a
    # category-4 finding, not unfinished exploration or a fabricated blockage.
    finished = set(region_report.get('locally_reviewed_zones') or [])
    finished.update(z for z, rec in (zone_records or {}).items()
        if rec.get('exploration_status') == 'COMPLETE' or
           rec.get('state') in ('EXPLORATION_COMPLETE', 'OBSERVATION_SUFFICIENT', 'OBSERVATION_LIMITED'))
    if region_report.get('stage') == 'SECOND':
        finished.discard(region_report.get('local_second_zone'))
    classified = set()
    known_tasks = {r.get('task_id') for key in ('inaccessible_regions', 'robot_unexplorable_regions')
                   for r in groups[key] if r.get('task_id')}
    for task in region_report.get('tasks') or []:
        if task.get('zone_id') not in finished or task.get('status') != 'UNVERIFIED':
            continue
        tid = task.get('task_id')
        if tid not in known_tasks:
            item = copy.deepcopy(task)
            item.update(source_status=task.get('status'),
                display_label='로봇 탐사 불가 영역',
                final_result_category='robot_unexplorable_regions',
                evaluation_complete=True, physical_blockage_proven=False,
                basis='finished_local_procedure_unverified_access',
                scope=task.get('completion_scope', 'bounded_sampled_policy'))
            groups['robot_unexplorable_regions'].append(item)
            known_tasks.add(tid)
        classified.add(tid)
    evaluation = result.setdefault('evaluation', {})
    evaluation['incomplete_tasks'] = [r for r in evaluation.get('incomplete_tasks', [])
                                    if r.get('task_id') not in classified]
    unified = [dict(p, geometry_kind='POINT') for p in groups['blocked_points']]
    for key, geometry_kind in (('inaccessible_zones','ZONE'),('inaccessible_regions','REGION'),
                               ('robot_unexplorable_regions','REGION')):
        for r in groups[key]:
            unified.append(dict(r, geometry_kind=geometry_kind, display_label='차단지역',
                access_status=r.get('access_status') or ('ROBOT_EXPLORATION_UNAVAILABLE'
                    if key=='robot_unexplorable_regions' else 'INACCESSIBLE')))
    unified.extend(r for r in areas if r.get('scope')=='local')
    result['blocked_areas'] = unified
    result['counts'] = {k: len(v) for k, v in groups.items()}
    result['classification_note'] = '기존 급변 판정 조건 충족 = 차단지점; 지점 차단으로 Zone 전체 접근불가를 추론하지 않음'
    from blocked_policy import normalize_report
    return normalize_report(result)


def project_results(region_report, zone_records, *, path_records=(),
                    reroute_summary=None, owned_goal_ids=()):
    """Project recorded evidence into the public exploration result categories."""
    return _project_results_with_baseline(_legacy_project_results_v16, region_report, zone_records,
        path_records=path_records, reroute_summary=reroute_summary,
        owned_goal_ids=owned_goal_ids)
