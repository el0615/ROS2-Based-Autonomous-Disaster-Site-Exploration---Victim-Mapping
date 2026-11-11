"""Pure offline-checkable DN-011 mission/home policy, ROS-independent."""
from __future__ import annotations

import math

THRESHOLD = 0.96
NORMAL_DONE = "ALL_REQUESTED_ZONES_OBSERVED"


def check_mission_complete(manifest: dict, summary: dict, *, threshold: float = THRESHOLD) -> tuple[bool, str]:
    """Return True ONLY if a supervised execute mission verifiably completed all 27 zones.

    A return-home task must never infer success from zero return code or merely
    exhausting a goal/time budget. This does not independently validate sensors.
    """
    opts = manifest.get("options", {})
    try:
        requested = float(opts.get("threshold", -1.0))
        observed = float(summary.get("threshold", -1.0))
    except (TypeError, ValueError):
        return False, "missing/invalid 96% threshold metadata"
    if not (abs(requested - threshold) < 1e-6 and abs(observed - threshold) < 1e-6):
        return False, f"threshold mismatch: requested={requested} summary={observed}"
    if not bool(opts.get("execute", False)) or not bool(summary.get("execute", False)):
        return False, "not an EXECUTE run"
    if opts.get("zone_ids") is not None:
        return False, "subselected zones: not a full-map mission"
    if summary.get("state") != NORMAL_DONE:
        return False, f"mission not complete: {summary.get('state')!r}"
    # The state plus per-zone outcomes are authoritative. Stop reason strings
    # are diagnostic and may differ across v2.4 builds.
    if (manifest.get("zone_bundle_signature") and summary.get("zone_bundle_signature")
            and manifest["zone_bundle_signature"] != summary["zone_bundle_signature"]):
        return False, "zone bundle changed"
    zones = summary.get("zones")
    if not isinstance(zones, dict) or len(zones) != 27:
        return False, f"expected 27 zone records, got {len(zones) if isinstance(zones,dict) else 'none'}"
    incomplete = []
    for name, entry in sorted(zones.items()):
        try:
            percent = float(entry.get("measured_percent", float("nan")))
        except (TypeError, ValueError):
            percent = float("nan")
        if (not math.isfinite(percent) or percent < threshold * 100 - 1e-5
                or entry.get("state") != "OBSERVATION_SUFFICIENT"):
            incomplete.append((name, entry.get("state"), entry.get("measured_percent")))
    if incomplete:
        return False, f"incomplete or <96% zones: {incomplete[:5]}"
    sent = int(summary.get("sent", 0))
    success = int(summary.get("nav_succeeded", -1))
    if sent < 1 or success != sent:
        return False, f"navigation result mismatch: sent={sent}, succeeded={success}"
    return True, f"27/27 zones >=96%; {sent} goals succeeded"


def pose_distance(a: dict, b: dict) -> tuple[float, float]:
    d = math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))
    yaw = math.atan2(math.sin(float(a["yaw"]) - float(b["yaw"])),
                     math.cos(float(a["yaw"]) - float(b["yaw"])))
    return d, abs(yaw)


def parse_cli_param_output(name: str, output: str):
    """Safely parse expected `ros2 param get` outputs in ROS2 Humble.

    Fail closed for unknown formats; never eval() arbitrary CLI text.
    """
    import ast
    import re
    supported = {'GridBased.plugin', 'GridBased.cost_travel_multiplier',
                 'inflation_layer.inflation_radius', 'max_velocity', 'min_velocity'}
    if name not in supported:
        raise ValueError('Unexpected parameter ' + name)
    if name == 'GridBased.plugin':
        match = re.search(r'(?m)^String value is: ([^\r\n]+)$', output)
        if not match:
            raise ValueError('Missing string value')
        return match.group(1).strip()
    if name in ('GridBased.cost_travel_multiplier', 'inflation_layer.inflation_radius'):
        match = re.search(r'(?m)^Double value is: ([^\r\n]+)$', output)
        if not match:
            raise ValueError('Missing double value')
        value = float(match.group(1).strip())
        if not math.isfinite(value):
            raise ValueError('Nonfinite double value')
        return value
    match = re.search(r'(?m)^Double values are: ([^\r\n]+)$', output)
    if not match:
        raise ValueError('Missing double-array value')
    array_text = re.search(r'\[[^\]\r\n]*\]', match.group(1))
    if not array_text:
        raise ValueError('Malformed double array')
    vals = ast.literal_eval(array_text.group(0))
    if not isinstance(vals, list) or len(vals) != 3:
        raise ValueError('Expected 3-element double array')
    parsed = [float(v) for v in vals]
    if not all(math.isfinite(v) for v in parsed):
        raise ValueError('Nonfinite double array')
    return parsed


def parse_saved_session_path(line: str) -> str | None:
    """Read the full session directory after [SAVED], preserving timestamp suffix.

    The caller separately verifies that the resolved directory belongs directly
    under the approved project logs/hospital_zone_mission directory.
    """
    import re
    m = re.fullmatch(r"\[SAVED\][ \t]+(.+?)\s*", line)
    return m.group(1) if m else None


def evaluate_return_eligibility(manifest: dict, summary: dict, *, threshold: float = THRESHOLD) -> dict:
    """Separate completeness of exploration from eligibility to go home.

    FULL: all 27 zones >= threshold and achieved normal terminal state.
    PARTIAL: terminal PARTIAL_DEFERRED_REMAIN, all 27 zones accounted for,
      remaining zones explicitly deferred, no Nav2 failures, no unassessed zones.
    Any uncertainty, timeout, interruption, mixed terminal state -> BLOCKED.
    A PARTIAL return is never labeled a successful exploration mission.
    """
    if manifest.get('options',{}).get('connectivity_aware'):
        return _evaluate_connectivity_return(manifest,summary,threshold=threshold)
    complete, full_detail = check_mission_complete(manifest, summary, threshold=threshold)
    if complete:
        return {'eligible': True, 'exploration': 'FULL', 'detail': full_detail,
                'partial_zones': []}
    def blocked(reason):
        return {'eligible': False, 'exploration': 'BLOCKED', 'detail': reason,
                'partial_zones': []}
    if summary.get('state') != 'PARTIAL_DEFERRED_REMAIN':
        return blocked(full_detail)
    opts = manifest.get('options', {})
    try:
        t1 = float(opts.get('threshold', -1))
        t2 = float(summary.get('threshold', -1))
        sent = int(summary.get('sent', 0))
        succeeded = int(summary.get('nav_succeeded', -1))
        canceled = int(summary.get('coverage_canceled', -1))
    except (TypeError, ValueError, OverflowError):
        return blocked('invalid mission metadata')
    if abs(t1-threshold)>1e-6 or abs(t2-threshold)>1e-6:
        return blocked('threshold mismatch')
    if not opts.get('execute') or not summary.get('execute') or opts.get('zone_ids') is not None:
        return blocked('not a full 27-zone EXECUTE attempt')
    if (manifest.get('zone_bundle_signature') and summary.get('zone_bundle_signature')
            and manifest['zone_bundle_signature'] != summary['zone_bundle_signature']):
        return blocked('zone bundle signature mismatch')
    if sent < 1 or succeeded != sent or canceled != 0:
        return blocked(f'navigation success mismatch: {succeeded}/{sent}, canceled={canceled}')
    if summary.get('owned_goal_id'):
        return blocked('owned goal still recorded in summary')
    zones = summary.get('zones')
    if not isinstance(zones, dict) or len(zones) != 27:
        return blocked('not exactly 27 zone records')
    deferred = []
    sufficient = 0
    for name, entry in sorted(zones.items()):
        if not isinstance(entry, dict):
            return blocked(f'malformed zone {name}')
        try:
            percent = float(entry.get('measured_percent', float('nan')))
        except (TypeError, ValueError, OverflowError):
            return blocked(f'malformed coverage for {name}')
        if not math.isfinite(percent) or percent < 0 or percent > 100:
            return blocked(f'invalid coverage for {name}')
        state = entry.get('state')
        if state == 'OBSERVATION_SUFFICIENT' and percent >= threshold*100-1e-5:
            sufficient += 1
        elif state == 'PARTIAL_DEFERRED' and 0 <= percent < threshold*100:
            reason = entry.get('deferred_reason')
            if not isinstance(reason, str) or not reason.strip():
                return blocked(f'unexplained deferred zone: {name}')
            deferred.append({'zone': name, 'measured_percent': percent, 'reason': reason})
        else:
            return blocked(f'unaccounted zone: {name} state={state} coverage={percent:.2f}')
    if not deferred or sufficient < 1:
        return blocked('partial mission has no verified sufficient/deferred combination')
    return {'eligible': True, 'exploration': 'PARTIAL',
            'detail': f'{sufficient}/27 zones >= {threshold*100:.0f}%; {len(deferred)} deferred; '
                      f'Nav2 succeeded {succeeded}/{sent}. Exploration remains PARTIAL.',
            'partial_zones': deferred}



def _approved_path_cancellations(manifest, summary):
    count=summary.get('path_blockage_canceled',0)
    rows=summary.get('path_blockage_cancellations',[])
    if type(count) is not int or count<0 or not isinstance(rows,list) or len(rows)!=count:return False,0
    if count and (manifest.get('options',{}).get('path_blockage_check') is not True
                  or summary.get('path_blockage_check') is not True):return False,0
    ids=set();zones={}
    for r in rows:
        if not isinstance(r,dict):return False,0
        e=r.get('evidence',{});gid=r.get('goal_id');z=e.get('zone') if isinstance(e,dict) else None
        if (r.get('reason')!='PATH_BLOCKAGE_REPLAN_REQUIRED' or r.get('terminal')!='CANCELED'
            or r.get('stop_confirmed') is not True or r.get('new_coverage') is not True
            or not isinstance(gid,str) or len(gid)!=32 or any(x not in '0123456789abcdef' for x in gid)
            or gid in ids or not isinstance(e,dict) or not isinstance(z,str)
            or z not in summary.get('zones',{}) or e.get('bundle_signature')!=summary.get('zone_bundle_signature')
            or e.get('persistent') is not True or e.get('translation_stalled') is not True
            or e.get('scan_supported') is not True or e.get('prior_free') is not True or e.get('cost')!=100
            or type(e.get('independent_samples')) is not int or e['independent_samples']<3):return False,0
        ids.add(gid);zones[z]=zones.get(z,0)+1
        if zones[z]>2:return False,0
    return True,count


def _evaluate_connectivity_return(manifest,summary,*,threshold=THRESHOLD):
    """Completion classification is separate from safe return authorization.

    No imports from ROS or the zone engine: validate serialized certificates
    defensively. Existing HomeSupervisor still checks live sensors, path,
    footprint, action state, arrival and stop before/after actual return.
    """
    def blocked(reason):
        return dict(eligible=False,exploration='BLOCKED',detail=reason,partial_zones=[])
    opts=manifest.get('options',{})
    try:
        t1=float(opts.get('threshold',float('nan')))
        t2=float(summary.get('threshold',float('nan')))
        sent=int(summary.get('sent',-1));ok=int(summary.get('nav_succeeded',-1))
        aborted=int(summary.get('nav_aborted',0));canceled=int(summary.get('coverage_canceled',-1))
    except (ValueError,TypeError,OverflowError):return blocked('invalid metadata')
    if not all(math.isfinite(t) and abs(t-threshold)<1e-6 for t in (t1,t2)):
        return blocked('threshold mismatch')
    if not opts.get('execute') or not summary.get('execute') or opts.get('zone_ids') is not None:
        return blocked('not a full-zone execute run')
    if (not summary.get('connectivity_aware') or not manifest.get('zone_bundle_signature')
            or manifest['zone_bundle_signature']!=summary.get('zone_bundle_signature')):
        return blocked('missing or mismatched assessment identity')
    terminal={'ALL_REQUESTED_ZONES_OBSERVED','ALL_ZONES_ASSESSED_WITH_LIMITS','PARTIAL_DEFERRED_REMAIN'}
    if summary.get('state') not in terminal or summary.get('stop_reason') or summary.get('owned_goal_id'):
        return blocked('not an orderly resolved/stopped mission')
    valid_path_cancels,path_canceled=_approved_path_cancellations(manifest,summary)
    if not valid_path_cancels:return blocked('unapproved path blockage cancellation')
    valid_operational,operational_canceled=_approved_operational_cancellations(manifest,summary)
    if not valid_operational:return blocked('unapproved operational cancellation')
    if sent<1 or min(ok,aborted)<0 or ok+aborted+path_canceled+operational_canceled!=sent or canceled!=0:
        return blocked('navigation terminal counts mismatch')
    if aborted and not opts.get('continue_on_nav_abort'):
        return blocked('unapproved aborted-navigation continuation')
    zones=summary.get('zones')
    if not isinstance(zones,dict) or len(zones)!=27:return blocked('expected 27 zones')
    sufficient=0;limited=0;partial=[]
    for name,rec in zones.items():
        if not isinstance(rec,dict):return blocked('malformed zone')
        try:value=float(rec.get('measured_percent',float('nan')))
        except (ValueError,TypeError):return blocked('malformed coverage')
        if not math.isfinite(value) or not 0<=value<=100:return blocked('invalid coverage')
        state=rec.get('state')
        if state=='OBSERVATION_SUFFICIENT' and value>=100*threshold-1e-5:
            sufficient+=1
        elif state=='OBSERVATION_LIMITED' and value<100*threshold:
            assessment=rec.get('assessment') or {};e=assessment.get('evidence') or {}
            flags=('catalog_complete','inputs_fresh','context_matches',
                   'connectivity_verified','all_potential_information_accounted')
            zero=('probes_failed','unrepresented_components','unknown_cells',
                  'unobserved_free_cells','remaining_gain_views')
            if (assessment.get('normal_termination') is not True
                    or not all(e.get(k) is True for k in flags)
                    or type(e.get('positions')) is not int or e['positions']<1
                    or type(e.get('probes_succeeded')) is not int or e['probes_succeeded']!=e['positions']
                    or any(type(e.get(k)) is not int or e[k]!=0 for k in zero)
                    or not e.get('cost_key') or e.get('bundle_signature')!=summary['zone_bundle_signature']):
                return blocked('invalid connectivity certificate: '+name)
            limited+=1
            partial.append(dict(zone=name,measured_percent=value,reason='VERIFIED_CONNECTED_OBSERVATION_LIMIT'))
        elif state=='PARTIAL_DEFERRED' and isinstance(rec.get('deferred_reason'),str) and rec['deferred_reason']:
            partial.append(dict(zone=name,measured_percent=value,reason=rec['deferred_reason']))
        else:return blocked('unassessed/inconsistent zone: '+name)
    if sufficient==27:
        if summary['state']!='ALL_REQUESTED_ZONES_OBSERVED':return blocked('terminal state mismatch')
        classification='FULL'
    elif sufficient+limited==27:
        if summary['state']!='ALL_ZONES_ASSESSED_WITH_LIMITS':return blocked('terminal state mismatch')
        classification='LIMITED'
    else:
        if summary['state']!='PARTIAL_DEFERRED_REMAIN':return blocked('terminal state mismatch')
        if sufficient+limited==0:return blocked('no resolved zone')
        classification='PARTIAL'
    return dict(eligible=True,exploration=classification,partial_zones=partial,
        detail=f'{sufficient}/27 coverage targets, {limited} connectivity-certified limits; '
               f'raw measured coverage unchanged; navigation succeeded={ok}, aborted={aborted}, path_canceled={path_canceled}, operational_canceled={operational_canceled}')


def _approved_operational_cancellations(manifest, summary):
    """Fail closed on new cancellation records; never authorize by a count alone.

    Return transport/path/footprint/stationarity checks stay in HomeSupervisor.
    These are serialized accounting checks, not a new physical safety proof.
    """
    count=summary.get('operational_canceled',0)
    rows=summary.get('operational_cancellations',[])
    if type(count) is not int or count<0 or not isinstance(rows,list) or len(rows)!=count:
        return False,0
    if not count:return True,0
    opts=manifest.get('options',{})
    if opts.get('mission_budget_guard') is not True or summary.get('mission_budget_guard') is not True:
        return False,0
    mapping={'stall_window_s':'guard_stall_window_s','stall_extent_m':'guard_stall_extent_m',
             'stall_recoveries':'guard_stall_recoveries','goal_recovery_limit':'guard_goal_recovery_limit',
             'zone_cancel_limit':'guard_zone_cancel_limit','work_base_s':'guard_zone_work_base_s',
             'work_max_s':'guard_zone_work_max_s','extension_window_s':'guard_extension_window_s',
             'extension_gain_pp':'guard_extension_gain_pp'}
    ids={r.get('goal_id') for r in summary.get('path_blockage_cancellations',[]) if isinstance(r,dict)}
    zone_counts={}
    def finite(v):return type(v) in (int,float) and math.isfinite(v)
    for row in rows:
        if not isinstance(row,dict):return False,0
        gid=row.get('goal_id');z=row.get('zone');e=row.get('evidence');cfg=row.get('settings')
        if (row.get('schema')!='dn012_operational_cancel_v1' or row.get('terminal')!='CANCELED'
                or row.get('stop_confirmed') is not True or row.get('new_coverage') is not True
                or row.get('local_start_clear') is not True or not isinstance(gid,str)
                or len(gid)!=32 or any(c not in '0123456789abcdef' for c in gid) or gid in ids
                or z not in summary.get('zones',{}) or row.get('goal_kind') not in ('ENTRY','OBSERVE')
                or row.get('bundle_signature')!=summary.get('zone_bundle_signature')
                or not row.get('coverage_session') or row['coverage_session']!=summary.get('coverage_session')
                or not isinstance(e,dict) or not isinstance(cfg,dict)
                or e.get('physical_blockage_proven') is not False or e.get('normal_termination') is not False):
            return False,0
        if any(not finite(cfg.get(k)) or not finite(opts.get(v)) or cfg[k]!=opts[v] for k,v in mapping.items()):return False,0
        if (not 5<=cfg['stall_window_s']<=120 or not .02<=cfg['stall_extent_m']<=1.
                or type(cfg['stall_recoveries']) is not int or not 1<=cfg['stall_recoveries']<=30
                or type(cfg['goal_recovery_limit']) is not int or not 0<=cfg['goal_recovery_limit']<=100
                or type(cfg['zone_cancel_limit']) is not int or not 1<=cfg['zone_cancel_limit']<=3
                or not 30<=cfg['work_base_s']<=cfg['work_max_s']<=1800):return False,0
        start=row.get('requested_ros_s');stop=row.get('confirmed_ros_s')
        if not finite(start) or not finite(stop) or stop<start:return False,0
        reason=row.get('reason')
        if e.get('reason')!=reason:return False,0
        if reason=='NAVIGATION_STAGNATION_UNVERIFIED':
            if (not finite(e.get('window_s')) or e['window_s']<cfg['stall_window_s']
                    or not finite(e.get('extent_m')) or not 0<=e['extent_m']<cfg['stall_extent_m']
                    or type(e.get('recovery_increase')) is not int or e['recovery_increase']<cfg['stall_recoveries']
                    or type(e.get('sample_count')) is not int or e['sample_count']<3):return False,0
        elif reason=='NAV_RECOVERY_BUDGET_UNVERIFIED':
            if (cfg['goal_recovery_limit']<=0 or e.get('recovery_limit')!=cfg['goal_recovery_limit']
                    or type(e.get('recoveries')) is not int or e['recoveries']<cfg['goal_recovery_limit']):return False,0
        elif reason=='ZONE_WORK_BUDGET_UNVERIFIED':
            if (not finite(e.get('work_time_ros_s')) or not finite(e.get('allowance_ros_s'))
                    or e.get('expired') is not True or e['work_time_ros_s']<e['allowance_ros_s']
                    or e['allowance_ros_s'] not in (cfg['work_base_s'],cfg['work_max_s'])):return False,0
        else:return False,0
        ids.add(gid);zone_counts[z]=zone_counts.get(z,0)+1
        if zone_counts[z]>cfg['zone_cancel_limit']:return False,0
    return True,count
