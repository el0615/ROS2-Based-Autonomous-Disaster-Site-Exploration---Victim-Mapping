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
