"""User-approved DN012 rule: existing confirmed large-reroute criteria => BLOCKED.

No changes to path comparison/confirmation, navigation masks, safety gates or
Zone completion. Public BLOCKED points never require physical_* proof flags,
sensor corroboration, or mask_eligible. Those remain raw evidence only.
"""
from __future__ import annotations
import copy
import math

VERSION = 'DN012_REROUTE_BLOCKED_V1_8_9'


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def reroute_decision(record):
    """Accept the detector's existing confirmation plus its recorded thresholds.

    The original detector still enforces same Goal, timestamps, pose continuity,
    +3m AND +50% defaults, >=2 samples and >=0.5s. Do not invent a new gate.
    """
    r = record or {}
    if isinstance(r.get('confirmation'), dict):
        r = r['confirmation']
    metric = r.get('metric') or r
    thresholds = r.get('thresholds') or {}
    values = (metric.get('extra_m'), metric.get('increase_fraction'),
              thresholds.get('min_extra_m'), thresholds.get('min_fraction'))
    if r.get('plan_change_confirmed') is not True:
        return False, 'EXISTING_DETECTOR_NOT_CONFIRMED'
    if not all(finite(x) for x in values):
        return False, 'RECORDED_THRESHOLDS_OR_METRICS_MISSING'
    delta, ratio, min_delta, min_ratio = values
    if delta + 1e-9 < min_delta or ratio + 1e-9 < min_ratio:
        return False, 'EXISTING_AND_THRESHOLDS_NOT_MET'
    # Full evidence contains a confirming metric; compact PointStore records do not.
    confirmation = r.get('confirm_metric')
    if isinstance(confirmation, dict):
        a, b = confirmation.get('extra_m'), confirmation.get('increase_fraction')
        if not finite(a) or not finite(b) or a + 1e-9 < min_delta or b + 1e-9 < min_ratio:
            return False, 'CONFIRMING_METRIC_NOT_MET'
    return True, 'EXISTING_REROUTE_RULE_SATISFIED'


def annotate_reroute(record):
    result = copy.deepcopy(record)
    blocked, reason = reroute_decision(record)
    result.update(blockage_status='BLOCKED' if blocked else 'NOT_BLOCKED',
                  decision_basis=reason, blockage_policy=VERSION,
                  display_label='차단지점' if blocked else '경로변경 진단',
                  zone_inaccessibility_inferred=False)
    return result


def _supported(item, kind=None):
    """Classify raw, projected, or clustered evidence without dropping provenance."""
    if not isinstance(item, dict):
        return False
    sources = item.get('sources') or []
    source = kind or item.get('source')
    if source == 'PATH_BLOCKAGE' or ('PATH_BLOCKAGE' in sources):
        return True  # the existing path-obstruction detector already decided
    if source == 'REROUTE' or 'REROUTE' in sources or 'plan_change_confirmed' in item:
        if reroute_decision(item)[0]:
            return True
    evidence = item.get('evidence')
    if isinstance(evidence, dict) and evidence is not item:
        if _supported(evidence, source):
            return True
    for ob in item.get('observations') or []:
        if _supported(ob, ob.get('source')):
            return True
    # v1.7 index entries store the full confirmed detector event here.
    if isinstance(item.get('confirmation'), dict):
        return reroute_decision(item)[0]
    return False


def normalize_points(records):
    accepted, diagnostics = [], []
    for raw in records:
        item = copy.deepcopy(raw)
        point = item.get('xy') or item.get('point_xy')
        valid = isinstance(point, (list, tuple)) and len(point) == 2 and all(finite(x) for x in point)
        if valid and _supported(item):
            item.update(xy=list(point), geometry_kind='POINT', classification='BLOCKED',
                        blockage_status='BLOCKED', display_label='차단지점',
                        blockage_policy=VERSION, zone_inaccessibility_inferred=False)
            accepted.append(item)
        else:
            item.update(blockage_status='NOT_BLOCKED', blockage_policy=VERSION)
            diagnostics.append(item)
    return accepted, diagnostics


def normalize_report(result):
    """Both category consumers and unified consumers get the SAME point verdict."""
    result = copy.deepcopy(result)
    groups = result.get('categories') or {}
    points, diagnostics = normalize_points(groups.get('blocked_points') or [])
    groups['blocked_points'] = points
    result['categories'] = groups
    result['reroute_diagnostics_v189'] = diagnostics
    result.setdefault('category_labels', {})['blocked_points'] = '차단지점'
    result['counts'] = {key: len(value) for key, value in groups.items()}
    result['blocked_areas'] = [x for x in result.get('blocked_areas', [])
                               if x.get('geometry_kind') != 'POINT'] + points
    result['blockage_policy'] = VERSION
    return result
