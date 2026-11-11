#!/usr/bin/env python3
"""Observation planner entry point with the existing guarded visibility opt-in.

Pure geometry and candidate calculations are shared with the zone explorer.
The visibility callback is passed per call, keeping their backends independent.
"""
from __future__ import annotations
from planning_core import (
    Settings, SensorModel, Geometry, ThresholdTracker, array_fingerprint,
    conservative_safe, oriented_footprint_clear, conservative_start_bridge,
    path_graph, generate_candidates, validate_coverage, current_input_error,
    rerank_plan, reportable,
)
import planning_core as _core


def build_plan(geom, labels, known_free, zone, observed, robot_pose, sensor, cfg,
               costmap=None, *, allow_navfn_fallback=False,
               force_navfn_fallback=False):
    return _core.build_plan(
        geom, labels, known_free, zone, observed, robot_pose, sensor, cfg, costmap,
        allow_navfn_fallback=allow_navfn_fallback,
        force_navfn_fallback=force_navfn_fallback, visibility_fn=visibility_cells)


def append_navfn_fallback_views(plan, geom, labels, free, safe, clearance, target,
                                observed, robot_pose, sensor, cfg, points,
                                costmap=None, max_evaluations=24):
    return _core.append_navfn_fallback_views(
        plan, geom, labels, free, safe, clearance, target, observed, robot_pose,
        sensor, cfg, points, costmap, max_evaluations,
        visibility_fn=visibility_cells)


# DN011_VISIBILITY_SUPERVISED_EXECUTE_V3_20261101
# Default stays exactly the original Python implementation.
_DN011_VISIBILITY_REFERENCE = _core.visibility_cells
import os as _dn011_visibility_os
import sys as _dn011_visibility_sys
_DN011_V3_EXECUTE = ('--execute' in _dn011_visibility_sys.argv)
_DN011_V3_GUARD = None
if (_dn011_visibility_os.environ.get('DN011_VISIBILITY_FAST', '0') == '1'
        and _DN011_V3_EXECUTE):
    # DN011_VISIBILITY_MULTIZONE_EXECUTE_V4_20261101
    from visibility_execute_guard_v4 import Guard as _DN011_V3_Guard
    _DN011_V3_GUARD = _DN011_V3_Guard(__file__, _dn011_visibility_sys.argv)

def visibility_cells(geom, free, target, x, y, yaw, sensor, range_cap):
    if _dn011_visibility_os.environ.get('DN011_VISIBILITY_FAST', '0') == '1':
        from visibility_fast import visibility_cells as _fast_visibility_cells
        if _DN011_V3_EXECUTE:
            if _DN011_V3_GUARD is None:
                raise RuntimeError('V3 execute guard missing')
            return _DN011_V3_GUARD(_DN011_VISIBILITY_REFERENCE, _fast_visibility_cells,
                geom, free, target, x, y, yaw, sensor, range_cap)
        return _fast_visibility_cells(geom, free, target, x, y, yaw, sensor, range_cap)
    return _DN011_VISIBILITY_REFERENCE(geom, free, target, x, y, yaw, sensor, range_cap)
