#!/usr/bin/env python3
"""Observation mission policies using the shared pure sequence implementation."""
from observation_planning_core import (
    ThresholdTracker, build_plan, rerank_plan,
    oriented_footprint_clear, conservative_safe,
)
from sequence_core import (
    angle_delta, fraction, ZoneRecord, MissionBook, pose_repeated, zone_order,
    refresh_candidate, validate_oriented_swept_path, ensure_execution_preflight,
    StopEvidence, safe_early_stop, json_candidate,
)
import sequence_core as _core


def build_batch(bundle, snapshot, zones, cfg, previous, locked=None, shortlist=3,
                entrance_overlay=None, navfn_fallback=False):
    return _core.build_batch(
        bundle, snapshot, zones, cfg, previous, locked, shortlist,
        entrance_overlay, navfn_fallback, plan_builder=build_plan)


def validate_nav_path(xys, frame, goal, robot_pose, geom, known_free, costmap,
                      end_tolerance=.12, start_tolerance=.60):
    return _core.validate_nav_path(
        xys, frame, goal, robot_pose, geom, known_free, costmap,
        end_tolerance, start_tolerance, diagnostics=False)
