#!/usr/bin/env python3
"""Scoped, fail-closed navigation-only entrance mask for the approved hospital map.

NOT a SLAM map patch. Does not modify the 27-zone mask or measured coverage.
Only approved unknown pixels with GLOBAL COSTMAP EXACTLY 0 can be temporarily
considered free for *candidate planning*. Never changes a Nav2 costmap or action.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math
import numpy as np


def _array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def apply_entrance_overlay(known_free, labels, geom, costmap, pose, config_path):
    """Return (copy of bool free map, diagnostic) or raise on unexpected config/map.

    Gates: exact reference hash and geometry, approved cells in zone ID=0,
    current pose in physically reviewed doorway, live costmap value exactly zero.
    An unknown/occupied costmap cell is NEVER overridden.
    """
    free = np.asarray(known_free, dtype=bool)
    labels = np.asarray(labels)
    cost = np.asarray(costmap)
    if free.shape != labels.shape or free.shape != cost.shape:
        raise ValueError('ENTRANCE_SHAPE_MISMATCH')
    c = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if c.get('schema') != 'j100_entrance_local_navigation_overlay_v1':
        raise ValueError('ENTRANCE_UNKNOWN_SCHEMA')
    if c.get('always_require_live_global_costmap_zero') is not True:
        raise ValueError('ENTRANCE_COSTMAP_GUARD_NOT_ENABLED')
    if _array_sha256(free) != c['reference_known_free_sha256']:
        raise ValueError('ENTRANCE_REFERENCE_MAP_MISMATCH')
    g = c['geometry']
    if (geom.width, geom.height, geom.resolution, tuple(geom.origin)) != (
        g['width'], g['height'], g['resolution'], tuple(g['origin'])):
        raise ValueError('ENTRANCE_GEOMETRY_MISMATCH')
    if not len(pose)>=3 or not all(math.isfinite(float(x)) for x in pose[:3]):
        raise ValueError('ENTRANCE_INVALID_POSE')
    win=c['start_pose_window']
    at_entrance=(win['x_min']<=pose[0]<=win['x_max'] and win['y_min']<=pose[1]<=win['y_max'])
    diagnostic=dict(enabled=True,at_entrance=at_entrance,approved_unknown_cells=c['cell_count'],
                    live_costmap_clear_cells=0,applied_cells=0,live_costmap_blocked_cells=0)
    if not at_entrance:return free.copy(),diagnostic
    approved = np.zeros_like(free,dtype=bool)
    for j,ranges in c['cell_rows']:
        if not 0<=j<geom.height:raise ValueError('ENTRANCE_ROW_OUT_OF_RANGE')
        for x0,x1 in ranges:
            if not 0<=x0<x1<=geom.width:raise ValueError('ENTRANCE_COLUMN_OUT_OF_RANGE')
            if approved[j,x0:x1].any():raise ValueError('ENTRANCE_DUPLICATE_CELL')
            approved[j,x0:x1]=True
    if np.count_nonzero(approved)!=c['cell_count'] or c['cell_count']!=81:
        raise ValueError('ENTRANCE_APPROVED_CELL_COUNT_CHANGED')
    if (labels[approved]!=0).any() or free[approved].any():
        raise ValueError('ENTRANCE_OVERRIDE_TOUCHES_ZONE_OR_PRIOR_FREE')
    # Require a current, non-lethal/non-inflated, positively known clear costmap cell.
    live_ok = approved & (cost == 0)
    diagnostic.update(live_costmap_clear_cells=int(np.count_nonzero(live_ok)),
                      applied_cells=int(np.count_nonzero(live_ok)),
                      live_costmap_blocked_cells=int(np.count_nonzero(approved & ~live_ok)))
    out=free.copy();out[live_ok]=True
    return out,diagnostic
