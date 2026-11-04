#!/usr/bin/env python3
"""Opt-in, visibility-only acceleration; requires locally built libzb_visibility.so.

Numerically matches the reference DDA for audited input distributions. Validation
must be run BEFORE enabling the DN011_VISIBILITY_FAST=1 environment toggle.
"""
from __future__ import annotations
import ctypes
import math
from pathlib import Path
import numpy as np

_DLL = None


def _load():
    global _DLL
    if _DLL is None:
        path = Path(__file__).with_name('libzb_visibility.so')
        if not path.is_file():
            raise RuntimeError('DN-011 visibility accelerator not compiled: ' + str(path))
        dll = ctypes.CDLL(str(path))
        dll.dn011_visibility_dda.argtypes = (
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_uint8))
        dll.dn011_visibility_dda.restype = None
        _DLL = dll
    return _DLL


def visibility_cells(geom, free, target, x, y, yaw, sensor, range_cap):
    """One-for-one replacement for the original visibility_cells() function."""
    sx = x + math.cos(yaw) * sensor.offset_x - math.sin(yaw) * sensor.offset_y
    sy = y + math.sin(yaw) * sensor.offset_x + math.cos(yaw) * sensor.offset_y
    gx, gy = geom.world_to_grid_float(sx, sy)
    ci, cj = math.floor(gx), math.floor(gy)
    if not geom.contains(ci, cj) or not free[cj, ci]:
        return np.empty(0, dtype=np.int32)
    angles = (yaw + sensor.offset_yaw + sensor.angle_min +
              np.arange(sensor.beams) * sensor.angle_increment - geom.origin[2])
    dx = np.cos(angles)
    dy = np.sin(angles)
    px = gx + sensor.range_min / geom.resolution * dx
    py = gy + sensor.range_min / geom.resolution * dy
    limit = (min(range_cap, sensor.range_max) - sensor.range_min) / geom.resolution
    if limit <= 0:
        return np.empty(0, dtype=np.int32)

    f = np.ascontiguousarray(free, dtype=np.uint8)
    dx, dy, px, py = [np.ascontiguousarray(v, dtype=np.float64) for v in (dx, dy, px, py)]
    out = np.zeros(geom.width * geom.height, dtype=np.uint8)
    ptr_u8 = ctypes.POINTER(ctypes.c_uint8)
    ptr_f64 = ctypes.POINTER(ctypes.c_double)
    _load().dn011_visibility_dda(
        f.ctypes.data_as(ptr_u8), geom.width, geom.height,
        dx.ctypes.data_as(ptr_f64), dy.ctypes.data_as(ptr_f64),
        px.ctypes.data_as(ptr_f64), py.ctypes.data_as(ptr_f64),
        sensor.beams, float(limit), out.ctypes.data_as(ptr_u8))
    return np.flatnonzero((out != 0) & target.ravel()).astype(np.int32)
