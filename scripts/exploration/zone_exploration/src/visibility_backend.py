"""Visibility-only backend with bounded cross-checks, not a route/motion authorizer.

The isolated C source is the previously supplied DDA implementation. Installed
v1-v4 modules/environment variables are not modified or imported by this backend.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import threading
import numpy as np
from planning_core import visibility_cells as reference


class VisibilityBackend:
    def __init__(self, mode='auto', check_interval=256):
        self.mode = mode
        self.fast = None
        self.calls = self.fast_calls = self.checked = 0
        self.check_interval = check_interval
        self._contexts = set()
        self._lock = threading.Lock()
        root = Path(__file__).resolve().parent
        receipt = root / 'visibility_build_receipt.json'
        if mode != 'python':
            try:
                r = json.loads(receipt.read_text())
                for filename in ('zb_visibility_dda.c', 'zb_visibility_c.py', 'libzb_visibility.so'):
                    digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
                    if r['sha256'].get(filename) != digest:
                        raise ValueError('Visibility build hash mismatch: '+filename)
                if r.get('equivalence_pass') is not True:
                    raise ValueError('No successful local equivalence test')
                from zb_visibility_c import visibility_cells
                self.fast = visibility_cells
            except (OSError, ValueError, KeyError, ImportError) as exc:
                if mode == 'c':
                    raise RuntimeError('C backend not ready; run check_install.py --build: '+str(exc)) from exc
                print('[VISIBILITY_REFERENCE] Local accelerator unavailable; bounded Python candidates used.', flush=True)
        print('[VISIBILITY_BACKEND] '+('C_WITH_CROSSCHECKS' if self.fast else 'PYTHON_REFERENCE'), flush=True)

    def __call__(self, geom, free, target, x, y, yaw, sensor, cap):
        # Called only by the single planner worker; lock also makes direct tests safe.
        with self._lock:
            self.calls += 1
            if self.fast is None:
                return reference(geom, free, target, x, y, yaw, sensor, cap)
            result = self.fast(geom, free, target, x, y, yaw, sensor, cap)
            self.fast_calls += 1
            # A new geometry/target/sensor checks one actual request, not all
            # requests. The grid/candidate cache separately fingerprints obstacles.
            key = (geom.width, geom.height, geom.resolution, tuple(geom.origin),
                   hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest(),
                   tuple(vars(sensor).values()), cap)
            if key not in self._contexts or self.calls % self.check_interval == 0:
                expected = reference(geom, free, target, x, y, yaw, sensor, cap)
                self.checked += 1
                if not np.array_equal(result, expected):
                    raise RuntimeError('VISIBILITY_MISMATCH: no Goal is authorized from this batch')
                self._contexts.add(key)
            return result

    def summary(self):
        return dict(calls=self.calls, fast_calls=self.fast_calls,
                    crosschecks=self.checked, backend='c' if self.fast else 'python')
