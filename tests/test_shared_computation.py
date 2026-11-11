"""Backend separation and validation boundaries after sharing pure calculations."""
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXPLORATION = Path(__file__).resolve().parents[1] / 'scripts/exploration'
sys.path.insert(0, str(EXPLORATION))
import planning_core as planning
import sequence_core as sequence
import observation_planning_core as observation_planning
import observation_sequence_core as observation_sequence
from hospital_coverage_core import Geometry


class SharedPlanningTests(unittest.TestCase):
    def setUp(self):
        self.geom = Geometry(40, 32, .15, (0., 0., 0.))
        self.free = np.ones((32, 40), dtype=bool)
        self.free[[0, -1], :] = False
        self.free[:, [0, -1]] = False
        self.labels = self.free.astype(np.int16)
        self.observed = np.zeros_like(self.free)
        self.zone = dict(id='A', mask_id=1, known_free_cells=int(self.free.sum()))
        self.pose = (1.5, 1.5, 0.)
        self.sensor = planning.SensorModel(-math.pi, math.tau / 48, 48, .05, 7.)
        self.cfg = planning.Settings(max_candidates=8)
        self.cost = np.where(self.free, 0, 100).astype(np.int8)
        self.args = (self.geom, self.labels, self.free, self.zone, self.observed,
                     self.pose, self.sensor, self.cfg, self.cost)

    def test_standalone_import_does_not_activate_observation_execute_guard(self):
        code = (
            "import os,sys; sys.path.insert(0,sys.argv[1]); "
            "os.environ['DN011_VISIBILITY_FAST']='1'; sys.argv.append('--execute'); "
            "import planning_core,sequence_core; "
            "assert 'observation_planning_core' not in sys.modules; "
            "assert 'visibility_execute_guard_v4' not in sys.modules"
        )
        result = subprocess.run([sys.executable, '-B', '-c', code, str(EXPLORATION)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_observation_visibility_override_does_not_change_zone_backend(self):
        def no_visibility(*args):
            return np.empty(0, dtype=np.int32)

        with patch.object(observation_planning, 'visibility_cells',
                          side_effect=no_visibility) as backend:
            observation = observation_planning.build_plan(*self.args)
            standalone = planning.build_plan(*self.args)
        self.assertGreater(backend.call_count, 0)
        self.assertIsNone(observation['selected'])
        self.assertIsNotNone(standalone['selected'])

    def test_explicit_backend_does_not_leak_to_later_calls(self):
        invisible = lambda *args: np.empty(0, dtype=np.int32)
        result = planning.build_plan(*self.args, visibility_fn=invisible,
                                     allow_navfn_fallback=True,
                                     force_navfn_fallback=True)
        self.assertIsNone(result['selected'])
        self.assertIsNotNone(planning.build_plan(*self.args)['selected'])

    def test_batch_uses_each_entrypoints_current_planner(self):
        bundle = ({}, self.geom, self.labels, self.free, 'fixture')
        snapshot = dict(observed=self.observed, pose=self.pose,
                        sensor=self.sensor, cost={'values': self.cost})
        args = (bundle, snapshot, [self.zone], self.cfg, [])
        empty_plan = dict(status='NO_CANDIDATE', candidates=[])
        with patch.object(observation_sequence, 'build_plan',
                          return_value=empty_plan) as observation_builder:
            result = observation_sequence.build_batch(*args)
            self.assertEqual(result['candidates'], [])
            self.assertGreater(len(sequence.build_batch(*args)['candidates']), 0)
        observation_builder.assert_called_once()
        with patch.object(sequence, 'build_plan',
                          return_value=empty_plan) as standalone_builder:
            self.assertEqual(sequence.build_batch(*args)['candidates'], [])
        standalone_builder.assert_called_once()

    def test_path_rejection_keeps_distinct_diagnostic_behavior(self):
        points = [(1.5, 1.5), (1.8, 1.5)]
        goal = dict(x=1.8, y=1.5)
        self.cost[10, 11] = 100
        for module, diagnostic in ((sequence, True), (observation_sequence, False)):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaisesRegex(
                    ValueError, '^PATH_CENTRELINE_NOT_FREE$'):
                module.validate_nav_path(points, 'map', goal, self.pose,
                                         self.geom, self.free, self.cost)
            self.assertEqual('[PATH_DIAG]' in output.getvalue(), diagnostic)


class GuardDependencyTests(unittest.TestCase):
    def fixture(self, folder, version):
        module = importlib.import_module(f'visibility_execute_guard_v{version}')
        names = (*module.SHARED_SOURCES, 'observation_planning_core.py',
                 'visibility_execute_guard_v3.py', 'visibility_execute_guard_v4.py')
        for name in names:
            shutil.copy2(EXPLORATION / name, folder / name)

        def digest(name):
            return hashlib.sha256((folder / name).read_bytes()).hexdigest()

        dependencies = {name: digest(name) for name in module.SHARED_SOURCES}
        s1 = dict(original_sha256=module.BASE_SHA, patched_sha256='historical-v1')
        s3 = dict(previous_sha256=s1['patched_sha256'],
                  patched_sha256=digest('observation_planning_core.py'),
                  guard_sha256=digest('visibility_execute_guard_v3.py'),
                  shared_sources_sha256=dependencies)
        s4 = dict(previous_sha256=s3['patched_sha256'],
                  patched_sha256=digest('observation_planning_core.py'),
                  guard_sha256=digest('visibility_execute_guard_v4.py'),
                  shared_sources_sha256=dependencies)
        for name, state in ((module.V1_STATE, s1), (module.V3_STATE, s3),
                            ('.DN011_visibility_multizone_v4_install.json', s4)):
            (folder / name).write_text(json.dumps(state))
        argv = ['guard', '--execute', '--max-goals', str(version - 2), '--zone-ids']
        argv += ['LOBBY'] if version == 3 else [
            'R14_T', 'R09', '--entrance-nav-only', '--ros-args', '-p',
            'use_sim_time:=true']
        return module, argv

    def test_both_guards_reject_changed_shared_code_before_using_live_audit(self):
        for version in (3, 4):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                module, argv = self.fixture(folder, version)
                environment = {f'DN011_VISIBILITY_EXECUTE_V{version}': '1',
                               'DN011_VISIBILITY_AUDIT_REPORT': ''}
                with patch.dict(os.environ, environment):
                    with self.assertRaisesRegex(RuntimeError, 'Set DN011_VISIBILITY_AUDIT_REPORT'):
                        module.Guard(folder / 'observation_planning_core.py', argv)
                    for name in module.SHARED_SOURCES:
                        original = (folder / name).read_bytes()
                        (folder / name).write_bytes(original + b'\n# modified\n')
                        with self.assertRaisesRegex(RuntimeError, 'Shared computation modified'):
                            module.Guard(folder / 'observation_planning_core.py', argv)
                        (folder / name).write_bytes(original)

    def test_missing_dependency_hash_cannot_disable_validation(self):
        for version in (3, 4):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                module, argv = self.fixture(folder, version)
                receipt = folder / (module.V3_STATE if version == 3 else module.V4_STATE)
                state = json.loads(receipt.read_text())
                state['shared_sources_sha256'].pop('planning_core.py')
                receipt.write_text(json.dumps(state))
                with patch.dict(os.environ, {f'DN011_VISIBILITY_EXECUTE_V{version}': '1'}):
                    with self.assertRaisesRegex(RuntimeError, 'Missing shared computation'):
                        module.Guard(folder / 'observation_planning_core.py', argv)


if __name__ == '__main__':
    unittest.main()
