#!/usr/bin/env python3
"""Guarded TWO-GOAL MULTI-ZONE fast visibility trial. Unknown input -> ORIGINAL.

This is only an optimization guard. Navfn/Nav2, collisions, physical stop must be
independently supervised; neither an audit nor a plan certifies robot safety.
"""
from __future__ import annotations
import atexit
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

V1_STATE = '.DN011_visibility_optin_v1_install.json'
V3_STATE = '.DN011_visibility_execute_v3_install.json'
V4_STATE = '.DN011_visibility_multizone_v4_install.json'
AUDIT_SCHEMA = 'DN011_LIVE_VISIBILITY_EQUIVALENCE_V2_20261101'
BASE_SHA = '4af92fbce0e6a528913e9dcd7d9b436f8271a122e0daba5364eb59f6bf19e093'
SHARED_SOURCES = ('planning_core.py', 'sequence_core.py', 'exploration_io.py', 'observation_sequence_core.py', 'observation_goal_planner.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure(predicate, msg):
    if not predicate:
        raise RuntimeError('[V4_EXECUTE_REFUSED] ' + msg)


def parse_multizone_args(argv):
    ensure('--execute' in argv, '--execute expected')
    ensure(os.environ.get('DN011_VISIBILITY_EXECUTE_V4') == '1',
           'Explicit DN011_VISIBILITY_EXECUTE_V4=1 is required')
    ensure('--inflation-aware-dry-run' not in argv,
           'Inflation-aware alternate preflight may not execute')
    for forbidden in ('--navfn-fallback', '--experimental-fallback-execute',
                      '--cancel-on-coverage'):
        ensure(forbidden not in argv, 'Forbidden pilot option ' + forbidden)
    ensure(argv.count('--max-goals') == 1 and argv.count('--zone-ids') == 1,
           'Exactly one goal-limit and zone-list argument is required')
    ensure(argv.count('--execute') == 1, 'Exactly one --execute is required')
    # We intentionally require an explicit non-lobby *multi-zone* list in the
    # first bounded pilot.  Goal count != number of distinct rooms visited.
    try:
        boundary = argv.index('--ros-args') if '--ros-args' in argv else len(argv)
        app_args = argv[:boundary]
        idx = app_args.index('--max-goals')
        ensure(app_args[idx + 1] == '2', 'V4 pilot requires --max-goals 2')
        idx = app_args.index('--zone-ids')
        zone_ids = []
        for value in app_args[idx + 1:]:
            if value.startswith('--'):
                break
            zone_ids.append(value)
        ensure(2 <= len(zone_ids) <= 8, 'V4 pilot requires 2..8 explicit zone IDs')
        ensure('LOBBY' not in zone_ids, 'V4 pilot excludes LOBBY')
        ensure(len(set(zone_ids)) == len(zone_ids), 'Duplicate zone IDs')
        ensure(set(zone_ids) <= {'R14_T','R09','R10A','R02B'},
               'This pilot is restricted to R14_T, R09, R10A, R02B')
    except (ValueError, IndexError):
        raise RuntimeError('[V4_EXECUTE_REFUSED] Explicit --max-goals 2 and '
                           '--zone-ids with 2..4 approved non-lobby zones required') from None
    # Keep the current 90% zone completion policy unchanged.
    for flag, value in (('--threshold', '0.9'), ('--max-views-per-zone','3')):
        if flag in app_args:
            try:
                actual = app_args[app_args.index(flag) + 1]
                if flag == '--threshold':
                    ensure(float(actual) == 0.9, 'The 90% threshold must remain unchanged')
                else:
                    ensure(actual == value, 'Per-zone viewpoint limit must remain 3')
            except (ValueError, IndexError):
                raise RuntimeError('[V4_EXECUTE_REFUSED] Invalid ' + flag) from None
    if '--planner-id' in app_args:
        try:
            ensure(app_args[app_args.index('--planner-id')+1] == 'GridBased',
                   'Use the approved Navfn GridBased planner')
        except IndexError:
            raise RuntimeError('[V4_EXECUTE_REFUSED] Missing planner ID') from None
    # No hidden additional execution options or experimental fallback features.
    ensure('--entrance-nav-only' in app_args,
           'This pilot requires the already tested entrance overlay')
    ensure('--ros-args' in argv and '-p' in argv and 'use_sim_time:=true' in argv,
           'Gazebo simulation clock required')
    return tuple(zone_ids)



class Guard:
    def __init__(self, planning_file, argv):
        self.zone_ids = parse_multizone_args(argv)
        self.planning_file = Path(planning_file).resolve()
        root = self.planning_file.parent
        v1 = root / V1_STATE
        v3 = root / V3_STATE
        v4 = root / V4_STATE
        ensure(v1.is_file() and v3.is_file() and v4.is_file(), 'Missing v1/v3/v4 installation state')
        s1 = json.loads(v1.read_text())
        s3 = json.loads(v3.read_text())
        s4 = json.loads(v4.read_text())
        ensure(s1.get('original_sha256') == BASE_SHA, 'Unexpected original source')
        ensure(s3.get('previous_sha256') == s1.get('patched_sha256'),
               'v3 was not installed over the expected v1 source')
        ensure(s4.get('previous_sha256') == s3.get('patched_sha256'),
               'v4 was not installed over the expected v3 source')
        ensure(digest(self.planning_file) == s4.get('patched_sha256'),
               'Planning source modified after guarded v4 install')
        ensure(digest(root / 'visibility_execute_guard_v3.py') == s3.get('guard_sha256'),
               'v3 guard module modified after installation')
        ensure(digest(root / 'visibility_execute_guard_v4.py') == s4.get('guard_sha256'),
               'v4 guard module modified after installation')
        dependencies = s4.get('shared_sources_sha256', {})
        ensure(set(dependencies) == set(SHARED_SOURCES),
               'Missing shared computation installation hashes')
        for name in SHARED_SOURCES:
            ensure(digest(root / name) == dependencies[name],
                   'Shared computation modified after installation: ' + name)
        report_setting = os.environ.get('DN011_VISIBILITY_AUDIT_REPORT', '')
        ensure(report_setting, 'Set DN011_VISIBILITY_AUDIT_REPORT to the PASS JSON')
        report_path = Path(report_setting).expanduser().resolve()
        ensure(report_path.is_file(), 'Audit report does not exist: ' + str(report_path))
        report = json.loads(report_path.read_text())
        ensure(report.get('schema') == AUDIT_SCHEMA, 'Unsupported audit schema')
        ensure(report.get('audit_status') == 'PASS' and report.get('final_manager_state') == 'DRY_RUN_PASS',
               'Audit was not a completed successful dry run')
        ensure(report.get('manager_exit_code') == 0 and report.get('manager_failure') is None,
               'Audit manager failed')
        ensure(report.get('results', {}).get('calls', 0) >= 3960 and
               report.get('results', {}).get('mismatches') == 0 and
               report.get('results', {}).get('last_error') is None,
               'Full equivalence audit is missing or incomplete')
        ensure(report.get('last_plan', {}).get('validation_mode') == 'STRICT',
               'Audit did not include STRICT Navfn route')
        ensure(report.get('manager_summary', {}).get('sent') == 0,
               'Audit unexpectedly sent a Goal')
        ensure(report.get('mode') == 'DRY_RUN_ONLY', 'Audit mode differs')
        ensure(Path(report.get('project', '')).resolve() == root.parents[1],
               'Audit belongs to a different project')
        expected = report.get('source_hashes', {})
        actual = {
            'manager_sha256': digest(root / 'hospital_observation_sequence.py'),
            'planning_sha256': s1['patched_sha256'],
            'accelerator_python_sha256': digest(root / 'visibility_fast.py'),
            'accelerator_shared_library_sha256': digest(root / 'libdn011_visibility.so'),
        }
        ensure(expected == actual, 'Audit source hashes differ from installed source')
        self.known_scenarios = frozenset(report.get('results', {}).get('scenarios', {}).keys())
        ensure(self.known_scenarios, 'No audited scenario signatures')
        self._array_hash_cache = {}
        self._scenario_count = {}
        self.fast_calls = 0
        self.reference_calls = 0
        self.spot_checks = 0
        self.mismatches = 0
        self.warned = set()
        atexit.register(self.summary)
        print(f'[V4_MULTI_ZONE_GUARD_READY] audited_calls={report["results"]["calls"]} '
              f'audited_scenarios={len(self.known_scenarios)} max_goals=2 zones={",".join(self.zone_ids)} '
              f'UNKNOWN_SCENE=ORIGINAL; navfn_fallback=OFF', flush=True)

    def array_hash(self, arr):
        idx = id(arr)
        pair = self._array_hash_cache.get(idx)
        if pair is not None:
            ensure(pair[0] is arr, 'Array identity changed unexpectedly')
            return pair[1]
        checksum = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        self._array_hash_cache[idx] = (arr, checksum)
        return checksum

    def signature(self, geom, free, target, sensor, range_cap):
        geometry = (geom.width, geom.height, float(geom.resolution),
                    tuple(float(v) for v in geom.origin))
        sensor_json = json.dumps(asdict(sensor), sort_keys=True)
        serialized = json.dumps([geometry, self.array_hash(free), self.array_hash(target),
                                 sensor_json, float(range_cap)], sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def __call__(self, original, accelerated, geom, free, target, x, y, yaw, sensor, range_cap):
        signature = self.signature(geom, free, target, sensor, range_cap)
        if signature not in self.known_scenarios:
            self.reference_calls += 1
            if signature not in self.warned:
                self.warned.add(signature)
                print('[V4_REFERENCE_FALLBACK] Unseen grid/sensor/target scenario; '
                      'using original Python calculation, NOT unaudited C acceleration', flush=True)
            return original(geom, free, target, x, y, yaw, sensor, range_cap)
        accelerated_result = accelerated(geom, free, target, x, y, yaw, sensor, range_cap)
        count = self._scenario_count.get(signature, 0) + 1
        self._scenario_count[signature] = count
        # Defense in depth for future numerical corner cases. Audit covered the
        # previous session; only the start and periodic calls are cross-checked.
        if count <= 3 or count % 128 == 0:
            reference = original(geom, free, target, x, y, yaw, sensor, range_cap)
            self.spot_checks += 1
            if not np.array_equal(reference, accelerated_result):
                self.mismatches += 1
                raise RuntimeError('[V4_VISIBILITY_MISMATCH] Halt planning; fast/ref visibility differs')
        self.fast_calls += 1
        return accelerated_result

    def summary(self):
        print(f'[V4_VISIBILITY_SUMMARY] fast_calls={self.fast_calls} '
              f'original_fallback_calls={self.reference_calls} '
              f'spot_checks={self.spot_checks} mismatches={self.mismatches}', flush=True)
