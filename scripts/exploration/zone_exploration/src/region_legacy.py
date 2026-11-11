"""Opt-in adapter. Reuses the original Nav2 state machine and audit lattice.

No publishers/actions are constructed here. Every move uses node.plan_request
and subsequently the ORIGINAL path, endpoint footprint, local-start, fresh
sensor/TF, one-owned-goal, cancellation and stop gates.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict
import hashlib
import math
import os
import time
import numpy as np
from regions import (VERSION, RegionSettings, PersistentCells, residual_regions,
    rank_region_views, scan_supported_cells, access_analysis)
from planning_core import oriented_footprint_clear
from exploration_io import atomic_json


def make_runtime(base):
    class RegionConnectivityRuntime(base):
        def __init__(self, node, bundle, cfg, args, output):
            super().__init__(node, bundle, cfg, args, output)
            self.region_settings = RegionSettings.from_environment()
            self.region_mode = os.getenv('DN012_REGION_MODE', 'rescue')
            if self.region_mode not in ('diagnostic', 'rescue'):
                raise ValueError('DN012_REGION_MODE must be diagnostic or rescue')
            self.region_cells = PersistentCells(bundle[2].shape, self.region_settings)
            self.region_witness = np.zeros_like(bundle[3], dtype=bool)
            self.region_ledgers = {}
            self.region_reports = {}
            self.region_probe_meta = {}
            self.region_probe_results = {}
            self.region_last_tick = -math.inf
            self.region_last_inp = None
            self.region_reports_written = 0
            node.event('REGION_EVIDENCE_ADAPTER', version=VERSION, enabled=True, mode=self.region_mode,
                       settings=asdict(self.region_settings), nav2_parameters_changed=False,
                       coverage_denominator_changed=False, hard_budgets_preserved=True,
                       cut_scope='sensor-supported finite-grid fixed-prior assumption')
            node.info('region_policy', '[REGION_EVIDENCE_ADAPTER] ' + VERSION +
                      '; mode=' + self.region_mode + '; original Nav2/safety/budget gates retained', 0)

        def _ledger(self, zid):
            return self.region_ledgers.setdefault(zid, dict(navigation_goals=0,
                low_yield_positions=[], outcomes=[], rescue_triggers=0,
                selected_regions=[], local_probe_attempts=0, region_preflight_attempts={}))

        def region_tick(self, inp):
            """Called from original poll with already validated inputs; no motion."""
            self.region_last_inp = inp
            wall = time.monotonic()
            if wall - self.region_last_tick < .25:
                return
            self.region_last_tick = wall
            monitor = getattr(self.n, 'path_blockage', None)
            hits = getattr(monitor, 'hits', None)
            now = self.n.ros_s()
            if (hits is None or not -.1 <= now - hits['stamp'] <= 1. or
                    wall - hits['wall'] > 3.):
                self.region_cells.reset()
                self.region_witness.fill(False)
                return
            reference = getattr(self.n, 'region_reference_occupancy', None)
            if reference is None:
                self.region_cells.reset(); self.region_witness.fill(False)
                return
            supported = scan_supported_cells(hits['points'], self.bundle[1], reference != 100,
                inp['cost']['values'], self.region_settings.sensor_tolerance_m)
            epoch = (inp['epoch'], self.bundle[4], inp['status']['session_id'])
            self.region_witness = self.region_cells.update(supported, now, hits['stamp'],
                                                           inp['cost']['stamp'], epoch)

        def _inspect(self, zid, inp):
            zone = self.n.zones_by_id[zid]
            regions, residual = residual_regions(self.bundle, inp, zone, self.region_settings)
            access_input = dict(inp, reference_occupancy=getattr(self.n, 'region_reference_occupancy', None))
            access = access_analysis(self.bundle, access_input, zone, self.region_witness,
                self.cfg, self.region_settings,
                bool(getattr(self.args, 'prior_topology_fixed', False)),
                endpoint_checker=oriented_footprint_clear)
            for region in access['local_regions']:
                key = (zid, region['region_id'], inp['cost']['key'])
                proof = self.region_probe_results.get(key)
                if proof and math.dist(proof['start_pose'][:2], inp['pose'][:2]) <= .15:
                    region['global_route_status'] = proof['status']
                    region['nav2_probe_evidence'] = proof
            result = dict(schema='dn012_region_review_v1', zone_id=zid,
                ros_time_s=self.n.ros_s(), coverage_session=inp['status']['session_id'],
                cost_key=inp['cost']['key'], residual=residual, access=access,
                ledger=self._ledger(zid),
                note='Unobserved area and inaccessible robot-center subset are separate metrics.')
            self.region_reports[zid] = result
            return regions, result

        def publish_region_report(self):
            # Report-only hook: original publish cadence is retained. A stale input
            # never manufactures a new certificate from a cached costmap.
            err, inp = self.n.inputs()
            if err is None and inp is not None:
                zids = {getattr(self.n.book, 'active_zone', None), self.zid}
                if self.n.current:
                    zids.add(self.n.current.get('zone_id'))
                for zid in sorted(z for z in zids if z in self.n.zones_by_id):
                    self._inspect(zid, inp)
            if not self.region_reports:
                return
            folder = self.output / 'region_review'
            folder.mkdir(exist_ok=True)
            atomic_json(folder / 'latest.json', self.region_summary())
            self.region_reports_written += 1

        def region_summary(self):
            current = self.region_last_inp
            current_key = None if current is None else current['cost']['key']
            now = self.n.ros_s()
            zones = {}
            for zid, report in self.region_reports.items():
                zones[zid] = dict(report, evidence_is_historical=(
                    current_key != report['cost_key'] or now - report['ros_time_s'] > 3.))
            return dict(version=VERSION, mode=self.region_mode, settings=asdict(self.region_settings), zones=zones,
                nav2_configuration_modified=False, measured_coverage_modified=False,
                disclaimer='Grid-conditioned cut evidence, not an unrestricted physical-world proof.')

        def consider_low_gain_plateau(self, record, coverage_session, confirmed_inside):
            """Same measured sliding window, but request residual review, NOT exit."""
            zid = record['zone_id']
            ledger = self._ledger(zid)
            err, inp = self.n.inputs()
            c = self.n.current or {}
            if confirmed_inside and record.get('nav_status') == 'SUCCEEDED' and inp is not None and err is None:
                target = (self.bundle[2] == self.n.zones_by_id[zid]['mask_id']) & self.bundle[3]
                gain = max(0., float(record['gain_pp'])) / 100 * int(target.sum()) * self.bundle[1].resolution**2
                expected = float(c.get('new_area_m2', 0.))
                weak = gain < self.region_settings.low_yield_area_m2 or (expected >= .5 and gain < .1 * expected)
                if weak:
                    # Requested XY and measured terminal XY both prevent another
                    # heading at effectively the same unproductive position.
                    for xy in ((record['x'], record['y']), inp['pose'][:2]):
                        if not any(math.dist(xy, p) < .1 for p in ledger['low_yield_positions']):
                            ledger['low_yield_positions'].append([float(xy[0]), float(xy[1])])
                outcome = dict(candidate_id=record['id'], actual_zone_gain_m2=gain,
                    predicted_view_gain_m2=expected, low_yield=weak,
                    residual_region_id=c.get('residual_region_id'),
                    measurement_scope='whole-zone change during motion and fresh post-stop scan')
                ledger['outcomes'].append(outcome)
                self.n.event('REGION_VIEW_OUTCOME', zone=zid, **outcome)
                self._inspect(zid, inp)
            if self.region_mode == 'diagnostic':
                return super().consider_low_gain_plateau(record, coverage_session, confirmed_inside)
            if (not confirmed_inside or record.get('goal_kind') != 'OBSERVE' or
                    record.get('nav_status') != 'SUCCEEDED'):
                self.reset_low_gain_history(zid)
                return False
            rec = self.n.book.records[zid]
            if self.n.book.is_complete(zid) or rec.measured >= self.cfg.threshold or rec.state == 'PARTIAL_DEFERRED':
                self.reset_low_gain_history(zid)
                return False
            window = int(getattr(self.args, 'low_gain_stop_window', 3))
            if window == 0:
                return False
            threshold = float(getattr(self.args, 'low_gain_stop_total_pp', 1.))
            history = self.low_gain_history.get(zid)
            if history is None or any(x['session'] != coverage_session for x in history):
                history = deque(maxlen=window)
                self.low_gain_history[zid] = history
            before, after = float(record['before_percent']), float(record['after_percent'])
            ident = str(record.get('id', ''))
            if (not ident or not all(math.isfinite(v) and -.001 <= v <= 100.001 for v in (before, after)) or
                    after < before - 1e-5 or (history and (before < history[-1]['after'] - 1e-5 or
                    any(x['id'] == ident for x in history)))):
                self.reset_low_gain_history(zid)
                return False
            history.append(dict(id=ident, session=coverage_session, before=before, after=after))
            if len(history) < window or after - history[0]['before'] >= threshold - 1e-8:
                return False
            ledger['rescue_triggers'] += 1
            self.n.event('REGION_LOW_GAIN_REVIEW_REQUESTED', zone=zid,
                total_gain_pp=after - history[0]['before'], threshold_total_pp=threshold,
                goal_ids=[r['id'] for r in history],
                distinct_goal_ids_do_not_imply_distinct_positions=True,
                decision='KEEP_ZONE_LOCK_AND_REVIEW_RESIDUAL_REGIONS')
            # Do not override safety or hard-budget deferrals. The caller checks
            # these before this hook; the existing finish_partial lock is retained.
            self.n.book.activate(zid)
            self.n.book.audit_requested.setdefault(zid, 'RESIDUAL_REGION_RECHECK_AFTER_LOW_GAIN')
            self.reset_low_gain_history(zid)
            return True

        def audit_observation_goal_sent(self, c):
            if c.get('region_investigation'):
                ledger = self._ledger(c['zone_id'])
                ledger['navigation_goals'] += 1
                ledger['selected_regions'].append(c['residual_region_id'])
                self.n.event('REGION_INVESTIGATION_NAV_SENT', zone=c['zone_id'],
                    region_id=c['residual_region_id'], candidate_id=c['id'],
                    residual_target_cells=c['residual_target_cells'],
                    predicted_overlap_cells=c['residual_overlap_cells'],
                    distance_to_region_m=c['residual_distance_m'],
                    region_progress_m=c['residual_progress_m'],
                    region_goal_count=ledger['navigation_goals'])
            super().audit_observation_goal_sent(c)

        def _offer_local_probe(self, inp, review):
            """Planning-only global detour check for independently supported local cuts."""
            zid = self.zid
            ledger = self._ledger(zid)
            if ledger['local_probe_attempts'] >= 4 or not self.n.local_start_clear(inp):
                return False
            for region in review['access']['local_regions']:
                if region['classification'] != 'SENSOR_SUPPORTED_LOCAL_CUT':
                    continue
                key = (zid, region['region_id'], inp['cost']['key'])
                if key in self.region_probe_results:
                    continue
                runs = region['cell_runs_ros']
                candidates = [p for p in self.catalog['positions'] if any(
                    p['cell'][1] == y and a <= p['cell'][0] < b for y, a, b in runs)]
                if not candidates:
                    continue
                pos = min(candidates, key=lambda p: p['probe']['path_est_m'])
                candidate = dict(pos['probe'], connectivity_probe=True, region_access_probe=True)
                pid = candidate['audit_position_id']
                self.region_probe_meta[pid] = (key, list(inp['pose']))
                self.region_probe_results[key] = dict(status='PROBE_REQUESTED', start_pose=list(inp['pose']))
                ledger['local_probe_attempts'] += 1
                self.probe_count += 1
                self.total_probes += 1
                self.n.event('REGION_LOCAL_DETOUR_PROBE', zone=zid, region_id=region['region_id'],
                             position_id=pid, planning_only=True)
                self.n.plan_request(candidate, inp)
                return True
            return False

        def probe_success(self, c, inp, points, length):
            super().probe_success(c, inp, points, length)
            pid = c.get('audit_position_id')
            meta = self.region_probe_meta.pop(pid, None) if c.get('region_access_probe') else None
            if meta and self.probes.get(pid, {}).get('success') is True:
                key, start_pose = meta
                self.region_probe_results[key] = dict(status='NAV2_GLOBAL_ROUTE_PLANNED_NOT_EXECUTED',
                    start_pose=start_pose, cost_key=inp['cost']['key'], ros_time_s=self.n.ros_s(),
                    path_length_m=float(length),
                    path_left_target_zone=self.probes[pid]['path_left_target_zone'],
                    physical_traversal_verified=False)

        def probe_failure(self, c, reason):
            pid = c.get('audit_position_id')
            meta = self.region_probe_meta.pop(pid, None) if c.get('region_access_probe') else None
            super().probe_failure(c, reason)
            if meta:
                key, start_pose = meta
                self.region_probe_results[key] = dict(status='NAV2_ACCESS_UNVERIFIED',
                    reason=reason, start_pose=start_pose, physical_blockage_proven=False)

        def _finish_regions(self, reason, result, extra=None):
            evidence = dict(schema='dn012_residual_assessment_v1', normal_termination=False,
                connectivity_verified=False, all_potential_information_accounted=False,
                scope='bounded residual-region investigation; unresolved evidence retained',
                region_review=result, **(extra or {}))
            zid = self.zid
            self.n.event('REGION_REVIEW_FINISHED', zone=zid, reason=reason,
                         remaining_area_m2=result['residual']['remaining_area_m2'])
            super().finish_partial(reason, evidence)

        def residual_step(self, inp):
            """Called by the original step AFTER fresh catalog + ENTRY handling.

            Inherited async build/context validation, audit timeout, entry recovery,
            path blockage replans and pending Nav2 ownership are untouched.
            """
            if self.region_mode == 'diagnostic':
                return None  # Original audit selection remains active.
            n = self.n
            if self.zid is None or self.catalog is None:
                raise RuntimeError('RESIDUAL_STEP_REQUIRES_VALID_AUDIT_CATALOG')
            zid = self.zid
            ledger = self._ledger(zid)
            regions, review = self._inspect(zid, inp)
            if self.probe_count >= self.settings.max_probes:
                self._finish_regions('RESIDUAL_PROBE_BUDGET_UNVERIFIED', review)
                return True
            if self._offer_local_probe(inp, review):
                return True
            if not regions:
                # Small remnants are explicitly recorded, not removed from the
                # denominator and not turned into a fabricated 96% certificate.
                self._finish_regions('NO_SIGNIFICANT_RESIDUAL_SMALL_REMAINDER_UNVERIFIED', review)
                return True
            if ledger['navigation_goals'] >= self.region_settings.max_rescue_goals:
                self._finish_regions('RESIDUAL_NAV_BUDGET_UNVERIFIED', review)
                return True
            limit = self.args.audit_max_view_preflights
            if self.view_preflight_attempts >= limit:
                self._finish_regions('RESIDUAL_VIEW_PREFLIGHT_BUDGET_UNVERIFIED', review,
                    dict(view_preflights=self.view_preflight_attempts, max_view_preflights=limit))
                return True
            history = list(n.book.visits) + list(n.book.rejected)
            held = {pid for pid, failures in self.position_view_failures.items() if len(failures) >= 2}
            available, stats = rank_region_views(self.catalog, regions, inp, self.bundle[1],
                history, ledger['low_yield_positions'], self.region_settings, held)
            n.event('REGION_VIEW_CATALOG', zone=zid, **stats,
                    significant_regions=len(regions),
                    actual_residual_area_m2=review['residual']['remaining_area_m2'])
            if not available:
                self._finish_regions('RESIDUAL_NO_DISTINCT_PROGRESS_VIEW_UNVERIFIED', review, stats)
                return True
            # Rotate after two trials per residual component instead of spending
            # the entire 24-preflight budget on one unreachable component.
            attempts = ledger['region_preflight_attempts']
            available.sort(key=lambda c: attempts.get(c['residual_region_id'], 0) // 2)
            candidate = available[0]
            # A meaningful residual is a sensing/investigation obligation, not an
            # exemption from local-start safety or the original normal preflight.
            if not n.local_start_clear(inp):
                n.request_stop('LOCAL_START_NOT_CLEAR_OR_STALE')
                return True
            self.view_preflight_attempts += 1
            rid = candidate['residual_region_id']
            attempts[rid] = attempts.get(rid, 0) + 1
            n.event('REGION_VIEW_SELECTED', zone=zid, candidate_id=candidate['id'],
                region_id=candidate['residual_region_id'], xy=[candidate['x'], candidate['y']],
                target_area_m2=candidate['residual_target_cells'] * self.bundle[1].resolution**2,
                view_preflight=self.view_preflight_attempts, max_view_preflights=limit)
            n.plan_request(candidate, inp)
            return True

    RegionConnectivityRuntime.__name__ = 'RegionConnectivityRuntime'
    return RegionConnectivityRuntime
