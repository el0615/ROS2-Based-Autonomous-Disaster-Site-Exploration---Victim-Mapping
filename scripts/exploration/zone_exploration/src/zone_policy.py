"""ROS-independent zone mission policy. No coordinates, percentages, or outcomes are invented.

ENTRY is transit into an incomplete zone, not an information-gain goal.
Coverage is external measured evidence. DEFERRED never means unreachable.
"""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import math
from sequence_core import MissionBook


@dataclass
class VisitState:
    entered: bool = False
    entry_goals: int = 0
    observation_goals: int = 0
    low_gain: int = 0
    retries: int = 0
    deferred_key: str | None = None
    deferred_at: float = 0.0
    deferred_progress: int = 0
    entry_arrival_retries: int = 0


class ZoneMission(MissionBook):
    def __init__(self, zones, threshold=.9, max_views=4, min_gain_pp=.2,
                 max_revisits=1, revisit_cooldown_s=30.):
        super().__init__(zones, threshold, max_views, min_gain_pp)
        self.visit_state = {z['id']: VisitState() for z in zones}
        self.active_zone: str | None = None
        self.progress = 0
        self.max_revisits = max_revisits
        self.cooldown = revisit_cooldown_s
        self.context_key = ''
        self.wall = 0.

    def context(self, cost_key: str, wall: float):
        self.context_key, self.wall = cost_key, wall

    def update(self, status):
        new = super().update(status)
        self.progress += len(new)
        if self.active_zone in new:
            self.active_zone = None
            self.locked_zone = None
        return new

    def activate(self, zone_id):
        if zone_id not in self.records:
            raise ValueError('Unknown zone')
        self.active_zone = self.locked_zone = zone_id

    def mark_entered(self, zone_id):
        self.visit_state[zone_id].entered = True

    def defer(self, zone_id, reason):
        state = self.visit_state[zone_id]
        state.deferred_key = self.context_key
        state.deferred_at = self.wall
        state.deferred_progress = self.progress
        super().defer(zone_id, reason)
        if self.active_zone == zone_id:
            self.active_zone = None
        # Never write ROBOT_UNREACHABLE or COMPLETE on a failed sample.

    def reopen_changed(self):
        reopened = []
        for zone_id, state in self.visit_state.items():
            rec = self.records[zone_id]
            if rec.state != 'PARTIAL_DEFERRED':
                continue
            changed = self.context_key != state.deferred_key
            progressed = self.progress > state.deferred_progress
            if (state.retries < self.max_revisits and (changed or progressed)
                    and self.wall - state.deferred_at >= self.cooldown):
                state.retries += 1
                state.low_gain = 0
                # Per-visit budget is renewed, cumulative evidence remains in logs.
                state.observation_goals = 0
                rec.deferred_reason = None
                rec.state = 'PARTIAL' if rec.measured > 0 else 'UNOBSERVED'
                self.rejected = [r for r in self.rejected if r['zone_id'] != zone_id]
                reopened.append(zone_id)
        return reopened

    def record_visit(self, c, before, after, nav_status, reason):
        zone_id = c['zone_id']
        rec = self.records[zone_id]
        state = self.visit_state[zone_id]
        kind = c['goal_kind']
        gain = 100. * (after - before)
        record = {k: c[k] for k in ('zone_id', 'id', 'x', 'y', 'yaw')}
        record.update(goal_kind=kind, before_percent=100*before,
                      after_percent=100*after, gain_pp=gain,
                      nav_status=nav_status, cancel_reason=reason)
        self.visits.append(record)
        rec.views += 1
        self.progress += 1
        if kind == 'ENTRY':
            state.entry_goals += 1
            # Zero-gain transit does not count against the local low-gain budget.
        elif kind == 'OBSERVE':
            state.observation_goals += 1
            state.low_gain = state.low_gain+1 if gain < self.min_gain_pp else 0
        else:
            raise ValueError('Unknown goal_kind')
        if not self.is_complete(zone_id):
            self.activate(zone_id)
            if state.observation_goals >= self.max_views:
                self.defer(zone_id, 'LOCAL_VIEW_BUDGET_PARTIAL')
            elif state.low_gain >= 2:
                self.defer(zone_id, 'LOW_MEASURED_GAIN_TWICE_PARTIAL')
        return record

    def summary(self):
        out = super().summary()
        for zone_id, info in out.items():
            st = self.visit_state[zone_id]
            info.update(entered=st.entered, entry_goals=st.entry_goals,
                        observation_goals=st.observation_goals,
                        revisit_count=st.retries, low_gain_count=st.low_gain,
                        entry_arrival_retries=st.entry_arrival_retries)
        return out


def room_priority(zone, policy):
    if policy == 'nearest':
        return 0
    kind = str(zone.get('kind', '')).lower()
    ident = zone['id']
    transit = kind in {'corridor', 'lobby', 'connector', 'hall'} or ident.startswith(
        ('LOBBY', 'CORRIDOR_', 'CONNECTOR_'))
    return int(transit)


def angle_diff(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


def repeated(candidate, history, radius=.35, yaw_tolerance=.35):
    """Reject the same position AND heading; a new heading may expose more of a 270deg scan."""
    return any(p.get('zone_id') == candidate['zone_id']
               and p.get('goal_kind', 'OBSERVE') == candidate['goal_kind']
               and math.hypot(p['x']-candidate['x'], p['y']-candidate['y']) <= radius
               and angle_diff(p['yaw'], candidate['yaw']) <= yaw_tolerance
               for p in history)


LIMITED = 'OBSERVATION_LIMITED'
AUDIT_TRIGGERS = {
    'LOCAL_VIEW_BUDGET_PARTIAL', 'LOW_MEASURED_GAIN_TWICE_PARTIAL',
    'NO_USEFUL_INTERNAL_VIEW_THIS_REVISION', 'LOCAL_PREFLIGHT_EXHAUSTED_PARTIAL',
    'ENTRY_PREFLIGHT_EXHAUSTED_PARTIAL',
    'NO_SAFE_INTERIOR_POSE_NOT_UNREACHABLE_PROOF',
    'NO_DISTINCT_INTERIOR_ENTRY_THIS_REVISION',
}


def limited_certificate_ok(e: dict) -> bool:
    """Require positive, complete evidence; absence of data is never success."""
    if not isinstance(e, dict):
        return False
    flags = ('catalog_complete', 'inputs_fresh', 'context_matches',
             'connectivity_verified', 'all_potential_information_accounted')
    if not all(e.get(k) is True for k in flags):
        return False
    counts = ('positions', 'probes_succeeded', 'probes_failed',
              'unrepresented_components', 'unknown_cells',
              'unobserved_free_cells', 'remaining_gain_views')
    if any(type(e.get(k)) is not int or e[k] < 0 for k in counts):
        return False
    if e['positions'] < 1 or e['probes_succeeded'] != e['positions']:
        return False
    if any(e[k] != 0 for k in counts[2:]):
        return False
    return bool(e.get('cost_key')) and bool(e.get('bundle_signature'))


class ConnectivityMission(ZoneMission):
    """Preserve baseline thresholds; use soft limits as audit requests instead."""
    def __init__(self, *args, hard_zone_goals=20, no_auto_revisit=False, **kwargs):
        kwargs['max_revisits'] = 0
        super().__init__(*args, **kwargs)
        if type(hard_zone_goals) is not int or not 1 <= hard_zone_goals <= 100:
            raise ValueError('Invalid hard_zone_goals')
        self.hard_zone_goals = hard_zone_goals
        self.no_auto_revisit = bool(no_auto_revisit)
        self.deferred_this_mission = set()
        # P0: scheduling lock, measured coverage and accessibility are independent.
        self.deferral_history: dict[str, list[dict]] = {z['id']: [] for z in self.zones}
        self.coverage_confirmation: dict[str, dict] = {}
        self.last_coverage_evidence: dict = {}
        self.audit_requested: dict[str, str] = {}
        self.outcomes: dict[str, dict] = {}
        self.nav_abort_counts = {z['id']: 0 for z in self.zones}

    def defer(self, zone_id, reason):
        if not self.is_complete(zone_id) and reason in AUDIT_TRIGGERS:
            self.audit_requested.setdefault(zone_id, reason)
            return
        self.finish_partial(zone_id, reason)

    def finish_partial(self, zone_id, reason, *, evidence=None):
        # A budget/low-gain verdict never downgrades already-confirmed Coverage.
        if self.is_complete(zone_id):
            return
        if self.no_auto_revisit and zone_id in self.deferred_this_mission:return
        if self.no_auto_revisit and not self.is_complete(zone_id):
            self.deferred_this_mission.add(zone_id)
        rec = self.records[zone_id]
        self.deferral_history[zone_id].append({
            'reason': reason,
            'measured_percent': 100. * rec.measured,
            'state_before_deferral': rec.state,
            'coverage_session': self.session,
            'coverage_evidence_at_deferral': deepcopy(self.last_coverage_evidence),
            'cost_key': self.context_key,
            'evidence': deepcopy(evidence or {}),
        })
        super().defer(zone_id, reason)
        self.audit_requested.pop(zone_id, None)
        self.outcomes[zone_id] = {
            'classification': ('CURRENT_MAP_UNREACHABLE' if reason=='CURRENT_MAP_UNREACHABLE_FIXED_PRIOR_TOPOLOGY'
                and (evidence or {}).get('current_map_unreachable') is True else
                'MAP_BLOCKED_CANDIDATE' if reason.startswith('MAP_BLOCKED') else 'UNVERIFIED'),
            'reason': reason, 'normal_termination': False,
            'evidence': deepcopy(evidence or {}),
        }

    def finish_limited(self, zone_id, evidence):
        if self.no_auto_revisit and zone_id in self.deferred_this_mission:
            raise ValueError('DEFERRED_ZONE_LOCKED_THIS_MISSION')
        if self.is_complete(zone_id):
            return
        if evidence.get('cost_key') != self.context_key or not limited_certificate_ok(evidence):
            raise ValueError('INSUFFICIENT_CONNECTIVITY_OR_OBSERVATION_EVIDENCE')
        rec = self.records[zone_id]
        rec.state = LIMITED
        rec.deferred_reason = None
        self.outcomes[zone_id] = {
            'classification': 'CONNECTED_OBSERVATION_LIMIT',
            'normal_termination': True, 'coverage_target_met': False,
            'reason': 'CURRENT_MAP_CONNECTED_FREE_OBSERVED_OCCUPIED_RESIDUAL',
            'evidence': evidence,
        }
        self.audit_requested.pop(zone_id, None)
        if self.active_zone == zone_id:
            self.active_zone = self.locked_zone = None

    def update(self, status):
        limited_before = {z for z, r in self.records.items() if r.state == LIMITED}
        new = super().update(status)
        # Called by zone_node only after existing fresh/matched input validation.
        # Do not replace ThresholdTracker: it still requires another scan and
        # >= 1 simulation second before OBSERVATION_SUFFICIENT is emitted.
        self.last_coverage_evidence = {
            'coverage_session': status['session_id'],
            'evidence_scans': int(status['evidence_scans']),
            'last_evidence_scan_end_s': float(status['last_evidence_scan_end_s']),
        }
        for z in new:
            rec = self.records[z]
            self.coverage_confirmation[z] = {
                **self.last_coverage_evidence,
                'measured_percent': 100. * rec.measured,
                'threshold_percent': 100. * self.threshold,
                'first_above_threshold_scan_end_s': float(rec.tracker.first[0]),
                'first_above_threshold_scan_sequence': int(rec.tracker.first[1]),
                'source': 'EXISTING_THRESHOLD_TRACKER',
                'confirms_accessibility': False,
            }
        for z in limited_before:
            r = self.records[z]
            if r.state in ('OBSERVATION_SUFFICIENT', 'THRESHOLD_PENDING'):
                continue
            old_key = self.outcomes[z]['evidence'].get('cost_key')
            if old_key == self.context_key:
                r.state = LIMITED
            else:
                self.outcomes.pop(z, None)
                self.audit_requested[z] = 'CONNECTIVITY_EVIDENCE_INVALIDATED_BY_MAP_CHANGE'
        if self.no_auto_revisit:
            # A deferred Zone remains excluded from future navigation goals.
            # Passive, validated observations MAY still satisfy Coverage.
            # Preserve the parent's THRESHOLD_PENDING / OBSERVATION_SUFFICIENT
            # state and progress count; never turn Coverage into topology proof.
            for z in self.deferred_this_mission:
                self.audit_requested.pop(z, None)
        for z in new:
            self.audit_requested.pop(z, None)
        return new

    def eligible(self):
        return [z for z in super().eligible() if self.records[z['id']].state != LIMITED
                and (not self.no_auto_revisit or z['id'] not in self.deferred_this_mission)]

    def all_resolved(self):
        return all(r.state in ('OBSERVATION_SUFFICIENT', LIMITED) for r in self.records.values())

    def reopen_changed(self):
        """At most one event-driven retry for uncertain blocked-entry zones.

        A clock tick alone, another zone's progress alone, or a costmap change
        alone cannot reopen a zone. Both new costmap information and subsequent
        mission progress are required after a cooldown. This is NOT a statement
        that the previously blocked entrance is traversable.
        """
        if self.no_auto_revisit:return []
        reopened=[]
        for zid,state in self.visit_state.items():
            rec=self.records[zid]
            outcome=self.outcomes.get(zid) or {}
            if not (rec.state=='PARTIAL_DEFERRED'
                    and outcome.get('reason') in ('ENTRY_ACCESS_UNVERIFIED_ALTERNATES_EXHAUSTED',
                        'PATH_BLOCKED_NO_CONFIRMED_DETOUR','CURRENT_MAP_UNREACHABLE_FIXED_PRIOR_TOPOLOGY')
                    and state.retries<1
                    and self.context_key!=state.deferred_key
                    and self.progress>state.deferred_progress
                    and self.wall-state.deferred_at>=self.cooldown):
                continue
            state.retries+=1
            state.low_gain=0
            state.observation_goals=0
            rec.deferred_reason=None
            rec.state='PARTIAL' if rec.measured>0 else 'UNOBSERVED'
            self.rejected=[c for c in self.rejected if c.get('zone_id')!=zid]
            self.outcomes.pop(zid,None)
            self.audit_requested.pop(zid,None)
            reopened.append(zid)
        return reopened

    def record_visit(self, c, before, after, nav_status, reason):
        result = super().record_visit(c, before, after, nav_status, reason)
        z = c['zone_id']
        if not self.is_complete(z) and self.records[z].views + self.nav_abort_counts[z] >= self.hard_zone_goals:
            self.finish_partial(z, 'ZONE_GOAL_BUDGET_UNVERIFIED')
        return result

    def summary(self):
        out = super().summary()
        for z, d in out.items():
            d['deferred_locked_this_mission'] = self.no_auto_revisit and z in self.deferred_this_mission
            d['assessment'] = self.outcomes.get(z)
            d['audit_requested'] = self.audit_requested.get(z)
            d['nav_aborted'] = self.nav_abort_counts[z]
            rec = self.records[z]
            reached = rec.measured >= self.threshold
            confirmed = self.is_complete(z)
            d['coverage_target_percent'] = 100. * self.threshold
            d['coverage_target_reached'] = bool(reached)
            d['coverage_confirmed'] = bool(confirmed)
            d['coverage_status'] = ('TARGET_CONFIRMED' if confirmed else
                'TARGET_REACHED_UNCONFIRMED' if reached else 'BELOW_TARGET')
            was_deferred = bool(self.deferral_history[z])
            d['exploration_status'] = ('DEFERRED' if
                (d['deferred_locked_this_mission'] or rec.deferred_reason) else
                'COVERAGE_SUFFICIENT' if confirmed else
                'OBSERVATION_LIMITED' if rec.state == LIMITED else rec.state)
            d['coverage_at_last_deferral_percent'] = (
                self.deferral_history[z][-1]['measured_percent'] if was_deferred else None)
            d['deferral_history'] = deepcopy(self.deferral_history[z])
            d['coverage_confirmation_evidence'] = deepcopy(self.coverage_confirmation.get(z))
            d['coverage_does_not_certify_connectivity'] = True
        return out
