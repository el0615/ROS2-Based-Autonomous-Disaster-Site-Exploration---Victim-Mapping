"""ROS-independent zone mission policy. No coordinates, percentages, or outcomes are invented.

ENTRY is transit into an incomplete zone, not an information-gain goal.
Coverage is external measured evidence. DEFERRED never means unreachable.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from zb_reference_sequence import MissionBook


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
