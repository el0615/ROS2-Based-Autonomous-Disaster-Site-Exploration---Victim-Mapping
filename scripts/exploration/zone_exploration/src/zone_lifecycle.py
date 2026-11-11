"""Zone lifecycle, completion evidence and saved terminal-state continuity.

Owns completion decisions used by checkpoint selection and mission resume.
Persisted versioned keys remain unchanged for compatibility with saved sessions.
A Zone lifecycle is distinct from measured Coverage and local accessibility.
An operator-verified legacy false completion is reconciled ONLY for the
identified 2026-11-10 mission/coverage lineage and only if its saved evidence
matches the reviewed case. Never modify the signed legacy checkpoint itself.

The underlying Nav2, stopped-odometry, freshness and goal ownership gates
remain in the original zone_node.py. This module never authorizes motion.
"""
from __future__ import annotations
from copy import deepcopy
import math

SCHEMA = 'dn012_explicit_zone_lifecycle_v1_8_7'
STATUS = frozenset({'NOT_STARTED', 'IN_PROGRESS', 'COMPLETE'})
ACCESS = frozenset({'UNKNOWN', 'ACCESSIBLE', 'INACCESSIBLE'})
PHASE = frozenset({'PRIMARY_ENTRY', 'PRIMARY_OBSERVE', 'SECOND', 'TERMINAL'})
LEGACY_MISSION = frozenset({
    'session_20261110_131743_816532',
    'session_20261110_133731_776040',
})
LEGACY_T5 = 'session_20261110_130514_234118'
LEGACY_CORRIDOR = 'CORRIDOR_E'


def _str(text):
    return isinstance(text, str) and bool(text.strip())


def validate_table(mission):
    """Check exact Zone scope, fields and terminal evidence before trusting v187."""
    table = mission.get('zone_lifecycle_v187')
    if not isinstance(table, dict) or table.get('schema') != SCHEMA:
        raise ValueError('V187_LIFECYCLE_SCHEMA_INVALID')
    zones = table.get('zones')
    scope = mission.get('scope')
    if (not isinstance(scope, list) or not scope or len(scope) != len(set(scope)) or
            not isinstance(zones, dict) or set(zones) != set(scope)):
        raise ValueError('V187_LIFECYCLE_ZONE_SCOPE_INVALID')
    for zid, item in zones.items():
        if not isinstance(item, dict):
            raise ValueError('V187_LIFECYCLE_RECORD_INVALID:' + zid)
        if item.get('exploration_status') not in STATUS or item.get('access_status') not in ACCESS:
            raise ValueError('V187_LIFECYCLE_STATUS_INVALID:' + zid)
        if item.get('phase') not in PHASE:
            raise ValueError('V187_LIFECYCLE_PHASE_INVALID:' + zid)
        reason = item.get('completion_reason')
        if item['exploration_status'] == 'COMPLETE':
            if not _str(reason) or item.get('phase') != 'TERMINAL':
                raise ValueError('V187_LIFECYCLE_COMPLETE_WITHOUT_REASON:' + zid)
        elif reason is not None:
            raise ValueError('V187_LIFECYCLE_NONCOMPLETE_WITH_REASON:' + zid)
        fraction = item.get('coverage_fraction')
        if (type(fraction) not in (int, float) or not math.isfinite(fraction) or
                not 0 <= fraction <= 1):
            raise ValueError('V187_LIFECYCLE_COVERAGE_INVALID:' + zid)
    return table


def is_exact_legacy_false_corridor(mission, coverage_source=None):
    """Operator-verified special correction, tightly bound to real uploaded logs.

    A routine later/new mission with the same Zone name MUST NOT inherit this
    exception. We require the original T6 and T5 session, zero confirmed entry,
    zero observed goals, <96% and the known unfinished two-region preflight.
    """
    if mission.get('zone_lifecycle_v187') is not None:
        return False
    if mission.get('source_session') not in LEGACY_MISSION:
        return False
    if coverage_source is not None and coverage_source != LEGACY_T5:
        return False
    if mission.get('counters', {}).get('sent_goals') != 52:
        return False
    rec = (mission.get('records') or {}).get(LEGACY_CORRIDOR) or {}
    visit = (mission.get('visit_states') or {}).get(LEGACY_CORRIDOR) or {}
    reviews = ((mission.get('connectivity') or {}).get('local_reviews') or {})
    results = (reviews.get(LEGACY_CORRIDOR) or {}).get('results') or []
    if (rec.get('state') != 'EXPLORATION_COMPLETE' or
            rec.get('views') != 0 or visit.get('entered') is not False or
            float(rec.get('measured', 1)) >= .96):
        return False
    return (len(results) >= 2 and all(isinstance(x, dict) and
            x.get('status') == 'UNVERIFIED' and
            x.get('reason') == 'SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES'
            for x in results))


def legacy_completion_facts(mission):
    """Backward compatibility. All legacy finished Zones remain finished, except
    the explicitly operator-confirmed false CORRIDOR_E from this exact lineage.
    """
    scope = mission.get('scope') or []
    records = mission.get('records') or {}
    if not isinstance(scope, list) or set(scope) != set(records):
        raise ValueError('V187_LEGACY_ZONE_RECORDS_MISMATCH')
    explicit = set((mission.get('completion_v183') or {}).get('finished_zones', []))
    reviewed = set((mission.get('connectivity') or {}).get('local_reviewed', []))
    results = {}
    for zid in scope:
        r = records[zid]
        measured = r.get('measured')
        if type(measured) not in (float, int) or not math.isfinite(measured) or not 0 <= measured <= 1:
            raise ValueError('V187_LEGACY_MEASURED_INVALID:' + zid)
        if measured >= .96 - 1e-12:
            results[zid] = 'COVERAGE_96_REACHED'
        elif zid in explicit or r.get('state') in ('EXPLORATION_COMPLETE', 'OBSERVATION_LIMITED'):
            results[zid] = 'SAVED_TERMINAL_ZONE'
        elif zid in reviewed:
            results[zid] = 'LOCAL_REVIEW_FINISHED'
    if is_exact_legacy_false_corridor(mission):
        results.pop(LEGACY_CORRIDOR, None)
    return results


def completion_facts_v187(mission):
    if mission.get('zone_lifecycle_v187') is not None:
        table = validate_table(legacy_upgrade(mission))
        return {z: v['completion_reason'] for z, v in table['zones'].items()
                if v['exploration_status'] == 'COMPLETE'}
    return legacy_completion_facts(mission)


def _old_access(zid, mission):
    # Retain only a categorical historical result, never convert it to a fresh
    # physical proof or a live costmap. The legacy history remains elsewhere.
    r = (mission.get('visit_states') or {}).get(zid) or {}
    if r.get('entered') is True:
        return 'ACCESSIBLE'
    outcome = (mission.get('outcomes') or {}).get(zid) or {}
    if outcome.get('classification') == 'INACCESSIBLE':
        return 'INACCESSIBLE'
    diag = outcome.get('historical_diagnosis') or {}
    if diag.get('classification') == 'INACCESSIBLE':
        return 'INACCESSIBLE'
    return 'UNKNOWN'


def legacy_upgrade(mission, *, source_coverage=None, source_digest=None):
    """Pure in-memory migration; original checkpoint and hashes stay unchanged."""
    if mission.get('zone_lifecycle_v187') is not None:
        validate_table(mission)
        upgraded = deepcopy(mission)
        conn = mission.get('connectivity') or {}
        # v1.8.9.2: preserve already-finished reviews on Resume, including the
        # former no-entry + UNVERIFIED case. Do not rewrite saved diagnostics.
        for zid in conn.get('local_reviewed', []):
            if conn.get('stage') == 'SECOND' and conn.get('local_second_zone') == zid:
                continue
            item = upgraded['zone_lifecycle_v187']['zones'].get(zid)
            if item is None or item['exploration_status'] == 'COMPLETE':
                continue
            item.update(exploration_status='COMPLETE', phase='TERMINAL',
                completion_reason='ZONE_LOCAL_PROCEDURE_FINISHED',
                access_status=('ACCESSIBLE' if item.get('entry_confirmed')
                               else _old_access(zid, mission)),
                provenance='SAVED_LOCAL_PROCEDURE_FINISHED_V1892')
        return upgraded
    if source_coverage is not None and (source_coverage != LEGACY_T5 and
            is_exact_legacy_false_corridor(mission)):
        raise ValueError('V187_LEGACY_CORRIDOR_T5_MISMATCH')
    facts = legacy_completion_facts(mission)
    exception = is_exact_legacy_false_corridor(mission, source_coverage)
    table = {}
    for zid in mission['scope']:
        rec = mission['records'][zid]
        seen = (mission.get('visit_states') or {}).get(zid) or {}
        done = zid in facts
        if done:
            # Preserve legacy terminal decisions without treating secondary
            # UNVERIFIED/low-Coverage as cancellation of Zone completion.
            # Explain *why* the Zone was closed as explicitly as old evidence
            # permits; do not pretend a historical entry was measured anew.
            if rec['measured'] >= .96 - 1e-12:
                reason = 'COVERAGE_96_REACHED'
            elif seen.get('entered') is True or int(rec.get('views', 0)) > 0:
                reason = 'SECOND_PASS_OR_LOCAL_PROCEDURE_FINISHED'
            elif _old_access(zid, mission) == 'INACCESSIBLE':
                reason = 'ACCESS_UNAVAILABLE_RECORDED'
            else:
                reason = 'HISTORICAL_ZONE_PROCEDURE_TERMINAL'
            phase = 'TERMINAL'
            state = 'COMPLETE'
        else:
            reason = None
            phase = 'PRIMARY_ENTRY' if seen.get('entered') is not True else 'PRIMARY_OBSERVE'
            state = 'NOT_STARTED' if zid == LEGACY_CORRIDOR and exception else 'IN_PROGRESS'
        access = 'UNKNOWN' if zid == LEGACY_CORRIDOR and exception else _old_access(zid, mission)
        table[zid] = dict(exploration_status=state, access_status=access,
            completion_reason=reason, phase=phase,
            coverage_fraction=float(rec['measured']),
            entry_confirmed=(seen.get('entered') is True) and not (zid == LEGACY_CORRIDOR and exception),
            historical_evidence=dict(
                old_record_state=rec.get('state'),
                legacy_local_reviewed=(zid in (mission.get('connectivity') or {}).get('local_reviewed', [])),
                historical_entry_confirmed=(seen.get('entered') is True),
                historical_observe_views=int(rec.get('views', 0)),
                historical_outcome_reason=((mission.get('outcomes') or {}).get(zid) or {}).get('reason')),
            provenance=('OPERATOR_CONFIRMED_NO_MEANINGFUL_ENTRY_20261110' if zid == LEGACY_CORRIDOR and exception
                        else 'PRESERVED_LEGACY_CHECKPOINT' if done else 'INFERRED_LEGACY_UNFINISHED'))
    upgraded = deepcopy(mission)
    upgraded['zone_lifecycle_v187'] = dict(schema=SCHEMA, zones=table,
        migration=dict(original_session=mission.get('source_session'),
            source_coverage_session=source_coverage, source_payload_sha256=source_digest,
            legacy_finished_count=sum(1 for r in mission['records'].values()
                if r.get('state') in ('EXPLORATION_COMPLETE','OBSERVATION_SUFFICIENT','OBSERVATION_LIMITED')),
            special_correction=['CORRIDOR_E'] if exception else [],
            original_checkpoint_not_modified=True,
            old_evidence_kept_in_mission_history=True))
    validate_table(upgraded)
    return upgraded


def review_terminal(node, zid):
    """Procedure end is COMPLETE, independently of entry, Coverage or verdict.

    The existing local_reviewed marker is written by _end_local. It records
    procedure end, not physical accessibility. UNVERIFIED/BUDGET remain evidence.
    """
    ledger = getattr(node, '_v187_lifecycle', None)
    if ledger is not None and ledger.zones[zid]['exploration_status'] == 'COMPLETE':
        return True
    if node.book.records[zid].measured + 1e-12 >= float(node.book.threshold):
        return True
    rt = getattr(node, 'connectivity', None)
    if rt is None:
        return False
    if getattr(rt, 'exploration_stage', '') == 'SECOND' and getattr(rt, 'local_second_zone', None) == zid:
        return False
    return zid in getattr(rt, 'local_reviewed', set())


class ZoneLifecycle:
    def __init__(self, scope):
        self.scope = tuple(scope)
        self.zones = {z:dict(exploration_status='NOT_STARTED',access_status='UNKNOWN',
            completion_reason=None,phase='PRIMARY_ENTRY',coverage_fraction=0.,
            entry_confirmed=False,provenance='NEW_MISSION') for z in self.scope}
        self.migration = None

    def load(self, mission):
        data = legacy_upgrade(mission)
        table = validate_table(data)
        if set(table['zones']) != set(self.scope):
            raise ValueError('V187_RESTORE_SCOPE_CHANGED')
        self.zones = deepcopy(table['zones'])
        self.migration = deepcopy(table.get('migration'))

    def event(self, kind, fields):
        zid = fields.get('zone') or fields.get('zone_id')
        if zid not in self.zones:
            return
        rec = self.zones[zid]
        if rec['exploration_status'] == 'COMPLETE':
            return
        if kind == 'ENTRY_CONFIRMED':
            rec.update(exploration_status='IN_PROGRESS',entry_confirmed=True,
                access_status='ACCESSIBLE',phase='PRIMARY_OBSERVE')
        elif kind == 'LOCAL_SECOND_PASS_START':
            rec['exploration_status']='IN_PROGRESS'
            rec['phase']='SECOND'
        elif kind in ('NAV_REQUESTED','SECOND_PASS_NAV_SENT'):
            rec['exploration_status']='IN_PROGRESS'
            if fields.get('goal_kind') == 'OBSERVE':rec['phase']='PRIMARY_OBSERVE'
        elif kind in ('ZONE_EXPLORATION_COMPLETE',):
            # A legacy completion event alone is not authoritative. sync() below
            # checks current review gates before recording COMPLETE.
            if rec['exploration_status'] == 'NOT_STARTED':
                rec['exploration_status'] = 'IN_PROGRESS'

    def sync(self, node):
        book = node.book
        rt = getattr(node, 'connectivity', None)
        finished = set(getattr(node,'_v183_finished_zones',set()))
        for zid in self.scope:
            rec = self.zones[zid]
            measured = float(book.records[zid].measured)
            if not math.isfinite(measured) or not 0 <= measured <= 1:
                raise ValueError('V187_NONFINITE_MEASURED:' + zid)
            rec['coverage_fraction'] = measured
            if rec['exploration_status'] == 'COMPLETE':
                continue  # irreversible within this mission, even if subregion was UNVERIFIED
            entered = getattr(book.visit_state[zid], 'entered', False)
            if entered:
                rec['entry_confirmed']=True
                rec['access_status']='ACCESSIBLE'
                if rec['exploration_status']=='NOT_STARTED':rec['exploration_status']='IN_PROGRESS'
            if measured + 1e-12 >= float(book.threshold):
                rec.update(exploration_status='COMPLETE', completion_reason='COVERAGE_96_REACHED',
                    phase='TERMINAL',provenance='FRESH_MEASURED_COVERAGE')
                continue
            if zid in finished and review_terminal(node,zid):
                status = ('ACCESSIBLE' if rec['entry_confirmed'] else
                          _old_access(zid, {'outcomes': getattr(book, 'outcomes', {})}))
                rec.update(exploration_status='COMPLETE', access_status=status,
                    completion_reason='ZONE_LOCAL_PROCEDURE_FINISHED' if rec['entry_confirmed']
                        else 'BOUNDED_ACCESS_REVIEW_FINISHED', phase='TERMINAL',
                    provenance='LIVE_REVIEW_TERMINAL')
            elif rt is not None and zid in getattr(rt,'local_reviewed',set()) and review_terminal(node,zid):
                rec.update(exploration_status='COMPLETE',
                    access_status=('ACCESSIBLE' if rec['entry_confirmed'] else
                        _old_access(zid, {'outcomes': getattr(book, 'outcomes', {})})),
                    completion_reason='ZONE_LOCAL_PROCEDURE_FINISHED' if rec['entry_confirmed']
                        else 'BOUNDED_ACCESS_REVIEW_FINISHED',phase='TERMINAL',
                    provenance='LIVE_REVIEW_TERMINAL')
            elif rt is not None and (getattr(rt, 'local_second_zone', None)==zid):
                rec['exploration_status']='IN_PROGRESS';rec['phase']='SECOND'

    def snapshot(self,node=None):
        if node is not None:self.sync(node)
        table=dict(schema=SCHEMA,zones=deepcopy(self.zones),
            migration=deepcopy(self.migration),completion_is_independent_of_coverage=True,
            access_findings_do_not_rewrite_coverage=True)
        validate_table(dict(scope=list(self.scope),zone_lifecycle_v187=table))
        return table


def completion_facts(mission: dict) -> dict[str, str]:
    """Finalized Zone status is durable. No automatic reopening on Resume.

    A completed local review is a Zone-level *operational* end, even when the
    stored SECOND-pass residual results contain UNVERIFIED/BUDGET/UNTESTED.
    Those diagnostic records and raw Coverage remain unchanged. Neither a low
    percentage nor a previous planner-failure reason overrides a recorded Zone
    completion. A Zone whose local SECOND phase is still ACTIVE is not closed
    merely because it has old review history from a prior episode.
    """
    # v1.8.7: explicit Zone lifecycle is authoritative when present.
    # A tightly scoped legacy correction excludes only the user-confirmed
    # unentered CORRIDOR_E from the identified 2026-11-10 checkpoint lineage.
    explicit_v187 = mission.get('zone_lifecycle_v187')
    if explicit_v187 is not None:
        return completion_facts_v187(mission)
    scope = mission.get('scope')
    if not isinstance(scope, list) or not scope or len(set(scope)) != len(scope):
        raise ValueError('RESUME_INVALID_SCOPE')
    records = mission.get('records')
    if not isinstance(records, dict) or set(records) != set(scope):
        raise ValueError('RESUME_ZONE_RECORDS_MISSING_OR_EXTRA')
    threshold = mission.get('threshold')
    if (not isinstance(threshold, (float, int)) or isinstance(threshold, bool)
            or not math.isfinite(threshold) or abs(threshold - .96) > 1e-9):
        raise ValueError('RESUME_EXPECTED_96_PERCENT_POLICY')
    conn = mission.get('connectivity') or {}
    if not isinstance(conn, dict):
        raise ValueError('RESUME_INVALID_CONNECTIVITY')
    reviewed = conn.get('local_reviewed', [])
    if not isinstance(reviewed, (list, tuple, set)) or not set(reviewed) <= set(scope):
        raise ValueError('RESUME_COMPLETION_UNKNOWN_ZONE')
    explicit = (mission.get('completion_v183') or {}).get('finished_zones', [])
    if not isinstance(explicit, list) or not set(explicit) <= set(scope):
        raise ValueError('RESUME_COMPLETION_UNKNOWN_ZONE')
    reviews = conn.get('local_reviews') or {}
    if not isinstance(reviews, dict) or not set(reviews) <= set(scope):
        raise ValueError('RESUME_LOCAL_REVIEW_UNKNOWN_ZONE')
    facts = {}
    for zid, rec in records.items():
        value = rec.get('measured', 0.)
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('RESUME_INVALID_MEASURED:' + zid)
        if value + 1e-12 >= threshold:
            facts[zid] = 'COVERAGE_96_REACHED'
            continue
        # An explicit committed Zone completion in this mission's own
        # checkpoint is a durable lifecycle fact, NOT a suggestion to audit.
        if rec.get('state') == 'EXPLORATION_COMPLETE' or zid in explicit:
            facts[zid] = 'SAVED_ZONE_EXPLORATION_COMPLETE'
            continue
        active_here = conn.get('stage') == 'SECOND' and conn.get('local_second_zone') == zid
        review = reviews.get(zid)
        if (not active_here and zid in reviewed and isinstance(review, dict)
                and isinstance(review.get('results'), list)):
            facts[zid] = 'LOCAL_ZONE_REVIEW_FINISHED'
            continue
        if rec.get('state') == 'OBSERVATION_LIMITED':
            outcome = (mission.get('outcomes') or {}).get(zid) or {}
            if (outcome.get('normal_termination') is True and
                    outcome.get('classification') == 'CONNECTED_OBSERVATION_LIMIT' and
                    isinstance(outcome.get('evidence'), dict) and
                    outcome['evidence'].get('connectivity_verified') is True):
                facts[zid] = 'CERTIFIED_LIMITED_OBSERVATION'
    if is_exact_legacy_false_corridor(mission):
        facts.pop('CORRIDOR_E', None)
    return facts


GOOD = frozenset({'OBSERVED', 'SMALL_REMAINDER', 'ROBOT_EXPLORATION_UNAVAILABLE'})
UNAVAILABLE_REASONS = frozenset({
    'ALL_BOUNDED_ROBOT_APPROACHES_FAILED',
    'ALL_SAMPLED_APPROACHES_FAILED',
    'NO_SAFE_VIEWPOINT_IN_COMPLETE_SAMPLED_CATALOG',
})
BAD_REASON_TOKENS = ('BUDGET', 'UNTESTED', 'TIMEOUT', 'STALE', 'INCOMPLETE', 'UNVERIFIED', 'NOT_CHECKED')


def is_terminal_result(item):
    if not isinstance(item, dict):
        return False
    status, reason = item.get('status'), item.get('reason')
    if not isinstance(reason, str) or any(token in reason.upper() for token in BAD_REASON_TOKENS):
        return False
    if status not in GOOD:
        return False
    if status == 'ROBOT_EXPLORATION_UNAVAILABLE' and reason not in UNAVAILABLE_REASONS:
        return False
    return True


def review_verdict(zid, connectivity, task_ledger=None):
    """(complete, cause): positive coverage is handled by caller.

    A saved local review with unfinished/untried candidates is *not* complete.
    Tasks are checked both in the immutable end-of-review summary and in the
    latest saved ledger, to reject stale/contradictory statuses.
    """
    if not isinstance(connectivity, dict):
        return False, 'REVIEW_MISSING'
    reviewed = connectivity.get('local_reviewed', [])
    if not isinstance(reviewed, (list, tuple, set)) or zid not in reviewed:
        return False, 'NO_FINISHED_ZONE_REVIEW'
    if connectivity.get('stage') == 'SECOND' and connectivity.get('local_second_zone') == zid:
        return False, 'LOCAL_SECOND_STILL_ACTIVE'
    review = (connectivity.get('local_reviews') or {}).get(zid)
    if not isinstance(review, dict) or not isinstance(review.get('results'), list):
        return False, 'LOCAL_REVIEW_EVIDENCE_MISSING'
    ledger = (task_ledger or {}).get('tasks', [])
    if not isinstance(ledger, list):
        return False, 'TASK_LEDGER_INVALID'
    own = {}
    for t in ledger:
        if not isinstance(t, dict):
            return False, 'TASK_LEDGER_INVALID'
        if t.get('zone_id') != zid:
            continue
        tid = t.get('task_id')
        if not isinstance(tid, str) or not tid or tid in own:
            return False, 'TASK_LEDGER_INVALID'
        own[tid] = t
    for t in own.values():
        if t.get('status') == 'SPLIT':
            children = t.get('children')
            if not isinstance(children, list) or not children or any(
                    not isinstance(cid, str) or cid not in own or not is_terminal_result(own[cid])
                    for cid in children):
                return False, 'SPLIT_CHILDREN_NOT_FINISHED'
        elif not is_terminal_result(t):
            return False, 'TASK_STILL_UNRESOLVED:' + str(t.get('reason', t.get('status')))
    for t in review['results']:
        if not isinstance(t, dict):
            return False, 'LOCAL_REVIEW_INVALID'
        if t.get('status') == 'SPLIT':
            parent = own.get(t.get('task_id'))
            if not parent or not parent.get('children') or any(
                    cid not in own or not is_terminal_result(own[cid])
                    for cid in parent['children']):
                return False, 'LOCAL_REVIEW_SPLIT_UNRESOLVED'
        elif not is_terminal_result(t):
            return False, 'LOCAL_REVIEW_UNRESOLVED:' + str(t.get('reason', t.get('status')))
    return True, 'LOCAL_PROCEDURE_TERMINAL_EVIDENCE'


def live_review_verdict(runtime, zid):
    ledger = getattr(runtime, 'tasks', None)
    tasks = (getattr(ledger, 'tasks', {}) or {}).values() if ledger is not None else ()
    conn = dict(local_reviewed=set(getattr(runtime, 'local_reviewed', ())),
                local_reviews=getattr(runtime, 'local_reviews', {}),
                stage=getattr(runtime, 'exploration_stage', 'PRIMARY'),
                local_second_zone=getattr(runtime, 'local_second_zone', None))
    return review_verdict(zid, conn, dict(tasks=list(tasks)))


def unresolved_zones(mission):
    """Re-open *only* previously false-positive terminal zones, no historical data loss."""
    facts = completion_facts(mission)
    conn = mission.get('connectivity') or {}
    old = set((mission.get('completion_v183') or {}).get('finished_zones', []))
    old.update(z for z,r in (mission.get('records') or {}).items()
               if r.get('state') == 'EXPLORATION_COMPLETE')
    old.update(conn.get('local_reviewed', []))
    return sorted((old - set(facts)) & set(mission.get('scope', [])))
