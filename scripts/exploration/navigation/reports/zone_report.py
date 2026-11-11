#!/usr/bin/env python3
"""DN-012 v3 read-only zone assessment: produce JSON/CSV/Markdown from a *finished* session.

No ROS import, no Nav2 calls, no motion, and no modification of DN-012 outputs.
Strict evidence semantics: a path blockage is not a local-topology cut, a Nav2
path is not a traversed route, and low coverage is not a completed search.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import sys
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

ASSESSMENT_SCHEMA = 'dn012_zone_assessment_report_v1'
FULL_STATES = {'ALL_REQUESTED_ZONES_OBSERVED', 'ALL_ZONES_ASSESSED_WITH_LIMITS'}
STOP_STATES = {'GOAL_LIMIT_REACHED', 'PLAN_ATTEMPT_LIMIT_PARTIAL', 'PARTIAL_DEFERRED_REMAIN'}
CSV_FIELDS = [
    'zone_id', 'measured_coverage_percent', 'coverage_target_percent', 'zone_state',
    'exploration_result', 'assessment_classification', 'reason', 'normal_termination',
    'entry_confirmed', 'local_connectivity', 'global_access',
    'blocked_path_events', 'blocked_path_stop_confirmed', 'blocked_points_xy',
    'nav_succeeded_events', 'nav_aborted_events', 'nav_canceled_events',
    'nav2_alternate_path_plans', 'connectivity_probe_paths', 'audit_started',
    'audit_probes_failed', 'audit_requested', 'evidence_limits'
]

class ReportError(ValueError):
    pass


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError('Cannot read valid JSON: {}: {}'.format(path, exc)) from exc


def read_events(path: Path):
    if not path.is_file():
        return [], ['EVENTS_JSONL_MISSING: per-zone route evidence cannot be confirmed']
    result = []
    try:
        with path.open(encoding='utf-8') as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get('event'), str):
                    raise ReportError('Bad event record at {}:{}'.format(path, line_no))
                result.append(event)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError('Malformed event stream, refusing partial evidence: {}'.format(exc)) from exc
    return result, []


def safe_float(value, name):
    if type(value) not in (float, int):
        raise ReportError('Non-numeric {}'.format(name))
    v = float(value)
    if not math.isfinite(v):
        raise ReportError('Non-finite {}'.format(name))
    return v


def safe_name(text):
    return str(text).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ').replace('`', "'")


def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _xy(e):
    witness = e.get('evidence') if isinstance(e.get('evidence'), dict) else {}
    xy = witness.get('xy')
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        return None
    try:
        x, y = (safe_float(t, 'blocked xy') for t in xy)
        return [round(x, 3), round(y, 3)]
    except ReportError:
        return None


def classify_zone(zid, item, events, requested_goal_result):
    if not isinstance(item, dict):
        raise ReportError('Invalid zone summary: {}'.format(zid))
    measured = safe_float(item.get('measured_percent'), 'measured_percent of ' + zid)
    if not 0 <= measured <= 100:
        raise ReportError('Out-of-range Coverage for ' + zid)
    state = item.get('state', 'UNKNOWN')
    assessment = item.get('assessment')
    if not isinstance(assessment, dict):
        assessment = {}
    classification = str(assessment.get('classification') or 'NOT_CLASSIFIED')
    reason = str(assessment.get('reason') or item.get('deferred_reason') or '')
    normal_limited = (state == 'OBSERVATION_LIMITED'
                      and classification == 'CONNECTED_OBSERVATION_LIMIT'
                      and assessment.get('normal_termination') is True)
    if state == 'OBSERVATION_SUFFICIENT':
        if measured < 96.0-1e-6:
            raise ReportError('COVERAGE_STATE_CONTRADICTION: {} measured below 96%'.format(zid))
        exploration = 'COVERAGE_96_CONFIRMED'
        normal_termination = True
    elif normal_limited:
        exploration = 'OBSERVATION_LIMITED_CERTIFIED'
        normal_termination = True
    elif state == 'THRESHOLD_PENDING':
        exploration = 'THRESHOLD_CONFIRMATION_PENDING'
        normal_termination = False
    elif state == 'PARTIAL_DEFERRED':
        exploration = 'PARTIAL_DEFERRED_UNRESOLVED'
        normal_termination = False
    elif measured > 0:
        exploration = 'PARTIAL_UNRESOLVED'
        normal_termination = False
    else:
        exploration = 'NOT_OBSERVED_OR_UNRESOLVED'
        normal_termination = False

    ze = [e for e in events if e.get('zone') == zid or
          (isinstance(e.get('candidate'), dict) and e['candidate'].get('zone_id') == zid) or
          (isinstance(e.get('evidence'), dict) and e['evidence'].get('zone') == zid)]
    blocked = [e for e in ze if e.get('event') == 'PATH_BLOCKED_EVIDENCE']
    # The runtime writes a completed, owned cancellation record only after
    # terminal CANCELED, fresh odometry stop and new Coverage. Never count a
    # bare event label if its payload is missing these confirmations.
    confirmed = [e for e in ze if e.get('event') == 'PATH_BLOCKAGE_STOP_CONFIRMED'
                 and isinstance(e.get('record'), dict)
                 and e['record'].get('stop_confirmed') is True
                 and e['record'].get('new_coverage') is True
                 and e['record'].get('terminal') == 'CANCELED'
                 and isinstance(e['record'].get('goal_id'), str)]
    points = [p for p in (_xy(e) for e in blocked) if p is not None]
    unique_points = list(dict.fromkeys((tuple(p) for p in points)))
    entry_confirmed = bool(item.get('entered') is True or
                           any(e.get('event') == 'ENTRY_CONFIRMED' for e in ze))
    alt_paths = sum(e.get('event') in ('ENTRY_ALTERNATE_PATH_VERIFIED', 'PATH_BLOCKAGE_REPLAN_OK') for e in ze)
    probes = sum(e.get('event') == 'CONNECTIVITY_PROBE_OK' for e in ze)
    failed_probes = sum(e.get('event') == 'CONNECTIVITY_PROBE_FAILED' for e in ze)
    audit = any(e.get('event') == 'ZONE_AUDIT_STARTED' for e in ze)

    # True topological cut only with explicit affirmative evidence from the mission.
    # A single blocked path, even after terminal stop, is not this certificate.
    a_evidence = assessment.get('evidence') if isinstance(assessment.get('evidence'), dict) else {}
    if a_evidence.get('local_connectivity_disconnected') is True and a_evidence.get('local_connectivity_verified') is True:
        local = 'LOCAL_DISCONNECTED_EVIDENCE_CERTIFIED'
    elif a_evidence.get('local_connectivity_connected') is True and a_evidence.get('local_connectivity_verified') is True:
        local = 'LOCAL_CONNECTED_EVIDENCE_CERTIFIED'
    elif confirmed:
        local = 'BLOCKED_PATH_CONFIRMED_LOCAL_TOPOLOGY_UNVERIFIED'
    elif blocked:
        local = 'BLOCKAGE_CANDIDATE_LOCAL_TOPOLOGY_UNVERIFIED'
    else:
        local = 'LOCAL_TOPOLOGY_UNVERIFIED'

    if classification == 'CURRENT_MAP_UNREACHABLE' and a_evidence.get('current_map_unreachable') is True:
        global_access = 'CURRENT_MAP_DISCONNECTED_CONDITIONAL'
    elif entry_confirmed:
        global_access = 'ZONE_POSE_REACHED_NOT_ALL_COMPONENTS'
    elif alt_paths or probes:
        global_access = 'PATH_PLANNED_ONLY_NOT_TRAVERSED'
    else:
        global_access = 'NOT_VERIFIED'

    nresult = requested_goal_result.get(zid, Counter())
    limitations = ['2D Coverage != victim discovery']
    if state not in ('OBSERVATION_SUFFICIENT', 'OBSERVATION_LIMITED'):
        limitations.append('no normal completion certificate')
    if blocked or confirmed:
        limitations.append('path obstruction != whole-zone cut')
    if entry_confirmed:
        limitations.append('one entry != all components accessible')
    if alt_paths or probes:
        limitations.append('planned route != traversed route')
    if not events:
        limitations.append('event evidence missing')
    return dict(
        zone_id=zid,
        measured_coverage_percent=round(measured, 4),
        zone_state=str(state), exploration_result=exploration,
        assessment_classification=classification, reason=reason,
        normal_termination=normal_termination, entry_confirmed=entry_confirmed,
        local_connectivity=local, global_access=global_access,
        blocked_path_events=len(blocked), blocked_path_stop_confirmed=len(confirmed),
        blocked_points_xy=[list(p) for p in unique_points],
        nav_succeeded_events=nresult.get('SUCCEEDED', 0),
        nav_aborted_events=nresult.get('ABORTED', 0),
        nav_canceled_events=nresult.get('CANCELED', 0),
        nav2_alternate_path_plans=alt_paths, connectivity_probe_paths=probes,
        audit_started=audit, audit_probes_failed=failed_probes,
        audit_requested=item.get('audit_requested'),
        evidence_limits=limitations,
    )


def assemble(session: Path, project: Path | None = None):
    session = session.resolve()
    manifest_path, summary_path = session / 'run_manifest.json', session / 'summary.json'
    if not summary_path.is_file():
        raise ReportError('FINISHED_SUMMARY_REQUIRED: no summary.json in ' + str(session))
    if not manifest_path.is_file():
        raise ReportError('RUN_MANIFEST_REQUIRED: no run_manifest.json in ' + str(session))
    summary, manifest = read_json(summary_path), read_json(manifest_path)
    if not isinstance(summary, dict) or not isinstance(manifest, dict):
        raise ReportError('Invalid mission summary or manifest')
    zones = summary.get('zones')
    if not isinstance(zones, dict) or not zones:
        raise ReportError('No zone summary')
    coverage_target = safe_float(summary.get('threshold'), 'mission threshold') * 100
    if coverage_target != 96.0:
        raise ReportError('Expected DN-012 96% mission; found threshold {}'.format(coverage_target))
    if summary.get('coverage_session') in (None, ''):
        raise ReportError('No measured Coverage session ID')
    events, warnings = read_events(session/'events.jsonl')
    event_summary = next((e for e in reversed(events) if e.get('event') == 'SUMMARY'), None)
    if event_summary and event_summary.get('state') != summary.get('state'):
        raise ReportError('SUMMARY_EVENT_MISMATCH')
    # Goal UUID is the authoritative matching key; never count results by order.
    goal_to_zone = {}
    for e in events:
        if e.get('event') == 'NAV_REQUESTED' and isinstance(e.get('candidate'), dict) and e.get('goal_id'):
            goal_to_zone[e['goal_id']] = e['candidate'].get('zone_id')
    counts = defaultdict(Counter)
    for e in events:
        if e.get('event') == 'NAV_RESULT' and e.get('goal_id') in goal_to_zone:
            status = str(e.get('status') or '')
            counts[goal_to_zone[e['goal_id']]][status] += 1
    requested = {k: classify_zone(k, r, events, counts) for k, r in sorted(zones.items())}
    # Scope is precisely the summary scope; other hospital zones are never
    # silently called complete. The manifest's --zone-ids carries subset intent.
    options = manifest.get('options') if isinstance(manifest.get('options'), dict) else {}
    selected = options.get('zone_ids')
    approved_zone_path = options.get('zones')
    approved_ids = None
    if isinstance(approved_zone_path, str):
        path = Path(approved_zone_path).expanduser()
        if path.is_file():
            # A post-run report must not silently use a *different* zones file.
            source_hashes = manifest.get('input_sha256')
            expected_hash = (source_hashes.get(str(path))
                             if isinstance(source_hashes, dict) else None)
            if expected_hash is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                warnings.append('ZONE_CONFIG_CHANGED_SINCE_MISSION: full scope not independently verified')
            else:
                source = read_json(path)
                records = source.get('zones') if isinstance(source, dict) else None
                if isinstance(records, list):
                    approved_ids = {z['id'] for z in records if isinstance(z, dict) and 'id' in z}
        else:
            warnings.append('APPROVED_ZONE_CONFIG_NOT_AVAILABLE: full scope unconfirmed')
    if approved_ids is None:
        warnings.append('APPROVED_ZONE_ID_SET_UNVERIFIED: cannot certify exact 27 scope')
    full_scope = (selected in (None, [], 'None') and len(zones) == 27
                  and approved_ids is not None and len(approved_ids) == 27
                  and set(zones) == approved_ids)
    if not full_scope:
        warnings.append('NOT_VERIFIED_FULL_27_SCOPE: only recorded zones are reported')
    if selected not in (None, [], 'None'):
        warnings.append('ZONE_ID_FILTER: full 27-zone coverage was not attempted')
    if not summary.get('execute'):
        warnings.append('DRY_RUN: routes were planned but no exploration motion requested')
    if summary.get('state') not in FULL_STATES:
        warnings.append('NONTERMINAL_ASSESSMENT: mission stopped without resolving every zone')
    if event_summary is None:
        warnings.append('NO_SUMMARY_EVENT_CROSSCHECK')
    counter = Counter(v['exploration_result'] for v in requested.values())
    report = dict(
        schema=ASSESSMENT_SCHEMA, generated_utc=datetime.now(timezone.utc).isoformat(),
        session=str(session), mission_state=summary.get('state'),
        mission_execute=summary.get('execute'),
        coverage_session=summary.get('coverage_session'),
        coverage_target_percent=round(coverage_target, 4),
        zone_bundle_signature=summary.get('zone_bundle_signature'),
        zones_in_scope=len(zones), full_27_scope=full_scope,
        normal_termination=bool(summary.get('state') in FULL_STATES and
                                all(r['normal_termination'] for r in requested.values())),
        mission_goals_sent=summary.get('sent'), mission_nav_succeeded=summary.get('nav_succeeded'),
        mission_nav_aborted=summary.get('nav_aborted'),
        mission_path_blockage_canceled=summary.get('path_blockage_canceled'),
        outcome_counts=dict(sorted(counter.items())),
        warnings=warnings, zones=requested,
        limits=[
            'Coverage is measured 2D visibility, not all victims detected.',
            'Dynamic obstacle witness supports a blocked *path*, not automatically a disconnected zone.',
            'Local connectivity requires explicit verified topology evidence.',
            'One successful zone entry never certifies reachability of all separated components.',
            'Nav2 path planning alone does not prove that a robot traversed an alternative route.',
            'Unobserved or uncertain spaces remain unresolved and must be reported.'
        ],
        inputs={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (manifest_path,summary_path,session/'events.jsonl') if p.is_file()}
    )
    return report


def to_markdown(report):
    lines = [
        '# DN-012 구역별 재난탐사·연결성 결과', '',
        '- 실행 세션: `{}`'.format(safe_name(report['session'])),
        '- 실제 주행: **{}**'.format('YES' if report['mission_execute'] else 'NO (Dry Run)'),
        '- 종료 상태: `{}`'.format(safe_name(report['mission_state'])),
        '- 대상: **{}개 Zone** (27개 전체 대상: {})'.format(report['zones_in_scope'], report['full_27_scope']),
        '- Coverage 기준: **{}%**, 세션: `{}`'.format(report['coverage_target_percent'], safe_name(report['coverage_session'])),
        '- 정상 완료/관측 한계 포함 전 구역 판정: **{}**'.format(report['normal_termination']),
        '- Goal: 전송 `{}`, 성공 `{}`, Nav2 Abort `{}`, 차단 근거 기반 취소 `{}`'.format(
            report['mission_goals_sent'], report['mission_nav_succeeded'],
            report['mission_nav_aborted'], report['mission_path_blockage_canceled']),
        '', '## Zone별 결과', '',
        '| Zone | 실측 Coverage | 탐사 판정 | 국소 연결성 | 전역 접근성 | 차단 경로 확인 |',
        '|---|---:|---|---|---|---:|',
    ]
    for z in report['zones'].values():
        lines.append('| `{}` | {:.2f}% | `{}` | `{}` | `{}` | {} |'.format(
            safe_name(z['zone_id']), z['measured_coverage_percent'],
            safe_name(z['exploration_result']), safe_name(z['local_connectivity']),
            safe_name(z['global_access']), z['blocked_path_stop_confirmed']))
    lines.extend(['', '## 구역별 근거', ''])
    for z in report['zones'].values():
        if (z['blocked_path_events'] or z['assessment_classification'] != 'NOT_CLASSIFIED'
                or z['audit_started'] or z['reason']):
            lines += [
                '### {}'.format(safe_name(z['zone_id'])),
                '- 판정: `{}` / `{}`'.format(safe_name(z['zone_state']),safe_name(z['assessment_classification'])),
                '- 이유: `{}`'.format(safe_name(z['reason'] or '근거 추가 필요')),
                '- 경로 차단 관측: {}회 / 정지 후 기록: {}회'.format(z['blocked_path_events'],z['blocked_path_stop_confirmed']),
                '- 장애물 관측점(차단선 아님): `{}`'.format(z['blocked_points_xy']),
                '- Nav2 대체 경로 계획 증거: {}회 / 연결성 표본 경로 계획: {}회'.format(z['nav2_alternate_path_plans'],z['connectivity_probe_paths']),
                '- 감사 시작: {} / 실패한 표본 경로 계획: {}'.format(z['audit_started'],z['audit_probes_failed']),
                '- 근거 한계: {}'.format('; '.join(z['evidence_limits'])), ''
            ]
    lines += ['## 주의 및 제한', '']
    for w in report['warnings'] + report['limits']:
        lines.append('- {}'.format(safe_name(w)))
    lines.append('')
    return '\n'.join(lines)


def to_csv(report):
    out = io.StringIO(newline='')
    writer=csv.DictWriter(out,fieldnames=CSV_FIELDS,extrasaction='ignore')
    writer.writeheader()
    for row in report['zones'].values():
        entry=dict(row,coverage_target_percent=report['coverage_target_percent'],
                   blocked_points_xy=json.dumps(row['blocked_points_xy']),
                   evidence_limits='; '.join(row['evidence_limits']))
        writer.writerow(entry)
    return out.getvalue()


def latest_session(project: Path):
    parent = project/'logs/hospital_zone_mission'
    choices=sorted([p for p in parent.glob('session_*') if p.is_dir()], key=lambda p:p.name)
    if not choices:
        raise ReportError('NO_SESSIONS_FOUND: '+str(parent))
    return choices[-1]


def assessment_main(argv=None):
    p=argparse.ArgumentParser(description='DN-012 v3 evidence-limited post-run zone report; read-only for mission outputs.')
    p.add_argument('--project',type=Path,default=Path.home()/('Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping'))
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--latest',action='store_true',help='Most recent session; fail closed if it is not finished')
    group.add_argument('--session',type=Path,help='Exact existing hospital_zone_mission/session_... directory')
    p.add_argument('--output',type=Path,help='Optional report destination; default <session>/zone_assessment')
    a=p.parse_args(argv)
    try:
        session=latest_session(a.project.expanduser()) if a.latest else a.session.expanduser()
        report=assemble(session,project=a.project.expanduser())
        dest=(a.output.expanduser() if a.output else session/'zone_assessment').resolve()
        atomic_write(dest/'zone_results.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        atomic_write(dest/'zone_results.csv',to_csv(report))
        atomic_write(dest/'zone_results.md',to_markdown(report))
        print('[ZONE_REPORT_SAVED]',dest)
        print('[ZONE_REPORT_SUMMARY] {} zones, state={}, normal_termination={}, measured coverage retained'.format(
            report['zones_in_scope'],report['mission_state'],report['normal_termination']))
        for w in report['warnings']:
            print('[ZONE_REPORT_WARNING]',w)
    except (ReportError, OSError) as exc:
        p.exit(2,'[ZONE_REPORT_ERROR] {}\n'.format(exc))


CONSISTENCY_SCHEMA = 'dn012_coverage_consistency_report_v1'


def number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('INVALID_NUMBER: ' + name)
    return float(value)


def read_coverage_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('JSON_OBJECT_REQUIRED: ' + str(path))
    return value


def evidence_valid(row, session, target):
    """Positive evidence only; this is not a replacement for live freshness gates."""
    e = row.get('coverage_confirmation_evidence')
    if not isinstance(e, dict) or not session:
        return False
    try:
        first = number(e.get('first_above_threshold_scan_end_s'), 'first scan')
        last = number(e.get('last_evidence_scan_end_s'), 'last scan')
        seq0, seq1 = e.get('first_above_threshold_scan_sequence'), e.get('evidence_scans')
        return bool(e.get('source') == 'EXISTING_THRESHOLD_TRACKER'
            and e.get('coverage_session') == session
            and e.get('confirms_accessibility') is False
            and type(seq0) is int and type(seq1) is int and 0 <= seq0 < seq1
            and last - first >= 1.0
            and number(e.get('threshold_percent'), 'threshold') == target
            and target <= number(e.get('measured_percent'), 'coverage') <= 100.)
    except (ValueError, TypeError):
        return False


def enrich(report, summary=None):
    if report.get('schema') != 'dn012_zone_assessment_report_v1':
        raise ValueError('EXPECTED_V1_ZONE_REPORT')
    target = number(report.get('coverage_target_percent'), 'target')
    if not 0. < target <= 100.:
        raise ValueError('INVALID_TARGET')
    zones = report.get('zones')
    if not isinstance(zones, dict) or not zones:
        raise ValueError('NO_ZONES')
    source_rows = (summary or {}).get('zones', {})
    out = deepcopy(report)
    out['schema'] = CONSISTENCY_SCHEMA
    out['source_report_schema'] = report['schema']
    out['review_generated_utc'] = datetime.now(timezone.utc).isoformat()
    out['historical_outcomes_rewritten'] = False
    out['return_home_evidence_included'] = False
    out['source_mode'] = 'FINISHED_SESSION' if summary is not None else 'EXPORTED_REPORT_ONLY'
    counts = Counter()
    discrepancies = []
    for zid, z in out['zones'].items():
        measured = number(z.get('measured_coverage_percent'), zid)
        if not 0. <= measured <= 100.:
            raise ValueError('OUT_OF_RANGE_COVERAGE: ' + zid)
        raw = source_rows.get(zid, {})
        if not isinstance(raw, dict):
            raise ValueError('INVALID_SOURCE_ROW: ' + zid)
        # Prefer unrounded runtime measurement when available, never round up at 96%.
        if raw:
            precise = number(raw.get('measured_percent'), zid + ' precise')
            if not 0. <= precise <= 100. or abs(precise - measured) > 0.00011:
                raise ValueError('SUMMARY_REPORT_MEASUREMENT_MISMATCH: ' + zid)
            measured = precise
        reached = measured >= target
        state = z.get('zone_state')
        confirmed = state == 'OBSERVATION_SUFFICIENT'
        if confirmed and not reached:
            raise ValueError('STATE_BELOW_THRESHOLD: ' + zid)
        if raw and 'coverage_confirmed' in raw:
            if raw['coverage_confirmed'] is not confirmed:
                raise ValueError('CONFIRMATION_STATE_MISMATCH: ' + zid)
            if raw.get('coverage_target_reached') is not reached:
                raise ValueError('TARGET_REACHED_MISMATCH: ' + zid)
            if confirmed and not evidence_valid(raw, out.get('coverage_session'), target):
                raise ValueError('CONFIRMATION_EVIDENCE_INVALID: ' + zid)
        coverage_status = ('TARGET_CONFIRMED' if confirmed else
                           'TARGET_REACHED_UNCONFIRMED' if reached else 'BELOW_TARGET')
        history = raw.get('deferral_history')
        if history is not None and not isinstance(history, list):
            raise ValueError('INVALID_DEFERRAL_HISTORY: ' + zid)
        # A historical assessment remains historical even if passive Coverage confirms later.
        historical_reason = (raw.get('deferred_reason') or z.get('reason') or '')
        if history:
            historical_reason = history[-1].get('reason') or historical_reason
        deferred = bool(history or historical_reason or state == 'PARTIAL_DEFERRED'
                        or raw.get('deferred_locked_this_mission') is True)
        exploration_status = ('DEFERRED' if deferred else
            'COVERAGE_SUFFICIENT' if confirmed else state)
        z.update(coverage_target_percent=target, coverage_target_reached=reached,
                 coverage_confirmed=confirmed, coverage_status=coverage_status,
                 exploration_status=exploration_status,
                 recorded_runtime_state=state,
                 deferral_reason_history=historical_reason,
                 coverage_at_last_deferral_percent=raw.get('coverage_at_last_deferral_percent'),
                 deferral_history=deepcopy(history),
                 coverage_confirmation_evidence=deepcopy(raw.get('coverage_confirmation_evidence')),
                 confirmation_source=('RUNTIME_THRESHOLD_TRACKER_EVIDENCE' if confirmed and raw and
                    'coverage_confirmed' in raw else 'RECORDED_STATE_ONLY' if confirmed else 'NOT_PROVEN'),
                 coverage_does_not_certify_connectivity=True)
        counts[coverage_status] += 1
        if reached:
            counts['numeric_target_reached'] += 1
        if deferred:
            counts['deferral_history_present'] += 1
        if reached and not confirmed:
            discrepancies.append(zid)
    out['coverage_counts'] = dict(counts)
    out['coverage_confirmation_gap_zones'] = discrepancies
    out['warnings'] = list(out.get('warnings', []))
    if discrepancies:
        out['warnings'].append('NUMERIC_THRESHOLD_WITHOUT_CONFIRMED_STATE: ' + ', '.join(discrepancies))
    out['limits'] = list(out.get('limits', [])) + [
        'Numeric threshold crossing is not retrospective proof of two fresh scans.',
        'No automatic revisit does not prohibit passive measured Coverage updates.',
        'Recorded mission/zone outcome and all blockage evidence are retained unchanged.',
        'This report cannot establish physical return success or local/global connectivity.',
    ]
    return out


def esc(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('`', "'")


def markdown(report):
    c = report['coverage_counts']
    lines = ['# DN-012 P0 관측 목표·보류 이력 분리 보고서', '',
        '- 원본 세션: `' + esc(report.get('session')) + '`',
        '- 원본 임무 종료 상태: `' + esc(report.get('mission_state')) + '` (변경하지 않음)',
        '- 실측 목표 이상: **{} / {}**'.format(c.get('numeric_target_reached', 0), len(report['zones'])),
        '- 원본 또는 신규 런타임에서 완료 확인: **{} / {}**'.format(c.get('TARGET_CONFIRMED', 0), len(report['zones'])),
        '- 목표 이상이나 완료 확인 부족: **{}**'.format(c.get('TARGET_REACHED_UNCONFIRMED', 0)),
        '', '**과거 결과를 완료로 덮어쓰지 않는다. 관측률 달성은 차단 해제·전역 접근·피해자 전원 발견의 증거가 아니다.**', '',
        '| Zone | 최종 Coverage | 관측 목표 | 원본 상태 | 탐사 운영 상태 | 보류 이력 사유 |',
        '|---|---:|---|---|---|---|']
    labels = {'TARGET_CONFIRMED':'목표 달성 확인', 'TARGET_REACHED_UNCONFIRMED':'수치 달성·확인 부족', 'BELOW_TARGET':'목표 미달'}
    for zid, z in sorted(report['zones'].items()):
        lines.append('| {} | {:.4f}% | {} | {} | {} | {} |'.format(
            esc(zid), z['measured_coverage_percent'], labels[z['coverage_status']],
            esc(z['recorded_runtime_state']), esc(z['exploration_status']), esc(z['deferral_reason_history'])))
    lines += ['', '## 접근성·차단 기록 (원본 유지)', '',
              '| Zone | 국소 연결성 | 전역 접근성 | 차단 관측점 |', '|---|---|---|---|']
    for zid,z in sorted(report['zones'].items()):
        lines.append('| {} | {} | {} | {} |'.format(esc(zid), esc(z.get('local_connectivity')),
                     esc(z.get('global_access')), esc(z.get('blocked_points_xy', []))))
    lines += ['', '## 주의', '']
    lines += ['- ' + esc(s) for s in report['warnings'] + report['limits']]
    return '\n'.join(lines) + '\n'


FIELDS = ['zone_id','measured_coverage_percent','coverage_target_percent','coverage_target_reached',
          'coverage_confirmed','coverage_status','recorded_runtime_state','exploration_status',
          'deferral_reason_history','coverage_at_last_deferral_percent','local_connectivity',
          'global_access','blocked_path_events','blocked_path_stop_confirmed','blocked_points_xy',
          'confirmation_source']


def csv_text(report):
    s=io.StringIO(newline='');w=csv.DictWriter(s, fieldnames=FIELDS, extrasaction='ignore');w.writeheader()
    for z in report['zones'].values():
        row=dict(z);row['blocked_points_xy']=json.dumps(row.get('blocked_points_xy', []))
        w.writerow(row)
    return s.getvalue()




def coverage_main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--session',type=Path,help='Exact finished session directory; do not guess the latest run')
    g.add_argument('--legacy-report',type=Path,help='Read an existing zone_results.json without changing its verdicts')
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    try:
        if a.session:
            session=a.session.expanduser().resolve()
            source=assemble(session)
            result=enrich(source, read_coverage_json(session/'summary.json'))
            dest=(a.output or session/'zone_assessment_p0').expanduser().resolve()
            inputs=[session/'summary.json',session/'run_manifest.json',session/'events.jsonl']
        else:
            file=a.legacy_report.expanduser().resolve()
            result=enrich(read_coverage_json(file))
            dest=(a.output or file.parent/'zone_assessment_p0').expanduser().resolve()
            inputs=[file]
        names=['zone_results_p0.json','zone_results_p0.csv','zone_results_p0.md']
        if any((dest/name).resolve() == q.resolve() for name in names for q in inputs):
            raise ValueError('OUTPUT_MUST_NOT_OVERWRITE_INPUT')
        result['p0_input_sha256']={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in inputs if q.is_file()}
        text=[json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',csv_text(result),markdown(result)]
        for name,value in zip(names,text):atomic_write(dest/name,value)
        print('[P0_REPORT_SAVED]',dest)
        print('[P0_COUNTS]',json.dumps(result['coverage_counts'],ensure_ascii=False))
        print('[P0_NOTE] Original runtime state, return policy and input logs unchanged.')
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print('[P0_REPORT_ERROR]',exc)
        return 2

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description='Saved zone assessment and coverage reports.')
    parser.add_argument('mode', choices=('assessment', 'coverage'))
    if not argv or argv[0] in ('-h', '--help'):
        parser.parse_args(argv)
    mode = parser.parse_args(argv[:1]).mode
    return (assessment_main if mode == 'assessment' else coverage_main)(argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
