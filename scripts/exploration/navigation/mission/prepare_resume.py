#!/usr/bin/env python3
"""Import existing DN012 logs into ONE validated v1.8 checkpoint; no ROS motion.

The exact Coverage session is required. Other sessions/older fallback directories
are never substituted. Legacy world identity requires explicit operator consent.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'scripts/exploration'), str(ROOT / 'scripts/exploration/zone_exploration/src')]
from hospital_coverage_core import load_bundle
from checkpoint import (SCHEMA, import_legacy_coverage, project_context,
                                  read_json, save_checkpoint, plain, sha)
from exploration_results import BlockageRegistry


def merge_reports(reports, bundle):
    if not reports:
        raise ValueError('NO_MATCHING_MISSION_REPORTS; refusing to invent mission history')
    m = dict(scope=[], threshold=.96, counters=dict(sent_goals=0, plan_attempts=0,
             succeeded_goals=0, aborted_goals=0, coverage_cancels=0), records={}, visit_states={},
             visits=[], rejected=[], progress=0, deferral_history={}, outcomes={}, nav_abort_counts={},
             connectivity=dict(local_reviewed=[], total_probes=0),
             task_ledger=dict(tasks=[], inventoried_zones=[], small_by_zone={}, order=0, initialized=False),
             original_summary=dict(path_blockage_cancellations=[], reroute_memory_v1_4=dict(points=[])))
    registry = BlockageRegistry(bundle)
    task_zones = {}
    guard = None
    point_records = {}
    for session, status, manifest in reports:
        if status.get('resume_v18'):
            raise ValueError('V18_SESSION_PRESENT: resume its checkpoint instead of adding cumulative counters twice')
        if status.get('zone_bundle_signature') != bundle[4]:
            raise ValueError('LEGACY_MISSION_BUNDLE_MISMATCH:' + session.name)
        threshold = float(status.get('threshold', .96))
        if abs(threshold - .96) > 1e-9:
            raise ValueError('LEGACY_MISSION_THRESHOLD_NOT_96:' + session.name)
        for target, source in [('sent_goals','sent'),('plan_attempts','plan_attempts'),
                               ('succeeded_goals','nav_succeeded'),('aborted_goals','nav_aborted'),
                               ('coverage_cancels','coverage_canceled')]:
            value = status.get(source, 0)
            if type(value) is not int or value < 0:
                raise ValueError('INVALID_LEGACY_COUNTER:' + source)
            m['counters'][target] += value
        for zid, rec in status.get('zones', {}).items():
            if zid not in m['scope']:
                m['scope'].append(zid)
            previous = m['records'].get(zid, {})
            m['records'][zid] = dict(state=rec['state'], measured=rec['measured_percent']/100.,
                views=previous.get('views', 0)+int(rec.get('views', 0)),
                low_gain_count=int(rec.get('low_gain_count', 0)), deferred_reason=rec.get('deferred_reason'))
            v = m['visit_states'].setdefault(zid, dict(entered=False, entry_goals=0,
                observation_goals=0, low_gain=0, retries=0, entry_arrival_retries=0))
            v['entered'] = v['entered'] or bool(rec.get('entered'))
            for key, source in [('entry_goals','entry_goals'),('observation_goals','observation_goals'),
                                ('retries','revisit_count'),('entry_arrival_retries','entry_arrival_retries')]:
                v[key] += int(rec.get(source, 0))
            v['low_gain'] = int(rec.get('low_gain_count', 0))
            m['deferral_history'].setdefault(zid, []).extend(copy.deepcopy(rec.get('deferral_history', [])))
            m['outcomes'][zid] = copy.deepcopy(rec.get('assessment') or {})
            m['nav_abort_counts'][zid] = m['nav_abort_counts'].get(zid,0)+int(rec.get('nav_aborted',0))
        for visit in status.get('visits', []):
            m['visits'].append(dict(visit, source_mission_session=session.name))
        registry.absorb_summary(status)
        m['original_summary']['path_blockage_cancellations'].extend(copy.deepcopy(status.get('path_blockage_cancellations', [])))
        for rec in (status.get('reroute_memory_v1_4') or {}).get('points', []):
            point_records[rec['id']] = copy.deepcopy(rec)
        region = status.get('region_review') or {}
        for zid in status.get('zones', {}):
            ts = [copy.deepcopy(t) for t in region.get('tasks', []) if t.get('zone_id') == zid]
            if ts:task_zones[zid] = ts
        m['connectivity']['local_reviewed'] = sorted(set(m['connectivity']['local_reviewed']) |
                                                    set(region.get('locally_reviewed_zones', [])))
        m['connectivity']['total_probes'] += int(status.get('connectivity_probes', 0))
        g = status.get('mission_budget')
        if g:
            if guard is None:
                guard = dict(settings=copy.deepcopy(g['settings']), records=[], zone_cancels={}, ledgers={})
            if g['settings'] != guard['settings']:
                raise ValueError('LEGACY_MISSION_GUARD_SETTINGS_DIFFER')
            for zid, rec in g.get('zone_work', {}).items():
                dest = guard['ledgers'].setdefault(zid, dict(rec, work_time_ros_s=0.))
                dest['work_time_ros_s'] += float(rec['work_time_ros_s'])
                dest['allowance_ros_s'] = max(dest['allowance_ros_s'],rec['allowance_ros_s'])
                dest['extended'] = dest['extended'] or rec['extended']
                dest['expired'] = dest['work_time_ros_s'] >= dest['allowance_ros_s']
            for rec in g.get('cancellations', []):
                guard['records'].append(copy.deepcopy(rec))
                z = rec['zone'];guard['zone_cancels'][z] = guard['zone_cancels'].get(z,0)+1
        m['connectivity']['local_reviews'] = dict(m['connectivity'].get('local_reviews', {}), **copy.deepcopy(region.get('local_reviews', {})))
        m['source_state'] = status.get('state')
        m['interrupted_goal'] = copy.deepcopy(status.get('current'))
        m['interrupted_goal_id'] = status.get('owned_goal_id')
    m['progress'] = len(m['visits'])
    m['original_summary']['reroute_memory_v1_4']['points'] = list(point_records.values())
    tasks = [t for ts in task_zones.values() for t in ts]
    m['task_ledger'].update(tasks=tasks, initialized=bool(tasks), inventoried_zones=sorted(task_zones),
                          order=max([0]+[int(t.get('sequence',0)) for t in tasks]))
    m['connectivity']['stage'] = 'PRIMARY'
    m['connectivity']['second_zone_nav'] = {z: sum(t.get('nav_sent',0) for t in ts) for z,ts in task_zones.items()}
    m['connectivity']['second_zone_preflights'] = {z: sum(t.get('checks',0) for t in ts) for z,ts in task_zones.items()}
    if guard is not None:
        m['guard'] = guard
    return plain(m), registry.public()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage-dir', type=Path, required=True)
    parser.add_argument('--mission-dir', type=Path, action='append', help='optional explicit sessions; otherwise match exact coverage_session in project logs')
    parser.add_argument('--confirm-same-scene', action='store_true', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    coverage_dir = args.coverage_dir.expanduser().resolve()
    bundle = load_bundle(ROOT/'config/exploration/navigation/hospital_zones.json',
                         ROOT/'config/exploration/navigation/hospital_coverage_reference.json.gz')
    coverage = import_legacy_coverage(coverage_dir, bundle)
    directories = args.mission_dir or list((ROOT/'logs/hospital_zone_mission').glob('session_*'))
    reports=[]
    for session in sorted(set(p.expanduser().resolve() for p in directories)):
        path = session/'latest_status.json'
        if not path.is_file():
            if args.mission_dir:raise ValueError('EXPLICIT_MISSION_STATUS_MISSING:' + str(session))
            continue
        status = read_json(path)
        if status.get('coverage_session') != coverage['source_session']:
            if args.mission_dir:raise ValueError('EXPLICIT_MISSION_COVERAGE_MISMATCH:' + str(session))
            continue
        manifest_path = session/'run_manifest.json'
        if not manifest_path.is_file():raise ValueError('MATCHED_MISSION_MANIFEST_MISSING:' + str(session))
        manifest = read_json(manifest_path)
        if manifest.get('options',{}).get('execute') is not True:
            continue
        reports.append((session,status,manifest))
    mission, records = merge_reports(reports,bundle)
    context = project_context(ROOT,bundle)
    historical = [dict(session=s.name, manifest=manifest, summary=status,
                       status_sha256=sha((s/'latest_status.json').read_bytes())) for s,status,manifest in reports]
    payload = dict(schema=SCHEMA, kind='LEGACY_IMPORT', context=context, coverage=coverage,
                   mission=mission, blocked_areas=records, historical_reports=historical,
                   mission_session_ids=[s.name for s,_,_ in reports],
                   legacy_world_identity='OPERATOR_CONFIRMED_NOT_RETROACTIVELY_PROVEN',
                   created_utc=datetime.now(timezone.utc).isoformat(), active_action_is_historical_only=True)
    out = args.out or (ROOT/'logs/dn012_resume_import'/datetime.now().strftime('import_%Y%m%d_%H%M%S_%f'))
    path,digest = save_checkpoint(out,payload)
    print('[IMPORT_PASS] No ROS nodes or motion were started.')
    print('COVERAGE_SESSION:',coverage['source_session'])
    print('OBSERVED_CELLS:',coverage['observed_cells'])
    print('MATCHED_MISSION_SESSIONS:',len(reports))
    print('CUMULATIVE_GOALS:',mission['counters']['sent_goals'])
    print('CHECKPOINT:',path)
    print('PAYLOAD_SHA256:',digest)
    print('LATEST:',Path(out).resolve()/'LATEST.json')

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SystemExit('[IMPORT_REFUSED] '+str(exc))
