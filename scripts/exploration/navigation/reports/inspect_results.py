#!/usr/bin/env python3
"""Inspect saved region results and reroute memory without starting ROS."""
import argparse
import json
from pathlib import Path
import sys

def regions_main(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--session',type=Path);g.add_argument('--latest',action='store_true')
    p.add_argument('--export-json',type=Path,help='Optional NEW output file; existing files are not overwritten')
    a=p.parse_args(argv);root=a.project.expanduser().resolve()
    if a.session:
        session=a.session.expanduser()
        if not session.is_absolute():session=root/session
    else:
        sessions=[]
        for f in (root/'logs/hospital_zone_mission').glob('session_*'):
            if any((f/'region_review'/name).is_file() for name in ('two_phase_latest.json','latest.json')):
                sessions.append(f)
        if not sessions:raise ValueError('NO_REGION_REPORTS_FOUND: select --session explicitly if output-dir differs')
        session=max(sessions,key=lambda f:f.name)
    path=session/'region_review/two_phase_latest.json'
    if not path.is_file():path=session/'region_review/latest.json'
    if not path.is_file():raise ValueError('REPORT_NOT_FOUND:'+str(path))
    data=json.loads(path.read_text(encoding='utf-8'))
    print('SESSION:',session)
    print('VERSION:',data.get('version'),'STAGE:',data.get('stage','legacy'))
    print('INVENTORY_COMPLETE:',data.get('inventory_complete','not recorded'))
    print('COUNTS:',json.dumps(data.get('counts',{}),ensure_ascii=False))
    print('SAFE_CANCELS:',data.get('second_pass_safety_cancels',0))
    print('\nZONE / AREA m2 / STATUS / REASON / CHECKS / NAV')
    for t in data.get('tasks',[]):
        print(f"{t['zone_id']} / {t['area_m2']:.3f} / {t['status']} / {t['reason']} / {t['checks']} / {t['nav_sent']}")
    print('\nOperational unavailable != physical blockage. Budgets/timeouts remain UNVERIFIED.')
    print('Blocked regions are scoped grid-conditioned robot-center subsets, not full inaccessible floor area.')
    print('This patch does not implement victim detection or render a final map/confirmed barrier line.')
    if a.export_json:
        with a.export_json.expanduser().open('x',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        print('EXPORTED:',a.export_json)
    return 0

def results_main(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    a=p.parse_args(argv);s=a.session.expanduser().resolve()
    path=s/'region_review/final_results_v16.json'
    if not path.is_file():
        raise SystemExit('MISSING_V16_FINAL_RESULTS: '+str(path))
    d=json.loads(path.read_text(encoding='utf-8'))
    if d.get('schema')!='dn012_four_results_v1_6':raise SystemExit('WRONG_REPORT_SCHEMA')
    print('SESSION:',s)
    for key,label in d['category_labels'].items():
        rows=d['categories'][key]
        print(f'\n=== {label}: {len(rows)} ===')
        for r in rows:
            keys=('zone_id','task_id','id','area_m2','centroid_map_m','xy','source',
                  'location_basis','historical','reason','checks','nav_sent','basis')
            print(json.dumps({k:r[k] for k in keys if k in r},ensure_ascii=False))
    print('\n검사 미완료 Task:',len(d['evaluation']['incomplete_tasks']))
    print('과거 기록 지점:',d['evaluation']['historical_points_count'])
    print('Coverage 분모/관측값은 변경하지 않음.')

def memory_main(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--session',type=Path,help='Exact mission output directory')
    group.add_argument('--latest',action='store_true',help='Latest mission (default)')
    group.add_argument('--legacy',action='store_true',help='Inspect old shared memory explicitly')
    a=p.parse_args(argv)
    root=a.project.expanduser().resolve()
    if a.legacy:
        files=sorted((root/'.dn012_reroute_memory').glob('points_*.json'))
    else:
        session=a.session
        if session is None:
            sessions=[s for s in (root/'logs/hospital_zone_mission').glob('session_*') if s.is_dir()]
            if not sessions:raise ValueError('NO_MISSION_SESSIONS_FOUND')
            session=max(sessions,key=lambda s:s.name)
        else:
            session=session.expanduser()
            if not session.is_absolute():session=root/session
        if not session.is_dir():raise ValueError('SESSION_DIRECTORY_NOT_FOUND')
        print('SESSION:',session)
        path=session/'memory/reroute_points.json'
        files=[path] if path.is_file() else []
    if not files:print('NO_REROUTE_POINT_FILE_YET')
    for path in files:
        d=json.loads(path.read_text());print('\nMEMORY:',path,'SCENE:',d['context']['scene_id'])
        for r in d['records']:
            print(r['id'],'point=',r['point_xy'],'anchor=',r['anchor_xy'],
                'extra_m=',round(r['extra_m'],3),'increase_pct=',round(100*r['increase_fraction'],1),
                'mask_eligible=',r['mask_eligible'],'source=',r['correction_source'],
                'spatial_zone=',r.get('spatial_zone'),'hold_reason=',r.get('mask_hold_reason'))
        print('POINTS:',len(d['records']),'NOTE: points are operational candidates, not physical cut proof')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('regions', 'results', 'memory'))
    if not argv or argv[0] in ('-h', '--help'):
        parser.parse_args(argv)
    mode = parser.parse_args(argv[:1]).mode
    try:
        return {'regions': regions_main, 'results': results_main, 'memory': memory_main}[mode](argv[1:])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit('REPORT_READ_ERROR: ' + str(exc)) from exc


if __name__ == '__main__':
    raise SystemExit(main())
