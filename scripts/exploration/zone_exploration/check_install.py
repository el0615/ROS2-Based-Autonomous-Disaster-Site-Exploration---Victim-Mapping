#!/usr/bin/env python3
"""Read project compatibility; optionally compile an isolated C predictor and run tests.
No ROS action, map changes, parameter changes, pip installs or legacy file writes.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
DEFAULT=Path.home()/'Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=DEFAULT)
    p.add_argument('--build',action='store_true')
    p.add_argument('--skip-project',action='store_true',help='Offline test only, never a deployment-ready claim')
    a=p.parse_args();project=a.project.expanduser().resolve()
    for code in ROOT.rglob('*.py'):
        ast.parse(code.read_text(),filename=str(code),feature_version=(3,10))
    if not a.skip_project:
        required=['scripts/exploration/hospital_coverage_core.py','scripts/exploration/hospital_zone_visualizer.py',
            'scripts/exploration/entrance_navigation_overlay.py','config/exploration/hospital_zones.json',
            'config/exploration/hospital_coverage_reference.json.gz']
        for relative in required:
            if not (project/relative).is_file():raise RuntimeError('Missing project file: '+relative)
        sys.path.insert(0,str(project/'scripts/exploration'));sys.path.insert(0,str(ROOT/'src'))
        from hospital_coverage_core import load_bundle
        from zb_reference_planning import Settings, SensorModel
        from hospital_zone_visualizer import verify_map
        from entrance_navigation_overlay import apply_entrance_overlay
        b=load_bundle(project/'config/exploration/hospital_zones.json',project/'config/exploration/hospital_coverage_reference.json.gz')
        zones=b[0]['zones'];ids=[z['id'] for z in zones]
        if len(ids)!=len(set(ids)) or len(ids)==0:raise RuntimeError('Duplicate/empty zone IDs')
        for zone in zones:
            if int((b[2]==zone['mask_id']).sum())!=int(zone['known_free_cells']):raise RuntimeError('Zone denominator mismatch')
        print('[PROJECT_BUNDLE_OK] zones='+str(len(zones))+' signature='+str(b[4]),flush=True)
        print('[PROJECT_UNCHANGED] Existing Coverage/masks/map/Nav2/v1-v4 files are read only.',flush=True)
    if a.build:
        cc=shutil.which('cc') or shutil.which('gcc')
        if not cc:raise RuntimeError('No C compiler. Install build-essential separately or use --visibility-backend python; no automatic package install.')
        src=ROOT/'src';lib=src/'libzb_visibility.so';tmp=src/('._libzb_'+str(os.getpid())+'.so')
        subprocess.run([cc,'-O3','-std=c99','-fPIC','-shared',str(src/'zb_visibility_dda.c'),'-o',str(tmp),'-lm'],check=True)
        previous=lib.read_bytes() if lib.exists() else None
        receipt=src/'visibility_build_receipt.json'
        previous_receipt=receipt.read_bytes() if receipt.exists() else None
        try:
            os.replace(tmp,lib)
            r=subprocess.run([sys.executable,str(ROOT/'tests/verify_visibility.py')],capture_output=True,text=True)
            print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
            if r.returncode:raise RuntimeError('Synthetic C equivalence checks failed')
            prefix='[VISIBILITY_EQUIVALENCE_PASS] '
            result=json.loads(next(x[len(prefix):] for x in r.stdout.splitlines() if x.startswith(prefix)))
            report=dict(equivalence_pass=True,synthetic_comparisons=result['calls'],
                created_utc=datetime.now(timezone.utc).isoformat(),
                sha256={n:sha(src/n) for n in ('zb_visibility_dda.c','zb_visibility_c.py','libzb_visibility.so')})
            receipt.write_text(json.dumps(report,indent=2)+'\n')
        except BaseException:
            if previous is None:lib.unlink(missing_ok=True)
            else:lib.write_bytes(previous)
            if previous_receipt is None:receipt.unlink(missing_ok=True)
            else:receipt.write_bytes(previous_receipt)
            raise
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_*.py','-v'],capture_output=True,text=True)
    print(test.stdout,end='');print(test.stderr,end='',file=sys.stderr)
    if test.returncode:raise RuntimeError('Offline/mock tests failed')
    if not a.skip_project:
        print('[ZONE_BASED_READY] Project inputs loaded; offline/mock tests passed. LIVE GAZEBO STILL UNTESTED.',flush=True)
    else:
        print('[OFFLINE_CHECK_PASS] Synthetic/mock only; no project files or ROS were read.',flush=True)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,ImportError,RuntimeError,subprocess.CalledProcessError,StopIteration) as exc:
        print('[CHECK_FAILED]',exc,file=sys.stderr);raise SystemExit(2)
