#!/usr/bin/env python3
"""Standalone entry point; leaves the installed DN-011 v1-v4 files unchanged."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

DEFAULT_PROJECT=Path.home()/'Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping'

def main():
    bootstrap=argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--project',type=Path,default=DEFAULT_PROJECT)
    opts,rest=bootstrap.parse_known_args()
    project=opts.project.expanduser().resolve()
    script_dir=project/'scripts/exploration'
    for name in ('hospital_coverage_core.py','hospital_zone_visualizer.py','entrance_navigation_overlay.py'):
        if not (script_dir/name).is_file():
            print('[MISSING_PROJECT_DEPENDENCY]',script_dir/name,file=sys.stderr)
            return 2
    os.environ['ZB_PROJECT_ROOT']=str(project)
    shared_dir=Path(__file__).resolve().parent.parent
    if shared_dir != script_dir:
        # Shared algorithms belong to this checkout; project inputs keep precedence.
        sys.path.insert(0,str(shared_dir))
    sys.path.insert(0,str(script_dir))
    sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
    sys.argv=[sys.argv[0],*rest]
    try:
        from zone_node import main as run
        return run()
    except (ValueError,ImportError,OSError,KeyError,RuntimeError) as exc:
        print('[ZONE_MISSION_ERROR]',type(exc).__name__+':',exc,file=sys.stderr)
        return 2

if __name__=='__main__':raise SystemExit(main())
