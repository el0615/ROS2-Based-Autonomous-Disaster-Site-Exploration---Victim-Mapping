#!/usr/bin/env python3
"""Imports + syntax + installed plugin metadata; creates NO ROS node or action."""
from pathlib import Path
import argparse
import ast
import importlib
import json
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--ros-interfaces',action='store_true');a=p.parse_args();root=a.project.expanduser().resolve()
    src=root/'scripts/exploration/zone_exploration/src'
    sys.path[:0]=[str(src),str(root/'scripts/exploration')]
    for name in ('reroute_core','reroute_runtime','path_blockage_runtime','zone_node'):
        ast.parse((src/(name+'.py')).read_text(),feature_version=(3,10));importlib.import_module(name)
    from reroute_core import RerouteSettings
    cfg=root/'config/exploration/navigation/dn012_reroute_v1_4.json'
    RerouteSettings.from_file(cfg)
    if a.ros_interfaces:
        from ament_index_python.packages import get_package_share_directory
        from nav2_msgs.msg import CostmapFilterInfo
        from nav_msgs.msg import OccupancyGrid
        from nav2_msgs.action import NavigateToPose
        import rclpy
        share=Path(get_package_share_directory('nav2_costmap_2d'))
        found=any('nav2_costmap_2d::KeepoutFilter' in x.read_text(errors='replace') for x in share.rglob('*.xml'))
        if not found:raise ValueError('HUMBLE_KEEPOUT_PLUGIN_NOT_FOUND_IN_INSTALLED_PACKAGE')
        print('KEEPOUT_PLUGIN_METADATA_OK:',share)
    print('IMPORT_SYNTAX_OK: no ROS node/action/Goal/cmd_vel created')
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print('VERIFY_FAILED:',type(exc).__name__,str(exc),file=sys.stderr);raise SystemExit(2)
