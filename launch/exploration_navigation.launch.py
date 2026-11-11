#!/usr/bin/env python3
"""Start only Nav2 navigation nodes in /j100_0001 using external online SLAM.

The external slam_toolbox publishes /map and map->odom TF on /j100_0001/tf.
No AMCL / map_server / second SLAM process is launched here.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    nav2_launch = os.path.join(
        get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
    namespace = LaunchConfiguration('namespace')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='j100_0001'),
        DeclareLaunchArgument('params_file', description='Absolute YAML path'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        GroupAction([
            PushRosNamespace(namespace),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav2_launch),
                launch_arguments={
                    'namespace': namespace,
                    'params_file': params_file,
                    'use_sim_time': use_sim_time,
                    'autostart': 'true',
                    'use_composition': 'False',
                }.items(),
            ),
        ]),
    ])
