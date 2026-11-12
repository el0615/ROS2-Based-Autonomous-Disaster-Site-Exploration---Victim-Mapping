"""Launch the project's world while keeping Clearpath robot setup and bridges."""
from pathlib import Path
import os
import shlex
import tempfile

from ament_index_python.packages import get_package_share_directory
from clearpath_config.clearpath_config import ClearpathConfig
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    root = Path(__file__).resolve().parents[1]
    world = LaunchConfiguration('world').perform(context)
    if world not in ('normal_hospital', 'disaster_hospital'):
        raise ValueError('UNKNOWN_HOSPITAL_WORLD: ' + world)
    world_file = root / 'gazebo/worlds' / world.removesuffix('_hospital') / (world + '.sdf')
    if not world_file.is_file():
        raise FileNotFoundError(world_file)
    setup = LaunchConfiguration('setup_path').perform(context)
    clearpath = Path(get_package_share_directory('clearpath_gz'))
    simulator = Path(get_package_share_directory('ros_gz_sim'))

    # Retain the installed launcher's namespace-aware GUI teleoperation topic.
    gui = clearpath / 'config/gui.config'
    robot_yaml = Path(setup) / 'robot.yaml'
    if robot_yaml.is_file():
        namespace = ClearpathConfig(str(robot_yaml)).system.namespace
        topic = f'/{namespace}/cmd_vel' if namespace not in ('', '/') else '/cmd_vel'
        content = gui.read_text().replace(
            '<plugin filename="Teleop">',
            f'<plugin filename="Teleop">\n    <topic>{topic}</topic>')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.config', prefix='hospital_gui_', delete=False) as stream:
            stream.write(content)
            gui = Path(stream.name)

    paths = [str(root / 'gazebo/models'), str(clearpath / 'worlds'),
             str(Path.home() / '.gazebo/models'),
             str(Path.home() / 'hospital_world_test3/models'),
             str(Path.home() / 'hospital_world_test3/fuel_models')]
    paths.extend(str(Path(p) / 'share') for p in os.environ.get('AMENT_PREFIX_PATH', '').split(':') if p)
    actions = []
    for variable in ('IGN_GAZEBO_RESOURCE_PATH', 'GZ_SIM_RESOURCE_PATH'):
        resources = paths + [p for p in os.environ.get(variable, '').split(':') if p]
        actions.append(SetEnvironmentVariable(variable, os.pathsep.join(dict.fromkeys(resources))))
    # ros_gz_sim executes gz_args through a shell; quote paths containing spaces / &.
    arguments = shlex.join([str(world_file), '-r', '-v', '4', '--gui-config', str(gui)])
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(simulator / 'launch/gz_sim.launch.py')),
        launch_arguments={'gz_args': arguments}.items()))
    actions.append(Node(package='ros_gz_bridge', executable='parameter_bridge',
        name='clock_bridge', output='screen',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock']))
    robot_arguments = {key: LaunchConfiguration(key) for key in
        ('world', 'setup_path', 'x', 'y', 'z', 'yaw', 'rviz', 'generate', 'use_sim_time')}
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(clearpath / 'launch/robot_spawn.launch.py')),
        launch_arguments=robot_arguments.items()))
    return actions


def generate_launch_description():
    defaults = dict(world='disaster_hospital', setup_path=str(Path.home() / 'clearpath'),
                    x='0.0', y='0.0', z='0.3', yaw='0.0', rviz='false',
                    generate='true', use_sim_time='true')
    return LaunchDescription([
        *[DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()],
        OpaqueFunction(function=launch_setup),
    ])
