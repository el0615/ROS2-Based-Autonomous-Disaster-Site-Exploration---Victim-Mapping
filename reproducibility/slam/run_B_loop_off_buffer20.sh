#!/bin/bash
set -e -o pipefail
WS=/home/robotics/slam_heading_ab_ws
# Avoid inheriting a different ROS/colcon overlay or a virtual Python environment.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source /opt/ros/humble/setup.bash

[[ -f "$WS/install/local_setup.bash" ]] || { echo "Build first."; exit 1; }
source "$WS/install/local_setup.bash"
prefix="$(ros2 pkg prefix slam_toolbox)"
[[ "$prefix" == "$WS/install/slam_toolbox" ]] || {
  echo "Wrong package prefix: $prefix"; exit 1;
}
if ! nodes="$(timeout 15s ros2 node list)"; then
  echo "Cannot verify active ROS nodes; no new SLAM node started."; exit 1
fi
if printf '%s\n' "$nodes" | grep -Fxq '/slam_toolbox'; then
  echo "An existing /slam_toolbox is running. Stop it before starting this test."; exit 1
fi
printf '\nTEST B: check_min_dist_and_heading_precisely=true\n'
printf 'Package: %s\nConfig: %s\n' "$prefix" "$WS/config/test_B_loop_off_buffer20.yaml"
echo "Use the unchanged Normal Hospital launch and RViz in separate terminals."
echo "EXPERIMENT: SLAM B | Loop OFF | scan_buffer_size=20"
exec ros2 run slam_toolbox async_slam_toolbox_node \
  --ros-args --params-file "$WS/config/test_B_loop_off_buffer20.yaml" \
  -p use_sim_time:=true \
  -r /tf:=/j100_0001/tf -r /tf_static:=/j100_0001/tf_static
