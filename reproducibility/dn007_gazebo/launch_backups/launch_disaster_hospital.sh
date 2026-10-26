#!/bin/bash

source /opt/ros/humble/setup.bash

PROJECT_ROOT="$HOME/Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping"

# RGB-D PointCloud 좌표계 보정 노드 자동 실행
python3 "$PROJECT_ROOT/scripts/pointcloud_to_optical.py" &
POINTCLOUD_PID=$!

# 이 스크립트가 종료될 때 보정 노드도 같이 종료
cleanup()
{
  kill "$POINTCLOUD_PID" 2>/dev/null
}

trap cleanup EXIT INT TERM

# Disaster Hospital + J100 실행
ros2 launch clearpath_gz simulation.launch.py \
  setup_path:=$HOME/clearpath/ \
  world:=disaster_hospital \
  x:=1.47 \
  y:=20.62 \
  yaw:=-1.58
