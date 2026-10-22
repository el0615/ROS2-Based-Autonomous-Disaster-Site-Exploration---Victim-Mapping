#!/bin/bash

source /opt/ros/humble/setup.bash

MODELS="$HOME/Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping/gazebo/models"
export IGN_GAZEBO_RESOURCE_PATH="$MODELS${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
export GZ_SIM_RESOURCE_PATH="$MODELS${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

ros2 launch clearpath_gz simulation.launch.py \
  setup_path:=$HOME/clearpath/ \
  world:=normal_hospital \
  x:=1.47 \
  y:=20.62 \
  yaw:=-1.58
