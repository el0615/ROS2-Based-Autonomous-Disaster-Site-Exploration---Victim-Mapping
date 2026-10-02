#!/bin/bash

source /opt/ros/humble/setup.bash

ros2 launch clearpath_gz simulation.launch.py \
  setup_path:=$HOME/clearpath/ \
  world:=disaster_hospital \
  x:=1.47 \
  y:=20.62 \
  yaw:=-1.58
