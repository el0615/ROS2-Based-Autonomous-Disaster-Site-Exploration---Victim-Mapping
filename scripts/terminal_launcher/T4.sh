#!/usr/bin/env bash
set -euo pipefail
rviz2 -d "$RVIZ_FILE" --ros-args \
  -p use_sim_time:=true \
  -r /tf:=/j100_0001/tf \
  -r /tf_static:=/j100_0001/tf_static
