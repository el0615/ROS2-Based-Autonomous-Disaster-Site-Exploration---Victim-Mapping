#!/usr/bin/env bash
# Shared Gazebo launcher; disaster scenes additionally need optical-frame correction.
WORLD="${1:-disaster}"
case "$WORLD" in
  normal|disaster) ;;
  *) echo 'Usage: launch_hospital.sh normal|disaster' >&2; exit 2 ;;
esac
source /opt/ros/humble/setup.bash
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$PROJECT_ROOT/gazebo/models"
export IGN_GAZEBO_RESOURCE_PATH="$MODELS${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
export GZ_SIM_RESOURCE_PATH="$MODELS${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

if [[ "$WORLD" == disaster ]]; then
  python3 "$PROJECT_ROOT/scripts/pointcloud_to_optical.py" &
  POINTCLOUD_PID=$!
  cleanup() { kill "$POINTCLOUD_PID" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
fi

ros2 launch clearpath_gz simulation.launch.py \
  setup_path:="$HOME/clearpath/" world:="${WORLD}_hospital" \
  x:=1.47 y:=20.62 yaw:=-1.58
