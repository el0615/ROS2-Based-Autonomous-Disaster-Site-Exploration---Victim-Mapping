#!/usr/bin/env bash
# New opt-in T6 entry point. Never overwrites zone_based_v2_4.
set -eo pipefail
MODE="${1:-dry-run}"
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  dry-run|execute) ;;
  return-only) MODE=execute ;;
  *) echo "Usage: bash run_mission.sh dry-run|execute|return-only [--source-report FILE] [--return-plan-only]" >&2; exit 2 ;;
esac
source /opt/ros/humble/setup.bash
# ROS 2 Humble setup may reference unset AMENT_TRACE_SETUP_FILES; enable nounset only after source.
set -u
PROJECT="${PROJECT:-$HOME/Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping}"
SCRIPT="$PROJECT/scripts/exploration/navigation/return_home/run_mission_with_home.py"
if [ ! -f "$SCRIPT" ]; then echo "[MISSING] $SCRIPT" >&2; exit 2; fi
exec env PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH:-}" PYTHONNOUSERSITE=1 \
 /usr/bin/python3 "$SCRIPT" --project "$PROJECT" --mode "$MODE" "$@" \
 --ros-args -p use_sim_time:=true -r /tf:=/j100_0001/tf -r /tf_static:=/j100_0001/tf_static
