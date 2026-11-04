#!/usr/bin/env bash
# Read-only environment setup; choose dry-run or an explicit supervised execute.
set -eo pipefail
MODE="${1:-dry-run}"
if [ "$#" -gt 0 ]; then shift; fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-$HOME/Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping}"
if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "[ERROR] /opt/ros/humble/setup.bash not found" >&2; exit 2
fi
source /opt/ros/humble/setup.bash
EXTRA=()
case "$MODE" in
  dry-run) echo "[MODE] DRY RUN: plans only; no navigation requested by this node." ;;
  execute) EXTRA=(--execute --supervised); echo "[MODE] EXECUTE: Gazebo J100 may move. Supervision and cancellation ready." ;;
  *) echo "Usage: bash run_pilot.sh dry-run|execute [zone/policy options]" >&2; exit 2 ;;
esac
exec env PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH:-}" PYTHONNOUSERSITE=1 \
 /usr/bin/python3 "$ROOT/run_zone_exploration.py" --project "$PROJECT" \
 "${EXTRA[@]}" --max-goals 6 --max-candidates 15 --entry-candidates 8 \
 --entrance-nav-only "$@" --ros-args -p use_sim_time:=true \
 -r /tf:=/j100_0001/tf -r /tf_static:=/j100_0001/tf_static
