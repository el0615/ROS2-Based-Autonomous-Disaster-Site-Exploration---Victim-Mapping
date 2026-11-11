#!/usr/bin/env bash
# Historical HOME supervisor with explicit policy presets.
set -eo pipefail
PRESET="${1:-home}"
if [[ $# -gt 0 ]]; then shift; fi
MODE="${1:-dry-run}"
if [[ $# -gt 0 ]]; then shift; fi
usage() {
  echo 'Usage: run_legacy_mission.sh home|connectivity|blockage|budget dry-run|execute|return-only [options]' >&2
}
DEFAULTS=()
case "$PRESET" in
  home) ;;
  connectivity|blockage|budget)
    DEFAULTS+=(--connectivity-aware --continue-on-nav-abort)
    if [[ "$PRESET" != connectivity ]]; then
      DEFAULTS+=(--path-blockage-check --prior-topology-fixed)
    fi
    if [[ "$PRESET" == budget ]]; then
      DEFAULTS+=(--mission-budget-guard --audit-max-view-preflights 24)
    fi
    ;;
  *) usage; exit 2 ;;
esac
case "$MODE" in
  dry-run|execute) ;;
  return-only)
    [[ "$PRESET" != budget ]] || { usage; exit 2; }
    MODE=execute
    ;;
  *) usage; exit 2 ;;
esac
source /opt/ros/humble/setup.bash
set -u
PROJECT="${PROJECT:-$HOME/Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping}"
SCRIPT="$PROJECT/scripts/exploration/navigation/return_home/run_mission_with_home.py"
[[ -f "$SCRIPT" ]] || { echo "[MISSING] $SCRIPT" >&2; exit 2; }
# Explicit arguments remain after preset defaults so callers can override them.
exec env PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH:-}" PYTHONNOUSERSITE=1 \
  /usr/bin/python3 "$SCRIPT" --project "$PROJECT" --mode "$MODE" "${DEFAULTS[@]}" "$@" \
  --ros-args -p use_sim_time:=true -r /tf:=/j100_0001/tf -r /tf_static:=/j100_0001/tf_static
