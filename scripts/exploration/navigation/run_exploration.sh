#!/usr/bin/env bash
# Exploration runner. The optional mission preset enables the current supervisor.
set -euo pipefail

PRESET=exploration
case "${1:-}" in mission|pilot) PRESET="$1"; shift ;; esac
MODE="${1:-dry-run}"
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  dry-run|execute) ;;
  *) echo "Usage: run_exploration.sh [mission|pilot] dry-run|execute [zone/policy options]" >&2; exit 2 ;;
esac
PROJECT="${PROJECT:-$HOME/Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping}"
if [[ "$PRESET" == mission ]]; then
  HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT="$(cd "$HERE/../../.." && pwd)"
fi
set +u
source /opt/ros/humble/setup.bash
set -u
DEFAULTS=()
if [[ "$PRESET" == mission ]]; then
  source "$PROJECT/config/exploration/mission_environment.sh"
  export PROJECT ZB_PROJECT_ROOT="$PROJECT" PYTHONNOUSERSITE=1
  export DN012_REROUTE_MODE=observe DN012_ZONE_POLICY=rooms-first
  DEFAULTS=(--connectivity-aware --path-blockage-check --prior-topology-fixed
    --continue-on-nav-abort --mission-budget-guard --audit-max-view-preflights 24
    --threshold 0.96 --max-goals 100 --max-plan-attempts 300)
  cd "$PROJECT"
fi
# Explicit, uniform region minimum; Nav2 remains unchanged.
if [[ "$PRESET" != pilot ]]; then
  export DN012_V18_ENABLE=1
  export DN012_REGION_MIN_AREA_M2=1.0 DN012_SECOND_ENTRY_MIN_AREA_M2=1.0 DN012_REGION_PRIORITY_AREA_M2=1.0
fi
if [[ "$PRESET" == mission && "${DN012_V189_CHILD:-0}" != 1 ]]; then
  exec /usr/bin/python3 "$PROJECT/scripts/exploration/navigation/return_home/mission_coordinator.py" mission "$MODE" "$@"
fi
BED="$PROJECT/config/exploration/navigation"
ZB="$PROJECT/scripts/exploration/zone_exploration"
if [[ "$PRESET" == pilot ]]; then
  # A separate PROJECT selects data; standalone pilots use this checkout's code.
  HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  ZB="$(cd "$HERE/../zone_exploration" && pwd)"
fi
MAP_ARGS=()
if [[ "$PRESET" != pilot ]]; then
  for req in "$BED/hospital_zones.json" "$BED/hospital_coverage_reference.json.gz" "$BED/hospital_zone_mask.json.gz" \
           "$BED/entrance_nav_only_overlay.json" "$ZB/run_zone_exploration.py"; do
    if [ ! -f "$req" ]; then echo "[BEDLESS_MISSING] $req" >&2; exit 2; fi
  done
  MAP_ARGS=(--zones "$BED/hospital_zones.json" --reference "$BED/hospital_coverage_reference.json.gz"
    --entrance-overlay-file "$BED/entrance_nav_only_overlay.json")
fi
EXTRA=()
if [[ "$MODE" == execute ]]; then
  if [[ "$PRESET" != pilot ]]; then
    echo "[BEDLESS_EXECUTE] Active Gazebo/Nav2 and LIVE matching Coverage required. Supervise robot and have cancellation available." >&2
  fi
  EXTRA=(--execute --supervised)
  echo '[MODE] EXECUTE: Gazebo J100 may move. Supervision and cancellation ready.'
else
  echo '[MODE] DRY RUN: plans only; no navigation requested by this node.'
fi
exec env PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH:-}" PYTHONNOUSERSITE=1 \
 /usr/bin/python3 "$ZB/run_zone_exploration.py" --project "$PROJECT" \
 "${EXTRA[@]}" --max-goals 6 --max-candidates 15 --entry-candidates 8 --entrance-nav-only \
 "${MAP_ARGS[@]}" "${DEFAULTS[@]}" "$@" --ros-args -p use_sim_time:=true \
 -r /tf:=/j100_0001/tf -r /tf_static:=/j100_0001/tf_static
