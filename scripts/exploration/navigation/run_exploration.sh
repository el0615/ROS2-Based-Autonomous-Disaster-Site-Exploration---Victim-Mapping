#!/usr/bin/env bash
# Bedless v1 wrapper for unchanged DN-011 v2.4. All 27 zones by default.
set -euo pipefail
MODE="${1:-dry-run}"
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  dry-run|execute) ;;
  *) echo "Usage: bash run_exploration.sh dry-run|execute [--zone-ids ...] [--max-goals ...]" >&2; exit 2 ;;
esac
PROJECT="${PROJECT:-$HOME/Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping}"
BED="$PROJECT/config/exploration/navigation"
ZB="$PROJECT/scripts/exploration/zone_exploration"
for req in "$BED/hospital_zones.json" "$BED/hospital_coverage_reference.json.gz" "$BED/hospital_zone_mask.json.gz" \
           "$BED/entrance_nav_only_overlay.json" "$ZB/run_pilot.sh"; do
  if [ ! -f "$req" ]; then echo "[BEDLESS_MISSING] $req" >&2; exit 2; fi
done
if [ "${MODE}" = execute ]; then
  echo "[BEDLESS_EXECUTE] Active Gazebo/Nav2 and LIVE matching Coverage required. Supervise robot and have cancellation available." >&2
fi
exec bash "$ZB/run_pilot.sh" "$MODE" \
 --zones "$BED/hospital_zones.json" \
 --reference "$BED/hospital_coverage_reference.json.gz" \
 --entrance-overlay-file "$BED/entrance_nav_only_overlay.json" \
 "$@"
