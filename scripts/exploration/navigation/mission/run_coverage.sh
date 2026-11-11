#!/usr/bin/env bash
# T5 only. Stop the old T5 before using this wrapper; no robot motion commands.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../../../.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
set -u
export ZB_PROJECT_ROOT="$PROJECT" PYTHONNOUSERSITE=1
export DN012_V18_ENABLE=1
cd "$PROJECT"
exec /usr/bin/python3 scripts/exploration/hospital_zone_coverage.py \
  --zones config/exploration/navigation/hospital_zones.json \
  --reference config/exploration/navigation/hospital_coverage_reference.json.gz \
  --output-dir logs/hospital_coverage_disaster_trial "$@" \
  --ros-args -p use_sim_time:=true -r /tf:=/j100_0001/tf -r /tf_static:=/j100_0001/tf_static
