#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT"

exec bash scripts/exploration/navigation/mission/run_coverage.sh \
  --map-topic /j100_0001/map \
  --scan-topic /j100_0001/sensors/lidar2d_0/scan
