#!/usr/bin/env bash
# DN012 v1.8.9: initial map HOME in Checkpoint; terminal Zones never reopened.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
set -u
export PROJECT="$ROOT" ZB_PROJECT_ROOT="$ROOT" PYTHONNOUSERSITE=1
cd "$ROOT"
exec /usr/bin/python3 "$ROOT/scripts/exploration/navigation/return_home/mission_coordinator.py" "$@"
