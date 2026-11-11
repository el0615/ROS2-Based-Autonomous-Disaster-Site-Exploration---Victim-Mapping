#!/usr/bin/env bash
# Read-only reporting; does not start ROS nodes.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE=assessment
case "${1:-}" in assessment|coverage) MODE="$1"; shift ;; esac
if [[ "$MODE" == assessment ]]; then
  PROJECT="${PROJECT:-$HOME/Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping}"
  exec /usr/bin/python3 "$HERE/zone_report.py" assessment --project "$PROJECT" "$@"
fi
exec "${PYTHON:-python3}" "$HERE/zone_report.py" coverage "$@"
