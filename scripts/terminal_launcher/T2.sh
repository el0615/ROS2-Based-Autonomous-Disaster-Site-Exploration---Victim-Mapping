#!/usr/bin/env bash
set -euo pipefail
exec bash "$PROJECT/scripts/exploration/navigation/reroute/run_nav2.sh" "${DN012_REROUTE_MODE:-observe}"
