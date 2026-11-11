#!/usr/bin/env bash
# Open the hospital terminal session with the saved tab settings.
set -euo pipefail

# Explicit, uniform region minimum.
export DN012_V18_ENABLE=1
export DN012_REGION_MIN_AREA_M2=1.0 DN012_SECOND_ENTRY_MIN_AREA_M2=1.0 DN012_REGION_PRIORITY_AREA_M2=1.0
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-observe}"
[[ "$MODE" == observe ]] || { echo '[V15_APPLY_DISABLED] Use observe.' >&2; exit 2; }
export DN012_PROJECT="$PROJECT"
export TERMINAL_LAUNCHER_CONFIG="$PROJECT/scripts/terminal_launcher"
export TERMINAL_LAUNCHER_LOGS="$PROJECT/logs/terminal_launcher"
export DN012_REROUTE_MODE=observe DN012_ZONE_POLICY=rooms-first
exec bash "$PROJECT/scripts/terminal_launcher/launcher.sh" "${2:-start}"
