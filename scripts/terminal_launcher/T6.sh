#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT"

echo "[MISSION] Running preflight checks..."
./scripts/mission.sh start --dry-run

echo
read -r -p "Checks passed. Type START when ready to explore: " ANSWER

if [[ "$ANSWER" != "START" ]]; then
    echo "[MISSION_NOT_STARTED]"
    exit 0
fi

echo "[MISSION] Starting exploration..."
exec ./scripts/mission.sh start
