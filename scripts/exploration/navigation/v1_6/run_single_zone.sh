#!/usr/bin/env bash
# DN012 v1.6 scoped pilot. No return-home supervisor; no C6/coverage changes.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${DN012_PROJECT:-$(cd "$HERE/../../../.." && pwd)}"
ZONE="${1:-R12}"
ACTION="${2:-dry-run}"
[[ $# -le 2 ]] || { echo 'Usage: run_single_zone.sh R12 dry-run|execute' >&2; exit 2; }
case "$ACTION" in dry-run|execute) ;; *) echo 'Invalid action' >&2; exit 2;; esac
# Validate scope before sourcing ROS or creating any nodes.
python3 - "$PROJECT" "$ZONE" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);z=sys.argv[2]
d=json.loads((root/'config/exploration/navigation/hospital_zones.json').read_text())
if z not in {i['id'] for i in d['zones']}:raise SystemExit('UNKNOWN_ZONE:'+z)
print('[V16_SCOPE]',z,'ONLY; transit through other Zones allowed; no return-home')
PY
set +u
source /opt/ros/humble/setup.bash
set -u
source "$PROJECT/config/exploration/mission_environment.sh"
export PROJECT PYTHONNOUSERSITE=1
export DN012_REROUTE_MODE=observe DN012_ZONE_POLICY=rooms-first
export DN012_REGION_RESCUE=1 DN012_REGION_TWO_PASS=1 DN012_REGION_MODE=rescue
export DN012_SECOND_ENTRY_MIN_AREA_M2=1.0
python3 - "$PROJECT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'scripts/exploration/zone_exploration/src'))
from two_phase import VERSION,TwoPassSettings
if VERSION!='DN012_REGION_ENTRY_V1_6':raise SystemExit('V16_POLICY_NOT_INSTALLED')
print('[V16_SINGLE_ZONE]',VERSION,TwoPassSettings.from_environment())
PY
cd "$PROJECT"
exec bash "$PROJECT/scripts/exploration/navigation/run_exploration.sh" "$ACTION" \
 --connectivity-aware --path-blockage-check --prior-topology-fixed \
 --continue-on-nav-abort --mission-budget-guard --audit-max-view-preflights 24 \
 --threshold 0.96 --max-goals 100 --max-plan-attempts 300 --zone-ids "$ZONE"
