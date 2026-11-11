#!/usr/bin/env bash
# DN012 v1.6: same T6 supervisor; per-Zone interior ENTRY + rooms-first + observe.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${DN012_PROJECT:-$(cd "$HERE/../../../.." && pwd)}"
MODE="${1:-observe}"
if [[ $# -gt 0 ]]; then shift; fi
ACTION="${1:-dry-run}"
if [[ $# -gt 0 ]]; then shift; fi
[[ "$MODE" == observe ]] || { echo '[V16_APPLY_DISABLED] Use observe; keepout is disabled.' >&2; exit 2; }
case "$ACTION" in execute|dry-run) ;; *) echo 'ACTION must be execute|dry-run' >&2; exit 2;; esac
export PYTHONNOUSERSITE=1
set +u
source /opt/ros/humble/setup.bash
set -u
source "$PROJECT/config/exploration/mission_environment.sh"

# DN012 v1.8: explicit, uniform region minimum; Nav2 remains unchanged.
export DN012_V18_ENABLE=1
export DN012_REGION_MIN_AREA_M2=1.0 DN012_SECOND_ENTRY_MIN_AREA_M2=1.0 DN012_REGION_PRIORITY_AREA_M2=1.0
# Explicit values defeat a leftover 'apply'/'nearest' from an older terminal.
export DN012_REROUTE_MODE=observe
export DN012_ZONE_POLICY=rooms-first
export DN012_REGION_RESCUE=1 DN012_REGION_TWO_PASS=1 DN012_REGION_MODE=rescue
# Separate from legacy cut-analysis size: only second-pass selection changes.
export DN012_SECOND_ENTRY_MIN_AREA_M2=1.0
export PROJECT
python3 "$HERE/verify_runtime.py" --project "$PROJECT" --ros-interfaces
python3 - "$PROJECT" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'scripts/exploration/zone_exploration/src'))
from two_phase import VERSION
from two_phase import TwoPassSettings
from region_entry import build_entry_catalog
if VERSION!='DN012_REGION_ENTRY_V1_6':raise SystemExit('V16_POLICY_NOT_INSTALLED')
print('[V16_ENTRY_SETTINGS]',TwoPassSettings.from_environment())
PY
printf '[V16_T6] mode=observe zone_policy=rooms-first schedule=zone-local interior-entry min-area=1.0 global-second=off action=%s\n' "$ACTION"
cd "$PROJECT"
exec bash "$PROJECT/scripts/exploration/navigation/run_legacy_mission.sh" budget "$ACTION" "$@"
