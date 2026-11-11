#!/usr/bin/env bash
# DN012 v1.5: ORIGINAL C6 YAML, no KeepoutFilter or mask server.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${DN012_PROJECT:-$(cd "$HERE/../../../.." && pwd)}"
MODE="${1:-observe}"
[[ "$MODE" == observe ]] || { echo '[V15_APPLY_DISABLED] Use observe; keepout is disabled.' >&2; exit 2; }
[[ $# -le 1 ]] || { echo 'Usage: run_nav2.sh observe' >&2; exit 2; }
export PYTHONNOUSERSITE=1
set +u
source /opt/ros/humble/setup.bash
set -u
PARAMS="$PROJECT/config/nav2/nav2_hospital_navigation.yaml"
[[ -f "$PARAMS" ]] || { echo "[MISSING] $PARAMS" >&2; exit 2; }
# Fail closed if the supposed original YAML itself contains a costmap filter.
python3 - "$PARAMS" <<'PY'
import sys, yaml
with open(sys.argv[1],encoding='utf-8') as f: doc=yaml.safe_load(f)
for name in ('global_costmap','local_costmap'):
    params=doc[name][name]['ros__parameters']
    if params.get('filters'):
        raise SystemExit('[V15_ORIGINAL_YAML_HAS_FILTERS] Refusing filtered Nav2 configuration')
print('[V15_T2] original C6 YAML; no reroute mask server; no KeepoutFilter')
PY
exec ros2 launch nav2_bringup bringup_launch.py \
  namespace:=j100_0001 use_namespace:=True \
  map:="$PROJECT/maps/normal/normal_hospital_navigation.yaml" \
  params_file:="$PARAMS" \
  use_sim_time:=True slam:=False use_composition:=False autostart:=True
