#!/usr/bin/env bash
# Open the configured simulation, navigation, visualization and mission tabs.
set -euo pipefail

SCRIPT_HOME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${DN012_PROJECT:-$(cd "$SCRIPT_HOME/../.." && pwd)}"
CONFIG="${TERMINAL_LAUNCHER_CONFIG:-${DN012_QUICKSTART_CONFIG:-$SCRIPT_HOME}}"
LOGS="${TERMINAL_LAUNCHER_LOGS:-${DN012_QUICKSTART_LOGS:-$PROJECT/logs/terminal_launcher}}"
RVIZ_FILE="$CONFIG/hospital.rviz"
SESSION_HELPER="$SCRIPT_HOME/session_processes.py"

help() {
  cat <<'EOF'
Usage:
  ./launcher.sh init        # Copy missing settings to a custom config directory
  ./launcher.sh check       # Check files without launching
  ./launcher.sh start       # Open T1-T6 and a T7 stop tab
  ./launcher.sh run-slot T1 # Run one tab (internal)

Tab commands and hospital.rviz live beside this launcher.
TERMINAL_LAUNCHER_CONFIG can select a custom settings directory; init keeps existing files.
EOF
  printf 'RViz config: %s\n' "$RVIZ_FILE"
}

copy_if_absent() {
  local source="$1" target="$2"
  if [[ ! -e "$target" ]]; then
    [[ -f "$source" ]] || { printf '[MISSING] %s\n' "$source" >&2; return 2; }
    install -m 600 "$source" "$target"
    printf '[CREATED] %s\n' "$target"
  else
    printf '[KEPT] %s\n' "$target"
  fi
}

init() {
  mkdir -p "$CONFIG" "$LOGS"
  local name
  for name in T1.sh T2.sh T3.sh T4.sh T5.sh T6.sh hospital.rviz; do
    copy_if_absent "$SCRIPT_HOME/$name" "$CONFIG/$name"
  done
  printf '\nRViz config: %s\n' "$RVIZ_FILE"
  printf 'Tab commands: %s/T{1..6}.sh\n' "$CONFIG"
  printf 'Existing files kept. Next: check\n'
}

check() {
  local errors=0 slot f
  [[ -d "$PROJECT" ]] || { echo "[MISSING] Project: $PROJECT"; errors=$((errors+1)); }
  for f in "$PROJECT/scripts/launch_hospital.sh" "$PROJECT/scripts/mission.sh" \
      "$PROJECT/config/exploration/mission_environment.sh"; do
    if [[ ! -f "$f" ]]; then echo "[MISSING] $f"; errors=$((errors+1));
    elif ! bash -n "$f"; then echo "[SYNTAX_ERROR] $f"; errors=$((errors+1)); fi
  done
  [[ -s "$RVIZ_FILE" ]] || { echo "[MISSING] RViz config: $RVIZ_FILE"; errors=$((errors+1)); }
  [[ -f "$SESSION_HELPER" ]] || { echo "[MISSING] Session manager: $SESSION_HELPER"; errors=$((errors+1)); }
  if ! command -v gnome-terminal >/dev/null; then echo '[MISSING] gnome-terminal'; errors=$((errors+1)); fi
  for slot in T1 T2 T3 T4 T5 T6; do
    f="$CONFIG/$slot.sh"
    if [[ ! -s "$f" ]]; then echo "[MISSING] $f"; errors=$((errors+1)); continue; fi
    if ! bash -n "$f"; then echo "[SYNTAX_ERROR] $f"; errors=$((errors+1)); fi
    if grep -q 'TODO_EXACT_COMMANDS' "$f"; then echo "[NOT_CONFIGURED] $f: add your working command."; errors=$((errors+1)); fi
  done
  if [[ $errors -gt 0 ]]; then
    echo "CHECK_INCOMPLETE: ${errors} issue(s). Nothing launched."
    return 2
  fi
  echo 'CHECK_OK: Files ready. Check ROS readiness after launch.'
}

run_command() {
  local slot="${1:-}"; [[ "$slot" =~ ^T[1-6]$ ]] || { echo 'BAD_SLOT'; return 2; }
  local cmd="$CONFIG/$slot.sh"
  [[ -f "$cmd" ]] || { echo "NO_COMMAND_FILE: $cmd"; return 2; }
  export PROJECT CONFIG RVIZ_FILE
  if [[ "$slot" == T6 ]]; then
    printf '\nCheck T1-T5 are ready and the robot is safe to move.\n'
    printf 'Press Enter to continue, or q then Enter to cancel.\n'
    local answer
    read -r answer || return 130
    if [[ "$answer" == q || "$answer" == Q ]]; then echo '[T6_CANCELED_BY_USER]'; return 0; fi
    if [[ -n "$answer" ]]; then echo '[T6_NOT_STARTED] Press Enter without typing.'; return 2; fi
  fi
  cd "$PROJECT"
  exec bash "$cmd"
}

run_slot() {
  local slot="${1:-}" session="${2:-}" terminal_pid="${3:-}"
  [[ "$slot" =~ ^T[1-7]$ ]] || { echo 'BAD_SLOT'; return 2; }
  [[ -n "$session" ]] || { echo '[NO_SESSION] Use launch_reroute.sh to open the tabs.'; return 2; }
  if [[ "$slot" == T7 ]]; then
    echo 'Stop all T1-T6 processes from this launch.'
    echo 'T6 stops first. Other programs are kept running.'
    local answer
    while read -r -p 'Type STOP to shut down, or q to cancel: ' answer; do
      case "$answer" in
        STOP) /usr/bin/python3 -B "$SESSION_HELPER" stop "$session"; return $? ;;
        q|Q) echo '[STOP_CANCELED]'; return 0 ;;
        *) echo 'Enter STOP or q.' ;;
      esac
    done
    return 130
  fi
  export PROJECT CONFIG RVIZ_FILE
  mkdir -p "$LOGS"
  local outfile="$LOGS/${slot}_$(date +%Y%m%d_%H%M%S).log"
  printf '\n[%s] %s\n' "$slot" "$(date --iso-8601=seconds)"
  printf 'Command: %s\nLog: %s\n' "$CONFIG/$slot.sh" "$outfile"
  local -a ownership_args=()
  if [[ -n "$terminal_pid" ]]; then ownership_args=(--terminal-pid "$terminal_pid"); fi
  set +e
  /usr/bin/python3 -B "$SESSION_HELPER" run "${ownership_args[@]}" "$session" "$slot" -- \
    bash "$SCRIPT_HOME/launcher.sh" run-command "$slot" 2>&1 | tee -a "$outfile"
  local code=${PIPESTATUS[0]}
  set -e
  printf '\n[SLOT_EXIT] %s exit=%s\n' "$slot" "$code"
  return "$code"
}

start() {
  check || return 2
  echo 'Close any running T1-T6 sessions and stop the robot.'
  read -r -p 'Type YES to open T1-T7 in one window: ' ans
  if [[ "$ans" != 'YES' ]]; then echo 'START_CANCELED'; return 1; fi
  local slot mode term_cmd tab_command quoted_launcher session quoted_session title
  local -a terminal_args=(--quiet)
  session="$(/usr/bin/python3 -B "$SESSION_HELPER" create)"
  quoted_session="${session//\'/\'\\\'\'}"
  # Quote the launcher path for GNOME Terminal's per-tab command parser.
  quoted_launcher="${SCRIPT_HOME}/launcher.sh"
  quoted_launcher="${quoted_launcher//\'/\'\\\'\'}"
  term_cmd='"$1" run-slot "$2" "$3" "$$"; rc=$?; echo; echo "[TERMINAL] Tab ended (exit=$rc). Logs are kept."; exec bash'
  for slot in T1 T2 T3 T4 T5 T6 T7; do
    if [[ "$slot" == T1 ]]; then mode=--window; else mode=--tab; fi
    title="$slot"; if [[ "$slot" == T7 ]]; then title='T7 - Stop'; fi
    printf -v tab_command "bash -ic '%s' _ '%s' '%s' '%s'" "$term_cmd" "$quoted_launcher" "$slot" "$quoted_session"
    terminal_args+=("$mode" "--title=$title" "--working-directory=$PROJECT" "--command=$tab_command")
  done
  # One invocation keeps every --tab attached to this new --window.
  gnome-terminal "${terminal_args[@]}"
  echo 'LAUNCH_TABS_OPENED: One window, seven tabs. T6 waits for Enter; T7 stops this session.'
  echo 'Wait for ROS nodes to be ready before continuing.'
}

case "${1:-help}" in
  init) init;;
  check) check;;
  start) start;;
  run-slot) shift; run_slot "${1:-}" "${2:-}" "${3:-}";;
  run-command) shift; run_command "${1:-}";;
  help|-h|--help) help;;
  *) help; exit 2;;
esac
