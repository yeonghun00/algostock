#!/bin/bash
# setup_scheduler.sh — install/uninstall the AlgoStock launchd scheduler.
#
# Usage:
#   ./scripts/setup_scheduler.sh start           # install with defaults
#   ./scripts/setup_scheduler.sh start --run myrun --hour 7 --min 30
#   ./scripts/setup_scheduler.sh stop            # unload and remove
#   ./scripts/setup_scheduler.sh status          # show current state
#
# Time is in YOUR LOCAL TIME (HKT = UTC+8).
# Default: 07:30 HKT = 08:30 KST (30 min before Korean market open).
#
# NOTE: To wake Mac from sleep before this fires, run once:
#   sudo pmset repeat wakeorpoweron MTWRF 07:25:00

PLIST_LABEL="com.algostock.live"
PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_LABEL.plist"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
HOUR=7
MIN=30
RUN_NAME=""          # empty = auto-select latest run
PORTFOLIO=100000000

# ── Parse args ────────────────────────────────────────────────────────────────
ACTION="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)       RUN_NAME="$2";    shift 2 ;;
    --hour)      HOUR="$2";        shift 2 ;;
    --min)       MIN="$2";         shift 2 ;;
    --portfolio) PORTFOLIO="$2";   shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
do_start() {
  echo "Installing AlgoStock scheduler..."
  echo "  Time:      $(printf '%02d:%02d' $HOUR $MIN) (local time)"
  echo "  Run:       ${RUN_NAME:-auto (latest in runs/)}"
  echo "  Portfolio: $PORTFOLIO KRW"
  echo "  Plist:     $PLIST_PATH"

  # Build env vars for the plist
  ENV_ENTRIES="<key>ALGOSTOCK_PORTFOLIO</key><string>$PORTFOLIO</string>"
  if [ -n "$RUN_NAME" ]; then
    ENV_ENTRIES="$ENV_ENTRIES<key>ALGOSTOCK_RUN</key><string>$RUN_NAME</string>"
  fi

  cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$PLIST_LABEL</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$SCRIPT_DIR/scripts/auto_live.sh</string>
  </array>

  <!-- Runs at $(printf '%02d:%02d' $HOUR $MIN) local time every day.     -->
  <!-- auto_live.sh exits early on weekends and non-rebalance days.        -->
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>   <integer>$HOUR</integer>
    <key>Minute</key> <integer>$MIN</integer>
  </dict>

  <!-- If Mac was asleep at scheduled time, run immediately on wake.       -->
  <!-- (This is launchd's catch-up behavior with StartCalendarInterval.)   -->

  <key>EnvironmentVariables</key>
  <dict>
    $ENV_ENTRIES
  </dict>

  <key>StandardOutPath</key>
  <string>$SCRIPT_DIR/live/logs/launchd_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$SCRIPT_DIR/live/logs/launchd_stderr.log</string>

  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
EOF

  mkdir -p "$SCRIPT_DIR/live/logs"
  launchctl load "$PLIST_PATH"
  echo ""
  echo "  Scheduler installed and loaded."
  WAKE_MIN=$((MIN - 5))
  WAKE_HOUR=$HOUR
  if [ $WAKE_MIN -lt 0 ]; then
    WAKE_MIN=$((60 + WAKE_MIN))
    WAKE_HOUR=$((HOUR - 1))
    [ $WAKE_HOUR -lt 0 ] && WAKE_HOUR=23
  fi
  echo "  To also wake Mac from sleep at $(printf '%02d:%02d' $WAKE_HOUR $WAKE_MIN):"
  echo "    sudo pmset repeat wakeorpoweron MTWRF $(printf '%02d:%02d:00' $WAKE_HOUR $WAKE_MIN)"
}

do_stop() {
  echo "Removing AlgoStock scheduler..."
  if launchctl list | grep -q "$PLIST_LABEL"; then
    launchctl unload "$PLIST_PATH" 2>/dev/null && echo "  Unloaded."
  else
    echo "  Not currently loaded."
  fi
  if [ -f "$PLIST_PATH" ]; then
    rm "$PLIST_PATH" && echo "  Plist removed."
  else
    echo "  Plist not found (already removed)."
  fi
  echo ""
  echo "  To also cancel the pmset wake schedule:"
  echo "    sudo pmset repeat cancel"
}

do_status() {
  echo "── Scheduler status ──────────────────────"
  if launchctl list | grep -q "$PLIST_LABEL"; then
    echo "  launchd:  LOADED ($PLIST_LABEL)"
    if [ -f "$PLIST_PATH" ]; then
      HOUR_VAL=$(grep -A1 "Hour" "$PLIST_PATH" | grep integer | grep -oE "[0-9]+")
      MIN_VAL=$(grep -A1 "Minute" "$PLIST_PATH" | grep integer | grep -oE "[0-9]+")
      echo "  Time:     $(printf '%02d:%02d' $HOUR_VAL $MIN_VAL) local"
    fi
  else
    echo "  launchd:  NOT loaded"
  fi

  if [ -f "$PLIST_PATH" ]; then
    echo "  Plist:    $PLIST_PATH"
  else
    echo "  Plist:    not found"
  fi

  echo ""
  PMSET=$(pmset -g sched 2>/dev/null | grep -i "wake\|poweron" || true)
  if [ -n "$PMSET" ]; then
    echo "  pmset wake: $PMSET"
  else
    echo "  pmset wake: not set"
  fi

  echo ""
  echo "── Recent logs ───────────────────────────"
  ls -lt "$SCRIPT_DIR/live/logs/"*.log 2>/dev/null | head -5 | sed 's/^/  /'
}

# ── Main ──────────────────────────────────────────────────────────────────────
case "$ACTION" in
  start)  do_start ;;
  stop)   do_stop  ;;
  status) do_status ;;
  *)
    echo "Usage: $0 {start|stop|status} [--run NAME] [--hour H] [--min M] [--portfolio KRW]"
    echo ""
    echo "Examples:"
    echo "  $0 start                           # 07:30 local, auto run"
    echo "  $0 start --run myrun --hour 7 --min 30"
    echo "  $0 start --run myrun_21 --hour 19 --min 30  # 7:30 PM"
    echo "  $0 stop"
    echo "  $0 status"
    exit 1
    ;;
esac
