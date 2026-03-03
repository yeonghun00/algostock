#!/bin/bash
# auto_live.sh — runs daily via cron or launchd.
# On non-rebalance days: exits early (safe to run every day).
# On rebalance day: updates DB, computes picks, submits orders automatically.

set -uo pipefail

# ── Config (override with environment variables) ──────────────────────────────
#
#   ALGOSTOCK_RUN        Which run to use. If empty, auto-picks the latest run
#                        in runs/ by most recently modified results.csv.
#   ALGOSTOCK_PORTFOLIO  Total KRW to deploy (default: 100,000,000 = 1억)
#   ALGOSTOCK_PYTHON     Python executable (default: python3)
#
PORTFOLIO="${ALGOSTOCK_PORTFOLIO:-100000000}"
PYTHON="${ALGOSTOCK_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LOG_DIR="$SCRIPT_DIR/live/logs"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d).log"

mkdir -p "$LOG_DIR"

# ── Run selection ─────────────────────────────────────────────────────────────
# If ALGOSTOCK_RUN is set, use it. Otherwise find the run with the most
# recently modified results.csv in runs/.

if [ -n "${ALGOSTOCK_RUN:-}" ]; then
  RUN_NAME="$ALGOSTOCK_RUN"
else
  RUN_NAME=$(
    find "$SCRIPT_DIR/runs" -name "results.csv" -print0 2>/dev/null \
    | xargs -0 ls -t 2>/dev/null \
    | head -1 \
    | xargs -I{} dirname {} \
    | xargs basename
  )
  if [ -z "$RUN_NAME" ]; then
    echo "[ERROR] No runs found in runs/. Run a backtest first."
    exit 1
  fi
fi

# Validate: runs/<name>/results.csv and picks.csv must both exist
RUN_DIR="$SCRIPT_DIR/runs/$RUN_NAME"
if [ ! -f "$RUN_DIR/results.csv" ] || [ ! -f "$RUN_DIR/picks.csv" ]; then
  echo "[ERROR] Run '$RUN_NAME' is missing results.csv or picks.csv in $RUN_DIR"
  echo "  Available runs:"
  ls "$SCRIPT_DIR/runs/" 2>/dev/null | sed 's/^/    /'
  exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
{
  echo "========================================"
  echo "  AlgoStock auto_live  $(date)"
  echo "  run=$RUN_NAME  portfolio=$PORTFOLIO"
  echo "========================================"

  cd "$SCRIPT_DIR"

  # Pipe "y" to auto-confirm the order prompt on rebalance day.
  # On non-rebalance days the script exits before the prompt — safe.
  echo "y" | "$PYTHON" scripts/run_live.py \
    --run "$RUN_NAME" \
    --portfolio "$PORTFOLIO" \
    --execute

  echo "Done: $(date)"
} 2>&1 | tee -a "$LOG_FILE"

# ── Summary report ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  REPORT  $(date +%Y-%m-%d)  [run: $RUN_NAME]"
echo "════════════════════════════════════════"

if grep -q "오늘이 실행일" "$LOG_FILE"; then
  echo "  [REBAL]  Rebalance day — orders attempted"
elif grep -q "거래일 후 실행일" "$LOG_FILE"; then
  DAYS=$(grep "거래일 후 실행일" "$LOG_FILE" | grep -oE "[0-9]+ 거래일" | tail -1)
  echo "  [SKIP]   Not rebalance day ($DAYS remaining)"
elif grep -q "내일.*실행일" "$LOG_FILE"; then
  echo "  [SKIP]   Rebalance is tomorrow — picks computed, no orders"
elif grep -q "이미 실행 완료" "$LOG_FILE"; then
  echo "  [SKIP]   Already executed today"
fi

if grep -q "WARNING: update-all" "$LOG_FILE"; then
  echo "  [WARN]   ETL update failed — picks used stale DB data"
else
  echo "  [OK]     ETL update succeeded"
fi

if grep -q "인증 실패" "$LOG_FILE"; then
  echo "  [ERROR]  Kiwoom authentication failed — NO orders placed"
elif grep -q "인증 성공" "$LOG_FILE"; then
  echo "  [OK]     Kiwoom authenticated"
fi

SOLD=$(grep -c "주문완료.*SELL" "$LOG_FILE" 2>/dev/null || echo 0)
BOUGHT=$(grep -c "주문완료.*BUY" "$LOG_FILE" 2>/dev/null || echo 0)
FAILED=$(grep -c "주문실패" "$LOG_FILE" 2>/dev/null || echo 0)

if grep -q "주문완료" "$LOG_FILE"; then
  echo "  [OK]     Orders placed — sell=$SOLD  buy=$BOUGHT"
fi
if [ "$FAILED" -gt 0 ]; then
  echo "  [ERROR]  $FAILED order(s) failed — check log for details"
  grep "주문실패" "$LOG_FILE" | tail -5 | sed 's/^/             /'
fi

if grep -q "✅ 주문 완료" "$LOG_FILE"; then
  echo "  [OK]     All done cleanly"
elif grep -q "ERROR:" "$LOG_FILE"; then
  echo "  [ERROR]  Script encountered errors — see $LOG_FILE"
fi

echo "  [LOG]    $LOG_FILE"
echo "════════════════════════════════════════"

# ── macOS notification ────────────────────────────────────────────────────────
if command -v osascript &>/dev/null; then
  if grep -q "인증 실패\|주문실패\|WARNING: update-all\|ERROR:" "$LOG_FILE"; then
    TITLE="AlgoStock ⚠️ Error"
    MSG="Network or order error on $(date +%Y-%m-%d). Check live/logs."
  elif grep -q "주문완료" "$LOG_FILE"; then
    TITLE="AlgoStock ✅ Rebalanced"
    MSG="Orders placed: sell=$SOLD buy=$BOUGHT on $(date +%Y-%m-%d) [run: $RUN_NAME]"
  else
    TITLE="AlgoStock — No action"
    MSG="Not a rebalance day on $(date +%Y-%m-%d) [run: $RUN_NAME]"
  fi
  osascript -e "display notification \"$MSG\" with title \"$TITLE\""
fi
