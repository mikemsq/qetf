#!/usr/bin/env bash
# Single-word wrapper: optimize
# Run the optimizer with a sane default snapshot (no args).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${VENV_PYTHON:-python3}"
SNAP="data/snapshots/snapshot_latest"

START_TIME=$(date +%s)
START_FMT=$(date '+%Y-%m-%d %H:%M:%S')

"$PYTHON" scripts/find_best_strategy.py \
  --snapshot "$SNAP" \
  --periods 1 \
  --parallel 8 \
  --scoring-method regime_weighted

END_TIME=$(date +%s)
END_FMT=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo ""
echo "Started:  $START_FMT"
echo "Ended:    $END_FMT"
printf "Duration: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS