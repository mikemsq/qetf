#!/usr/bin/env bash
# Single-word wrapper: backtest
# Runs the standard backtest with sensible defaults (no args required).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${VENV_PYTHON:-python3}"

# If an explicit existing wrapper is present, use it (keeps parity with repo logic)
if [ -x "$SCRIPT_DIR/run_backtest.sh" ]; then
  exec "$SCRIPT_DIR/run_backtest.sh"
fi

# Fallback: call Python script with default args
exec "$PYTHON" scripts/run_backtest.py \
  --snapshot "data/snapshots/snapshot_latest" \
  --start "2016-01-01" \
  --end "2026-01-15" \
  --strategy "momentum-ew-top5" \
  --capital 100000 \
  --top-n 5 \
  --lookback 252
