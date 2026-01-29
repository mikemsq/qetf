#!/usr/bin/env bash
# Single-word wrapper: optimize
# Run the optimizer with a sane default snapshot (no args).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${VENV_PYTHON:-python3}"
SNAP="data/snapshots/snapshot_latest"

"$PYTHON" scripts/find_best_strategy.py \
  --snapshot "$SNAP" \
  --periods 1 \
  --parallel 8