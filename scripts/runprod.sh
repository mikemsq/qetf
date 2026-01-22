#!/usr/bin/env bash
# Single-word wrapper: runprod
# Run production pipeline in dry-run mode using a default production config (no args).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${VENV_PYTHON:-python3}"
CONFIG="configs/strategies/production_value_momentum.yaml"
SNAP="data/snapshots/snapshot_20260122_010523"

exec "$PYTHON" scripts/run_production_pipeline.py \
  --config "$CONFIG" \
  --snapshot "$SNAP" \
  --dry-run
