#!/bin/bash
# Script to run production pipeline with default parameters
# Example usage: ./run_production.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

cd "$PROJECT_ROOT"

python3 scripts/run_production_pipeline.py \
    --config configs/strategies/production_value_momentum.yaml \
    --snapshot data/snapshots/snapshot_20260122_010523 \
    --dry-run 
