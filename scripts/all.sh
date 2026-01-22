#!/usr/bin/env bash
# Single-word wrapper: all
# Run the default full workflow (optimize -> backtest -> walk-forward) using existing orchestration if available.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Prefer existing orchestrator
if [ -x "$SCRIPT_DIR/run_all.sh" ]; then
  exec "$SCRIPT_DIR/run_all.sh"
fi

# Fallback: run a minimal sequence
./scripts/backtest.sh
./scripts/walkforward
