#!/bin/bash
# Regime-aware rebalance wrapper
cd "$(dirname "$0")/.."
python scripts/run_regime_rebalance.py "$@"
