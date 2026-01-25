#!/usr/bin/env bash
# Single-word wrapper: ingest
# Ingest ETF data using default universe and lookback (no args).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export FRED_API_KEY="123"
PYTHON="${VENV_PYTHON:-python3}"
UNIVERSE="tier4_broad_200"

exec "$PYTHON" scripts/ingest_etf_data.py \
  --universe "$UNIVERSE" \
  #--lookback-years 5

# Ingest FRED data
exec "$PYTHON" scripts/ingest_fred_data.py

# Create a snapshot from the default universe (from snapshot script)
exec "$PYTHON" scripts/create_snapshot.py \
  --universe "$UNIVERSE"
