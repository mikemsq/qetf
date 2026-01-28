#!/usr/bin/env bash
# Single-word wrapper: ingest
# Ingest ETF data using default universe and lookback (no args).
set -euo pipefail

# Load environment variables from .env if it exists
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${FRED_API_KEY:-}" ]]; then
  echo "Error: FRED_API_KEY environment variable is not set."
  echo "Please set FRED_API_KEY before running this script."
  exit 1
fi
PYTHON="${VENV_PYTHON:-python3}"
UNIVERSE="tier4_broad_200"

"$PYTHON" scripts/ingest_etf_data.py \
  --universe "$UNIVERSE"

# Ingest FRED data
"$PYTHON" scripts/ingest_fred_data.py

# Create a snapshot from the default universe (from snapshot script)
"$PYTHON" scripts/create_snapshot.py \
  --universe "$UNIVERSE" \
  --name snapshot_latest

  