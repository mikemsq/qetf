#!/bin/bash
# Daily regime monitor wrapper
cd "$(dirname "$0")/.."
python scripts/run_daily_monitor.py "$@"
