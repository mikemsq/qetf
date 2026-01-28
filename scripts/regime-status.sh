#!/bin/bash
# Show regime status wrapper
cd "$(dirname "$0")/.."
python scripts/show_regime_status.py "$@"
