#!/usr/bin/env bash
# Strategy optimization pipeline
#
# Runs the full optimization in three steps:
#   1. run_backtests.py   - Run all backtests, save results
#   2. rank_strategies.py - Rank strategies by scoring method
#   3. analyze_regimes.py - Run regime analysis
#
# Usage:
#   ./scripts/optimize.sh                    # Use defaults
#   ./scripts/optimize.sh --max-configs 20   # Quick test
#   ./scripts/optimize.sh --help             # Show options
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Defaults
PYTHON="${VENV_PYTHON:-python3}"
SNAPSHOT="data/snapshots/snapshot_latest"
OUTPUT="artifacts/optimization"
PERIODS="1"
PARALLEL="8"
SCORING_METHOD="regime_weighted"
MAX_CONFIGS=""
VERBOSE=""
DRY_RUN=""
SKIP_REGIME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --snapshot)
            SNAPSHOT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --periods)
            PERIODS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --scoring-method)
            SCORING_METHOD="$2"
            shift 2
            ;;
        --max-configs)
            MAX_CONFIGS="--max-configs $2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --skip-regime)
            SKIP_REGIME="1"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --snapshot PATH      Data snapshot path (default: data/snapshots/snapshot_latest)"
            echo "  --output PATH        Output directory (default: artifacts/optimization)"
            echo "  --periods LIST       Evaluation periods in years (default: 1)"
            echo "  --parallel N         Number of parallel workers (default: 8)"
            echo "  --scoring-method M   Scoring method: multi_period, trailing_1y, regime_weighted (default: regime_weighted)"
            echo "  --max-configs N      Limit configs for testing"
            echo "  --skip-regime        Skip regime analysis step"
            echo "  --verbose, -v        Enable verbose logging"
            echo "  --dry-run            Just count configs, don't run"
            echo "  --help, -h           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

START_TIME=$(date +%s)
START_FMT=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo "STRATEGY OPTIMIZATION PIPELINE"
echo "============================================================"
echo "Snapshot:       $SNAPSHOT"
echo "Output:         $OUTPUT"
echo "Periods:        $PERIODS years"
echo "Parallel:       $PARALLEL workers"
echo "Scoring:        $SCORING_METHOD"
echo "Started:        $START_FMT"
echo "============================================================"
echo ""

# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT/$TIMESTAMP"
mkdir -p "$RUN_DIR"

# Step 1: Run backtests
echo ">>> Step 1/3: Running backtests..."
"$PYTHON" scripts/run_backtests.py \
    --snapshot "$SNAPSHOT" \
    --output "$RUN_DIR" \
    --periods "$PERIODS" \
    --parallel "$PARALLEL" \
    $MAX_CONFIGS \
    $VERBOSE \
    $DRY_RUN

if [[ -n "$DRY_RUN" ]]; then
    echo ""
    echo "Dry run complete."
    exit 0
fi

# Find the results file (run_backtests creates a subdirectory)
RESULTS_FILE=$(find "$RUN_DIR" -name "backtest_results.pkl" -type f | head -1)
if [[ -z "$RESULTS_FILE" ]]; then
    echo "ERROR: backtest_results.pkl not found in $RUN_DIR"
    exit 1
fi
echo ""
echo "Backtest results: $RESULTS_FILE"
echo ""

# Step 2: Rank strategies
echo ">>> Step 2/3: Ranking strategies..."
"$PYTHON" scripts/rank_strategies.py \
    --results "$RESULTS_FILE" \
    --output "$RUN_DIR" \
    --scoring-method "$SCORING_METHOD" \
    $VERBOSE

echo ""

# Step 3: Regime analysis (optional)
if [[ -z "$SKIP_REGIME" ]]; then
    echo ">>> Step 3/3: Running regime analysis..."
    "$PYTHON" scripts/analyze_regimes.py \
        --results "$RESULTS_FILE" \
        --snapshot "$SNAPSHOT" \
        --output "$RUN_DIR" \
        $VERBOSE
else
    echo ">>> Step 3/3: Skipped (--skip-regime)"
fi

# Summary
END_TIME=$(date +%s)
END_FMT=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Started:  $START_FMT"
echo "Ended:    $END_FMT"
printf "Duration: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
echo ""
echo "Results:  $RUN_DIR"
echo "============================================================"
