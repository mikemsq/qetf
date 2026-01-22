#!/bin/bash
# Run walk-forward validation on best strategy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default parameters
SNAPSHOT="data/snapshots/snapshot_20260122_010523"
TRAIN_YEARS=5
TEST_YEARS=1
STEP_MONTHS=12
TOP_N=5
LOOKBACK=252
COST_BPS=10
REBALANCE="monthly"
CAPITAL=100000
PLOTS=false
VERBOSE=false

# Help text
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run walk-forward validation to test strategy robustness on out-of-sample data.

OPTIONS:
    -s, --snapshot PATH     Path to snapshot (default: $SNAPSHOT)
    --train-years N         Training window size (default: $TRAIN_YEARS)
    --test-years N          Testing window size (default: $TEST_YEARS)
    --step-months N         Step size between windows (default: $STEP_MONTHS)
    --top-n N              Number of holdings (default: $TOP_N)
    --lookback N           Momentum lookback days (default: $LOOKBACK)
    --cost-bps N           Transaction cost in bps (default: $COST_BPS)
    --rebalance FREQ       Rebalance frequency: weekly/monthly/quarterly (default: $REBALANCE)
    --capital N            Initial capital (default: $CAPITAL)
    -p, --plots            Save visualization plots
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message

EXAMPLES:
    # Run with defaults
    $(basename "$0")

    # Save plots to output
    $(basename "$0") --plots

    # Custom parameters
    $(basename "$0") --train-years 3 --test-years 1 --step-months 6

    # Different strategy parameters
    $(basename "$0") --top-n 7 --lookback 126 --cost-bps 5

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--snapshot)
            SNAPSHOT="$2"
            shift 2
            ;;
        --train-years)
            TRAIN_YEARS="$2"
            shift 2
            ;;
        --test-years)
            TEST_YEARS="$2"
            shift 2
            ;;
        --step-months)
            STEP_MONTHS="$2"
            shift 2
            ;;
        --top-n)
            TOP_N="$2"
            shift 2
            ;;
        --lookback)
            LOOKBACK="$2"
            shift 2
            ;;
        --cost-bps)
            COST_BPS="$2"
            shift 2
            ;;
        --rebalance)
            REBALANCE="$2"
            shift 2
            ;;
        --capital)
            CAPITAL="$2"
            shift 2
            ;;
        -p|--plots)
            PLOTS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate snapshot
if [ ! -d "$SNAPSHOT" ]; then
    echo -e "${RED}Error: Snapshot not found: $SNAPSHOT${NC}"
    exit 1
fi

# Build command
CMD="python scripts/walk_forward_test.py"
CMD="$CMD --snapshot $SNAPSHOT"
CMD="$CMD --start 2016-01-15"
CMD="$CMD --end 2026-01-15"
CMD="$CMD --train-years $TRAIN_YEARS"
CMD="$CMD --test-years $TEST_YEARS"
CMD="$CMD --step-months $STEP_MONTHS"
CMD="$CMD --top-n $TOP_N"
CMD="$CMD --lookback $LOOKBACK"
CMD="$CMD --cost-bps $COST_BPS"
CMD="$CMD --rebalance-frequency $REBALANCE"
CMD="$CMD --capital $CAPITAL"
CMD="$CMD --output artifacts/walk_forward/$(date +%Y%m%d_%H%M%S)"
[ "$PLOTS" = true ] && CMD="$CMD --save-plots"
[ "$VERBOSE" = true ] && CMD="$CMD --verbose"

# Print header
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}WALK-FORWARD VALIDATION${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Snapshot:${NC}          $SNAPSHOT"
echo -e "${GREEN}Training window:${NC}    $TRAIN_YEARS years"
echo -e "${GREEN}Testing window:${NC}     $TEST_YEARS year(s)"
echo -e "${GREEN}Step size:${NC}          $STEP_MONTHS months"
echo -e "${GREEN}Strategy params:${NC}    top_n=$TOP_N, lookback=$LOOKBACK, cost=$COST_BPS bps"
[ "$PLOTS" = true ] && echo -e "${GREEN}Visualizations:${NC}     ENABLED"
echo ""
echo -e "${YELLOW}Command:${NC} $CMD"
echo ""
echo -e "${BLUE}Starting walk-forward validation...${NC}"
echo ""

# Run validation
$CMD

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Walk-forward validation complete${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Results saved to: artifacts/walk_forward/"
echo "Check summary.json for detailed metrics"
echo ""
