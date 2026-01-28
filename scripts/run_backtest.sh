#!/bin/bash
# Run backtest with cycle metrics and analysis

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
SNAPSHOT="data/snapshots/snapshot_latest"
UNIVERSE="US_CORE_ETFS"
STRATEGY="trend_filtered_momentum"
TOP_N=5
LOOKBACK=252
COST_BPS=10
REBALANCE="monthly"
START_DATE="2016-01-15"
END_DATE="2026-01-15"
CAPITAL=100000
ANALYSIS=false
VERBOSE=false

# Help text
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run full backtest with cycle metrics, drawdown analysis, and performance reports.

OPTIONS:
    -s, --snapshot PATH       Path to snapshot (default: $SNAPSHOT)
    -u, --universe NAME       Universe config name (default: $UNIVERSE)
    --strategy NAME           Strategy name: trend_filtered_momentum, ensemble, etc
    --top-n N                Number of holdings (default: $TOP_N)
    --lookback N             Momentum lookback days (default: $LOOKBACK)
    --cost-bps N             Transaction cost in bps (default: $COST_BPS)
    --rebalance FREQ         Rebalance frequency: weekly/monthly/quarterly (default: $REBALANCE)
    --start DATE             Start date YYYY-MM-DD (default: $START_DATE)
    --end DATE               End date YYYY-MM-DD (default: $END_DATE)
    --capital N              Initial capital (default: $CAPITAL)
    -a, --analysis           Run enhanced analysis report
    -v, --verbose            Enable verbose output
    -h, --help               Show this help message

EXAMPLES:
    # Run standard backtest with defaults
    $(basename "$0")

    # Run with analysis report
    $(basename "$0") --analysis

    # Custom date range
    $(basename "$0") --start 2020-01-01 --end 2024-12-31

    # Different strategy parameters
    $(basename "$0") --top-n 7 --lookback 126 --cost-bps 5

    # Recent period only
    $(basename "$0") --start 2023-01-01 --end 2026-01-15 --analysis

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--snapshot)
            SNAPSHOT="$2"
            shift 2
            ;;
        -u|--universe)
            UNIVERSE="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
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
        --start)
            START_DATE="$2"
            shift 2
            ;;
        --end)
            END_DATE="$2"
            shift 2
            ;;
        --capital)
            CAPITAL="$2"
            shift 2
            ;;
        -a|--analysis)
            ANALYSIS=true
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

# Generate output directory
OUTPUT_DIR="artifacts/backtests/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python scripts/run_backtest.py"
CMD="$CMD --snapshot $SNAPSHOT"
CMD="$CMD --universe $UNIVERSE"
CMD="$CMD --strategy $STRATEGY"
CMD="$CMD --top-n $TOP_N"
CMD="$CMD --lookback $LOOKBACK"
CMD="$CMD --transaction-cost-bps $COST_BPS"
CMD="$CMD --rebalance-frequency $REBALANCE"
CMD="$CMD --start-date $START_DATE"
CMD="$CMD --end-date $END_DATE"
CMD="$CMD --initial-capital $CAPITAL"
CMD="$CMD --output $OUTPUT_DIR"
[ "$ANALYSIS" = true ] && CMD="$CMD --with-analysis"
[ "$VERBOSE" = true ] && CMD="$CMD --verbose"

# Print header
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}BACKTEST ENGINE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Snapshot:${NC}          $SNAPSHOT"
echo -e "${GREEN}Universe:${NC}          $UNIVERSE"
echo -e "${GREEN}Strategy:${NC}          $STRATEGY"
echo -e "${GREEN}Period:${NC}             $START_DATE to $END_DATE"
echo -e "${GREEN}Capital:${NC}            \$$CAPITAL"
echo -e "${GREEN}Strategy params:${NC}    top_n=$TOP_N, lookback=$LOOKBACK, cost=$COST_BPS bps"
[ "$ANALYSIS" = true ] && echo -e "${GREEN}Analysis mode:${NC}       ENABLED"
echo ""
echo -e "${BLUE}Output directory:${NC}   $OUTPUT_DIR"
echo ""
echo -e "${YELLOW}Command:${NC} $CMD"
echo ""
echo -e "${BLUE}Starting backtest...${NC}"
echo ""

# Run backtest
$CMD

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Backtest complete${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Key files:"
echo "  - cycle_metrics.json          (Detailed monthly/daily metrics)"
echo "  - backtest_results.csv        (Price and return history)"
echo "  - performance_analysis.json   (Sharpe, sortino, drawdown, etc)"
echo ""
[ "$ANALYSIS" = true ] && echo "Enhanced analysis files generated. See output directory for details."
echo ""
