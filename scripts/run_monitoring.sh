#!/bin/bash
# Run daily monitoring and performance tracking

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
STRATEGY_CONFIG="configs/strategies/best_strategy.yaml"
MONITOR_DAYS=30
EMAIL_ALERTS=false
VERBOSE=false
CONTINUOUS=false

# Help text
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Monitor strategy performance in real-time or over recent periods.

OPTIONS:
    -c, --config PATH       Strategy config file (default: $STRATEGY_CONFIG)
    --monitor-days N        Number of recent days to monitor (default: $MONITOR_DAYS)
    --continuous            Run continuous monitoring (every 5 minutes)
    --email-alerts          Send email alerts for drawdowns > 5%
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

EXAMPLES:
    # Monitor last 30 days
    $(basename "$0")

    # Monitor last 90 days with verbose output
    $(basename "$0") --monitor-days 90 --verbose

    # Continuous monitoring with email alerts
    $(basename "$0") --continuous --email-alerts

    # Monitor custom strategy config
    $(basename "$0") --config configs/strategies/custom_strategy.yaml

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            STRATEGY_CONFIG="$2"
            shift 2
            ;;
        --monitor-days)
            MONITOR_DAYS="$2"
            shift 2
            ;;
        --continuous)
            CONTINUOUS=true
            shift
            ;;
        --email-alerts)
            EMAIL_ALERTS=true
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

# Validate strategy config
if [ ! -f "$STRATEGY_CONFIG" ]; then
    echo -e "${RED}Error: Strategy config not found: $STRATEGY_CONFIG${NC}"
    exit 1
fi

# Create output directory
OUTPUT_DIR="artifacts/monitoring/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Print header
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}REAL-TIME PERFORMANCE MONITORING${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Strategy config:${NC}     $STRATEGY_CONFIG"
echo -e "${GREEN}Monitor period:${NC}      Last $MONITOR_DAYS days"
[ "$EMAIL_ALERTS" = true ] && echo -e "${GREEN}Email alerts:${NC}        ENABLED"
echo -e "${GREEN}Output directory:${NC}    $OUTPUT_DIR"
echo ""

# Run monitoring function
run_monitoring() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[$timestamp]${NC} Running monitoring check..."
    
    # Call Python monitoring script
    python scripts/run_daily_monitoring.py \
        --config "$STRATEGY_CONFIG" \
        --monitor-days "$MONITOR_DAYS" \
        --output "$OUTPUT_DIR" \
        $([ "$VERBOSE" = true ] && echo "--verbose" || true) \
        $([ "$EMAIL_ALERTS" = true ] && echo "--email-alerts" || true)
    
    # Display key metrics
    if [ -f "$OUTPUT_DIR/latest_metrics.json" ]; then
        echo ""
        echo -e "${GREEN}Recent metrics:${NC}"
        cat "$OUTPUT_DIR/latest_metrics.json" | python -m json.tool | grep -E '(date|return|sharpe|drawdown|volatility)' || true
    fi
}

# If continuous mode, loop with 5-minute interval
if [ "$CONTINUOUS" = true ]; then
    echo -e "${BLUE}Continuous monitoring enabled (updates every 5 minutes)${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    while true; do
        run_monitoring
        echo ""
        echo -e "${YELLOW}Next check in 5 minutes... ($(date))${NC}"
        sleep 300
    done
else
    # Single run
    run_monitoring
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Monitoring check complete${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
fi
