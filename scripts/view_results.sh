#!/bin/bash
# View detailed results from completed analyses

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

View detailed results from analysis outputs.

OPTIONS:
    --optimization      View optimization results
    --backtest ID       View specific backtest (ID or 'latest')
    --walk-forward ID   View walk-forward results (ID or 'latest')
    --list-all          List all available results
    --metrics-only      Show only key metrics (no detailed tables)
    -h, --help          Show this help message

EXAMPLES:
    # View latest optimization
    $(basename "$0") --optimization

    # View latest backtest
    $(basename "$0") --backtest latest

    # View latest walk-forward validation
    $(basename "$0") --walk-forward latest

    # List all available results
    $(basename "$0") --list-all

    # View specific backtest
    $(basename "$0") --backtest 20260117_182400

EOF
}

# Helper to display section header
section_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Parse arguments
SHOW_OPTIMIZATION=false
SHOW_BACKTEST=false
SHOW_WALK_FORWARD=false
LIST_ALL=false
METRICS_ONLY=false
BACKTEST_ID=""
WALK_FORWARD_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --optimization)
            SHOW_OPTIMIZATION=true
            shift
            ;;
        --backtest)
            SHOW_BACKTEST=true
            BACKTEST_ID="${2:-latest}"
            shift 2
            ;;
        --walk-forward)
            SHOW_WALK_FORWARD=true
            WALK_FORWARD_ID="${2:-latest}"
            shift 2
            ;;
        --list-all)
            LIST_ALL=true
            shift
            ;;
        --metrics-only)
            METRICS_ONLY=true
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

# List all results
list_all_results() {
    section_header "ALL AVAILABLE RESULTS"
    
    echo -e "${GREEN}Optimization Results:${NC}"
    ls -td artifacts/optimization/*/ 2>/dev/null | head -5 | while read dir; do
        timestamp=$(basename "$dir")
        echo "  $timestamp"
    done
    
    echo -e "${GREEN}Backtest Results:${NC}"
    ls -td artifacts/backtests/*/ 2>/dev/null | head -5 | while read dir; do
        timestamp=$(basename "$dir")
        echo "  $timestamp"
    done
    
    echo -e "${GREEN}Walk-Forward Results:${NC}"
    ls -td artifacts/walk_forward/*/ 2>/dev/null | head -5 | while read dir; do
        timestamp=$(basename "$dir")
        echo "  $timestamp"
    done
}

# View optimization results
view_optimization() {
    section_header "OPTIMIZATION RESULTS"
    
    local latest_dir=$(ls -td artifacts/optimization/*/ 2>/dev/null | head -1)
    
    if [ -z "$latest_dir" ]; then
        echo -e "${RED}No optimization results found${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Results directory: $latest_dir${NC}"
    echo ""
    
    # List all files
    echo -e "${GREEN}Files in directory:${NC}"
    ls -lh "$latest_dir" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}' 
    
    echo ""
    
    # Show best strategy
    if [ -f "$latest_dir/best_strategy.yaml" ]; then
        echo -e "${GREEN}Best Strategy Configuration:${NC}"
        cat "$latest_dir/best_strategy.yaml" | sed 's/^/  /'
        echo ""
    fi
    
    # Show summary if not metrics-only
    if [ "$METRICS_ONLY" = false ] && [ -f "$latest_dir/results_summary.json" ]; then
        echo -e "${GREEN}Complete Summary:${NC}"
        python3 -m json.tool < "$latest_dir/results_summary.json" 2>/dev/null | sed 's/^/  /' || \
            cat "$latest_dir/results_summary.json" | sed 's/^/  /'
    else
        # Show metrics only
        if [ -f "$latest_dir/results_summary.json" ]; then
            echo -e "${GREEN}Summary Statistics:${NC}"
            python3 << PYEOF 2>/dev/null || echo "  (Results file found)"
import json
try:
    with open("$latest_dir/results_summary.json") as f:
        data = json.load(f)
        for key, value in list(data.items())[:15]:  # Show first 15 items
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
except: pass
PYEOF
        fi
    fi
}

# View backtest results
view_backtest() {
    local id=$1
    
    # Get directory
    local latest_dir=""
    if [ "$id" = "latest" ]; then
        latest_dir=$(ls -td artifacts/backtests/*/ 2>/dev/null | head -1)
    else
        latest_dir="artifacts/backtests/$id/"
    fi
    
    if [ -z "$latest_dir" ] || [ ! -d "$latest_dir" ]; then
        echo -e "${RED}Backtest results not found: $id${NC}"
        return 1
    fi
    
    section_header "BACKTEST RESULTS - $(basename $latest_dir)"
    
    echo -e "${YELLOW}Results directory: $latest_dir${NC}"
    echo ""
    
    # List files
    echo -e "${GREEN}Files in directory:${NC}"
    ls -lh "$latest_dir" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}' 
    
    echo ""
    
    # Show performance analysis
    if [ -f "$latest_dir/performance_analysis.json" ]; then
        echo -e "${GREEN}Performance Analysis:${NC}"
        if [ "$METRICS_ONLY" = true ]; then
            python3 << PYEOF 2>/dev/null || echo "  (Results file found)"
import json
with open("$latest_dir/performance_analysis.json") as f:
    data = json.load(f)
    for key, value in list(data.items())[:20]:
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
PYEOF
        else
            python3 -m json.tool < "$latest_dir/performance_analysis.json" 2>/dev/null | sed 's/^/  /'
        fi
    fi
    
    # Show cycle metrics sample
    echo ""
    if [ -f "$latest_dir/cycle_metrics.json" ]; then
        echo -e "${GREEN}Cycle Metrics Sample (first 5 periods):${NC}"
        python3 << PYEOF 2>/dev/null || echo "  (Results file found)"
import json
with open("$latest_dir/cycle_metrics.json") as f:
    data = json.load(f)
    if 'monthly' in data:
        for i, month in enumerate(list(data['monthly'].items())[:5]):
            print(f"  {month[0]}: return={month[1].get('return', 0):.4f}, sharpe={month[1].get('sharpe', 0):.4f}")
PYEOF
    fi
}

# View walk-forward results
view_walk_forward() {
    local id=$1
    
    # Get directory
    local latest_dir=""
    if [ "$id" = "latest" ]; then
        latest_dir=$(ls -td artifacts/walk_forward/*/ 2>/dev/null | head -1)
    else
        latest_dir="artifacts/walk_forward/$id/"
    fi
    
    if [ -z "$latest_dir" ] || [ ! -d "$latest_dir" ]; then
        echo -e "${RED}Walk-forward results not found: $id${NC}"
        return 1
    fi
    
    section_header "WALK-FORWARD VALIDATION - $(basename $latest_dir)"
    
    echo -e "${YELLOW}Results directory: $latest_dir${NC}"
    echo ""
    
    # List files
    echo -e "${GREEN}Files in directory:${NC}"
    ls -lh "$latest_dir" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}' 
    
    echo ""
    
    # Show summary
    if [ -f "$latest_dir/summary.json" ]; then
        echo -e "${GREEN}Validation Summary:${NC}"
        if [ "$METRICS_ONLY" = true ]; then
            python3 << PYEOF 2>/dev/null || echo "  (Results file found)"
import json
with open("$latest_dir/summary.json") as f:
    data = json.load(f)
    metrics = ['num_windows', 'oos_sharpe_mean', 'oos_sharpe_std', 'oos_return_mean', 
               'pct_windows_oos_positive', 'pct_windows_oos_beats_is', 'sharpe_degradation']
    for metric in metrics:
        if metric in data:
            val = data[metric]
            if isinstance(val, float):
                print(f"  {metric}: {val:.6f}")
            else:
                print(f"  {metric}: {val}")
PYEOF
        else
            python3 -m json.tool < "$latest_dir/summary.json" 2>/dev/null | sed 's/^/  /'
        fi
    fi
    
    # Show window results table
    echo ""
    if [ -f "$latest_dir/window_results.csv" ]; then
        echo -e "${GREEN}Window-by-Window Performance:${NC}"
        column -t -s',' "$latest_dir/window_results.csv" | sed 's/^/  /'
    fi
    
    # Show plot info
    echo ""
    if [ -f "$latest_dir/walk_forward_analysis.png" ]; then
        echo -e "${GREEN}Visualization:${NC}"
        echo "  walk_forward_analysis.png ($(ls -lh "$latest_dir/walk_forward_analysis.png" | awk '{print $5}'))"
        echo "  Run: open $latest_dir/walk_forward_analysis.png"
    fi
}

# Main
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                          VIEW RESULTS                                         ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"

if [ "$LIST_ALL" = true ]; then
    list_all_results
elif [ "$SHOW_OPTIMIZATION" = true ]; then
    view_optimization
elif [ "$SHOW_BACKTEST" = true ]; then
    view_backtest "$BACKTEST_ID"
elif [ "$SHOW_WALK_FORWARD" = true ]; then
    view_walk_forward "$WALK_FORWARD_ID"
else
    echo -e "${YELLOW}No options specified. Showing latest results...${NC}"
    view_walk_forward "latest"
fi

echo ""
