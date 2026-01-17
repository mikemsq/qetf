#!/bin/bash
# Check status of running processes and recent outputs

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

Check status of running processes and display recent outputs.

OPTIONS:
    --all                 Show all status checks
    --processes           Show running quantetf processes
    --optimization        Show latest optimization results
    --backtest            Show latest backtest results
    --walk-forward        Show latest walk-forward results
    --logs N              Show last N lines of recent logs (default: 20)
    -h, --help            Show this help message

EXAMPLES:
    # Full status check
    $(basename "$0") --all

    # Check running processes
    $(basename "$0") --processes

    # Show latest optimization results
    $(basename "$0") --optimization

    # Show last 50 lines of logs
    $(basename "$0") --logs 50

EOF
}

# Default to show all
SHOW_ALL=true
SHOW_PROCESSES=false
SHOW_OPTIMIZATION=false
SHOW_BACKTEST=false
SHOW_WALK_FORWARD=false
LOG_LINES=20

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            SHOW_ALL=true
            shift
            ;;
        --processes)
            SHOW_ALL=false
            SHOW_PROCESSES=true
            shift
            ;;
        --optimization)
            SHOW_ALL=false
            SHOW_OPTIMIZATION=true
            shift
            ;;
        --backtest)
            SHOW_ALL=false
            SHOW_BACKTEST=true
            shift
            ;;
        --walk-forward)
            SHOW_ALL=false
            SHOW_WALK_FORWARD=true
            shift
            ;;
        --logs)
            LOG_LINES="$2"
            shift 2
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

# Helper to display section header
section_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Check running processes
check_processes() {
    section_header "RUNNING PROCESSES"
    
    local found=0
    
    if pgrep -f "find_best_strategy.py" > /dev/null; then
        echo -e "${GREEN}✓${NC} Optimization running"
        pgrep -f "find_best_strategy.py" | while read pid; do
            echo "  PID: $pid, started: $(ps -p $pid -o lstart= | cut -d' ' -f1-3)"
        done
        found=$((found + 1))
    fi
    
    if pgrep -f "run_backtest.py" > /dev/null; then
        echo -e "${GREEN}✓${NC} Backtest running"
        pgrep -f "run_backtest.py" | while read pid; do
            echo "  PID: $pid, started: $(ps -p $pid -o lstart= | cut -d' ' -f1-3)"
        done
        found=$((found + 1))
    fi
    
    if pgrep -f "walk_forward_test.py" > /dev/null; then
        echo -e "${GREEN}✓${NC} Walk-forward validation running"
        pgrep -f "walk_forward_test.py" | while read pid; do
            echo "  PID: $pid, started: $(ps -p $pid -o lstart= | cut -d' ' -f1-3)"
        done
        found=$((found + 1))
    fi
    
    if [ $found -eq 0 ]; then
        echo -e "${YELLOW}⊘${NC} No quantetf processes running"
    fi
}

# Show latest optimization results
show_optimization() {
    section_header "LATEST OPTIMIZATION RESULTS"
    
    local latest_dir=$(ls -td artifacts/optimization/*/ 2>/dev/null | head -1)
    
    if [ -z "$latest_dir" ]; then
        echo -e "${YELLOW}No optimization results found${NC}"
        return
    fi
    
    echo "Directory: $latest_dir"
    echo ""
    
    if [ -f "$latest_dir/best_strategy.yaml" ]; then
        echo -e "${GREEN}Best Strategy:${NC}"
        head -20 "$latest_dir/best_strategy.yaml" | sed 's/^/  /'
    fi
    
    if [ -f "$latest_dir/results_summary.json" ]; then
        echo ""
        echo -e "${GREEN}Summary Statistics:${NC}"
        python3 << PYEOF 2>/dev/null || echo "  (Results file found)"
import json
with open("$latest_dir/results_summary.json") as f:
    data = json.load(f)
    for key in ['best_sharpe', 'best_return', 'mean_sharpe', 'mean_return', 'total_configs']:
        if key in data:
            print(f"  {key}: {data[key]}")
PYEOF
    fi
}

# Show latest backtest results
show_backtest() {
    section_header "LATEST BACKTEST RESULTS"
    
    local latest_dir=$(ls -td artifacts/backtests/*/ 2>/dev/null | head -1)
    
    if [ -z "$latest_dir" ]; then
        echo -e "${YELLOW}No backtest results found${NC}"
        return
    fi
    
    echo "Directory: $latest_dir"
    echo ""
    
    if [ -f "$latest_dir/performance_analysis.json" ]; then
        echo -e "${GREEN}Performance Metrics:${NC}"
        python3 << PYEOF 2>/dev/null || echo "  (Results file found)"
import json
with open("$latest_dir/performance_analysis.json") as f:
    data = json.load(f)
    metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'sortino_ratio']
    for metric in metrics:
        if metric in data:
            print(f"  {metric}: {data[metric]}")
PYEOF
    fi
}

# Show latest walk-forward results
show_walk_forward() {
    section_header "LATEST WALK-FORWARD VALIDATION"
    
    local latest_dir=$(ls -td artifacts/walk_forward/*/ 2>/dev/null | head -1)
    
    if [ -z "$latest_dir" ]; then
        echo -e "${YELLOW}No walk-forward results found${NC}"
        return
    fi
    
    echo "Directory: $latest_dir"
    echo ""
    
    if [ -f "$latest_dir/summary.json" ]; then
        echo -e "${GREEN}Validation Summary:${NC}"
        python3 << PYEOF 2>/dev/null || echo "  (Results file found)"
import json
with open("$latest_dir/summary.json") as f:
    data = json.load(f)
    metrics = ['num_windows', 'oos_sharpe_mean', 'oos_return_mean', 'pct_windows_oos_positive', 'pct_windows_oos_beats_is', 'sharpe_degradation']
    for metric in metrics:
        if metric in data:
            val = data[metric]
            if isinstance(val, float):
                print(f"  {metric}: {val:.4f}")
            else:
                print(f"  {metric}: {val}")
PYEOF
    fi
    
    echo ""
    echo -e "${GREEN}Window Performance:${NC}"
    if [ -f "$latest_dir/window_results.csv" ]; then
        tail -6 "$latest_dir/window_results.csv" | column -t -s',' | sed 's/^/  /'
    fi
}

# Show recent logs
show_logs() {
    section_header "RECENT LOG FILES"
    
    echo "Most recent log files:"
    ls -lt *.log 2>/dev/null | head -5 | while read line; do
        echo "  $line"
    done
    
    if [ -f "walk_forward_"*.log ]; then
        echo ""
        echo -e "${GREEN}Latest walk-forward log (last $LOG_LINES lines):${NC}"
        tail -n "$LOG_LINES" "$(ls -t walk_forward_*.log 2>/dev/null | head -1)" | sed 's/^/  /'
    fi
}

# Main
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                            STATUS CHECK                                       ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"

if [ "$SHOW_ALL" = true ]; then
    check_processes
    show_optimization
    show_backtest
    show_walk_forward
    show_logs
else
    [ "$SHOW_PROCESSES" = true ] && check_processes
    [ "$SHOW_OPTIMIZATION" = true ] && show_optimization
    [ "$SHOW_BACKTEST" = true ] && show_backtest
    [ "$SHOW_WALK_FORWARD" = true ] && show_walk_forward
fi

echo ""
