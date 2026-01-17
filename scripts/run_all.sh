#!/bin/bash
# Orchestrate complete workflow: optimize, backtest, validate, monitor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default parameters
STEPS="optimize,backtest,walk-forward"
PARALLEL=4
DRY_RUN=false
SKIP_VALIDATION=false

# Help text
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Orchestrate complete workflow: strategy optimization, backtesting, and validation.

OPTIONS:
    --steps STEPS              Comma-separated steps to run (default: $STEPS)
                               Available: optimize, backtest, walk-forward, monitor
    --parallel N               Number of parallel jobs (default: $PARALLEL)
    --skip-validation          Skip walk-forward validation step
    --dry-run                  Show commands without executing
    -h, --help                 Show this help message

WORKFLOW STEPS:
    1. optimize          Find optimal strategy parameters via grid search
    2. backtest          Run backtest on best strategy from optimization
    3. walk-forward      Validate strategy robustness on out-of-sample data
    4. monitor           Check real-time performance vs benchmarks

EXAMPLES:
    # Complete workflow
    $(basename "$0")

    # Only optimization and backtest
    $(basename "$0") --steps optimize,backtest

    # Just validation of existing best strategy
    $(basename "$0") --steps walk-forward

    # Optimize with 8 parallel jobs, then validate
    $(basename "$0") --parallel 8

    # Preview commands without running
    $(basename "$0") --dry-run

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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

# Print banner
print_banner() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                          QUANTETF COMPLETE WORKFLOW                            ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Print step header
print_step() {
    local step_num=$1
    local step_name=$2
    local description=$3
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Step $step_num: $step_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "$description"
    echo ""
}

# Run step with error handling
run_step() {
    local step_name=$1
    local cmd=$2
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC} $cmd"
        return 0
    else
        echo -e "${YELLOW}Running:${NC} $cmd"
        echo ""
        if eval "$cmd"; then
            echo -e "${GREEN}✓ $step_name completed successfully${NC}"
            return 0
        else
            echo -e "${RED}✗ $step_name failed${NC}"
            return 1
        fi
    fi
}

# Main workflow
print_banner

echo -e "${GREEN}Configuration:${NC}"
echo "  Steps:      $STEPS"
echo "  Parallel:   $PARALLEL"
echo "  Dry-run:    $DRY_RUN"
echo ""

# Track start time
START_TIME=$(date +%s)

# Split steps by comma
IFS=',' read -ra STEP_ARRAY <<< "$STEPS"
STEP_COUNT=0

for step in "${STEP_ARRAY[@]}"; do
    step=$(echo "$step" | xargs)  # Trim whitespace
    STEP_COUNT=$((STEP_COUNT + 1))
    TOTAL_STEPS=${#STEP_ARRAY[@]}
    
    case "$step" in
        optimize)
            print_step "$STEP_COUNT/$TOTAL_STEPS" "STRATEGY OPTIMIZATION" \
                "Finding optimal strategy parameters across configuration space..."
            run_step "Optimization" \
                "bash scripts/run_optimization.sh --parallel $PARALLEL --verbose" || exit 1
            echo ""
            ;;
        backtest)
            print_step "$STEP_COUNT/$TOTAL_STEPS" "BACKTEST ANALYSIS" \
                "Running full backtest with cycle metrics and performance analysis..."
            run_step "Backtest" \
                "bash scripts/run_backtest.sh --analysis --verbose" || exit 1
            echo ""
            ;;
        walk-forward)
            if [ "$SKIP_VALIDATION" = false ]; then
                print_step "$STEP_COUNT/$TOTAL_STEPS" "WALK-FORWARD VALIDATION" \
                    "Testing strategy robustness on out-of-sample data..."
                run_step "Walk-Forward" \
                    "bash scripts/run_walk_forward.sh --plots --verbose" || exit 1
                echo ""
            else
                echo -e "${YELLOW}⊘ Skipping walk-forward validation${NC}"
                echo ""
            fi
            ;;
        monitor)
            print_step "$STEP_COUNT/$TOTAL_STEPS" "PERFORMANCE MONITORING" \
                "Checking real-time strategy performance..."
            run_step "Monitoring" \
                "bash scripts/run_monitoring.sh --monitor-days 30 --verbose" || exit 1
            echo ""
            ;;
        *)
            echo -e "${RED}Unknown step: $step${NC}"
            exit 1
            ;;
    esac
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# Print summary
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                          WORKFLOW COMPLETE                                    ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ All steps completed successfully${NC}"
echo ""
echo "Summary:"
echo "  Steps executed:   $STEP_COUNT of $TOTAL_STEPS"
echo "  Total time:       ${MINUTES}m ${SECONDS}s"
echo "  Artifacts:        artifacts/optimization/, artifacts/backtests/, artifacts/walk_forward/"
echo ""
echo "Next steps:"
echo "  1. Review results in artifacts/ directory"
echo "  2. Check walk_forward results for validation metrics"
echo "  3. Deploy best strategy to production"
echo ""
