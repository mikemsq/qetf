#!/bin/bash
# Run strategy optimization across configuration space

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
SNAPSHOT="data/snapshots/snapshot_20260113_232157"
PARALLEL=4
DRY_RUN=false
MAX_CONFIGS=""
VERBOSE=""

# Help text
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run strategy optimization to find winning configurations.

OPTIONS:
    -s, --snapshot PATH     Path to snapshot (default: $SNAPSHOT)
    -p, --parallel N        Number of parallel workers (default: $PARALLEL)
    -d, --dry-run          Count configurations without running
    -m, --max-configs N    Limit number of configurations to test
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message

EXAMPLES:
    # Run full optimization with 4 workers
    $(basename "$0")

    # Dry-run to count configurations
    $(basename "$0") --dry-run

    # Run with 8 parallel workers
    $(basename "$0") --parallel 8

    # Test only first 50 configurations
    $(basename "$0") --max-configs 50

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--snapshot)
            SNAPSHOT="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -m|--max-configs)
            MAX_CONFIGS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
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
CMD="python scripts/find_best_strategy.py"
CMD="$CMD --snapshot $SNAPSHOT"
CMD="$CMD --parallel $PARALLEL"
[ "$DRY_RUN" = true ] && CMD="$CMD --dry-run"
[ -n "$MAX_CONFIGS" ] && CMD="$CMD --max-configs $MAX_CONFIGS"
[ -n "$VERBOSE" ] && CMD="$CMD $VERBOSE"

# Print header
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STRATEGY OPTIMIZATION${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Snapshot:${NC}         $SNAPSHOT"
echo -e "${GREEN}Parallel workers:${NC} $PARALLEL"
[ "$DRY_RUN" = true ] && echo -e "${GREEN}Mode:${NC}             DRY-RUN (count only, no evaluation)"
[ -n "$MAX_CONFIGS" ] && echo -e "${GREEN}Max configs:${NC}       $MAX_CONFIGS"
echo ""
echo -e "${YELLOW}Command:${NC} $CMD"
echo ""
echo -e "${BLUE}Starting optimization...${NC}"
echo ""

# Run optimization
$CMD

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Optimization complete${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Results saved to: artifacts/optimization/"
echo ""
