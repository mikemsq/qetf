#!/bin/bash
# Quick reference card for bash scripts

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║                   QUANTETF BASH SCRIPTS QUICK REFERENCE                   ║
╚════════════════════════════════════════════════════════════════════════════╝

MAIN OPERATIONS
═══════════════════════════════════════════════════════════════════════════════

  run_optimization.sh              Find optimal strategy parameters
    Usage: ./run_optimization.sh [--parallel N] [--max-configs N]
    Time:  15-60 minutes
    Output: artifacts/optimization/*/best_strategy.yaml

  run_backtest.sh                  Execute full backtest with analysis
    Usage: ./run_backtest.sh [--analysis] [--start DATE] [--end DATE]
    Time:  5-15 minutes
    Output: artifacts/backtests/*/performance_analysis.json

  run_walk_forward.sh              Validate on out-of-sample data
    Usage: ./run_walk_forward.sh [--plots] [--train-years N]
    Time:  30-120 minutes
    Output: artifacts/walk_forward/*/summary.json

  run_monitoring.sh                Monitor real-time performance
    Usage: ./run_monitoring.sh [--continuous] [--email-alerts]
    Time:  1-5 minutes per check
    Output: artifacts/monitoring/*/latest_metrics.json

  run_all.sh                       Run complete workflow
    Usage: ./run_all.sh [--steps STEPS] [--parallel N]
    Time:  60-240 minutes (full workflow)
    Output: All of the above


UTILITY COMMANDS
═══════════════════════════════════════════════════════════════════════════════

  check_status.sh                  Show process and output status
    Usage: ./check_status.sh [--all|--processes|--walk-forward]

  view_results.sh                  Display detailed analysis results
    Usage: ./view_results.sh [--walk-forward latest] [--metrics-only]


QUICK WORKFLOWS
═══════════════════════════════════════════════════════════════════════════════

  FIND BEST STRATEGY
    ./run_all.sh                              # Runs: optimize → backtest → walk-forward

  JUST VALIDATE EXISTING STRATEGY
    ./run_walk_forward.sh --plots --verbose
    ./view_results.sh --walk-forward latest

  CHECK CURRENT STATUS
    ./check_status.sh --all

  VIEW LATEST RESULTS
    ./view_results.sh --walk-forward latest   # Walk-forward validation
    ./view_results.sh --backtest latest       # Backtest results
    ./view_results.sh --optimization          # Optimization results

  MONITOR CONTINUOUSLY
    ./run_monitoring.sh --continuous --email-alerts


COMMON OPTIONS
═══════════════════════════════════════════════════════════════════════════════

  --help                          Show detailed help for any script
  --verbose                       Enable verbose output
  --plots                         Save visualization plots
  --analysis                      Generate enhanced analysis
  --dry-run                       Show commands without executing
  --parallel N                    Use N parallel jobs
  --start DATE                    Start date (YYYY-MM-DD)
  --end DATE                      End date (YYYY-MM-DD)


EXAMPLE COMMANDS
═══════════════════════════════════════════════════════════════════════════════

  # Check what's currently running
  ./check_status.sh --processes

  # Start optimization with 8 parallel jobs
  ./run_optimization.sh --parallel 8

  # Run backtest with analysis
  ./run_backtest.sh --analysis --verbose

  # Run walk-forward validation with plots
  ./run_walk_forward.sh --plots --verbose

  # View latest walk-forward results with metrics only
  ./view_results.sh --walk-forward latest --metrics-only

  # Run complete workflow (optimize + backtest + validate)
  ./run_all.sh

  # Preview workflow without executing
  ./run_all.sh --dry-run


KEY OUTPUT FILES
═══════════════════════════════════════════════════════════════════════════════

  Optimization:
    artifacts/optimization/[timestamp]/best_strategy.yaml    ← Use for deployment

  Backtest:
    artifacts/backtests/[timestamp]/performance_analysis.json
    artifacts/backtests/[timestamp]/cycle_metrics.json

  Walk-Forward (VALIDATION):
    artifacts/walk_forward/[timestamp]/summary.json
    artifacts/walk_forward/[timestamp]/window_results.csv
    artifacts/walk_forward/[timestamp]/walk_forward_analysis.png


WHAT EACH STEP DOES
═══════════════════════════════════════════════════════════════════════════════

  1. OPTIMIZATION (--parallel 8)
     Tests 438 strategy configurations using grid search
     Finds: top_n, lookback, cost_bps, rebalance_frequency
     Output: best_strategy.yaml with optimal parameters

  2. BACKTEST (--analysis)
     Runs strategy on full 10-year historical data
     Calculates: Sharpe, Sortino, max drawdown, monthly returns
     Output: Detailed performance metrics and cycle analysis

  3. WALK-FORWARD (--plots)
     Validates strategy on 5 rolling windows of unseen data
     Checks: Out-of-sample returns, Sharpe ratio, overfitting
     Output: Window-by-window validation, degradation analysis
     ✓ PASS: 80% of windows beat training performance

  4. MONITORING (--continuous)
     Tracks real-time strategy performance
     Alerts: Email on drawdowns > 5%
     Output: Latest metrics and performance updates


VALIDATION CRITERIA
═══════════════════════════════════════════════════════════════════════════════

  Strategy is PRODUCTION-READY when:
    ✓ Walk-forward: 80%+ windows beat training
    ✓ Walk-forward: Sharpe degradation < 0.05
    ✓ Walk-forward: 75%+ of test windows positive
    ✓ Backtest: Sharpe ratio > 0.5
    ✓ Backtest: Max drawdown < 30%
    ✓ Recent results (last 90 days): Positive returns


PERFORMANCE EXPECTATIONS
═══════════════════════════════════════════════════════════════════════════════

  Optimization:     15-60 min   (depends on --parallel value)
  Backtest:         5-15 min
  Walk-Forward:     30-120 min
  Monitoring:       1-5 min per check
  Complete Flow:    60-240 min (full workflow)


FOR DETAILED HELP
═══════════════════════════════════════════════════════════════════════════════

  Each script has full documentation:

    ./run_optimization.sh --help
    ./run_backtest.sh --help
    ./run_walk_forward.sh --help
    ./run_monitoring.sh --help
    ./run_all.sh --help
    ./check_status.sh --help
    ./view_results.sh --help

  Or read full guide:
    cat scripts/BASH_SCRIPTS_README.md


LATEST RESULTS SUMMARY
═══════════════════════════════════════════════════════════════════════════════

  Walk-Forward Validation: ✓ PASSED
    • 5 rolling windows tested
    • 80% of windows beat training performance
    • Out-of-sample Sharpe: 0.24 (positive)
    • Sharpe degradation: -0.016 (no overfitting)
    • 4 of 5 windows had positive returns on unseen data

  Best Strategy: trend_filtered_momentum
    • Parameters: top_n=5, lookback=252, cost_bps=10
    • Rebalance: monthly
    • Status: READY FOR PRODUCTION


═══════════════════════════════════════════════════════════════════════════════

  Created: 2025-01-17
  Version: 1.0
  
═══════════════════════════════════════════════════════════════════════════════

EOF
