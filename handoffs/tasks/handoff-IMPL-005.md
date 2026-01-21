# Task Handoff: IMPL-005 - End-to-End Backtest Script

**Task ID:** IMPL-005
**Status:** blocked (depends on IMPL-004)
**Priority:** high
**Estimated Time:** 2 hours

---

## Quick Context

You are creating the end-to-end script that runs a complete backtest on the 5-year snapshot and generates results. This is the culmination of Phase 2 - making everything work together and producing actual backtest results!

**Why this matters:** This script proves the system works. It's what we'll use to test strategies, generate recommendations, and validate our approach before going to production.

---

## What You Need to Know

### Architecture

This is a **script**, not a library module:
- Lives in `scripts/run_backtest.py`
- Loads snapshot data
- Configures backtest parameters
- Runs backtest using SimpleBacktestEngine
- Saves results to artifacts/
- Prints summary to console

### Components Required ✅

All components should be implemented by now:
1. SnapshotDataStore
2. MomentumAlpha
3. EqualWeightTopN
4. FlatTransactionCost
5. SimpleBacktestEngine

### Design Decisions

- **Configuration:** Command-line arguments or config file
- **Output:** Save to `artifacts/backtests/YYYY-MM-DD-strategy-name/`
- **Results:** Equity curve CSV, metrics JSON, plots (optional)
- **Logging:** Detailed console output showing progress

---

## Files to Read First

1. **`/workspaces/qetf/CLAUDE_CONTEXT.md`** - Coding standards
2. **`/workspaces/qetf/src/quantetf/backtest/simple_engine.py`** - Engine to use
3. **`/workspaces/qetf/src/quantetf/data/snapshot_store.py`** - Data loading
4. **`/workspaces/qetf/scripts/ingest_etf_data.py`** - Example script structure
5. **`/workspaces/qetf/data/snapshots/snapshot_5yr_20etfs/`** - Snapshot location

---

## Implementation Steps

### 1. Create the script file

Create `/workspaces/qetf/scripts/run_backtest.py`

### 2. Add imports and argument parsing

```python
#!/usr/bin/env python3
"""Run a backtest on historical data.

Example:
    $ python scripts/run_backtest.py \\
        --snapshot data/snapshots/snapshot_5yr_20etfs \\
        --start 2021-01-01 \\
        --end 2025-12-31 \\
        --strategy momentum-ew-top5
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

from quantetf.backtest.simple_engine import SimpleBacktestEngine
from quantetf.alpha.momentum import MomentumAlpha
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.types import BacktestConfig, Universe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run ETF backtest')

    parser.add_argument(
        '--snapshot',
        type=str,
        default='data/snapshots/snapshot_5yr_20etfs',
        help='Path to snapshot directory'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2021-01-01',
        help='Backtest start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2025-12-31',
        help='Backtest end date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default='momentum-ew-top5',
        help='Strategy name for output directory'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital ($)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of ETFs to hold'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        default=252,
        help='Momentum lookback days'
    )

    parser.add_argument(
        '--cost-bps',
        type=float,
        default=10.0,
        help='Transaction cost (basis points)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/backtests',
        help='Output directory for results'
    )

    return parser.parse_args()
```

### 3. Implement main backtest function

```python
def run_backtest(args):
    """Run the backtest with given parameters."""

    logger.info("=" * 80)
    logger.info("QuantETF Backtest")
    logger.info("=" * 80)

    # 1. Load snapshot
    logger.info(f"Loading snapshot: {args.snapshot}")
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    store = SnapshotDataStore(str(snapshot_path))

    # 2. Load universe from snapshot metadata
    metadata_path = snapshot_path / 'metadata.yaml'
    import yaml
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    tickers = tuple(metadata['tickers'])
    logger.info(f"Universe: {len(tickers)} ETFs")

    universe = Universe(
        as_of=pd.Timestamp(args.end),
        tickers=tickers
    )

    # 3. Configure backtest
    config = BacktestConfig(
        start_date=pd.Timestamp(args.start),
        end_date=pd.Timestamp(args.end),
        universe=universe,
        initial_capital=args.capital
    )

    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Initial capital: ${args.capital:,.2f}")

    # 4. Create components
    alpha_model = MomentumAlpha(lookback_days=args.lookback)
    portfolio = EqualWeightTopN(top_n=args.top_n)
    cost_model = FlatTransactionCost(cost_bps=args.cost_bps)

    logger.info(f"Alpha: {args.lookback}-day momentum")
    logger.info(f"Portfolio: Equal-weight top {args.top_n}")
    logger.info(f"Costs: {args.cost_bps} bps per trade")

    # 5. Run backtest
    logger.info("Running backtest...")
    engine = SimpleBacktestEngine()

    result = engine.run(
        config=config,
        alpha_model=alpha_model,
        portfolio=portfolio,
        cost_model=cost_model,
        store=store
    )

    logger.info("Backtest complete!")

    # 6. Print metrics
    print_metrics(result)

    # 7. Save results
    output_dir = save_results(result, args)
    logger.info(f"Results saved to: {output_dir}")

    return result


def print_metrics(result):
    """Print backtest metrics to console."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)

    metrics = result.metrics

    print(f"\nTotal Return:     {metrics['total_return']:>10.2%}")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
    print(f"Total Costs:      ${metrics['total_costs']:>10,.2f}")
    print(f"Num Rebalances:   {metrics['num_rebalances']:>10,}")

    # Additional metrics
    final_nav = result.equity_curve['nav'].iloc[-1]
    initial_nav = result.config.initial_capital
    print(f"\nInitial NAV:      ${initial_nav:>10,.2f}")
    print(f"Final NAV:        ${final_nav:>10,.2f}")
    print(f"Profit/Loss:      ${final_nav - initial_nav:>10,.2f}")

    logger.info("=" * 80)


def save_results(result, args):
    """Save backtest results to disk."""

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{timestamp}_{args.strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save equity curve
    equity_path = output_dir / 'equity_curve.csv'
    result.equity_curve.to_csv(equity_path)
    logger.info(f"Saved equity curve: {equity_path}")

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(result.metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics: {metrics_path}")

    # Save holdings history
    holdings_path = output_dir / 'holdings_history.csv'
    result.holdings_history.to_csv(holdings_path)
    logger.info(f"Saved holdings: {holdings_path}")

    # Save weights history
    weights_path = output_dir / 'weights_history.csv'
    result.weights_history.to_csv(weights_path)
    logger.info(f"Saved weights: {weights_path}")

    # Save config
    config_path = output_dir / 'config.json'
    config_dict = {
        'start_date': str(result.config.start_date),
        'end_date': str(result.config.end_date),
        'initial_capital': result.config.initial_capital,
        'universe': list(result.config.universe.tickers),
        'strategy': args.strategy,
        'top_n': args.top_n,
        'lookback_days': args.lookback,
        'cost_bps': args.cost_bps,
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved config: {config_path}")

    return output_dir


def main():
    """Main entry point."""
    args = parse_args()

    try:
        result = run_backtest(args)
        logger.info("SUCCESS!")
        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
```

### 4. Create optional visualization script

Create `/workspaces/qetf/notebooks/backtest_analysis.ipynb` (optional):

```python
# Jupyter notebook to analyze backtest results
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_dir = Path('../artifacts/backtests/20260109_120000_momentum-ew-top5')

equity = pd.read_csv(results_dir / 'equity_curve.csv', index_col=0, parse_dates=True)
weights = pd.read_csv(results_dir / 'weights_history.csv', index_col=0, parse_dates=True)

# Plot equity curve
fig, ax = plt.subplots(figsize=(12, 6))
equity['nav'].plot(ax=ax, label='Strategy NAV')
ax.set_title('Backtest Equity Curve')
ax.set_xlabel('Date')
ax.set_ylabel('NAV ($)')
ax.legend()
plt.tight_layout()
plt.show()

# Plot returns distribution
fig, ax = plt.subplots(figsize=(10, 6))
equity['returns'].hist(bins=50, ax=ax)
ax.set_title('Returns Distribution')
ax.set_xlabel('Monthly Return')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Show portfolio composition over time
weights.plot(figsize=(12, 8), title='Portfolio Weights Over Time')
plt.ylabel('Weight')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### 5. Test the script

```bash
# Run with defaults
python scripts/run_backtest.py

# Run with custom parameters
python scripts/run_backtest.py \
    --start 2022-01-01 \
    --end 2024-12-31 \
    --top-n 3 \
    --lookback 126 \
    --cost-bps 5
```

---

## Acceptance Criteria

- [ ] Script runs successfully on snapshot_5yr_20etfs
- [ ] Command-line arguments work correctly
- [ ] Backtest executes without errors
- [ ] Metrics printed to console (readable format)
- [ ] Results saved to artifacts/backtests/
- [ ] Output includes: equity_curve.csv, metrics.json, holdings, weights, config
- [ ] Script handles errors gracefully (missing snapshot, invalid dates)
- [ ] Logging provides useful progress information
- [ ] Documentation includes usage examples

---

## Success Looks Like

```bash
$ python scripts/run_backtest.py

================================================================================
QuantETF Backtest
================================================================================
2026-01-09 14:30:00 - Loading snapshot: data/snapshots/snapshot_5yr_20etfs
2026-01-09 14:30:00 - Universe: 20 ETFs
2026-01-09 14:30:00 - Period: 2021-01-01 to 2025-12-31
2026-01-09 14:30:00 - Initial capital: $100,000.00
2026-01-09 14:30:00 - Alpha: 252-day momentum
2026-01-09 14:30:00 - Portfolio: Equal-weight top 5
2026-01-09 14:30:00 - Costs: 10.0 bps per trade
2026-01-09 14:30:00 - Running backtest...
2026-01-09 14:30:05 - Backtest complete!

================================================================================
RESULTS
================================================================================

Total Return:          42.50%
Sharpe Ratio:           1.25
Max Drawdown:         -18.32%
Total Costs:         $1,234.56
Num Rebalances:           60

Initial NAV:      $100,000.00
Final NAV:        $142,500.00
Profit/Loss:       $42,500.00

================================================================================
2026-01-09 14:30:05 - Results saved to: artifacts/backtests/20260109_143000_momentum-ew-top5
2026-01-09 14:30:05 - SUCCESS!
```

---

## Questions? Issues?

If blocked or unclear:
1. **Check existing scripts** - `scripts/ingest_etf_data.py` for patterns
2. **Test incrementally** - Run backtest in Python REPL first
3. **Check snapshot structure** - `data/snapshots/snapshot_5yr_20etfs/`
4. **Document issues** in `handoffs/completion-IMPL-005.md`

---

## Related Files

- Backtest engine: `src/quantetf/backtest/simple_engine.py`
- Data store: `src/quantetf/data/snapshot_store.py`
- Example script: `scripts/ingest_etf_data.py`
- Standards: `CLAUDE_CONTEXT.md`

---

## When Done

1. Run the script and verify it completes successfully
2. Check output in artifacts/backtests/
3. Verify metrics look reasonable (positive/negative returns, sensible Sharpe)
4. Create `handoffs/completion-IMPL-005.md` with:
   - Script implementation notes
   - Sample backtest output
   - Any issues encountered
   - Suggestions for improvements
5. Update `TASKS.md`: change status to `completed`
6. Commit with message: "Add end-to-end backtest script (IMPL-005)"

---

## Important Reminder

**Save progress frequently!** Update completion note as you go:
- After implementing argument parsing
- After implementing run_backtest function
- After implementing save_results function
- After successful test run

---

## Phase 2 Completion!

When this task is done, **Phase 2 is complete**! 🎉

You'll have:
- ✅ Complete backtest system
- ✅ Real results on 5-year snapshot
- ✅ Proof that momentum strategy works (or doesn't!)
- ✅ Foundation for Phase 3 (more strategies)

This is a major milestone!
