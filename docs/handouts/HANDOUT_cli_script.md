# Task 4: CLI Script

## File to Create
`scripts/find_best_strategy.py`

## Purpose
Command-line interface for running the strategy optimizer.

## Implementation

```python
#!/usr/bin/env python
"""
Find the best strategy that beats SPY across multiple time periods.

Usage:
    python scripts/find_best_strategy.py \
        --snapshot data/snapshots/snapshot_20260113_232157 \
        --output artifacts/optimization \
        --periods 3,5,10 \
        --max-configs 500 \
        --parallel 4
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from quantetf.optimization.optimizer import StrategyOptimizer
from quantetf.optimization.grid import count_configs


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Find strategies that beat SPY across multiple time periods'
    )

    parser.add_argument(
        '--snapshot',
        type=str,
        required=True,
        help='Path to data snapshot directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='artifacts/optimization',
        help='Output directory for results (default: artifacts/optimization)'
    )

    parser.add_argument(
        '--periods',
        type=str,
        default='3,5,10',
        help='Comma-separated evaluation periods in years (default: 3,5,10)'
    )

    parser.add_argument(
        '--max-configs',
        type=int,
        default=None,
        help='Maximum number of configs to test (for debugging)'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1 = sequential)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Just count configs without running optimization'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Parse periods
    periods = [int(p.strip()) for p in args.periods.split(',')]
    logger.info(f"Evaluation periods: {periods} years")

    # Validate snapshot exists
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        logger.error(f"Snapshot not found: {snapshot_path}")
        sys.exit(1)

    # Count configs
    counts = count_configs()
    logger.info(f"Configuration count by schedule:")
    logger.info(f"  Weekly: {counts['weekly']}")
    logger.info(f"  Monthly: {counts['monthly']}")
    logger.info(f"  Total: {counts['total']}")

    if args.dry_run:
        print("\nDry run complete. Use without --dry-run to execute optimization.")
        return

    # Run optimization
    logger.info("Starting optimization...")

    optimizer = StrategyOptimizer(
        snapshot_path=str(snapshot_path),
        output_dir=args.output,
        periods_years=periods,
        max_workers=args.parallel
    )

    result = optimizer.run(max_configs=args.max_configs)

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total configs tested: {result.successful_configs}")
    print(f"Failed configs: {result.failed_configs}")
    print(f"Strategies that beat SPY in ALL periods: {len(result.winners)}")

    if result.winners:
        print(f"\nBest strategy: {result.best_config.generate_name()}")
        best = result.winners[0]
        print(f"  Composite score: {best.composite_score:.3f}")
        for period_name, metrics in best.periods.items():
            print(f"  {period_name}: {metrics.active_return*100:.1f}% excess return, IR={metrics.information_ratio:.2f}")

        print(f"\nResults saved to: {optimizer.run_dir}")
        print(f"  - all_results.csv ({len(result.all_results)} strategies)")
        print(f"  - winners.csv ({len(result.winners)} strategies)")
        print(f"  - best_strategy.yaml")
        print(f"  - optimization_report.md")
    else:
        print("\nNo strategy beat SPY in all periods.")
        print(f"Results saved to: {optimizer.run_dir}")


if __name__ == '__main__':
    main()
```

## Usage Examples

### Basic run (sequential)
```bash
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --output artifacts/optimization
```

### Parallel execution with 4 workers
```bash
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --output artifacts/optimization \
    --parallel 4
```

### Quick test with 20 configs
```bash
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --max-configs 20 \
    --verbose
```

### Dry run (just count configs)
```bash
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --dry-run
```

### Custom evaluation periods
```bash
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --periods 1,3,5
```

## Expected Output

```
2026-01-14 14:30:22 - quantetf.optimization.optimizer - INFO - Generated 354 configurations
2026-01-14 14:30:22 - __main__ - INFO - Configuration count by schedule:
2026-01-14 14:30:22 - __main__ - INFO -   Weekly: {'momentum': 36, 'momentum_acceleration': 18, ...}
2026-01-14 14:30:22 - __main__ - INFO -   Monthly: {'momentum': 36, 'momentum_acceleration': 27, ...}
2026-01-14 14:30:22 - __main__ - INFO -   Total: 354
2026-01-14 14:30:22 - __main__ - INFO - Starting optimization...
Evaluating strategies: 100%|████████████████| 354/354 [15:42<00:00,  2.66s/it]

============================================================
OPTIMIZATION COMPLETE
============================================================
Total configs tested: 354
Failed configs: 3
Strategies that beat SPY in ALL periods: 12

Best strategy: vol_adjusted_momentum_lookback63_vol0.01_min50_top5_monthly
  Composite score: 1.247
  3yr: 18.4% excess return, IR=0.89
  5yr: 22.1% excess return, IR=0.95
  10yr: 31.2% excess return, IR=0.87

Results saved to: artifacts/optimization/20260114_143022
  - all_results.csv (351 strategies)
  - winners.csv (12 strategies)
  - best_strategy.yaml
  - optimization_report.md
```

## Dependencies

- `argparse` (stdlib)
- `pathlib` (stdlib)
- `logging` (stdlib)
- `quantetf.optimization.optimizer`
- `quantetf.optimization.grid`

## Make it Executable

```bash
chmod +x scripts/find_best_strategy.py
```

## Notes

- Always test with `--max-configs 10 --verbose` first
- Use `--dry-run` to verify config count before long runs
- Parallel execution may not work well with all backtest engines due to pickling
- Results are timestamped so you can run multiple times without overwriting
