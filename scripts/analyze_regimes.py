#!/usr/bin/env python3
"""Run regime analysis on saved backtest results.

This script loads previously saved backtest results (with daily returns)
and runs regime analysis to produce regime->strategy mapping.

Example:
    $ python scripts/analyze_regimes.py \
        --results artifacts/optimization/20260131/backtest_results.pkl \
        --snapshot data/snapshots/snapshot_latest

    $ python scripts/analyze_regimes.py \
        --results artifacts/optimization/20260131/backtest_results.pkl \
        --snapshot data/snapshots/snapshot_latest \
        --output artifacts/optimization/regimes
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from quantetf.data.access import DataAccessFactory
from quantetf.optimization.optimizer import OptimizationResult, StrategyOptimizer

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run regime analysis on saved backtest results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to backtest_results.pkl file',
    )

    parser.add_argument(
        '--snapshot',
        type=str,
        required=True,
        help='Path to data snapshot (needed for macro/regime data)',
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: same as results file parent)',
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging',
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Validate results file
    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return 1

    # Validate snapshot
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        logger.error(f"Snapshot not found: {snapshot_path}")
        return 1
    if snapshot_path.is_dir():
        data_file = snapshot_path / 'data.parquet'
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return 1
        snapshot_path = data_file

    # Find macro data directory (needed for regime analysis)
    macro_data_dir = Path("data/raw/macro")
    if not macro_data_dir.exists():
        logger.error(f"Macro data directory not found: {macro_data_dir}")
        logger.error("Regime analysis requires VIX and other macro data.")
        return 1

    logger.info(f"Loading results from: {results_path}")
    logger.info(f"Using macro data from: {macro_data_dir}")

    try:
        # Load saved results
        saved_data = StrategyOptimizer.load_backtest_results(results_path)
        results = saved_data['results']
        periods_years = saved_data.get('periods_years', [1, 3])
        cost_bps = saved_data.get('cost_bps', 10.0)

        logger.info(f"Loaded {len(results)} strategies from {saved_data.get('timestamp')}")

        # Create data access for regime analysis
        data_access = DataAccessFactory.create_context(
            config={
                "snapshot_path": str(snapshot_path),
                "macro_data_dir": str(macro_data_dir),
            },
            enable_caching=True,
        )

        output_dir = Path(args.output) if args.output else results_path.parent.parent
        optimizer = StrategyOptimizer(
            data_access=data_access,
            output_dir=output_dir,
            periods_years=periods_years,
            cost_bps=cost_bps,
            regime_analysis_enabled=True,
        )

        # Create output directory
        optimizer._run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        optimizer._run_dir = optimizer.output_dir / f"regime_{optimizer._run_timestamp}"
        optimizer._run_dir.mkdir(parents=True, exist_ok=True)

        # Find winners and run regime analysis
        winners = [r for r in results if r.beats_spy_all_periods]
        winners.sort(key=lambda r: r.composite_score, reverse=True)

        if not winners:
            logger.warning("No winning strategies found for regime analysis")
            print("\nNo strategies beat SPY in all periods - skipping regime analysis")
            return 0

        # Extract configs from results
        configs = [r.config for r in results if r.config is not None]

        # Run regime analysis
        regime_outputs = optimizer._run_regime_analysis(winners, configs)

        # Save regime outputs
        if regime_outputs:
            result = OptimizationResult(
                all_results=results,
                winners=winners,
                best_config=winners[0].config if winners else None,
                run_timestamp=optimizer._run_timestamp,
                total_configs=len(results),
                successful_configs=len(results),
                failed_configs=0,
                output_dir=optimizer._run_dir,
                regime_outputs=regime_outputs,
            )
            optimizer._save_regime_outputs(result)

    except Exception as e:
        logger.error(f"Regime analysis failed: {e}", exc_info=args.verbose)
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("REGIME ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Strategies analyzed: {len(winners)}")

    if regime_outputs:
        mapping = regime_outputs.get('regime_mapping', {})
        print(f"Regimes mapped: {len(mapping)}")
        for regime, info in mapping.items():
            strategy = info.get('strategy', 'N/A')
            print(f"  {regime}: {strategy[:40]}")

        dist = regime_outputs.get('regime_distribution', {})
        if dist:
            print(f"\nRegime distribution:")
            for regime, pct in sorted(dist.items(), key=lambda x: -x[1]):
                print(f"  {regime}: {pct*100:.1f}%")

    print(f"\nResults saved to: {optimizer.run_dir}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
