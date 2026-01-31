#!/usr/bin/env python3
"""Rank strategies from saved backtest results.

This script loads previously saved backtest results and applies a scoring
method to rank strategies without rerunning backtests.

Example:
    $ python scripts/rank_strategies.py \
        --results artifacts/optimization/20260131/backtest_results.pkl

    $ python scripts/rank_strategies.py \
        --results artifacts/optimization/20260131/backtest_results.pkl \
        --scoring-method trailing_1y \
        --output artifacts/optimization/ranked
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from quantetf.optimization.optimizer import StrategyOptimizer

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
        description='Rank strategies from saved backtest results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to backtest_results.pkl file',
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: same as results file parent)',
    )

    parser.add_argument(
        '--scoring-method',
        type=str,
        choices=['trailing_1y', 'regime_weighted', 'multi_period'],
        default='multi_period',
        help='Scoring method for ranking (default: multi_period)',
    )

    parser.add_argument(
        '--trailing-days',
        type=int,
        default=252,
        help='Days for trailing window when using trailing_1y (default: 252)',
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

    logger.info(f"Loading results from: {results_path}")
    logger.info(f"Scoring method: {args.scoring_method}")

    try:
        # Load saved results
        saved_data = StrategyOptimizer.load_backtest_results(results_path)
        results = saved_data['results']
        periods_years = saved_data.get('periods_years', [1, 3])
        cost_bps = saved_data.get('cost_bps', 10.0)

        logger.info(f"Loaded {len(results)} strategies from {saved_data.get('timestamp')}")

        # Determine output directory
        output_dir = Path(args.output) if args.output else results_path.parent.parent

        # Create minimal optimizer for ranking (no data access needed)
        optimizer = StrategyOptimizer.__new__(StrategyOptimizer)
        optimizer.data_access = None  # Not needed for ranking
        optimizer.output_dir = output_dir
        optimizer.periods_years = periods_years
        optimizer.cost_bps = cost_bps
        optimizer.scoring_method = args.scoring_method
        optimizer.trailing_days = args.trailing_days
        optimizer.regime_analysis_enabled = False
        optimizer.max_workers = 1
        optimizer._run_dir = None
        optimizer._run_timestamp = None

        # Create output directory
        optimizer._run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        optimizer._run_dir = optimizer.output_dir / f"ranked_{optimizer._run_timestamp}"
        optimizer._run_dir.mkdir(parents=True, exist_ok=True)

        # Apply ranking
        all_results, winners = optimizer.rank_results(
            results,
            scoring_method=args.scoring_method,
        )

        # Save all results
        df = pd.DataFrame([r.to_dict() for r in all_results])
        df = df.sort_values('composite_score', ascending=False)
        df.to_csv(optimizer.run_dir / 'all_results.csv', index=False)

        # Save winners
        if winners:
            winners_df = pd.DataFrame([r.to_dict() for r in winners])
            winners_df = winners_df.sort_values('composite_score', ascending=False)
            winners_df.to_csv(optimizer.run_dir / 'winners.csv', index=False)

            # Save best strategy YAML
            best_result = winners[0]
            if best_result.config:
                config_dict = best_result.config.to_dict()
                config_dict['_ranking_metadata'] = {
                    'scoring_method': args.scoring_method,
                    'composite_score': round(best_result.composite_score, 4),
                    'source_results': str(args.results),
                    'ranked_at': optimizer._run_timestamp,
                }
                with open(optimizer.run_dir / 'best_strategy.yaml', 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    except Exception as e:
        logger.error(f"Ranking failed: {e}", exc_info=args.verbose)
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("RANKING COMPLETE")
    print("=" * 60)
    print(f"Scoring method: {args.scoring_method}")
    print(f"Total strategies ranked: {len(all_results)}")
    print(f"Winners (beat SPY all periods): {len(winners)}")

    if winners:
        print(f"\nTop 5 strategies:")
        for i, w in enumerate(winners[:5], 1):
            print(f"  {i}. {w.config_name[:50]} (score={w.composite_score:.3f})")

        print(f"\nBest: {winners[0].config_name}")

    print(f"\nResults saved to: {optimizer.run_dir}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
