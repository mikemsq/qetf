#!/usr/bin/env python3
"""
Regime-Aware Production Rebalancer

Run this script on rebalance dates to generate portfolio recommendations.

Usage:
    python scripts/run_regime_rebalance.py
    python scripts/run_regime_rebalance.py --dry-run
    python scripts/run_regime_rebalance.py --date 2026-01-24 --portfolio-value 500000
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantetf.data.access import DataAccessFactory
from quantetf.production.regime_monitor import DailyRegimeMonitor
from quantetf.production.rebalancer import RegimeAwareRebalancer


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Execute regime-aware portfolio rebalance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (no state changes)
    python scripts/run_regime_rebalance.py --dry-run

    # Execute rebalance
    python scripts/run_regime_rebalance.py

    # Custom portfolio value
    python scripts/run_regime_rebalance.py --portfolio-value 500000

    # Use specific regime mapping
    python scripts/run_regime_rebalance.py --regime-mapping artifacts/optimization/latest/regime_mapping.yaml
        """,
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Rebalance date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - don't persist state changes",
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=1_000_000.0,
        help="Total portfolio value for position sizing (default: $1,000,000)",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="data/snapshots/snapshot_20260115_*/data.parquet",
        help="Path to data snapshot",
    )
    parser.add_argument(
        "--regime-mapping",
        type=str,
        default=None,
        help="Path to regime_mapping.yaml (uses default if not specified)",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default="data/state",
        help="Directory for state persistence",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/rebalance",
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Parse date
    if args.date:
        as_of = pd.Timestamp(args.date)
    else:
        as_of = pd.Timestamp.now().normalize()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info(f"Running rebalance for {as_of.date()} [{mode}]")

    try:
        # Create data access
        data_access = DataAccessFactory.create_context(
            config={"snapshot_path": args.snapshot},
            enable_caching=True,
        )

        # Create monitor
        monitor = DailyRegimeMonitor(
            data_access=data_access,
            state_dir=Path(args.state_dir),
        )

        # Create rebalancer
        rebalancer = RegimeAwareRebalancer(
            data_access=data_access,
            regime_monitor=monitor,
            regime_mapping_path=Path(args.regime_mapping) if args.regime_mapping else None,
            artifacts_dir=Path(args.output_dir),
            portfolio_value=args.portfolio_value,
        )

        # Execute rebalance
        result = rebalancer.rebalance(as_of=as_of, dry_run=args.dry_run)

        # Print summary
        print("\n" + "=" * 60)
        print(f"REBALANCE SUMMARY {'(DRY RUN)' if args.dry_run else ''}")
        print("=" * 60)
        print(f"Date:           {result.as_of.date()}")
        print(f"Regime:         {result.regime}")
        print(f"Strategy:       {result.strategy_used}")
        print(f"Portfolio Value: {format_currency(args.portfolio_value)}")
        print("-" * 60)

        # Trades summary
        buys = [t for t in result.trades if t.action == "BUY"]
        sells = [t for t in result.trades if t.action == "SELL"]
        holds = [t for t in result.trades if t.action == "HOLD"]

        print(f"Trades:         {len(buys)} buys, {len(sells)} sells, {len(holds)} holds")
        print(f"Turnover:       {result._calculate_turnover():.1%}")

        if buys:
            print("\nBUYS:")
            for t in sorted(buys, key=lambda x: -x.notional_value)[:10]:
                print(f"  {t.ticker:6s}  {t.target_shares:8.1f} shares  {format_currency(t.notional_value):>12s}")

        if sells:
            print("\nSELLS:")
            for t in sorted(sells, key=lambda x: -x.notional_value)[:10]:
                print(f"  {t.ticker:6s}  {abs(t.shares_delta):8.1f} shares  {format_currency(t.notional_value):>12s}")

        # Target portfolio summary
        print("\nTARGET PORTFOLIO (top 10):")
        top_positions = result.target_portfolio.nlargest(10, "weight")
        for ticker, row in top_positions.iterrows():
            print(f"  {ticker:6s}  {row['weight']:6.1%}  {format_currency(row['notional']):>12s}")

        print("=" * 60)

        if not args.dry_run:
            output_dir = Path(args.output_dir) / as_of.strftime("%Y%m%d")
            print(f"\nArtifacts saved to: {output_dir}")

        print()
        logger.info("Rebalance complete")
        return 0

    except Exception as e:
        logger.error(f"Rebalance failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
