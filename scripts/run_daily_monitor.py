#!/usr/bin/env python3
"""
Daily Regime Monitor

Run this script daily (e.g., via cron) to update regime state.

Usage:
    python scripts/run_daily_monitor.py
    python scripts/run_daily_monitor.py --date 2026-01-24
    python scripts/run_daily_monitor.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantetf.data.access import DataAccessFactory
from quantetf.production.regime_monitor import DailyRegimeMonitor


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Update daily regime state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run for today
    python scripts/run_daily_monitor.py

    # Run for specific date
    python scripts/run_daily_monitor.py --date 2026-01-24

    # Verbose output
    python scripts/run_daily_monitor.py --verbose

    # Use custom snapshot
    python scripts/run_daily_monitor.py --snapshot data/snapshots/snapshot_20260115_*/data.parquet
        """,
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to evaluate (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="data/snapshots/snapshot_20260115_*/data.parquet",
        help="Path to data snapshot",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default="data/state",
        help="Directory for state persistence",
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

    logger.info(f"Running regime monitor for {as_of.date()}")

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

        # Update regime
        state = monitor.update(as_of=as_of)

        # Print summary
        summary = monitor.get_regime_summary()
        print("\n" + "=" * 50)
        print("REGIME STATUS")
        print("=" * 50)
        print(f"Date:       {summary['as_of']}")
        print(f"Regime:     {summary['regime']}")
        print(f"Trend:      {summary['trend']}")
        print(f"Volatility: {summary['volatility']}")
        print("-" * 50)
        print("Indicators:")
        print(f"  SPY Price:  ${summary['indicators']['spy_price']:.2f}")
        print(f"  SPY 200MA:  ${summary['indicators']['spy_200ma']:.2f}")
        print(f"  SPY vs MA:  {summary['indicators']['spy_vs_ma_pct']:.1f}%")
        print(f"  VIX:        {summary['indicators']['vix']:.2f}")
        print("=" * 50 + "\n")

        logger.info(f"Regime update complete: {state.name}")
        return 0

    except Exception as e:
        logger.error(f"Failed to update regime: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
