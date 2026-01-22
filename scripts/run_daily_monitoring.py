#!/usr/bin/env python3
"""Run daily monitoring checks for production portfolio.

This script performs daily operational checks including data quality,
market regime detection, and NAV/drawdown monitoring.

Examples:
    # Run all checks
    $ python scripts/run_daily_monitoring.py \
        --config configs/strategies/production_value_momentum.yaml \
        --check-quality \
        --check-regime \
        --update-nav

    # Check data quality only
    $ python scripts/run_daily_monitoring.py \
        --config configs/strategies/production_value_momentum.yaml \
        --check-quality

    # Check regime and update NAV
    $ python scripts/run_daily_monitoring.py \
        --config configs/strategies/production_value_momentum.yaml \
        --check-regime \
        --update-nav \
        --nav-value 105000.00
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantetf.data.access import DataAccessContext, DataAccessFactory
from quantetf.monitoring import (
    AlertManager,
    ConsoleAlertHandler,
    DataQualityChecker,
    DrawdownThreshold,
    FileAlertHandler,
    NAVTracker,
    RegimeMonitor,
    create_default_alert_manager,
)
from quantetf.production import (
    InMemoryStateManager,
    PortfolioState,
    SQLiteStateManager,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run daily monitoring checks for production portfolio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all checks
  python scripts/run_daily_monitoring.py \\
    --config configs/strategies/production_value_momentum.yaml \\
    --check-quality --check-regime --update-nav

  # Check data quality only
  python scripts/run_daily_monitoring.py \\
    --config configs/strategies/production_value_momentum.yaml \\
    --check-quality

  # Update NAV with specific value
  python scripts/run_daily_monitoring.py \\
    --config configs/strategies/production_value_momentum.yaml \\
    --update-nav --nav-value 105000.00
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to production strategy config YAML file",
    )

    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Path to snapshot directory (required for --check-quality)",
    )

    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Date to run checks for (YYYY-MM-DD, default: today)",
    )

    # Check options
    parser.add_argument(
        "--check-quality",
        action="store_true",
        help="Run data quality checks",
    )

    parser.add_argument(
        "--check-regime",
        action="store_true",
        help="Check market regime",
    )

    parser.add_argument(
        "--update-nav",
        action="store_true",
        help="Update NAV and check drawdown thresholds",
    )

    parser.add_argument(
        "--nav-value",
        type=float,
        default=None,
        help="NAV value to record (required with --update-nav unless using state)",
    )

    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for results (default: text)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/monitoring",
        help="Output directory for results (default: artifacts/monitoring)",
    )

    parser.add_argument(
        "--alert-file",
        type=str,
        default=None,
        help="File to write alerts to (in addition to console)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load config YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Raw config dictionary
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_state_manager(config: dict[str, Any]):
    """Create state manager from config.

    Args:
        config: Raw config dict with state section

    Returns:
        PortfolioStateManager instance
    """
    state_config = config.get("state", {})
    backend = state_config.get("backend", "sqlite")

    if backend == "sqlite":
        db_path = Path(state_config.get("path", "data/production/portfolio_state.db"))
        if db_path.exists():
            logger.info(f"Using existing SQLite state: {db_path}")
            return SQLiteStateManager(db_path=db_path)
        else:
            logger.warning(f"State DB not found: {db_path}, using in-memory")
            return InMemoryStateManager()
    else:
        return InMemoryStateManager()


def create_alert_manager(alert_file: str | None) -> AlertManager:
    """Create alert manager with appropriate handlers.

    Args:
        alert_file: Optional file path for alert logging

    Returns:
        Configured AlertManager
    """
    alert_manager = AlertManager()

    # Always add console handler
    alert_manager.add_handler(ConsoleAlertHandler())

    # Add file handler if specified
    if alert_file:
        alert_path = Path(alert_file)
        alert_path.parent.mkdir(parents=True, exist_ok=True)
        alert_manager.add_handler(FileAlertHandler(alert_path))
        logger.info(f"Alerts will be written to: {alert_path}")

    return alert_manager


def run_quality_check(
    data_access: DataAccessContext,
    tickers: list[str],
    alert_manager: AlertManager,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    """Run data quality checks.

    Args:
        data_access: Data access context
        tickers: Tickers to check
        alert_manager: Alert manager
        as_of: Date for checks

    Returns:
        Quality check result dictionary
    """
    logger.info("Running data quality checks...")

    checker = DataQualityChecker(
        data_access=data_access,
        alert_manager=alert_manager,
        stale_threshold_days=3,
        gap_threshold_days=5,
        spike_threshold=0.10,
    )

    # Use DataAccessContext to fetch and check prices
    try:
        result = checker.check_all(
            tickers=tickers,
            as_of=as_of,
        )
    except Exception as e:
        logger.warning(f"Could not run quality check: {e}")
        return {
            "error": str(e),
            "overall_status": "ERROR",
            "summary": {"stale_count": 0, "gap_count": 0, "anomaly_count": 0},
        }

    return result.to_dict()


def run_regime_check(
    data_access: DataAccessContext,
    alert_manager: AlertManager,
    as_of: str,
) -> dict[str, Any]:
    """Check market regime.

    Args:
        data_access: Data access context
        alert_manager: Alert manager
        as_of: Date string for check

    Returns:
        Regime check result dictionary
    """
    logger.info("Checking market regime...")

    monitor = RegimeMonitor(
        data_access=data_access,
        alert_manager=alert_manager,
    )

    result = monitor.check(as_of)

    return result.to_dict()


def run_nav_update(
    state_manager,
    alert_manager: AlertManager,
    as_of: pd.Timestamp,
    nav_value: float | None,
    drawdown_thresholds: list[float],
) -> dict[str, Any]:
    """Update NAV and check drawdown thresholds.

    Args:
        state_manager: Portfolio state manager
        alert_manager: Alert manager
        as_of: Date for update
        nav_value: NAV value to record
        drawdown_thresholds: Drawdown threshold levels

    Returns:
        NAV update result dictionary
    """
    logger.info("Updating NAV and checking drawdowns...")

    # Create custom thresholds from config
    thresholds = []
    for level in sorted(drawdown_thresholds):
        if level <= 0.10:
            alert_level = "INFO"
        elif level <= 0.20:
            alert_level = "WARNING"
        else:
            alert_level = "CRITICAL"
        thresholds.append(
            DrawdownThreshold(
                level=level,
                alert_level=alert_level,
                name=f"{int(level * 100)}% Drawdown",
            )
        )

    tracker = NAVTracker(
        state_manager=state_manager,
        alert_manager=alert_manager,
        drawdown_thresholds=tuple(thresholds) if thresholds else None,
    )

    # Get current state to determine NAV
    if nav_value is not None:
        nav = nav_value
    else:
        latest_state = state_manager.get_latest_state()
        if latest_state is not None:
            nav = latest_state.nav
            logger.info(f"Using NAV from state: {nav}")
        else:
            logger.error("No NAV value provided and no existing state found")
            return {"error": "No NAV value available"}

    result = tracker.update(as_of=as_of, nav=nav)

    return result.to_dict()


def print_text_output(results: dict[str, Any], strategy_name: str) -> None:
    """Print monitoring results as formatted text.

    Args:
        results: Combined results dictionary
        strategy_name: Strategy name
    """
    print("=" * 80)
    print("DAILY MONITORING REPORT")
    print("=" * 80)
    print(f"Strategy:  {strategy_name}")
    print(f"As Of:     {results.get('as_of', 'N/A')}")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print()

    # Data quality results
    if "quality_check" in results:
        quality = results["quality_check"]
        print("DATA QUALITY CHECK:")
        print("-" * 40)
        print(f"  Status: {quality.get('overall_status', 'N/A')}")
        summary = quality.get("summary", {})
        print(f"  Stale tickers:  {summary.get('stale_count', 0)}")
        print(f"  Gap issues:     {summary.get('gap_count', 0)}")
        print(f"  Anomalies:      {summary.get('anomaly_count', 0)}")

        if quality.get("stale_tickers"):
            print("  Stale tickers:")
            for stale in quality["stale_tickers"]:
                print(f"    - {stale['ticker']}: {stale['days_stale']} days stale")

        if quality.get("anomalies"):
            print("  Anomalies detected:")
            for anomaly in quality["anomalies"]:
                print(
                    f"    - {anomaly['ticker']}: {anomaly['anomaly_type']} "
                    f"on {anomaly['anomaly_date']}"
                )
        print()

    # Regime check results
    if "regime_check" in results:
        regime = results["regime_check"]
        state = regime.get("current_state", {})
        print("MARKET REGIME CHECK:")
        print("-" * 40)
        print(f"  Regime:        {state.get('regime', 'N/A')}")
        print(f"  VIX:           {state.get('vix', 'N/A')}")
        print(f"  Yield Spread:  {state.get('yield_curve_spread', 'N/A')}")
        print(f"  Changed:       {regime.get('regime_changed', False)}")

        if state.get("previous_regime"):
            print(f"  Previous:      {state['previous_regime']}")
        print()

    # NAV update results
    if "nav_update" in results:
        nav = results["nav_update"]
        if "error" not in nav:
            print("NAV UPDATE:")
            print("-" * 40)
            print(f"  Current NAV:   ${nav.get('nav', 0):,.2f}")
            print(f"  Peak NAV:      ${nav.get('peak_nav', 0):,.2f}")
            print(f"  Drawdown:      {nav.get('drawdown', 0):.2%}")

            if nav.get("thresholds_breached"):
                print(f"  Thresholds:    {', '.join(nav['thresholds_breached'])}")
        else:
            print(f"  Error: {nav['error']}")
        print()

    # Summary
    print("ALERTS SUMMARY:")
    print("-" * 40)
    total_alerts = 0
    for section in ["quality_check", "regime_check", "nav_update"]:
        if section in results and "alerts_emitted" in results[section]:
            alerts = results[section]["alerts_emitted"]
            total_alerts += len(alerts)
            if alerts:
                print(f"  {section.replace('_', ' ').title()}:")
                for alert in alerts:
                    print(f"    [{alert.get('level', 'INFO')}] {alert.get('message', '')}")
    print(f"  Total alerts: {total_alerts}")

    print("=" * 80)


def save_results(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save monitoring results to disk.

    Args:
        results: Combined results dictionary
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results as JSON
    output_file = output_dir / f"monitoring_result_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results: {output_file}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate arguments
    if not args.check_quality and not args.check_regime and not args.update_nav:
        logger.error(
            "At least one check must be specified: "
            "--check-quality, --check-regime, or --update-nav"
        )
        return 1

    if args.check_quality and not args.snapshot:
        logger.error("--snapshot is required for --check-quality")
        return 1

    try:
        # Load config
        config_path = Path(args.config)
        logger.info(f"Loading config: {config_path}")

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1

        config = load_config(config_path)
        strategy_name = config.get("name", config_path.stem)

        # Determine as_of date
        if args.as_of:
            as_of = pd.Timestamp(args.as_of)
            as_of_str = args.as_of
        else:
            as_of = pd.Timestamp.now().normalize()
            as_of_str = as_of.strftime("%Y-%m-%d")

        # Create alert manager
        alert_manager = create_alert_manager(args.alert_file)

        # Initialize results
        results: dict[str, Any] = {
            "strategy_name": strategy_name,
            "as_of": as_of_str,
            "timestamp": datetime.now().isoformat(),
            "checks_performed": [],
        }

        # Create DataAccessContext if needed for quality or regime checks
        data_access: DataAccessContext | None = None
        if args.check_quality or args.check_regime:
            # Build data access config from args and strategy config
            dal_config: dict[str, Any] = {}

            if args.snapshot:
                snapshot_path = Path(args.snapshot)
                if snapshot_path.is_dir():
                    dal_config["snapshot_path"] = str(snapshot_path / "data.parquet")
                else:
                    dal_config["snapshot_path"] = str(snapshot_path)

                if not Path(dal_config["snapshot_path"]).exists():
                    logger.error(f"Snapshot data not found: {dal_config['snapshot_path']}")
                    return 1

            # Get macro data dir from config
            macro_dir = config.get("risk_overlays", {}).get(
                "vix_regime", {}
            ).get("macro_data_dir", "data/raw/macro")
            dal_config["macro_data_dir"] = str(macro_dir)

            # Get universe config dir
            universe_path = config.get("universe")
            if universe_path:
                universe_config_dir = str(config_path.parent.parent.parent / Path(universe_path).parent)
                dal_config["universe_config_dir"] = universe_config_dir

            try:
                data_access = DataAccessFactory.create_context(config=dal_config)
            except Exception as e:
                logger.error(f"Failed to create DataAccessContext: {e}")
                return 1

        # Run data quality check
        if args.check_quality:
            results["checks_performed"].append("quality")

            # Get tickers from universe config
            universe_path = config.get("universe")
            if universe_path:
                universe_full_path = config_path.parent.parent.parent / universe_path
                with open(universe_full_path) as f:
                    universe_config = yaml.safe_load(f)
                tickers = universe_config.get("source", {}).get(
                    "tickers", universe_config.get("tickers", [])
                )
            else:
                # Try to get tickers from data access
                try:
                    ohlcv = data_access.prices.read_prices_as_of(as_of, lookback_days=1)
                    tickers = list(ohlcv.columns.get_level_values("Ticker").unique())
                except Exception:
                    tickers = []

            results["quality_check"] = run_quality_check(
                data_access=data_access,
                tickers=tickers,
                alert_manager=alert_manager,
                as_of=as_of,
            )

        # Run regime check
        if args.check_regime:
            results["checks_performed"].append("regime")

            try:
                results["regime_check"] = run_regime_check(
                    data_access=data_access,
                    alert_manager=alert_manager,
                    as_of=as_of_str,
                )
            except Exception as e:
                logger.warning(f"Regime check failed: {e}")
                results["regime_check"] = {"error": str(e)}

        # Run NAV update
        if args.update_nav:
            results["checks_performed"].append("nav")

            state_manager = create_state_manager(config)

            drawdown_thresholds = config.get("monitoring", {}).get(
                "drawdown_alerts", [0.10, 0.20, 0.30]
            )

            results["nav_update"] = run_nav_update(
                state_manager=state_manager,
                alert_manager=alert_manager,
                as_of=as_of,
                nav_value=args.nav_value,
                drawdown_thresholds=drawdown_thresholds,
            )

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output results
        if args.output_format == "text":
            print_text_output(results, strategy_name)
        else:
            print(json.dumps(results, indent=2, default=str))

        # Save results
        save_results(results, output_dir)

        logger.info("Monitoring completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Monitoring failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
