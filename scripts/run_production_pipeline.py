#!/usr/bin/env python3
"""Run the production portfolio management pipeline.

This script loads a production strategy configuration and runs the enhanced
pipeline with risk overlays, pre-trade checks, and state management.

IMPL-028: Updated to use DataAccessContext (DAL) instead of snapshot paths.

Examples:
    # Run in dry-run mode (no state changes)
    $ python scripts/run_production_pipeline.py \
        --config configs/strategies/production_value_momentum.yaml \
        --snapshot data/snapshots/snapshot_20260122_010523 \
        --data-config configs/data_access.yaml \
        --dry-run

    # Execute with state updates (uses default data config)
    $ python scripts/run_production_pipeline.py \
        --config configs/strategies/production_value_momentum.yaml \
        --snapshot data/snapshots/snapshot_20260122_010523 \
        --dry-run

    # Output as JSON
    $ python scripts/run_production_pipeline.py \
        --config configs/strategies/production_value_momentum.yaml \
        --snapshot data/snapshots/snapshot_20260122_010523 \
        --dry-run --output-format json
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

from quantetf.config.loader import load_strategy_config
from quantetf.data.access import DataAccessContext, DataAccessFactory
from quantetf.production import (
    MaxTurnoverCheck,
    MinTradeThresholdCheck,
    PipelineConfig,
    ProductionPipeline,
    SQLiteStateManager,
    InMemoryStateManager,
)
from quantetf.risk.overlays import (
    DrawdownCircuitBreaker,
    PositionLimitOverlay,
    VIXRegimeOverlay,
    VolatilityTargeting,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run production portfolio management pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no state changes)
  python scripts/run_production_pipeline.py \\
    --config configs/strategies/production_value_momentum.yaml \\
    --snapshot data/snapshots/snapshot_20260122_010523 \\
    --dry-run

  # With explicit data config
  python scripts/run_production_pipeline.py \\
    --config configs/strategies/production_value_momentum.yaml \\
    --data-config configs/data_access.yaml \\
    --dry-run

  # Execute with state updates
  python scripts/run_production_pipeline.py \\
    --config configs/strategies/production_value_momentum.yaml \\
    --snapshot data/snapshots/snapshot_20260122_010523 \\
    --execute
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to production strategy config YAML file",
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Path to data access config YAML file (default: uses strategy config's data_access section)",
    )

    parser.add_argument(
        "--snapshot-path",
        type=str,
        default=None,
        help="Path to snapshot data file (alternative to --data-config)",
    )

    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Date to run pipeline for (YYYY-MM-DD, default: latest available)",
    )

    # Mutually exclusive: --dry-run or --execute
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without persisting state changes",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        help="Run and persist state changes",
    )

    parser.add_argument(
        "--force-rebalance",
        action="store_true",
        help="Force rebalance even if not a scheduled rebalance date",
    )

    parser.add_argument(
        "--output-format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format for recommendations (default: text)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/production",
        help="Output directory for results (default: artifacts/production)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_production_config(config_path: Path) -> dict[str, Any]:
    """Load production config YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Raw config dictionary
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_risk_overlays(config: dict[str, Any]) -> list:
    """Create risk overlay instances from config.

    Args:
        config: Raw config dict with risk_overlays section

    Returns:
        List of RiskOverlay instances
    """
    overlays = []
    overlay_config = config.get("risk_overlays", {})

    # Volatility targeting
    vol_config = overlay_config.get("volatility_targeting", {})
    if vol_config.get("enabled", False):
        overlays.append(
            VolatilityTargeting(
                target_vol=vol_config.get("target_vol", 0.15),
                lookback_days=vol_config.get("lookback_days", 60),
                min_scale=vol_config.get("min_scale", 0.25),
                max_scale=vol_config.get("max_scale", 1.50),
            )
        )
        logger.info("Added VolatilityTargeting overlay")

    # Position limits
    pos_config = overlay_config.get("position_limits", {})
    if pos_config.get("enabled", False):
        overlays.append(
            PositionLimitOverlay(
                max_weight=pos_config.get("max_weight", 0.25),
                redistribute=pos_config.get("redistribute", True),
            )
        )
        logger.info("Added PositionLimitOverlay overlay")

    # Drawdown circuit breaker
    dd_config = overlay_config.get("drawdown_circuit_breaker", {})
    if dd_config.get("enabled", False):
        overlays.append(
            DrawdownCircuitBreaker(
                soft_threshold=dd_config.get("soft_threshold", 0.10),
                hard_threshold=dd_config.get("hard_threshold", 0.20),
                exit_threshold=dd_config.get("exit_threshold", 0.30),
            )
        )
        logger.info("Added DrawdownCircuitBreaker overlay")

    # VIX regime
    vix_config = overlay_config.get("vix_regime", {})
    if vix_config.get("enabled", False):
        defensive_tickers = vix_config.get(
            "defensive_tickers", ["AGG", "TLT", "GLD", "USMV", "SPLV"]
        )
        overlays.append(
            VIXRegimeOverlay(
                high_vix_threshold=vix_config.get("high_vix_threshold", 30.0),
                elevated_vix_threshold=vix_config.get("elevated_vix_threshold", 25.0),
                defensive_tickers=tuple(defensive_tickers),
            )
        )
        logger.info("Added VIXRegimeOverlay overlay")

    return overlays


def create_state_manager(config: dict[str, Any], dry_run: bool):
    """Create state manager from config.

    Args:
        config: Raw config dict with state section
        dry_run: If True, use in-memory state manager

    Returns:
        PortfolioStateManager instance
    """
    state_config = config.get("state", {})
    backend = state_config.get("backend", "sqlite")

    if dry_run:
        logger.info("Dry-run mode: using in-memory state manager")
        return InMemoryStateManager()

    if backend == "sqlite":
        db_path = Path(state_config.get("path", "data/production/portfolio_state.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using SQLite state manager: {db_path}")
        return SQLiteStateManager(db_path=db_path)
    else:
        logger.warning(f"Unknown backend '{backend}', using in-memory")
        return InMemoryStateManager()


def generate_target_weights(
    strategy_config,
    data_access: DataAccessContext,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Generate target weights using strategy's alpha model and portfolio construction.

    Args:
        strategy_config: Loaded StrategyConfig
        data_access: DataAccessContext for price/macro data
        as_of: Date for weight generation

    Returns:
        Series of target weights
    """
    # Get alpha scores
    alpha_model = strategy_config.alpha_model
    portfolio_constructor = strategy_config.portfolio_construction

    # Create universe
    universe = strategy_config.create_universe(as_of=as_of)

    # Generate alpha scores
    alpha_scores = alpha_model.score(
        as_of=as_of,
        universe=universe,
        features=None,  # Not used for simple alpha models
        data_access=data_access,
    )

    # Construct portfolio
    target_weights = portfolio_constructor.construct(
        as_of=as_of,
        universe=universe,
        alpha=alpha_scores,
        risk=None,  # Not using risk model
        data_access=data_access,
    )

    return target_weights.weights


def print_text_output(result, strategy_name: str) -> None:
    """Print pipeline result as formatted text.

    Args:
        result: PipelineResult
        strategy_name: Strategy name
    """
    print("=" * 80)
    print("PRODUCTION PIPELINE RESULT")
    print("=" * 80)
    print(f"Strategy:      {strategy_name}")
    print(f"As Of:         {result.as_of}")
    print(f"Status:        {result.execution_status}")
    print(f"Checks Passed: {result.pre_trade_checks_passed}")
    print()

    # Pre-trade check results
    if result.check_results:
        print("PRE-TRADE CHECKS:")
        print("-" * 40)
        for name, passed, reason in result.check_results:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {reason}")
        print()

    # Risk overlay diagnostics
    if result.overlay_diagnostics:
        print("RISK OVERLAY DIAGNOSTICS:")
        print("-" * 40)
        for name, diag_tuple in result.overlay_diagnostics:
            diag = dict(diag_tuple)
            print(f"  {name}:")
            for key, value in diag.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        print()

    # Target vs adjusted weights
    print("WEIGHTS:")
    print("-" * 40)
    print(f"{'Ticker':<10} {'Target':>10} {'Adjusted':>10} {'Change':>10}")
    print("-" * 40)

    all_tickers = set(result.target_weights.index) | set(result.adjusted_weights.index)
    for ticker in sorted(all_tickers):
        target = result.target_weights.get(ticker, 0.0)
        adjusted = result.adjusted_weights.get(ticker, 0.0)
        if target > 0.001 or adjusted > 0.001:
            change = adjusted - target
            print(f"{ticker:<10} {target:>10.2%} {adjusted:>10.2%} {change:>+10.2%}")
    print()

    # Trades
    if not result.trades.empty:
        print("RECOMMENDED TRADES:")
        print("-" * 60)
        print(f"{'Ticker':<10} {'Current':>12} {'Target':>12} {'Delta':>12}")
        print("-" * 60)
        for _, row in result.trades.iterrows():
            print(
                f"{row['ticker']:<10} "
                f"{row['current_weight']:>12.2%} "
                f"{row['target_weight']:>12.2%} "
                f"{row['delta_weight']:>+12.2%}"
            )
        print()
        turnover = 0.5 * result.trades["delta_weight"].abs().sum()
        print(f"Total Turnover: {turnover:.2%}")
    else:
        print("NO TRADES REQUIRED")

    print("=" * 80)


def save_results(
    result,
    args: argparse.Namespace,
    strategy_name: str,
    output_dir: Path,
) -> None:
    """Save pipeline results to disk.

    Args:
        result: PipelineResult
        args: Command-line arguments
        strategy_name: Strategy name
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_format == "json":
        result_dict = result.to_dict()
        result_dict["strategy_name"] = strategy_name
        result_dict["config_file"] = str(args.config)
        result_dict["data_config"] = str(args.data_config) if args.data_config else None
        result_dict["snapshot_path"] = str(args.snapshot_path) if args.snapshot_path else None
        result_dict["mode"] = "dry-run" if args.dry_run else "execute"

        output_file = output_dir / f"pipeline_result_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        logger.info(f"Saved JSON result: {output_file}")

    elif args.output_format == "csv":
        # Save trades as CSV
        if not result.trades.empty:
            trades_file = output_dir / f"trades_{timestamp}.csv"
            result.trades.to_csv(trades_file, index=False)
            logger.info(f"Saved trades CSV: {trades_file}")

        # Save weights as CSV
        weights_df = pd.DataFrame(
            {
                "ticker": result.adjusted_weights.index,
                "target_weight": result.target_weights.reindex(
                    result.adjusted_weights.index
                ).fillna(0),
                "adjusted_weight": result.adjusted_weights.values,
            }
        )
        weights_file = output_dir / f"weights_{timestamp}.csv"
        weights_df.to_csv(weights_file, index=False)
        logger.info(f"Saved weights CSV: {weights_file}")

    # Always save a summary JSON
    summary = {
        "timestamp": timestamp,
        "strategy_name": strategy_name,
        "as_of": str(result.as_of),
        "execution_status": result.execution_status,
        "pre_trade_checks_passed": result.pre_trade_checks_passed,
        "num_trades": len(result.trades),
        "mode": "dry-run" if args.dry_run else "execute",
    }
    summary_file = output_dir / f"summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_file}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load configs
        config_path = Path(args.config)
        logger.info(f"Loading config: {config_path}")

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1

        raw_config = load_production_config(config_path)
        strategy_config = load_strategy_config(config_path)

        # Create DataAccessContext using DAL (IMPL-028)
        data_access_config = {}

        if args.data_config:
            # Use explicit data config file
            data_config_path = Path(args.data_config)
            logger.info(f"Loading data config: {data_config_path}")
            data_access = DataAccessFactory.create_context(config_file=data_config_path)
        elif args.snapshot_path:
            # Use snapshot path directly
            snapshot_path = Path(args.snapshot_path)
            logger.info(f"Loading snapshot: {snapshot_path}")

            if snapshot_path.is_dir():
                data_path = snapshot_path / "data.parquet"
            else:
                data_path = snapshot_path

            if not data_path.exists():
                logger.error(f"Snapshot data not found: {data_path}")
                return 1

            data_access_config["snapshot_path"] = str(data_path)
            data_access = DataAccessFactory.create_context(config=data_access_config)
        elif "data_access" in raw_config:
            # Use data_access section from strategy config
            logger.info("Using data_access config from strategy file")
            data_access_config = raw_config["data_access"]
            data_access = DataAccessFactory.create_context(config=data_access_config)
        else:
            logger.error("No data source specified. Use --data-config, --snapshot-path, or add data_access section to config.")
            return 1

        # Determine as_of date
        if args.as_of:
            as_of = pd.Timestamp(args.as_of)
        else:
            # Use latest available date from price data
            try:
                # Read a small sample to get latest date
                ohlcv = data_access.prices.read_prices_as_of(
                    as_of=pd.Timestamp.now(),
                    tickers=["SPY"],  # Use SPY as reference ticker
                    lookback_days=5,
                )
                as_of = ohlcv.index.max()
                logger.info(f"Using latest available date: {as_of}")
            except Exception as e:
                logger.error(f"Could not determine latest date: {e}")
                return 1

        # Create components
        risk_overlays = create_risk_overlays(raw_config)
        state_manager = create_state_manager(raw_config, args.dry_run)

        # Create pre-trade checks
        pre_trade_checks = [
            MaxTurnoverCheck(max_turnover=0.50),
            MinTradeThresholdCheck(min_trade_weight=0.005),
        ]

        # Create pipeline config
        pipeline_config = PipelineConfig(
            strategy_config_path=config_path,
            risk_overlays=risk_overlays,
            pre_trade_checks=pre_trade_checks,
            state_manager=state_manager,
            rebalance_schedule=strategy_config.rebalance_frequency,
            dry_run=args.dry_run,
        )

        # Create pipeline
        pipeline = ProductionPipeline(config=pipeline_config)

        # Generate target weights
        logger.info("Generating target weights...")
        target_weights = generate_target_weights(strategy_config, data_access, as_of)
        logger.info(f"Generated weights for {len(target_weights)} positions")

        # Run enhanced pipeline
        logger.info("Running enhanced pipeline...")
        result = pipeline.run_enhanced(
            as_of=as_of,
            target_weights=target_weights,
            data_access=data_access,
            force_rebalance=args.force_rebalance,
        )

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output results
        if args.output_format == "text":
            print_text_output(result, strategy_config.name)
        elif args.output_format == "json":
            print(json.dumps(result.to_dict(), indent=2, default=str))

        # Save results
        save_results(result, args, strategy_config.name, output_dir)

        logger.info("Pipeline completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
