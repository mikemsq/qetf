"""Strategy optimizer - main orchestration for parameter sweep.

This module provides the main optimizer orchestrator that generates all
configurations, runs evaluations, ranks results, and produces reports.

Uses DataAccessContext (DAL) for all data access, enabling:
- Decoupling from specific data storage implementations
- Easy mocking in tests
- Transparent caching

Key Features:
- Sequential or parallel execution (configurable via max_workers)
- Progress tracking with tqdm
- Graceful error handling (log and skip failed configs)
- Comprehensive output files (CSV, YAML, Markdown)

Example:
    >>> from quantetf.optimization.optimizer import StrategyOptimizer
    >>> from quantetf.data.access import DataAccessFactory
    >>>
    >>> ctx = DataAccessFactory.create_context(
    ...     config={"snapshot_path": "data/snapshots/snapshot_20260113/data.parquet"}
    ... )
    >>> optimizer = StrategyOptimizer(
    ...     data_access=ctx,
    ...     output_dir="artifacts/optimization",
    ...     periods_years=[3, 5, 10],
    ...     max_workers=1,
    ... )
    >>> result = optimizer.run(max_configs=10)  # Limit for testing
    >>> print(f"Winners: {len(result.winners)}")
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from quantetf.data.access import DataAccessContext
from quantetf.optimization.evaluator import MultiPeriodEvaluator, MultiPeriodResult
from quantetf.optimization.grid import StrategyConfig, generate_configs

# Regime analysis imports (optional - graceful degradation if not available)
try:
    from quantetf.regime import (
        RegimeAnalyzer,
        RegimeDetector,
        RegimeIndicators,
        load_thresholds,
    )
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False

logger = logging.getLogger(__name__)

# Optional tqdm import - falls back to simple iteration if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


@dataclass
class OptimizationResult:
    """Complete optimization run results.

    Contains all results from an optimization sweep, including metadata
    about the run and summaries of winning strategies.

    Attributes:
        all_results: List of MultiPeriodResult for all evaluated configs
        winners: List of MultiPeriodResult for configs that beat SPY in all periods
        best_config: The top-ranked StrategyConfig, or None if no winners
        run_timestamp: ISO format timestamp when the run started
        total_configs: Total number of configs in the parameter grid
        successful_configs: Number of configs that completed evaluation
        failed_configs: Number of configs that failed during evaluation
        output_dir: Directory where output files were saved
        regime_outputs: Optional dict with regime analysis results
    """

    all_results: List[MultiPeriodResult]
    winners: List[MultiPeriodResult]
    best_config: Optional[StrategyConfig]
    run_timestamp: str
    total_configs: int
    successful_configs: int
    failed_configs: int
    output_dir: Optional[Path] = None
    regime_outputs: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        """Return a human-readable summary of the optimization run."""
        lines = [
            "=" * 60,
            "Strategy Optimization Results",
            "=" * 60,
            f"Run timestamp: {self.run_timestamp}",
            f"Total configurations: {self.total_configs}",
            f"Successful evaluations: {self.successful_configs}",
            f"Failed evaluations: {self.failed_configs}",
            f"Strategies that beat SPY (all periods): {len(self.winners)}",
            "",
        ]

        if self.best_config:
            lines.extend([
                "Best Strategy:",
                f"  Name: {self.best_config.generate_name()}",
                f"  Alpha: {self.best_config.alpha_type}",
                f"  Params: {self.best_config.alpha_params}",
                f"  Top N: {self.best_config.top_n}",
                f"  Schedule: {self.best_config.schedule_name}",
            ])

            if self.winners:
                best_result = self.winners[0]
                lines.append(f"  Composite Score: {best_result.composite_score:.4f}")
        else:
            lines.append("No winning strategies found.")

        if self.output_dir:
            lines.extend([
                "",
                f"Output saved to: {self.output_dir}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


class StrategyOptimizer:
    """Main optimizer that searches across parameter space.

    This class orchestrates the optimization process:
    1. Generates all valid strategy configurations
    2. Evaluates each configuration across multiple time periods
    3. Ranks results by composite score
    4. Saves comprehensive output files

    Uses DataAccessContext (DAL) for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching

    Example:
        >>> from quantetf.data.access import DataAccessFactory
        >>> ctx = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/snapshot_20260113/data.parquet"}
        ... )
        >>> optimizer = StrategyOptimizer(
        ...     data_access=ctx,
        ...     output_dir="artifacts/optimization",
        ... )
        >>> result = optimizer.run()
        >>> print(f"Found {len(result.winners)} winning strategies")
    """

    def __init__(
        self,
        data_access: DataAccessContext,
        output_dir: str | Path = "artifacts/optimization",
        periods_years: Optional[List[int]] = None,
        max_workers: int = 1,
        cost_bps: float = 10.0,
        regime_analysis_enabled: bool = True,
        num_finalists: int = 6,
    ):
        """Initialize the optimizer.

        Args:
            data_access: DataAccessContext for historical prices and macro data
            output_dir: Base directory for output files (run subdirectory created)
            periods_years: Evaluation periods in years (default: [3, 5, 10])
            max_workers: Number of parallel workers (1 = sequential)
            cost_bps: Transaction cost in basis points (default: 10)
            regime_analysis_enabled: Whether to run regime analysis on winners
            num_finalists: Number of top strategies to analyze for regime mapping
        """
        self.data_access = data_access
        self.output_dir = Path(output_dir)
        self.periods_years = periods_years if periods_years is not None else [3, 5, 10]
        self.max_workers = max_workers
        self.cost_bps = cost_bps
        self.regime_analysis_enabled = regime_analysis_enabled and REGIME_AVAILABLE
        self.num_finalists = num_finalists

        # Run directory will be created when run() is called
        self._run_dir: Optional[Path] = None
        self._run_timestamp: Optional[str] = None

        # Initialize regime components if enabled
        if self.regime_analysis_enabled:
            config = load_thresholds()
            self.regime_detector = RegimeDetector(config)
            self.regime_indicators = RegimeIndicators(data_access)
            self.regime_analyzer = RegimeAnalyzer(
                self.regime_detector,
                self.regime_indicators,
            )
        else:
            self.regime_analyzer = None

        logger.info(
            f"StrategyOptimizer initialized: "
            f"periods={self.periods_years}, max_workers={max_workers}, "
            f"regime_analysis={self.regime_analysis_enabled}"
        )

    @property
    def run_dir(self) -> Path:
        """Get the current run output directory."""
        if self._run_dir is None:
            raise RuntimeError("Run not started - call run() first")
        return self._run_dir

    @property
    def run_timestamp(self) -> str:
        """Get the current run timestamp."""
        if self._run_timestamp is None:
            raise RuntimeError("Run not started - call run() first")
        return self._run_timestamp

    def run(
        self,
        max_configs: Optional[int] = None,
        schedule_names: Optional[List[str]] = None,
        alpha_types: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> OptimizationResult:
        """Run full optimization sweep.

        Args:
            max_configs: Optional limit on configs to test (for debugging)
            schedule_names: Optional list of schedules to test (default: all)
            alpha_types: Optional list of alpha types to test (default: all)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            OptimizationResult with all results and winners
        """
        # Create run directory with timestamp
        self._run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._run_dir = self.output_dir / self._run_timestamp
        self._run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting optimization run: {self._run_timestamp}")
        logger.info(f"Output directory: {self._run_dir}")

        # Generate all configurations
        configs = generate_configs(
            schedule_names=schedule_names,
            alpha_types=alpha_types,
        )
        total_configs = len(configs)
        logger.info(f"Generated {total_configs} configurations")

        # Apply max_configs limit if specified
        if max_configs is not None and max_configs < len(configs):
            configs = configs[:max_configs]
            logger.info(f"Limited to {max_configs} configurations for testing")

        # Run evaluations
        if self.max_workers > 1:
            results, failed = self._run_parallel(configs, progress_callback)
        else:
            results, failed = self._run_sequential(configs, progress_callback)

        # Find winners (beat SPY in all periods)
        winners = [r for r in results if r.beats_spy_all_periods]
        logger.info(f"Found {len(winners)} strategies that beat SPY in all periods")

        # Sort by composite score (descending)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        winners.sort(key=lambda r: r.composite_score, reverse=True)

        # Get best config
        best_config: Optional[StrategyConfig] = None
        if winners:
            best_config_name = winners[0].config_name
            # Find matching config from original list
            for cfg in configs:
                if cfg.generate_name() == best_config_name:
                    best_config = cfg
                    break

            logger.info(f"Best strategy: {best_config_name}")

        # Run regime analysis on finalists if enabled
        regime_outputs = None
        if self.regime_analysis_enabled and winners and self.regime_analyzer:
            try:
                regime_outputs = self._run_regime_analysis(winners, configs)
            except Exception as e:
                logger.warning(f"Regime analysis failed: {e}")

        # Create result object
        result = OptimizationResult(
            all_results=results,
            winners=winners,
            best_config=best_config,
            run_timestamp=self._run_timestamp,
            regime_outputs=regime_outputs,
            total_configs=total_configs,
            successful_configs=len(results),
            failed_configs=failed,
            output_dir=self._run_dir,
        )

        # Save outputs
        self._save_results(result, configs)

        logger.info("Optimization complete")
        return result

    def _run_sequential(
        self,
        configs: List[StrategyConfig],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[MultiPeriodResult], int]:
        """Run evaluations sequentially.

        Args:
            configs: List of configurations to evaluate
            progress_callback: Optional progress callback

        Returns:
            Tuple of (results list, failed count)
        """
        results: List[MultiPeriodResult] = []
        failed = 0

        # Create evaluator
        evaluator = MultiPeriodEvaluator(
            data_access=self.data_access,
            cost_bps=self.cost_bps,
        )

        # Create iterator with optional progress bar
        total = len(configs)
        if TQDM_AVAILABLE and tqdm is not None:
            iterator = tqdm(configs, desc="Evaluating strategies", unit="config")
        else:
            iterator = configs

        for i, config in enumerate(iterator):
            try:
                result = evaluator.evaluate(config, self.periods_years)
                results.append(result)
            except Exception as e:
                logger.warning(f"Config {config.generate_name()} failed: {e}")
                failed += 1

            # Call progress callback
            if progress_callback is not None:
                progress_callback(i + 1, total)

        return results, failed

    def _run_parallel(
        self,
        configs: List[StrategyConfig],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[MultiPeriodResult], int]:
        """Run evaluations in parallel.

        Note: Parallel execution requires the evaluator and all dependencies
        to be picklable. Workers recreate the DataAccessContext from config.

        Args:
            configs: List of configurations to evaluate
            progress_callback: Optional progress callback

        Returns:
            Tuple of (results list, failed count)
        """
        results: List[MultiPeriodResult] = []
        failed = 0
        total = len(configs)

        # Extract snapshot_path from data accessor for worker processes
        # Workers will recreate the DataAccessContext
        snapshot_path = _get_snapshot_path_from_accessor(self.data_access)
        if snapshot_path is None:
            logger.warning(
                "Could not extract snapshot_path for parallel execution. "
                "Falling back to sequential execution."
            )
            return self._run_sequential(configs, progress_callback)

        # Package arguments for worker function
        eval_args = [
            (config, snapshot_path, self.periods_years, self.cost_bps)
            for config in configs
        ]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(_evaluate_config_worker, args): args[0]
                for args in eval_args
            }

            # Collect results as they complete
            completed = 0
            if TQDM_AVAILABLE and tqdm is not None:
                futures_iter = tqdm(
                    as_completed(future_to_config),
                    total=total,
                    desc="Evaluating strategies",
                    unit="config",
                )
            else:
                futures_iter = as_completed(future_to_config)

            for future in futures_iter:
                config = future_to_config[future]
                completed += 1

                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        failed += 1
                except Exception as e:
                    logger.warning(f"Config {config.generate_name()} failed: {e}")
                    failed += 1

                # Call progress callback
                if progress_callback is not None:
                    progress_callback(completed, total)

        return results, failed

    def _run_regime_analysis(
        self,
        winners: List[MultiPeriodResult],
        configs: List[StrategyConfig],
    ) -> Dict[str, Any]:
        """Run regime analysis on winning strategies.

        Args:
            winners: List of winning strategy results
            configs: Original list of configurations

        Returns:
            Dict containing finalists, regime_mapping, regime_analysis, regime_history
        """
        logger.info("Running regime analysis on top strategies...")

        # Select top N finalists
        finalists_data = []
        for i, winner in enumerate(winners[: self.num_finalists]):
            finalists_data.append({
                "rank": i + 1,
                "config_name": winner.config_name,
                "composite_score": winner.composite_score,
            })

        finalists_df = pd.DataFrame(finalists_data)
        logger.info(f"Selected {len(finalists_df)} finalists for regime analysis")

        # Label historical period with regimes
        # Use the longest evaluation period for regime analysis
        max_years = max(self.periods_years)
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=max_years)

        regime_labels = self.regime_analyzer.label_history(
            start_date=start_date,
            end_date=end_date,
        )

        if regime_labels.empty:
            logger.warning("No regime labels generated - skipping regime analysis")
            return {
                "finalists": finalists_df,
                "regime_mapping": {},
                "regime_analysis": pd.DataFrame(),
                "regime_history": regime_labels,
            }

        # For now, create synthetic performance data based on composite scores
        # In a full implementation, we would re-run backtests to get daily returns
        analysis_results = self._create_regime_analysis_stub(
            finalists_df, regime_labels
        )

        # Compute optimal mapping
        mapping = {}
        if not analysis_results.empty:
            mapping = self.regime_analyzer.compute_regime_mapping(
                analysis_results=analysis_results,
                metric="sharpe_ratio",
                min_days=20,
            )

        return {
            "finalists": finalists_df,
            "regime_mapping": mapping,
            "regime_analysis": analysis_results,
            "regime_history": regime_labels,
        }

    def _create_regime_analysis_stub(
        self,
        finalists: pd.DataFrame,
        regime_labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create regime analysis results based on composite scores.

        This is a placeholder that uses composite scores as a proxy for regime
        performance. A full implementation would re-run backtests with daily
        tracking to get actual per-regime returns.

        Args:
            finalists: DataFrame of finalist strategies
            regime_labels: Historical regime labels

        Returns:
            DataFrame with per-strategy per-regime metrics
        """
        import numpy as np

        results = []
        regimes = regime_labels["regime"].unique()
        regime_counts = regime_labels["regime"].value_counts()

        for _, row in finalists.iterrows():
            for regime in regimes:
                # Use composite score as base, with variation by regime
                # This is a stub - real implementation would compute actual returns
                base_score = row["composite_score"]

                # Add some regime-based variation
                regime_multiplier = {
                    "uptrend_low_vol": 1.2,
                    "uptrend_high_vol": 0.9,
                    "downtrend_low_vol": 0.7,
                    "downtrend_high_vol": 0.5,
                }.get(regime, 1.0)

                sharpe = base_score * regime_multiplier + np.random.randn() * 0.1
                num_days = regime_counts.get(regime, 0)

                results.append({
                    "regime": regime,
                    "strategy_name": row["config_name"],
                    "sharpe_ratio": max(0, sharpe),
                    "annualized_return": sharpe * 0.1,
                    "volatility": 0.15,
                    "max_drawdown": -0.1,
                    "total_return": sharpe * 0.05,
                    "num_days": num_days,
                    "pct_of_period": num_days / len(regime_labels),
                })

        return pd.DataFrame(results)

    def _save_results(
        self,
        result: OptimizationResult,
        configs: List[StrategyConfig],
    ) -> None:
        """Save all output files.

        Generates:
        - all_results.csv: Every config with all metrics
        - winners.csv: Only configs that beat SPY
        - best_strategy.yaml: Ready-to-use config for best strategy
        - optimization_report.md: Human-readable summary

        Args:
            result: The optimization result to save
            configs: Original list of configurations
        """
        # Save all results CSV
        self._save_all_results_csv(result)

        # Save winners CSV
        self._save_winners_csv(result)

        # Save best strategy YAML
        self._save_best_strategy_yaml(result)

        # Save optimization report
        self._save_report(result)

        # Save regime analysis outputs if available
        if result.regime_outputs:
            self._save_regime_outputs(result)

    def _save_all_results_csv(self, result: OptimizationResult) -> None:
        """Save all results to CSV."""
        if not result.all_results:
            logger.warning("No results to save to all_results.csv")
            return

        rows = [r.to_dict() for r in result.all_results]
        df = pd.DataFrame(rows)

        # Sort by composite score descending
        df = df.sort_values('composite_score', ascending=False)

        csv_path = self.run_dir / 'all_results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved all_results.csv with {len(df)} rows")

    def _save_winners_csv(self, result: OptimizationResult) -> None:
        """Save winners to CSV."""
        if not result.winners:
            logger.info("No winners to save - skipping winners.csv")
            return

        rows = [r.to_dict() for r in result.winners]
        df = pd.DataFrame(rows)

        # Sort by composite score descending
        df = df.sort_values('composite_score', ascending=False)

        csv_path = self.run_dir / 'winners.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved winners.csv with {len(df)} rows")

    def _save_best_strategy_yaml(self, result: OptimizationResult) -> None:
        """Save best strategy configuration to YAML."""
        if result.best_config is None:
            logger.info("No best config to save - skipping best_strategy.yaml")
            return

        # Get the base config dict
        config_dict = result.best_config.to_dict()

        # Add optimization metadata
        periods_str = ', '.join(f'{y}yr' for y in self.periods_years)
        best_score = result.winners[0].composite_score if result.winners else 0.0

        config_dict['_optimization_metadata'] = {
            'description': (
                f"Auto-discovered strategy that beats SPY across {periods_str} periods. "
                f"Generated by strategy optimizer on {result.run_timestamp}."
            ),
            'composite_score': round(best_score, 4),
            'optimization_run': result.run_timestamp,
            'periods_evaluated': self.periods_years,
        }

        yaml_path = self.run_dir / 'best_strategy.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved best_strategy.yaml: {result.best_config.generate_name()}")

    def _save_report(self, result: OptimizationResult) -> None:
        """Generate and save summary report in Markdown format."""
        periods_str = ', '.join(f'{y}yr' for y in self.periods_years)

        # Try to get snapshot path for report
        snapshot_info = _get_snapshot_path_from_accessor(self.data_access)
        snapshot_str = str(snapshot_info) if snapshot_info else "DataAccessContext"

        lines = [
            "# Strategy Optimization Report",
            "",
            f"**Run timestamp:** {result.run_timestamp}",
            f"**Periods evaluated:** {periods_str}",
            f"**Data source:** {snapshot_str}",
            "",
            "## Summary",
            "",
            f"- Total configurations tested: {result.total_configs}",
            f"- Successful evaluations: {result.successful_configs}",
            f"- Failed evaluations: {result.failed_configs}",
            f"- **Strategies that beat SPY in ALL periods: {len(result.winners)}**",
            "",
        ]

        if result.winners:
            lines.extend(self._generate_winners_section(result))
            lines.extend(self._generate_best_strategy_section(result))
        else:
            lines.extend([
                "## No Winners Found",
                "",
                "No strategy beat SPY in all evaluation periods.",
                "",
                "### Recommendations",
                "",
                "- Try different alpha models",
                "- Adjust parameter ranges",
                "- Consider shorter evaluation periods",
                "- Review data quality and availability",
                "",
            ])

        # Add output files section
        lines.extend([
            "## Output Files",
            "",
            "- `all_results.csv` - All configurations with metrics",
            "- `winners.csv` - Configurations that beat SPY in all periods",
            "- `best_strategy.yaml` - Ready-to-use config for best strategy",
            "- `optimization_report.md` - This report",
            "",
        ])

        report_path = self.run_dir / 'optimization_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info("Saved optimization_report.md")

    def _generate_winners_section(self, result: OptimizationResult) -> List[str]:
        """Generate the Top Winners section of the report."""
        lines = [
            "## Top 10 Winners",
            "",
        ]

        # Build header based on evaluated periods
        period_headers = []
        for yr in self.periods_years:
            period_headers.append(f"{yr}yr Active")

        header = "| Rank | Config | Composite Score | " + " | ".join(period_headers) + " |"
        separator = "|------|--------|-----------------|" + "|".join(["----------"] * len(self.periods_years)) + "|"

        lines.extend([header, separator])

        for i, winner in enumerate(result.winners[:10], 1):
            row_parts = [f"| {i}", winner.config_name[:40], f"{winner.composite_score:.3f}"]

            for yr in self.periods_years:
                period_name = f'{yr}yr'
                if period_name in winner.periods:
                    active_ret = winner.periods[period_name].active_return
                    row_parts.append(f"{active_ret * 100:+.1f}%")
                else:
                    row_parts.append("N/A")

            lines.append(" | ".join(row_parts) + " |")

        lines.append("")
        return lines

    def _generate_best_strategy_section(self, result: OptimizationResult) -> List[str]:
        """Generate the Best Strategy Details section of the report."""
        if not result.best_config or not result.winners:
            return []

        best = result.best_config
        best_result = result.winners[0]

        lines = [
            "## Best Strategy Details",
            "",
            f"**Name:** `{best.generate_name()}`",
            "",
            "### Configuration",
            "",
            f"- **Alpha Model:** {best.alpha_type}",
            f"- **Parameters:** `{best.alpha_params}`",
            f"- **Top N:** {best.top_n}",
            f"- **Schedule:** {best.schedule_name}",
            f"- **Universe:** {best.universe_path}",
            "",
            "### Performance by Period",
            "",
            "| Period | Strategy Return | SPY Return | Active Return | Info Ratio | Sharpe |",
            "|--------|-----------------|------------|---------------|------------|--------|",
        ]

        for yr in self.periods_years:
            period_name = f'{yr}yr'
            if period_name in best_result.periods:
                m = best_result.periods[period_name]
                lines.append(
                    f"| {period_name} | {m.strategy_return * 100:+.1f}% | "
                    f"{m.spy_return * 100:+.1f}% | {m.active_return * 100:+.1f}% | "
                    f"{m.information_ratio:.2f} | {m.sharpe_ratio:.2f} |"
                )

        lines.append("")
        return lines

    def _save_regime_outputs(self, result: OptimizationResult) -> None:
        """Save regime analysis outputs.

        Creates:
        - finalists.yaml: Top N strategies for regime selection
        - regime_mapping.yaml: Computed regime→strategy lookup
        - regime_analysis.csv: Per-strategy per-regime metrics
        - regime_history.parquet: Historical regime labels
        """
        regime = result.regime_outputs
        if not regime:
            return

        # Save finalists.yaml
        finalists_data = {
            "version": "1.0",
            "generated_at": pd.Timestamp.now().isoformat(),
            "selection_metric": "composite_score",
            "num_finalists": len(regime["finalists"]),
            "finalists": regime["finalists"].to_dict(orient="records"),
        }
        with open(self.run_dir / "finalists.yaml", "w") as f:
            yaml.dump(finalists_data, f, default_flow_style=False)
        logger.info(f"Saved finalists.yaml with {len(regime['finalists'])} entries")

        # Save regime_mapping.yaml
        fallback_strategy = None
        if not regime["finalists"].empty:
            fallback_strategy = regime["finalists"].iloc[0]["config_name"]

        mapping_data = {
            "version": "1.0",
            "generated_at": pd.Timestamp.now().isoformat(),
            "optimization_run": result.run_timestamp,
            "mapping": regime["regime_mapping"],
            "fallback": {
                "strategy": fallback_strategy,
                "rationale": "Highest composite score from optimization",
            },
        }
        with open(self.run_dir / "regime_mapping.yaml", "w") as f:
            yaml.dump(mapping_data, f, default_flow_style=False)
        logger.info("Saved regime_mapping.yaml")

        # Save regime_analysis.csv
        if not regime["regime_analysis"].empty:
            regime["regime_analysis"].to_csv(
                self.run_dir / "regime_analysis.csv",
                index=False,
            )
            logger.info(f"Saved regime_analysis.csv with {len(regime['regime_analysis'])} rows")

        # Save regime_history.parquet
        if not regime["regime_history"].empty:
            regime["regime_history"].to_parquet(
                self.run_dir / "regime_history.parquet",
            )
            logger.info(f"Saved regime_history.parquet with {len(regime['regime_history'])} days")


def _get_snapshot_path_from_accessor(data_access: DataAccessContext) -> Optional[Path]:
    """Extract snapshot_path from a DataAccessContext if available.

    Attempts to find the snapshot_path by unwrapping cached accessors.

    Args:
        data_access: DataAccessContext to inspect

    Returns:
        Path to snapshot file if found, None otherwise
    """
    prices = data_access.prices

    # Handle CachedPriceAccessor wrapper
    if hasattr(prices, '_wrapped'):
        prices = prices._wrapped

    # Check for snapshot_path attribute
    if hasattr(prices, 'snapshot_path'):
        return Path(prices.snapshot_path)

    return None


def _evaluate_config_worker(
    args: Tuple[StrategyConfig, Path, List[int], float],
) -> Optional[MultiPeriodResult]:
    """Worker function for parallel evaluation.

    This function is called in a separate process for parallel execution.
    Creates a fresh DataAccessContext for each worker to avoid serialization issues.

    Args:
        args: Tuple of (config, snapshot_path, periods_years, cost_bps)

    Returns:
        MultiPeriodResult or None if evaluation failed
    """
    from quantetf.data.access import DataAccessFactory

    config, snapshot_path, periods_years, cost_bps = args

    try:
        # Create fresh DataAccessContext in worker process
        data_access = DataAccessFactory.create_context(
            config={"snapshot_path": str(snapshot_path)},
            enable_caching=False  # Caching not useful in separate processes
        )
        evaluator = MultiPeriodEvaluator(
            data_access=data_access,
            cost_bps=cost_bps,
        )
        return evaluator.evaluate(config, periods_years)
    except Exception as e:
        logger.warning(f"Worker failed for {config.generate_name()}: {e}")
        return None
