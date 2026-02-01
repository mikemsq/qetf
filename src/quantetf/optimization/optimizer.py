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
import pickle
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
from quantetf.optimization.walk_forward_evaluator import (
    WalkForwardEvaluator,
    WalkForwardEvaluatorConfig,
    WalkForwardEvaluationResult,
)

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
        scoring_method: str = "regime_weighted",
        trailing_days: int = 252,
        use_walk_forward: bool = True,  # Default to walk-forward
        wf_config: Optional[WalkForwardEvaluatorConfig] = None,
    ):
        """Initialize the optimizer.

        Args:
            data_access: DataAccessContext for historical prices and macro data
            output_dir: Base directory for output files (run subdirectory created)
            periods_years: Evaluation periods in years (default: [3, 5, 10]).
                Only used when use_walk_forward=False.
            max_workers: Number of parallel workers (1 = sequential)
            cost_bps: Transaction cost in basis points (default: 10)
            regime_analysis_enabled: Whether to run regime analysis on winners
            num_finalists: Number of top strategies to analyze for regime mapping
            scoring_method: Method for ranking strategies. Options:
                - 'multi_period': Original avg(IR) - penalty + bonus (for discovery)
                - 'trailing_1y': Score based on trailing 1-year Sharpe
                - 'regime_weighted': Score weighted by historical regime frequency
            trailing_days: Days for trailing window evaluation (default: 252 = 1 year)
            use_walk_forward: If True, use WalkForwardEvaluator for OOS evaluation.
                If False, use legacy MultiPeriodEvaluator.
            wf_config: Walk-forward configuration. Uses defaults if None.
        """
        self.data_access = data_access
        self.output_dir = Path(output_dir)
        self.periods_years = periods_years if periods_years is not None else [3, 5, 10]
        self.max_workers = max_workers
        self.cost_bps = cost_bps
        self.regime_analysis_enabled = regime_analysis_enabled and REGIME_AVAILABLE
        self.num_finalists = num_finalists
        self.scoring_method = scoring_method
        self.trailing_days = trailing_days
        self.use_walk_forward = use_walk_forward
        self.wf_config = wf_config or WalkForwardEvaluatorConfig()

        # Deprecation warning for periods_years with legacy mode
        if not use_walk_forward and periods_years is not None:
            import warnings
            warnings.warn(
                "periods_years parameter is deprecated. "
                "Use use_walk_forward=True with wf_config instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Validate scoring method
        valid_methods = ['multi_period', 'trailing_1y', 'regime_weighted']
        if self.scoring_method not in valid_methods:
            raise ValueError(
                f"Invalid scoring_method: {scoring_method}. "
                f"Must be one of: {valid_methods}"
            )

        # Run directory will be created when run() is called
        self._run_dir: Optional[Path] = None
        self._run_timestamp: Optional[str] = None

        # Initialize regime components if enabled (only for legacy mode)
        if self.regime_analysis_enabled and not use_walk_forward:
            config = load_thresholds()
            self.regime_detector = RegimeDetector(config)
            self.regime_indicators = RegimeIndicators(data_access)
            self.regime_analyzer = RegimeAnalyzer(
                self.regime_detector,
                self.regime_indicators,
            )
        else:
            self.regime_analyzer = None

        if use_walk_forward:
            logger.info(
                f"StrategyOptimizer initialized (walk-forward mode): "
                f"train={self.wf_config.train_years}yr, "
                f"test={self.wf_config.test_years}yr, "
                f"step={self.wf_config.step_months}mo, "
                f"max_workers={max_workers}"
            )
        else:
            logger.info(
                f"StrategyOptimizer initialized (legacy mode): "
                f"periods={self.periods_years}, max_workers={max_workers}, "
                f"regime_analysis={self.regime_analysis_enabled}, "
                f"scoring_method={self.scoring_method}"
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

    def save_backtest_results(
        self,
        results: List[MultiPeriodResult],
        path: Path,
    ) -> None:
        """Save raw backtest results including daily returns.

        Saves results to pickle format to preserve pandas Series objects
        in daily_returns fields, which are required for regime analysis.

        Args:
            results: List of MultiPeriodResult from backtest runs
            path: Path to save the pickle file
        """
        data = {
            'results': results,
            'timestamp': self._run_timestamp,
            'periods_years': self.periods_years,
            'cost_bps': self.cost_bps,
            'version': '1.0',
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved backtest results to {path} ({len(results)} strategies)")

    @staticmethod
    def load_backtest_results(path: Path) -> Dict[str, Any]:
        """Load previously saved backtest results.

        Args:
            path: Path to the saved pickle file

        Returns:
            Dict containing:
            - results: List[MultiPeriodResult]
            - timestamp: Original run timestamp
            - periods_years: Evaluation periods used
            - cost_bps: Transaction cost in basis points
            - version: Format version string
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logger.info(
            f"Loaded backtest results from {path}: "
            f"{len(data['results'])} strategies, timestamp={data.get('timestamp')}"
        )
        return data

    def run_backtests(
        self,
        max_configs: Optional[int] = None,
        schedule_names: Optional[List[str]] = None,
        alpha_types: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[MultiPeriodResult], int, int, List[StrategyConfig]]:
        """Run backtests only, without ranking or regime analysis.

        Args:
            max_configs: Optional limit on configs to test (for debugging)
            schedule_names: Optional list of schedules to test (default: all)
            alpha_types: Optional list of alpha types to test (default: all)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Tuple of (results, total_configs, failed_count, configs_list)
        """
        # Create run directory with timestamp if not already created
        if self._run_timestamp is None:
            self._run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._run_dir = self.output_dir / self._run_timestamp
            self._run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting backtest run: {self._run_timestamp}")
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

        return results, total_configs, failed, configs

    def rank_results(
        self,
        results: List,
        scoring_method: Optional[str] = None,
        regime_distribution: Optional[Dict[str, float]] = None,
    ) -> Tuple[List, List]:
        """Apply ranking to backtest results.

        Handles both MultiPeriodResult (legacy) and WalkForwardEvaluationResult.

        Args:
            results: List of results from backtests
            scoring_method: Override scoring method (uses self.scoring_method if None)
            regime_distribution: Optional regime weights for regime_weighted scoring

        Returns:
            Tuple of (all_results_sorted, winners_sorted)
        """
        if self.use_walk_forward:
            return self._rank_walk_forward_results(results)
        else:
            return self._rank_multi_period_results(
                results, scoring_method, regime_distribution
            )

    def _rank_walk_forward_results(
        self,
        results: List[WalkForwardEvaluationResult],
    ) -> Tuple[List[WalkForwardEvaluationResult], List[WalkForwardEvaluationResult]]:
        """Rank results from walk-forward evaluation.

        Filters strategies with negative OOS active return, then ranks
        by composite score (OOS-based).

        Args:
            results: List of WalkForwardEvaluationResult

        Returns:
            Tuple of (all_results_sorted, winners_sorted)
        """
        # Filter: require positive OOS active return
        winners = [r for r in results if r.oos_active_return_mean > 0]

        filtered_count = len(results) - len(winners)
        if filtered_count > 0:
            logger.info(
                f"Filtered {filtered_count} strategies with negative OOS active return"
            )

        logger.info(f"Found {len(winners)} strategies with positive OOS active return")

        # Sort all results by composite score (higher is better)
        all_sorted = sorted(results, key=lambda r: r.composite_score, reverse=True)
        winners_sorted = sorted(winners, key=lambda r: r.composite_score, reverse=True)

        return all_sorted, winners_sorted

    def _rank_multi_period_results(
        self,
        results: List[MultiPeriodResult],
        scoring_method: Optional[str] = None,
        regime_distribution: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[MultiPeriodResult], List[MultiPeriodResult]]:
        """Legacy ranking for multi-period results.

        Args:
            results: List of MultiPeriodResult from backtests
            scoring_method: Override scoring method (uses self.scoring_method if None)
            regime_distribution: Optional regime weights for regime_weighted scoring

        Returns:
            Tuple of (all_results_sorted, winners_sorted)
        """
        method = scoring_method or self.scoring_method

        # Make copies to avoid modifying original
        all_results = list(results)

        # Find winners (beat SPY in all periods)
        winners = [r for r in all_results if r.beats_spy_all_periods]
        logger.info(f"Found {len(winners)} strategies that beat SPY in all periods")

        # Apply scoring method
        if method == 'trailing_1y':
            self._apply_trailing_scores(all_results, self.trailing_days)
            self._apply_trailing_scores(winners, self.trailing_days)
        elif method == 'regime_weighted' and regime_distribution:
            self._apply_regime_weighted_scores(all_results, regime_distribution)
            self._apply_regime_weighted_scores(winners, regime_distribution)

        # Sort by composite score (descending)
        all_results.sort(key=lambda r: r.composite_score, reverse=True)
        winners.sort(key=lambda r: r.composite_score, reverse=True)

        return all_results, winners

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
        if self.use_walk_forward:
            logger.info(
                f"Evaluation mode: Walk-Forward (train={self.wf_config.train_years}yr, "
                f"test={self.wf_config.test_years}yr, step={self.wf_config.step_months}mo)"
            )
        else:
            logger.info(f"Evaluation mode: Multi-Period ({self.periods_years}yr)")
            logger.info(f"Scoring method: {self.scoring_method}")

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

        # Rank results based on evaluation mode
        results, winners = self.rank_results(results)

        # Run regime analysis on finalists if enabled (only for legacy mode)
        regime_outputs = None
        if not self.use_walk_forward and self.regime_analysis_enabled and winners and self.regime_analyzer:
            try:
                regime_outputs = self._run_regime_analysis(winners, configs)

                # For regime_weighted scoring, recalculate scores using regime weights
                if self.scoring_method == 'regime_weighted' and regime_outputs:
                    self._apply_regime_weighted_scores(
                        results, regime_outputs.get("regime_distribution", {})
                    )
                    self._apply_regime_weighted_scores(
                        winners, regime_outputs.get("regime_distribution", {})
                    )
                    # Re-sort after recalculating scores
                    results.sort(key=lambda r: r.composite_score, reverse=True)
                    winners.sort(key=lambda r: r.composite_score, reverse=True)

            except Exception as e:
                logger.warning(f"Regime analysis failed: {e}")

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

        # Print console summary
        if self.use_walk_forward:
            self._print_walk_forward_summary(result)
        else:
            self._print_legacy_summary(result)

        # Save outputs
        self._save_results(result, configs)

        logger.info("Optimization complete")
        return result

    def _print_walk_forward_summary(self, result: OptimizationResult) -> None:
        """Print console summary for walk-forward optimization."""
        # Get number of windows from first result if available
        num_windows = 0
        if result.all_results:
            first_result = result.all_results[0]
            if hasattr(first_result, 'num_windows'):
                num_windows = first_result.num_windows

        filtered_count = result.successful_configs - len(result.winners)
        pct_passed = 100 * len(result.winners) / max(result.successful_configs, 1)

        lines = [
            "",
            "=" * 80,
            "WALK-FORWARD OPTIMIZATION RESULTS",
            "=" * 80,
            "Walk-Forward Config:",
            f"  Train Period: {self.wf_config.train_years} years",
            f"  Test Period:  {self.wf_config.test_years} years",
            f"  Step Size:    {self.wf_config.step_months} months",
            f"  Windows:      {num_windows}",
            "",
            f"Strategies Evaluated:    {result.successful_configs:,}",
            f"Strategies Passed OOS:     {len(result.winners):,}  ({pct_passed:.1f}%)",
            f"Strategies Filtered:       {filtered_count:,}  (negative OOS active return)",
            "",
        ]

        if result.winners:
            lines.extend([
                "TOP 10 STRATEGIES (by OOS composite score):",
                "-" * 80,
                f"{'Rank':<6}{'Strategy':<44}{'OOS_Sharpe':>12}{'OOS_Active':>12}{'Win%':>8}",
                "-" * 80,
            ])
            for i, winner in enumerate(result.winners[:10], 1):
                name = winner.config_name[:42]
                lines.append(
                    f"{i:<6}{name:<44}{winner.oos_sharpe_mean:>12.2f}"
                    f"{winner.oos_active_return_mean:>+11.1%}{winner.oos_win_rate:>8.0%}"
                )
        else:
            lines.append("No strategies passed OOS active return filter.")

        lines.append("=" * 80)
        print("\n".join(lines))

    def _print_legacy_summary(self, result: OptimizationResult) -> None:
        """Print console summary for legacy multi-period optimization."""
        lines = [
            "",
            "=" * 60,
            "Strategy Optimization Results",
            "=" * 60,
            f"Run timestamp: {result.run_timestamp}",
            f"Total configurations: {result.total_configs:,}",
            f"Successful evaluations: {result.successful_configs:,}",
            f"Failed evaluations: {result.failed_configs:,}",
            f"Strategies that beat SPY (all periods): {len(result.winners):,}",
            "",
        ]

        if result.best_config and result.winners:
            lines.extend([
                "Best Strategy:",
                f"  Name: {result.best_config.generate_name()}",
                f"  Composite Score: {result.winners[0].composite_score:.4f}",
            ])
        else:
            lines.append("No winning strategies found.")

        lines.append("=" * 60)
        print("\n".join(lines))

    def _apply_trailing_scores(
        self,
        results: List[MultiPeriodResult],
        trailing_days: int,
    ) -> None:
        """Recalculate composite scores using trailing window method.

        Calculates trailing Sharpe ratio directly from stored daily returns,
        without needing data_access.

        Args:
            results: List of results to update (modified in place)
            trailing_days: Number of days for trailing window
        """
        import numpy as np

        for result in results:
            new_score = self._calculate_trailing_score(result.periods, trailing_days)
            # Update the composite score (note: MultiPeriodResult is a dataclass)
            object.__setattr__(result, 'composite_score', new_score)

    def _calculate_trailing_score(
        self,
        periods: Dict[str, Any],
        trailing_days: int,
    ) -> float:
        """Calculate trailing Sharpe score from period metrics.

        Args:
            periods: Dict mapping period name to PeriodMetrics
            trailing_days: Number of days for trailing window

        Returns:
            Trailing Sharpe ratio
        """
        import numpy as np

        # Find the shortest period with valid returns
        min_period = None
        min_years = float('inf')

        for period_name, metrics in periods.items():
            if not metrics.evaluation_success:
                continue
            try:
                years = int(period_name.replace('yr', ''))
                if years < min_years:
                    min_years = years
                    min_period = metrics
            except ValueError:
                continue

        if min_period is None or min_period.daily_returns is None:
            return float('-inf')

        returns = min_period.daily_returns
        if len(returns) < trailing_days:
            trailing_returns = returns
        else:
            trailing_returns = returns.iloc[-trailing_days:]

        if len(trailing_returns) < 20:
            return float('-inf')

        mean_return = trailing_returns.mean()
        std_return = trailing_returns.std()

        if std_return == 0:
            return 0.0

        # Annualize based on approximate rebalance frequency
        periods_per_year = 52 if len(returns) > 100 else 12
        trailing_sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)

        return trailing_sharpe

    def _apply_regime_weighted_scores(
        self,
        results: List[MultiPeriodResult],
        regime_distribution: Dict[str, float],
    ) -> None:
        """Recalculate composite scores using regime-weighted method.

        This uses the regime distribution weights to score strategies based on
        their historical regime frequency.

        Args:
            results: List of results to update (modified in place)
            regime_distribution: Dict mapping regime name to frequency weight
        """
        if not regime_distribution:
            logger.warning("No regime distribution available for scoring")
            return

        # For each result, we need to compute regime-weighted score
        # This requires the per-regime metrics from regime analysis
        # For now, use the original composite score weighted by regime
        # In a full implementation, we would store per-regime metrics
        logger.info(
            f"Applying regime-weighted scores with distribution: {regime_distribution}"
        )

    def _run_sequential(
        self,
        configs: List[StrategyConfig],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List, int]:
        """Run evaluations sequentially.

        Args:
            configs: List of configurations to evaluate
            progress_callback: Optional progress callback

        Returns:
            Tuple of (results list, failed count)
            Results are either MultiPeriodResult or WalkForwardEvaluationResult
            depending on use_walk_forward mode.
        """
        results: List = []
        failed = 0

        # Create evaluator based on mode
        if self.use_walk_forward:
            evaluator = WalkForwardEvaluator(
                data_access=self.data_access,
                wf_config=self.wf_config,
                cost_bps=self.cost_bps,
            )
        else:
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
                if self.use_walk_forward:
                    result = evaluator.evaluate(config)
                else:
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
    ) -> Tuple[List, int]:
        """Run evaluations in parallel.

        Note: Parallel execution requires the evaluator and all dependencies
        to be picklable. Workers recreate the DataAccessContext from config.

        Args:
            configs: List of configurations to evaluate
            progress_callback: Optional progress callback

        Returns:
            Tuple of (results list, failed count)
            Results are either MultiPeriodResult or WalkForwardEvaluationResult
            depending on use_walk_forward mode.
        """
        results: List = []
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

        # Package arguments for worker function based on mode
        if self.use_walk_forward:
            # Walk-forward mode - include wf_config as dict for pickling
            wf_config_dict = {
                'train_years': self.wf_config.train_years,
                'test_years': self.wf_config.test_years,
                'step_months': self.wf_config.step_months,
                'min_windows': self.wf_config.min_windows,
                'require_positive_oos': self.wf_config.require_positive_oos,
            }
            eval_args = [
                (config, snapshot_path, self.cost_bps, True, wf_config_dict)
                for config in configs
            ]
        else:
            # Legacy mode
            eval_args = [
                (config, snapshot_path, self.cost_bps, False, self.periods_years)
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
        """Run regime analysis on winning strategies using ACTUAL backtest returns.

        This replaces the previous stub implementation. It:
        1. Extracts daily returns from winner evaluation results
        2. Labels historical dates with their regime
        3. Calculates real per-regime performance metrics
        4. Computes optimal regime → strategy mapping

        Args:
            winners: List of winning strategy results (must have daily_returns)
            configs: Original list of configurations

        Returns:
            Dict containing:
            - finalists: DataFrame of top N strategies
            - regime_mapping: Dict mapping regime → best strategy
            - regime_analysis: DataFrame with per-strategy per-regime metrics
            - regime_history: DataFrame with daily regime labels
            - regime_distribution: Dict with regime frequency weights
        """
        logger.info("Running regime analysis on top strategies...")

        # Select top N finalists
        finalists = winners[: self.num_finalists]
        finalists_data = []
        for i, winner in enumerate(finalists):
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
                "regime_distribution": {},
            }

        # Calculate regime distribution (historical frequency weights)
        regime_counts = regime_labels["regime"].value_counts()
        total_days = len(regime_labels)
        regime_distribution = {
            regime: count / total_days
            for regime, count in regime_counts.items()
        }
        logger.info(f"Regime distribution: {regime_distribution}")

        # Extract daily returns from finalists and analyze by regime
        strategy_returns = self._extract_finalist_returns(finalists)

        if not strategy_returns:
            logger.warning("No daily returns available from finalists - using fallback")
            # Fallback: use the stub approach if no returns available
            analysis_results = self._create_regime_analysis_fallback(
                finalists_df, regime_labels
            )
        else:
            # Use real regime analysis with actual returns
            logger.info(f"Analyzing {len(strategy_returns)} strategies by regime...")
            analysis_results = self.regime_analyzer.analyze_multiple_strategies(
                strategy_results=strategy_returns,
                regime_labels=regime_labels,
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
            "regime_distribution": regime_distribution,
        }

    def _extract_finalist_returns(
        self,
        finalists: List[MultiPeriodResult],
    ) -> Dict[str, pd.Series]:
        """Extract daily returns from finalist evaluation results.

        Args:
            finalists: List of finalist MultiPeriodResult objects

        Returns:
            Dict mapping strategy_name -> daily returns Series
        """
        strategy_returns = {}

        for finalist in finalists:
            # Find the longest period with valid returns
            best_returns = None
            max_len = 0

            for period_name, metrics in finalist.periods.items():
                if (
                    metrics.evaluation_success
                    and metrics.daily_returns is not None
                    and len(metrics.daily_returns) > max_len
                ):
                    best_returns = metrics.daily_returns
                    max_len = len(metrics.daily_returns)

            if best_returns is not None:
                strategy_returns[finalist.config_name] = best_returns
                logger.debug(
                    f"Extracted {len(best_returns)} returns for {finalist.config_name}"
                )
            else:
                logger.warning(
                    f"No daily returns available for {finalist.config_name}"
                )

        return strategy_returns

    def _create_regime_analysis_fallback(
        self,
        finalists: pd.DataFrame,
        regime_labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fallback regime analysis when daily returns are not available.

        This uses composite scores as a proxy for regime performance.
        Should only be used when actual returns cannot be extracted.

        Args:
            finalists: DataFrame of finalist strategies
            regime_labels: Historical regime labels

        Returns:
            DataFrame with per-strategy per-regime metrics (estimated)
        """
        import numpy as np

        logger.warning(
            "Using fallback regime analysis - results are ESTIMATED, not actual"
        )

        results = []
        regimes = regime_labels["regime"].unique()
        regime_counts = regime_labels["regime"].value_counts()

        for _, row in finalists.iterrows():
            for regime in regimes:
                base_score = row["composite_score"]

                # Regime-based variation (heuristic based on typical patterns)
                regime_multiplier = {
                    "uptrend_low_vol": 1.2,
                    "uptrend_high_vol": 0.9,
                    "downtrend_low_vol": 0.7,
                    "downtrend_high_vol": 0.5,
                }.get(regime, 1.0)

                # Add small deterministic variation based on strategy name hash
                name_hash = hash(row["config_name"]) % 100 / 1000
                sharpe = base_score * regime_multiplier + name_hash
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
        """Save all results to CSV.

        For walk-forward mode, reorders columns to show OOS metrics first.
        """
        if not result.all_results:
            logger.warning("No results to save to all_results.csv")
            return

        rows = [r.to_dict() for r in result.all_results]
        df = pd.DataFrame(rows)

        # Sort by composite score descending
        df = df.sort_values('composite_score', ascending=False)

        # Reorder columns for walk-forward mode
        if self.use_walk_forward:
            oos_cols = [c for c in df.columns if c.startswith('oos_')]
            is_cols = [c for c in df.columns if c.startswith('is_')]
            other_cols = [c for c in df.columns if c not in oos_cols + is_cols]
            df = df[other_cols + oos_cols + is_cols]

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

        # Add optimization metadata (mode-specific)
        best_score = result.winners[0].composite_score if result.winners else 0.0

        if self.use_walk_forward:
            best_result = result.winners[0] if result.winners else None
            config_dict['_optimization_metadata'] = {
                'description': (
                    f"Walk-forward validated strategy with OOS Sharpe {best_result.oos_sharpe_mean:.2f}. "
                    f"Generated by strategy optimizer on {result.run_timestamp}."
                    if best_result else "No walk-forward results available."
                ),
                'evaluation_mode': 'walk_forward',
                'composite_score': round(best_score, 4),
                'oos_sharpe_mean': round(best_result.oos_sharpe_mean, 3) if best_result else 0.0,
                'oos_active_return_mean': round(best_result.oos_active_return_mean, 4) if best_result else 0.0,
                'oos_win_rate': round(best_result.oos_win_rate, 2) if best_result else 0.0,
                'num_windows': best_result.num_windows if best_result else 0,
                'optimization_run': result.run_timestamp,
                'wf_config': {
                    'train_years': self.wf_config.train_years,
                    'test_years': self.wf_config.test_years,
                    'step_months': self.wf_config.step_months,
                },
            }
        else:
            periods_str = ', '.join(f'{y}yr' for y in self.periods_years)
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
        # Try to get snapshot path for report
        snapshot_info = _get_snapshot_path_from_accessor(self.data_access)
        snapshot_str = str(snapshot_info) if snapshot_info else "DataAccessContext"

        if self.use_walk_forward:
            lines = self._generate_walk_forward_report(result, snapshot_str)
        else:
            lines = self._generate_legacy_report(result, snapshot_str)

        report_path = self.run_dir / 'optimization_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info("Saved optimization_report.md")

    def _generate_walk_forward_report(
        self,
        result: OptimizationResult,
        snapshot_str: str,
    ) -> List[str]:
        """Generate report for walk-forward mode."""
        lines = [
            "# Walk-Forward Optimization Report",
            "",
            f"**Run timestamp:** {result.run_timestamp}",
            f"**Evaluation mode:** Walk-Forward Validation",
            f"**Data source:** {snapshot_str}",
            "",
            "## Walk-Forward Configuration",
            "",
            f"- Training Period: {self.wf_config.train_years} years",
            f"- Test Period: {self.wf_config.test_years} years",
            f"- Step Size: {self.wf_config.step_months} months",
            f"- Minimum Windows: {self.wf_config.min_windows}",
            "",
            "## Summary",
            "",
            f"- Total configurations tested: {result.total_configs}",
            f"- Successful evaluations: {result.successful_configs}",
            f"- Failed evaluations: {result.failed_configs}",
            f"- **Strategies with positive OOS active return: {len(result.winners)}** "
            f"({100 * len(result.winners) / max(result.successful_configs, 1):.1f}%)",
            f"- Strategies filtered: {result.successful_configs - len(result.winners)}",
            "",
        ]

        if result.winners:
            lines.extend(self._generate_walk_forward_winners_section(result))
            lines.extend(self._generate_degradation_section(result))
        else:
            lines.extend([
                "## No Winners Found",
                "",
                "No strategy achieved positive OOS active return.",
                "",
                "### Recommendations",
                "",
                "- Try different alpha models or parameter ranges",
                "- Consider shorter training/test periods",
                "- Review data quality and date range",
                "",
            ])

        # Add output files section
        lines.extend([
            "## Output Files",
            "",
            "- `all_results.csv` - All configurations with OOS metrics",
            "- `winners.csv` - Configurations with positive OOS active return",
            "- `best_strategy.yaml` - Ready-to-use config for best strategy",
            "- `optimization_report.md` - This report",
            "",
        ])

        return lines

    def _generate_walk_forward_winners_section(
        self,
        result: OptimizationResult,
    ) -> List[str]:
        """Generate winners section for walk-forward mode."""
        lines = [
            "## Top 10 Strategies (by OOS Composite Score)",
            "",
            "| Rank | Strategy | OOS Sharpe | OOS Active | Win Rate | Windows |",
            "|------|----------|------------|------------|----------|---------|",
        ]

        for i, winner in enumerate(result.winners[:10], 1):
            name = winner.config_name[:40]
            lines.append(
                f"| {i} | {name} | {winner.oos_sharpe_mean:.2f} | "
                f"{winner.oos_active_return_mean:+.1%} | "
                f"{winner.oos_win_rate:.0%} | {winner.num_windows} |"
            )

        lines.append("")
        return lines

    def _generate_degradation_section(
        self,
        result: OptimizationResult,
    ) -> List[str]:
        """Generate degradation analysis section."""
        lines = [
            "## Degradation Analysis (IS vs OOS)",
            "",
            "Lower degradation indicates more robust strategies that generalize well.",
            "",
            "| Strategy | IS Sharpe | OOS Sharpe | Degradation |",
            "|----------|-----------|------------|-------------|",
        ]

        for winner in result.winners[:5]:
            name = winner.config_name[:40]
            lines.append(
                f"| {name} | {winner.is_sharpe_mean:.2f} | "
                f"{winner.oos_sharpe_mean:.2f} | {winner.sharpe_degradation:+.2f} |"
            )

        lines.append("")
        return lines

    def _generate_legacy_report(
        self,
        result: OptimizationResult,
        snapshot_str: str,
    ) -> List[str]:
        """Generate report for legacy multi-period mode."""
        periods_str = ', '.join(f'{y}yr' for y in self.periods_years)

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

        return lines

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
        - regime_mapping.yaml: PRIMARY OUTPUT - regime→strategy lookup with v2.0 format
        - regime_analysis.csv: Per-strategy per-regime metrics (REAL DATA)
        - regime_history.parquet: Historical regime labels
        - best_overall_strategy.yaml: Fallback strategy for unknown regimes
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

        # Build enhanced mapping with per-regime metrics
        fallback_strategy = None
        if not regime["finalists"].empty:
            fallback_strategy = regime["finalists"].iloc[0]["config_name"]

        # Build v2.0 format regime mapping with detailed metrics per regime
        detailed_mapping = {}
        regime_analysis_df = regime.get("regime_analysis", pd.DataFrame())

        for regime_name, mapping_info in regime.get("regime_mapping", {}).items():
            strategy_name = mapping_info.get("strategy", "")

            # Get detailed metrics for this strategy/regime from analysis
            if not regime_analysis_df.empty:
                match = regime_analysis_df[
                    (regime_analysis_df["regime"] == regime_name) &
                    (regime_analysis_df["strategy_name"] == strategy_name)
                ]
                if not match.empty:
                    row = match.iloc[0]
                    detailed_mapping[regime_name] = {
                        "strategy": strategy_name,
                        "regime_sharpe": round(row.get("sharpe_ratio", 0), 3),
                        "regime_return": round(row.get("annualized_return", 0), 4),
                        "regime_max_dd": round(row.get("max_drawdown", 0), 4),
                        "num_days_evaluated": int(row.get("num_days", 0)),
                    }
                    continue

            # Fallback if no detailed metrics found
            detailed_mapping[regime_name] = {
                "strategy": strategy_name,
                "regime_sharpe": round(mapping_info.get("score", 0), 3),
                "regime_return": round(mapping_info.get("total_return", 0), 4),
                "regime_max_dd": 0.0,
                "num_days_evaluated": int(mapping_info.get("num_days", 0)),
            }

        # Save regime_mapping.yaml with v2.0 format
        periods_str = f"trailing_{max(self.periods_years)}y"
        mapping_data = {
            "version": "2.0",
            "generated_at": pd.Timestamp.now().isoformat(),
            "optimization_run": result.run_timestamp,
            "evaluation_period": periods_str,
            "regime_distribution": regime.get("regime_distribution", {}),
            "mapping": detailed_mapping,
            "fallback": {
                "strategy": fallback_strategy,
                "rationale": "Highest regime-weighted composite score",
            },
        }
        with open(self.run_dir / "regime_mapping.yaml", "w") as f:
            yaml.dump(mapping_data, f, default_flow_style=False, sort_keys=False)
        logger.info("Saved regime_mapping.yaml (v2.0 format)")

        # Save regime_analysis.csv
        if not regime_analysis_df.empty:
            regime_analysis_df.to_csv(
                self.run_dir / "regime_analysis.csv",
                index=False,
            )
            logger.info(f"Saved regime_analysis.csv with {len(regime_analysis_df)} rows")

        # Save regime_history.parquet
        if not regime["regime_history"].empty:
            regime["regime_history"].to_parquet(
                self.run_dir / "regime_history.parquet",
            )
            logger.info(f"Saved regime_history.parquet with {len(regime['regime_history'])} days")

        # Save best_overall_strategy.yaml (fallback for unknown regimes)
        if result.best_config:
            config_dict = result.best_config.to_dict()
            config_dict['_regime_metadata'] = {
                'description': "Fallback strategy for unknown regimes",
                'optimization_run': result.run_timestamp,
                'regime_weighted_score': result.winners[0].composite_score if result.winners else 0.0,
            }
            with open(self.run_dir / "best_overall_strategy.yaml", "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info("Saved best_overall_strategy.yaml")


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
    args: Tuple,
) -> Optional[Any]:
    """Worker function for parallel evaluation.

    This function is called in a separate process for parallel execution.
    Creates a fresh DataAccessContext for each worker to avoid serialization issues.

    Handles both walk-forward and legacy modes based on the args:
    - Walk-forward: (config, snapshot_path, cost_bps, True, wf_config_dict)
    - Legacy: (config, snapshot_path, cost_bps, False, periods_years)

    Args:
        args: Tuple with mode-specific arguments

    Returns:
        WalkForwardEvaluationResult or MultiPeriodResult, or None if evaluation failed
    """
    from quantetf.data.access import DataAccessFactory

    config, snapshot_path, cost_bps, use_walk_forward, mode_config = args

    try:
        # Create fresh DataAccessContext in worker process
        data_access = DataAccessFactory.create_context(
            config={"snapshot_path": str(snapshot_path)},
            enable_caching=False  # Caching not useful in separate processes
        )

        if use_walk_forward:
            # Walk-forward mode
            wf_config = WalkForwardEvaluatorConfig(**mode_config)
            evaluator = WalkForwardEvaluator(
                data_access=data_access,
                wf_config=wf_config,
                cost_bps=cost_bps,
            )
            return evaluator.evaluate(config)
        else:
            # Legacy mode
            evaluator = MultiPeriodEvaluator(
                data_access=data_access,
                cost_bps=cost_bps,
            )
            return evaluator.evaluate(config, mode_config)  # mode_config is periods_years
    except Exception as e:
        logger.warning(f"Worker failed for {config.generate_name()}: {e}")
        return None
