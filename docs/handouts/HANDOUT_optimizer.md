# Task 3: Strategy Optimizer

## File to Create
`src/quantetf/optimization/optimizer.py`

## Purpose
Main orchestrator that generates all configurations, runs evaluations, ranks results, and produces reports.

## Implementation

```python
"""Strategy optimizer - main orchestration."""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from quantetf.data.snapshot import SnapshotDataStore
from quantetf.optimization.grid import generate_configs, StrategyConfig
from quantetf.optimization.evaluator import MultiPeriodEvaluator, MultiPeriodResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Complete optimization run results."""
    all_results: List[MultiPeriodResult]
    winners: List[MultiPeriodResult]
    best_config: Optional[StrategyConfig]
    run_timestamp: str
    total_configs: int
    successful_configs: int
    failed_configs: int


class StrategyOptimizer:
    """
    Main optimizer that searches across parameter space to find strategies
    that beat SPY across all evaluation periods.
    """

    def __init__(
        self,
        snapshot_path: str,
        output_dir: str,
        periods_years: List[int] = [3, 5, 10],
        max_workers: int = 1
    ):
        """
        Args:
            snapshot_path: Path to data snapshot
            output_dir: Directory for output files
            periods_years: Evaluation periods in years
            max_workers: Number of parallel workers (1 = sequential)
        """
        self.snapshot = SnapshotDataStore(snapshot_path)
        self.output_dir = Path(output_dir)
        self.periods_years = periods_years
        self.max_workers = max_workers
        self.evaluator = MultiPeriodEvaluator(self.snapshot)

        # Create output directory
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / self.run_timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run(self, max_configs: Optional[int] = None) -> OptimizationResult:
        """
        Run full optimization sweep.

        Args:
            max_configs: Optional limit on configs to test (for debugging)

        Returns:
            OptimizationResult with all results and winners
        """
        # Generate all configurations
        configs = generate_configs()
        total_configs = len(configs)
        logger.info(f"Generated {total_configs} configurations")

        if max_configs:
            configs = configs[:max_configs]
            logger.info(f"Limited to {max_configs} configurations")

        # Run evaluations
        results = []
        failed = 0

        if self.max_workers > 1:
            results, failed = self._run_parallel(configs)
        else:
            results, failed = self._run_sequential(configs)

        # Find winners (beat SPY in all periods)
        winners = [r for r in results if r.beats_spy_all_periods]

        # Sort by composite score
        results.sort(key=lambda r: r.composite_score, reverse=True)
        winners.sort(key=lambda r: r.composite_score, reverse=True)

        # Get best config
        best_config = None
        if winners:
            best_config_name = winners[0].config_name
            best_config = next(c for c in configs if c.generate_name() == best_config_name)

        result = OptimizationResult(
            all_results=results,
            winners=winners,
            best_config=best_config,
            run_timestamp=self.run_timestamp,
            total_configs=total_configs,
            successful_configs=len(results),
            failed_configs=failed
        )

        # Save outputs
        self._save_results(result, configs)

        return result

    def _run_sequential(self, configs: List[StrategyConfig]) -> tuple:
        """Run evaluations sequentially."""
        results = []
        failed = 0

        for config in tqdm(configs, desc="Evaluating strategies"):
            try:
                result = self.evaluator.evaluate(config, self.periods_years)
                results.append(result)
            except Exception as e:
                logger.warning(f"Config {config.generate_name()} failed: {e}")
                failed += 1

        return results, failed

    def _run_parallel(self, configs: List[StrategyConfig]) -> tuple:
        """Run evaluations in parallel."""
        results = []
        failed = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_config, config): config
                for config in configs
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating strategies"):
                config = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        failed += 1
                except Exception as e:
                    logger.warning(f"Config {config.generate_name()} failed: {e}")
                    failed += 1

        return results, failed

    def _evaluate_config(self, config: StrategyConfig) -> Optional[MultiPeriodResult]:
        """Evaluate a single config (for parallel execution)."""
        try:
            return self.evaluator.evaluate(config, self.periods_years)
        except Exception as e:
            logger.warning(f"Config {config.generate_name()} failed: {e}")
            return None

    def _save_results(self, result: OptimizationResult, configs: List[StrategyConfig]):
        """Save all output files."""
        # Save all results CSV
        all_df = pd.DataFrame([r.to_dict() for r in result.all_results])
        all_df.to_csv(self.run_dir / 'all_results.csv', index=False)
        logger.info(f"Saved all_results.csv with {len(all_df)} rows")

        # Save winners CSV
        if result.winners:
            winners_df = pd.DataFrame([r.to_dict() for r in result.winners])
            winners_df.to_csv(self.run_dir / 'winners.csv', index=False)
            logger.info(f"Saved winners.csv with {len(winners_df)} rows")

        # Save best strategy YAML
        if result.best_config:
            best_config_dict = result.best_config.to_dict()
            best_config_dict['description'] = f"""
Auto-discovered strategy that beats SPY across {', '.join(f'{y}yr' for y in self.periods_years)} periods.
Generated by strategy optimizer on {self.run_timestamp}.
Composite score: {result.winners[0].composite_score:.3f}
"""
            with open(self.run_dir / 'best_strategy.yaml', 'w') as f:
                yaml.dump(best_config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved best_strategy.yaml: {result.best_config.generate_name()}")

        # Save optimization report
        self._save_report(result)

    def _save_report(self, result: OptimizationResult):
        """Generate and save summary report."""
        report_lines = [
            "# Strategy Optimization Report",
            "",
            f"**Run timestamp:** {result.run_timestamp}",
            f"**Periods evaluated:** {', '.join(f'{y}yr' for y in self.periods_years)}",
            "",
            "## Summary",
            "",
            f"- Total configurations: {result.total_configs}",
            f"- Successful evaluations: {result.successful_configs}",
            f"- Failed evaluations: {result.failed_configs}",
            f"- **Strategies that beat SPY in ALL periods: {len(result.winners)}**",
            "",
        ]

        if result.winners:
            report_lines.extend([
                "## Top 10 Winners",
                "",
                "| Rank | Config | Composite Score | 3yr Active | 5yr Active | 10yr Active |",
                "|------|--------|-----------------|------------|------------|-------------|",
            ])

            for i, winner in enumerate(result.winners[:10], 1):
                p3 = winner.periods.get('3yr')
                p5 = winner.periods.get('5yr')
                p10 = winner.periods.get('10yr')
                report_lines.append(
                    f"| {i} | {winner.config_name} | {winner.composite_score:.3f} | "
                    f"{p3.active_return*100:.1f}% | {p5.active_return*100:.1f}% | {p10.active_return*100:.1f}% |"
                )

            report_lines.extend([
                "",
                "## Best Strategy Details",
                "",
                f"**Name:** {result.best_config.generate_name()}",
                f"**Alpha Model:** {result.best_config.alpha_type}",
                f"**Parameters:** {result.best_config.alpha_params}",
                f"**Top N:** {result.best_config.top_n}",
                f"**Schedule:** {result.best_config.schedule_name}",
                "",
            ])
        else:
            report_lines.extend([
                "## No Winners Found",
                "",
                "No strategy beat SPY in all evaluation periods.",
                "Consider adjusting parameters or using different alpha models.",
                "",
            ])

        with open(self.run_dir / 'optimization_report.md', 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info("Saved optimization_report.md")
```

## Key Features

1. **Sequential or parallel execution**: Use `max_workers=1` for debugging, higher for production
2. **Progress tracking**: Uses tqdm for visual progress
3. **Graceful error handling**: Logs and skips failed configs
4. **Comprehensive output**:
   - `all_results.csv` - every config with metrics
   - `winners.csv` - only configs that beat SPY
   - `best_strategy.yaml` - ready-to-use config file
   - `optimization_report.md` - human-readable summary

## Dependencies

- `src/quantetf/optimization/grid.py`
- `src/quantetf/optimization/evaluator.py`
- `src/quantetf/data/snapshot.py`
- `tqdm` (for progress bars)
- `yaml` (for config output)

## Testing

```python
def test_optimizer_small_run():
    optimizer = StrategyOptimizer(
        snapshot_path='data/snapshots/snapshot_20260113_232157',
        output_dir='artifacts/optimization',
        periods_years=[3, 5, 10],
        max_workers=1
    )

    # Run with just 10 configs for quick test
    result = optimizer.run(max_configs=10)

    assert result.total_configs == 10
    assert result.successful_configs > 0
    assert len(result.all_results) > 0

    print(f"Winners: {len(result.winners)}")
    if result.best_config:
        print(f"Best: {result.best_config.generate_name()}")
```

## Output Directory Structure

```
artifacts/optimization/20260114_143022/
├── all_results.csv           # 354 rows with all metrics
├── winners.csv               # Only winning strategies
├── best_strategy.yaml        # Top strategy config
└── optimization_report.md    # Summary report
```

## Notes

- Parallel execution with `ProcessPoolExecutor` requires pickling - ensure all objects are picklable
- For debugging, always use `max_workers=1` first
- The composite score is designed to reward consistency across time periods
