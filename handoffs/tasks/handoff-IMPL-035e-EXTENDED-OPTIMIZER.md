# Task Handoff: IMPL-035e - Extended Optimizer with Regime Analysis

> ⚠️ **SUPERSEDED BY IMPL-040**
>
> This task has been superseded by [IMPL-040 (Optimizer Redesign)](handoff-IMPL-040-OPTIMIZER-REDESIGN.md).
> IMPL-040 addresses fundamental issues in the optimizer that make this task's approach obsolete:
> 1. Replaces fake/stub regime analysis with real backtest data
> 2. Fixes multi-period scoring for quarterly runs
> 3. Makes `regime_mapping.yaml` the primary output
>
> **Do not implement this task. Implement IMPL-040 instead.**
>
> See: [IMPL-035-040-ANALYSIS.md](../architecture/IMPL-035-040-ANALYSIS.md) for full rationale.

**Task ID:** IMPL-035e
**Parent Task:** IMPL-035 (Regime-Based Strategy Selection System)
**Status:** ~~ready~~ → **SUPERSEDED**
**Priority:** ~~HIGH~~ → N/A
**Type:** Enhancement
**Estimated Effort:** 2-3 hours
**Dependencies:** IMPL-035d (Regime Analyzer)
**Superseded By:** IMPL-040

---

## Summary

Extend the strategy optimizer to include regime analysis after finding winning strategies. The optimizer will automatically generate regime-to-strategy mappings based on historical performance.

---

## Deliverables

1. **Updated `src/quantetf/optimization/optimizer.py`** - Add regime analysis step
2. **New output files** in optimization artifacts:
   - `finalists.yaml` - Top N strategies
   - `regime_mapping.yaml` - Computed regime→strategy mapping
   - `regime_analysis.csv` - Detailed per-regime performance
3. **Configuration:** `configs/optimization/regime_analysis.yaml`
4. **Tests:** Verify regime analysis integration

---

## Technical Specification

### New Output Files

After running optimization, the output directory should contain:

```
artifacts/optimization/{timestamp}/
├── all_results.csv          # Existing
├── winners.csv              # Existing
├── best_strategy.yaml       # Existing (now also serves as fallback)
├── optimization_report.md   # Existing
├── finalists.yaml           # NEW: Top N strategies for regime selection
├── regime_mapping.yaml      # NEW: Computed regime→strategy lookup
├── regime_analysis.csv      # NEW: Per-strategy per-regime metrics
└── regime_history.parquet   # NEW: Historical regime labels
```

### `finalists.yaml` Format

```yaml
# Top strategies from optimization, used for regime mapping
version: "1.0"
generated_at: "2026-01-24T12:00:00"
selection_metric: "composite_score"
num_finalists: 6

finalists:
  - name: "momentum_acceleration_long_lookback_days126_..."
    composite_score: 2.84
    1yr_active_return: 0.182
    1yr_sharpe: 2.30
    config_path: "configs/strategies/generated/finalist_1.yaml"

  - name: "momentum_lookback_days63_..."
    composite_score: 2.58
    1yr_active_return: 0.400
    1yr_sharpe: 2.32
    config_path: "configs/strategies/generated/finalist_2.yaml"

  # ... up to N finalists
```

### `regime_mapping.yaml` Format

```yaml
# Computed regime → strategy mapping from optimization
version: "1.0"
generated_at: "2026-01-24T12:00:00"
optimization_run: "20260124_044939"
analysis_period:
  start: "2016-01-01"
  end: "2026-01-20"

mapping:
  uptrend_low_vol:
    strategy: "momentum_acceleration_long_lookback_days126_..."
    config_path: "configs/strategies/generated/finalist_1.yaml"
    regime_sharpe: 2.45
    regime_return: 0.28
    regime_days: 450

  uptrend_high_vol:
    strategy: "vol_adjusted_momentum_lookback_days63_..."
    config_path: "configs/strategies/generated/finalist_3.yaml"
    regime_sharpe: 1.82
    regime_return: 0.15
    regime_days: 180

  downtrend_low_vol:
    strategy: "momentum_lookback_days63_..."
    config_path: "configs/strategies/generated/finalist_2.yaml"
    regime_sharpe: 0.95
    regime_return: 0.08
    regime_days: 120

  downtrend_high_vol:
    strategy: "vol_adjusted_momentum_lookback_days63_..."
    config_path: "configs/strategies/generated/finalist_3.yaml"
    regime_sharpe: 0.42
    regime_return: -0.02
    regime_days: 85

fallback:
  strategy: "momentum_acceleration_long_lookback_days126_..."
  config_path: "configs/strategies/generated/finalist_1.yaml"
  rationale: "Highest composite score across all periods"
```

### Code Changes

```python
# src/quantetf/optimization/optimizer.py

from quantetf.regime.detector import RegimeDetector
from quantetf.regime.analyzer import RegimeAnalyzer
from quantetf.regime.indicators import RegimeIndicators
from quantetf.regime.config import load_thresholds


class StrategyOptimizer:
    """Extended with regime analysis."""

    def __init__(
        self,
        data_access: DataAccessContext,
        regime_analysis_enabled: bool = True,
        num_finalists: int = 6,
    ):
        self.data_access = data_access
        self.regime_analysis_enabled = regime_analysis_enabled
        self.num_finalists = num_finalists

        # Initialize regime components if enabled
        if regime_analysis_enabled:
            config = load_thresholds()
            self.regime_detector = RegimeDetector(config)
            self.regime_indicators = RegimeIndicators(data_access)
            self.regime_analyzer = RegimeAnalyzer(
                self.regime_detector,
                self.regime_indicators,
            )

    def run(self, ...) -> OptimizationResult:
        """Run optimization with regime analysis."""
        # ... existing optimization logic ...
        results = self._run_optimization(...)

        # Select winners
        winners = self._select_winners(results)

        # NEW: Run regime analysis on finalists
        if self.regime_analysis_enabled and len(winners) > 0:
            regime_outputs = self._run_regime_analysis(winners)
        else:
            regime_outputs = None

        return OptimizationResult(
            all_results=results,
            winners=winners,
            regime_outputs=regime_outputs,
        )

    def _run_regime_analysis(
        self,
        winners: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run regime analysis on winning strategies.

        Args:
            winners: DataFrame of winning strategy results

        Returns:
            Dict containing:
            - finalists: List of top N strategies
            - regime_mapping: Dict of regime → strategy
            - regime_analysis: DataFrame of per-regime metrics
            - regime_history: DataFrame of historical regime labels
        """
        # Select top N finalists
        finalists = winners.nsmallest(self.num_finalists, "composite_score_rank")

        # Label historical period with regimes
        regime_labels = self.regime_analyzer.label_history(
            start_date=self.backtest_start_date,
            end_date=self.backtest_end_date,
        )

        # Get daily returns for each finalist
        # (Need to extract from backtest results or re-run)
        finalist_returns = self._extract_finalist_returns(finalists)

        # Analyze each finalist by regime
        analysis = self.regime_analyzer.analyze_multiple_strategies(
            strategy_results=finalist_returns,
            regime_labels=regime_labels,
        )

        # Compute optimal mapping
        mapping = self.regime_analyzer.compute_regime_mapping(
            analysis_results=analysis,
            metric="sharpe_ratio",
        )

        return {
            "finalists": finalists,
            "regime_mapping": mapping,
            "regime_analysis": analysis,
            "regime_history": regime_labels,
        }

    def _extract_finalist_returns(
        self,
        finalists: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        """
        Extract or reconstruct daily returns for finalist strategies.

        This may require re-running backtests with daily equity curves
        if not already stored.
        """
        # Implementation depends on what's stored in backtest results
        # Option 1: If daily equity curves are stored, extract returns
        # Option 2: Re-run backtest for finalists with daily tracking

        returns = {}
        for _, row in finalists.iterrows():
            config_name = row["config_name"]

            # Try to load from cached backtest
            backtest_path = self._get_backtest_path(config_name)
            if backtest_path and backtest_path.exists():
                equity_curve = pd.read_parquet(backtest_path / "equity_curve.parquet")
                returns[config_name] = equity_curve["returns"]
            else:
                # Re-run backtest for this config
                logger.info(f"Re-running backtest for {config_name} to get daily returns")
                result = self._run_single_backtest(row["config"])
                returns[config_name] = result.equity_curve["returns"]

        return returns

    def save_outputs(self, result: OptimizationResult, output_dir: Path) -> None:
        """Save all optimization outputs including regime analysis."""
        # ... existing save logic ...

        if result.regime_outputs:
            self._save_regime_outputs(result.regime_outputs, output_dir)

    def _save_regime_outputs(
        self,
        regime_outputs: Dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Save regime analysis outputs."""
        import yaml

        # Save finalists.yaml
        finalists_data = {
            "version": "1.0",
            "generated_at": pd.Timestamp.now().isoformat(),
            "selection_metric": "composite_score",
            "num_finalists": len(regime_outputs["finalists"]),
            "finalists": regime_outputs["finalists"].to_dict(orient="records"),
        }
        with open(output_dir / "finalists.yaml", "w") as f:
            yaml.dump(finalists_data, f, default_flow_style=False)

        # Save regime_mapping.yaml
        mapping_data = {
            "version": "1.0",
            "generated_at": pd.Timestamp.now().isoformat(),
            "optimization_run": output_dir.name,
            "mapping": regime_outputs["regime_mapping"],
            "fallback": {
                "strategy": regime_outputs["finalists"].iloc[0]["config_name"],
                "rationale": "Highest composite score",
            },
        }
        with open(output_dir / "regime_mapping.yaml", "w") as f:
            yaml.dump(mapping_data, f, default_flow_style=False)

        # Save regime_analysis.csv
        regime_outputs["regime_analysis"].to_csv(
            output_dir / "regime_analysis.csv",
            index=False,
        )

        # Save regime_history.parquet
        regime_outputs["regime_history"].to_parquet(
            output_dir / "regime_history.parquet",
        )

        logger.info(f"Regime analysis outputs saved to {output_dir}")
```

### Configuration File

```yaml
# configs/optimization/regime_analysis.yaml

# Regime analysis settings for optimizer
regime_analysis:
  enabled: true

  # Number of top strategies to analyze
  num_finalists: 6

  # Metric to optimize for regime mapping
  # Options: sharpe_ratio, annualized_return, sortino_ratio
  optimization_metric: "sharpe_ratio"

  # Minimum days in a regime to consider for mapping
  min_regime_days: 20

  # Whether to generate strategy config files for finalists
  generate_finalist_configs: true
```

---

## Test Cases

```python
# tests/optimization/test_optimizer_regime.py
import pytest
import pandas as pd
from pathlib import Path

from quantetf.optimization.optimizer import StrategyOptimizer


class TestOptimizerRegimeAnalysis:
    """Test regime analysis integration in optimizer."""

    def test_optimizer_produces_regime_outputs(self, data_access, tmp_path):
        """Optimizer should produce regime analysis when enabled."""
        optimizer = StrategyOptimizer(
            data_access=data_access,
            regime_analysis_enabled=True,
            num_finalists=3,
        )

        # Run small optimization
        result = optimizer.run(max_configs=10)

        assert result.regime_outputs is not None
        assert "finalists" in result.regime_outputs
        assert "regime_mapping" in result.regime_outputs

    def test_regime_mapping_covers_all_regimes(self, data_access):
        """Regime mapping should have entry for each regime."""
        optimizer = StrategyOptimizer(data_access=data_access)
        result = optimizer.run(max_configs=10)

        mapping = result.regime_outputs["regime_mapping"]

        required_regimes = [
            "uptrend_low_vol",
            "uptrend_high_vol",
            "downtrend_low_vol",
            "downtrend_high_vol",
        ]
        for regime in required_regimes:
            assert regime in mapping, f"Missing mapping for {regime}"

    def test_finalists_yaml_created(self, data_access, tmp_path):
        """Optimizer should create finalists.yaml."""
        optimizer = StrategyOptimizer(data_access=data_access)
        result = optimizer.run(max_configs=10)
        optimizer.save_outputs(result, tmp_path)

        assert (tmp_path / "finalists.yaml").exists()

    def test_regime_mapping_yaml_created(self, data_access, tmp_path):
        """Optimizer should create regime_mapping.yaml."""
        optimizer = StrategyOptimizer(data_access=data_access)
        result = optimizer.run(max_configs=10)
        optimizer.save_outputs(result, tmp_path)

        assert (tmp_path / "regime_mapping.yaml").exists()

    def test_disabled_regime_analysis(self, data_access):
        """Regime analysis can be disabled."""
        optimizer = StrategyOptimizer(
            data_access=data_access,
            regime_analysis_enabled=False,
        )
        result = optimizer.run(max_configs=10)

        assert result.regime_outputs is None
```

---

## Files to Modify/Create

| File | Action |
|------|--------|
| `src/quantetf/optimization/optimizer.py` | Modify: Add regime analysis |
| `src/quantetf/optimization/types.py` | Modify: Add OptimizationResult fields |
| `configs/optimization/regime_analysis.yaml` | Create |
| `tests/optimization/test_optimizer_regime.py` | Create |

---

## Acceptance Criteria

- [ ] Optimizer produces `finalists.yaml` with top N strategies
- [ ] Optimizer produces `regime_mapping.yaml` with regime→strategy lookup
- [ ] Optimizer produces `regime_analysis.csv` with per-regime metrics
- [ ] Optimizer produces `regime_history.parquet` with historical labels
- [ ] Regime analysis can be disabled via config
- [ ] All existing optimizer tests still pass
- [ ] New regime analysis tests pass

---

## Integration Notes

1. **Script update:** Update `scripts/find_best_strategy.py` to pass regime config
2. **Performance:** Regime analysis adds ~30s to optimization (can be parallelized)
3. **Backward compatibility:** Old optimization runs still work (no regime outputs)

---

**Document Version:** 1.0
**Created:** 2026-01-24
**For:** Coding Agent
