# Task Handoff: IMPL-035d - Regime Analyzer

**Task ID:** IMPL-035d
**Parent Task:** IMPL-035 (Regime-Based Strategy Selection System)
**Status:** ready
**Priority:** HIGH
**Type:** Analytics
**Estimated Effort:** 3-4 hours
**Dependencies:** IMPL-035a (Regime Detector)

---

## Summary

Implement the Regime Analyzer that labels historical periods by regime and analyzes strategy performance within each regime. This enables data-driven regime→strategy mapping.

---

## Deliverables

1. **`src/quantetf/regime/analyzer.py`** - RegimeAnalyzer class
2. **`tests/regime/test_analyzer.py`** - Unit tests
3. **Output capability:** Generate `regime_history.parquet` and per-regime performance reports

---

## Technical Specification

### Interface Design

```python
# src/quantetf/regime/analyzer.py
"""Analyze strategy performance by market regime."""

from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from .detector import RegimeDetector
from .indicators import RegimeIndicators
from .types import RegimeState, RegimeConfig

logger = logging.getLogger(__name__)


class RegimeAnalyzer:
    """
    Analyzes historical regimes and strategy performance within each regime.

    This class:
    1. Labels historical dates with their regime
    2. Calculates strategy performance metrics per regime
    3. Recommends regime→strategy mappings based on performance
    """

    def __init__(
        self,
        detector: RegimeDetector,
        indicators: RegimeIndicators,
    ):
        """
        Initialize analyzer.

        Args:
            detector: Configured RegimeDetector
            indicators: RegimeIndicators for fetching SPY/VIX data
        """
        self.detector = detector
        self.indicators = indicators

    def label_history(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Label historical dates with their regime.

        Args:
            start_date: Start of period to label
            end_date: End of period to label

        Returns:
            DataFrame with columns:
            - date (index)
            - regime: str (e.g., "uptrend_low_vol")
            - trend: str ("uptrend" or "downtrend")
            - vol: str ("low_vol" or "high_vol")
            - spy_price: float
            - spy_200ma: float
            - vix: float
        """
        # Get indicator data for full period
        spy_data = self.indicators.get_spy_data(
            as_of=end_date,
            lookback_days=(end_date - start_date).days + 250,  # Extra for 200MA warmup
        )
        vix_data = self.indicators.get_vix(
            as_of=end_date,
            lookback_days=(end_date - start_date).days + 30,
        )

        # Filter to requested date range
        spy_data = spy_data[(spy_data.index >= start_date) & (spy_data.index <= end_date)]

        # Align VIX to SPY dates
        vix_aligned = vix_data.reindex(spy_data.index, method="ffill")

        # Label each date
        records = []
        previous_state = None

        for date in spy_data.index:
            spy_price = spy_data.loc[date, "close"]
            spy_200ma = spy_data.loc[date, "ma_200"]
            vix = vix_aligned.loc[date]

            # Skip if 200MA not yet available
            if pd.isna(spy_200ma):
                continue

            state = self.detector.detect(
                spy_price=spy_price,
                spy_200ma=spy_200ma,
                vix=vix,
                previous_state=previous_state,
                as_of=date,
            )

            records.append({
                "date": date,
                "regime": state.name,
                "trend": state.trend.value,
                "vol": state.vol.value,
                "spy_price": spy_price,
                "spy_200ma": spy_200ma,
                "vix": vix,
            })

            previous_state = state

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)

        logger.info(
            f"Labeled {len(df)} trading days from {start_date} to {end_date}. "
            f"Regime distribution: {df['regime'].value_counts().to_dict()}"
        )

        return df

    def analyze_strategy_by_regime(
        self,
        strategy_returns: pd.Series,
        regime_labels: pd.DataFrame,
        strategy_name: str = "strategy",
    ) -> pd.DataFrame:
        """
        Calculate strategy performance metrics for each regime.

        Args:
            strategy_returns: Daily returns of strategy (indexed by date)
            regime_labels: Output from label_history()
            strategy_name: Name for identification

        Returns:
            DataFrame with one row per regime, columns:
            - regime
            - strategy_name
            - total_return
            - annualized_return
            - volatility
            - sharpe_ratio
            - max_drawdown
            - num_days
            - pct_of_period
        """
        # Align returns to regime labels
        common_dates = strategy_returns.index.intersection(regime_labels.index)
        returns_aligned = strategy_returns.loc[common_dates]
        regimes_aligned = regime_labels.loc[common_dates, "regime"]

        results = []
        for regime in regimes_aligned.unique():
            regime_mask = regimes_aligned == regime
            regime_returns = returns_aligned[regime_mask]

            if len(regime_returns) < 5:
                logger.warning(f"Only {len(regime_returns)} days for {regime}, skipping")
                continue

            # Calculate metrics
            total_return = (1 + regime_returns).prod() - 1
            num_days = len(regime_returns)
            annualized_return = (1 + total_return) ** (252 / num_days) - 1
            volatility = regime_returns.std() * np.sqrt(252)
            sharpe = annualized_return / volatility if volatility > 0 else 0

            # Max drawdown
            cumulative = (1 + regime_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            results.append({
                "regime": regime,
                "strategy_name": strategy_name,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "num_days": num_days,
                "pct_of_period": num_days / len(returns_aligned),
            })

        return pd.DataFrame(results)

    def analyze_multiple_strategies(
        self,
        strategy_results: Dict[str, pd.Series],
        regime_labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze multiple strategies across all regimes.

        Args:
            strategy_results: Dict of {strategy_name: daily_returns}
            regime_labels: Output from label_history()

        Returns:
            Combined DataFrame with all strategies and regimes
        """
        all_results = []
        for name, returns in strategy_results.items():
            analysis = self.analyze_strategy_by_regime(
                strategy_returns=returns,
                regime_labels=regime_labels,
                strategy_name=name,
            )
            all_results.append(analysis)

        return pd.concat(all_results, ignore_index=True)

    def compute_regime_mapping(
        self,
        analysis_results: pd.DataFrame,
        metric: str = "sharpe_ratio",
        min_days: int = 20,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute optimal regime→strategy mapping.

        Args:
            analysis_results: Output from analyze_multiple_strategies()
            metric: Metric to optimize ("sharpe_ratio", "annualized_return", etc.)
            min_days: Minimum days required in a regime to consider

        Returns:
            Dict mapping regime name to strategy info:
            {
                "uptrend_low_vol": {
                    "strategy": "momentum_acceleration",
                    "score": 2.3,
                    "metric": "sharpe_ratio",
                },
                ...
            }
        """
        # Filter by minimum days
        filtered = analysis_results[analysis_results["num_days"] >= min_days]

        mapping = {}
        for regime in filtered["regime"].unique():
            regime_data = filtered[filtered["regime"] == regime]

            # Find best strategy by metric
            best_idx = regime_data[metric].idxmax()
            best_row = regime_data.loc[best_idx]

            mapping[regime] = {
                "strategy": best_row["strategy_name"],
                "score": best_row[metric],
                "metric": metric,
                "total_return": best_row["total_return"],
                "num_days": best_row["num_days"],
            }

            logger.info(
                f"Regime '{regime}': best strategy = {best_row['strategy_name']} "
                f"({metric}={best_row[metric]:.3f})"
            )

        return mapping

    def generate_report(
        self,
        analysis_results: pd.DataFrame,
        regime_labels: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """
        Generate regime analysis report files.

        Creates:
        - regime_history.parquet: Daily regime labels
        - regime_performance.csv: Strategy performance by regime
        - regime_summary.md: Markdown report
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save regime history
        regime_labels.to_parquet(output_dir / "regime_history.parquet")

        # Save performance analysis
        analysis_results.to_csv(output_dir / "regime_performance.csv", index=False)

        # Generate markdown summary
        summary = self._generate_summary_md(analysis_results, regime_labels)
        (output_dir / "regime_summary.md").write_text(summary)

        logger.info(f"Regime analysis report saved to {output_dir}")

    def _generate_summary_md(
        self,
        analysis: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> str:
        """Generate markdown summary report."""
        lines = [
            "# Regime Analysis Summary",
            "",
            f"**Period:** {labels.index.min().date()} to {labels.index.max().date()}",
            f"**Total Trading Days:** {len(labels)}",
            "",
            "## Regime Distribution",
            "",
        ]

        # Regime distribution
        dist = labels["regime"].value_counts()
        for regime, count in dist.items():
            pct = 100 * count / len(labels)
            lines.append(f"- **{regime}:** {count} days ({pct:.1f}%)")

        lines.extend(["", "## Best Strategy by Regime", ""])

        # Best strategy per regime
        for regime in analysis["regime"].unique():
            regime_data = analysis[analysis["regime"] == regime]
            best = regime_data.loc[regime_data["sharpe_ratio"].idxmax()]
            lines.append(
                f"- **{regime}:** {best['strategy_name']} "
                f"(Sharpe={best['sharpe_ratio']:.2f}, Return={best['annualized_return']:.1%})"
            )

        return "\n".join(lines)
```

---

## Test Cases

```python
# tests/regime/test_analyzer.py
import pytest
import pandas as pd
import numpy as np

from quantetf.regime.analyzer import RegimeAnalyzer
from quantetf.regime.detector import RegimeDetector
from quantetf.regime.indicators import RegimeIndicators


class TestRegimeAnalyzer:
    """Test regime analysis functionality."""

    @pytest.fixture
    def mock_indicators(self):
        """Create mock indicators with synthetic data."""
        class MockIndicators:
            def get_spy_data(self, as_of, lookback_days):
                dates = pd.date_range(end=as_of, periods=300, freq="B")
                close = 500 + np.cumsum(np.random.randn(300) * 2)
                ma_200 = pd.Series(close).rolling(200).mean().values
                return pd.DataFrame({
                    "close": close,
                    "ma_200": ma_200,
                }, index=dates)

            def get_vix(self, as_of, lookback_days):
                dates = pd.date_range(end=as_of, periods=lookback_days, freq="B")
                vix = 15 + np.random.randn(lookback_days) * 3
                return pd.Series(vix, index=dates)

        return MockIndicators()

    @pytest.fixture
    def analyzer(self, mock_indicators):
        detector = RegimeDetector()
        return RegimeAnalyzer(detector, mock_indicators)

    def test_label_history_returns_dataframe(self, analyzer):
        """label_history should return DataFrame with required columns."""
        labels = analyzer.label_history(
            start_date=pd.Timestamp("2025-06-01"),
            end_date=pd.Timestamp("2025-12-31"),
        )

        assert isinstance(labels, pd.DataFrame)
        assert "regime" in labels.columns
        assert "trend" in labels.columns
        assert "vol" in labels.columns
        assert "spy_price" in labels.columns

    def test_all_regimes_valid(self, analyzer):
        """All regime labels should be one of 4 valid regimes."""
        labels = analyzer.label_history(
            start_date=pd.Timestamp("2025-01-01"),
            end_date=pd.Timestamp("2025-12-31"),
        )

        valid_regimes = {
            "uptrend_low_vol",
            "uptrend_high_vol",
            "downtrend_low_vol",
            "downtrend_high_vol",
        }
        assert set(labels["regime"].unique()).issubset(valid_regimes)

    def test_analyze_strategy_by_regime(self, analyzer):
        """Strategy analysis should return metrics per regime."""
        # Create synthetic strategy returns
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="B")
        returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

        # Create synthetic regime labels
        regime_labels = pd.DataFrame({
            "regime": np.random.choice(
                ["uptrend_low_vol", "downtrend_high_vol"],
                size=len(dates),
            ),
        }, index=dates)

        analysis = analyzer.analyze_strategy_by_regime(
            strategy_returns=returns,
            regime_labels=regime_labels,
            strategy_name="test_strategy",
        )

        assert "regime" in analysis.columns
        assert "sharpe_ratio" in analysis.columns
        assert "max_drawdown" in analysis.columns
        assert len(analysis) > 0

    def test_compute_regime_mapping(self, analyzer):
        """Regime mapping should pick best strategy per regime."""
        # Create mock analysis results
        analysis = pd.DataFrame([
            {"regime": "uptrend_low_vol", "strategy_name": "A", "sharpe_ratio": 2.0, "num_days": 50, "total_return": 0.1},
            {"regime": "uptrend_low_vol", "strategy_name": "B", "sharpe_ratio": 1.5, "num_days": 50, "total_return": 0.08},
            {"regime": "downtrend_low_vol", "strategy_name": "A", "sharpe_ratio": 0.5, "num_days": 30, "total_return": 0.02},
            {"regime": "downtrend_low_vol", "strategy_name": "B", "sharpe_ratio": 1.0, "num_days": 30, "total_return": 0.04},
        ])

        mapping = analyzer.compute_regime_mapping(analysis)

        assert mapping["uptrend_low_vol"]["strategy"] == "A"  # Higher Sharpe
        assert mapping["downtrend_low_vol"]["strategy"] == "B"  # Higher Sharpe
```

---

## Acceptance Criteria

- [ ] `label_history()` correctly labels historical dates with regimes
- [ ] Labels use hysteresis (consistent with detector)
- [ ] `analyze_strategy_by_regime()` calculates correct metrics
- [ ] `compute_regime_mapping()` picks best strategy per regime
- [ ] `generate_report()` creates all output files
- [ ] All tests pass
- [ ] No lookahead bias in historical labeling

---

## Integration with Optimizer

After this task, the optimizer (IMPL-035e) will use RegimeAnalyzer to:

1. Label the backtest period by regime
2. Analyze finalist strategies by regime
3. Generate `regime_mapping.yaml` as output

---

**Document Version:** 1.0
**Created:** 2026-01-24
**For:** Coding Agent
