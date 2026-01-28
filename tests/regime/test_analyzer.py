"""Tests for regime analysis functionality."""

import pytest
import pandas as pd
import numpy as np

from quantetf.regime.analyzer import RegimeAnalyzer
from quantetf.regime.detector import RegimeDetector


class MockIndicators:
    """Mock indicators for testing."""

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


class TestRegimeAnalyzer:
    """Test regime analysis functionality."""

    @pytest.fixture
    def mock_indicators(self):
        """Create mock indicators with synthetic data."""
        return MockIndicators()

    @pytest.fixture
    def analyzer(self, mock_indicators):
        """Create analyzer with mock dependencies."""
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
        assert "spy_200ma" in labels.columns
        assert "vix" in labels.columns

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

    def test_label_history_respects_date_range(self, analyzer):
        """Labels should only include dates within the requested range."""
        start = pd.Timestamp("2025-06-01")
        end = pd.Timestamp("2025-12-31")

        labels = analyzer.label_history(start_date=start, end_date=end)

        assert labels.index.min() >= start
        assert labels.index.max() <= end

    def test_analyze_strategy_by_regime_returns_metrics(self, analyzer):
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
        assert "strategy_name" in analysis.columns
        assert "sharpe_ratio" in analysis.columns
        assert "max_drawdown" in analysis.columns
        assert "annualized_return" in analysis.columns
        assert "volatility" in analysis.columns
        assert "num_days" in analysis.columns
        assert len(analysis) > 0

    def test_analyze_strategy_skips_insufficient_data(self, analyzer):
        """Regimes with fewer than 5 days should be skipped."""
        dates = pd.date_range("2025-01-01", "2025-01-10", freq="B")  # Only 8 days
        returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

        # One regime has only 2 days
        regime_labels = pd.DataFrame({
            "regime": ["uptrend_low_vol"] * 2 + ["downtrend_high_vol"] * 6,
        }, index=dates)

        analysis = analyzer.analyze_strategy_by_regime(
            strategy_returns=returns,
            regime_labels=regime_labels,
            strategy_name="test",
        )

        # Only downtrend_high_vol should be in results
        assert len(analysis) == 1
        assert analysis.iloc[0]["regime"] == "downtrend_high_vol"

    def test_analyze_multiple_strategies(self, analyzer):
        """Multiple strategies should be analyzed together."""
        dates = pd.date_range("2025-01-01", "2025-06-30", freq="B")
        strategy_a = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        strategy_b = pd.Series(np.random.randn(len(dates)) * 0.015, index=dates)

        regime_labels = pd.DataFrame({
            "regime": np.random.choice(
                ["uptrend_low_vol", "downtrend_low_vol"],
                size=len(dates),
            ),
        }, index=dates)

        results = analyzer.analyze_multiple_strategies(
            {"strategy_a": strategy_a, "strategy_b": strategy_b},
            regime_labels,
        )

        assert "strategy_a" in results["strategy_name"].values
        assert "strategy_b" in results["strategy_name"].values

    def test_compute_regime_mapping_picks_best(self, analyzer):
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

    def test_compute_regime_mapping_respects_min_days(self, analyzer):
        """Mapping should filter out regimes with insufficient data."""
        analysis = pd.DataFrame([
            {"regime": "uptrend_low_vol", "strategy_name": "A", "sharpe_ratio": 2.0, "num_days": 50, "total_return": 0.1},
            {"regime": "downtrend_low_vol", "strategy_name": "A", "sharpe_ratio": 0.5, "num_days": 10, "total_return": 0.02},
        ])

        mapping = analyzer.compute_regime_mapping(analysis, min_days=20)

        assert "uptrend_low_vol" in mapping
        assert "downtrend_low_vol" not in mapping  # Only 10 days

    def test_compute_regime_mapping_custom_metric(self, analyzer):
        """Mapping should optimize by specified metric."""
        analysis = pd.DataFrame([
            {"regime": "uptrend_low_vol", "strategy_name": "A", "sharpe_ratio": 2.0, "annualized_return": 0.10, "num_days": 50, "total_return": 0.1},
            {"regime": "uptrend_low_vol", "strategy_name": "B", "sharpe_ratio": 1.5, "annualized_return": 0.20, "num_days": 50, "total_return": 0.15},
        ])

        # By sharpe, A wins
        mapping_sharpe = analyzer.compute_regime_mapping(analysis, metric="sharpe_ratio")
        assert mapping_sharpe["uptrend_low_vol"]["strategy"] == "A"

        # By return, B wins
        mapping_return = analyzer.compute_regime_mapping(analysis, metric="annualized_return")
        assert mapping_return["uptrend_low_vol"]["strategy"] == "B"

    def test_generate_report(self, analyzer, tmp_path):
        """Report generation should create all files."""
        # Create mock data
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="B")
        regime_labels = pd.DataFrame({
            "regime": np.random.choice(
                ["uptrend_low_vol", "downtrend_high_vol"],
                size=len(dates),
            ),
        }, index=dates)

        analysis = pd.DataFrame([
            {"regime": "uptrend_low_vol", "strategy_name": "A", "sharpe_ratio": 2.0, "annualized_return": 0.15, "num_days": 100, "total_return": 0.1},
            {"regime": "downtrend_high_vol", "strategy_name": "B", "sharpe_ratio": 1.0, "annualized_return": 0.08, "num_days": 100, "total_return": 0.05},
        ])

        output_dir = tmp_path / "regime_report"
        analyzer.generate_report(analysis, regime_labels, output_dir)

        assert (output_dir / "regime_history.parquet").exists()
        assert (output_dir / "regime_performance.csv").exists()
        assert (output_dir / "regime_summary.md").exists()

    def test_summary_md_content(self, analyzer, tmp_path):
        """Summary markdown should contain expected sections."""
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="B")
        regime_labels = pd.DataFrame({
            "regime": ["uptrend_low_vol"] * len(dates),
        }, index=dates)

        analysis = pd.DataFrame([
            {"regime": "uptrend_low_vol", "strategy_name": "momentum", "sharpe_ratio": 2.0, "annualized_return": 0.15, "num_days": 200, "total_return": 0.1},
        ])

        output_dir = tmp_path / "regime_report"
        analyzer.generate_report(analysis, regime_labels, output_dir)

        summary = (output_dir / "regime_summary.md").read_text()

        assert "# Regime Analysis Summary" in summary
        assert "Regime Distribution" in summary
        assert "Best Strategy by Regime" in summary
        assert "uptrend_low_vol" in summary


class TestRegimeMetrics:
    """Test metric calculations."""

    @pytest.fixture
    def analyzer(self):
        return RegimeAnalyzer(RegimeDetector(), MockIndicators())

    def test_sharpe_ratio_calculation(self, analyzer):
        """Sharpe ratio should be annualized return / annualized vol."""
        # Create predictable returns
        dates = pd.date_range("2025-01-01", periods=252, freq="B")
        returns = pd.Series([0.001] * 252, index=dates)  # 0.1% daily

        regime_labels = pd.DataFrame({"regime": ["uptrend_low_vol"] * 252}, index=dates)

        analysis = analyzer.analyze_strategy_by_regime(returns, regime_labels)

        # With constant 0.1% daily: annualized ~28%, vol ~0, sharpe high
        assert analysis.iloc[0]["sharpe_ratio"] > 0

    def test_max_drawdown_calculation(self, analyzer):
        """Max drawdown should be correctly calculated."""
        dates = pd.date_range("2025-01-01", periods=10, freq="B")
        # Up 5%, then down 10%, then up 3%
        returns = pd.Series([0.05, -0.10, 0.03, 0, 0, 0, 0, 0, 0, 0], index=dates)

        regime_labels = pd.DataFrame({"regime": ["uptrend_low_vol"] * 10}, index=dates)

        analysis = analyzer.analyze_strategy_by_regime(returns, regime_labels)

        # Max drawdown should be negative
        assert analysis.iloc[0]["max_drawdown"] < 0
