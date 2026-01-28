"""Analyze strategy performance by market regime."""

from typing import Any, Dict
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from .detector import RegimeDetector
from .indicators import RegimeIndicators

logger = logging.getLogger(__name__)


class RegimeAnalyzer:
    """Analyzes historical regimes and strategy performance within each regime.

    This class:
    1. Labels historical dates with their regime
    2. Calculates strategy performance metrics per regime
    3. Recommends regime→strategy mappings based on performance

    Usage:
        from quantetf.regime import RegimeDetector, RegimeIndicators, RegimeAnalyzer

        detector = RegimeDetector()
        indicators = RegimeIndicators(data_access)
        analyzer = RegimeAnalyzer(detector, indicators)

        # Label historical regimes
        labels = analyzer.label_history(start, end)

        # Analyze strategy performance by regime
        results = analyzer.analyze_multiple_strategies(
            {"mom": mom_returns, "vol_adj": vol_adj_returns},
            labels
        )

        # Get optimal mapping
        mapping = analyzer.compute_regime_mapping(results)
    """

    def __init__(
        self,
        detector: RegimeDetector,
        indicators: RegimeIndicators,
    ):
        """Initialize analyzer.

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
        """Label historical dates with their regime.

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
        lookback = (end_date - start_date).days + 250  # Extra for 200MA warmup
        spy_data = self.indicators.get_spy_data(
            as_of=end_date,
            lookback_days=lookback,
        )
        vix_data = self.indicators.get_vix(
            as_of=end_date,
            lookback_days=(end_date - start_date).days + 30,
        )

        # Filter to requested date range
        spy_data = spy_data[
            (spy_data.index >= start_date) & (spy_data.index <= end_date)
        ]

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
            if pd.isna(spy_200ma) or pd.isna(vix):
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
        if not df.empty:
            df.set_index("date", inplace=True)

        logger.info(
            f"Labeled {len(df)} trading days from {start_date.date()} to {end_date.date()}. "
            f"Regime distribution: {df['regime'].value_counts().to_dict() if not df.empty else {}}"
        )

        return df

    def analyze_strategy_by_regime(
        self,
        strategy_returns: pd.Series,
        regime_labels: pd.DataFrame,
        strategy_name: str = "strategy",
    ) -> pd.DataFrame:
        """Calculate strategy performance metrics for each regime.

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
        """Analyze multiple strategies across all regimes.

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

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()

    def compute_regime_mapping(
        self,
        analysis_results: pd.DataFrame,
        metric: str = "sharpe_ratio",
        min_days: int = 20,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute optimal regime→strategy mapping.

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
                "num_days": int(best_row["num_days"]),
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
        """Generate regime analysis report files.

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
