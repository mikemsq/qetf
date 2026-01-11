from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityIssue:
    severity: str  # "info", "warning", "error"
    message: str
    context: dict


def check_missingness(df: pd.DataFrame, *, max_missing_frac: float = 0.05) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    missing = df.isna().mean()
    bad = missing[missing > max_missing_frac]
    for col, frac in bad.items():
        issues.append(
            DataQualityIssue(
                severity="warning",
                message="Column exceeds missingness threshold",
                context={"column": col, "missing_frac": float(frac), "threshold": max_missing_frac},
            )
        )
    return issues


def winsorize_returns(r: pd.DataFrame, *, lower_q: float = 0.001, upper_q: float = 0.999) -> pd.DataFrame:
    """Optional helper for robust statistics; do not use to hide data issues."""
    lo = r.quantile(lower_q)
    hi = r.quantile(upper_q)
    return r.clip(lower=lo, upper=hi, axis=1)


def check_missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing data across all columns or tickers.

    Args:
        df: DataFrame with MultiIndex columns (ticker, price_type) or simple columns

    Returns:
        DataFrame with missing data summary
        Columns: ticker/column, nan_pct, total_rows, missing_rows

    Example:
        >>> df = pd.read_parquet('snapshot.parquet')
        >>> missing = check_missing_data_summary(df)
        >>> print(missing[missing['nan_pct'] > 5.0])
    """
    results = []

    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex columns (Ticker, Price)
        tickers = df.columns.get_level_values(0).unique()
        for ticker in tickers:
            ticker_cols = [col for col in df.columns if col[0] == ticker]
            ticker_data = df[ticker_cols]
            nan_pct = (ticker_data.isna().sum().sum() / (len(df) * len(ticker_cols))) * 100
            missing_rows = ticker_data.isna().any(axis=1).sum()

            results.append({
                'ticker': ticker,
                'nan_pct': nan_pct,
                'total_rows': len(df),
                'missing_rows': missing_rows,
            })
    else:
        # Simple columns
        for col in df.columns:
            nan_pct = (df[col].isna().sum() / len(df)) * 100
            results.append({
                'ticker': str(col),
                'nan_pct': nan_pct,
                'total_rows': len(df),
                'missing_rows': df[col].isna().sum(),
            })

    return pd.DataFrame(results)


def detect_price_spikes_simple(
    df: pd.DataFrame,
    threshold: float = 0.15
) -> pd.DataFrame:
    """Detect suspicious single-day price spikes across all tickers.

    Args:
        df: DataFrame with MultiIndex columns (ticker, price_type) or simple columns
        threshold: Return threshold for spike (default 0.15 = 15%)

    Returns:
        DataFrame of detected spikes with date, ticker, return

    Example:
        >>> spikes = detect_price_spikes_simple(df, threshold=0.15)
        >>> print(f"Found {len(spikes)} suspicious spikes")
    """
    spikes = []

    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique()
        for ticker in tickers:
            close_col = (ticker, 'Close')
            if close_col in df.columns:
                prices = df[close_col].dropna()
                if len(prices) < 2:
                    continue
                returns = prices.pct_change()
                spike_mask = returns.abs() > threshold
                for date in returns[spike_mask].index:
                    spikes.append({
                        'date': date,
                        'ticker': ticker,
                        'return_pct': returns.loc[date] * 100,
                    })
    else:
        for col in df.columns:
            prices = df[col].dropna()
            if len(prices) < 2:
                continue
            returns = prices.pct_change()
            spike_mask = returns.abs() > threshold
            for date in returns[spike_mask].index:
                spikes.append({
                    'date': date,
                    'ticker': str(col),
                    'return_pct': returns.loc[date] * 100,
                })

    return pd.DataFrame(spikes)


def calculate_dataset_quality_score(df: pd.DataFrame) -> dict:
    """Calculate overall quality score for dataset.

    Scoring:
    - Base score: 100
    - Missing data: -10 points per 1% missing (max -30)
    - Price spikes: -5 points per spike (max -20)

    Args:
        df: DataFrame to analyze

    Returns:
        Dict with score (0-100) and breakdown

    Example:
        >>> score = calculate_dataset_quality_score(df)
        >>> print(f"Quality score: {score['total']}/100")
    """
    score = 100
    breakdown = {}

    # Missing data penalty
    missing_summary = check_missing_data_summary(df)
    avg_missing_pct = missing_summary['nan_pct'].mean()
    missing_penalty = min(30, int(avg_missing_pct * 10))
    score -= missing_penalty
    breakdown['missing_penalty'] = missing_penalty
    breakdown['avg_missing_pct'] = avg_missing_pct

    # Price spike penalty
    spikes = detect_price_spikes_simple(df, threshold=0.15)
    spike_count = len(spikes)
    spike_penalty = min(20, spike_count * 5)
    score -= spike_penalty
    breakdown['spike_penalty'] = spike_penalty
    breakdown['spike_count'] = spike_count

    breakdown['total'] = max(0, score)
    return breakdown


def generate_quality_report_text(df: pd.DataFrame) -> str:
    """Generate comprehensive quality report as text.

    Args:
        df: DataFrame to analyze

    Returns:
        Report as string

    Example:
        >>> report = generate_quality_report_text(df)
        >>> print(report)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("DATA QUALITY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Dataset info
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    if isinstance(df.index, pd.DatetimeIndex):
        lines.append(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        lines.append(f"Duration: {(df.index.max() - df.index.min()).days} days")
    lines.append("")

    # Missing data
    lines.append("-" * 70)
    lines.append("Missing Data Summary")
    lines.append("-" * 70)
    missing = check_missing_data_summary(df)
    if len(missing) > 0:
        # Show top 10 by missing percentage
        top_missing = missing.nlargest(10, 'nan_pct')
        lines.append(top_missing.to_string(index=False))
        lines.append(f"\nAverage missing: {missing['nan_pct'].mean():.2f}%")
    else:
        lines.append("No missing data detected.")
    lines.append("")

    # Price spikes
    lines.append("-" * 70)
    lines.append("Price Spikes (>15% single-day moves)")
    lines.append("-" * 70)
    spikes = detect_price_spikes_simple(df, threshold=0.15)
    if len(spikes) > 0:
        lines.append(f"Found {len(spikes)} suspicious price spikes:")
        lines.append(spikes.head(15).to_string(index=False))
        if len(spikes) > 15:
            lines.append(f"\n... and {len(spikes) - 15} more")
    else:
        lines.append("No suspicious price spikes detected (>15%).")
    lines.append("")

    # Overall score
    lines.append("-" * 70)
    lines.append("Overall Quality Score")
    lines.append("-" * 70)
    score = calculate_dataset_quality_score(df)
    lines.append(f"Total Score: {score['total']}/100")
    lines.append(f"  Missing Data: -{score['missing_penalty']} (avg {score['avg_missing_pct']:.2f}%)")
    lines.append(f"  Price Spikes: -{score['spike_penalty']} ({score['spike_count']} spikes)")

    if score['total'] >= 90:
        grade = "A (Excellent)"
    elif score['total'] >= 75:
        grade = "B (Good)"
    elif score['total'] >= 60:
        grade = "C (Fair)"
    else:
        grade = "D (Poor - investigate issues)"

    lines.append(f"\n  Grade: {grade}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
