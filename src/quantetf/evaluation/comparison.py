"""Strategy comparison and analysis tools.

This module provides functionality to compare multiple strategy backtests,
generate comparison tables, visualizations, and statistical significance tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from quantetf.evaluation.metrics import (
    sharpe, sortino_ratio, calmar_ratio, max_drawdown, cagr, win_rate,
    value_at_risk, conditional_value_at_risk, information_ratio
)


@dataclass
class StrategyResult:
    """Container for a single strategy's backtest results.

    Attributes:
        name: Strategy name/identifier
        config: Configuration dictionary
        metrics: Performance metrics dictionary
        equity_curve: Time series of portfolio NAV
        weights_history: Time series of portfolio weights
        holdings_history: Time series of holdings (shares)
        backtest_dir: Path to backtest results directory
    """
    name: str
    config: Dict
    metrics: Dict
    equity_curve: pd.Series
    weights_history: pd.DataFrame
    holdings_history: pd.DataFrame
    backtest_dir: Path


def load_backtest_result(backtest_dir: Path | str) -> StrategyResult:
    """Load a backtest result from directory.

    Args:
        backtest_dir: Path to backtest results directory containing:
            - config.json
            - metrics.json
            - equity_curve.csv
            - weights_history.csv
            - holdings_history.csv

    Returns:
        StrategyResult object with all loaded data

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data format is invalid

    Example:
        >>> result = load_backtest_result('artifacts/backtests/20260111_023934_momentum-ew-top5')
        >>> print(result.name)
        'momentum-ew-top5'
        >>> print(result.metrics['sharpe_ratio'])
        1.45
    """
    backtest_dir = Path(backtest_dir)

    if not backtest_dir.exists():
        raise FileNotFoundError(f"Backtest directory not found: {backtest_dir}")

    # Load config
    config_path = backtest_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # Load metrics
    metrics_path = backtest_dir / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Load equity curve
    equity_path = backtest_dir / 'equity_curve.csv'
    if not equity_path.exists():
        raise FileNotFoundError(f"Equity curve not found: {equity_path}")
    equity_df = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    equity_curve = equity_df['nav']

    # Load weights history
    weights_path = backtest_dir / 'weights_history.csv'
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights history not found: {weights_path}")
    weights_history = pd.read_csv(weights_path, index_col=0, parse_dates=True)

    # Load holdings history
    holdings_path = backtest_dir / 'holdings_history.csv'
    if not holdings_path.exists():
        raise FileNotFoundError(f"Holdings history not found: {holdings_path}")
    holdings_history = pd.read_csv(holdings_path, index_col=0, parse_dates=True)

    # Extract strategy name from config or directory name
    strategy_name = config.get('strategy', backtest_dir.name)

    return StrategyResult(
        name=strategy_name,
        config=config,
        metrics=metrics,
        equity_curve=equity_curve,
        weights_history=weights_history,
        holdings_history=holdings_history,
        backtest_dir=backtest_dir
    )


def compute_comparison_metrics(results: List[StrategyResult]) -> pd.DataFrame:
    """Compute comprehensive metrics for all strategies.

    Args:
        results: List of StrategyResult objects to compare

    Returns:
        DataFrame with strategies as rows and metrics as columns

    Example:
        >>> results = [load_backtest_result(dir1), load_backtest_result(dir2)]
        >>> df = compute_comparison_metrics(results)
        >>> print(df[['total_return', 'sharpe_ratio', 'max_drawdown']])
    """
    comparison_data = []

    for result in results:
        # Calculate returns for additional metrics
        returns = result.equity_curve.pct_change().dropna()

        # Compute comprehensive metrics
        metrics_dict = {
            'strategy': result.name,
            'total_return': result.metrics.get('total_return', np.nan),
            'cagr': cagr(result.equity_curve) if len(result.equity_curve) > 1 else np.nan,
            'sharpe_ratio': result.metrics.get('sharpe_ratio', sharpe(returns) if len(returns) > 0 else np.nan),
            'sortino_ratio': sortino_ratio(returns) if len(returns) > 0 else np.nan,
            'calmar_ratio': calmar_ratio(result.equity_curve) if len(result.equity_curve) > 1 else np.nan,
            'max_drawdown': result.metrics.get('max_drawdown', max_drawdown(result.equity_curve)),
            'win_rate': win_rate(returns) if len(returns) > 0 else np.nan,
            'var_95': value_at_risk(returns, confidence_level=0.95) if len(returns) > 0 else np.nan,
            'cvar_95': conditional_value_at_risk(returns, confidence_level=0.95) if len(returns) > 0 else np.nan,
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else np.nan,
            'total_costs': result.metrics.get('total_costs', np.nan),
            'num_rebalances': result.metrics.get('num_rebalances', np.nan),
            'final_nav': result.metrics.get('final_nav', result.equity_curve.iloc[-1] if len(result.equity_curve) > 0 else np.nan),
            'initial_nav': result.metrics.get('initial_nav', result.equity_curve.iloc[0] if len(result.equity_curve) > 0 else np.nan),
        }

        comparison_data.append(metrics_dict)

    df = pd.DataFrame(comparison_data)

    if len(df) > 0:
        df.set_index('strategy', inplace=True)

    return df


def compute_returns_correlation(results: List[StrategyResult]) -> pd.DataFrame:
    """Compute correlation matrix of strategy returns.

    Args:
        results: List of StrategyResult objects

    Returns:
        Correlation matrix of daily returns between strategies

    Example:
        >>> results = [load_backtest_result(dir1), load_backtest_result(dir2)]
        >>> corr = compute_returns_correlation(results)
        >>> print(corr)
    """
    returns_dict = {}

    for result in results:
        returns = result.equity_curve.pct_change().dropna()
        returns_dict[result.name] = returns

    # Align all returns to common dates
    returns_df = pd.DataFrame(returns_dict)

    return returns_df.corr()


def sharpe_ratio_ttest(result1: StrategyResult, result2: StrategyResult) -> Dict:
    """Perform t-test for difference in Sharpe ratios.

    Uses the Jobson-Korkie test for comparing Sharpe ratios.

    Args:
        result1: First strategy result
        result2: Second strategy result

    Returns:
        Dictionary with t-statistic, p-value, and significance flag

    Example:
        >>> r1 = load_backtest_result(dir1)
        >>> r2 = load_backtest_result(dir2)
        >>> test = sharpe_ratio_ttest(r1, r2)
        >>> print(f"p-value: {test['p_value']:.4f}, significant: {test['is_significant']}")
    """
    returns1 = result1.equity_curve.pct_change().dropna()
    returns2 = result2.equity_curve.pct_change().dropna()

    # Align returns
    aligned = pd.DataFrame({'r1': returns1, 'r2': returns2}).dropna()

    if len(aligned) < 2:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'is_significant': False,
            'message': 'Insufficient data for t-test'
        }

    r1 = aligned['r1'].values
    r2 = aligned['r2'].values

    # Compute Sharpe ratios
    sr1 = np.mean(r1) / np.std(r1, ddof=1) if np.std(r1, ddof=1) > 0 else 0
    sr2 = np.mean(r2) / np.std(r2, ddof=1) if np.std(r2, ddof=1) > 0 else 0

    # Compute standard error using Jobson-Korkie method
    n = len(aligned)
    rho = np.corrcoef(r1, r2)[0, 1]

    var1 = np.var(r1, ddof=1)
    var2 = np.var(r2, ddof=1)

    if var1 == 0 or var2 == 0:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'is_significant': False,
            'message': 'Zero variance in returns'
        }

    # Simplified Jobson-Korkie variance formula
    theta = (1.0 / n) * (2 - 2 * rho * np.sqrt(var1 / var2) - 0.5 * sr1**2 - 0.5 * sr2**2 +
                         sr1 * sr2 * rho)

    if theta <= 0:
        # Fall back to simple difference test
        t_stat, p_value = stats.ttest_rel(r1, r2)
    else:
        std_error = np.sqrt(theta)
        t_stat = (sr1 - sr2) / std_error if std_error > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'is_significant': bool(p_value < 0.05),
        'sharpe_1': float(sr1),
        'sharpe_2': float(sr2),
        'message': f"Sharpe difference: {sr1:.3f} vs {sr2:.3f}"
    }


def create_equity_overlay_chart(results: List[StrategyResult],
                                output_path: Optional[Path] = None,
                                figsize: tuple = (12, 6)) -> plt.Figure:
    """Create overlay chart of equity curves for all strategies.

    Args:
        results: List of StrategyResult objects
        output_path: Optional path to save chart
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object

    Example:
        >>> results = [load_backtest_result(dir1), load_backtest_result(dir2)]
        >>> fig = create_equity_overlay_chart(results, output_path='comparison.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    for result in results:
        # Normalize to start at 100
        normalized = 100 * result.equity_curve / result.equity_curve.iloc[0]
        ax.plot(normalized.index, normalized.values, label=result.name, linewidth=2)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized NAV (Base=100)', fontsize=12)
    ax.set_title('Strategy Comparison - Equity Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def create_risk_return_scatter(results: List[StrategyResult],
                               output_path: Optional[Path] = None,
                               figsize: tuple = (10, 8)) -> plt.Figure:
    """Create risk-return scatter plot for all strategies.

    Args:
        results: List of StrategyResult objects
        output_path: Optional path to save chart
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object

    Example:
        >>> results = [load_backtest_result(dir1), load_backtest_result(dir2)]
        >>> fig = create_risk_return_scatter(results, output_path='risk_return.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    risk_values = []
    return_values = []
    labels = []

    for result in results:
        returns = result.equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        annual_return = cagr(result.equity_curve)

        risk_values.append(volatility)
        return_values.append(annual_return)
        labels.append(result.name)

    # Plot points
    ax.scatter(risk_values, return_values, s=200, alpha=0.6, edgecolors='black', linewidth=2)

    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, (risk_values[i], return_values[i]),
                   fontsize=10, ha='center', va='bottom',
                   xytext=(0, 10), textcoords='offset points')

    ax.set_xlabel('Annualized Volatility', fontsize=12)
    ax.set_ylabel('CAGR', fontsize=12)
    ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Format as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def create_comparison_table_html(comparison_df: pd.DataFrame,
                                 output_path: Optional[Path] = None) -> str:
    """Generate HTML table from comparison DataFrame.

    Args:
        comparison_df: DataFrame from compute_comparison_metrics()
        output_path: Optional path to save HTML file

    Returns:
        HTML string

    Example:
        >>> df = compute_comparison_metrics(results)
        >>> html = create_comparison_table_html(df, 'comparison_table.html')
    """
    # Format specific columns as percentages
    formatted_df = comparison_df.copy()

    pct_cols = ['total_return', 'cagr', 'max_drawdown', 'win_rate', 'var_95', 'cvar_95', 'volatility']
    for col in pct_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:.2%}' if not pd.isna(x) else 'N/A')

    # Format ratio columns
    ratio_cols = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    for col in ratio_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:.3f}' if not pd.isna(x) else 'N/A')

    # Format currency columns
    currency_cols = ['final_nav', 'initial_nav', 'total_costs']
    for col in currency_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f'${x:,.2f}' if not pd.isna(x) else 'N/A')

    # Format integer columns
    int_cols = ['num_rebalances']
    for col in int_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f'{int(x)}' if not pd.isna(x) else 'N/A')

    # Generate HTML with styling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: right;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #ddd;
            }}
            .strategy-name {{
                text-align: left;
                font-weight: bold;
            }}
            h1 {{
                color: #333;
            }}
        </style>
    </head>
    <body>
        <h1>Strategy Comparison Table</h1>
        {formatted_df.to_html(classes='comparison-table', escape=False)}
    </body>
    </html>
    """

    if output_path:
        with open(output_path, 'w') as f:
            f.write(html)

    return html


def generate_comparison_report(results: List[StrategyResult],
                               output_dir: Path | str,
                               report_name: str = 'comparison_report') -> Dict[str, Path]:
    """Generate comprehensive comparison report with all charts and tables.

    Args:
        results: List of StrategyResult objects to compare
        output_dir: Directory to save report files
        report_name: Base name for report files

    Returns:
        Dictionary mapping file types to their paths

    Example:
        >>> results = [load_backtest_result(dir1), load_backtest_result(dir2)]
        >>> paths = generate_comparison_report(results, 'artifacts/comparisons')
        >>> print(paths['html'])
        PosixPath('artifacts/comparisons/comparison_report.html')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    # 1. Compute metrics comparison table
    comparison_df = compute_comparison_metrics(results)
    csv_path = output_dir / f'{report_name}_metrics.csv'
    comparison_df.to_csv(csv_path)
    output_paths['metrics_csv'] = csv_path

    # 2. Generate HTML table
    html_path = output_dir / f'{report_name}_table.html'
    create_comparison_table_html(comparison_df, html_path)
    output_paths['table_html'] = html_path

    # 3. Create equity overlay chart
    equity_chart_path = output_dir / f'{report_name}_equity.png'
    create_equity_overlay_chart(results, equity_chart_path)
    output_paths['equity_chart'] = equity_chart_path

    # 4. Create risk-return scatter
    scatter_path = output_dir / f'{report_name}_risk_return.png'
    create_risk_return_scatter(results, scatter_path)
    output_paths['risk_return_chart'] = scatter_path

    # 5. Compute returns correlation
    if len(results) > 1:
        corr_df = compute_returns_correlation(results)
        corr_path = output_dir / f'{report_name}_correlation.csv'
        corr_df.to_csv(corr_path)
        output_paths['correlation_csv'] = corr_path

    # 6. Perform pairwise Sharpe ratio tests
    if len(results) > 1:
        test_results = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                test = sharpe_ratio_ttest(results[i], results[j])
                test_results.append({
                    'strategy_1': results[i].name,
                    'strategy_2': results[j].name,
                    **test
                })

        tests_df = pd.DataFrame(test_results)
        tests_path = output_dir / f'{report_name}_significance_tests.csv'
        tests_df.to_csv(tests_path, index=False)
        output_paths['significance_tests'] = tests_path

    return output_paths
