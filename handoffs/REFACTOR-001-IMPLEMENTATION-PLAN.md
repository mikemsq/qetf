# REFACTOR-001: Active Returns Focus - Detailed Implementation Plan

**Created:** January 12, 2026
**Status:** Ready for Implementation
**Priority:** CRITICAL
**Estimated Time:** 2-3 hours

---

## Executive Summary

After analyzing the existing codebase, the good news is that **most building blocks already exist**. The main work involves:
1. Adding a single helper function to compute active metrics
2. Updating the notebook to load SPY data and overlay comparisons
3. Minor updates to comparison scripts to make SPY the default

**Current State:** Implementation is 70% aligned with requirements
**Effort Required:** 30% new code, 70% reorganization/presentation

---

## Detailed Analysis

### What Already Exists ✅

**Metrics Module (`src/quantetf/evaluation/metrics.py`):**
- ✅ `information_ratio()` - calculates IR vs benchmark (lines 337-406)
- ✅ `sharpe()`, `sortino_ratio()`, `calmar_ratio()`, `max_drawdown()` - all strategy metrics
- ✅ `value_at_risk()`, `conditional_value_at_risk()` - risk metrics
- ✅ `rolling_sharpe_ratio()` - time-series metrics

**Benchmark Module (`src/quantetf/evaluation/benchmarks.py`):**
- ✅ `run_spy_benchmark()` - SPY buy-and-hold implementation (lines 42-149)
- ✅ `run_60_40_benchmark()` - 60/40 portfolio
- ✅ Complete benchmark framework with 5 benchmarks

**Scripts:**
- ✅ `scripts/benchmark_comparison.py` - runs strategy vs benchmarks
- ✅ `scripts/compare_strategies.py` - compares multiple strategies

**Notebook (`notebooks/backtest_analysis.ipynb`):**
- ✅ 8 comprehensive visualizations
- ✅ Equity curve plotting infrastructure
- ✅ Metrics display capabilities

### What Needs to Change 🔧

**1. Metrics Module - Add Active Metrics Helper**
**2. Notebook - Add SPY Comparison Throughout**
**3. Scripts - Make SPY Comparison Default**

---

## File-by-File Implementation Plan

### File 1: `src/quantetf/evaluation/metrics.py`

**Location:** After `information_ratio()` function (after line 406)

**Add New Function:**
```python
def calculate_active_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> dict[str, float]:
    """Calculate comprehensive active performance metrics vs benchmark.

    This is the STANDARD format for all performance reporting in QuantETF.
    The goal is to beat SPY, so active metrics are the primary focus.

    Args:
        strategy_returns: Strategy period returns (e.g., daily)
        benchmark_returns: Benchmark period returns (typically SPY)
        periods_per_year: Trading periods per year (default 252 for daily)
        risk_free_rate: Annualized risk-free rate (default 0.0)

    Returns:
        Dictionary with comprehensive active metrics:

        Strategy Metrics:
        - strategy_total_return: Cumulative strategy return
        - strategy_sharpe: Strategy Sharpe ratio
        - strategy_sortino: Strategy Sortino ratio
        - strategy_max_dd: Strategy maximum drawdown
        - strategy_cagr: Strategy CAGR

        Benchmark Metrics:
        - benchmark_total_return: Cumulative benchmark return
        - benchmark_sharpe: Benchmark Sharpe ratio
        - benchmark_sortino: Benchmark Sortino ratio
        - benchmark_max_dd: Benchmark maximum drawdown
        - benchmark_cagr: Benchmark CAGR

        Active Metrics (THE KEY METRICS):
        - active_return: Total excess return vs benchmark
        - active_return_ann: Annualized excess return
        - tracking_error: Volatility of excess returns (annualized)
        - information_ratio: Active return / tracking error
        - beta: Strategy beta to benchmark
        - alpha: Jensen's alpha (risk-adjusted excess return)
        - sharpe_difference: Strategy Sharpe - Benchmark Sharpe
        - win_rate: % of periods where strategy beat benchmark

    Raises:
        ValueError: If returns series are empty or misaligned

    Example:
        >>> strategy_rets = pd.Series([0.01, 0.02, -0.01, 0.015])
        >>> spy_rets = pd.Series([0.008, 0.015, -0.008, 0.012])
        >>> metrics = calculate_active_metrics(strategy_rets, spy_rets)
        >>> print(f"Beat SPY by {metrics['active_return']:.2%}")
        >>> print(f"Information Ratio: {metrics['information_ratio']:.2f}")
    """
    # Input validation
    strat = strategy_returns.dropna()
    bench = benchmark_returns.dropna()

    if len(strat) == 0:
        raise ValueError("Strategy returns series is empty or all NaN")
    if len(bench) == 0:
        raise ValueError("Benchmark returns series is empty or all NaN")

    # Align indices
    aligned_strat = strat.reindex(bench.index)
    aligned_bench = bench.reindex(strat.index)

    # Drop any NaN after alignment
    mask = ~(aligned_strat.isna() | aligned_bench.isna())
    aligned_strat = aligned_strat[mask]
    aligned_bench = aligned_bench[mask]

    if len(aligned_strat) == 0:
        raise ValueError("No overlapping returns between strategy and benchmark")

    # Calculate strategy metrics
    strat_equity = (1 + aligned_strat).cumprod()
    bench_equity = (1 + aligned_bench).cumprod()

    strategy_metrics = {
        'strategy_total_return': float(strat_equity.iloc[-1] - 1.0),
        'strategy_sharpe': sharpe(aligned_strat, periods_per_year),
        'strategy_sortino': sortino_ratio(aligned_strat, risk_free_rate, periods_per_year),
        'strategy_max_dd': max_drawdown(strat_equity),
        'strategy_cagr': cagr(strat_equity, periods_per_year),
    }

    # Calculate benchmark metrics
    benchmark_metrics = {
        'benchmark_total_return': float(bench_equity.iloc[-1] - 1.0),
        'benchmark_sharpe': sharpe(aligned_bench, periods_per_year),
        'benchmark_sortino': sortino_ratio(aligned_bench, risk_free_rate, periods_per_year),
        'benchmark_max_dd': max_drawdown(bench_equity),
        'benchmark_cagr': cagr(bench_equity, periods_per_year),
    }

    # Calculate active metrics
    excess_returns = aligned_strat - aligned_bench
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)

    # Information ratio
    ir = information_ratio(aligned_strat, aligned_bench, periods_per_year)

    # Beta and alpha (using linear regression)
    if aligned_bench.std() > 0:
        covariance = aligned_strat.cov(aligned_bench)
        beta = covariance / aligned_bench.var()

        # Jensen's alpha: alpha = Rp - [Rf + beta * (Rb - Rf)]
        rf_per_period = risk_free_rate / periods_per_year
        alpha = (aligned_strat.mean() - rf_per_period) - beta * (aligned_bench.mean() - rf_per_period)
        alpha_annualized = alpha * periods_per_year
    else:
        beta = 0.0
        alpha_annualized = 0.0

    # Win rate (% periods beating benchmark)
    win_rate_pct = float((excess_returns > 0).sum() / len(excess_returns) * 100.0)

    active_metrics = {
        'active_return': float(strategy_metrics['strategy_total_return'] - benchmark_metrics['benchmark_total_return']),
        'active_return_ann': float(excess_returns.mean() * periods_per_year),
        'tracking_error': float(tracking_error),
        'information_ratio': float(ir),
        'beta': float(beta),
        'alpha': float(alpha_annualized),
        'sharpe_difference': float(strategy_metrics['strategy_sharpe'] - benchmark_metrics['benchmark_sharpe']),
        'win_rate': float(win_rate_pct),
    }

    # Combine all metrics
    return {**strategy_metrics, **benchmark_metrics, **active_metrics}
```

**Tests to Add:** `tests/test_advanced_metrics.py` (add to existing file)
```python
def test_calculate_active_metrics_basic():
    """Test basic active metrics calculation."""
    strategy = pd.Series([0.01, 0.02, -0.01, 0.015, 0.02])
    benchmark = pd.Series([0.008, 0.015, -0.008, 0.012, 0.018])

    metrics = calculate_active_metrics(strategy, benchmark)

    # Check all keys present
    assert 'active_return' in metrics
    assert 'information_ratio' in metrics
    assert 'beta' in metrics
    assert 'alpha' in metrics
    assert 'tracking_error' in metrics
    assert 'win_rate' in metrics

    # Check strategy outperformed
    assert metrics['active_return'] > 0

def test_calculate_active_metrics_underperformance():
    """Test when strategy underperforms benchmark."""
    strategy = pd.Series([0.005, 0.01, -0.015, 0.008, 0.01])
    benchmark = pd.Series([0.01, 0.02, -0.01, 0.015, 0.02])

    metrics = calculate_active_metrics(strategy, benchmark)

    assert metrics['active_return'] < 0
    assert metrics['win_rate'] < 50.0

def test_calculate_active_metrics_perfect_tracking():
    """Test when strategy perfectly tracks benchmark."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.02])

    metrics = calculate_active_metrics(returns, returns)

    assert np.isclose(metrics['active_return'], 0.0, atol=1e-10)
    assert np.isclose(metrics['tracking_error'], 0.0, atol=1e-10)
    assert np.isclose(metrics['beta'], 1.0, atol=1e-6)
```

**Estimated Time:** 1 hour (function + tests)

---

### File 2: `notebooks/backtest_analysis.ipynb`

**Changes Required:** 6 cells to modify

#### Cell 1: Load Data (add SPY loading)

**Current:** Only loads strategy backtest
**New:** Also load SPY benchmark data

**After line:** `print(f"Final NAV: ${metrics['final_nav']:,.2f}")`

**Add:**
```python
# Load SPY benchmark data for comparison
print("\n" + "="*60)
print("Loading SPY Benchmark Data")
print("="*60)

from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.backtest.simple_engine import BacktestConfig
from quantetf.evaluation.benchmarks import run_spy_benchmark
from quantetf.evaluation.metrics import calculate_active_metrics

# Determine which snapshot was used
snapshot_path = config.get('snapshot_dir', 'data/snapshots/snapshot_5yr_20etfs')
print(f"Using snapshot: {snapshot_path}")

# Load snapshot data
store = SnapshotDataStore(snapshot_path)

# Create config for SPY benchmark
spy_config = BacktestConfig(
    start_date=equity_df.index.min(),
    end_date=equity_df.index.max(),
    initial_capital=config.get('initial_capital', 100000.0),
    rebalance_frequency='monthly'
)

# Run SPY benchmark
spy_result = run_spy_benchmark(config=spy_config, store=store)

# Calculate strategy returns
strategy_returns = equity_df['returns'].fillna(0)

# Calculate SPY returns
spy_returns = spy_result.equity_curve['nav'].pct_change().fillna(0)

# Align indices
aligned_dates = strategy_returns.index.intersection(spy_returns.index)
strategy_returns_aligned = strategy_returns[aligned_dates]
spy_returns_aligned = spy_returns[aligned_dates]

# Calculate active metrics
active_metrics = calculate_active_metrics(
    strategy_returns_aligned,
    spy_returns_aligned,
    periods_per_year=252
)

print(f"\nSPY Total Return: {active_metrics['benchmark_total_return']:.2%}")
print(f"SPY Sharpe Ratio: {active_metrics['benchmark_sharpe']:.2f}")
print(f"SPY Max Drawdown: {active_metrics['benchmark_max_dd']:.2%}")

print(f"\n{'='*60}")
print("🎯 ACTIVE PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f"Strategy Return:     {active_metrics['strategy_total_return']:>8.2%}")
print(f"SPY Return:          {active_metrics['benchmark_total_return']:>8.2%}")
print(f"Active Return:       {active_metrics['active_return']:>8.2%}  {'✅ OUTPERFORM' if active_metrics['active_return'] > 0 else '❌ UNDERPERFORM'}")
print(f"\nInformation Ratio:   {active_metrics['information_ratio']:>8.2f}")
print(f"Tracking Error:      {active_metrics['tracking_error']:>8.2%}")
print(f"Beta:                {active_metrics['beta']:>8.2f}")
print(f"Alpha:               {active_metrics['alpha']:>8.2%}")
print(f"\nStrategy Sharpe:     {active_metrics['strategy_sharpe']:>8.2f}")
print(f"SPY Sharpe:          {active_metrics['benchmark_sharpe']:>8.2f}")
print(f"Sharpe Difference:   {active_metrics['sharpe_difference']:>8.2f}")
print(f"\nWin Rate vs SPY:     {active_metrics['win_rate']:>8.1f}%")
print(f"{'='*60}\n")
```

#### Cell 2: Update Equity Curve Chart (Add SPY Overlay)

**Current:** Shows only strategy equity curve
**New:** Shows strategy vs SPY overlaid with outperformance shading

**Replace the entire viz1_code cell with:**
```python
# Calculate cumulative equity curves
strategy_equity = equity_df['nav']
spy_equity = spy_result.equity_curve['nav']

# Align to same index
aligned_dates = strategy_equity.index.intersection(spy_equity.index)
strategy_equity_aligned = strategy_equity[aligned_dates]
spy_equity_aligned = spy_equity[aligned_dates]

# Normalize to same starting point for comparison
strategy_norm = strategy_equity_aligned / strategy_equity_aligned.iloc[0] * 100
spy_norm = spy_equity_aligned / spy_equity_aligned.iloc[0] * 100

# Calculate drawdowns
strategy_dd = (strategy_equity_aligned / strategy_equity_aligned.cummax()) - 1
spy_dd = (spy_equity_aligned / spy_equity_aligned.cummax()) - 1

# Create figure with two y-axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

# ============ TOP: Equity Curves with Outperformance Shading ============
ax1.plot(strategy_norm.index, strategy_norm.values,
         color='darkblue', linewidth=2.5, label='Strategy', zorder=3)
ax1.plot(spy_norm.index, spy_norm.values,
         color='gray', linewidth=2, label='SPY (Benchmark)', alpha=0.8, zorder=2)

# Shade outperformance/underperformance
ax1.fill_between(strategy_norm.index, strategy_norm.values, spy_norm.values,
                 where=(strategy_norm >= spy_norm),
                 alpha=0.25, color='green', label='Outperformance', zorder=1)
ax1.fill_between(strategy_norm.index, strategy_norm.values, spy_norm.values,
                 where=(strategy_norm < spy_norm),
                 alpha=0.25, color='red', label='Underperformance', zorder=1)

ax1.set_ylabel('Equity (Normalized to 100)', fontsize=12, fontweight='bold')
ax1.set_title('Strategy vs SPY Benchmark - Equity Curve',
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)

# Add performance summary text box
textstr = f"""
Strategy: {active_metrics['strategy_total_return']:.1%} return
SPY: {active_metrics['benchmark_total_return']:.1%} return
Active: {active_metrics['active_return']:+.1%} ({'+' if active_metrics['active_return'] > 0 else ''}{'beat' if active_metrics['active_return'] > 0 else 'trail'} SPY)
IR: {active_metrics['information_ratio']:.2f}
"""
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr.strip(), transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# ============ BOTTOM: Comparative Drawdowns ============
ax2.fill_between(strategy_dd.index, strategy_dd.values * 100, 0,
                 color='darkblue', alpha=0.4, label='Strategy DD')
ax2.fill_between(spy_dd.index, spy_dd.values * 100, 0,
                 color='gray', alpha=0.3, label='SPY DD')
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_title('Comparative Drawdown Analysis', fontsize=12, fontweight='bold')
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Strategy Max Drawdown: {strategy_dd.min():.2%}")
print(f"SPY Max Drawdown: {spy_dd.min():.2%}")
print(f"Drawdown Difference: {(strategy_dd.min() - spy_dd.min()):.2%}")
```

#### Cell 3: Add New Cell - Active Returns Analysis

**Insert NEW CELL after viz1:**

```python
# ============================================================
# ACTIVE RETURNS OVER TIME
# ============================================================

# Calculate rolling active returns
rolling_window = 252  # 1 year
strategy_rolling_ret = strategy_returns_aligned.rolling(rolling_window).apply(
    lambda x: (1 + x).prod() - 1
)
spy_rolling_ret = spy_returns_aligned.rolling(rolling_window).apply(
    lambda x: (1 + x).prod() - 1
)
active_rolling_ret = strategy_rolling_ret - spy_rolling_ret

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top: Rolling returns
ax1.plot(strategy_rolling_ret.index, strategy_rolling_ret * 100,
         label='Strategy (Rolling 1Y)', color='darkblue', linewidth=2)
ax1.plot(spy_rolling_ret.index, spy_rolling_ret * 100,
         label='SPY (Rolling 1Y)', color='gray', linewidth=2, alpha=0.7)
ax1.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax1.set_ylabel('Rolling Return (%)', fontsize=12, fontweight='bold')
ax1.set_title('Rolling 1-Year Returns: Strategy vs SPY', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)

# Bottom: Active returns
ax2.fill_between(active_rolling_ret.index, active_rolling_ret * 100, 0,
                 where=(active_rolling_ret >= 0),
                 color='green', alpha=0.4, label='Outperformance')
ax2.fill_between(active_rolling_ret.index, active_rolling_ret * 100, 0,
                 where=(active_rolling_ret < 0),
                 color='red', alpha=0.4, label='Underperformance')
ax2.axhline(0, color='black', linestyle='-', linewidth=1.5)
ax2.set_ylabel('Active Return (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_title('Rolling 1-Year Active Returns vs SPY', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Average Rolling Active Return: {active_rolling_ret.mean():.2%}")
print(f"Best Rolling Active Return: {active_rolling_ret.max():.2%}")
print(f"Worst Rolling Active Return: {active_rolling_ret.min():.2%}")
print(f"% Time Outperforming: {(active_rolling_ret > 0).sum() / len(active_rolling_ret.dropna()) * 100:.1f}%")
```

#### Cell 4: Update Monthly Returns Heatmap (Add Comparative)

**Add at end of viz2_code cell:**
```python
# Calculate SPY monthly returns for comparison
spy_monthly_returns = spy_result.equity_curve['nav'].resample('M').last().pct_change()

print("\n" + "="*60)
print("MONTHLY PERFORMANCE COMPARISON")
print("="*60)

# Find months where strategy beat SPY
monthly_active = monthly_returns - spy_monthly_returns
months_with_data = monthly_active.dropna()

if len(months_with_data) > 0:
    outperform_months = (months_with_data > 0).sum()
    total_months = len(months_with_data)
    outperform_pct = outperform_months / total_months * 100

    print(f"Months Outperforming SPY: {outperform_months}/{total_months} ({outperform_pct:.1f}%)")
    print(f"Best Active Month: {months_with_data.max():.2%}")
    print(f"Worst Active Month: {months_with_data.min():.2%}")
    print(f"Average Active Return: {months_with_data.mean():.2%}")
```

#### Cell 5: Update Rolling Sharpe (Add SPY Comparison)

**Replace viz3_code cell with:**
```python
# Calculate rolling Sharpe for both strategy and SPY
window = 252  # 1 year

# Strategy rolling Sharpe
strategy_rolling_mean = strategy_returns_aligned.rolling(window=window).mean()
strategy_rolling_std = strategy_returns_aligned.rolling(window=window).std()
strategy_rolling_sharpe = (strategy_rolling_mean / strategy_rolling_std) * np.sqrt(252)

# SPY rolling Sharpe
spy_rolling_mean = spy_returns_aligned.rolling(window=window).mean()
spy_rolling_std = spy_returns_aligned.rolling(window=window).std()
spy_rolling_sharpe = (spy_rolling_mean / spy_rolling_std) * np.sqrt(252)

# Plot comparison
plt.figure(figsize=(14, 6))
plt.plot(strategy_rolling_sharpe.index, strategy_rolling_sharpe.values,
         linewidth=2.5, color='darkblue', label='Strategy Rolling Sharpe')
plt.plot(spy_rolling_sharpe.index, spy_rolling_sharpe.values,
         linewidth=2, color='gray', alpha=0.7, label='SPY Rolling Sharpe')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero')
plt.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1.0')

# Shade outperformance periods
plt.fill_between(strategy_rolling_sharpe.index,
                 strategy_rolling_sharpe.values,
                 spy_rolling_sharpe.values,
                 where=(strategy_rolling_sharpe >= spy_rolling_sharpe),
                 alpha=0.2, color='green', label='Strategy > SPY')

plt.title('Rolling Sharpe Ratio (252-Day): Strategy vs SPY', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sharpe Ratio', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

sharpe_diff = strategy_rolling_sharpe - spy_rolling_sharpe
print(f"Strategy Current Rolling Sharpe: {strategy_rolling_sharpe.iloc[-1]:.2f}")
print(f"SPY Current Rolling Sharpe: {spy_rolling_sharpe.iloc[-1]:.2f}")
print(f"Sharpe Difference: {sharpe_diff.iloc[-1]:+.2f}")
print(f"Average Sharpe Advantage: {sharpe_diff.mean():.2f}")
print(f"% Time with Higher Sharpe: {(sharpe_diff > 0).sum() / len(sharpe_diff.dropna()) * 100:.1f}%")
```

#### Cell 6: Update Returns Distribution (Add SPY Overlay)

**In viz5_code, after the strategy histogram, add:**
```python
# Overlay SPY returns distribution
spy_returns_aligned.plot.kde(ax=ax, color='gray', linewidth=2,
                             label='SPY KDE', linestyle='--', alpha=0.7)

# Update legend
ax.legend(loc='upper right', fontsize=10)
```

**And at the end, add comparison:**
```python
print("\n" + "="*60)
print("RETURNS DISTRIBUTION COMPARISON")
print("="*60)
print(f"{'Metric':<30} {'Strategy':>12} {'SPY':>12} {'Difference':>12}")
print("-" * 66)
print(f"{'Mean Daily Return':<30} {strategy_returns_aligned.mean():>11.4%} {spy_returns_aligned.mean():>11.4%} {(strategy_returns_aligned.mean() - spy_returns_aligned.mean()):>11.4%}")
print(f"{'Median Daily Return':<30} {strategy_returns_aligned.median():>11.4%} {spy_returns_aligned.median():>11.4%} {(strategy_returns_aligned.median() - spy_returns_aligned.median()):>11.4%}")
print(f"{'Daily Volatility':<30} {strategy_returns_aligned.std():>11.4%} {spy_returns_aligned.std():>11.4%} {(strategy_returns_aligned.std() - spy_returns_aligned.std()):>11.4%}")
print(f"{'Best Day':<30} {strategy_returns_aligned.max():>11.2%} {spy_returns_aligned.max():>11.2%} {(strategy_returns_aligned.max() - spy_returns_aligned.max()):>11.2%}")
print(f"{'Worst Day':<30} {strategy_returns_aligned.min():>11.2%} {spy_returns_aligned.min():>11.2%} {(strategy_returns_aligned.min() - spy_returns_aligned.min()):>11.2%}")
```

**Estimated Time:** 1.5 hours (6 cell modifications + testing)

---

### File 3: `scripts/compare_strategies.py`

**Change:** Make SPY benchmark comparison automatic

**Location:** In `main()` function, after loading backtests

**Add (around line 200):**
```python
# Automatically add SPY benchmark for context
logger.info("Adding SPY benchmark for comparison context")
try:
    # Load snapshot from first backtest config
    first_config_path = Path(results[0].backtest_dir) / 'config.json'
    with open(first_config_path) as f:
        first_config = json.load(f)

    snapshot_path = first_config.get('snapshot_dir', 'data/snapshots/snapshot_5yr_20etfs')
    store = SnapshotDataStore(snapshot_path)

    # Create SPY benchmark config
    spy_config = BacktestConfig(
        start_date=results[0].equity_curve.index.min(),
        end_date=results[0].equity_curve.index.max(),
        initial_capital=first_config.get('initial_capital', 100000.0),
        rebalance_frequency='monthly'
    )

    # Run SPY benchmark
    spy_result = run_spy_benchmark(config=spy_config, store=store)

    # Convert to StrategyResult format
    spy_strategy_result = StrategyResult(
        name='SPY Benchmark',
        backtest_dir='benchmark_spy',
        equity_curve=spy_result.equity_curve,
        metrics=spy_result.metrics,
        config={'description': 'SPY buy-and-hold benchmark'}
    )

    # Add to results list
    results.append(spy_strategy_result)
    logger.info("SPY benchmark added successfully")

except Exception as e:
    logger.warning(f"Could not add SPY benchmark: {e}")
```

**Add imports at top:**
```python
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.backtest.simple_engine import BacktestConfig
from quantetf.evaluation.benchmarks import run_spy_benchmark
```

**Estimated Time:** 20 minutes

---

### File 4: `scripts/benchmark_comparison.py`

**Change:** Minor - already SPY-focused, just ensure report emphasizes active returns

**Location:** In report generation section (around line 350)

**Modify summary output:**
```python
# Emphasize active returns in console output
print("\n" + "="*70)
print("🎯 ACTIVE PERFORMANCE VS SPY")
print("="*70)

strategy_metrics = # ... load from results
spy_metrics = # ... load from spy benchmark

active_return = strategy_metrics['total_return'] - spy_metrics['total_return']
status = "BEAT SPY ✅" if active_return > 0 else "TRAIL SPY ❌"

print(f"\n{status}")
print(f"Active Return: {active_return:+.2%}")
print(f"Strategy: {strategy_metrics['total_return']:.2%}")
print(f"SPY: {spy_metrics['total_return']:.2%}")
# ... rest of output
```

**Estimated Time:** 15 minutes

---

### File 5: `tests/test_advanced_metrics.py`

**Add:** Tests for `calculate_active_metrics()` function (see code above in File 1)

**Estimated Time:** 15 minutes (included in File 1 time)

---

## Summary of Changes

| File | Type | Effort | Lines Changed |
|------|------|--------|---------------|
| `src/quantetf/evaluation/metrics.py` | Add function | 1.0 hr | +150 lines |
| `tests/test_advanced_metrics.py` | Add tests | (incl above) | +50 lines |
| `notebooks/backtest_analysis.ipynb` | Modify 6 cells | 1.5 hr | ~300 lines |
| `scripts/compare_strategies.py` | Add SPY auto | 0.3 hr | +30 lines |
| `scripts/benchmark_comparison.py` | Tweak output | 0.25 hr | ~10 lines |

**Total Estimated Time:** 2.5-3 hours
**Total Lines Changed:** ~540 lines

---

## Testing Plan

### Unit Tests
1. Run `pytest tests/test_advanced_metrics.py -v`
2. Verify all 3 new tests pass
3. Check edge cases (empty series, perfect tracking, underperformance)

### Integration Tests
1. Run a backtest: `python scripts/run_backtest.py`
2. Open notebook: `notebooks/backtest_analysis.ipynb`
3. Verify:
   - SPY data loads automatically
   - Active performance summary displays correctly
   - All charts show strategy vs SPY overlays
   - Numbers match between notebook cells

### Script Tests
1. Run strategy comparison with automatic SPY:
   ```bash
   python scripts/compare_strategies.py \
       --backtest-dirs artifacts/backtests/2026*/ \
       --output artifacts/comparisons/test
   ```
2. Verify SPY benchmark appears in comparison report
3. Check HTML report emphasizes active returns

---

## Rollout Strategy

**Phase 1: Metrics Foundation (30 min)**
- Add `calculate_active_metrics()` to metrics.py
- Add tests
- Verify tests pass

**Phase 2: Notebook Updates (1.5 hrs)**
- Update notebook cells 1-6
- Test notebook end-to-end
- Verify visualizations render correctly

**Phase 3: Scripts (30 min)**
- Update compare_strategies.py
- Update benchmark_comparison.py
- Test both scripts

**Phase 4: Documentation (15 min)**
- Update PROGRESS_LOG.md to mark REFACTOR-001 complete
- Commit all changes

---

## Success Criteria

- [ ] `calculate_active_metrics()` function implemented and tested
- [ ] All 3 new unit tests pass
- [ ] Notebook shows "🎯 ACTIVE PERFORMANCE SUMMARY" at top
- [ ] Equity curve chart shows strategy vs SPY overlaid
- [ ] Active returns chart displays rolling 1Y comparison
- [ ] All visualizations have SPY context
- [ ] Compare strategies script auto-adds SPY benchmark
- [ ] Benchmark comparison script emphasizes active returns
- [ ] 275 tests still passing (added 3, total 278)

---

## Notes for Implementer

1. **Import Organization:** All imports for `calculate_active_metrics()` are already in metrics.py (np, pd)
2. **Backward Compatibility:** This is purely additive - no existing code breaks
3. **Performance:** SPY benchmark adds ~2-3 seconds to notebook load time (acceptable)
4. **Future:** This pattern can be extended to any benchmark (not just SPY)

---

**Ready to implement!** 🚀
