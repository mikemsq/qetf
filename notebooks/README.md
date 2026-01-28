# notebooks/

Exploratory research notebooks and analysis dashboards.

## Available Notebooks

### backtest_analysis.ipynb
Comprehensive backtest analysis dashboard with 8 key visualizations:

1. **Equity Curve with Dual-Axis Drawdown Overlay** - Portfolio value over time with drawdowns on secondary axis
2. **Monthly/Yearly Returns Heatmap** - Calendar heatmap showing returns by month and year
3. **Rolling Sharpe Ratio (252-day window)** - Evolution of risk-adjusted returns over time
4. **Drawdown Waterfall Chart** - Identification and visualization of individual drawdown periods
5. **Returns Distribution Histogram** - Daily returns distribution with statistical measures
6. **Underwater Plot** - Time spent below high-water mark
7. **Holdings Evolution Over Time** - Stacked area chart showing portfolio composition changes
8. **Turnover Analysis** - Portfolio turnover, concentration (HHI), and trade size analysis

**Usage:**
- Automatically loads the most recent valid backtest from `artifacts/backtests/`
- Generates professional visualizations with matplotlib/seaborn
- Includes comprehensive statistics for each analysis section
- All 8 visualizations as specified in VIZ-001 requirements

**To Run:**
```bash
jupyter notebook notebooks/backtest_analysis.ipynb
```

### regime_detection_analysis.ipynb
Market regime detection analysis using SPY price data and VIX from macro indicators.

**Features:**
1. **Regime Detection** - Classifies market into 4 regimes based on trend (SPY vs 200MA) and volatility (VIX)
2. **Daily Time Series Chart** - 3-panel visualization showing SPY with regime shading, VIX, and regime timeline
3. **Yearly Breakdown** - Stacked bar chart of regime distribution by year
4. **Returns by Regime** - SPY daily returns analysis segmented by regime
5. **Regime Statistics** - Distribution, transitions, and current regime status

**Regimes:**
| Regime | Conditions |
|--------|------------|
| uptrend_low_vol | SPY > 200MA, VIX < 20 |
| uptrend_high_vol | SPY > 200MA, VIX > 25 |
| downtrend_low_vol | SPY < 200MA, VIX < 20 |
| downtrend_high_vol | SPY < 200MA, VIX > 25 |

**Data Sources:**
- `data/snapshots/snapshot_latest/` - SPY price data
- `data/snapshots/macro.parquet` - VIX and macro indicators

**Outputs:**
- `artifacts/regime_detection_chart.png` - Main visualization
- `artifacts/regime_yearly_breakdown.png` - Yearly distribution
- `artifacts/regime_returns_boxplot.png` - Returns by regime
- `data/snapshots/regime_timeseries.parquet` - Daily regime classifications

**To Run:**
```bash
jupyter notebook notebooks/regime_detection_analysis.ipynb
```

## Guidelines

- Keep notebooks thin and call into `src/quantetf/` for logic
- Save figures and reports into `artifacts/`
- Record the dataset snapshot ID and config used for any results
- All analysis notebooks should be self-contained and well-documented
