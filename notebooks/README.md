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

## Guidelines

- Keep notebooks thin and call into `src/quantetf/` for logic
- Save figures and reports into `artifacts/`
- Record the dataset snapshot ID and config used for any results
- All analysis notebooks should be self-contained and well-documented
