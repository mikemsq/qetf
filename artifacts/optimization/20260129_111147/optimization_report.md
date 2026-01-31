# Strategy Optimization Report

**Run timestamp:** 20260129_111147
**Periods evaluated:** 1yr
**Data source:** DataAccessContext

## Summary

- Total configurations tested: 438
- Successful evaluations: 438
- Failed evaluations: 0
- **Strategies that beat SPY in ALL periods: 109**

## Top 10 Winners

| Rank | Config | Composite Score | 1yr Active |
|------|--------|-----------------|----------|
| 1 | momentum_acceleration_long_lookback_days | 2.839 | +18.2% |
| 2 | momentum_acceleration_long_lookback_days | 2.796 | +32.3% |
| 3 | momentum_acceleration_long_lookback_days | 2.796 | +32.3% |
| 4 | momentum_acceleration_long_lookback_days | 2.728 | +45.2% |
| 5 | momentum_acceleration_long_lookback_days | 2.728 | +45.2% |
| 6 | momentum_lookback_days63_min_periods50_t | 2.583 | +40.0% |
| 7 | momentum_lookback_days63_min_periods15_t | 2.581 | +68.1% |
| 8 | momentum_lookback_days63_min_periods30_t | 2.581 | +68.1% |
| 9 | momentum_lookback_days63_min_periods50_t | 2.581 | +68.1% |
| 10 | momentum_acceleration_long_lookback_days | 2.579 | +42.7% |

## Best Strategy Details

**Name:** `momentum_acceleration_long_lookback_days126_min_periods100_short_lookback_days63_top7_monthly`

### Configuration

- **Alpha Model:** momentum_acceleration
- **Parameters:** `{'short_lookback_days': 63, 'long_lookback_days': 126, 'min_periods': 100}`
- **Top N:** 7
- **Schedule:** monthly
- **Universe:** configs/universes/tier3_expanded_100.yaml

### Performance by Period

| Period | Strategy Return | SPY Return | Active Return | Info Ratio | Sharpe |
|--------|-----------------|------------|---------------|------------|--------|
| 1yr | +41.3% | +23.2% | +18.2% | 2.34 | 2.30 |

## Output Files

- `all_results.csv` - All configurations with metrics
- `winners.csv` - Configurations that beat SPY in all periods
- `best_strategy.yaml` - Ready-to-use config for best strategy
- `optimization_report.md` - This report
