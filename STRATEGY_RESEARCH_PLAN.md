# Strategy Research Plan: Beat SPY

## Executive Summary
This document outlines the research plan for implementing and testing new trading strategies to outperform the S&P 500 (SPY) benchmark. We will implement 10+ strategies across multiple categories, leveraging the existing QuantETF framework.

## Current State
- **Existing Strategy**: Cross-sectional momentum (12-month trailing returns)
- **Framework**: Modular architecture with alpha models, portfolio construction, risk models
- **Universe**: 20 ETFs across equity, bond, commodity sectors
- **Evaluation**: 30+ metrics focused on active returns vs SPY
- **Infrastructure**: Walk-forward validation, multiple benchmarks, transaction cost models

## Strategy Categories & Implementation Priority

### Phase 1: Enhanced Momentum (Priority: HIGH)
**Timeline**: Week 1-2

#### 1.1 Residual Momentum (Beta-Neutral)
- **Location**: `src/quantetf/alpha/residual_momentum.py`
- **Logic**: Regress returns on SPY, rank by residuals
- **Rationale**: Extract pure alpha independent of market beta
- **Expected Benefit**: Lower correlation to SPY, better risk-adjusted returns
- **Complexity**: Medium (requires regression)

#### 1.2 Volatility-Adjusted Momentum
- **Location**: `src/quantetf/alpha/vol_adjusted_momentum.py`
- **Logic**: `score = returns / realized_volatility` (Sharpe-style ranking)
- **Rationale**: Risk-adjusted momentum signals
- **Expected Benefit**: Better drawdown control, smoother returns
- **Complexity**: Low (simple division)

#### 1.3 Momentum Acceleration
- **Location**: `src/quantetf/alpha/momentum_acceleration.py`
- **Logic**: `score = returns_3m - returns_12m` (trend strength)
- **Rationale**: Capture momentum inflection points
- **Expected Benefit**: Earlier entry/exit signals
- **Complexity**: Low (feature subtraction)

### Phase 2: Defensive Strategies (Priority: HIGH)
**Timeline**: Week 2-3

#### 2.1 Minimum Volatility
- **Location**: `src/quantetf/alpha/min_volatility.py`
- **Logic**: Rank by `-1 * realized_volatility`
- **Rationale**: Low-volatility anomaly (Ang et al.)
- **Expected Benefit**: Lower drawdowns, defensive positioning
- **Complexity**: Low (simple volatility ranking)

#### 2.2 Maximum Sharpe
- **Location**: `src/quantetf/alpha/max_sharpe.py`
- **Logic**: Rank by trailing Sharpe ratio
- **Rationale**: Risk-adjusted selection
- **Expected Benefit**: Quality-focused portfolio
- **Complexity**: Low (Sharpe calculation exists)

### Phase 3: Multi-Factor Ensembles (Priority: HIGH)
**Timeline**: Week 3-4

#### 3.1 Momentum + Low Vol Ensemble
- **Location**: Configuration file using existing `WeightedEnsemble`
- **Logic**: 50% momentum + 50% min volatility
- **Rationale**: Combine trend-following with defensive tilt
- **Expected Benefit**: Better Sharpe, lower tail risk
- **Complexity**: Very Low (use existing ensemble)

#### 3.2 Quality-Momentum
- **Location**: `src/quantetf/alpha/quality_momentum.py`
- **Logic**: Filter by Sharpe > threshold, then apply momentum
- **Rationale**: Avoid "falling knives"
- **Expected Benefit**: Fewer momentum crashes
- **Complexity**: Medium (two-stage filtering)

### Phase 4: Timing & Regime Strategies (Priority: MEDIUM)
**Timeline**: Week 4-5

#### 4.1 Trend Following with Cash
- **Location**: `src/quantetf/alpha/trend_following.py`
- **Logic**: Hold when SPY > 200-day SMA, cash when below
- **Rationale**: Simple market timing (Faber 2007)
- **Expected Benefit**: Avoid major bear markets
- **Complexity**: Low (moving average)

#### 4.2 Dynamic Volatility Allocation
- **Location**: `src/quantetf/portfolio/vol_target.py`
- **Logic**: Scale portfolio leverage to target volatility
- **Rationale**: Volatility clustering
- **Expected Benefit**: Stable risk profile
- **Complexity**: Medium (requires portfolio constructor)

### Phase 5: Mean Reversion (Priority: MEDIUM)
**Timeline**: Week 5-6

#### 5.1 Short-Term Reversal
- **Location**: `src/quantetf/alpha/short_term_reversal.py`
- **Logic**: `score = -1 * returns_1w` (contrarian)
- **Rationale**: Short-term overreaction correction
- **Expected Benefit**: Low correlation to momentum
- **Complexity**: Low (negative momentum)

#### 5.2 Z-Score Mean Reversion
- **Location**: `src/quantetf/alpha/zscore_reversion.py`
- **Logic**: Rank by `(price - mean) / std` (z-score)
- **Rationale**: Statistical reversion to mean
- **Expected Benefit**: Range-bound market performance
- **Complexity**: Medium (rolling statistics)

### Phase 6: Advanced Strategies (Priority: LOW)
**Timeline**: Week 7+

#### 6.1 Machine Learning (XGBoost)
- **Location**: `src/quantetf/alpha/ml_ensemble.py`
- **Features**: Momentum, volatility, volume, RSI, correlation
- **Target**: Next month excess return vs SPY
- **Rationale**: Non-linear feature interactions
- **Expected Benefit**: Adaptive to regime changes
- **Complexity**: High (ML pipeline)

#### 6.2 Cross-Asset Allocation
- **Location**: `src/quantetf/alpha/flight_to_quality.py`
- **Logic**: Shift to bonds when equity vol spikes
- **Rationale**: Stock-bond negative correlation in crises
- **Expected Benefit**: Crisis alpha
- **Complexity**: Medium (multi-asset logic)

## Implementation Components

### For Each Strategy:

1. **Alpha Model** (`src/quantetf/alpha/`)
   - Inherit from `AlphaModel` base class
   - Implement `score()` method
   - Return `AlphaScores` with rankings
   - Ensure point-in-time compliance

2. **Configuration File** (`configs/strategies/`)
   - YAML config specifying alpha model parameters
   - Portfolio construction rules
   - Rebalancing schedule
   - Cost model

3. **Backtest Script** (`scripts/backtest_*.py`)
   - Load strategy config
   - Run backtest over full period (2020-2026)
   - Compare against SPY benchmark
   - Generate performance report

4. **Walk-Forward Validation** (`scripts/walk_forward_*.py`)
   - 2-year train, 1-year test windows
   - 6-month step forward
   - Track IS vs OOS degradation

5. **Unit Tests** (`tests/alpha/test_*.py`)
   - Test score calculation logic
   - Test edge cases (missing data, single asset)
   - Test point-in-time compliance

## Success Metrics

### Primary Goal: Beat SPY
- **Active Return** > 0 (CAGR strategy - CAGR SPY)
- **Information Ratio** > 0.5 (active return / tracking error)
- **Sharpe Ratio** > SPY Sharpe + 0.2

### Risk Constraints
- **Max Drawdown** < SPY max drawdown + 5%
- **Volatility** < SPY volatility + 5%
- **Calmar Ratio** > SPY Calmar

### Robustness Tests
- **OOS Sharpe** > 0.5 (walk-forward validation)
- **Win Rate** > 55% (days beating SPY)
- **Consistency** > 60% (% of years beating SPY)

## Data Requirements

### Existing Data Sources (Available)
- Daily OHLCV from Yahoo Finance
- 20 ETF universe
- SPY benchmark data
- Transaction cost models

### Additional Data Needed
- **VIX data** (for volatility regime strategies)
- **Interest rate data** (for carry strategies)
- **Dividend yield data** (for value strategies)
- **Volume data** (already available, needs feature engineering)

## Risk Management

### Overfitting Prevention
1. Use walk-forward validation (not just in-sample)
2. Limit parameter tuning (max 2-3 parameters per strategy)
3. Compare to random selection benchmark
4. Require statistical significance (Sharpe t-test)

### Transaction Cost Reality Check
- Use realistic 10 bps cost assumption
- Test sensitivity to cost (5 bps, 20 bps scenarios)
- Monitor turnover (penalize high turnover strategies)

### Diversification
- Test strategy combinations via ensemble
- Correlation matrix across strategies
- Portfolio of strategies approach

## Deliverables

### Per Strategy
1. Alpha model implementation (`.py` file)
2. Configuration file (`.yaml`)
3. Backtest results (metrics table)
4. Walk-forward validation results
5. Unit tests (`.py` file)
6. Performance summary (markdown report)

### Overall Research
1. Strategy comparison matrix (all strategies vs benchmarks)
2. Equity curve overlay chart
3. Risk-return scatter plot
4. Correlation heatmap
5. Final recommendation report

## Development Workflow

### Standard Process for Each Strategy

1. **Design Phase**
   - Review academic literature
   - Define scoring logic mathematically
   - Identify data dependencies

2. **Implementation Phase**
   - Create alpha model class
   - Write unit tests
   - Create configuration file
   - Write backtest script

3. **Validation Phase**
   - Run in-sample backtest
   - Run walk-forward validation
   - Compare to SPY benchmark
   - Check for lookahead bias

4. **Analysis Phase**
   - Generate performance metrics
   - Review equity curves
   - Analyze drawdown periods
   - Document findings

5. **Iteration Phase**
   - Tune parameters (if needed)
   - Test ensemble combinations
   - Sensitivity analysis

## Timeline Summary

- **Phase 1 (Enhanced Momentum)**: Week 1-2 → 3 strategies
- **Phase 2 (Defensive)**: Week 2-3 → 2 strategies
- **Phase 3 (Multi-Factor)**: Week 3-4 → 2 strategies
- **Phase 4 (Timing)**: Week 4-5 → 2 strategies
- **Phase 5 (Mean Reversion)**: Week 5-6 → 2 strategies
- **Phase 6 (Advanced)**: Week 7+ → 2+ strategies

**Total**: 10+ strategies over 6-7 weeks

## Next Steps

1. ✅ Document this plan
2. ✅ Create architect/planner agent task breakdown
3. ✅ Generate handover documents for coding agents (3/3 complete!)
4. ⏳ Begin Phase 1: Enhanced Momentum strategies

### Phase 1 Handouts Available

All three Phase 1 strategy handouts are complete and ready for implementation:

- ✅ **HANDOUT_momentum_acceleration.md** (615 lines) - Simplest, start here!
- ✅ **HANDOUT_vol_adjusted_momentum.md** (685 lines) - Second implementation
- ✅ **HANDOUT_residual_momentum.md** (868 lines) - Most complex, implement last

**Location**: `/workspaces/qetf/docs/handouts/`

**Quick Start**: See `/workspaces/qetf/docs/handouts/QUICKSTART.md` for implementation guide

---

**Document Version**: 1.1
**Last Updated**: 2026-01-13
**Author**: Quant Research Agent
**Status**: Planning Complete → Ready for Implementation ✅
