# Research Findings: RESEARCH-001 - Can We Beat SPY?

**Research ID:** RESEARCH-001
**Status:** FINDINGS DOCUMENTED
**Date:** 2026-01-15
**Researcher:** Quant Agent

---

## Executive Summary

### The Hypothesis
*"Is it technically possible to find a strategy that beats SPY over any 1-year period using Tier 4 ETFs, given that different market regimes may require different strategies?"*

### The Answer: **YES, with significant caveats**

| Finding | Result |
|---------|--------|
| Existence (perfect foresight) | ✅ 10/10 years beatable |
| Simple momentum | ⚠️ 33% win rate |
| Regime patterns exist | ✅ Confirmed |
| Predictable regime-switching | ❌ Not yet proven |

---

## 1. Data and Universe

### Dataset Used
- **Snapshot:** `data/snapshots/snapshot_20260113_232157/data.parquet`
- **Date Range:** 2016-01-13 to 2026-01-12 (10 years)
- **Universe:** Tier 4 (200 ETFs, 178 after excluding leveraged/inverse)
- **Benchmark:** SPY

### Tier 4 Characteristics
- Includes: Global equities, sectors, countries, bonds, commodities, currencies
- Excludes for analysis: Leveraged (2x), Inverse (-1x), VIX products (decay issues)
- Suitable for: Global macro, sector rotation, defensive positioning

---

## 2. Perfect Foresight Analysis (Existence Proof)

**Question:** Was it even POSSIBLE to beat SPY each year?

| Year | SPY Return | Best Single ETF | Oracle Top-5 | Spread vs SPY |
|------|------------|-----------------|--------------|---------------|
| 2016 | +20.9% | EWZ +83.1% | +69.6% | **+48.6%** |
| 2017 | +20.8% | ARKW +84.9% | +69.1% | **+48.3%** |
| 2018 | -5.2% | PALL +14.0% | +10.6% | **+15.9%** |
| 2019 | +31.1% | TAN +64.4% | +60.5% | **+29.4%** |
| 2020 | +17.2% | TAN +222.0% | +181.1% | **+163.9%** |
| 2021 | +30.5% | USO +67.9% | +55.6% | **+25.1%** |
| 2022 | -18.6% | TUR +91.2% | +55.7% | **+74.4%** |
| 2023 | +26.7% | ARKW +101.2% | +77.0% | **+50.3%** |
| 2024 | +25.6% | ARKW +48.1% | +41.6% | **+16.0%** |
| 2025 | +18.0% | SLV +139.2% | +103.6% | **+85.6%** |

**Conclusion:** With perfect hindsight, the oracle portfolio beats SPY by **+55.8%/year average**. The opportunity exists.

### Winners by Theme
- **2016:** Emerging markets recovery (EWZ Brazil)
- **2017-2020:** Tech/Innovation (ARKW, TAN solar)
- **2021-2022:** Energy/Commodities (USO, XLE, TUR)
- **2023-2025:** Mixed (Tech comeback, precious metals)

---

## 3. Market Regime Analysis

### Regime Classification Method
Using observable (lagging) indicators:
- **SPY vs 200-day MA:** Above = bullish, Below = bearish
- **Realized Volatility:** 21-day annualized vol
- **SPY Annual Return:** Direction and magnitude

### Regimes Identified (2016-2025)

| Year | SPY | Max DD | % > MA200 | Avg Vol | Regime |
|------|-----|--------|-----------|---------|--------|
| 2016 | +20.9% | -5.6% | 19% | 11.3% | MODERATE |
| 2017 | +20.8% | -2.6% | 100% | 6.7% | **BULL TREND** |
| 2018 | -5.2% | -19.3% | 84% | 14.3% | MODERATE |
| 2019 | +31.1% | -6.6% | 89% | 12.7% | **BULL TREND** |
| 2020 | +17.2% | -33.7% | 77% | 26.1% | **HIGH VOL** |
| 2021 | +30.5% | -5.1% | 100% | 12.4% | **BULL TREND** |
| 2022 | -18.6% | -24.5% | 19% | 23.7% | **BEAR** |
| 2023 | +26.7% | -10.0% | 94% | 13.1% | **BULL TREND** |
| 2024 | +25.6% | -8.4% | 100% | 11.9% | **BULL TREND** |
| 2025 | +18.0% | -18.8% | 83% | 16.3% | **BULL TREND** |

### Regime Distribution
- Bull Trend: 7 years
- Moderate: 2 years
- Bear: 1 year
- High Vol: 1 year (2020, but ended as bull)

---

## 4. Strategy Performance by Regime

### Strategies Tested
1. **12M Momentum:** Buy top-5 by trailing 12-month return
2. **3M Momentum:** Buy top-5 by trailing 3-month return
3. **Value (Mean Reversion):** Buy bottom-5 by trailing 12-month return
4. **Low Volatility:** Buy lowest-volatility 5 ETFs

### Results (Spread vs SPY by Year)

| Year | Regime | 12M Mom | 3M Mom | Value | Low Vol | Winner |
|------|--------|---------|--------|-------|---------|--------|
| 2017 | BULL | -5.7% | +2.1% | -12.3% | -8.4% | 3M Mom |
| 2018 | MOD | -12.8% | -8.2% | +6.1% | +3.2% | Value |
| 2019 | BULL | -7.7% | -3.4% | -15.6% | -12.8% | 3M Mom |
| 2020 | HIGH_VOL | +92.3% | +48.5% | -10.4% | -15.4% | **12M Mom** |
| 2021 | BULL | -56.7% | -48.2% | +18.8% | -31.4% | **Value** |
| 2022 | BEAR | +36.9% | -9.5% | +14.0% | +18.3% | **12M Mom** |
| 2023 | BULL | -26.7% | -14.1% | +8.2% | -5.6% | Value |
| 2024 | BULL | -0.6% | +5.2% | -18.3% | -10.1% | 3M Mom |
| 2025 | BULL | +10.4% | -1.0% | +23.1% | +3.5% | **Value** |

### Key Patterns
1. **12M Momentum works in:** High volatility (2020), Bear markets (2022)
2. **Value works in:** Bull markets after momentum crashes (2021, 2025)
3. **3M Momentum:** Marginal improvement in steady bulls
4. **No single strategy dominates**

---

## 5. Simple Momentum Baseline (Annual Rebalance)

Strategy: At end of year T-1, select top-5 ETFs by 12M return. Hold for year T.

| Year | SPY | Momentum | vs SPY | Beat? |
|------|-----|----------|--------|-------|
| 2017 | +20.8% | +15.0% | -5.7% | NO |
| 2018 | -5.2% | -18.0% | -12.8% | NO |
| 2019 | +31.1% | +23.4% | -7.7% | NO |
| 2020 | +17.2% | +109.5% | **+92.3%** | YES |
| 2021 | +30.5% | -26.2% | -56.7% | NO |
| 2022 | -18.6% | +18.2% | **+36.9%** | YES |
| 2023 | +26.7% | +0.0% | -26.7% | NO |
| 2024 | +25.6% | +25.0% | -0.6% | NO |
| 2025 | +18.0% | +28.4% | **+10.4%** | YES |

**Win Rate:** 3/9 years (33%)
**Average Spread:** +4.4%/year (high variance)

### The Momentum Paradox
- When momentum wins, it wins BIG (+36% to +92%)
- When momentum loses, it can be catastrophic (-56.7% in 2021)
- This suggests momentum needs **risk management** or **regime filtering**

---

## 6. Why Regime-Switching Is Hard

### The Core Problem: Lagging Detection

Regime signals are only clear **after** the regime is established:
- By the time SPY falls below MA200, you've already lost 10-20%
- By the time vol spikes, the crash has happened
- By the time vol normalizes, the recovery is underway

### 2020 Case Study: Regime Whipsaw
| Month | SPY | Regime Signal | Actual |
|-------|-----|---------------|--------|
| Jan | +0% | Bull (>MA200) | Correct |
| Feb | -8% | Bull (>MA200) | Wrong |
| Mar | -34% | Bear (<MA200) | Correct (too late) |
| Apr | +13% | Bear (<MA200) | Wrong (recovery) |
| May | +5% | Neutral | Missed bull |
| Jun-Dec | +20% | Bull (>MA200) | Correct (late) |

**Lesson:** Regime transitions are where strategies fail most.

---

## 7. Research Agenda: Next Steps

### High Priority Experiments

#### EXPERIMENT-001: Monthly Rebalancing
**Hypothesis:** Monthly rebalancing captures regime shifts faster than annual.

**Test:**
- Run 12M momentum with monthly vs annual rebalance
- Compare: win rate, max drawdown, Sharpe ratio
- Expected: Higher win rate, lower variance

**Implementation:** Modify `run_backtest.py` to support monthly selection.

---

#### EXPERIMENT-002: Momentum + Trend Filter
**Hypothesis:** Only use momentum when SPY > 200MA; go defensive otherwise.

**Rules:**
```
IF SPY > MA200:
    Use 12M momentum (top-5)
ELSE:
    Hold defensive basket (AGG, TLT, GLD, USMV, SPLV)
```

**Test:**
- Backtest 2016-2025 with this rule
- Compare to pure momentum
- Expected: Avoid 2018, 2022 crashes

**Implementation:** New alpha model: `TrendFilteredMomentum`

---

#### EXPERIMENT-003: Ensemble Strategy
**Hypothesis:** Blending strategies reduces variance vs switching.

**Method:**
```
score = 0.4 * momentum_score + 0.3 * value_score + 0.3 * low_vol_score
```

**Test:**
- Backtest equal-weight blend
- Compare to individual strategies
- Expected: More stable returns, lower max drawdown

**Implementation:** Extend `EnsembleAlphaModel` with configurable weights.

---

#### EXPERIMENT-004: Volatility-Based Position Sizing
**Hypothesis:** Scale exposure based on realized volatility.

**Rules:**
```
target_vol = 15%
realized_vol = 21-day rolling vol of portfolio
position_size = target_vol / realized_vol
```

**Test:**
- Apply to momentum strategy
- Compare raw vs vol-scaled returns
- Expected: Smoother equity curve, similar Sharpe

**Implementation:** New portfolio construction: `VolTargetedConstruction`

---

#### EXPERIMENT-005: Walk-Forward Validation
**Hypothesis:** In-sample regime rules don't work out-of-sample.

**Method:**
1. Train regime detector on 2016-2020
2. Define strategy-regime mapping
3. Test on 2021-2025 (unseen)
4. Measure degradation

**Implementation:** Use existing `walk_forward.py` infrastructure.

---

### Medium Priority

#### EXPERIMENT-006: Factor Timing
Test if factor returns are predictable:
- Momentum factor premium (vs value)
- Quality vs growth rotation
- US vs International timing

#### EXPERIMENT-007: Sector Rotation
Instead of individual ETFs, rotate among sectors:
- XLK (tech), XLE (energy), XLF (financials), etc.
- May have more stable regime patterns

#### EXPERIMENT-008: Drawdown Control
Stop-loss rules:
- If portfolio drawdown > 10%, rotate to cash/bonds
- Test various thresholds (5%, 10%, 15%)

---

## 8. Additional Strategies to Implement

### New Alpha Models Needed

| Strategy | Priority | Complexity | File |
|----------|----------|------------|------|
| TrendFilteredMomentum | HIGH | Low | `trend_filtered_momentum.py` |
| ValueMomentum (blend) | HIGH | Medium | `value_momentum.py` |
| SectorRotation | MEDIUM | Medium | `sector_rotation.py` |
| RelativeStrength | MEDIUM | Low | `relative_strength.py` |
| DualMomentum (abs + rel) | HIGH | Low | `dual_momentum.py` |

### TrendFilteredMomentum Spec

```python
class TrendFilteredMomentum(AlphaModel):
    """
    Momentum strategy with trend filter.

    Only go long when SPY > 200MA.
    When SPY < 200MA, hold defensive assets.

    Parameters:
        momentum_lookback: int = 252 (12 months)
        ma_period: int = 200
        defensive_tickers: list = ['AGG', 'TLT', 'GLD', 'USMV']
    """

    def score(self, prices: pd.DataFrame, date: datetime) -> pd.Series:
        spy_price = prices['SPY'].loc[:date].iloc[-1]
        spy_ma = prices['SPY'].loc[:date].rolling(self.ma_period).mean().iloc[-1]

        if spy_price > spy_ma:
            # Bull regime: use momentum
            return self._momentum_scores(prices, date)
        else:
            # Defensive regime: score defensive tickers high
            return self._defensive_scores(prices)
```

### DualMomentum Spec (Gary Antonacci style)

```python
class DualMomentum(AlphaModel):
    """
    Combines absolute and relative momentum.

    1. Absolute: Only invest if asset return > T-bill (positive momentum)
    2. Relative: Among positive momentum assets, pick best

    If no assets have positive momentum, go to bonds.
    """

    def score(self, prices: pd.DataFrame, date: datetime) -> pd.Series:
        returns = self._calculate_returns(prices, date, self.lookback)
        tbill_return = self.risk_free_rate * (self.lookback / 252)

        # Absolute filter: only positive momentum
        positive_mom = returns[returns > tbill_return]

        if len(positive_mom) == 0:
            # All negative: return bond scores
            return self._bond_scores(prices)

        # Relative: rank positive momentum
        return positive_mom.rank(ascending=False)
```

---

## 9. Additional Data Sources

### Currently Available
- **Stooq:** Free OHLCV data for ETFs (in use)
- **Coverage:** 300+ ETFs, 10+ years

### Additional Free Sources to Investigate

| Source | Data Type | Coverage | Access |
|--------|-----------|----------|--------|
| **FRED** | Macro data (yields, VIX, spreads) | 50+ years | API (free) |
| **Yahoo Finance** | OHLCV, fundamentals | Global | yfinance lib |
| **Alpha Vantage** | OHLCV, technicals | US stocks | API (free tier) |
| **Tiingo** | OHLCV, news | US stocks | API (free tier) |
| **Quandl/Nasdaq** | Economic data | Various | API (some free) |

### High-Value Additional Data

#### 1. FRED Economic Indicators
**Why:** Regime signals may come from macro data.

| Indicator | FRED Code | Use Case |
|-----------|-----------|----------|
| VIX | VIXCLS | Volatility regime |
| 10Y Treasury | DGS10 | Risk-on/off signal |
| 2Y-10Y Spread | T10Y2Y | Recession indicator |
| Corporate Spreads | BAMLH0A0HYM2 | Credit conditions |
| Unemployment | UNRATE | Economic regime |

**Implementation:** Add `scripts/ingest_fred_data.py`

```python
from fredapi import Fred
fred = Fred(api_key='YOUR_KEY')

indicators = {
    'VIX': 'VIXCLS',
    'DGS10': 'DGS10',
    'SPREAD_10Y2Y': 'T10Y2Y',
    'HY_SPREAD': 'BAMLH0A0HYM2',
}

for name, code in indicators.items():
    data = fred.get_series(code, start='2015-01-01')
    data.to_parquet(f'data/raw/macro/{name}.parquet')
```

#### 2. Sector ETF Fundamentals
**Why:** Value signals may improve sector rotation.

| Data | Source | Use |
|------|--------|-----|
| P/E ratios | Yahoo Finance | Value signal |
| Earnings growth | Yahoo Finance | Growth signal |
| Dividend yield | Yahoo Finance | Income signal |

---

## 10. Implementation Priorities

### Wave 1 (Immediate - This Week)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Implement TrendFilteredMomentum | Coding Agent | `src/quantetf/alpha/trend_filtered_momentum.py` |
| Implement DualMomentum | Coding Agent | `src/quantetf/alpha/dual_momentum.py` |
| Add monthly rebalance option | Coding Agent | Update `SimpleBacktestEngine` |
| FRED data ingestion | Coding Agent | `scripts/ingest_fred_data.py` |

### Wave 2 (Next Week)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Run EXPERIMENT-001 (monthly rebal) | Quant Agent | Analysis report |
| Run EXPERIMENT-002 (trend filter) | Quant Agent | Analysis report |
| Ensemble strategy implementation | Coding Agent | `src/quantetf/alpha/blended_ensemble.py` |
| Walk-forward validation | Quant Agent | EXPERIMENT-005 results |

### Wave 3 (Following Week)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Volatility targeting | Coding Agent | `VolTargetedConstruction` class |
| Sector rotation alpha | Coding Agent | `sector_rotation.py` |
| Comprehensive regime analysis | Quant Agent | Research report |

---

## 11. Success Metrics

### For Individual Experiments
- **Beat SPY:** >50% of rolling 1-year windows
- **Sharpe Ratio:** >0.8 (vs SPY ~0.6)
- **Max Drawdown:** <25% (vs SPY ~34% in 2020)
- **Information Ratio:** >0.3 vs SPY

### For Overall Research Program
- Identify at least ONE strategy that:
  - Beats SPY in 6/10 years (60% win rate)
  - Has positive risk-adjusted alpha (IR > 0)
  - Survives walk-forward validation

---

## 12. Key Insights for Planner/Coding Agents

### What Works (Probably)
1. **Trend filtering** reduces catastrophic losses
2. **Monthly rebalancing** captures more alpha than annual
3. **Ensemble blending** is safer than regime switching
4. **Volatility targeting** smooths returns

### What Doesn't Work (Probably)
1. Pure momentum with annual rebalance (too much variance)
2. Perfect regime switching (detection is too slow)
3. Over-optimization on historical regimes (overfitting)

### Implementation Principles
1. Start simple, add complexity only if it helps OOS
2. Always run walk-forward validation
3. Focus on Sharpe ratio, not just returns
4. Transaction costs matter more at higher frequencies

---

## Appendix: Code for Analysis

### Perfect Foresight Analysis
```python
# See: scripts/analysis/perfect_foresight.py
for year in range(2016, 2026):
    year_data = prices.loc[f"{year}-01-01":f"{year}-12-31"]
    fwd_ret = (year_data.iloc[-1] / year_data.iloc[0] - 1).dropna()
    spy_ret = fwd_ret['SPY']
    top5_avg = fwd_ret.nlargest(5).mean()
    print(f"{year}: Oracle={top5_avg:.1%}, SPY={spy_ret:.1%}, Spread={top5_avg-spy_ret:.1%}")
```

### Regime Classification
```python
# See: scripts/analysis/regime_detector.py
def classify_regime(spy, date, ma_period=200, vol_period=21):
    spy_price = spy.loc[:date].iloc[-1]
    spy_ma = spy.loc[:date].rolling(ma_period).mean().iloc[-1]
    spy_vol = spy.pct_change().loc[:date].rolling(vol_period).std().iloc[-1] * np.sqrt(252)

    if spy_price < spy_ma:
        return "DEFENSIVE"
    elif spy_vol > 0.20:
        return "HIGH_VOL"
    else:
        return "BULL"
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-15
**Next Review:** After EXPERIMENT-001 and EXPERIMENT-002 complete
