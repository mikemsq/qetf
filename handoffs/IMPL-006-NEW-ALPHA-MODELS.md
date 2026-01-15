# Handoff: IMPL-006 - New Alpha Models for Regime Research

**Task ID:** IMPL-006
**Status:** ready
**Priority:** HIGH
**Estimated Effort:** Medium (3-4 files, ~400 lines)
**Dependencies:** None
**Assigned to:** Coding Agent

---

## Context & Motivation

### What are we building?

Three new alpha models to test regime-aware strategies identified in RESEARCH-001:

1. **TrendFilteredMomentum** - Momentum with trend filter (only long when SPY > MA200)
2. **DualMomentum** - Combines absolute and relative momentum (Gary Antonacci style)
3. **ValueMomentum** - Blends momentum and mean-reversion signals

### Why does this matter?

Research findings (RESEARCH-001-REGIME-HYPOTHESIS.md) showed:
- Simple 12M momentum only beats SPY 33% of years
- When momentum fails, it fails catastrophically (2021: -56.7% vs SPY)
- Trend filtering and dual momentum may reduce these drawdowns
- Value signals work in certain regimes (2021, 2025)

These models enable testing the core research hypotheses.

---

## Specification 1: TrendFilteredMomentum

### Concept

Use momentum when the market trend is bullish (SPY > 200MA).
Go defensive when trend is bearish (SPY < 200MA).

### Implementation

**File:** `src/quantetf/alpha/trend_filtered_momentum.py`

```python
"""Trend-filtered momentum alpha model.

This model applies a trend filter before using momentum signals.
When SPY is above its 200-day moving average (bullish), use momentum.
When SPY is below (bearish), allocate to defensive assets.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from quantetf.alpha.base import AlphaModel


@dataclass
class TrendFilteredMomentum(AlphaModel):
    """Momentum strategy with trend filter.

    Only use momentum when market trend is bullish (SPY > MA200).
    When bearish, allocate to defensive assets.

    Attributes:
        momentum_lookback: Days for momentum calculation (default: 252)
        ma_period: Moving average period for trend filter (default: 200)
        trend_ticker: Ticker to use for trend detection (default: 'SPY')
        defensive_tickers: Tickers to use in defensive mode
        top_n: Number of assets to select in momentum mode
        min_periods: Minimum data points required
    """
    momentum_lookback: int = 252
    ma_period: int = 200
    trend_ticker: str = 'SPY'
    defensive_tickers: List[str] = field(default_factory=lambda: ['AGG', 'TLT', 'GLD', 'USMV', 'SPLV'])
    top_n: int = 5
    min_periods: int = 200

    def score(
        self,
        prices: pd.DataFrame,
        date: datetime,
        universe: Optional[List[str]] = None
    ) -> pd.Series:
        """Calculate scores based on regime.

        Args:
            prices: DataFrame with price data (tickers as columns)
            date: Date for which to calculate scores
            universe: Optional list of tickers to consider

        Returns:
            Series of scores (higher = more attractive)
        """
        # Get trend signal
        is_bullish = self._is_bullish(prices, date)

        if is_bullish:
            # Bullish regime: use momentum
            return self._momentum_scores(prices, date, universe)
        else:
            # Bearish regime: score defensive assets high
            return self._defensive_scores(prices, universe)

    def _is_bullish(self, prices: pd.DataFrame, date: datetime) -> bool:
        """Check if market is in bullish regime.

        Returns True if trend_ticker is above its moving average.
        """
        if self.trend_ticker not in prices.columns:
            # If no trend ticker, default to bullish (use momentum)
            return True

        trend_prices = prices[self.trend_ticker].loc[:date].dropna()

        if len(trend_prices) < self.ma_period:
            # Not enough data for MA, default to bullish
            return True

        current_price = trend_prices.iloc[-1]
        ma_value = trend_prices.rolling(self.ma_period).mean().iloc[-1]

        return current_price > ma_value

    def _momentum_scores(
        self,
        prices: pd.DataFrame,
        date: datetime,
        universe: Optional[List[str]] = None
    ) -> pd.Series:
        """Calculate momentum scores (trailing return)."""
        if universe is None:
            universe = prices.columns.tolist()

        # Filter to available tickers
        available = [t for t in universe if t in prices.columns]
        price_data = prices[available].loc[:date]

        if len(price_data) < self.min_periods:
            return pd.Series(dtype=float)

        # Calculate lookback return
        lookback_start = max(0, len(price_data) - self.momentum_lookback)
        start_prices = price_data.iloc[lookback_start]
        end_prices = price_data.iloc[-1]

        returns = (end_prices / start_prices - 1).dropna()

        # Return as scores (higher return = higher score)
        return returns

    def _defensive_scores(
        self,
        prices: pd.DataFrame,
        universe: Optional[List[str]] = None
    ) -> pd.Series:
        """Score defensive assets high, others zero.

        In defensive mode, we want to hold safe assets.
        """
        if universe is None:
            universe = prices.columns.tolist()

        # Start with zero scores
        scores = pd.Series(0.0, index=universe)

        # Score defensive assets highly (equal weight among them)
        for ticker in self.defensive_tickers:
            if ticker in scores.index:
                scores[ticker] = 1.0

        return scores

    def get_regime(self, prices: pd.DataFrame, date: datetime) -> str:
        """Return current regime for logging/analysis."""
        return "BULLISH" if self._is_bullish(prices, date) else "DEFENSIVE"
```

### Test Cases

**File:** `tests/test_trend_filtered_momentum.py`

```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from quantetf.alpha.trend_filtered_momentum import TrendFilteredMomentum


class TestTrendFilteredMomentum:
    """Tests for TrendFilteredMomentum alpha model."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        np.random.seed(42)

        # SPY trending up
        spy_prices = 100 * np.cumprod(1 + np.random.randn(300) * 0.01 + 0.0005)

        # Other tickers
        qqq_prices = 100 * np.cumprod(1 + np.random.randn(300) * 0.015 + 0.0008)
        agg_prices = 100 * np.cumprod(1 + np.random.randn(300) * 0.003 + 0.0001)
        tlt_prices = 100 * np.cumprod(1 + np.random.randn(300) * 0.008 + 0.0002)

        return pd.DataFrame({
            'SPY': spy_prices,
            'QQQ': qqq_prices,
            'AGG': agg_prices,
            'TLT': tlt_prices,
        }, index=dates)

    def test_bullish_regime_uses_momentum(self, sample_prices):
        """When SPY > MA200, should use momentum scores."""
        model = TrendFilteredMomentum(
            momentum_lookback=60,
            ma_period=50,  # Short MA to ensure bullish
        )

        date = sample_prices.index[-1]
        scores = model.score(sample_prices, date)

        # Should have scores for all tickers
        assert len(scores) > 0
        # QQQ should have higher score (higher vol + drift)
        # This depends on random seed

    def test_bearish_regime_uses_defensive(self):
        """When SPY < MA200, should score defensive assets."""
        # Create data where SPY is below MA
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        spy_prices = np.concatenate([
            100 * np.ones(200),  # Flat for 200 days
            100 * np.linspace(1, 0.8, 100)  # Then drop 20%
        ])

        prices = pd.DataFrame({
            'SPY': spy_prices,
            'QQQ': spy_prices * 1.1,
            'AGG': 100 * np.ones(300),
            'TLT': 100 * np.ones(300),
        }, index=dates)

        model = TrendFilteredMomentum(
            ma_period=200,
            defensive_tickers=['AGG', 'TLT']
        )

        date = prices.index[-1]
        scores = model.score(prices, date)

        # Defensive tickers should have high scores
        assert scores['AGG'] == 1.0
        assert scores['TLT'] == 1.0
        # Non-defensive should have zero
        assert scores['QQQ'] == 0.0

    def test_regime_detection(self, sample_prices):
        """Test regime detection method."""
        model = TrendFilteredMomentum(ma_period=50)
        date = sample_prices.index[-1]

        regime = model.get_regime(sample_prices, date)
        assert regime in ['BULLISH', 'DEFENSIVE']

    def test_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = pd.DataFrame({
            'SPY': np.random.randn(50).cumsum() + 100,
        }, index=dates)

        model = TrendFilteredMomentum(min_periods=200)
        scores = model.score(prices, prices.index[-1])

        # Should return empty series
        assert len(scores) == 0

    def test_missing_trend_ticker(self):
        """Should default to bullish if trend ticker missing."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        prices = pd.DataFrame({
            'QQQ': np.random.randn(300).cumsum() + 100,
        }, index=dates)

        model = TrendFilteredMomentum(trend_ticker='SPY')  # SPY not in data

        # Should not raise, should use momentum
        scores = model.score(prices, prices.index[-1])
        assert len(scores) > 0
```

---

## Specification 2: DualMomentum

### Concept

Gary Antonacci's Dual Momentum combines:
1. **Absolute momentum:** Only invest if asset return > risk-free rate
2. **Relative momentum:** Among qualifying assets, pick the best

If no assets pass absolute momentum test, go to bonds.

### Implementation

**File:** `src/quantetf/alpha/dual_momentum.py`

```python
"""Dual momentum alpha model (Gary Antonacci style).

Combines absolute momentum (vs risk-free) and relative momentum (vs peers).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from quantetf.alpha.base import AlphaModel


@dataclass
class DualMomentum(AlphaModel):
    """Dual momentum strategy combining absolute and relative momentum.

    1. Absolute momentum: Filter out assets with return < risk-free rate
    2. Relative momentum: Rank remaining assets by return

    If no assets pass absolute filter, allocate to safe assets.

    Attributes:
        lookback: Days for momentum calculation (default: 252)
        risk_free_rate: Annual risk-free rate for absolute filter (default: 0.02)
        safe_tickers: Tickers to use when all momentum is negative
        top_n: Number of assets to select
        min_periods: Minimum data points required
    """
    lookback: int = 252
    risk_free_rate: float = 0.02  # 2% annual
    safe_tickers: List[str] = field(default_factory=lambda: ['AGG', 'BND', 'SHY'])
    top_n: int = 5
    min_periods: int = 200

    def score(
        self,
        prices: pd.DataFrame,
        date: datetime,
        universe: Optional[List[str]] = None
    ) -> pd.Series:
        """Calculate dual momentum scores.

        Args:
            prices: DataFrame with price data
            date: Date for calculation
            universe: Optional ticker list

        Returns:
            Series of scores
        """
        if universe is None:
            universe = [t for t in prices.columns if t not in self.safe_tickers]

        available = [t for t in universe if t in prices.columns]
        price_data = prices[available].loc[:date]

        if len(price_data) < self.min_periods:
            return pd.Series(dtype=float)

        # Calculate returns
        lookback_start = max(0, len(price_data) - self.lookback)
        start_prices = price_data.iloc[lookback_start]
        end_prices = price_data.iloc[-1]

        returns = (end_prices / start_prices - 1).dropna()

        # Absolute momentum threshold (annualized rf rate * lookback fraction)
        threshold = self.risk_free_rate * (self.lookback / 252)

        # Filter: only positive absolute momentum
        positive_momentum = returns[returns > threshold]

        if len(positive_momentum) == 0:
            # All negative: return safe asset scores
            return self._safe_scores(prices.columns.tolist())

        # Relative momentum: rank by return
        # Higher return = higher score
        return positive_momentum

    def _safe_scores(self, all_tickers: List[str]) -> pd.Series:
        """Score safe assets high when momentum is negative."""
        scores = pd.Series(0.0, index=all_tickers)
        for ticker in self.safe_tickers:
            if ticker in scores.index:
                scores[ticker] = 1.0
        return scores

    def get_signal_type(
        self,
        prices: pd.DataFrame,
        date: datetime,
        universe: Optional[List[str]] = None
    ) -> str:
        """Return whether using momentum or safe assets."""
        scores = self.score(prices, date, universe)

        # Check if we're in safe mode
        safe_in_scores = any(t in scores[scores > 0].index for t in self.safe_tickers)
        momentum_in_scores = any(t not in self.safe_tickers for t in scores[scores > 0].index)

        if safe_in_scores and not momentum_in_scores:
            return "SAFE"
        else:
            return "MOMENTUM"
```

### Test Cases

```python
class TestDualMomentum:
    """Tests for DualMomentum alpha model."""

    def test_positive_momentum_uses_ranking(self):
        """When assets have positive momentum, rank by return."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Create assets with clear return ranking
        prices = pd.DataFrame({
            'A': 100 * np.cumprod(1 + np.ones(300) * 0.001),  # +35% annual
            'B': 100 * np.cumprod(1 + np.ones(300) * 0.0005),  # +18% annual
            'C': 100 * np.cumprod(1 + np.ones(300) * 0.0002),  # +7% annual
            'AGG': 100 * np.ones(300),
        }, index=dates)

        model = DualMomentum(lookback=252, risk_free_rate=0.02)
        scores = model.score(prices, prices.index[-1])

        # A should have highest score, then B, then C
        # AGG should not be in scores (it's a safe ticker)
        assert scores['A'] > scores['B'] > scores['C']

    def test_negative_momentum_uses_safe(self):
        """When all momentum negative, use safe assets."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # All assets declining
        prices = pd.DataFrame({
            'A': 100 * np.cumprod(1 - np.ones(300) * 0.001),  # Declining
            'B': 100 * np.cumprod(1 - np.ones(300) * 0.0005),  # Declining
            'AGG': 100 * np.ones(300),
            'BND': 100 * np.ones(300),
        }, index=dates)

        model = DualMomentum(
            lookback=252,
            risk_free_rate=0.02,
            safe_tickers=['AGG', 'BND']
        )
        scores = model.score(prices, prices.index[-1])

        # Safe assets should have scores
        assert scores['AGG'] == 1.0
        assert scores['BND'] == 1.0
        # Risky assets should have zero
        assert scores['A'] == 0.0

    def test_absolute_momentum_threshold(self):
        """Test that absolute momentum filter works."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Asset with 1% return (below 2% threshold)
        prices = pd.DataFrame({
            'A': 100 * np.cumprod(1 + np.ones(300) * 0.00004),  # ~1% annual
            'AGG': 100 * np.ones(300),
        }, index=dates)

        model = DualMomentum(lookback=252, risk_free_rate=0.02)
        scores = model.score(prices, prices.index[-1])

        # A should be filtered out (below threshold)
        # Should fall back to safe assets
        assert scores.get('A', 0) == 0.0
        assert scores['AGG'] == 1.0
```

---

## Specification 3: ValueMomentum (Blended)

### Concept

Blend momentum and value (mean-reversion) signals:
- Momentum: Buy recent winners
- Value: Buy recent losers (mean reversion)
- Blend: Weighted combination

### Implementation

**File:** `src/quantetf/alpha/value_momentum.py`

```python
"""Value-momentum blend alpha model.

Combines momentum (trend-following) with value (mean-reversion) signals.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from quantetf.alpha.base import AlphaModel


@dataclass
class ValueMomentum(AlphaModel):
    """Blended value and momentum strategy.

    Score = momentum_weight * momentum_score + value_weight * value_score

    Where:
    - momentum_score: Trailing return (higher = better)
    - value_score: Negative of trailing return (lower past = higher score)

    Attributes:
        momentum_weight: Weight for momentum signal (default: 0.5)
        value_weight: Weight for value signal (default: 0.5)
        momentum_lookback: Days for momentum (default: 252)
        value_lookback: Days for value (default: 252)
        min_periods: Minimum data required
    """
    momentum_weight: float = 0.5
    value_weight: float = 0.5
    momentum_lookback: int = 252
    value_lookback: int = 252
    min_periods: int = 200

    def __post_init__(self):
        """Validate weights sum to 1."""
        total = self.momentum_weight + self.value_weight
        if abs(total - 1.0) > 0.001:
            # Normalize
            self.momentum_weight /= total
            self.value_weight /= total

    def score(
        self,
        prices: pd.DataFrame,
        date: datetime,
        universe: Optional[List[str]] = None
    ) -> pd.Series:
        """Calculate blended value-momentum scores.

        Args:
            prices: DataFrame with price data
            date: Date for calculation
            universe: Optional ticker list

        Returns:
            Series of blended scores (z-scored)
        """
        if universe is None:
            universe = prices.columns.tolist()

        available = [t for t in universe if t in prices.columns]
        price_data = prices[available].loc[:date]

        if len(price_data) < self.min_periods:
            return pd.Series(dtype=float)

        # Momentum scores (trailing return)
        mom_start = max(0, len(price_data) - self.momentum_lookback)
        mom_returns = (price_data.iloc[-1] / price_data.iloc[mom_start] - 1).dropna()

        # Value scores (negative of trailing return = mean reversion)
        val_start = max(0, len(price_data) - self.value_lookback)
        val_returns = (price_data.iloc[-1] / price_data.iloc[val_start] - 1).dropna()

        # Z-score both
        mom_z = self._zscore(mom_returns)
        val_z = self._zscore(-val_returns)  # Negative: losers get high value score

        # Blend
        common = mom_z.index.intersection(val_z.index)
        blended = (
            self.momentum_weight * mom_z[common] +
            self.value_weight * val_z[common]
        )

        return blended

    def _zscore(self, series: pd.Series) -> pd.Series:
        """Convert to z-scores."""
        if len(series) == 0 or series.std() == 0:
            return series
        return (series - series.mean()) / series.std()
```

---

## File Structure

After implementation:

```
src/quantetf/alpha/
├── __init__.py  # Add new exports
├── base.py
├── momentum.py
├── momentum_acceleration.py
├── residual_momentum.py
├── vol_adjusted_momentum.py
├── ensemble.py
├── trend_filtered_momentum.py   # NEW
├── dual_momentum.py             # NEW
└── value_momentum.py            # NEW

tests/
├── test_trend_filtered_momentum.py  # NEW
├── test_dual_momentum.py            # NEW
└── test_value_momentum.py           # NEW
```

---

## Update __init__.py

```python
# src/quantetf/alpha/__init__.py

from quantetf.alpha.base import AlphaModel
from quantetf.alpha.momentum import SimpleMomentum
from quantetf.alpha.momentum_acceleration import MomentumAcceleration
from quantetf.alpha.residual_momentum import ResidualMomentum
from quantetf.alpha.vol_adjusted_momentum import VolAdjustedMomentum
from quantetf.alpha.ensemble import EnsembleAlphaModel

# New models
from quantetf.alpha.trend_filtered_momentum import TrendFilteredMomentum
from quantetf.alpha.dual_momentum import DualMomentum
from quantetf.alpha.value_momentum import ValueMomentum

__all__ = [
    'AlphaModel',
    'SimpleMomentum',
    'MomentumAcceleration',
    'ResidualMomentum',
    'VolAdjustedMomentum',
    'EnsembleAlphaModel',
    'TrendFilteredMomentum',
    'DualMomentum',
    'ValueMomentum',
]
```

---

## Acceptance Criteria

- [ ] TrendFilteredMomentum implemented with:
  - [ ] Trend detection (SPY vs MA200)
  - [ ] Momentum scoring in bullish regime
  - [ ] Defensive scoring in bearish regime
  - [ ] 5+ unit tests
- [ ] DualMomentum implemented with:
  - [ ] Absolute momentum filter
  - [ ] Relative momentum ranking
  - [ ] Safe asset fallback
  - [ ] 5+ unit tests
- [ ] ValueMomentum implemented with:
  - [ ] Momentum scoring
  - [ ] Value scoring (mean reversion)
  - [ ] Configurable blending
  - [ ] 3+ unit tests
- [ ] All tests pass: `pytest tests/test_*momentum*.py -v`
- [ ] Exports updated in `__init__.py`

---

## Validation

After implementation, run:

```bash
# Run all momentum tests
pytest tests/test_*momentum*.py -v

# Quick integration test
python -c "
from quantetf.alpha import TrendFilteredMomentum, DualMomentum, ValueMomentum
print('TrendFilteredMomentum:', TrendFilteredMomentum())
print('DualMomentum:', DualMomentum())
print('ValueMomentum:', ValueMomentum())
print('All imports successful!')
"
```

---

## Related Documents

- [RESEARCH-001-REGIME-HYPOTHESIS.md](./RESEARCH-001-REGIME-HYPOTHESIS.md) - Research findings
- [IMPL-007-DATA-INGESTION.md](./IMPL-007-DATA-INGESTION.md) - FRED data ingestion

---

**Ready to implement!**
