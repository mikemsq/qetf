from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

# Special ticker for cash holdings
CASH_TICKER = "$CASH$"


@dataclass(frozen=True)
class DatasetVersion:
    """Identifies an immutable dataset snapshot used for reproducible runs."""
    id: str
    created_at: datetime
    description: str = ""


@dataclass(frozen=True)
class Universe:
    """A set of eligible tickers as-of a given timestamp."""
    as_of: pd.Timestamp
    tickers: tuple[str, ...]


@dataclass(frozen=True)
class FeatureFrame:
    """Features for a universe as-of a given timestamp."""
    as_of: pd.Timestamp
    frame: pd.DataFrame  # index: ticker, columns: feature names


@dataclass(frozen=True)
class AlphaScores:
    """Alpha scores for a universe as-of a given timestamp."""
    as_of: pd.Timestamp
    scores: pd.Series  # index: ticker, values: score


@dataclass(frozen=True)
class RiskModelOutput:
    """Risk model output as-of a given timestamp."""
    as_of: pd.Timestamp
    cov: pd.DataFrame  # index/columns: ticker
    exposures: Optional[pd.DataFrame] = None  # index: ticker, columns: factors
    diagnostics: Dict[str, Any] = None


@dataclass(frozen=True)
class TargetWeights:
    """Target portfolio weights as-of a given timestamp."""
    as_of: pd.Timestamp
    weights: pd.Series  # index: ticker, values: weight (sum ~ 1.0)
    diagnostics: Dict[str, Any] = None


@dataclass(frozen=True)
class BacktestResult:
    """Primary artifacts produced by a backtest run."""
    equity_curve: pd.Series               # index: date, values: portfolio value
    returns: pd.Series                    # index: date, values: daily returns
    positions: pd.DataFrame               # index: date, columns: tickers, values: weights
    trades: pd.DataFrame                  # index: date, columns: trade fields
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RecommendationPacket:
    """Production output for a given run date."""
    as_of: pd.Timestamp
    target_weights: pd.Series
    trades: pd.DataFrame
    summary: Mapping[str, Any]
    manifest: Mapping[str, Any]


@dataclass
class RegimeMetrics:
    """Performance metrics for a strategy within a specific market regime.

    Used for regime-based strategy selection, where different strategies
    may perform better in different market conditions.

    Attributes:
        regime: Name of the regime (e.g., 'uptrend_low_vol')
        sharpe_ratio: Annualized Sharpe ratio within this regime
        annualized_return: Annualized return within this regime
        volatility: Annualized volatility within this regime
        max_drawdown: Maximum drawdown within this regime
        total_return: Total cumulative return within this regime
        num_days: Number of trading days in this regime
        pct_of_period: Percentage of evaluation period spent in this regime
    """
    regime: str
    sharpe_ratio: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    total_return: float
    num_days: int
    pct_of_period: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'regime': self.regime,
            'sharpe_ratio': self.sharpe_ratio,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'total_return': self.total_return,
            'num_days': self.num_days,
            'pct_of_period': self.pct_of_period,
        }
