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
