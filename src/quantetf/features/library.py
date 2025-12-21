from __future__ import annotations

import pandas as pd

from quantetf.types import FeatureFrame, Universe
from quantetf.data.store import DataStore


def compute_momentum_score(
    prices_total_return: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    universe: Universe,
    lookback_days: int,
    skip_recent_days: int = 0,
) -> pd.Series:
    """Cross-sectional momentum signal.

    Returns a score per ticker based on trailing cumulative return.
    Uses business days as a proxy; align with your calendar in production.
    """
    end = as_of - pd.tseries.offsets.BDay(skip_recent_days)
    start = end - pd.tseries.offsets.BDay(lookback_days)
    window = prices_total_return.loc[start:end, list(universe.tickers)]

    # If input is daily returns, convert to cumulative growth.
    growth = (1.0 + window).prod(axis=0)
    score = growth - 1.0
    return score


def compute_realized_vol(
    prices_total_return: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    universe: Universe,
    lookback_days: int = 60,
) -> pd.Series:
    end = as_of
    start = end - pd.tseries.offsets.BDay(lookback_days)
    window = prices_total_return.loc[start:end, list(universe.tickers)]
    return window.std(axis=0)
