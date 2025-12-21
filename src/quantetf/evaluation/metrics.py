from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    years = len(equity) / float(periods_per_year)
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)

def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())

def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))
