from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from quantetf.types import BacktestResult
from quantetf.evaluation.metrics import cagr, max_drawdown, sharpe


@dataclass
class Evaluator:
    """Computes a standard metrics bundle for a backtest result."""

    def evaluate(self, result: BacktestResult) -> Mapping[str, Any]:
        equity = result.equity_curve
        r = result.returns
        return {
            "cagr": cagr(equity),
            "sharpe": sharpe(r),
            "max_drawdown": max_drawdown(equity),
        }
