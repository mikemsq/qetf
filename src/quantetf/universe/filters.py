from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class EligibilityRules:
    min_history_days: int = 252
    min_avg_dollar_volume: float = 0.0
    min_aum: float = 0.0


def apply_basic_rules(
    tickers: Sequence[str],
    instrument_master: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    rules: EligibilityRules,
) -> list[str]:
    """Apply common eligibility filters.

    This is a placeholder implementation. In a real system you would:
    - compute trailing ADV from OHLCV
    - use point-in-time AUM
    - enforce inception date and trading status
    """
    df = instrument_master.copy()

    if "ticker" in df.columns:
        df = df.set_index("ticker")

    eligible = []
    for t in tickers:
        if t not in df.index:
            continue
        # Optional fields
        inception = df.loc[t].get("inception_date", None)
        if pd.notna(inception):
            inception = pd.Timestamp(inception)
            if (as_of - inception).days < rules.min_history_days:
                continue

        aum = df.loc[t].get("aum", None)
        if pd.notna(aum) and float(aum) < rules.min_aum:
            continue

        eligible.append(t)

    return eligible
