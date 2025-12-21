from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class TradingCalendar:
    """Minimal trading calendar wrapper.

    In production, you may want a proper exchange calendar implementation.
    """
    timezone: str = "America/New_York"

    def to_trading_days(self, dates: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
        # Placeholder: assumes dates are already trading days.
        return list(pd.DatetimeIndex(list(dates)).tz_localize(None))

    def every_n_trading_days(self, start: pd.Timestamp, end: pd.Timestamp, n: int) -> list[pd.Timestamp]:
        idx = pd.date_range(start=start, end=end, freq="B")  # business days proxy
        return [idx[i] for i in range(0, len(idx), n)]
