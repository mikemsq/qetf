from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from quantetf.utils.time import TradingCalendar


class RebalanceSchedule(ABC):
    """Produces rebalance timestamps given a date range."""

    @abstractmethod
    def dates(self, *, start: pd.Timestamp, end: pd.Timestamp, calendar: TradingCalendar) -> list[pd.Timestamp]:
        raise NotImplementedError


@dataclass(frozen=True)
class EveryNTradingDays(RebalanceSchedule):
    n: int

    def dates(self, *, start: pd.Timestamp, end: pd.Timestamp, calendar: TradingCalendar) -> list[pd.Timestamp]:
        return calendar.every_n_trading_days(start, end, self.n)


@dataclass(frozen=True)
class WeeklyOnWeekday(RebalanceSchedule):
    weekday: int = 4  # 0=Mon ... 4=Fri

    def dates(self, *, start: pd.Timestamp, end: pd.Timestamp, calendar: TradingCalendar) -> list[pd.Timestamp]:
        idx = pd.date_range(start=start, end=end, freq="B")
        days = [d for d in idx if d.weekday() == self.weekday]
        return [pd.Timestamp(d) for d in days]
