from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quantetf.types import BacktestResult, DatasetVersion
from quantetf.data.store import DataStore
from quantetf.universe.base import UniverseBuilder
from quantetf.features.base import FeatureComputer
from quantetf.alpha.base import AlphaModel
from quantetf.risk.base import RiskModel
from quantetf.portfolio.base import PortfolioConstructor, CostModel
from quantetf.portfolio.schedule import RebalanceSchedule
from quantetf.utils.time import TradingCalendar


class BacktestEngine(ABC):
    @abstractmethod
    def run(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        dataset_version: DatasetVersion,
        store: DataStore,
        calendar: TradingCalendar,
        schedule: RebalanceSchedule,
        universe_builder: UniverseBuilder,
        feature_computer: FeatureComputer,
        alpha_model: AlphaModel,
        risk_model: RiskModel,
        portfolio_constructor: PortfolioConstructor,
        cost_model: Optional[CostModel] = None,
    ) -> BacktestResult:
        raise NotImplementedError
