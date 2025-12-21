from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from quantetf.universe.base import UniverseBuilder
from quantetf.types import DatasetVersion, Universe
from quantetf.data.store import DataStore
from quantetf.universe.filters import EligibilityRules, apply_basic_rules


@dataclass(frozen=True)
class StaticUniverseBuilder(UniverseBuilder):
    tickers: Sequence[str]
    rules: EligibilityRules = EligibilityRules()

    def build(
        self,
        *,
        as_of: pd.Timestamp,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> Universe:
        instrument_master = store.read_instrument_master(version=dataset_version)
        eligible = apply_basic_rules(self.tickers, instrument_master, as_of=as_of, rules=self.rules)
        return Universe(as_of=as_of, tickers=tuple(eligible))
