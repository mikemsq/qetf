from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quantetf.backtest.base import BacktestEngine
from quantetf.backtest.accounting import compute_portfolio_returns
from quantetf.backtest.execution import compute_turnover
from quantetf.types import BacktestResult, DatasetVersion
from quantetf.data.store import DataStore
from quantetf.universe.base import UniverseBuilder
from quantetf.features.base import FeatureComputer
from quantetf.alpha.base import AlphaModel
from quantetf.risk.base import RiskModel
from quantetf.portfolio.base import PortfolioConstructor, CostModel
from quantetf.portfolio.schedule import RebalanceSchedule
from quantetf.utils.time import TradingCalendar


@dataclass
class SimpleBacktestEngine(BacktestEngine):
    """A minimal, correct-by-construction backtest loop.

    Assumptions:
    - daily total returns are available for all instruments
    - rebalances occur on schedule dates
    - weights are applied from next trading day onward (close-to-close style)
    """

    initial_value: float = 1.0

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
        ret = store.read_prices_total_return(version=dataset_version)

        # Define rebalance dates and a daily simulation index.
        rebalance_dates = schedule.dates(start=start, end=end, calendar=calendar)
        daily_idx = pd.date_range(start=start, end=end, freq="B")

        # Positions will be held constant between rebalances.
        positions = pd.DataFrame(index=daily_idx, columns=ret.columns, dtype=float)

        prev_w = pd.Series(dtype=float)
        trades_rows = []

        for i, as_of in enumerate(rebalance_dates):
            if as_of not in daily_idx:
                continue

            universe = universe_builder.build(as_of=as_of, store=store, dataset_version=dataset_version)
            features = feature_computer.compute(
                as_of=as_of, universe=universe, store=store, dataset_version=dataset_version
            )
            alpha = alpha_model.score(
                as_of=as_of, universe=universe, features=features, store=store, dataset_version=dataset_version
            )
            risk = risk_model.estimate(as_of=as_of, universe=universe, store=store, dataset_version=dataset_version)
            target = portfolio_constructor.construct(
                as_of=as_of,
                universe=universe,
                alpha=alpha,
                risk=risk,
                store=store,
                dataset_version=dataset_version,
                prev_weights=prev_w if len(prev_w) else None,
            )

            next_w = target.weights
            turnover = compute_turnover(prev_w, next_w) if len(prev_w) else float(next_w.abs().sum())
            cost = 0.0
            if cost_model is not None and len(prev_w):
                cost = float(cost_model.estimate_rebalance_cost(prev_weights=prev_w, next_weights=next_w))

            trades_rows.append(
                {
                    "date": as_of,
                    "turnover": turnover,
                    "cost_frac": cost,
                    "selected": target.diagnostics.get("selected") if target.diagnostics else None,
                }
            )

            # Apply weights from the next business day (simple close-to-close assumption).
            start_apply = as_of + pd.tseries.offsets.BDay(1)
            end_apply = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end
            apply_idx = pd.date_range(start=start_apply, end=end_apply, freq="B")

            # Fill weights across the interval
            for d in apply_idx:
                if d in positions.index:
                    positions.loc[d, next_w.index] = next_w.values

            prev_w = next_w

        positions = positions.fillna(0.0)
        returns = ret.reindex(daily_idx).fillna(0.0)
        port_ret = compute_portfolio_returns(returns, positions)

        # Apply costs as a one-time drag on rebalance dates, distributed to next day for simplicity.
        trades = pd.DataFrame(trades_rows).set_index("date") if trades_rows else pd.DataFrame()
        if not trades.empty:
            for d, row in trades.iterrows():
                apply_day = (pd.Timestamp(d) + pd.tseries.offsets.BDay(1))
                if apply_day in port_ret.index:
                    port_ret.loc[apply_day] = port_ret.loc[apply_day] - float(row["cost_frac"])

        equity = (1.0 + port_ret).cumprod() * self.initial_value

        metrics = {
            "cagr": float(equity.iloc[-1] ** (252.0 / max(1, len(equity))) - 1.0),
            "vol": float(port_ret.std() * (252.0 ** 0.5)),
            "sharpe": float((port_ret.mean() / (port_ret.std() + 1e-12)) * (252.0 ** 0.5)),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        }
        metadata = {"dataset_version": dataset_version.id, "start": str(start), "end": str(end)}

        return BacktestResult(
            equity_curve=equity,
            returns=port_ret,
            positions=positions,
            trades=trades,
            metrics=metrics,
            metadata=metadata,
        )
