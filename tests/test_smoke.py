import pandas as pd

from quantetf.data.inmemory import InMemoryDataStore
from quantetf.types import DatasetVersion
from quantetf.universe.static import StaticUniverseBuilder
from quantetf.features.base_feature_computer import BasicFeatureComputer
from quantetf.alpha.momentum import CrossSectionalMomentum
from quantetf.risk.covariance import EWMACovariance
from quantetf.portfolio.construction import TopXEqualWeight
from quantetf.portfolio.constraints import WeightConstraints
from quantetf.backtest.engine import SimpleBacktestEngine
from quantetf.portfolio.schedule import EveryNTradingDays
from quantetf.utils.time import TradingCalendar


def test_end_to_end_smoke():
    dates = pd.date_range("2020-01-01", periods=260, freq="B")
    tickers = ["AAA", "BBB", "CCC"]
    # synthetic daily returns
    ret = pd.DataFrame(0.0002, index=dates, columns=tickers)
    ret["BBB"] = 0.0001
    ret["CCC"] = -0.00005

    instrument_master = pd.DataFrame(
        {
            "ticker": tickers,
            "inception_date": [pd.Timestamp("2010-01-01")] * 3,
            "aum": [1e9, 1e9, 1e9],
        }
    )

    store = InMemoryDataStore(prices_total_return=ret, instrument_master=instrument_master)
    dataset = DatasetVersion(id="test_snapshot", created_at=pd.Timestamp.utcnow().to_pydatetime())

    universe_builder = StaticUniverseBuilder(tickers=tickers)
    feature_computer = BasicFeatureComputer(momentum_lookback_days=60, momentum_skip_recent_days=0, vol_lookback_days=60)
    alpha_model = CrossSectionalMomentum(feature_name="momentum")
    risk_model = EWMACovariance(halflife_days=30, lookback_days=120)
    portfolio_constructor = TopXEqualWeight(x=1, constraints=WeightConstraints(max_weight=1.0))

    engine = SimpleBacktestEngine()
    schedule = EveryNTradingDays(n=20)
    calendar = TradingCalendar()

    result = engine.run(
        start=pd.Timestamp("2020-02-01"),
        end=pd.Timestamp("2020-12-31"),
        dataset_version=dataset,
        store=store,
        calendar=calendar,
        schedule=schedule,
        universe_builder=universe_builder,
        feature_computer=feature_computer,
        alpha_model=alpha_model,
        risk_model=risk_model,
        portfolio_constructor=portfolio_constructor,
        cost_model=None,
    )

    assert "cagr" in result.metrics
    assert len(result.equity_curve) > 10
