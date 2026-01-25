# Task Handoff: IMPL-035g - Regime-Aware Production Rebalancer

**Task ID:** IMPL-035g
**Parent Task:** IMPL-035 (Regime-Based Strategy Selection System)
**Status:** ready
**Priority:** HIGH
**Type:** Production Infrastructure
**Estimated Effort:** 3-4 hours
**Dependencies:** IMPL-035f (Daily Monitor)

---

## Summary

Implement the Regime-Aware Production Rebalancer that executes the appropriate strategy based on current regime. This is the core production component that generates portfolio recommendations on rebalance dates.

---

## Deliverables

1. **`src/quantetf/production/rebalancer.py`** - RegimeAwareRebalancer class
2. **`tests/production/test_rebalancer.py`** - Unit tests
3. **Output artifacts:** Rebalance recommendations in `artifacts/rebalance/{date}/`

---

## Technical Specification

### Interface Design

```python
# src/quantetf/production/rebalancer.py
"""Regime-aware production rebalancer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import yaml
import json
import logging

from quantetf.production.regime_monitor import DailyRegimeMonitor
from quantetf.alpha import AlphaFactory
from quantetf.portfolio import PortfolioConstructor
from quantetf.data.access import DataAccessContext
from quantetf.regime.config import load_regime_mapping, get_strategy_for_regime

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path("artifacts/rebalance")


@dataclass
class Trade:
    """Represents a single trade to execute."""
    ticker: str
    action: str  # "BUY", "SELL", "HOLD"
    current_shares: float
    target_shares: float
    current_weight: float
    target_weight: float
    notional_value: float  # Dollar amount of trade

    @property
    def shares_delta(self) -> float:
        return self.target_shares - self.current_shares


@dataclass
class RebalanceResult:
    """Result of a rebalance operation."""
    as_of: pd.Timestamp
    regime: str
    strategy_used: str
    target_portfolio: pd.DataFrame  # Columns: ticker, weight, shares
    trades: List[Trade]
    current_portfolio: pd.DataFrame
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "regime": self.regime,
            "strategy_used": self.strategy_used,
            "num_positions": len(self.target_portfolio),
            "num_trades": len([t for t in self.trades if t.action != "HOLD"]),
            "turnover": self._calculate_turnover(),
        }

    def _calculate_turnover(self) -> float:
        """Calculate portfolio turnover."""
        total_traded = sum(abs(t.notional_value) for t in self.trades if t.action != "HOLD")
        portfolio_value = sum(abs(t.notional_value) for t in self.trades)
        return total_traded / portfolio_value if portfolio_value > 0 else 0


class RegimeAwareRebalancer:
    """
    Production rebalancer that selects strategy based on current regime.

    This component:
    1. Gets current regime from monitor
    2. Looks up appropriate strategy from regime mapping
    3. Runs the strategy's alpha model
    4. Constructs target portfolio
    5. Generates trades from current to target
    6. Saves results to artifacts
    """

    def __init__(
        self,
        data_access: DataAccessContext,
        regime_monitor: DailyRegimeMonitor,
        regime_mapping_path: Optional[Path] = None,
        strategy_configs_dir: Path = Path("configs/strategies"),
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        portfolio_value: float = 1_000_000.0,  # $1M default
    ):
        """
        Initialize rebalancer.

        Args:
            data_access: Data access context
            regime_monitor: Monitor for getting current regime
            regime_mapping_path: Path to regime_mapping.yaml
            strategy_configs_dir: Directory containing strategy configs
            artifacts_dir: Output directory for rebalance artifacts
            portfolio_value: Total portfolio value for position sizing
        """
        self.data_access = data_access
        self.regime_monitor = regime_monitor
        self.strategy_configs_dir = Path(strategy_configs_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.portfolio_value = portfolio_value

        # Load regime mapping
        self.regime_mapping = load_regime_mapping(regime_mapping_path)

        # Track current holdings
        self.holdings_file = Path("data/state/current_holdings.json")

    def rebalance(
        self,
        as_of: Optional[pd.Timestamp] = None,
        dry_run: bool = False,
    ) -> RebalanceResult:
        """
        Execute rebalance based on current regime.

        Args:
            as_of: Rebalance date (defaults to today)
            dry_run: If True, don't persist state changes

        Returns:
            RebalanceResult with target portfolio and trades
        """
        as_of = as_of or pd.Timestamp.now().normalize()

        logger.info(f"Starting rebalance for {as_of.date()}")

        # Step 1: Get current regime
        regime_state = self.regime_monitor.update(as_of)
        regime_name = regime_state.name

        logger.info(f"Current regime: {regime_name}")

        # Step 2: Look up strategy for regime
        strategy_info = get_strategy_for_regime(regime_name, self.regime_mapping)
        strategy_name = strategy_info["strategy"]
        config_path = strategy_info.get("config_path")

        logger.info(f"Selected strategy: {strategy_name}")

        # Step 3: Load strategy config
        strategy_config = self._load_strategy_config(strategy_name, config_path)

        # Step 4: Run alpha model
        alpha_scores = self._run_alpha_model(strategy_config, as_of)

        # Step 5: Construct target portfolio
        target_portfolio = self._construct_portfolio(
            alpha_scores=alpha_scores,
            strategy_config=strategy_config,
            as_of=as_of,
        )

        # Step 6: Get current holdings
        current_portfolio = self._load_current_holdings()

        # Step 7: Generate trades
        trades = self._generate_trades(
            current=current_portfolio,
            target=target_portfolio,
            as_of=as_of,
        )

        # Create result
        result = RebalanceResult(
            as_of=as_of,
            regime=regime_name,
            strategy_used=strategy_name,
            target_portfolio=target_portfolio,
            trades=trades,
            current_portfolio=current_portfolio,
            metadata={
                "regime_indicators": {
                    "spy_price": regime_state.spy_price,
                    "spy_200ma": regime_state.spy_200ma,
                    "vix": regime_state.vix,
                },
                "strategy_config_path": str(config_path),
                "portfolio_value": self.portfolio_value,
            },
        )

        # Step 8: Save artifacts
        if not dry_run:
            self._save_artifacts(result)
            self._update_holdings(target_portfolio)

        logger.info(
            f"Rebalance complete: {len(trades)} trades, "
            f"turnover={result._calculate_turnover():.1%}"
        )

        return result

    def dry_run(self, as_of: Optional[pd.Timestamp] = None) -> RebalanceResult:
        """Execute rebalance in dry-run mode (no state changes)."""
        return self.rebalance(as_of=as_of, dry_run=True)

    def _load_strategy_config(
        self,
        strategy_name: str,
        config_path: Optional[str],
    ) -> dict:
        """Load strategy configuration."""
        if config_path:
            path = Path(config_path)
        else:
            # Try to find config by strategy name
            path = self.strategy_configs_dir / f"{strategy_name}.yaml"

        if not path.exists():
            raise FileNotFoundError(f"Strategy config not found: {path}")

        with open(path) as f:
            return yaml.safe_load(f)

    def _run_alpha_model(
        self,
        strategy_config: dict,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Run alpha model to get scores."""
        alpha_model = AlphaFactory.create(
            model_type=strategy_config["alpha_model"],
            params=strategy_config.get("parameters", {}),
        )

        # Get universe
        universe = self.data_access.universe.get_universe_as_of(as_of)

        # Get price data
        prices = self.data_access.prices.read_prices_as_of(
            as_of=as_of,
            tickers=universe,
        )

        # Compute scores
        scores = alpha_model.compute_scores(prices, as_of=as_of)

        return scores

    def _construct_portfolio(
        self,
        alpha_scores: pd.Series,
        strategy_config: dict,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Construct target portfolio from alpha scores."""
        constructor = PortfolioConstructor(
            top_n=strategy_config.get("top_n", 10),
            weighting=strategy_config.get("weighting", "equal"),
        )

        weights = constructor.construct(alpha_scores)

        # Get prices for position sizing
        prices = self.data_access.prices.read_prices_as_of(
            as_of=as_of,
            tickers=weights.index.tolist(),
        )
        latest_prices = prices.xs("Close", level="Price", axis=1).iloc[-1]

        # Calculate shares
        portfolio = pd.DataFrame({
            "ticker": weights.index,
            "weight": weights.values,
        })
        portfolio["price"] = portfolio["ticker"].map(latest_prices)
        portfolio["notional"] = portfolio["weight"] * self.portfolio_value
        portfolio["shares"] = portfolio["notional"] / portfolio["price"]

        return portfolio.set_index("ticker")

    def _load_current_holdings(self) -> pd.DataFrame:
        """Load current portfolio holdings."""
        if not self.holdings_file.exists():
            return pd.DataFrame(columns=["ticker", "weight", "shares", "price", "notional"])

        with open(self.holdings_file) as f:
            data = json.load(f)

        return pd.DataFrame(data["holdings"]).set_index("ticker")

    def _generate_trades(
        self,
        current: pd.DataFrame,
        target: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> List[Trade]:
        """Generate trades to move from current to target."""
        trades = []

        # All tickers involved
        all_tickers = set(current.index) | set(target.index)

        for ticker in all_tickers:
            current_shares = current.loc[ticker, "shares"] if ticker in current.index else 0
            current_weight = current.loc[ticker, "weight"] if ticker in current.index else 0
            target_shares = target.loc[ticker, "shares"] if ticker in target.index else 0
            target_weight = target.loc[ticker, "weight"] if ticker in target.index else 0

            delta = target_shares - current_shares
            price = target.loc[ticker, "price"] if ticker in target.index else current.loc[ticker, "price"]

            if abs(delta) < 0.01:  # Threshold for "no trade"
                action = "HOLD"
            elif delta > 0:
                action = "BUY"
            else:
                action = "SELL"

            trades.append(Trade(
                ticker=ticker,
                action=action,
                current_shares=current_shares,
                target_shares=target_shares,
                current_weight=current_weight,
                target_weight=target_weight,
                notional_value=abs(delta * price),
            ))

        return trades

    def _save_artifacts(self, result: RebalanceResult) -> None:
        """Save rebalance artifacts."""
        output_dir = self.artifacts_dir / result.as_of.strftime("%Y%m%d")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save target portfolio
        result.target_portfolio.to_csv(output_dir / "target_portfolio.csv")

        # Save trades
        trades_df = pd.DataFrame([
            {
                "ticker": t.ticker,
                "action": t.action,
                "current_shares": t.current_shares,
                "target_shares": t.target_shares,
                "shares_delta": t.shares_delta,
                "notional_value": t.notional_value,
            }
            for t in result.trades
        ])
        trades_df.to_csv(output_dir / "trades.csv", index=False)

        # Save execution log
        with open(output_dir / "execution_log.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save full result for audit
        with open(output_dir / "rebalance_result.json", "w") as f:
            json.dump({
                **result.to_dict(),
                "metadata": result.metadata,
            }, f, indent=2, default=str)

        logger.info(f"Artifacts saved to {output_dir}")

    def _update_holdings(self, target_portfolio: pd.DataFrame) -> None:
        """Update current holdings state."""
        self.holdings_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "updated_at": pd.Timestamp.now().isoformat(),
            "holdings": target_portfolio.reset_index().to_dict(orient="records"),
        }

        with open(self.holdings_file, "w") as f:
            json.dump(data, f, indent=2)
```

---

## Test Cases

```python
# tests/production/test_rebalancer.py
import pytest
import pandas as pd
from pathlib import Path

from quantetf.production.rebalancer import RegimeAwareRebalancer, Trade
from quantetf.production.regime_monitor import DailyRegimeMonitor


class TestRegimeAwareRebalancer:
    """Test regime-aware rebalancing."""

    @pytest.fixture
    def rebalancer(self, data_access, tmp_path):
        monitor = DailyRegimeMonitor(
            data_access=data_access,
            state_dir=tmp_path / "state",
        )
        return RegimeAwareRebalancer(
            data_access=data_access,
            regime_monitor=monitor,
            artifacts_dir=tmp_path / "artifacts",
        )

    def test_rebalance_returns_result(self, rebalancer):
        """Rebalance should return RebalanceResult."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        assert result is not None
        assert result.regime in [
            "uptrend_low_vol",
            "uptrend_high_vol",
            "downtrend_low_vol",
            "downtrend_high_vol",
        ]
        assert len(result.target_portfolio) > 0

    def test_dry_run_no_state_change(self, rebalancer, tmp_path):
        """Dry run should not update holdings."""
        rebalancer.rebalance(as_of=pd.Timestamp("2026-01-20"), dry_run=True)

        # Holdings file should not exist
        assert not rebalancer.holdings_file.exists()

    def test_rebalance_saves_artifacts(self, rebalancer, tmp_path):
        """Rebalance should save artifacts."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=False,
        )

        artifacts_dir = tmp_path / "artifacts" / "20260120"
        assert (artifacts_dir / "target_portfolio.csv").exists()
        assert (artifacts_dir / "trades.csv").exists()
        assert (artifacts_dir / "execution_log.json").exists()

    def test_strategy_selected_by_regime(self, rebalancer):
        """Strategy should be selected based on regime."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        # Strategy should match regime mapping
        assert result.strategy_used is not None
        assert len(result.strategy_used) > 0

    def test_trades_generated(self, rebalancer):
        """Trades should be generated."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        assert len(result.trades) > 0
        for trade in result.trades:
            assert trade.action in ["BUY", "SELL", "HOLD"]


class TestTrade:
    """Test Trade dataclass."""

    def test_shares_delta(self):
        trade = Trade(
            ticker="SPY",
            action="BUY",
            current_shares=10,
            target_shares=15,
            current_weight=0.1,
            target_weight=0.15,
            notional_value=500,
        )
        assert trade.shares_delta == 5
```

---

## Output Artifacts

Each rebalance creates:

```
artifacts/rebalance/20260124/
├── target_portfolio.csv    # Target weights and shares
├── trades.csv              # Individual trades to execute
├── execution_log.json      # Summary metrics
└── rebalance_result.json   # Full result for audit
```

---

## Acceptance Criteria

- [ ] Rebalancer selects strategy based on current regime
- [ ] Target portfolio is correctly constructed from alpha scores
- [ ] Trades are correctly generated (BUY/SELL/HOLD)
- [ ] Artifacts are saved to correct directory
- [ ] Dry-run mode works without state changes
- [ ] Holdings state is updated after rebalance
- [ ] All tests pass

---

**Document Version:** 1.0
**Created:** 2026-01-24
**For:** Coding Agent
