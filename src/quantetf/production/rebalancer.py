"""Regime-aware production rebalancer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json
import logging

import pandas as pd
import yaml

from quantetf.production.regime_monitor import DailyRegimeMonitor
from quantetf.regime.config import load_regime_mapping, get_strategy_for_regime

if TYPE_CHECKING:
    from quantetf.data.access import DataAccessContext

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
    notional_value: float

    @property
    def shares_delta(self) -> float:
        return self.target_shares - self.current_shares


@dataclass
class RebalanceResult:
    """Result of a rebalance operation."""

    as_of: pd.Timestamp
    regime: str
    strategy_used: str
    target_portfolio: pd.DataFrame
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
        total_traded = sum(
            abs(t.notional_value) for t in self.trades if t.action != "HOLD"
        )
        portfolio_value = sum(abs(t.notional_value) for t in self.trades)
        return total_traded / portfolio_value if portfolio_value > 0 else 0


class RegimeAwareRebalancer:
    """Production rebalancer that selects strategy based on current regime.

    This component:
    1. Gets current regime from monitor
    2. Looks up appropriate strategy from regime mapping
    3. Runs the strategy's alpha model
    4. Constructs target portfolio
    5. Generates trades from current to target
    6. Saves results to artifacts

    Usage:
        from quantetf.data.access import DataAccessFactory
        from quantetf.production.regime_monitor import DailyRegimeMonitor

        ctx = DataAccessFactory.create_context(...)
        monitor = DailyRegimeMonitor(ctx)
        rebalancer = RegimeAwareRebalancer(ctx, monitor)

        result = rebalancer.rebalance(as_of=pd.Timestamp("2026-01-24"))
    """

    def __init__(
        self,
        data_access: "DataAccessContext",
        regime_monitor: DailyRegimeMonitor,
        regime_mapping_path: Optional[Path] = None,
        strategy_configs_dir: Path = Path("configs/strategies"),
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        portfolio_value: float = 1_000_000.0,
    ):
        """Initialize rebalancer.

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
        """Execute rebalance based on current regime.

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

        # Step 4: Run alpha model and construct portfolio
        target_portfolio = self._run_strategy(strategy_config, as_of)

        # Step 5: Get current holdings
        current_portfolio = self._load_current_holdings()

        # Step 6: Generate trades
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
                "strategy_config_path": str(config_path) if config_path else None,
                "portfolio_value": self.portfolio_value,
            },
        )

        # Step 7: Save artifacts
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
            logger.warning(f"Strategy config not found: {path}, using defaults")
            return {
                "alpha_model": strategy_name,
                "parameters": {},
                "top_n": 5,
                "weighting": "equal",
            }

        with open(path) as f:
            return yaml.safe_load(f)

    def _run_strategy(
        self,
        strategy_config: dict,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Run strategy to get target portfolio.

        This is a simplified implementation. A full implementation would use
        AlphaModelRegistry and PortfolioConstructor.
        """
        from quantetf.alpha.factory import AlphaModelRegistry

        # Get alpha model type and parameters
        alpha_model_config = strategy_config.get("alpha_model", {})
        if isinstance(alpha_model_config, dict):
            alpha_type = alpha_model_config.get("type", "momentum_acceleration")
            params = {k: v for k, v in alpha_model_config.items() if k != "type"}
        else:
            # Legacy format: alpha_model is a string
            alpha_type = alpha_model_config or "momentum_acceleration"
            params = strategy_config.get("parameters", {})
        top_n = strategy_config.get("top_n", 5)

        # Create alpha model
        try:
            alpha_model = AlphaModelRegistry.create(alpha_type, params)
        except Exception as e:
            logger.warning(f"Failed to create alpha model: {e}, using fallback")
            return self._create_fallback_portfolio(as_of, top_n)

        # Get universe and prices
        try:
            prices = self.data_access.prices.read_prices_as_of(
                as_of=as_of,
                tickers=None,  # All available
            )
        except Exception as e:
            logger.error(f"Failed to get prices: {e}")
            return self._create_fallback_portfolio(as_of, top_n)

        # Extract close prices
        if isinstance(prices.columns, pd.MultiIndex):
            close_prices = prices.xs("Close", level=1, axis=1)
        else:
            close_prices = prices

        # Get universe from prices
        universe = list(close_prices.columns)

        # Compute alpha scores
        try:
            scores = alpha_model.score(
                as_of=as_of,
                universe=universe,
                features=pd.DataFrame(),
                store=None,
            )
        except Exception as e:
            logger.warning(f"Alpha scoring failed: {e}, using fallback")
            return self._create_fallback_portfolio(as_of, top_n)

        # Select top N
        top_tickers = scores.nlargest(top_n).index.tolist()

        # Equal weight
        weight = 1.0 / len(top_tickers)

        # Get latest prices
        latest_prices = close_prices.iloc[-1]

        # Build portfolio
        portfolio_data = []
        for ticker in top_tickers:
            price = latest_prices.get(ticker, 100.0)
            notional = weight * self.portfolio_value
            shares = notional / price

            portfolio_data.append({
                "ticker": ticker,
                "weight": weight,
                "price": price,
                "notional": notional,
                "shares": shares,
            })

        return pd.DataFrame(portfolio_data).set_index("ticker")

    def _create_fallback_portfolio(
        self,
        as_of: pd.Timestamp,
        top_n: int,
    ) -> pd.DataFrame:
        """Create a simple fallback portfolio."""
        # Simple fallback to major ETFs
        fallback_tickers = ["SPY", "QQQ", "IWM", "VTI", "VEA"][:top_n]
        weight = 1.0 / len(fallback_tickers)

        portfolio_data = []
        for ticker in fallback_tickers:
            notional = weight * self.portfolio_value
            portfolio_data.append({
                "ticker": ticker,
                "weight": weight,
                "price": 100.0,  # Placeholder
                "notional": notional,
                "shares": notional / 100.0,
            })

        return pd.DataFrame(portfolio_data).set_index("ticker")

    def _load_current_holdings(self) -> pd.DataFrame:
        """Load current portfolio holdings."""
        if not self.holdings_file.exists():
            return pd.DataFrame(
                columns=["ticker", "weight", "shares", "price", "notional"]
            )

        try:
            with open(self.holdings_file) as f:
                data = json.load(f)
            return pd.DataFrame(data["holdings"]).set_index("ticker")
        except Exception as e:
            logger.warning(f"Failed to load holdings: {e}")
            return pd.DataFrame(
                columns=["ticker", "weight", "shares", "price", "notional"]
            )

    def _generate_trades(
        self,
        current: pd.DataFrame,
        target: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> List[Trade]:
        """Generate trades to move from current to target."""
        trades = []

        # All tickers involved
        current_tickers = set(current.index) if not current.empty else set()
        target_tickers = set(target.index) if not target.empty else set()
        all_tickers = current_tickers | target_tickers

        for ticker in all_tickers:
            current_shares = (
                current.loc[ticker, "shares"] if ticker in current.index else 0
            )
            current_weight = (
                current.loc[ticker, "weight"] if ticker in current.index else 0
            )
            target_shares = (
                target.loc[ticker, "shares"] if ticker in target.index else 0
            )
            target_weight = (
                target.loc[ticker, "weight"] if ticker in target.index else 0
            )

            delta = target_shares - current_shares

            # Get price
            if ticker in target.index:
                price = target.loc[ticker, "price"]
            elif ticker in current.index:
                price = current.loc[ticker, "price"]
            else:
                price = 100.0

            # Determine action
            if abs(delta) < 0.01:
                action = "HOLD"
            elif delta > 0:
                action = "BUY"
            else:
                action = "SELL"

            trades.append(
                Trade(
                    ticker=ticker,
                    action=action,
                    current_shares=current_shares,
                    target_shares=target_shares,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    notional_value=abs(delta * price),
                )
            )

        return trades

    def _save_artifacts(self, result: RebalanceResult) -> None:
        """Save rebalance artifacts."""
        output_dir = self.artifacts_dir / result.as_of.strftime("%Y%m%d")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save target portfolio
        result.target_portfolio.to_csv(output_dir / "target_portfolio.csv")

        # Save trades
        trades_df = pd.DataFrame(
            [
                {
                    "ticker": t.ticker,
                    "action": t.action,
                    "current_shares": t.current_shares,
                    "target_shares": t.target_shares,
                    "shares_delta": t.shares_delta,
                    "notional_value": t.notional_value,
                }
                for t in result.trades
            ]
        )
        trades_df.to_csv(output_dir / "trades.csv", index=False)

        # Save execution log
        with open(output_dir / "execution_log.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save full result for audit
        with open(output_dir / "rebalance_result.json", "w") as f:
            json.dump(
                {
                    **result.to_dict(),
                    "metadata": result.metadata,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Artifacts saved to {output_dir}")

    def _update_holdings(self, target_portfolio: pd.DataFrame) -> None:
        """Update current holdings state."""
        self.holdings_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "updated_at": pd.Timestamp.now().isoformat(),
            "holdings": target_portfolio.reset_index().to_dict(orient="records"),
        }

        with open(self.holdings_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
