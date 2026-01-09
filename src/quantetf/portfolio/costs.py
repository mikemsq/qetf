from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from quantetf.portfolio.base import CostModel


@dataclass(frozen=True)
class SimpleLinearCostModel(CostModel):
    """Linear cost model using a single bps-per-dollar-turnover style assumption."""

    total_bps_per_turnover: float = 10.0  # combined commission + slippage + spread

    def estimate_rebalance_cost(
        self,
        *,
        prev_weights: pd.Series,
        next_weights: pd.Series,
        prices: pd.Series | None = None,
    ) -> float:
        prev = prev_weights.fillna(0.0)
        nxt = next_weights.fillna(0.0)
        tickers = sorted(set(prev.index).union(set(nxt.index)))
        prev = prev.reindex(tickers, fill_value=0.0)
        nxt = nxt.reindex(tickers, fill_value=0.0)

        turnover = 0.5 * (nxt - prev).abs().sum()  # 0..1 for fully invested portfolios
        return (self.total_bps_per_turnover / 10_000.0) * float(turnover)


@dataclass(frozen=True)
class FlatTransactionCost(CostModel):
    """Flat transaction cost model - charges a fixed percentage per trade.

    This is the simplest cost model: charge a flat percentage of the absolute
    change in portfolio weights (turnover). Common for discount brokers with ETFs.

    The cost is applied to one-sided turnover: the sum of absolute weight changes
    divided by 2. For example, if you fully rotate from one ETF to another,
    the turnover is 1.0 (100% of portfolio), not 2.0.

    Args:
        cost_bps: Cost in basis points per dollar of turnover (default: 10 bps = 0.10%)

    Example:
        >>> cost_model = FlatTransactionCost(cost_bps=10.0)  # 10 bps = 0.10%
        >>> cost = cost_model.estimate_rebalance_cost(
        ...     prev_weights=pd.Series([0.5, 0.5], index=['SPY', 'QQQ']),
        ...     next_weights=pd.Series([1.0, 0.0], index=['SPY', 'QQQ'])
        ... )
        >>> # Turnover = 0.5 * (|0.5| + |0.5|) = 0.5
        >>> # Cost = 0.5 * 0.0010 = 0.0005 (5 bps of NAV)
    """

    cost_bps: float = 10.0

    def __post_init__(self):
        """Validate cost_bps parameter."""
        if self.cost_bps < 0:
            raise ValueError(f"cost_bps must be >= 0, got {self.cost_bps}")

    def estimate_rebalance_cost(
        self,
        *,
        prev_weights: pd.Series,
        next_weights: pd.Series,
        prices: pd.Series | None = None,
    ) -> float:
        """Calculate transaction cost as a fraction of NAV.

        Cost = turnover * (cost_bps / 10000)

        where turnover = 0.5 * sum(|weight_change|)

        Args:
            prev_weights: Previous portfolio weights (sum ~1.0)
            next_weights: Target portfolio weights (sum ~1.0)
            prices: Not used in flat model (kept for interface compatibility)

        Returns:
            Transaction cost as a fraction of NAV (e.g., 0.001 = 10 bps)

        Raises:
            ValueError: If inputs are invalid
        """
        # Handle empty or None inputs
        if prev_weights is None or len(prev_weights) == 0:
            prev = pd.Series(dtype=float)
        else:
            prev = prev_weights.fillna(0.0)

        if next_weights is None or len(next_weights) == 0:
            nxt = pd.Series(dtype=float)
        else:
            nxt = next_weights.fillna(0.0)

        # Align tickers
        tickers = sorted(set(prev.index).union(set(nxt.index)))
        if not tickers:
            return 0.0

        prev = prev.reindex(tickers, fill_value=0.0)
        nxt = nxt.reindex(tickers, fill_value=0.0)

        # Calculate one-sided turnover
        turnover = 0.5 * (nxt - prev).abs().sum()

        # Convert bps to decimal and multiply by turnover
        cost_rate = self.cost_bps / 10_000.0
        cost = float(turnover) * cost_rate

        return cost
