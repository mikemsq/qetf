from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
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


@dataclass(frozen=True)
class SlippageCostModel(CostModel):
    """Volume-based slippage cost model.

    Costs increase with trade size relative to position changes.
    Models the impact of larger trades incurring more slippage.

    Formula: cost_bps = base_spread + |weight_change| * impact_coefficient * 100

    Args:
        base_spread_bps: Minimum spread cost (default 5.0 bps)
        impact_coefficient: Impact per percentage point of position change (default 2.0)

    Example:
        >>> model = SlippageCostModel(base_spread_bps=5.0, impact_coefficient=2.0)
        >>> # Going from 0% to 10% position → 5 + 10*2 = 25 bps cost
        >>> cost = model.estimate_rebalance_cost(
        ...     prev_weights=pd.Series([0.5, 0.5], index=['SPY', 'QQQ']),
        ...     next_weights=pd.Series([0.6, 0.4], index=['SPY', 'QQQ'])
        ... )
    """

    base_spread_bps: float = 5.0
    impact_coefficient: float = 2.0

    def __post_init__(self):
        """Validate parameters."""
        object.__setattr__(self, 'base_spread_bps', float(self.base_spread_bps))
        object.__setattr__(self, 'impact_coefficient', float(self.impact_coefficient))

        if self.base_spread_bps < 0:
            raise ValueError(f"base_spread_bps must be >= 0, got {self.base_spread_bps}")
        if self.impact_coefficient < 0:
            raise ValueError(f"impact_coefficient must be >= 0, got {self.impact_coefficient}")

    def estimate_rebalance_cost(
        self,
        *,
        prev_weights: pd.Series,
        next_weights: pd.Series,
        prices: pd.Series | None = None,
    ) -> float:
        """Calculate slippage cost based on position changes.

        Cost = base_spread + |weight_change| * impact_coefficient

        Args:
            prev_weights: Series of old portfolio weights (ticker -> weight)
            next_weights: Series of new portfolio weights (ticker -> weight)
            prices: Not used in this simplified model

        Returns:
            Total cost as fraction of NAV

        Example:
            >>> old_weights = pd.Series({'SPY': 0.5, 'QQQ': 0.5})
            >>> new_weights = pd.Series({'SPY': 0.7, 'QQQ': 0.3})
            >>> model = SlippageCostModel(base_spread_bps=5.0, impact_coefficient=2.0)
            >>> cost = model.estimate_rebalance_cost(
            ...     prev_weights=old_weights,
            ...     next_weights=new_weights,
            ...     prices=None
            ... )
            >>> # SPY: |0.7 - 0.5| = 0.2 → 5 + 0.2*2*100 = 45 bps
            >>> # QQQ: |0.3 - 0.5| = 0.2 → 5 + 0.2*2*100 = 45 bps
            >>> # Cost weighted by turnover
        """
        # Handle empty inputs
        if prev_weights is None or len(prev_weights) == 0:
            prev = pd.Series(dtype=float)
        else:
            prev = prev_weights.fillna(0.0)

        if next_weights is None or len(next_weights) == 0:
            nxt = pd.Series(dtype=float)
        else:
            nxt = next_weights.fillna(0.0)

        # Align tickers
        all_tickers = sorted(set(prev.index).union(set(nxt.index)))
        if not all_tickers:
            return 0.0

        prev = prev.reindex(all_tickers, fill_value=0.0)
        nxt = nxt.reindex(all_tickers, fill_value=0.0)

        # Calculate per-ticker weight changes
        weight_changes = (nxt - prev).abs()

        # Calculate per-ticker costs in bps
        costs_bps = self.base_spread_bps + weight_changes * self.impact_coefficient * 100.0

        # Weight-average cost by trade size
        total_trade_size = weight_changes.sum()
        if total_trade_size == 0:
            return 0.0

        # Weighted average of costs
        avg_cost_bps = (costs_bps * weight_changes).sum() / total_trade_size

        # Return as fraction of NAV
        return float(avg_cost_bps / 10_000.0)


@dataclass(frozen=True)
class SpreadCostModel(CostModel):
    """Bid-ask spread cost model.

    Different ETFs have different spreads based on liquidity.
    Liquid ETFs (SPY, QQQ) have ~1 bp spread, illiquid ETFs can have 50+ bp.

    Args:
        spread_map: Dict mapping ticker -> spread in bps (default spreads if not specified)
        default_spread_bps: Spread for tickers not in map (default 10.0 bps)

    Example:
        >>> model = SpreadCostModel()  # Uses default spreads
        >>> model = SpreadCostModel(spread_map={'SPY': 1.0, 'ARKK': 20.0})
        >>> cost = model.estimate_rebalance_cost(
        ...     prev_weights=pd.Series([1.0], index=['SPY']),
        ...     next_weights=pd.Series([1.0], index=['QQQ'])
        ... )
    """

    spread_map: dict[str, float] | None = None
    default_spread_bps: float = 10.0

    def __post_init__(self):
        """Initialize with default spreads if not provided."""
        # Initialize spread_map with defaults
        if self.spread_map is None:
            default_spreads = {
                # Highly liquid
                'SPY': 1.0,
                'QQQ': 1.0,
                'IWM': 2.0,
                'VOO': 1.5,
                'VTI': 1.0,
                'VEA': 2.0,
                'VWO': 3.0,
                # Moderately liquid
                'EEM': 5.0,
                'TLT': 3.0,
                'GLD': 2.0,
                'VNQ': 5.0,
                'AGG': 2.0,
                'BND': 2.0,
                'XLF': 2.0,
                'XLE': 2.0,
                'XLK': 2.0,
                'XLV': 2.0,
                # Less liquid
                'ARKK': 15.0,
                'ARKG': 15.0,
                'TAN': 20.0,
                'ICLN': 15.0,
            }
            object.__setattr__(self, 'spread_map', default_spreads)

        if self.default_spread_bps < 0:
            raise ValueError(f"default_spread_bps must be >= 0, got {self.default_spread_bps}")

    def estimate_rebalance_cost(
        self,
        *,
        prev_weights: pd.Series,
        next_weights: pd.Series,
        prices: pd.Series | None = None,
    ) -> float:
        """Calculate cost based on bid-ask spreads.

        For each traded ticker, apply its spread cost.

        Args:
            prev_weights: Old portfolio weights
            next_weights: New portfolio weights
            prices: Not used in this model

        Returns:
            Total cost as fraction of NAV

        Example:
            >>> old_weights = pd.Series({'SPY': 1.0})
            >>> new_weights = pd.Series({'QQQ': 1.0})
            >>> model = SpreadCostModel()
            >>> cost = model.estimate_rebalance_cost(
            ...     prev_weights=old_weights,
            ...     next_weights=new_weights,
            ...     prices=None
            ... )
            >>> # Trade 100% out of SPY (1 bp) and 100% into QQQ (1 bp)
        """
        # Handle empty inputs
        if prev_weights is None or len(prev_weights) == 0:
            prev = pd.Series(dtype=float)
        else:
            prev = prev_weights.fillna(0.0)

        if next_weights is None or len(next_weights) == 0:
            nxt = pd.Series(dtype=float)
        else:
            nxt = next_weights.fillna(0.0)

        all_tickers = sorted(set(prev.index).union(set(nxt.index)))
        if not all_tickers:
            return 0.0

        prev = prev.reindex(all_tickers, fill_value=0.0)
        nxt = nxt.reindex(all_tickers, fill_value=0.0)

        # Calculate turnover per ticker
        turnover = (nxt - prev).abs()

        # Get spread for each ticker
        spreads = pd.Series({
            ticker: self.spread_map.get(ticker, self.default_spread_bps)
            for ticker in all_tickers
        })

        # Cost = turnover-weighted average spread
        total_turnover = turnover.sum()
        if total_turnover == 0:
            return 0.0

        avg_spread_bps = (spreads * turnover).sum() / total_turnover

        # Return as fraction of NAV
        return float(avg_spread_bps / 10_000.0)


@dataclass(frozen=True)
class ImpactCostModel(CostModel):
    """Market impact cost model using square-root law.

    Larger trades incur disproportionately higher costs due to market impact.
    The square-root law captures this non-linear relationship.

    Formula: impact_bps = coefficient * sqrt(abs(weight_change))

    This gives:
    - 1% trade → ~0.05% cost
    - 4% trade → ~0.10% cost (2x for 4x size)
    - 16% trade → ~0.20% cost (4x for 16x size)

    Args:
        impact_coefficient: Scaling factor for impact (default 5.0)

    Example:
        >>> model = ImpactCostModel(impact_coefficient=5.0)
        >>> cost = model.estimate_rebalance_cost(
        ...     prev_weights=pd.Series({'SPY': 0.5}),
        ...     next_weights=pd.Series({'SPY': 0.7})
        ... )
        >>> # weight_change = 0.2 (20%)
        >>> # impact = 5.0 * sqrt(0.2) = 2.24 bps
    """

    impact_coefficient: float = 5.0

    def __post_init__(self):
        """Validate parameters."""
        if self.impact_coefficient < 0:
            raise ValueError(f"impact_coefficient must be >= 0, got {self.impact_coefficient}")

    def estimate_rebalance_cost(
        self,
        *,
        prev_weights: pd.Series,
        next_weights: pd.Series,
        prices: pd.Series | None = None,
    ) -> float:
        """Calculate market impact using square-root law.

        Formula: impact_bps = coefficient * sqrt(abs(weight_change))

        This captures the non-linear nature of market impact where
        larger trades incur disproportionately higher costs.

        Args:
            prev_weights: Old portfolio weights
            next_weights: New portfolio weights
            prices: Not used in this model

        Returns:
            Total cost as fraction of NAV

        Example:
            >>> old_weights = pd.Series({'SPY': 0.5})
            >>> new_weights = pd.Series({'SPY': 0.7})
            >>> model = ImpactCostModel(impact_coefficient=5.0)
            >>> cost = model.estimate_rebalance_cost(
            ...     prev_weights=old_weights,
            ...     next_weights=new_weights,
            ...     prices=None
            ... )
            >>> # weight_change = 0.2 (20%)
            >>> # impact = 5.0 * sqrt(0.2) = ~2.24 bps
        """
        # Handle empty inputs
        if prev_weights is None or len(prev_weights) == 0:
            prev = pd.Series(dtype=float)
        else:
            prev = prev_weights.fillna(0.0)

        if next_weights is None or len(next_weights) == 0:
            nxt = pd.Series(dtype=float)
        else:
            nxt = next_weights.fillna(0.0)

        all_tickers = sorted(set(prev.index).union(set(nxt.index)))
        if not all_tickers:
            return 0.0

        prev = prev.reindex(all_tickers, fill_value=0.0)
        nxt = nxt.reindex(all_tickers, fill_value=0.0)

        # Calculate weight changes
        weight_changes = (nxt - prev).abs()

        # Apply square-root law
        impacts_bps = self.impact_coefficient * np.sqrt(weight_changes)

        # Weight-average by trade size
        total_trade_size = weight_changes.sum()
        if total_trade_size == 0:
            return 0.0

        avg_impact_bps = (impacts_bps * weight_changes).sum() / total_trade_size

        return float(avg_impact_bps / 10_000.0)
