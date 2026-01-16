"""Risk overlay implementations for production portfolio management.

Risk overlays modify target weights to enforce risk constraints and
respond to market conditions. They are applied as a chain of responsibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantetf.data.store import DataStore
    from quantetf.production.state import PortfolioState


class RiskOverlay(ABC):
    """Base class for risk overlays.

    Risk overlays transform target weights to enforce risk constraints.
    They are applied sequentially in a chain-of-responsibility pattern.
    """

    @abstractmethod
    def apply(
        self,
        target_weights: pd.Series,
        as_of: pd.Timestamp,
        store: "DataStore",
        portfolio_state: Optional["PortfolioState"],
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Apply the overlay to target weights.

        Args:
            target_weights: Target portfolio weights (ticker -> weight)
            as_of: Current date for overlay calculation
            store: Data store for price/return data access
            portfolio_state: Current portfolio state (may be None for initial)

        Returns:
            Tuple of (adjusted_weights, diagnostics_dict)
        """
        raise NotImplementedError


@dataclass(frozen=True)
class VolatilityTargeting(RiskOverlay):
    """Scale portfolio exposure to target a specific volatility level.

    This overlay estimates realized portfolio volatility and scales all
    positions to achieve the target volatility. This helps maintain
    consistent risk exposure across different market regimes.

    Example:
        If realized vol is 30% and target is 15%, scale factor = 0.5,
        so all weights are halved (remaining goes to cash).
    """

    target_vol: float = 0.15
    lookback_days: int = 60
    min_scale: float = 0.25
    max_scale: float = 1.50
    halflife_days: int = 20

    def apply(
        self,
        target_weights: pd.Series,
        as_of: pd.Timestamp,
        store: "DataStore",
        portfolio_state: Optional["PortfolioState"],
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Scale weights to target volatility."""
        # Get returns for volatility estimation
        start = as_of - pd.tseries.offsets.BDay(self.lookback_days)
        tickers = list(target_weights[target_weights > 0].index)

        if not tickers:
            return target_weights, {"scale_factor": 1.0, "realized_vol": 0.0}

        try:
            prices = store.get_close_prices(tickers=tickers, start=start, end=as_of)
            returns = prices.pct_change().dropna()
        except Exception:
            # If we can't get data, pass through unchanged
            return target_weights, {"scale_factor": 1.0, "error": "Could not load price data"}

        if len(returns) < 20:
            return target_weights, {"scale_factor": 1.0, "error": "Insufficient data"}

        # Calculate EWMA volatility of portfolio
        weights_aligned = target_weights.reindex(returns.columns).fillna(0)
        portfolio_returns = (returns * weights_aligned).sum(axis=1)

        # EWMA volatility
        lam = 0.5 ** (1.0 / self.halflife_days)
        n = len(portfolio_returns)
        weights_ewma = np.array([(1 - lam) * lam ** i for i in range(n - 1, -1, -1)])
        weights_ewma = weights_ewma / weights_ewma.sum()

        variance = (weights_ewma * portfolio_returns.values ** 2).sum()
        realized_vol = np.sqrt(variance * 252)  # Annualize

        # Calculate scale factor
        if realized_vol < 0.01:  # Avoid division by near-zero
            scale_factor = 1.0
        else:
            scale_factor = self.target_vol / realized_vol

        # Clip to bounds
        scale_factor = np.clip(scale_factor, self.min_scale, self.max_scale)

        # Apply scaling
        adjusted = target_weights * scale_factor

        diagnostics = {
            "realized_vol": float(realized_vol),
            "target_vol": self.target_vol,
            "scale_factor": float(scale_factor),
            "lookback_days": self.lookback_days,
        }

        return adjusted, diagnostics


@dataclass(frozen=True)
class PositionLimitOverlay(RiskOverlay):
    """Cap individual position weights and redistribute excess.

    This overlay enforces maximum position sizes to prevent concentration
    risk. Excess weight from capped positions is redistributed proportionally
    to other positions (if redistribute=True) or allocated to cash.
    """

    max_weight: float = 0.25
    min_weight: float = 0.0
    redistribute: bool = True

    def apply(
        self,
        target_weights: pd.Series,
        as_of: pd.Timestamp,
        store: "DataStore",
        portfolio_state: Optional["PortfolioState"],
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Apply position limits."""
        adjusted = target_weights.copy()
        capped_tickers = []
        excess_weight = 0.0

        # Identify and cap positions exceeding max_weight
        for ticker, weight in adjusted.items():
            if weight > self.max_weight:
                excess_weight += weight - self.max_weight
                adjusted[ticker] = self.max_weight
                capped_tickers.append(ticker)

        # Redistribute excess if enabled
        if self.redistribute and excess_weight > 0:
            uncapped = adjusted[(adjusted > 0) & (adjusted < self.max_weight)]
            if len(uncapped) > 0:
                # Distribute proportionally
                uncapped_sum = uncapped.sum()
                if uncapped_sum > 0:
                    for ticker in uncapped.index:
                        share = uncapped[ticker] / uncapped_sum
                        add_weight = excess_weight * share
                        new_weight = adjusted[ticker] + add_weight
                        # Ensure redistributed weight doesn't exceed max
                        adjusted[ticker] = min(new_weight, self.max_weight)

        diagnostics = {
            "capped_tickers": capped_tickers,
            "num_capped": len(capped_tickers),
            "excess_weight": float(excess_weight),
            "max_weight": self.max_weight,
            "total_weight": float(adjusted.sum()),
        }

        return adjusted, diagnostics


@dataclass(frozen=True)
class DrawdownCircuitBreaker(RiskOverlay):
    """Reduce portfolio exposure based on drawdown from peak.

    This overlay monitors portfolio drawdown and reduces exposure when
    drawdowns exceed specified thresholds. This implements a systematic
    de-risking mechanism to limit losses during adverse market conditions.

    Thresholds:
        soft_threshold (10%): Reduce to 75% exposure
        hard_threshold (20%): Reduce to 50% exposure
        exit_threshold (30%): Reduce to 25% exposure (defensive mode)
    """

    soft_threshold: float = 0.10
    hard_threshold: float = 0.20
    exit_threshold: float = 0.30
    soft_exposure: float = 0.75
    hard_exposure: float = 0.50
    exit_exposure: float = 0.25

    def apply(
        self,
        target_weights: pd.Series,
        as_of: pd.Timestamp,
        store: "DataStore",
        portfolio_state: Optional["PortfolioState"],
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Apply drawdown-based exposure reduction."""
        if portfolio_state is None:
            # No state means no drawdown history - pass through
            return target_weights, {
                "drawdown": 0.0,
                "exposure_factor": 1.0,
                "level": "NONE",
            }

        # Calculate current drawdown
        current_nav = portfolio_state.nav
        peak_nav = portfolio_state.peak_nav if hasattr(portfolio_state, "peak_nav") else current_nav

        if peak_nav <= 0:
            drawdown = 0.0
        else:
            drawdown = (peak_nav - current_nav) / peak_nav

        # Determine exposure factor based on drawdown level
        if drawdown >= self.exit_threshold:
            exposure_factor = self.exit_exposure
            level = "EXIT"
        elif drawdown >= self.hard_threshold:
            exposure_factor = self.hard_exposure
            level = "HARD"
        elif drawdown >= self.soft_threshold:
            exposure_factor = self.soft_exposure
            level = "SOFT"
        else:
            exposure_factor = 1.0
            level = "NONE"

        adjusted = target_weights * exposure_factor

        diagnostics = {
            "drawdown": float(drawdown),
            "peak_nav": float(peak_nav),
            "current_nav": float(current_nav),
            "exposure_factor": float(exposure_factor),
            "level": level,
            "thresholds": {
                "soft": self.soft_threshold,
                "hard": self.hard_threshold,
                "exit": self.exit_threshold,
            },
        }

        return adjusted, diagnostics


@dataclass(frozen=True)
class VIXRegimeOverlay(RiskOverlay):
    """Shift to defensive assets when VIX indicates high volatility regime.

    This overlay monitors the VIX index and shifts portfolio allocation
    toward defensive assets (bonds, gold, low-vol ETFs) when VIX exceeds
    specified thresholds. This helps protect the portfolio during market
    stress periods.
    """

    high_vix_threshold: float = 30.0
    elevated_vix_threshold: float = 25.0
    defensive_tickers: tuple[str, ...] = ("AGG", "TLT", "GLD", "USMV", "SPLV")
    high_vix_defensive_weight: float = 0.50
    elevated_vix_defensive_weight: float = 0.25
    macro_data_dir: str = "data/raw/macro"

    def apply(
        self,
        target_weights: pd.Series,
        as_of: pd.Timestamp,
        store: "DataStore",
        portfolio_state: Optional["PortfolioState"],
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Apply VIX-based regime shift."""
        from pathlib import Path

        from quantetf.data.macro_loader import MacroDataLoader

        # Load VIX data
        try:
            macro_loader = MacroDataLoader(data_dir=Path(self.macro_data_dir))
            vix = macro_loader.get_vix(str(as_of.date()))
        except Exception as e:
            # If VIX data unavailable, pass through unchanged
            return target_weights, {
                "vix": None,
                "regime": "UNKNOWN",
                "error": str(e),
            }

        # Determine regime and defensive weight
        if vix >= self.high_vix_threshold:
            regime = "HIGH_VOL"
            defensive_weight = self.high_vix_defensive_weight
        elif vix >= self.elevated_vix_threshold:
            regime = "ELEVATED_VOL"
            defensive_weight = self.elevated_vix_defensive_weight
        else:
            regime = "NORMAL"
            defensive_weight = 0.0

        if defensive_weight == 0.0:
            return target_weights, {
                "vix": float(vix),
                "regime": regime,
                "defensive_weight": 0.0,
            }

        # Shift allocation toward defensive assets
        adjusted = target_weights.copy()

        # Reduce all non-defensive positions proportionally
        non_defensive_mask = ~adjusted.index.isin(self.defensive_tickers)
        non_defensive_sum = adjusted[non_defensive_mask].sum()

        if non_defensive_sum > 0:
            # Scale down non-defensive positions
            scale = (1 - defensive_weight)
            adjusted[non_defensive_mask] *= scale / non_defensive_sum * non_defensive_sum

            # Distribute defensive_weight among available defensive tickers
            available_defensive = [t for t in self.defensive_tickers if t in store.tickers]
            if available_defensive:
                per_ticker_weight = defensive_weight / len(available_defensive)
                for ticker in available_defensive:
                    if ticker in adjusted.index:
                        adjusted[ticker] = adjusted.get(ticker, 0) + per_ticker_weight
                    else:
                        adjusted[ticker] = per_ticker_weight

        diagnostics = {
            "vix": float(vix),
            "regime": regime,
            "defensive_weight": float(defensive_weight),
            "defensive_tickers_used": [t for t in self.defensive_tickers if t in adjusted.index],
            "thresholds": {
                "high": self.high_vix_threshold,
                "elevated": self.elevated_vix_threshold,
            },
        }

        return adjusted, diagnostics


def apply_overlay_chain(
    target_weights: pd.Series,
    overlays: list[RiskOverlay],
    as_of: pd.Timestamp,
    store: "DataStore",
    portfolio_state: Optional["PortfolioState"],
) -> tuple[pd.Series, dict[str, dict[str, Any]]]:
    """Apply a chain of risk overlays to target weights.

    Args:
        target_weights: Initial target portfolio weights
        overlays: List of RiskOverlay instances to apply in order
        as_of: Current date
        store: Data store for market data access
        portfolio_state: Current portfolio state

    Returns:
        Tuple of (final_adjusted_weights, dict_of_all_diagnostics)
    """
    current_weights = target_weights.copy()
    all_diagnostics: dict[str, dict[str, Any]] = {}

    for overlay in overlays:
        overlay_name = type(overlay).__name__
        current_weights, diag = overlay.apply(
            target_weights=current_weights,
            as_of=as_of,
            store=store,
            portfolio_state=portfolio_state,
        )
        all_diagnostics[overlay_name] = diag

    return current_weights, all_diagnostics
