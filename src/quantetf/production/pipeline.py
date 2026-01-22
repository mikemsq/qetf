"""Enhanced production pipeline for portfolio management.

This module provides an enhanced production pipeline that integrates:
- Risk overlays for constraint enforcement
- Pre-trade checks for risk validation
- Portfolio state management
- Rebalance scheduling

IMPL-028: Updated to use DataAccessContext (DAL) instead of legacy store parameter.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from quantetf.production.recommendations import diff_trades
from quantetf.types import DatasetVersion, RecommendationPacket

if TYPE_CHECKING:
    from quantetf.data.access import DataAccessContext
    from quantetf.production.state import PortfolioState, PortfolioStateManager
    from quantetf.risk.overlays import RiskOverlay

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pre-Trade Checks
# -----------------------------------------------------------------------------


class PreTradeCheck(ABC):
    """Base class for pre-trade validation checks.

    Pre-trade checks validate proposed trades before execution.
    They can block trades that violate risk constraints or policies.
    """

    @abstractmethod
    def check(
        self,
        trades: pd.DataFrame,
        state: Optional["PortfolioState"],
        as_of: pd.Timestamp,
    ) -> tuple[bool, str]:
        """Validate proposed trades.

        Args:
            trades: DataFrame with columns: ticker, current_weight, target_weight, delta_weight
            state: Current portfolio state (may be None for initial portfolio)
            as_of: Current date

        Returns:
            Tuple of (passed: bool, reason: str). If passed is False, reason
            explains why the check failed.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class MaxTurnoverCheck(PreTradeCheck):
    """Block trades if total turnover exceeds threshold.

    Turnover is calculated as half the sum of absolute weight changes,
    representing the fraction of the portfolio being traded.

    Example:
        With max_turnover=0.50, a rebalance that sells 60% of one position
        and buys 60% of another (total turnover = 0.60) would be blocked.
    """

    max_turnover: float = 0.50

    def check(
        self,
        trades: pd.DataFrame,
        state: Optional["PortfolioState"],
        as_of: pd.Timestamp,
    ) -> tuple[bool, str]:
        """Check if turnover is within limits.

        Args:
            trades: Trade DataFrame
            state: Portfolio state (unused)
            as_of: Current date (unused)

        Returns:
            (passed, reason) tuple
        """
        if trades.empty:
            return True, "No trades"

        turnover = 0.5 * trades["delta_weight"].abs().sum()

        if turnover > self.max_turnover:
            return (
                False,
                f"Turnover {turnover:.2%} exceeds max {self.max_turnover:.2%}",
            )

        return True, f"Turnover {turnover:.2%} within limit"


@dataclass(frozen=True)
class SectorConcentrationCheck(PreTradeCheck):
    """Block trades if any sector exceeds concentration limit.

    This check requires a sector mapping to validate sector exposures.
    If no sector mapping is provided, the check passes by default.
    """

    max_sector_weight: float = 0.40
    sector_map: tuple[tuple[str, str], ...] = ()

    def check(
        self,
        trades: pd.DataFrame,
        state: Optional["PortfolioState"],
        as_of: pd.Timestamp,
    ) -> tuple[bool, str]:
        """Check if sector concentrations are within limits.

        Args:
            trades: Trade DataFrame with target_weight column
            state: Portfolio state (unused)
            as_of: Current date (unused)

        Returns:
            (passed, reason) tuple
        """
        if trades.empty:
            return True, "No trades"

        if not self.sector_map:
            return True, "No sector mapping provided, check skipped"

        # Convert tuple of tuples to dict
        sector_dict = dict(self.sector_map)

        # Calculate sector weights from target weights
        sector_weights: dict[str, float] = {}
        for _, row in trades.iterrows():
            ticker = row["ticker"]
            target_weight = row["target_weight"]

            if target_weight <= 0:
                continue

            sector = sector_dict.get(ticker, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + target_weight

        # Check for violations
        violations = []
        for sector, weight in sector_weights.items():
            if weight > self.max_sector_weight:
                violations.append(f"{sector}: {weight:.2%}")

        if violations:
            return (
                False,
                f"Sector concentration exceeded: {', '.join(violations)}",
            )

        return True, "Sector concentrations within limits"


@dataclass(frozen=True)
class MinTradeThresholdCheck(PreTradeCheck):
    """Filter out trades below minimum threshold.

    This check doesn't block execution but marks trades below threshold
    as filtered. Useful for avoiding small, costly trades.
    """

    min_trade_weight: float = 0.005

    def check(
        self,
        trades: pd.DataFrame,
        state: Optional["PortfolioState"],
        as_of: pd.Timestamp,
    ) -> tuple[bool, str]:
        """Check trades against minimum threshold.

        This check always passes but provides information about
        trades that are below the threshold.

        Args:
            trades: Trade DataFrame
            state: Portfolio state (unused)
            as_of: Current date (unused)

        Returns:
            (passed, reason) tuple - always passes
        """
        if trades.empty:
            return True, "No trades"

        below_threshold = (trades["delta_weight"].abs() < self.min_trade_weight).sum()

        return (
            True,
            f"{below_threshold} trades below {self.min_trade_weight:.2%} threshold",
        )


# -----------------------------------------------------------------------------
# Rebalance Scheduling
# -----------------------------------------------------------------------------


def should_rebalance(
    as_of: pd.Timestamp,
    schedule: str = "monthly",
) -> bool:
    """Determine if a rebalance should occur on the given date.

    Args:
        as_of: Date to check
        schedule: Rebalance schedule. One of:
            - "monthly": Last business day of month
            - "weekly": Friday
            - "daily": Every business day

    Returns:
        True if as_of is a rebalance date

    Raises:
        ValueError: If schedule is not recognized
    """
    if schedule == "daily":
        return True

    if schedule == "weekly":
        # Friday = weekday 4
        return as_of.weekday() == 4

    if schedule == "monthly":
        # Check if next business day is in a different month
        next_bday = as_of + pd.tseries.offsets.BDay(1)
        return as_of.month != next_bday.month

    raise ValueError(f"Unknown schedule: {schedule}. Use 'daily', 'weekly', or 'monthly'")


def get_next_rebalance_date(
    as_of: pd.Timestamp,
    schedule: str = "monthly",
) -> pd.Timestamp:
    """Get the next rebalance date after as_of.

    Args:
        as_of: Starting date
        schedule: Rebalance schedule

    Returns:
        Next rebalance date
    """
    current = as_of

    # Move to next business day if on weekend
    if current.weekday() >= 5:
        current = current + pd.tseries.offsets.BDay(1)

    # Find next rebalance date
    max_iterations = 100
    for _ in range(max_iterations):
        current = current + pd.tseries.offsets.BDay(1)
        if should_rebalance(current, schedule):
            return current

    return current


# -----------------------------------------------------------------------------
# Pipeline Configuration and Result
# -----------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the enhanced production pipeline.

    Attributes:
        strategy_config_path: Path to strategy YAML configuration
        risk_overlays: List of risk overlays to apply (in order)
        pre_trade_checks: List of pre-trade validation checks
        state_manager: Portfolio state persistence manager
        rebalance_schedule: When to rebalance ("monthly", "weekly", "daily")
        trade_threshold: Minimum trade size to execute (weight fraction)
        dry_run: If True, don't persist state changes
    """

    strategy_config_path: Optional[Path] = None
    risk_overlays: list["RiskOverlay"] = field(default_factory=list)
    pre_trade_checks: list[PreTradeCheck] = field(default_factory=list)
    state_manager: Optional["PortfolioStateManager"] = None
    rebalance_schedule: str = "monthly"
    trade_threshold: float = 0.005
    dry_run: bool = False


@dataclass(frozen=True)
class PipelineResult:
    """Result of a production pipeline run.

    Attributes:
        as_of: Date of the pipeline run
        target_weights: Raw target weights from portfolio construction
        adjusted_weights: Weights after risk overlays applied
        trades: DataFrame of proposed trades
        pre_trade_checks_passed: Whether all pre-trade checks passed
        check_results: Results from each pre-trade check
        overlay_diagnostics: Diagnostics from each risk overlay
        execution_status: Status of the run ("pending", "executed", "blocked", "skipped")
    """

    as_of: pd.Timestamp
    target_weights: pd.Series
    adjusted_weights: pd.Series
    trades: pd.DataFrame
    pre_trade_checks_passed: bool
    check_results: tuple[tuple[str, bool, str], ...]
    overlay_diagnostics: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]
    execution_status: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "as_of": str(self.as_of),
            "target_weights": self.target_weights.to_dict(),
            "adjusted_weights": self.adjusted_weights.to_dict(),
            "trades": self.trades.to_dict(orient="records"),
            "pre_trade_checks_passed": self.pre_trade_checks_passed,
            "check_results": {
                name: {"passed": passed, "reason": reason}
                for name, passed, reason in self.check_results
            },
            "overlay_diagnostics": {
                name: dict(diag) for name, diag in self.overlay_diagnostics
            },
            "execution_status": self.execution_status,
        }

    def get_check_results_dict(self) -> dict[str, tuple[bool, str]]:
        """Get check results as a dictionary.

        Returns:
            Dictionary mapping check name to (passed, reason) tuple
        """
        return {name: (passed, reason) for name, passed, reason in self.check_results}

    def get_overlay_diagnostics_dict(self) -> dict[str, dict[str, Any]]:
        """Get overlay diagnostics as a dictionary.

        Returns:
            Dictionary mapping overlay name to diagnostics dict
        """
        return {name: dict(diag) for name, diag in self.overlay_diagnostics}


# -----------------------------------------------------------------------------
# Production Pipeline (Enhanced)
# -----------------------------------------------------------------------------


@dataclass
class ProductionPipeline:
    """Enhanced production pipeline for portfolio management.

    This class orchestrates the end-to-end pipeline:
    1. Load current portfolio state
    2. Check if rebalance is needed
    3. Generate target weights (from alpha scores)
    4. Apply risk overlay chain
    5. Generate trade list
    6. Run pre-trade validation checks
    7. Return PipelineResult

    The pipeline integrates:
    - Risk overlays from IMPL-010
    - Portfolio state management from IMPL-011
    - Pre-trade checks for risk validation
    - Rebalance scheduling

    IMPL-028: Now uses DataAccessContext (DAL) for all data access.

    Example:
        >>> from quantetf.production import ProductionPipeline, PipelineConfig
        >>> from quantetf.production.state import InMemoryStateManager
        >>> from quantetf.risk import PositionLimitOverlay, VolatilityTargeting
        >>> from quantetf.data.access import DataAccessFactory
        >>>
        >>> config = PipelineConfig(
        ...     risk_overlays=[
        ...         VolatilityTargeting(target_vol=0.15),
        ...         PositionLimitOverlay(max_weight=0.25),
        ...     ],
        ...     pre_trade_checks=[MaxTurnoverCheck(max_turnover=0.50)],
        ...     state_manager=InMemoryStateManager(),
        ...     rebalance_schedule="monthly",
        ... )
        >>> pipeline = ProductionPipeline(config=config)
        >>> data_access = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/latest/data.parquet"}
        ... )
        >>> result = pipeline.run_enhanced(
        ...     as_of=pd.Timestamp("2024-01-31"),
        ...     data_access=data_access,
        ...     target_weights=target_weights,
        ... )
    """

    artifacts_root: Path = field(default_factory=lambda: Path("artifacts"))
    config: Optional[PipelineConfig] = None

    def run(
        self,
        *,
        as_of: pd.Timestamp,
        dataset_version: DatasetVersion,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> RecommendationPacket:
        """Legacy run method for backward compatibility.

        This method provides the original simple interface without
        risk overlays or pre-trade checks.

        Args:
            as_of: Current date
            dataset_version: Dataset version
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            RecommendationPacket with trades and summary
        """
        trades = diff_trades(current_weights, target_weights, threshold=0.0)

        summary = {
            "as_of": str(as_of),
            "dataset_id": dataset_version.id,
            "num_trades": int(len(trades)),
            "gross_turnover": float(0.5 * trades["delta_weight"].abs().sum()),
        }

        manifest = {
            "dataset_id": dataset_version.id,
            "as_of": str(as_of),
        }

        packet = RecommendationPacket(
            as_of=as_of,
            target_weights=target_weights,
            trades=trades,
            summary=summary,
            manifest=manifest,
        )
        return packet

    def run_enhanced(
        self,
        *,
        as_of: pd.Timestamp,
        target_weights: pd.Series,
        data_access: Optional["DataAccessContext"] = None,
        current_weights: Optional[pd.Series] = None,
        force_rebalance: bool = False,
    ) -> PipelineResult:
        """Run the enhanced production pipeline.

        Args:
            as_of: Current date for the pipeline run
            target_weights: Target portfolio weights from portfolio construction
            data_access: DataAccessContext for market/macro data (required for risk overlays)
            current_weights: Current portfolio weights (optional, loaded from state if not provided)
            force_rebalance: If True, ignore rebalance schedule

        Returns:
            PipelineResult with adjusted weights, trades, and diagnostics
        """
        config = self.config or PipelineConfig()
        logger.info(f"Running enhanced production pipeline for {as_of}")

        # Step 1: Check rebalance schedule
        if not force_rebalance and not should_rebalance(as_of, config.rebalance_schedule):
            logger.info(f"Not a rebalance date for {config.rebalance_schedule} schedule")
            return PipelineResult(
                as_of=as_of,
                target_weights=target_weights,
                adjusted_weights=target_weights,
                trades=pd.DataFrame(columns=["ticker", "current_weight", "target_weight", "delta_weight"]),
                pre_trade_checks_passed=True,
                check_results=(),
                overlay_diagnostics=(),
                execution_status="skipped",
            )

        # Step 2: Load current state if available
        portfolio_state: Optional["PortfolioState"] = None
        if config.state_manager is not None:
            portfolio_state = config.state_manager.get_latest_state()
            logger.debug(f"Loaded portfolio state: NAV={portfolio_state.nav if portfolio_state else 'N/A'}")

        # Step 3: Get current weights (from state or provided)
        if current_weights is None:
            if portfolio_state is not None:
                current_weights = portfolio_state.weights
            else:
                current_weights = pd.Series(dtype=float)

        # Step 4: Apply risk overlay chain
        adjusted_weights = target_weights.copy()
        overlay_diagnostics_dict: dict[str, dict[str, Any]] = {}

        if config.risk_overlays and data_access is not None:
            from quantetf.risk.overlays import apply_overlay_chain

            adjusted_weights, overlay_diagnostics_dict = apply_overlay_chain(
                target_weights=target_weights,
                overlays=config.risk_overlays,
                as_of=as_of,
                data_access=data_access,
                portfolio_state=portfolio_state,
            )
            logger.info(f"Applied {len(config.risk_overlays)} risk overlays")

        # Convert diagnostics to frozen format
        overlay_diagnostics: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = tuple(
            (name, tuple(diag.items())) for name, diag in overlay_diagnostics_dict.items()
        )

        # Step 5: Generate trade list
        trades = diff_trades(
            current_weights,
            adjusted_weights,
            threshold=config.trade_threshold,
        )
        logger.info(f"Generated {len(trades)} trades")

        # Step 6: Run pre-trade checks
        check_results_list: list[tuple[str, bool, str]] = []
        all_checks_passed = True

        for check in config.pre_trade_checks:
            check_name = type(check).__name__
            passed, reason = check.check(trades, portfolio_state, as_of)
            check_results_list.append((check_name, passed, reason))

            if not passed:
                all_checks_passed = False
                logger.warning(f"Pre-trade check failed: {check_name} - {reason}")
            else:
                logger.debug(f"Pre-trade check passed: {check_name} - {reason}")

        check_results = tuple(check_results_list)

        # Step 7: Determine execution status
        if not all_checks_passed:
            execution_status = "blocked"
        elif trades.empty:
            execution_status = "pending"
        else:
            execution_status = "pending"

        logger.info(f"Pipeline completed: status={execution_status}, checks_passed={all_checks_passed}")

        return PipelineResult(
            as_of=as_of,
            target_weights=target_weights,
            adjusted_weights=adjusted_weights,
            trades=trades,
            pre_trade_checks_passed=all_checks_passed,
            check_results=check_results,
            overlay_diagnostics=overlay_diagnostics,
            execution_status=execution_status,
        )
