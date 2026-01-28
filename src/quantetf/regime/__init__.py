"""Regime detection module for market regime classification.

This module provides tools for classifying market regimes based on
trend (SPY vs 200MA) and volatility (VIX level) with hysteresis support.
"""

from .types import RegimeConfig, RegimeState, TrendState, VolatilityState
from .detector import RegimeDetector
from .config import (
    load_thresholds,
    load_regime_mapping,
    get_strategy_for_regime,
)
from .indicators import RegimeIndicators
from .analyzer import RegimeAnalyzer

__all__ = [
    "RegimeAnalyzer",
    "RegimeConfig",
    "RegimeDetector",
    "RegimeIndicators",
    "RegimeState",
    "TrendState",
    "VolatilityState",
    "get_strategy_for_regime",
    "load_regime_mapping",
    "load_thresholds",
]
