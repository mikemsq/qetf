"""
Strategy optimization module.

This module provides tools for automatically finding strategies that beat SPY
across multiple time periods (3yr, 5yr, 10yr).

Components:
- grid: Parameter grid generator for all alpha models
- evaluator: Multi-period strategy evaluation
- optimizer: Main orchestrator for optimization sweeps
"""

from quantetf.optimization.grid import (
    generate_configs,
    count_configs,
    StrategyConfig,
    PARAMETER_GRIDS_WEEKLY,
    PARAMETER_GRIDS_MONTHLY,
    PORTFOLIO_GRIDS,
    UNIVERSE_OPTIONS,
    SCHEDULE_OPTIONS,
)

__all__ = [
    'generate_configs',
    'count_configs',
    'StrategyConfig',
    'PARAMETER_GRIDS_WEEKLY',
    'PARAMETER_GRIDS_MONTHLY',
    'PORTFOLIO_GRIDS',
    'UNIVERSE_OPTIONS',
    'SCHEDULE_OPTIONS',
]
