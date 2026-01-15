"""
Strategy optimization module.

This module provides tools for automatically finding strategies that beat SPY
across multiple time periods (3yr, 5yr, 10yr).

Components:
- grid: Parameter grid generator for all alpha models
- evaluator: Multi-period strategy evaluation
- optimizer: Main orchestrator for optimization sweeps

Example:
    >>> from quantetf.optimization import (
    ...     generate_configs,
    ...     MultiPeriodEvaluator,
    ...     StrategyConfig,
    ... )
    >>> from pathlib import Path
    >>>
    >>> # Generate all strategy configurations
    >>> configs = generate_configs()
    >>> print(f"Generated {len(configs)} configurations")
    >>>
    >>> # Evaluate a single configuration
    >>> evaluator = MultiPeriodEvaluator(Path("data/snapshots/snapshot/data.parquet"))
    >>> result = evaluator.evaluate(configs[0], periods_years=[3, 5, 10])
    >>> print(f"Beats SPY: {result.beats_spy_all_periods}")
"""

from quantetf.optimization.grid import (
    generate_configs,
    count_configs,
    is_valid_config,
    get_parameter_grid,
    get_alpha_types,
    get_schedule_names,
    StrategyConfig,
    PARAMETER_GRIDS_WEEKLY,
    PARAMETER_GRIDS_MONTHLY,
    PORTFOLIO_GRIDS,
    UNIVERSE_OPTIONS,
    SCHEDULE_OPTIONS,
)

from quantetf.optimization.evaluator import (
    PeriodMetrics,
    MultiPeriodResult,
    MultiPeriodEvaluator,
)

from quantetf.optimization.optimizer import (
    OptimizationResult,
    StrategyOptimizer,
)

__all__ = [
    # Grid module
    'generate_configs',
    'count_configs',
    'is_valid_config',
    'get_parameter_grid',
    'get_alpha_types',
    'get_schedule_names',
    'StrategyConfig',
    'PARAMETER_GRIDS_WEEKLY',
    'PARAMETER_GRIDS_MONTHLY',
    'PORTFOLIO_GRIDS',
    'UNIVERSE_OPTIONS',
    'SCHEDULE_OPTIONS',
    # Evaluator module
    'PeriodMetrics',
    'MultiPeriodResult',
    'MultiPeriodEvaluator',
    # Optimizer module
    'OptimizationResult',
    'StrategyOptimizer',
]
