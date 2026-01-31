"""Parameter grid generator for strategy optimization.

This module generates all valid parameter combinations for strategy optimization,
with schedule-specific parameter grids.

Key Design Insight:
- Weekly rebalancing uses shorter lookback periods (signals must be fast enough
  to match frequent trading)
- Monthly rebalancing uses longer lookback periods (signals should be stable
  for infrequent trading)

Example:
    >>> from quantetf.optimization.grid import generate_configs, count_configs
    >>> configs = generate_configs()
    >>> print(f"Generated {len(configs)} configurations")
    >>> counts = count_configs()
    >>> print(f"Weekly: {counts['weekly']['subtotal']}, Monthly: {counts['monthly']['subtotal']}")
"""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

# Weekly schedule: shorter lookbacks for faster signal generation
# 21 days ≈ 1 month, 42 days ≈ 2 months, 63 days ≈ 3 months, 126 days ≈ 6 months
PARAMETER_GRIDS_WEEKLY: Dict[str, Dict[str, List[Any]]] = {
    'momentum': {
        'lookback_days': [21, 42, 63, 126],
        'min_periods': [15, 30, 50],
    },
    'momentum_acceleration': {
        'short_lookback_days': [21, 42, 63],
        'long_lookback_days': [63, 126, 189],
        'min_periods': [30, 50],
    },
    'vol_adjusted_momentum': {
        'lookback_days': [42, 63, 126],
        'vol_floor': [0.005, 0.01, 0.02],
        'min_periods': [30, 50],
    },
    'residual_momentum': {
        'lookback_days': [63, 126],
        'min_periods': [50, 100],
    },
    # New regime-based alpha models (IMPL-006)
    'trend_filtered_momentum': {
        'momentum_lookback': [63, 126],
        'ma_period': [100, 150],
        'min_periods': [50],
    },
    'dual_momentum': {
        'lookback': [63, 126],
        'risk_free_rate': [0.0, 0.02],
        'min_periods': [50],
    },
    'value_momentum': {
        'momentum_weight': [0.3, 0.5, 0.7],
        'momentum_lookback': [63, 126],
        'min_periods': [50],
    },
}

# Monthly schedule: longer lookbacks for stable signals
# 63 days ≈ 3 months, 126 days ≈ 6 months, 189 days ≈ 9 months, 252 days ≈ 1 year
PARAMETER_GRIDS_MONTHLY: Dict[str, Dict[str, List[Any]]] = {
    'momentum': {
        'lookback_days': [63, 126, 189, 252],
        'min_periods': [50, 100, 150],
    },
    'momentum_acceleration': {
        'short_lookback_days': [21, 42, 63],
        'long_lookback_days': [126, 189, 252],
        'min_periods': [50, 100, 150],
    },
    'vol_adjusted_momentum': {
        'lookback_days': [63, 126, 189, 252],
        'vol_floor': [0.005, 0.01, 0.02],
        'min_periods': [50, 100, 150],
    },
    'residual_momentum': {
        'lookback_days': [126, 189, 252],
        'min_periods': [100, 150],
    },
    # New regime-based alpha models (IMPL-006)
    'trend_filtered_momentum': {
        'momentum_lookback': [126, 189, 252],
        'ma_period': [150, 200],
        'min_periods': [100],
    },
    'dual_momentum': {
        'lookback': [126, 189, 252],
        'risk_free_rate': [0.0, 0.02, 0.04],
        'min_periods': [100],
    },
    'value_momentum': {
        'momentum_weight': [0.3, 0.5, 0.7],
        'momentum_lookback': [126, 189, 252],
        'min_periods': [100],
    },
}

# Portfolio construction options
PORTFOLIO_GRIDS: Dict[str, List[Any]] = {
    'top_n': [3, 5, 7],
}

# Universe options - focusing on Tier 3 (100 liquid ETFs)
UNIVERSE_OPTIONS: Dict[str, str] = {
    'tier2': 'configs/universes/tier2_core_50.yaml',
    'tier3': 'configs/universes/tier3_expanded_100.yaml',
    'tier4': 'configs/universes/tier4_broad_200.yaml',
}

# Schedule options
SCHEDULE_OPTIONS: Dict[str, str] = {
    'weekly': 'configs/schedules/weekly_friday.yaml',
    'monthly': 'configs/schedules/monthly_rebalance.yaml',
}

# Cost model (fixed for optimization)
DEFAULT_COST_CONFIG: str = 'configs/costs/flat_10bps.yaml'


@dataclass
class StrategyConfig:
    """Represents a complete strategy configuration for backtesting.

    This dataclass encapsulates all parameters needed to run a backtest:
    - Alpha model type and parameters
    - Portfolio construction parameters (top_n)
    - Universe and schedule configuration paths
    - Schedule name for parameter grid selection

    Attributes:
        alpha_type: Type of alpha model ('momentum', 'momentum_acceleration',
            'vol_adjusted_momentum', 'residual_momentum')
        alpha_params: Dictionary of parameters for the alpha model
        top_n: Number of top-ranked assets to include in portfolio
        universe_path: Path to universe configuration YAML file
        universe_name: Universe name
        schedule_path: Path to schedule configuration YAML file
        schedule_name: Schedule identifier ('weekly' or 'monthly')
        cost_config_path: Path to cost model configuration (default: flat_10bps)
    """

    alpha_type: str
    alpha_params: Dict[str, Any]
    top_n: int
    universe_path: str
    universe_name: str
    schedule_path: str
    schedule_name: str
    cost_config_path: str = field(default=DEFAULT_COST_CONFIG)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for YAML serialization.

        Returns:
            Dictionary suitable for saving as YAML or passing to backtest engine.

        Example:
            >>> config = StrategyConfig(
            ...     alpha_type='momentum',
            ...     alpha_params={'lookback_days': 252, 'min_periods': 200},
            ...     top_n=5,
            ...     universe_path='configs/universes/tier3_expanded_100.yaml',
            ...     schedule_path='configs/schedules/monthly_rebalance.yaml',
            ...     schedule_name='monthly'
            ... )
            >>> d = config.to_dict()
            >>> d['name']
            'momentum_lookback_days252_min_periods200_top5_monthly'
        """
        return {
            'name': self.generate_name(),
            'universe': self.universe_path,
            'schedule': self.schedule_path,
            'cost_model': self.cost_config_path,
            'alpha_model': {
                'type': self.alpha_type,
                **self.alpha_params,
            },
            'portfolio_construction': {
                'type': 'equal_weight_top_n',
                'top_n': self.top_n,
                'constraints': {
                    'max_weight': 0.60,
                    'min_weight': 0.00,
                },
            },
        }

    def generate_name(self) -> str:
        """Generate a descriptive, unique name for this configuration.

        The name format is:
            {alpha_type}_{param1}{value1}_{param2}{value2}_...top{top_n}_{schedule}

        Returns:
            String name suitable for file naming and identification.

        Example:
            >>> config = StrategyConfig(
            ...     alpha_type='momentum_acceleration',
            ...     alpha_params={'short_lookback_days': 63, 'long_lookback_days': 252},
            ...     top_n=5,
            ...     universe_path='configs/universes/tier3_expanded_100.yaml',
            ...     schedule_path='configs/schedules/monthly_rebalance.yaml',
            ...     schedule_name='monthly'
            ... )
            >>> config.generate_name()
            'momentum_acceleration_short_lookback_days63_long_lookback_days252_top5_monthly'
        """
        # Sort params by key for consistent naming
        sorted_params = sorted(self.alpha_params.items())
        param_str = '_'.join(f"{k}{v}" for k, v in sorted_params)
        return f"{self.alpha_type}_{param_str}_top{self.top_n}_{self.universe_name}_{self.schedule_name}"

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"StrategyConfig(alpha={self.alpha_type}, "
            f"top_n={self.top_n}, schedule={self.schedule_name})"
        )


def is_valid_config(alpha_type: str, params: Dict[str, Any]) -> bool:
    """Validate parameter combinations for a given alpha type.

    This function checks that parameter combinations are valid:
    - momentum_acceleration: short_lookback_days < long_lookback_days
    - All models: min_periods < lookback_days (or long_lookback_days)

    Args:
        alpha_type: Type of alpha model
        params: Dictionary of parameters for the alpha model

    Returns:
        True if the parameter combination is valid, False otherwise.

    Example:
        >>> is_valid_config('momentum', {'lookback_days': 252, 'min_periods': 200})
        True
        >>> is_valid_config('momentum', {'lookback_days': 100, 'min_periods': 150})
        False
        >>> is_valid_config('momentum_acceleration',
        ...                 {'short_lookback_days': 126, 'long_lookback_days': 63,
        ...                  'min_periods': 50})
        False
    """
    # Validation for momentum_acceleration: short < long
    if alpha_type == 'momentum_acceleration':
        short = params.get('short_lookback_days', 0)
        long = params.get('long_lookback_days', 0)
        if short >= long:
            return False

    # Determine the relevant lookback key based on alpha type
    lookback_key = None
    if alpha_type == 'momentum_acceleration':
        lookback_key = 'long_lookback_days'
    elif alpha_type in ('trend_filtered_momentum', 'value_momentum'):
        lookback_key = 'momentum_lookback'
    elif alpha_type == 'dual_momentum':
        lookback_key = 'lookback'
    else:
        lookback_key = 'lookback_days'

    # Validate min_periods < lookback_days
    if lookback_key in params and 'min_periods' in params:
        if params['min_periods'] >= params[lookback_key]:
            return False

    return True


def get_parameter_grid(schedule_name: str) -> Dict[str, Dict[str, List[Any]]]:
    """Get the parameter grid for a given schedule type.

    Args:
        schedule_name: Either 'weekly' or 'monthly'

    Returns:
        Dictionary mapping alpha types to their parameter spaces.

    Raises:
        ValueError: If schedule_name is not 'weekly' or 'monthly'.

    Example:
        >>> grid = get_parameter_grid('weekly')
        >>> 'momentum' in grid
        True
        >>> grid['momentum']['lookback_days']
        [21, 42, 63, 126]
    """
    if schedule_name == 'weekly':
        return PARAMETER_GRIDS_WEEKLY
    elif schedule_name == 'monthly':
        return PARAMETER_GRIDS_MONTHLY
    else:
        raise ValueError(
            f"Unknown schedule: '{schedule_name}'. Must be 'weekly' or 'monthly'"
        )


def generate_configs(
    schedule_names: Optional[List[str]] = None,
    alpha_types: Optional[List[str]] = None,
    universes: Optional[Dict[str, str]] = None,
) -> List[StrategyConfig]:
    """Generate all valid strategy configurations.

    This function creates the Cartesian product of all parameter combinations,
    filtering out invalid combinations (e.g., short_lookback >= long_lookback).

    Args:
        schedule_names: List of schedules to include (default: ['weekly', 'monthly'])
        alpha_types: List of alpha types to include (default: all 4 types)
        universe_paths: List of universe configs to include (default: UNIVERSE_OPTIONS)

    Returns:
        List of StrategyConfig objects, one for each valid combination.

    Example:
        >>> configs = generate_configs()
        >>> len(configs) > 0
        True
        >>> all(isinstance(c, StrategyConfig) for c in configs)
        True
        >>> # All configs should be valid
        >>> all(is_valid_config(c.alpha_type, c.alpha_params) for c in configs)
        True
    """
    if schedule_names is None:
        schedule_names = list(SCHEDULE_OPTIONS.keys())

    if universes is None:
        universes = UNIVERSE_OPTIONS

    configs: List[StrategyConfig] = []

    for schedule_name in schedule_names:
        if schedule_name not in SCHEDULE_OPTIONS:
            raise ValueError(f"Unknown schedule: '{schedule_name}'")

        schedule_path = SCHEDULE_OPTIONS[schedule_name]
        param_grids = get_parameter_grid(schedule_name)

        # Filter alpha types if specified
        alpha_types_to_use = alpha_types if alpha_types else list(param_grids.keys())

        for alpha_type in alpha_types_to_use:
            if alpha_type not in param_grids:
                continue

            param_space = param_grids[alpha_type]
            param_keys = list(param_space.keys())
            param_values = list(param_space.values())

            # Generate all parameter combinations
            for param_combo in product(*param_values):
                params = dict(zip(param_keys, param_combo))

                # Skip invalid combinations
                if not is_valid_config(alpha_type, params):
                    continue

                # Generate configs for each top_n and universe combination
                universe_names = list(universes.keys())
                for top_n in PORTFOLIO_GRIDS['top_n']:
                    for universe_name in universe_names:
                        config = StrategyConfig(
                            alpha_type=alpha_type,
                            alpha_params=params,
                            top_n=top_n,
                            universe_path=universes[universe_name],
                            universe_name=universe_name,
                            schedule_path=schedule_path,
                            schedule_name=schedule_name,
                        )
                        configs.append(config)

    return configs


def count_configs(
    schedule_names: Optional[List[str]] = None,
    alpha_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Count configurations by schedule and alpha type.

    This function provides a breakdown of how many valid configurations exist
    for each schedule and alpha type combination, useful for planning and
    progress reporting.

    Args:
        schedule_names: List of schedules to count (default: ['weekly', 'monthly'])
        alpha_types: List of alpha types to count (default: all 4 types)

    Returns:
        Dictionary with structure:
            {
                'weekly': {'momentum': N, 'momentum_acceleration': N, ..., 'subtotal': N},
                'monthly': {'momentum': N, 'momentum_acceleration': N, ..., 'subtotal': N},
                'total': N
            }

    Example:
        >>> counts = count_configs()
        >>> counts['total'] > 0
        True
        >>> counts['weekly']['subtotal'] + counts['monthly']['subtotal'] == counts['total']
        True
    """
    if schedule_names is None:
        schedule_names = list(SCHEDULE_OPTIONS.keys())

    counts: Dict[str, Any] = {}

    for schedule_name in schedule_names:
        counts[schedule_name] = {}
        param_grids = get_parameter_grid(schedule_name)

        alpha_types_to_use = alpha_types if alpha_types else list(param_grids.keys())

        for alpha_type in alpha_types_to_use:
            if alpha_type not in param_grids:
                continue

            param_space = param_grids[alpha_type]
            param_keys = list(param_space.keys())
            param_values = list(param_space.values())

            valid_count = 0
            for param_combo in product(*param_values):
                params = dict(zip(param_keys, param_combo))
                if is_valid_config(alpha_type, params):
                    valid_count += 1

            # Multiply by top_n options and universe options
            valid_count *= len(PORTFOLIO_GRIDS['top_n'])
            valid_count *= len(UNIVERSE_OPTIONS)

            counts[schedule_name][alpha_type] = valid_count

        counts[schedule_name]['subtotal'] = sum(
            v for k, v in counts[schedule_name].items() if k != 'subtotal'
        )

    counts['total'] = sum(
        counts[sched]['subtotal']
        for sched in schedule_names
        if sched in counts
    )

    return counts


def get_alpha_types() -> List[str]:
    """Get list of all supported alpha model types.

    Returns:
        List of alpha type strings.

    Example:
        >>> types = get_alpha_types()
        >>> 'momentum' in types
        True
        >>> len(types) == 4
        True
    """
    # Use weekly grid keys (same as monthly)
    return list(PARAMETER_GRIDS_WEEKLY.keys())


def get_schedule_names() -> List[str]:
    """Get list of all supported schedule names.

    Returns:
        List of schedule name strings.

    Example:
        >>> schedules = get_schedule_names()
        >>> 'weekly' in schedules
        True
        >>> 'monthly' in schedules
        True
    """
    return list(SCHEDULE_OPTIONS.keys())
