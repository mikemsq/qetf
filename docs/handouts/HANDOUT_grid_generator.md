# Task 1: Parameter Grid Generator

## File to Create
`src/quantetf/optimization/grid.py`

## Purpose
Generate all valid parameter combinations for strategy optimization, with schedule-specific parameter grids.

## Key Insight
Weekly rebalancing needs shorter lookback periods (signals must be fast enough to match frequent trading).
Monthly rebalancing needs longer lookback periods (signals should be stable for infrequent trading).

## Implementation

```python
"""Parameter grid generator for strategy optimization."""
from itertools import product
from typing import Dict, List, Any
from dataclasses import dataclass

# Weekly schedule: shorter lookbacks for faster signal generation
PARAMETER_GRIDS_WEEKLY = {
    'momentum': {
        'lookback_days': [21, 42, 63, 126],
        'min_periods': [15, 30, 50]
    },
    'momentum_acceleration': {
        'short_lookback_days': [10, 21, 42],
        'long_lookback_days': [42, 63, 126],
        'min_periods': [30, 50]
    },
    'vol_adjusted_momentum': {
        'lookback_days': [42, 63, 126],
        'vol_floor': [0.005, 0.01, 0.02],
        'min_periods': [30, 50]
    },
    'residual_momentum': {
        'lookback_days': [63, 126],
        'min_periods': [50, 100]
    }
}

# Monthly schedule: longer lookbacks for stable signals
PARAMETER_GRIDS_MONTHLY = {
    'momentum': {
        'lookback_days': [63, 126, 189, 252],
        'min_periods': [50, 100, 150]
    },
    'momentum_acceleration': {
        'short_lookback_days': [21, 42, 63],
        'long_lookback_days': [126, 189, 252],
        'min_periods': [50, 100, 150]
    },
    'vol_adjusted_momentum': {
        'lookback_days': [63, 126, 189, 252],
        'vol_floor': [0.005, 0.01, 0.02],
        'min_periods': [50, 100, 150]
    },
    'residual_momentum': {
        'lookback_days': [126, 189, 252],
        'min_periods': [100, 150]
    }
}

PORTFOLIO_GRIDS = {
    'top_n': [3, 5, 7]
}

UNIVERSE_OPTIONS = [
    'configs/universes/tier3_expanded_100.yaml'
]

SCHEDULE_OPTIONS = {
    'weekly': 'configs/schedules/weekly_rebalance.yaml',
    'monthly': 'configs/schedules/monthly_rebalance.yaml'
}


@dataclass
class StrategyConfig:
    """Represents a complete strategy configuration."""
    alpha_type: str
    alpha_params: Dict[str, Any]
    top_n: int
    universe_path: str
    schedule_path: str
    schedule_name: str  # 'weekly' or 'monthly'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'name': self.generate_name(),
            'universe': self.universe_path,
            'schedule': self.schedule_path,
            'cost_model': 'configs/costs/flat_10bps.yaml',
            'alpha_model': {
                'type': self.alpha_type,
                **self.alpha_params
            },
            'portfolio_construction': {
                'type': 'equal_weight_top_n',
                'top_n': self.top_n,
                'constraints': {
                    'max_weight': 0.60,
                    'min_weight': 0.00
                }
            }
        }

    def generate_name(self) -> str:
        """Generate descriptive config name."""
        param_str = '_'.join(f"{k}{v}" for k, v in self.alpha_params.items())
        return f"{self.alpha_type}_{param_str}_top{self.top_n}_{self.schedule_name}"


def is_valid_config(alpha_type: str, params: Dict[str, Any]) -> bool:
    """Validate parameter combinations."""
    if alpha_type == 'momentum_acceleration':
        # Short lookback must be less than long lookback
        if params['short_lookback_days'] >= params['long_lookback_days']:
            return False

    # min_periods should be reasonable relative to lookback
    lookback_key = 'lookback_days'
    if alpha_type == 'momentum_acceleration':
        lookback_key = 'long_lookback_days'

    if lookback_key in params and 'min_periods' in params:
        if params['min_periods'] >= params[lookback_key]:
            return False

    return True


def generate_configs() -> List[StrategyConfig]:
    """Generate all valid strategy configurations."""
    configs = []

    for schedule_name, schedule_path in SCHEDULE_OPTIONS.items():
        # Select appropriate parameter grid for schedule
        param_grids = (PARAMETER_GRIDS_WEEKLY
                      if schedule_name == 'weekly'
                      else PARAMETER_GRIDS_MONTHLY)

        for alpha_type, param_space in param_grids.items():
            param_keys = list(param_space.keys())
            param_values = list(param_space.values())

            for param_combo in product(*param_values):
                params = dict(zip(param_keys, param_combo))

                # Skip invalid combinations
                if not is_valid_config(alpha_type, params):
                    continue

                for top_n in PORTFOLIO_GRIDS['top_n']:
                    for universe in UNIVERSE_OPTIONS:
                        config = StrategyConfig(
                            alpha_type=alpha_type,
                            alpha_params=params,
                            top_n=top_n,
                            universe_path=universe,
                            schedule_path=schedule_path,
                            schedule_name=schedule_name
                        )
                        configs.append(config)

    return configs


def count_configs() -> Dict[str, int]:
    """Count configurations by schedule and alpha type."""
    counts = {'weekly': {}, 'monthly': {}, 'total': 0}

    for schedule_name in ['weekly', 'monthly']:
        param_grids = (PARAMETER_GRIDS_WEEKLY
                      if schedule_name == 'weekly'
                      else PARAMETER_GRIDS_MONTHLY)

        for alpha_type, param_space in param_grids.items():
            valid_count = 0
            param_keys = list(param_space.keys())
            param_values = list(param_space.values())

            for param_combo in product(*param_values):
                params = dict(zip(param_keys, param_combo))
                if is_valid_config(alpha_type, params):
                    valid_count += 1

            # Multiply by top_n options
            valid_count *= len(PORTFOLIO_GRIDS['top_n'])
            counts[schedule_name][alpha_type] = valid_count

        counts[schedule_name]['subtotal'] = sum(counts[schedule_name].values())

    counts['total'] = counts['weekly']['subtotal'] + counts['monthly']['subtotal']
    return counts
```

## Testing

```python
def test_grid_generator():
    configs = generate_configs()
    counts = count_configs()

    # Verify count matches
    assert len(configs) == counts['total']

    # Verify all configs are valid
    for config in configs:
        assert is_valid_config(config.alpha_type, config.alpha_params)

    # Verify momentum_acceleration constraint
    for config in configs:
        if config.alpha_type == 'momentum_acceleration':
            assert config.alpha_params['short_lookback_days'] < config.alpha_params['long_lookback_days']

    print(f"Generated {len(configs)} valid configurations")
    print(f"Weekly: {counts['weekly']['subtotal']}")
    print(f"Monthly: {counts['monthly']['subtotal']}")
```

## Expected Output

Running `count_configs()` should return approximately:
- Weekly: ~138 configurations
- Monthly: ~216 configurations
- Total: ~354 configurations

## Dependencies

None - this module is pure Python configuration generation.

## Notes

- All lookback values are in trading days (not calendar days)
- 21 days ≈ 1 month, 63 days ≈ 3 months, 126 days ≈ 6 months, 252 days ≈ 1 year
- vol_floor is the minimum volatility floor for vol_adjusted_momentum
