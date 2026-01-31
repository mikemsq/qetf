"""Tests for the parameter grid generator module.

This module tests the grid.py functionality for generating valid
strategy configurations for optimization.
"""

import pytest
from quantetf.optimization.grid import (
    StrategyConfig,
    generate_configs,
    count_configs,
    is_valid_config,
    get_parameter_grid,
    get_alpha_types,
    get_schedule_names,
    PARAMETER_GRIDS_WEEKLY,
    PARAMETER_GRIDS_MONTHLY,
    PORTFOLIO_GRIDS,
    UNIVERSE_OPTIONS,
    SCHEDULE_OPTIONS,
    DEFAULT_COST_CONFIG,
)


class TestParameterGrids:
    """Tests for parameter grid definitions."""

    def test_weekly_grids_have_all_alpha_types(self):
        """Weekly grids should have all 7 alpha types."""
        expected_types = {'momentum', 'momentum_acceleration',
                         'vol_adjusted_momentum', 'residual_momentum',
                         'trend_filtered_momentum', 'dual_momentum', 'value_momentum'}
        assert set(PARAMETER_GRIDS_WEEKLY.keys()) == expected_types

    def test_monthly_grids_have_all_alpha_types(self):
        """Monthly grids should have all 7 alpha types."""
        expected_types = {'momentum', 'momentum_acceleration',
                         'vol_adjusted_momentum', 'residual_momentum',
                         'trend_filtered_momentum', 'dual_momentum', 'value_momentum'}
        assert set(PARAMETER_GRIDS_MONTHLY.keys()) == expected_types

    def test_weekly_lookbacks_are_shorter(self):
        """Weekly momentum lookbacks should be shorter than monthly."""
        weekly_max = max(PARAMETER_GRIDS_WEEKLY['momentum']['lookback_days'])
        monthly_max = max(PARAMETER_GRIDS_MONTHLY['momentum']['lookback_days'])
        assert weekly_max < monthly_max

    def test_monthly_lookbacks_include_252(self):
        """Monthly momentum should include 1-year lookback."""
        assert 252 in PARAMETER_GRIDS_MONTHLY['momentum']['lookback_days']

    def test_momentum_acceleration_has_required_params(self):
        """Momentum acceleration should have short and long lookback params."""
        for grid in [PARAMETER_GRIDS_WEEKLY, PARAMETER_GRIDS_MONTHLY]:
            params = grid['momentum_acceleration']
            assert 'short_lookback_days' in params
            assert 'long_lookback_days' in params
            assert 'min_periods' in params

    def test_vol_adjusted_has_vol_floor(self):
        """Vol adjusted momentum should have vol_floor parameter."""
        for grid in [PARAMETER_GRIDS_WEEKLY, PARAMETER_GRIDS_MONTHLY]:
            params = grid['vol_adjusted_momentum']
            assert 'vol_floor' in params
            assert all(v > 0 for v in params['vol_floor'])


class TestIsValidConfig:
    """Tests for the is_valid_config validation function."""

    def test_valid_momentum_config(self):
        """Valid momentum config should return True."""
        params = {'lookback_days': 252, 'min_periods': 200}
        assert is_valid_config('momentum', params) is True

    def test_invalid_momentum_min_periods_too_high(self):
        """Momentum with min_periods >= lookback should be invalid."""
        params = {'lookback_days': 100, 'min_periods': 150}
        assert is_valid_config('momentum', params) is False

    def test_invalid_momentum_min_periods_equal(self):
        """Momentum with min_periods == lookback should be invalid."""
        params = {'lookback_days': 100, 'min_periods': 100}
        assert is_valid_config('momentum', params) is False

    def test_valid_momentum_acceleration(self):
        """Valid momentum acceleration config should return True."""
        params = {
            'short_lookback_days': 63,
            'long_lookback_days': 252,
            'min_periods': 100,
        }
        assert is_valid_config('momentum_acceleration', params) is True

    def test_invalid_momentum_acceleration_short_ge_long(self):
        """Momentum acceleration with short >= long should be invalid."""
        params = {
            'short_lookback_days': 252,
            'long_lookback_days': 126,
            'min_periods': 100,
        }
        assert is_valid_config('momentum_acceleration', params) is False

    def test_invalid_momentum_acceleration_short_equal_long(self):
        """Momentum acceleration with short == long should be invalid."""
        params = {
            'short_lookback_days': 126,
            'long_lookback_days': 126,
            'min_periods': 100,
        }
        assert is_valid_config('momentum_acceleration', params) is False

    def test_invalid_momentum_acceleration_min_periods_too_high(self):
        """Momentum acceleration with min_periods >= long_lookback invalid."""
        params = {
            'short_lookback_days': 63,
            'long_lookback_days': 126,
            'min_periods': 200,  # > long_lookback
        }
        assert is_valid_config('momentum_acceleration', params) is False

    def test_valid_vol_adjusted_momentum(self):
        """Valid vol adjusted momentum config should return True."""
        params = {
            'lookback_days': 252,
            'min_periods': 200,
            'vol_floor': 0.01,
        }
        assert is_valid_config('vol_adjusted_momentum', params) is True

    def test_valid_residual_momentum(self):
        """Valid residual momentum config should return True."""
        params = {'lookback_days': 252, 'min_periods': 200}
        assert is_valid_config('residual_momentum', params) is True


class TestStrategyConfig:
    """Tests for the StrategyConfig dataclass."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample strategy config for testing."""
        return StrategyConfig(
            alpha_type='momentum',
            alpha_params={'lookback_days': 252, 'min_periods': 200},
            top_n=5,
            universe_path='configs/universes/tier3_expanded_100.yaml',
            universe_name='tier3',
            schedule_path='configs/schedules/monthly_rebalance.yaml',
            schedule_name='monthly',
        )

    def test_generate_name(self, sample_config):
        """Generate name should produce consistent, unique names."""
        name = sample_config.generate_name()
        assert 'momentum' in name
        assert 'top5' in name
        assert 'monthly' in name
        assert 'lookback_days252' in name

    def test_generate_name_is_deterministic(self, sample_config):
        """Same config should generate same name."""
        name1 = sample_config.generate_name()
        name2 = sample_config.generate_name()
        assert name1 == name2

    def test_generate_name_sorts_params(self):
        """Params should be sorted for consistent naming."""
        config1 = StrategyConfig(
            alpha_type='momentum_acceleration',
            alpha_params={'long_lookback_days': 252, 'short_lookback_days': 63,
                         'min_periods': 100},
            top_n=5,
            universe_path='configs/universes/tier3_expanded_100.yaml',
            universe_name='tier3',
            schedule_path='configs/schedules/monthly_rebalance.yaml',
            schedule_name='monthly',
        )
        config2 = StrategyConfig(
            alpha_type='momentum_acceleration',
            alpha_params={'short_lookback_days': 63, 'min_periods': 100,
                         'long_lookback_days': 252},
            top_n=5,
            universe_path='configs/universes/tier3_expanded_100.yaml',
            universe_name='tier3',
            schedule_path='configs/schedules/monthly_rebalance.yaml',
            schedule_name='monthly',
        )
        # Names should be same regardless of dict order
        assert config1.generate_name() == config2.generate_name()

    def test_to_dict_structure(self, sample_config):
        """to_dict should return properly structured dict."""
        d = sample_config.to_dict()

        assert 'name' in d
        assert 'universe' in d
        assert 'schedule' in d
        assert 'cost_model' in d
        assert 'alpha_model' in d
        assert 'portfolio_construction' in d

    def test_to_dict_alpha_model(self, sample_config):
        """Alpha model section should have type and params."""
        d = sample_config.to_dict()
        alpha = d['alpha_model']

        assert alpha['type'] == 'momentum'
        assert alpha['lookback_days'] == 252
        assert alpha['min_periods'] == 200

    def test_to_dict_portfolio_construction(self, sample_config):
        """Portfolio construction should have top_n and constraints."""
        d = sample_config.to_dict()
        pc = d['portfolio_construction']

        assert pc['type'] == 'equal_weight_top_n'
        assert pc['top_n'] == 5
        assert 'constraints' in pc
        assert pc['constraints']['max_weight'] == 0.60

    def test_default_cost_config(self, sample_config):
        """Default cost config should be flat_10bps."""
        assert sample_config.cost_config_path == DEFAULT_COST_CONFIG

    def test_repr(self, sample_config):
        """repr should be concise and informative."""
        r = repr(sample_config)
        assert 'momentum' in r
        assert 'top_n=5' in r
        assert 'monthly' in r


class TestGenerateConfigs:
    """Tests for the generate_configs function."""

    def test_generates_non_empty_list(self):
        """Should generate at least one configuration."""
        configs = generate_configs()
        assert len(configs) > 0

    def test_all_configs_are_strategy_config(self):
        """All items should be StrategyConfig instances."""
        configs = generate_configs()
        assert all(isinstance(c, StrategyConfig) for c in configs)

    def test_all_configs_are_valid(self):
        """All generated configs should pass validation."""
        configs = generate_configs()
        for config in configs:
            assert is_valid_config(config.alpha_type, config.alpha_params), \
                f"Invalid config: {config}"

    def test_momentum_acceleration_constraint(self):
        """All momentum_acceleration configs should have short < long."""
        configs = generate_configs()
        for config in configs:
            if config.alpha_type == 'momentum_acceleration':
                short = config.alpha_params['short_lookback_days']
                long = config.alpha_params['long_lookback_days']
                assert short < long, f"Invalid: short={short}, long={long}"

    def test_filter_by_schedule(self):
        """Should be able to filter by schedule."""
        weekly_configs = generate_configs(schedule_names=['weekly'])
        monthly_configs = generate_configs(schedule_names=['monthly'])

        assert all(c.schedule_name == 'weekly' for c in weekly_configs)
        assert all(c.schedule_name == 'monthly' for c in monthly_configs)
        assert len(weekly_configs) > 0
        assert len(monthly_configs) > 0

    def test_filter_by_alpha_type(self):
        """Should be able to filter by alpha type."""
        momentum_configs = generate_configs(alpha_types=['momentum'])

        assert all(c.alpha_type == 'momentum' for c in momentum_configs)
        assert len(momentum_configs) > 0

    def test_all_top_n_values_present(self):
        """All top_n values should be represented."""
        configs = generate_configs()
        top_n_values = {c.top_n for c in configs}

        for expected_n in PORTFOLIO_GRIDS['top_n']:
            assert expected_n in top_n_values

    def test_invalid_schedule_raises(self):
        """Invalid schedule name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown schedule"):
            generate_configs(schedule_names=['invalid'])

    def test_configs_have_unique_names(self):
        """All configs should have unique generated names."""
        configs = generate_configs()
        names = [c.generate_name() for c in configs]
        assert len(names) == len(set(names)), "Duplicate names found"


class TestCountConfigs:
    """Tests for the count_configs function."""

    def test_count_matches_generated(self):
        """Count should match actual number of generated configs."""
        configs = generate_configs()
        counts = count_configs()

        assert len(configs) == counts['total']

    def test_subtotals_sum_to_total(self):
        """Subtotals should sum to total."""
        counts = count_configs()

        subtotal_sum = counts['weekly']['subtotal'] + counts['monthly']['subtotal']
        assert subtotal_sum == counts['total']

    def test_has_all_schedule_keys(self):
        """Should have keys for all schedules plus total."""
        counts = count_configs()

        assert 'weekly' in counts
        assert 'monthly' in counts
        assert 'total' in counts

    def test_weekly_has_all_alpha_types(self):
        """Weekly counts should have all alpha types."""
        counts = count_configs()

        for alpha_type in get_alpha_types():
            assert alpha_type in counts['weekly']

    def test_filter_by_alpha_type(self):
        """Should be able to count only specific alpha types."""
        momentum_counts = count_configs(alpha_types=['momentum'])

        # Only momentum should have counts
        for schedule in ['weekly', 'monthly']:
            for alpha in ['momentum_acceleration', 'vol_adjusted_momentum',
                         'residual_momentum']:
                assert alpha not in momentum_counts[schedule]

    def test_reasonable_total_count(self):
        """Total should be in expected range (with 7 alpha types and 3 universes)."""
        counts = count_configs()

        # With 7 alpha types and 3 universe tiers, expect larger counts
        assert 500 < counts['total'] < 2000


class TestGetParameterGrid:
    """Tests for the get_parameter_grid helper."""

    def test_weekly_returns_weekly_grid(self):
        """'weekly' should return weekly parameter grid."""
        grid = get_parameter_grid('weekly')
        assert grid is PARAMETER_GRIDS_WEEKLY

    def test_monthly_returns_monthly_grid(self):
        """'monthly' should return monthly parameter grid."""
        grid = get_parameter_grid('monthly')
        assert grid is PARAMETER_GRIDS_MONTHLY

    def test_invalid_schedule_raises(self):
        """Invalid schedule should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown schedule"):
            get_parameter_grid('quarterly')


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_alpha_types_returns_all(self):
        """get_alpha_types should return all 7 types."""
        types = get_alpha_types()
        assert len(types) == 7
        assert 'momentum' in types
        assert 'momentum_acceleration' in types
        assert 'vol_adjusted_momentum' in types
        assert 'residual_momentum' in types
        assert 'trend_filtered_momentum' in types
        assert 'dual_momentum' in types
        assert 'value_momentum' in types

    def test_get_schedule_names_returns_both(self):
        """get_schedule_names should return weekly and monthly."""
        names = get_schedule_names()
        assert 'weekly' in names
        assert 'monthly' in names
        assert len(names) == 2


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_universe_options_not_empty(self):
        """UNIVERSE_OPTIONS should not be empty."""
        assert len(UNIVERSE_OPTIONS) > 0

    def test_schedule_options_has_weekly_and_monthly(self):
        """SCHEDULE_OPTIONS should have weekly and monthly."""
        assert 'weekly' in SCHEDULE_OPTIONS
        assert 'monthly' in SCHEDULE_OPTIONS

    def test_portfolio_grids_has_top_n(self):
        """PORTFOLIO_GRIDS should have top_n."""
        assert 'top_n' in PORTFOLIO_GRIDS
        assert len(PORTFOLIO_GRIDS['top_n']) > 0

    def test_default_cost_config_exists(self):
        """DEFAULT_COST_CONFIG should be set."""
        assert DEFAULT_COST_CONFIG is not None
        assert 'flat_10bps' in DEFAULT_COST_CONFIG
