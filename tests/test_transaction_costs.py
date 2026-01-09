"""Tests for transaction cost models."""

import pytest
import pandas as pd

from quantetf.portfolio.costs import FlatTransactionCost


class TestFlatTransactionCost:
    """Test suite for FlatTransactionCost model."""

    def test_initialization_default(self):
        """Test FlatTransactionCost initializes with default parameters."""
        cost_model = FlatTransactionCost()
        assert cost_model.cost_bps == 10.0

    def test_initialization_custom_bps(self):
        """Test FlatTransactionCost initializes with custom cost_bps."""
        cost_model = FlatTransactionCost(cost_bps=5.0)
        assert cost_model.cost_bps == 5.0

    def test_negative_cost_bps_raises_error(self):
        """Test that negative cost_bps raises ValueError."""
        with pytest.raises(ValueError, match="cost_bps must be >= 0"):
            FlatTransactionCost(cost_bps=-1.0)

    def test_zero_cost_bps(self):
        """Test that zero cost_bps is allowed (no-cost model)."""
        cost_model = FlatTransactionCost(cost_bps=0.0)
        assert cost_model.cost_bps == 0.0

    def test_no_rebalance_zero_cost(self):
        """Test that no rebalancing results in zero cost."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        prev_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        assert cost == pytest.approx(0.0)

    def test_full_rotation(self):
        """Test cost of fully rotating from one position to another."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # Sell 100% SPY, buy 100% QQQ
        # Turnover = 0.5 * (|1.0 - 0| + |0 - 1.0|) = 0.5 * 2.0 = 1.0
        # Cost = 1.0 * 0.0010 = 0.0010 (10 bps)
        prev_weights = pd.Series([1.0, 0.0], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.0, 1.0], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        assert cost == pytest.approx(0.0010)

    def test_partial_rebalance(self):
        """Test cost of partial rebalancing."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # Rebalance from 50/50 to 70/30
        # Turnover = 0.5 * (|0.7 - 0.5| + |0.3 - 0.5|) = 0.5 * 0.4 = 0.2
        # Cost = 0.2 * 0.0010 = 0.0002 (2 bps)
        prev_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.7, 0.3], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        assert cost == pytest.approx(0.0002)

    def test_add_new_position(self):
        """Test cost of adding a new position."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # Add 20% position in TLT, reduce SPY and QQQ
        # Turnover = 0.5 * (|0.4 - 0.5| + |0.4 - 0.5| + |0.2 - 0|) = 0.5 * 0.4 = 0.2
        prev_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.4, 0.4, 0.2], index=['SPY', 'QQQ', 'TLT'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        assert cost == pytest.approx(0.0002)

    def test_close_position(self):
        """Test cost of closing a position."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # Close 20% TLT position, increase SPY and QQQ
        # Turnover = 0.5 * (|0.5 - 0.4| + |0.5 - 0.4| + |0 - 0.2|) = 0.5 * 0.4 = 0.2
        prev_weights = pd.Series([0.4, 0.4, 0.2], index=['SPY', 'QQQ', 'TLT'])
        next_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        assert cost == pytest.approx(0.0002)

    def test_empty_prev_weights(self):
        """Test with empty previous weights (initial portfolio setup)."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # Start from empty portfolio
        # Turnover = 0.5 * (|0.5 - 0| + |0.5 - 0|) = 0.5 * 1.0 = 0.5
        # Cost = 0.5 * 0.0010 = 0.0005 (5 bps)
        prev_weights = pd.Series(dtype=float)
        next_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        assert cost == pytest.approx(0.0005)

    def test_empty_next_weights(self):
        """Test with empty next weights (liquidating entire portfolio)."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # Liquidate entire portfolio
        # Turnover = 0.5 * (|0 - 0.5| + |0 - 0.5|) = 0.5 * 1.0 = 0.5
        # Cost = 0.5 * 0.0010 = 0.0005 (5 bps)
        prev_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        next_weights = pd.Series(dtype=float)

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        assert cost == pytest.approx(0.0005)

    def test_both_empty(self):
        """Test with both empty (no portfolio, no trades)."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        prev_weights = pd.Series(dtype=float)
        next_weights = pd.Series(dtype=float)

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        assert cost == 0.0

    def test_none_prev_weights(self):
        """Test with None previous weights."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        next_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=None,
            next_weights=next_weights
        )

        assert cost == pytest.approx(0.0005)

    def test_none_next_weights(self):
        """Test with None next weights."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        prev_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=None
        )

        assert cost == pytest.approx(0.0005)

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # NaN should be treated as 0.0
        prev_weights = pd.Series([0.5, float('nan')], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.3, 0.7], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        # Turnover = 0.5 * (|0.3 - 0.5| + |0.7 - 0|) = 0.5 * 0.9 = 0.45
        # Cost = 0.45 * 0.0010 = 0.00045
        assert cost == pytest.approx(0.00045)

    def test_misaligned_tickers(self):
        """Test with non-overlapping ticker sets."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # Different tickers entirely
        prev_weights = pd.Series([1.0], index=['SPY'])
        next_weights = pd.Series([1.0], index=['QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        # Turnover = 0.5 * (|0 - 1.0| + |1.0 - 0|) = 0.5 * 2.0 = 1.0
        # Cost = 1.0 * 0.0010 = 0.0010 (10 bps)
        assert cost == pytest.approx(0.0010)

    def test_multi_asset_complex(self):
        """Test complex multi-asset rebalancing scenario."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # 5-asset portfolio rebalancing
        prev_weights = pd.Series(
            [0.30, 0.25, 0.20, 0.15, 0.10],
            index=['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ']
        )
        next_weights = pd.Series(
            [0.35, 0.20, 0.20, 0.15, 0.10],
            index=['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ']
        )

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        # Turnover = 0.5 * (|0.05| + |-0.05| + |0| + |0| + |0|) = 0.5 * 0.10 = 0.05
        # Cost = 0.05 * 0.0010 = 0.00005 (0.5 bps)
        assert cost == pytest.approx(0.00005)

    def test_custom_cost_bps(self):
        """Test with custom cost_bps values."""
        # Test with 5 bps cost
        cost_model_5bps = FlatTransactionCost(cost_bps=5.0)

        prev_weights = pd.Series([1.0, 0.0], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.0, 1.0], index=['SPY', 'QQQ'])

        cost = cost_model_5bps.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        # Turnover = 1.0, Cost = 1.0 * 0.0005 = 0.0005 (5 bps)
        assert cost == pytest.approx(0.0005)

        # Test with 20 bps cost
        cost_model_20bps = FlatTransactionCost(cost_bps=20.0)

        cost = cost_model_20bps.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        # Turnover = 1.0, Cost = 1.0 * 0.0020 = 0.0020 (20 bps)
        assert cost == pytest.approx(0.0020)

    def test_prices_parameter_ignored(self):
        """Test that prices parameter is ignored (interface compatibility)."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        prev_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.7, 0.3], index=['SPY', 'QQQ'])
        prices = pd.Series([400.0, 350.0], index=['SPY', 'QQQ'])

        # With prices
        cost_with_prices = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights,
            prices=prices
        )

        # Without prices
        cost_without_prices = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        # Should be identical
        assert cost_with_prices == cost_without_prices

    def test_weights_sum_not_normalized(self):
        """Test behavior when weights don't sum to 1.0."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # Weights sum to 0.8 (partially invested)
        prev_weights = pd.Series([0.4, 0.4], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        # Turnover = 0.5 * (|0.5 - 0.4| + |0.5 - 0.4|) = 0.5 * 0.2 = 0.1
        # Cost = 0.1 * 0.0010 = 0.0001
        assert cost == pytest.approx(0.0001)

    def test_very_small_rebalance(self):
        """Test with very small rebalancing amounts."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        # Tiny adjustment (0.01%)
        prev_weights = pd.Series([0.5000, 0.5000], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.5001, 0.4999], index=['SPY', 'QQQ'])

        cost = cost_model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        # Turnover = 0.5 * 0.0002 = 0.0001
        # Cost = 0.0001 * 0.0010 = 0.0000001
        assert cost == pytest.approx(0.0000001, abs=1e-10)

    def test_dataclass_frozen(self):
        """Test that FlatTransactionCost is immutable (frozen dataclass)."""
        cost_model = FlatTransactionCost(cost_bps=10.0)

        with pytest.raises(Exception):  # FrozenInstanceError
            cost_model.cost_bps = 20.0
