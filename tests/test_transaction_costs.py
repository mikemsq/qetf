"""Tests for transaction cost models."""

import pytest
import pandas as pd
import numpy as np

from quantetf.portfolio.costs import (
    FlatTransactionCost,
    SlippageCostModel,
    SpreadCostModel,
    ImpactCostModel,
)


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


class TestSlippageCostModel:
    """Test suite for SlippageCostModel."""

    def test_initialization_default(self):
        """Test Slipp ageCostModel initializes with default parameters."""
        model = SlippageCostModel()
        assert model.base_spread_bps == 5.0
        assert model.impact_coefficient == 2.0

    def test_initialization_custom(self):
        """Test SlippageCostModel with custom parameters."""
        model = SlippageCostModel(base_spread_bps=10.0, impact_coefficient=3.0)
        assert model.base_spread_bps == 10.0
        assert model.impact_coefficient == 3.0

    def test_no_trade_zero_cost(self):
        """Test that no rebalancing results in zero cost."""
        model = SlippageCostModel()
        prev_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])

        cost = model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )
        assert cost == 0.0

    def test_small_trade_low_cost(self):
        """Test that small trades have low impact."""
        model = SlippageCostModel(base_spread_bps=5.0, impact_coefficient=2.0)
        prev_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.51, 0.49], index=['SPY', 'QQQ'])

        cost = model.estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )
        # Small 1% change should have cost close to base_spread
        assert cost > 0
        assert cost < 0.001  # Less than 10 bps

    def test_large_trade_high_cost(self):
        """Test that larger trades incur higher costs."""
        model = SlippageCostModel(base_spread_bps=5.0, impact_coefficient=2.0)

        # Small trade
        small_cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.5, 0.5], index=['SPY', 'QQQ']),
            next_weights=pd.Series([0.55, 0.45], index=['SPY', 'QQQ'])
        )

        # Large trade
        large_cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.5, 0.5], index=['SPY', 'QQQ']),
            next_weights=pd.Series([0.8, 0.2], index=['SPY', 'QQQ'])
        )

        # Larger trade should cost more
        assert large_cost > small_cost

    def test_impact_increases_with_size(self):
        """Verify that impact scales with trade size."""
        model = SlippageCostModel(base_spread_bps=0.0, impact_coefficient=1.0)

        # 10% position change
        cost_10pct = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.0], index=['SPY']),
            next_weights=pd.Series([0.1], index=['SPY'])
        )

        # 20% position change (2x larger)
        cost_20pct = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.0], index=['SPY']),
            next_weights=pd.Series([0.2], index=['SPY'])
        )

        # Cost should scale roughly linearly with size (when base_spread = 0)
        assert cost_20pct > cost_10pct

    def test_empty_weights(self):
        """Test handling of empty weight series."""
        model = SlippageCostModel()
        cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series(dtype=float),
            next_weights=pd.Series(dtype=float)
        )
        assert cost == 0.0


class TestSpreadCostModel:
    """Test suite for SpreadCostModel."""

    def test_initialization_default(self):
        """Test SpreadCostModel initializes with default spread map."""
        model = SpreadCostModel()
        assert model.spread_map is not None
        assert 'SPY' in model.spread_map
        assert model.spread_map['SPY'] == 1.0

    def test_initialization_custom_spread_map(self):
        """Test SpreadCostModel with custom spread map."""
        custom_spreads = {'SPY': 0.5, 'QQQ': 1.5}
        model = SpreadCostModel(spread_map=custom_spreads)
        assert model.spread_map['SPY'] == 0.5
        assert model.spread_map['QQQ'] == 1.5

    def test_liquid_etf_low_cost(self):
        """Test that liquid ETFs (SPY) have low spread cost."""
        model = SpreadCostModel()

        # Full rotation from SPY to QQQ (both very liquid)
        cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series([1.0], index=['SPY']),
            next_weights=pd.Series([1.0], index=['QQQ'])
        )

        # Both SPY and QQQ have 1 bp spread, so average should be 1 bp
        assert cost == pytest.approx(0.0001, abs=1e-6)

    def test_illiquid_etf_high_cost(self):
        """Test that illiquid ETFs have higher costs."""
        model = SpreadCostModel()

        # Trade into ARKK (higher spread)
        cost_arkk = model.estimate_rebalance_cost(
            prev_weights=pd.Series([1.0], index=['SPY']),
            next_weights=pd.Series([1.0], index=['ARKK'])
        )

        # Trade into QQQ (lower spread)
        cost_qqq = model.estimate_rebalance_cost(
            prev_weights=pd.Series([1.0], index=['SPY']),
            next_weights=pd.Series([1.0], index=['QQQ'])
        )

        # ARKK trade should cost more
        assert cost_arkk > cost_qqq

    def test_unknown_ticker_uses_default(self):
        """Test that unknown tickers use default spread."""
        model = SpreadCostModel(default_spread_bps=20.0)

        cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series([1.0], index=['UNKNOWN']),
            next_weights=pd.Series([1.0], index=['ALSO_UNKNOWN'])
        )

        # Should use default spread of 20 bps
        assert cost == pytest.approx(0.002, abs=1e-6)

    def test_no_trade_zero_cost(self):
        """Test no rebalancing gives zero cost."""
        model = SpreadCostModel()
        cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.5, 0.5], index=['SPY', 'QQQ']),
            next_weights=pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        )
        assert cost == 0.0


class TestImpactCostModel:
    """Test suite for ImpactCostModel."""

    def test_initialization_default(self):
        """Test ImpactCostModel initializes with default coefficient."""
        model = ImpactCostModel()
        assert model.impact_coefficient == 5.0

    def test_initialization_custom(self):
        """Test ImpactCostModel with custom coefficient."""
        model = ImpactCostModel(impact_coefficient=10.0)
        assert model.impact_coefficient == 10.0

    def test_square_root_scaling(self):
        """Verify square-root relationship: 4x trade → 2x cost."""
        model = ImpactCostModel(impact_coefficient=10.0)

        # 1% trade
        cost_1pct = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.0], index=['SPY']),
            next_weights=pd.Series([0.01], index=['SPY'])
        )

        # 4% trade (4x larger)
        cost_4pct = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.0], index=['SPY']),
            next_weights=pd.Series([0.04], index=['SPY'])
        )

        # Should be approximately 2x (sqrt(4) = 2)
        ratio = cost_4pct / cost_1pct
        assert 1.9 < ratio < 2.1  # Allow some tolerance

    def test_large_trade_higher_relative_cost(self):
        """Test that larger trades have disproportionately higher cost."""
        model = ImpactCostModel(impact_coefficient=5.0)

        small_cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.0], index=['SPY']),
            next_weights=pd.Series([0.05], index=['SPY'])
        )

        large_cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.0], index=['SPY']),
            next_weights=pd.Series([0.25], index=['SPY'])
        )

        # 5x larger trade should cost more than 5x (due to sqrt scaling)
        assert large_cost > small_cost
        # But less than linear (5x)
        assert large_cost < 5 * small_cost

    def test_no_trade_zero_cost(self):
        """Test no rebalancing gives zero cost."""
        model = ImpactCostModel()
        cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series([0.5, 0.5], index=['SPY', 'QQQ']),
            next_weights=pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        )
        assert cost == 0.0

    def test_empty_weights(self):
        """Test handling of empty weight series."""
        model = ImpactCostModel()
        cost = model.estimate_rebalance_cost(
            prev_weights=pd.Series(dtype=float),
            next_weights=pd.Series(dtype=float)
        )
        assert cost == 0.0


class TestCostModelComparison:
    """Integration tests comparing different cost models."""

    def test_all_models_on_same_trade(self):
        """Test all cost models on the same trade for comparison."""
        prev_weights = pd.Series([0.5, 0.5], index=['SPY', 'QQQ'])
        next_weights = pd.Series([0.7, 0.3], index=['SPY', 'QQQ'])

        flat_cost = FlatTransactionCost(cost_bps=10.0).estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        slippage_cost = SlippageCostModel(base_spread_bps=5.0, impact_coefficient=2.0).estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        spread_cost = SpreadCostModel().estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        impact_cost = ImpactCostModel(impact_coefficient=5.0).estimate_rebalance_cost(
            prev_weights=prev_weights,
            next_weights=next_weights
        )

        # All should return valid costs
        assert flat_cost > 0
        assert slippage_cost > 0
        assert spread_cost > 0
        assert impact_cost > 0

        # Spread cost should be lowest (SPY/QQQ are very liquid)
        assert spread_cost < flat_cost
