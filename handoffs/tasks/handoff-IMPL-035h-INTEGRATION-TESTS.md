# Task Handoff: IMPL-035h - Integration Tests

**Task ID:** IMPL-035h
**Parent Task:** IMPL-035 (Regime-Based Strategy Selection System)
**Status:** ready
**Priority:** HIGH
**Type:** Testing
**Estimated Effort:** 2-3 hours
**Dependencies:** All other IMPL-035 subtasks

---

## Summary

Create comprehensive integration tests that verify the entire regime-based strategy selection system works end-to-end. These tests ensure all components work together correctly.

---

## Deliverables

1. **`tests/integration/test_regime_system.py`** - End-to-end integration tests
2. **`tests/integration/conftest.py`** - Shared fixtures for integration tests
3. **Test data fixtures** if needed

---

## Technical Specification

### Integration Test Scenarios

#### Scenario 1: Full Optimization → Regime Analysis Flow

```python
# tests/integration/test_regime_system.py
"""Integration tests for regime-based strategy selection system."""

import pytest
import pandas as pd
from pathlib import Path
import yaml
import json

from quantetf.optimization.optimizer import StrategyOptimizer
from quantetf.regime.detector import RegimeDetector
from quantetf.regime.analyzer import RegimeAnalyzer
from quantetf.regime.indicators import RegimeIndicators
from quantetf.production.regime_monitor import DailyRegimeMonitor
from quantetf.production.rebalancer import RegimeAwareRebalancer
from quantetf.data.access import DataAccessFactory


class TestOptimizationToRegimeMapping:
    """Test optimization produces valid regime mappings."""

    @pytest.fixture
    def data_access(self):
        """Create data access context."""
        return DataAccessFactory.create_context(
            config={"snapshot_path": "data/snapshots/snapshot_20260115_*/data.parquet"},
            enable_caching=True,
        )

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "optimization"

    def test_optimizer_produces_regime_mapping(self, data_access, output_dir):
        """
        Full optimization run should produce regime_mapping.yaml.

        This tests the flow:
        1. Run optimizer with regime analysis enabled
        2. Verify finalists.yaml created
        3. Verify regime_mapping.yaml created with all 4 regimes
        4. Verify regime_analysis.csv has performance metrics
        """
        optimizer = StrategyOptimizer(
            data_access=data_access,
            regime_analysis_enabled=True,
            num_finalists=3,
        )

        # Run small optimization
        result = optimizer.run(
            max_configs=10,
            periods=[1],  # 1-year only for speed
        )

        # Save outputs
        optimizer.save_outputs(result, output_dir)

        # Verify files created
        assert (output_dir / "finalists.yaml").exists()
        assert (output_dir / "regime_mapping.yaml").exists()
        assert (output_dir / "regime_analysis.csv").exists()

        # Verify regime mapping structure
        with open(output_dir / "regime_mapping.yaml") as f:
            mapping = yaml.safe_load(f)

        assert "mapping" in mapping
        assert "uptrend_low_vol" in mapping["mapping"]
        assert "uptrend_high_vol" in mapping["mapping"]
        assert "downtrend_low_vol" in mapping["mapping"]
        assert "downtrend_high_vol" in mapping["mapping"]
        assert "fallback" in mapping

        # Each regime should have a strategy assigned
        for regime in ["uptrend_low_vol", "uptrend_high_vol", "downtrend_low_vol", "downtrend_high_vol"]:
            assert "strategy" in mapping["mapping"][regime]

    def test_regime_mapping_strategies_are_valid(self, data_access, output_dir):
        """Regime mapping should reference valid strategy configs."""
        optimizer = StrategyOptimizer(
            data_access=data_access,
            regime_analysis_enabled=True,
        )

        result = optimizer.run(max_configs=10, periods=[1])
        optimizer.save_outputs(result, output_dir)

        with open(output_dir / "regime_mapping.yaml") as f:
            mapping = yaml.safe_load(f)

        # Each strategy should have a config path that exists or can be generated
        for regime, info in mapping["mapping"].items():
            strategy = info["strategy"]
            assert len(strategy) > 0, f"Empty strategy for {regime}"
```

#### Scenario 2: Monitor → Rebalancer Flow

```python
class TestMonitorToRebalancerFlow:
    """Test daily monitor integrates with rebalancer."""

    @pytest.fixture
    def state_dir(self, tmp_path):
        return tmp_path / "state"

    @pytest.fixture
    def artifacts_dir(self, tmp_path):
        return tmp_path / "artifacts"

    @pytest.fixture
    def regime_mapping_file(self, tmp_path):
        """Create test regime mapping."""
        mapping = {
            "version": "1.0",
            "mapping": {
                "uptrend_low_vol": {
                    "strategy": "momentum_acceleration",
                    "config_path": "configs/strategies/momentum_acceleration.yaml",
                },
                "uptrend_high_vol": {
                    "strategy": "vol_adjusted_momentum",
                    "config_path": "configs/strategies/vol_adjusted_momentum.yaml",
                },
                "downtrend_low_vol": {
                    "strategy": "momentum",
                    "config_path": "configs/strategies/momentum.yaml",
                },
                "downtrend_high_vol": {
                    "strategy": "vol_adjusted_momentum",
                    "config_path": "configs/strategies/vol_adjusted_momentum.yaml",
                },
            },
            "fallback": {
                "strategy": "momentum_acceleration",
                "config_path": "configs/strategies/momentum_acceleration.yaml",
            },
        }
        path = tmp_path / "regime_mapping.yaml"
        with open(path, "w") as f:
            yaml.dump(mapping, f)
        return path

    def test_rebalance_uses_current_regime(
        self,
        data_access,
        state_dir,
        artifacts_dir,
        regime_mapping_file,
    ):
        """
        Rebalancer should use regime from monitor.

        Flow:
        1. Monitor updates regime state
        2. Rebalancer reads regime
        3. Rebalancer selects correct strategy
        4. Rebalancer generates portfolio
        """
        # Create monitor
        monitor = DailyRegimeMonitor(
            data_access=data_access,
            state_dir=state_dir,
        )

        # Update regime
        regime_state = monitor.update(as_of=pd.Timestamp("2026-01-20"))

        # Create rebalancer
        rebalancer = RegimeAwareRebalancer(
            data_access=data_access,
            regime_monitor=monitor,
            regime_mapping_path=regime_mapping_file,
            artifacts_dir=artifacts_dir,
        )

        # Run rebalance
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        # Verify regime matches
        assert result.regime == regime_state.name

        # Verify strategy was selected from mapping
        assert result.strategy_used is not None

    def test_multiple_day_rebalance_sequence(
        self,
        data_access,
        state_dir,
        artifacts_dir,
        regime_mapping_file,
    ):
        """
        Test sequence of rebalances maintains state correctly.

        Flow:
        1. Day 1: Monitor + Rebalance
        2. Day 2: Monitor (check hysteresis) + Rebalance
        3. Verify holdings state persisted
        """
        monitor = DailyRegimeMonitor(
            data_access=data_access,
            state_dir=state_dir,
        )

        rebalancer = RegimeAwareRebalancer(
            data_access=data_access,
            regime_monitor=monitor,
            regime_mapping_path=regime_mapping_file,
            artifacts_dir=artifacts_dir,
        )

        # Day 1
        result1 = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-17"),
            dry_run=False,
        )

        # Verify holdings saved
        assert rebalancer.holdings_file.exists()

        # Day 2 (next week - rebalance day)
        result2 = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-24"),
            dry_run=False,
        )

        # Should have trades (even if small) based on alpha changes
        assert len(result2.trades) > 0

        # Check history
        history = monitor.get_history()
        assert len(history) >= 2
```

#### Scenario 3: Regime Change Triggers Strategy Switch

```python
class TestRegimeChangeStrategySwitch:
    """Test that regime changes cause strategy switching."""

    def test_different_regimes_different_strategies(self, data_access, tmp_path):
        """
        Different regimes should (potentially) select different strategies.

        Note: This test verifies the *mechanism* works, not that strategies
        are always different (they might legitimately be the same).
        """
        # Create components
        monitor = DailyRegimeMonitor(
            data_access=data_access,
            state_dir=tmp_path / "state",
        )

        # Create mapping with different strategies per regime
        mapping = {
            "mapping": {
                "uptrend_low_vol": {"strategy": "strategy_A", "config_path": "..."},
                "uptrend_high_vol": {"strategy": "strategy_B", "config_path": "..."},
                "downtrend_low_vol": {"strategy": "strategy_C", "config_path": "..."},
                "downtrend_high_vol": {"strategy": "strategy_D", "config_path": "..."},
            },
            "fallback": {"strategy": "strategy_A", "config_path": "..."},
        }

        # Test lookup for each regime
        from quantetf.regime.config import get_strategy_for_regime

        results = {}
        for regime in mapping["mapping"].keys():
            info = get_strategy_for_regime(regime, mapping)
            results[regime] = info["strategy"]

        # Verify each regime has a strategy
        assert len(results) == 4

        # Verify at least some differentiation exists
        unique_strategies = set(results.values())
        assert len(unique_strategies) >= 2, "Should have different strategies per regime"
```

#### Scenario 4: End-to-End No-Lookahead Verification

```python
class TestNoLookahead:
    """Verify no lookahead bias in regime system."""

    def test_regime_labels_use_point_in_time_data(self, data_access):
        """
        Regime labels should only use data available at that time.
        """
        detector = RegimeDetector()
        indicators = RegimeIndicators(data_access)
        analyzer = RegimeAnalyzer(detector, indicators)

        # Label historical period
        labels = analyzer.label_history(
            start_date=pd.Timestamp("2025-01-01"),
            end_date=pd.Timestamp("2025-06-30"),
        )

        # Each label should have indicators from that date
        for date, row in labels.iterrows():
            # The 200MA should be calculated from data up to that date
            # (This is implicit in the as_of pattern but worth verifying)
            assert row["spy_price"] > 0
            assert row["spy_200ma"] > 0 or pd.isna(row["spy_200ma"])  # NaN ok for warmup

    def test_rebalance_uses_as_of_data_only(self, data_access, tmp_path):
        """
        Rebalance should not use future data.
        """
        monitor = DailyRegimeMonitor(
            data_access=data_access,
            state_dir=tmp_path / "state",
        )

        # Rebalance for historical date
        historical_date = pd.Timestamp("2025-06-15")

        # The regime indicators should be from June 15, not later
        state = monitor.update(as_of=historical_date)

        # Verify as_of is respected
        assert state.as_of == historical_date
```

---

## Shared Fixtures

```python
# tests/integration/conftest.py
"""Shared fixtures for integration tests."""

import pytest
import pandas as pd
from pathlib import Path

from quantetf.data.access import DataAccessFactory


@pytest.fixture(scope="module")
def data_access():
    """
    Create data access context for integration tests.

    Uses real snapshot data for realistic testing.
    Scope is module to avoid repeated loading.
    """
    return DataAccessFactory.create_context(
        config={"snapshot_path": "data/snapshots/snapshot_20260115_*/data.parquet"},
        enable_caching=True,
    )


@pytest.fixture
def clean_state(tmp_path):
    """Provide clean state directory for each test."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return state_dir


@pytest.fixture
def sample_regime_mapping(tmp_path):
    """Create sample regime mapping for tests."""
    import yaml

    mapping = {
        "version": "1.0",
        "mapping": {
            "uptrend_low_vol": {
                "strategy": "momentum_acceleration",
                "config_path": "configs/strategies/momentum_acceleration.yaml",
            },
            "uptrend_high_vol": {
                "strategy": "vol_adjusted_momentum",
                "config_path": "configs/strategies/vol_adjusted_momentum.yaml",
            },
            "downtrend_low_vol": {
                "strategy": "momentum",
                "config_path": "configs/strategies/momentum.yaml",
            },
            "downtrend_high_vol": {
                "strategy": "vol_adjusted_momentum",
                "config_path": "configs/strategies/vol_adjusted_momentum.yaml",
            },
        },
        "fallback": {
            "strategy": "momentum_acceleration",
            "config_path": "configs/strategies/momentum_acceleration.yaml",
        },
    }

    path = tmp_path / "regime_mapping.yaml"
    with open(path, "w") as f:
        yaml.dump(mapping, f)

    return path
```

---

## Test Commands

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test class
pytest tests/integration/test_regime_system.py::TestOptimizationToRegimeMapping -v

# Run with coverage
pytest tests/integration/ --cov=src/quantetf/regime --cov=src/quantetf/production

# Run slow tests (full optimization)
pytest tests/integration/ -v -m "not slow"
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `tests/integration/__init__.py` | Package init |
| `tests/integration/conftest.py` | Shared fixtures |
| `tests/integration/test_regime_system.py` | Main integration tests |

---

## Acceptance Criteria

- [ ] Optimization → regime mapping flow works end-to-end
- [ ] Monitor → rebalancer flow works correctly
- [ ] State persists across component restarts
- [ ] No lookahead bias in regime labeling or rebalancing
- [ ] All 4 regimes are handled correctly
- [ ] Fallback strategy works when regime unknown
- [ ] Tests run in < 5 minutes (use small configs)
- [ ] All tests pass

---

## Notes for Implementer

1. **Use real data:** Integration tests should use actual snapshot data
2. **Keep tests fast:** Limit optimizer to 10-20 configs
3. **Test edge cases:** What if VIX data missing? What if all strategies fail?
4. **Mark slow tests:** Use `@pytest.mark.slow` for full optimization tests

---

**Document Version:** 1.0
**Created:** 2026-01-24
**For:** Coding Agent
