# Task Handoff: IMPL-035b - Regime Configuration Files

**Task ID:** IMPL-035b
**Parent Task:** IMPL-035 (Regime-Based Strategy Selection System)
**Status:** ready
**Priority:** HIGH
**Type:** Configuration
**Estimated Effort:** 1 hour
**Dependencies:** IMPL-035a (uses types defined there)

---

## Summary

Create YAML configuration files for regime detection thresholds and the default regime-to-strategy mapping. These configs externalize the regime system parameters so they can be tuned without code changes.

---

## Deliverables

1. **`configs/regimes/thresholds.yaml`** - Regime detection thresholds
2. **`configs/regimes/default_mapping.yaml`** - Default regime→strategy mapping
3. **`src/quantetf/regime/config.py`** - Config loader functions
4. **`tests/regime/test_config.py`** - Config loading tests

---

## Technical Specification

### File 1: `configs/regimes/thresholds.yaml`

```yaml
# Regime Detection Thresholds
# See ADR-001 for rationale on these values
#
# Trend detection uses SPY price vs 200-day moving average
# Volatility detection uses VIX level

version: "1.0"

trend:
  indicator: "spy_vs_200ma"
  # Hysteresis band: ±2% around 200MA
  # Enter downtrend: SPY < 200MA × 0.98
  # Exit downtrend:  SPY > 200MA × 1.02
  hysteresis_pct: 0.02

volatility:
  indicator: "vix"
  # Enter high_vol: VIX > 25
  # Exit high_vol:  VIX < 20
  high_threshold: 25.0
  low_threshold: 20.0

# Regime definitions (for reference/validation)
regimes:
  - name: "uptrend_low_vol"
    description: "Calm bull market - clean trends, ride momentum"
    trend: "uptrend"
    vol: "low_vol"

  - name: "uptrend_high_vol"
    description: "Volatile rally - trends exist but noisy"
    trend: "uptrend"
    vol: "high_vol"

  - name: "downtrend_low_vol"
    description: "Grinding bear - reduce exposure, faster signals"
    trend: "downtrend"
    vol: "low_vol"

  - name: "downtrend_high_vol"
    description: "Crisis/panic - preserve capital"
    trend: "downtrend"
    vol: "high_vol"
```

### File 2: `configs/regimes/default_mapping.yaml`

```yaml
# Default Regime → Strategy Mapping
#
# This file provides fallback mappings when optimization-derived
# mappings are not available. Updated by optimization runs.
#
# Strategy names must match configs in configs/strategies/

version: "1.0"

# Mapping based on financial intuition (see ADR-001)
mapping:
  uptrend_low_vol:
    strategy: "momentum_acceleration"
    rationale: "Clean trends, aggressive momentum works well"
    config_path: "configs/strategies/momentum_acceleration.yaml"

  uptrend_high_vol:
    strategy: "vol_adjusted_momentum"
    rationale: "Trends exist but noisy, need vol adjustment"
    config_path: "configs/strategies/vol_adjusted_momentum.yaml"

  downtrend_low_vol:
    strategy: "momentum"
    rationale: "Shorter lookback for faster signals in declining market"
    config_path: "configs/strategies/momentum_short.yaml"

  downtrend_high_vol:
    strategy: "vol_adjusted_momentum"
    rationale: "Defensive positioning, minimize drawdown"
    config_path: "configs/strategies/vol_adjusted_momentum.yaml"

# Fallback when regime is unknown or data missing
fallback:
  strategy: "momentum_acceleration"
  rationale: "Best overall performer from optimization"
  config_path: "configs/strategies/momentum_acceleration.yaml"

# Metadata
metadata:
  generated_by: "manual"
  generated_at: "2026-01-24"
  notes: "Initial defaults based on ADR-001 intuition. Will be updated after regime analysis."
```

### File 3: `src/quantetf/regime/config.py`

```python
"""Configuration loading utilities for regime system."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

from .types import RegimeConfig

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS_PATH = Path("configs/regimes/thresholds.yaml")
DEFAULT_MAPPING_PATH = Path("configs/regimes/default_mapping.yaml")


def load_thresholds(path: Optional[Path] = None) -> RegimeConfig:
    """
    Load regime detection thresholds from YAML.

    Args:
        path: Path to thresholds.yaml. Uses default if not provided.

    Returns:
        RegimeConfig with loaded thresholds

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is malformed
    """
    config_path = path or DEFAULT_THRESHOLDS_PATH

    if not config_path.exists():
        logger.warning(f"Thresholds not found at {config_path}, using defaults")
        return RegimeConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f)

    try:
        return RegimeConfig(
            trend_hysteresis_pct=data["trend"]["hysteresis_pct"],
            vix_high_threshold=data["volatility"]["high_threshold"],
            vix_low_threshold=data["volatility"]["low_threshold"],
        )
    except KeyError as e:
        raise ValueError(f"Missing required field in thresholds config: {e}")


def load_regime_mapping(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load regime-to-strategy mapping from YAML.

    Args:
        path: Path to mapping.yaml. Uses default if not provided.

    Returns:
        Dict mapping regime names to strategy info:
        {
            "uptrend_low_vol": {
                "strategy": "momentum_acceleration",
                "config_path": "configs/strategies/...",
                "rationale": "..."
            },
            ...
            "fallback": {...}
        }

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is malformed
    """
    config_path = path or DEFAULT_MAPPING_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Regime mapping not found at {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    mapping = data.get("mapping", {})
    mapping["fallback"] = data.get("fallback", {})

    # Validate all 4 regimes are present
    required_regimes = [
        "uptrend_low_vol",
        "uptrend_high_vol",
        "downtrend_low_vol",
        "downtrend_high_vol",
    ]
    for regime in required_regimes:
        if regime not in mapping:
            raise ValueError(f"Missing mapping for regime: {regime}")

    return mapping


def get_strategy_for_regime(
    regime_name: str,
    mapping: Optional[Dict[str, Dict[str, Any]]] = None,
    mapping_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Get strategy config for a regime.

    Args:
        regime_name: One of the 4 regime names or "fallback"
        mapping: Pre-loaded mapping dict, or None to load from file
        mapping_path: Path to mapping file if loading fresh

    Returns:
        Dict with "strategy", "config_path", "rationale"
    """
    if mapping is None:
        mapping = load_regime_mapping(mapping_path)

    if regime_name in mapping:
        return mapping[regime_name]
    else:
        logger.warning(f"Unknown regime '{regime_name}', using fallback")
        return mapping["fallback"]
```

---

## Test Cases

```python
# tests/regime/test_config.py
import pytest
from pathlib import Path
import tempfile
import yaml

from quantetf.regime.config import (
    load_thresholds,
    load_regime_mapping,
    get_strategy_for_regime,
)
from quantetf.regime.types import RegimeConfig


class TestLoadThresholds:
    """Test threshold configuration loading."""

    def test_load_default_thresholds(self, tmp_path):
        """Load thresholds from valid YAML."""
        config_file = tmp_path / "thresholds.yaml"
        config_file.write_text("""
version: "1.0"
trend:
  hysteresis_pct: 0.03
volatility:
  high_threshold: 30
  low_threshold: 18
        """)

        config = load_thresholds(config_file)

        assert config.trend_hysteresis_pct == 0.03
        assert config.vix_high_threshold == 30
        assert config.vix_low_threshold == 18

    def test_missing_file_returns_defaults(self, tmp_path):
        """Missing file should return default config."""
        config = load_thresholds(tmp_path / "nonexistent.yaml")

        assert config.trend_hysteresis_pct == 0.02
        assert config.vix_high_threshold == 25
        assert config.vix_low_threshold == 20

    def test_malformed_config_raises(self, tmp_path):
        """Malformed config should raise ValueError."""
        config_file = tmp_path / "thresholds.yaml"
        config_file.write_text("""
trend:
  wrong_field: 0.03
        """)

        with pytest.raises(ValueError, match="Missing required field"):
            load_thresholds(config_file)


class TestLoadRegimeMapping:
    """Test regime mapping loading."""

    def test_load_valid_mapping(self, tmp_path):
        """Load mapping with all 4 regimes."""
        config_file = tmp_path / "mapping.yaml"
        config_file.write_text("""
mapping:
  uptrend_low_vol:
    strategy: momentum_accel
    config_path: path/to/config.yaml
  uptrend_high_vol:
    strategy: vol_mom
    config_path: path/to/config2.yaml
  downtrend_low_vol:
    strategy: short_mom
    config_path: path/to/config3.yaml
  downtrend_high_vol:
    strategy: defensive
    config_path: path/to/config4.yaml
fallback:
  strategy: momentum_accel
  config_path: path/to/fallback.yaml
        """)

        mapping = load_regime_mapping(config_file)

        assert mapping["uptrend_low_vol"]["strategy"] == "momentum_accel"
        assert mapping["downtrend_high_vol"]["strategy"] == "defensive"
        assert mapping["fallback"]["strategy"] == "momentum_accel"

    def test_missing_regime_raises(self, tmp_path):
        """Missing regime should raise ValueError."""
        config_file = tmp_path / "mapping.yaml"
        config_file.write_text("""
mapping:
  uptrend_low_vol:
    strategy: test
fallback:
  strategy: test
        """)

        with pytest.raises(ValueError, match="Missing mapping for regime"):
            load_regime_mapping(config_file)


class TestGetStrategyForRegime:
    """Test strategy lookup."""

    @pytest.fixture
    def mapping(self):
        return {
            "uptrend_low_vol": {"strategy": "momentum", "config_path": "a.yaml"},
            "uptrend_high_vol": {"strategy": "vol_adj", "config_path": "b.yaml"},
            "downtrend_low_vol": {"strategy": "short", "config_path": "c.yaml"},
            "downtrend_high_vol": {"strategy": "defensive", "config_path": "d.yaml"},
            "fallback": {"strategy": "default", "config_path": "default.yaml"},
        }

    def test_get_known_regime(self, mapping):
        """Known regime returns correct strategy."""
        result = get_strategy_for_regime("uptrend_low_vol", mapping)
        assert result["strategy"] == "momentum"

    def test_unknown_regime_returns_fallback(self, mapping):
        """Unknown regime returns fallback."""
        result = get_strategy_for_regime("invalid_regime", mapping)
        assert result["strategy"] == "default"
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `configs/regimes/thresholds.yaml` | Detection thresholds |
| `configs/regimes/default_mapping.yaml` | Strategy mapping |
| `src/quantetf/regime/config.py` | Config loader |
| `tests/regime/test_config.py` | Unit tests |

---

## Acceptance Criteria

- [ ] `thresholds.yaml` contains all required fields per ADR-001
- [ ] `default_mapping.yaml` maps all 4 regimes + fallback
- [ ] `load_thresholds()` returns `RegimeConfig` dataclass
- [ ] `load_regime_mapping()` validates all regimes present
- [ ] `get_strategy_for_regime()` returns fallback for unknown regimes
- [ ] All tests pass
- [ ] Configs are valid YAML (parseable)

---

## Notes for Implementer

1. **Match existing strategy names:** Check `configs/strategies/` for actual strategy config file names
2. **YAML formatting:** Use consistent indentation (2 spaces)
3. **Version field:** Include for future compatibility checking
4. **Rationale fields:** Helpful for documentation but optional for code

---

**Document Version:** 1.0
**Created:** 2026-01-24
**For:** Coding Agent
