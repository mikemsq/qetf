"""Configuration loading utilities for regime system."""

from pathlib import Path
from typing import Any, Dict, Optional
import logging

import yaml

from .types import RegimeConfig

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS_PATH = Path("configs/regimes/thresholds.yaml")
DEFAULT_MAPPING_PATH = Path("configs/regimes/default_mapping.yaml")


def load_thresholds(path: Optional[Path] = None) -> RegimeConfig:
    """Load regime detection thresholds from YAML.

    Args:
        path: Path to thresholds.yaml. Uses default if not provided.

    Returns:
        RegimeConfig with loaded thresholds

    Raises:
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
    """Load regime-to-strategy mapping from YAML.

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
    """Get strategy config for a regime.

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
