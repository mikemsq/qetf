from .base import AlphaModel, AlphaModelSpec
from .momentum import CrossSectionalMomentum, MomentumAlpha
from .trend_filtered_momentum import TrendFilteredMomentum
from .dual_momentum import DualMomentum
from .value_momentum import ValueMomentum
from .selector import (
    AlphaSelector,
    AlphaSelection,
    MarketRegime,
    RegimeBasedSelector,
    RegimeWeightedSelector,
    ConfigurableSelector,
    compute_alpha_with_selection,
)
from .regime_aware import (
    RegimeDetector,
    RegimeAwareAlpha,
)

__all__ = [
    'AlphaModel',
    'AlphaModelSpec',
    'CrossSectionalMomentum',
    'MomentumAlpha',
    'TrendFilteredMomentum',
    'DualMomentum',
    'ValueMomentum',
    'AlphaSelector',
    'AlphaSelection',
    'MarketRegime',
    'RegimeBasedSelector',
    'RegimeWeightedSelector',
    'ConfigurableSelector',
    'compute_alpha_with_selection',
    'RegimeDetector',
    'RegimeAwareAlpha',
]
