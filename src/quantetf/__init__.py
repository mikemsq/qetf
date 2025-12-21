"""QuantETF strategy platform.

This package provides modular components for:
- data management and snapshotting
- universe definition
- feature computation
- alpha and risk models
- portfolio construction
- backtesting and evaluation
- production recommendation generation
"""

from .types import (
    DatasetVersion,
    Universe,
    FeatureFrame,
    AlphaScores,
    RiskModelOutput,
    TargetWeights,
    BacktestResult,
    RecommendationPacket,
)
