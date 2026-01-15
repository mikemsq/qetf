"""Factory for creating alpha models from configuration.

This module provides a registry-based factory pattern for instantiating
alpha models from YAML configuration files.
"""

from typing import Dict, Any, Type
import logging

from quantetf.alpha.base import AlphaModel

logger = logging.getLogger(__name__)


class AlphaModelRegistry:
    """Registry for alpha model types.

    This registry maps string identifiers (used in YAML configs) to
    alpha model class constructors.
    """

    _registry: Dict[str, Type[AlphaModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[AlphaModel]) -> None:
        """Register an alpha model class.

        Args:
            name: String identifier for this model type (e.g., 'momentum')
            model_class: AlphaModel subclass to register

        Raises:
            ValueError: If name is already registered
        """
        if name in cls._registry:
            raise ValueError(f"Alpha model '{name}' is already registered")

        cls._registry[name] = model_class
        logger.debug(f"Registered alpha model: {name} -> {model_class.__name__}")

    @classmethod
    def create(cls, model_type: str, params: Dict[str, Any]) -> AlphaModel:
        """Create an alpha model instance from config.

        Args:
            model_type: String identifier for the model type
            params: Dictionary of parameters to pass to constructor

        Returns:
            Instantiated alpha model

        Raises:
            ValueError: If model_type is not registered
            TypeError: If params are invalid for the model constructor
        """
        if model_type not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Unknown alpha model type: '{model_type}'. "
                f"Available types: {available}"
            )

        model_class = cls._registry[model_type]

        try:
            # Filter params to only include those accepted by the constructor
            # This allows configs to have extra fields that are ignored
            import inspect
            sig = inspect.signature(model_class.__init__)
            valid_params = {
                k: v for k, v in params.items()
                if k in sig.parameters
            }

            logger.info(f"Creating {model_type} alpha model with params: {valid_params}")
            return model_class(**valid_params)

        except TypeError as e:
            raise TypeError(
                f"Failed to create {model_type} model with params {params}: {e}"
            ) from e

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model types.

        Returns:
            List of registered model type names
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model type is registered.

        Args:
            name: Model type name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry


# Auto-register all standard alpha models
def _register_standard_models():
    """Register all standard alpha models on module import."""
    try:
        from quantetf.alpha.momentum import MomentumAlpha
        AlphaModelRegistry.register('momentum', MomentumAlpha)
    except ImportError:
        logger.warning("Could not import MomentumAlpha")

    try:
        from quantetf.alpha.momentum_acceleration import MomentumAccelerationAlpha
        AlphaModelRegistry.register('momentum_acceleration', MomentumAccelerationAlpha)
    except ImportError:
        logger.warning("Could not import MomentumAccelerationAlpha")

    try:
        from quantetf.alpha.vol_adjusted_momentum import VolAdjustedMomentumAlpha
        AlphaModelRegistry.register('vol_adjusted_momentum', VolAdjustedMomentumAlpha)
    except ImportError:
        logger.warning("Could not import VolAdjustedMomentumAlpha")

    try:
        from quantetf.alpha.residual_momentum import ResidualMomentumAlpha
        AlphaModelRegistry.register('residual_momentum', ResidualMomentumAlpha)
    except ImportError:
        logger.warning("Could not import ResidualMomentumAlpha")

    # New regime-based alpha models (IMPL-006)
    try:
        from quantetf.alpha.trend_filtered_momentum import TrendFilteredMomentum
        AlphaModelRegistry.register('trend_filtered_momentum', TrendFilteredMomentum)
    except ImportError:
        logger.warning("Could not import TrendFilteredMomentum")

    try:
        from quantetf.alpha.dual_momentum import DualMomentum
        AlphaModelRegistry.register('dual_momentum', DualMomentum)
    except ImportError:
        logger.warning("Could not import DualMomentum")

    try:
        from quantetf.alpha.value_momentum import ValueMomentum
        AlphaModelRegistry.register('value_momentum', ValueMomentum)
    except ImportError:
        logger.warning("Could not import ValueMomentum")


# Register models on import
_register_standard_models()


def create_alpha_model(config: Dict[str, Any]) -> AlphaModel:
    """Convenience function to create alpha model from config dict.

    Args:
        config: Dictionary with 'type' key and optional parameter keys

    Returns:
        Instantiated alpha model

    Example:
        >>> config = {
        ...     'type': 'momentum_acceleration',
        ...     'short_lookback_days': 63,
        ...     'long_lookback_days': 252
        ... }
        >>> model = create_alpha_model(config)
    """
    if 'type' not in config:
        raise ValueError("Config must have 'type' field specifying model type")

    model_type = config['type']
    params = {k: v for k, v in config.items() if k != 'type'}

    return AlphaModelRegistry.create(model_type, params)
