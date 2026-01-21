"""Factory for creating configured DAL accessors."""

from pathlib import Path
from typing import Optional, Dict, Any
from .context import DataAccessContext
from .abstract import (
    PriceDataAccessor,
    MacroDataAccessor,
    UniverseDataAccessor,
    ReferenceDataAccessor,
)


class DataAccessFactory:
    """Factory for creating configured DAL accessors.
    
    Handles instantiation and configuration of all accessor types.
    Provides convenient methods for creating accessors individually or
    as a complete context.
    
    Usage:
        # Create from defaults
        ctx = DataAccessFactory.create_context()
        
        # Create with custom config
        ctx = DataAccessFactory.create_context(
            config_file="configs/data_access.yaml"
        )
        
        # Create individual accessors
        prices = DataAccessFactory.create_price_accessor(
            source="snapshot",
            config={"snapshot_dir": "data/snapshots/latest"}
        )
    """
    
    @staticmethod
    def create_price_accessor(
        source: str = "snapshot",
        config: Optional[Dict[str, Any]] = None,
    ) -> PriceDataAccessor:
        """Create price accessor from source type.
        
        Args:
            source: "snapshot" (default), "live", or "mock"
            config: Source-specific configuration dict
                For "snapshot": must contain "snapshot_path" key
                
        Returns:
            Configured PriceDataAccessor instance
            
        Raises:
            ValueError: If source type not supported or config invalid
            
        Note:
            Implementation of specific sources in:
            - IMPL-020: SnapshotPriceAccessor
            - IMPL-032: LivePriceAccessor
            - Tests: MockPriceAccessor
        """
        if source == "snapshot":
            # IMPL-020: SnapshotPriceAccessor
            from .snapshot_price import SnapshotPriceAccessor
            
            if config is None or "snapshot_path" not in config:
                raise ValueError(
                    "snapshot source requires config with 'snapshot_path' key"
                )
            
            snapshot_path = Path(config["snapshot_path"])
            return SnapshotPriceAccessor(snapshot_path)
        elif source == "live":
            # Will be implemented in IMPL-032
            raise NotImplementedError(
                "LivePriceAccessor implementation in IMPL-032"
            )
        elif source == "mock":
            # For testing
            raise NotImplementedError("MockPriceAccessor for testing")
        else:
            raise ValueError(f"Unknown price accessor source: {source}")
    
    @staticmethod
    def create_macro_accessor(
        source: str = "fred",
        config: Optional[Dict[str, Any]] = None,
    ) -> MacroDataAccessor:
        """Create macro accessor from source type.
        
        Args:
            source: "fred" (default) or "mock"
            config: Source-specific configuration
                For "fred": optional "data_dir" key for macro data directory
                
        Returns:
            Configured MacroDataAccessor instance
            
        Raises:
            ValueError: If source type not supported
            
        Note:
            Implementation of specific sources in:
            - IMPL-021: FREDMacroAccessor
            - Tests: MockMacroAccessor
        """
        if source == "fred":
            # IMPL-021: FREDMacroAccessor
            from .fred_macro import FREDMacroAccessor
            from quantetf.data.macro_loader import MacroDataLoader
            
            if config is None:
                config = {}
            
            data_dir = config.get("data_dir", None)
            
            # Create MacroDataLoader
            if data_dir is not None:
                loader = MacroDataLoader(data_dir=Path(data_dir))
            else:
                loader = MacroDataLoader()
            
            return FREDMacroAccessor(loader)
        elif source == "mock":
            # For testing
            raise NotImplementedError("MockMacroAccessor for testing")
        else:
            raise ValueError(f"Unknown macro accessor source: {source}")
    
    @staticmethod
    def create_universe_accessor(
        config: Optional[Dict[str, Any]] = None,
    ) -> UniverseDataAccessor:
        """Create universe accessor.

        Args:
            config: Universe configuration dictionary. Optional keys:
                - config_dir: Path to universe config directory
                             (default: configs/universes)
                - cache: Whether to cache parsed configs (default: True)

        Returns:
            Configured UniverseDataAccessor instance

        Example:
            >>> accessor = DataAccessFactory.create_universe_accessor()
            >>> tickers = accessor.get_universe("tier1_initial_20")

            >>> # With custom config directory
            >>> accessor = DataAccessFactory.create_universe_accessor(
            ...     config={"config_dir": "my_configs/universes"}
            ... )
        """
        from .universe import ConfigFileUniverseAccessor

        if config is None:
            config = {}

        # Default config directory
        config_dir = config.get("config_dir", "configs/universes")
        cache = config.get("cache", True)

        return ConfigFileUniverseAccessor(
            config_dir=Path(config_dir),
            cache=cache,
        )
    
    @staticmethod
    def create_reference_accessor(
        config: Optional[Dict[str, Any]] = None,
    ) -> ReferenceDataAccessor:
        """Create reference data accessor.

        Args:
            config: Reference data configuration dictionary. Optional keys:
                - config_dir: Path to directory containing reference YAML files
                              (tickers.yaml, exchanges.yaml)
                              Defaults to "configs/reference"

        Returns:
            Configured ReferenceDataAccessor instance

        Raises:
            ValueError: If config_dir does not exist

        Example:
            >>> accessor = DataAccessFactory.create_reference_accessor()
            >>> spy_info = accessor.get_ticker_info("SPY")
            >>> print(spy_info.sector)
            "Broad Market"

            >>> # With custom config directory
            >>> accessor = DataAccessFactory.create_reference_accessor(
            ...     config={"config_dir": "my_configs/reference"}
            ... )
        """
        from .reference import StaticReferenceDataAccessor

        if config is None:
            config = {}

        # Default to configs/reference directory
        config_dir = config.get("config_dir", "configs/reference")

        return StaticReferenceDataAccessor(config_dir=Path(config_dir))
    
    @staticmethod
    def create_context(
        config_file: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True,
        cache_config: Optional[Dict[str, Any]] = None,
    ) -> DataAccessContext:
        """Create a complete DataAccessContext with all accessors configured.

        Convenience method that creates all accessors and combines them
        into a single context object. Supports optional caching layer.

        Args:
            config_file: Path to data_access.yaml config (optional)
                        If provided, overrides other config parameters
            config: Direct configuration dictionary with keys:
                - price_source: "snapshot" (default) or "live"
                - snapshot_path: Path to snapshot parquet file (required for snapshot)
                - macro_source: "fred" (default)
                - macro_data_dir: Optional path to macro data
                - universe_config_dir: Path to universe configs (default: configs/universes)
                - reference_config_dir: Path to reference configs (default: configs/reference)
            enable_caching: Whether to wrap accessors with caching (default: True)
            cache_config: Caching configuration with keys:
                - price_max_cache_mb: Max price cache size (default: 500)
                - macro_cache_ttl_seconds: Macro cache TTL (default: 86400)

        Returns:
            Fully configured DataAccessContext with all accessors

        Raises:
            ValueError: If required configuration is missing
            FileNotFoundError: If config_file doesn't exist

        Example:
            # Basic usage with snapshot
            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": "data/snapshots/latest/data.parquet"}
            )

            # With caching disabled
            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": "..."},
                enable_caching=False
            )

            # With custom cache settings
            ctx = DataAccessFactory.create_context(
                config={"snapshot_path": "..."},
                cache_config={"price_max_cache_mb": 1000}
            )
        """
        import yaml

        # Load config from file if provided
        if config_file is not None:
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                config = yaml.safe_load(f)

        if config is None:
            config = {}

        if cache_config is None:
            cache_config = {}

        # Extract configuration
        price_source = config.get("price_source", "snapshot")
        macro_source = config.get("macro_source", "fred")

        # Create price accessor
        price_config = {}
        if "snapshot_path" in config:
            price_config["snapshot_path"] = config["snapshot_path"]

        prices = DataAccessFactory.create_price_accessor(
            source=price_source,
            config=price_config if price_config else None,
        )

        # Create macro accessor
        macro_config = {}
        if "macro_data_dir" in config:
            macro_config["data_dir"] = config["macro_data_dir"]

        macro = DataAccessFactory.create_macro_accessor(
            source=macro_source,
            config=macro_config if macro_config else None,
        )

        # Create universe accessor
        universe_config = {}
        if "universe_config_dir" in config:
            universe_config["config_dir"] = config["universe_config_dir"]

        universes = DataAccessFactory.create_universe_accessor(
            config=universe_config if universe_config else None,
        )

        # Create reference accessor
        reference_config = {}
        if "reference_config_dir" in config:
            reference_config["config_dir"] = config["reference_config_dir"]

        references = DataAccessFactory.create_reference_accessor(
            config=reference_config if reference_config else None,
        )

        # Apply caching layer if enabled
        if enable_caching:
            from .caching import CachedPriceAccessor, CachedMacroAccessor

            price_max_mb = cache_config.get("price_max_cache_mb", 500)
            macro_ttl = cache_config.get("macro_cache_ttl_seconds", 86400)

            prices = CachedPriceAccessor(prices, max_cache_mb=price_max_mb)
            macro = CachedMacroAccessor(macro, cache_ttl_seconds=macro_ttl)

        return DataAccessContext(
            prices=prices,
            macro=macro,
            universes=universes,
            references=references,
        )
