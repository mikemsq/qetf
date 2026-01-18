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
            # Will be implemented in IMPL-021
            raise NotImplementedError(
                "FREDMacroAccessor implementation in IMPL-021"
            )
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
            config: Universe configuration (config_dir, etc.)
            
        Returns:
            Configured UniverseDataAccessor instance
            
        Note:
            Implementation in IMPL-022: ConfigFileUniverseAccessor
        """
        # Will be implemented in IMPL-022
        raise NotImplementedError(
            "ConfigFileUniverseAccessor implementation in IMPL-022"
        )
    
    @staticmethod
    def create_reference_accessor(
        config: Optional[Dict[str, Any]] = None,
    ) -> ReferenceDataAccessor:
        """Create reference data accessor.
        
        Args:
            config: Reference data configuration
            
        Returns:
            Configured ReferenceDataAccessor instance
            
        Note:
            Implementation in IMPL-023: ReferenceDataAccessor
        """
        # Will be implemented in IMPL-023
        raise NotImplementedError(
            "ReferenceDataAccessor implementation in IMPL-023"
        )
    
    @staticmethod
    def create_context(
        config_file: Optional[Path] = None,
    ) -> DataAccessContext:
        """Create a complete DataAccessContext.
        
        Convenience method that creates all accessors and combines them
        into a single context object.
        
        Args:
            config_file: Path to data_access.yaml config (optional)
                        If not provided, uses default configuration
            
        Returns:
            Fully configured DataAccessContext with all accessors
            
        Note:
            Will be fully implemented once individual accessors
            are completed in IMPL-020, 021, 022, 023.
        """
        # Will use individual create_* methods above
        # Once all accessors are implemented
        raise NotImplementedError(
            "Complete DataAccessFactory.create_context in IMPL-024 "
            "after all individual accessors are implemented"
        )
