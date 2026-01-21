"""ConfigFileUniverseAccessor - Universe accessor that reads from YAML config files.

This module implements the UniverseDataAccessor interface for reading
universe definitions from YAML configuration files. Supports both static
universes (fixed ticker lists) and graduated universes (tickers added over time).
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

import pandas as pd
import yaml

from .abstract import UniverseDataAccessor

logger = logging.getLogger(__name__)


class ConfigFileUniverseAccessor(UniverseDataAccessor):
    """Universe accessor that reads from YAML config files.

    Reads universe definitions from configs/universes/*.yaml files.
    Supports static universes (fixed ticker lists) and graduated universes
    (tickers added over time for point-in-time backtesting).

    Args:
        config_dir: Path to directory containing universe YAML files.
                   Defaults to configs/universes/ relative to project root.
        cache: Whether to cache parsed universe definitions. Default True.

    Example:
        >>> accessor = ConfigFileUniverseAccessor(Path("configs/universes"))
        >>> tickers = accessor.get_universe("tier1_initial_20")
        >>> print(len(tickers))
        20

        >>> # Point-in-time access for graduated universes
        >>> tickers = accessor.get_universe_as_of(
        ...     "tier4_broad_200",
        ...     pd.Timestamp("2020-01-01")
        ... )
    """

    def __init__(
        self,
        config_dir: Union[str, Path],
        cache: bool = True,
    ) -> None:
        """Initialize ConfigFileUniverseAccessor.

        Args:
            config_dir: Path to directory containing universe YAML files.
            cache: Whether to cache parsed universe definitions.
        """
        self._config_dir = Path(config_dir)
        self._cache_enabled = cache
        self._cache: Dict[str, Dict[str, Any]] = {}

        if not self._config_dir.exists():
            raise FileNotFoundError(
                f"Universe config directory not found: {self._config_dir}"
            )

        if not self._config_dir.is_dir():
            raise ValueError(
                f"config_dir must be a directory: {self._config_dir}"
            )

        # Build index of available universes
        self._universe_files = self._scan_universe_files()

        logger.info(
            f"ConfigFileUniverseAccessor initialized with {len(self._universe_files)} "
            f"universes from {self._config_dir}"
        )

    def _scan_universe_files(self) -> Dict[str, Path]:
        """Scan config directory for universe YAML files.

        Returns:
            Dict mapping universe names (lowercase) to file paths.
        """
        universe_files: Dict[str, Path] = {}

        for yaml_file in self._config_dir.glob("*.yaml"):
            try:
                config = self._load_yaml(yaml_file)
                if config and "name" in config:
                    name = config["name"].lower()
                    universe_files[name] = yaml_file

                    # Also map by filename (without extension) for convenience
                    filename_key = yaml_file.stem.lower()
                    if filename_key not in universe_files:
                        universe_files[filename_key] = yaml_file
            except Exception as e:
                logger.warning(f"Error scanning {yaml_file}: {e}")

        return universe_files

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Parsed YAML content as dictionary.
        """
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _get_universe_config(self, universe_name: str) -> Dict[str, Any]:
        """Get universe configuration by name.

        Args:
            universe_name: Name of universe (case-insensitive).

        Returns:
            Universe configuration dictionary.

        Raises:
            ValueError: If universe not found.
        """
        name_lower = universe_name.lower()

        # Check cache first
        if self._cache_enabled and name_lower in self._cache:
            return self._cache[name_lower]

        # Find matching file
        if name_lower not in self._universe_files:
            available = list(set(self._universe_files.keys()))
            raise ValueError(
                f"Universe '{universe_name}' not found. "
                f"Available universes: {sorted(available)}"
            )

        # Load config
        config_path = self._universe_files[name_lower]
        config = self._load_yaml(config_path)

        # Cache if enabled
        if self._cache_enabled:
            self._cache[name_lower] = config

        return config

    def _extract_tickers_static(self, config: Dict[str, Any]) -> List[str]:
        """Extract tickers from a static universe configuration.

        Args:
            config: Universe configuration dictionary.

        Returns:
            List of ticker symbols.
        """
        source = config.get("source", {})

        # Handle different config formats
        if "tickers" in source:
            tickers = source["tickers"]
        elif "tickers" in config:
            # Direct tickers in root (legacy format)
            tickers = config["tickers"]
        else:
            raise ValueError(
                f"Universe config missing 'tickers' field: {config.get('name', 'unknown')}"
            )

        # Ensure all tickers are strings and uppercase
        return [str(t).upper().strip() for t in tickers if t]

    def _extract_tickers_graduated(
        self,
        config: Dict[str, Any],
        as_of: Optional[pd.Timestamp] = None,
    ) -> List[str]:
        """Extract tickers from a graduated universe configuration.

        Graduated universes have tickers with added_date fields.
        Only tickers added on or before as_of are included.

        Args:
            config: Universe configuration dictionary.
            as_of: Point-in-time date. If None, returns all tickers.

        Returns:
            List of ticker symbols available as of the given date.
        """
        source = config.get("source", {})
        ticker_entries = source.get("tickers", [])

        tickers = []
        for entry in ticker_entries:
            if isinstance(entry, dict):
                ticker = entry.get("ticker", "").upper().strip()
                added_date_str = entry.get("added_date")

                if as_of is None:
                    # No date filter, include all
                    if ticker:
                        tickers.append(ticker)
                elif added_date_str:
                    # Filter by added_date
                    added_date = pd.Timestamp(added_date_str)
                    if added_date <= as_of and ticker:
                        tickers.append(ticker)
                else:
                    # No added_date, assume always available
                    if ticker:
                        tickers.append(ticker)
            elif isinstance(entry, str):
                # Simple string ticker (no date)
                tickers.append(entry.upper().strip())

        return tickers

    def get_universe(self, universe_name: str) -> List[str]:
        """Get current/latest universe tickers.

        Returns all tickers in the universe (no point-in-time filtering).

        Args:
            universe_name: Name of the universe (case-insensitive).

        Returns:
            List of tickers in the universe.

        Raises:
            ValueError: If universe not found.
        """
        config = self._get_universe_config(universe_name)
        source = config.get("source", {})
        source_type = source.get("type", "static_list")

        if source_type == "graduated":
            tickers = self._extract_tickers_graduated(config, as_of=None)
        else:
            # static_list or unspecified
            tickers = self._extract_tickers_static(config)

        if not tickers:
            warnings.warn(
                f"Universe '{universe_name}' is empty",
                UserWarning
            )

        return tickers

    def get_universe_as_of(
        self,
        universe_name: str,
        as_of: pd.Timestamp,
    ) -> List[str]:
        """Get universe membership at specific point in time.

        For graduated universes, only includes tickers added by as_of date.
        For static universes, returns the full universe (no time variation).

        Args:
            universe_name: Name of the universe (case-insensitive).
            as_of: Point-in-time date.

        Returns:
            List of tickers in universe as of that date.

        Raises:
            ValueError: If universe not found.
        """
        config = self._get_universe_config(universe_name)
        source = config.get("source", {})
        source_type = source.get("type", "static_list")

        if source_type == "graduated":
            tickers = self._extract_tickers_graduated(config, as_of=as_of)
        else:
            # Static universes don't change over time
            tickers = self._extract_tickers_static(config)

        if not tickers:
            warnings.warn(
                f"Universe '{universe_name}' is empty as of {as_of}",
                UserWarning
            )

        return tickers

    def list_available_universes(self) -> List[str]:
        """Return list of available universe names.

        Returns:
            Sorted list of unique universe names.
        """
        # Get unique universe names (not filename aliases)
        unique_names = set()
        for yaml_file in self._config_dir.glob("*.yaml"):
            try:
                config = self._load_yaml(yaml_file)
                if config and "name" in config:
                    unique_names.add(config["name"])
            except Exception:
                pass

        return sorted(unique_names)

    def get_universe_metadata(self, universe_name: str) -> Dict[str, Any]:
        """Get metadata for a universe.

        Returns information about the universe including description,
        size, tier, and other configuration details.

        Args:
            universe_name: Name of the universe (case-insensitive).

        Returns:
            Dictionary with metadata fields:
            - name: Universe name
            - description: Description text
            - size: Number of tickers (current)
            - tier: Tier level (if specified)
            - type: Universe type (static or graduated)
            - liquidity_profile: Liquidity profile (if specified)
            - expense_ratio_profile: Expense ratio profile (if specified)
            - eligibility: Eligibility criteria (if specified)

        Raises:
            ValueError: If universe not found.
        """
        config = self._get_universe_config(universe_name)

        # Get current ticker count
        tickers = self.get_universe(universe_name)

        source = config.get("source", {})
        source_type = source.get("type", "static_list")

        return {
            "name": config.get("name", universe_name),
            "description": config.get("description", ""),
            "size": len(tickers),
            "tier": config.get("tier"),
            "type": source_type,
            "liquidity_profile": config.get("liquidity_profile"),
            "expense_ratio_profile": config.get("expense_ratio_profile"),
            "eligibility": config.get("eligibility", {}),
            "notes": config.get("notes", ""),
        }

    def clear_cache(self) -> None:
        """Clear the universe configuration cache."""
        self._cache.clear()
        logger.debug("Universe configuration cache cleared")

    @property
    def config_dir(self) -> Path:
        """Return the configuration directory path."""
        return self._config_dir
