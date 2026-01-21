"""Static reference data accessor implementation."""

from pathlib import Path
from typing import Union, Optional
import yaml
import logging

from .abstract import ReferenceDataAccessor
from .types import TickerMetadata, ExchangeInfo


logger = logging.getLogger(__name__)


class StaticReferenceDataAccessor(ReferenceDataAccessor):
    """Reference data accessor that reads from YAML config files.

    Provides access to static/slow-changing reference data including:
    - Ticker metadata (name, sector, exchange, currency)
    - Sector mappings
    - Exchange information

    All data is cached in memory on first access for performance.

    Usage:
        accessor = StaticReferenceDataAccessor(
            config_dir=Path("configs/reference")
        )

        # Get ticker info
        spy_info = accessor.get_ticker_info("SPY")
        print(spy_info.sector)  # "Broad Market"

        # Get all sectors
        sectors = accessor.get_sectors()

        # Get tickers by sector
        tech_tickers = accessor.get_tickers_by_sector("Technology")
    """

    def __init__(self, config_dir: Union[str, Path]):
        """Initialize reference data accessor.

        Args:
            config_dir: Directory containing reference YAML files.
                       Expected files:
                       - tickers.yaml: Ticker metadata
                       - exchanges.yaml: Exchange information
        """
        self._config_dir = Path(config_dir)

        if not self._config_dir.exists():
            raise ValueError(f"Config directory does not exist: {self._config_dir}")

        # Lazy-loaded caches
        self._tickers_cache: Optional[dict[str, TickerMetadata]] = None
        self._exchanges_cache: Optional[dict[str, ExchangeInfo]] = None
        self._sector_mapping_cache: Optional[dict[str, str]] = None

        logger.info(f"Initialized StaticReferenceDataAccessor with config_dir={config_dir}")

    def _load_tickers(self) -> dict[str, TickerMetadata]:
        """Load and cache ticker metadata from YAML file."""
        if self._tickers_cache is not None:
            return self._tickers_cache

        tickers_file = self._config_dir / "tickers.yaml"
        if not tickers_file.exists():
            raise FileNotFoundError(
                f"Tickers config file not found: {tickers_file}"
            )

        with open(tickers_file) as f:
            data = yaml.safe_load(f)

        if "tickers" not in data:
            raise ValueError(
                f"Invalid tickers.yaml format: missing 'tickers' key"
            )

        self._tickers_cache = {}
        for ticker, info in data["tickers"].items():
            self._tickers_cache[ticker.upper()] = TickerMetadata(
                ticker=ticker.upper(),
                name=info.get("name", ticker),
                sector=info.get("sector", "Unknown"),
                exchange=info.get("exchange", "Unknown"),
                currency=info.get("currency", "USD"),
            )

        logger.debug(f"Loaded {len(self._tickers_cache)} tickers from {tickers_file}")
        return self._tickers_cache

    def _load_exchanges(self) -> dict[str, ExchangeInfo]:
        """Load and cache exchange information from YAML file."""
        if self._exchanges_cache is not None:
            return self._exchanges_cache

        exchanges_file = self._config_dir / "exchanges.yaml"
        if not exchanges_file.exists():
            raise FileNotFoundError(
                f"Exchanges config file not found: {exchanges_file}"
            )

        with open(exchanges_file) as f:
            data = yaml.safe_load(f)

        if "exchanges" not in data:
            raise ValueError(
                f"Invalid exchanges.yaml format: missing 'exchanges' key"
            )

        self._exchanges_cache = {}
        for exchange_code, info in data["exchanges"].items():
            self._exchanges_cache[exchange_code] = ExchangeInfo(
                name=info.get("name", exchange_code),
                trading_hours=info.get("trading_hours", "09:30-16:00"),
                timezone=info.get("timezone", "US/Eastern"),
                settlement_days=info.get("settlement_days", 2),
            )

        logger.debug(f"Loaded {len(self._exchanges_cache)} exchanges from {exchanges_file}")
        return self._exchanges_cache

    def _build_sector_mapping(self) -> dict[str, str]:
        """Build and cache ticker → sector mapping."""
        if self._sector_mapping_cache is not None:
            return self._sector_mapping_cache

        tickers = self._load_tickers()
        self._sector_mapping_cache = {
            ticker: meta.sector for ticker, meta in tickers.items()
        }
        return self._sector_mapping_cache

    def get_ticker_info(self, ticker: str) -> TickerMetadata:
        """Get metadata for a ticker.

        Args:
            ticker: Ticker symbol (case-insensitive)

        Returns:
            TickerMetadata with ticker information

        Raises:
            ValueError: If ticker not found in reference data
        """
        tickers = self._load_tickers()
        ticker_upper = ticker.upper()

        if ticker_upper not in tickers:
            raise ValueError(
                f"Ticker '{ticker}' not found in reference data. "
                f"Available tickers: {len(tickers)} total"
            )

        return tickers[ticker_upper]

    def get_sector_mapping(self) -> dict[str, str]:
        """Return ticker → sector mapping for all tickers.

        Returns:
            Dictionary mapping ticker symbols to sector names
        """
        return self._build_sector_mapping().copy()

    def get_exchange_info(self) -> dict[str, ExchangeInfo]:
        """Return exchange → metadata mapping.

        Returns:
            Dictionary mapping exchange codes to ExchangeInfo objects
        """
        return self._load_exchanges().copy()

    def get_sectors(self) -> list[str]:
        """Return list of unique sector names.

        Returns:
            Sorted list of unique sector names
        """
        sector_mapping = self._build_sector_mapping()
        return sorted(set(sector_mapping.values()))

    def get_tickers_by_sector(self, sector: str) -> list[str]:
        """Return all tickers in a given sector.

        Args:
            sector: Sector name (case-sensitive)

        Returns:
            Sorted list of tickers in the sector

        Raises:
            ValueError: If sector not found
        """
        sector_mapping = self._build_sector_mapping()

        tickers_in_sector = [
            ticker for ticker, s in sector_mapping.items()
            if s == sector
        ]

        if not tickers_in_sector:
            available_sectors = self.get_sectors()
            raise ValueError(
                f"Sector '{sector}' not found. "
                f"Available sectors: {available_sectors}"
            )

        return sorted(tickers_in_sector)

    def get_available_tickers(self) -> list[str]:
        """Return list of all available tickers.

        Returns:
            Sorted list of all ticker symbols
        """
        tickers = self._load_tickers()
        return sorted(tickers.keys())

    def clear_cache(self) -> None:
        """Clear all cached data.

        Useful for refreshing data after config files are updated.
        """
        self._tickers_cache = None
        self._exchanges_cache = None
        self._sector_mapping_cache = None
        logger.debug("Cleared reference data cache")
