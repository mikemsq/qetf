"""Data Access Layer type definitions and enums."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd


class Regime(Enum):
    """Market regime classification.
    
    Represents different market conditions for strategic decision-making.
    """
    RISK_ON = "risk_on"
    ELEVATED_VOL = "elevated_vol"
    HIGH_VOL = "high_vol"
    RECESSION_WARNING = "recession_warning"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TickerMetadata:
    """Metadata about a ticker/ETF.
    
    Immutable container for static ticker information.
    """
    ticker: str
    name: str
    sector: str
    exchange: str
    currency: str


@dataclass(frozen=True)
class ExchangeInfo:
    """Metadata about an exchange.
    
    Contains trading hours, timezone, and settlement information.
    """
    name: str
    trading_hours: str  # e.g., "09:30-16:00 EST"
    timezone: str  # e.g., "US/Eastern"
    settlement_days: int  # Usually 2 or 3


@dataclass(frozen=True)
class DataAccessMetadata:
    """Metadata about data from an accessor.
    
    Provides information about data source, freshness, and quality.
    """
    source: str
    timestamp: pd.Timestamp
    lookback_date: Optional[pd.Timestamp]
    data_quality_score: float  # 0.0 to 1.0
