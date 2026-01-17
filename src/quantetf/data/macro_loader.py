"""Macro data loader for regime detection.

Loads FRED data and provides regime signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import numpy as np


class MacroIndicator(Enum):
    """Available macro indicators with FRED/data source mappings."""
    
    # Volatility
    VIX = "VIX"
    
    # Interest Rates
    TREASURY_3M = "DGS3MO"
    TREASURY_2Y = "DGS2"
    TREASURY_10Y = "DGS10"
    FED_FUNDS = "FED_FUNDS"
    
    # Yield Spreads
    YIELD_CURVE_10Y2Y = "T10Y2Y"
    YIELD_CURVE_10Y3M = "T10Y3M"
    
    # Credit Spreads
    HIGH_YIELD_SPREAD = "HY_SPREAD"
    INVESTMENT_GRADE_SPREAD = "IG_SPREAD"
    
    # Economic Indicators
    CPI = "CPI"
    UNEMPLOYMENT = "UNRATE"
    INDUSTRIAL_PRODUCTION = "INDPRO"
    
    # Market Reference
    SP500 = "SP500"
    
    # Legacy aliases for backward compatibility
    TREASURY_SPREAD_10Y2Y = "T10Y2Y"


@dataclass
class MacroDataLoader:
    """Load and process macroeconomic data.

    Provides point-in-time access to macro indicators for regime detection
    and conditioning alpha models.

    Attributes:
        data_dir: Directory containing macro parquet files
    """

    data_dir: Path = field(default_factory=lambda: Path("data/raw/macro"))
    _cache: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        self._cache = {}  # Cache loaded DataFrames

    # ==================== Core Data Loading ====================

    def _load_dataframe(self, indicator_value: str) -> Optional[pd.DataFrame]:
        """Load a single indicator DataFrame with caching.
        
        Args:
            indicator_value: Indicator enum value (e.g., 'VIX', 'DGS10')
            
        Returns:
            DataFrame with datetime index, or None if file not found
        """
        if indicator_value in self._cache:
            return self._cache[indicator_value]
        
        path = self.data_dir / f"{indicator_value}.parquet"
        if not path.exists():
            return None
        
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        self._cache[indicator_value] = df
        return df

    def load_indicator(self, name: str) -> pd.Series:
        """Load a single indicator (legacy method).

        Args:
            name: Indicator name (e.g., 'VIX', 'TREASURY_10Y')

        Returns:
            pandas Series with indicator values
        """
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Indicator {name} not found at {path}")

        df = pd.read_parquet(path)
        return df.iloc[:, 0]  # Return first column as Series

    def load_all(self) -> pd.DataFrame:
        """Load all available indicators.

        Returns:
            DataFrame with all indicators as columns
        """
        path = self.data_dir / "combined.parquet"
        if path.exists():
            return pd.read_parquet(path)

        # Combine individual files
        dfs = []
        for parquet_file in self.data_dir.glob("*.parquet"):
            if parquet_file.name in ["combined.parquet", "manifest.yaml"]:
                continue
            df = pd.read_parquet(parquet_file)
            df.columns = [parquet_file.stem]
            dfs.append(df)

        if not dfs:
            raise ValueError(f"No data found in {self.data_dir}")

        return pd.concat(dfs, axis=1).sort_index()

    # ==================== Point-in-Time Access ====================

    def get(
        self,
        indicator: MacroIndicator,
        as_of: pd.Timestamp,
        column: str = None,
    ) -> Optional[float]:
        """Get indicator value as-of a date.

        Args:
            indicator: Which indicator to retrieve
            as_of: Point-in-time date
            column: Column name (if None, uses first column)

        Returns:
            Most recent value on or before as_of, or None if no data
            
        Example:
            >>> loader = MacroDataLoader()
            >>> vix = loader.get(MacroIndicator.VIX, pd.Timestamp("2024-01-15"))
        """
        df = self._load_dataframe(indicator.value)
        if df is None:
            return None
        
        valid = df.loc[:as_of]
        if valid.empty:
            return None
        
        # Get specified column or first column
        col = column if column is not None else df.columns[0]
        return float(valid[col].iloc[-1])

    def get_series(
        self,
        indicator: MacroIndicator,
        start: pd.Timestamp,
        end: pd.Timestamp,
        column: str = None,
    ) -> pd.Series:
        """Get indicator time series for date range.

        Args:
            indicator: Which indicator
            start: Start date (inclusive)
            end: End date (inclusive)
            column: Column name (if None, uses first column)

        Returns:
            Series of values within date range
            
        Example:
            >>> loader = MacroDataLoader()
            >>> vix_range = loader.get_series(
            ...     MacroIndicator.VIX,
            ...     pd.Timestamp("2024-01-01"),
            ...     pd.Timestamp("2024-12-31")
            ... )
        """
        df = self._load_dataframe(indicator.value)
        if df is None:
            return pd.Series(dtype=float)
        
        mask = (df.index >= start) & (df.index <= end)
        col = column if column is not None else df.columns[0]
        return df.loc[mask, col]

    def get_lookback(
        self,
        indicator: MacroIndicator,
        as_of: pd.Timestamp,
        lookback_days: int,
        column: str = None,
    ) -> pd.Series:
        """Get indicator values for lookback window.

        Args:
            indicator: Which indicator
            as_of: End date of lookback window
            lookback_days: Number of calendar days to look back
            column: Column name (if None, uses first column)

        Returns:
            Series of values in lookback window
            
        Example:
            >>> loader = MacroDataLoader()
            >>> vix_20d = loader.get_lookback(
            ...     MacroIndicator.VIX,
            ...     pd.Timestamp("2024-01-15"),
            ...     lookback_days=20
            ... )
        """
        start = as_of - pd.Timedelta(days=lookback_days)
        return self.get_series(indicator, start, as_of, column)

    # ==================== Convenience Getters ====================

    def get_vix(self, as_of: Optional[pd.Timestamp] = None) -> Optional[float]:
        """Get VIX value as-of date.
        
        Args:
            as_of: Point-in-time date (uses latest if None)
            
        Returns:
            VIX value or None
        """
        if as_of is None:
            # Return latest value
            df = self._load_dataframe(MacroIndicator.VIX.value)
            if df is None:
                return None
            return float(df.iloc[-1, 0])
        return self.get(MacroIndicator.VIX, as_of)

    def get_yield_curve_spread(
        self,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Optional[float]:
        """Get 10Y-2Y Treasury spread as-of date.
        
        Args:
            as_of: Point-in-time date (uses latest if None)
            
        Returns:
            Spread in percentage points or None
        """
        if as_of is None:
            df = self._load_dataframe(MacroIndicator.YIELD_CURVE_10Y2Y.value)
            if df is None:
                return None
            return float(df.iloc[-1, 0])
        return self.get(MacroIndicator.YIELD_CURVE_10Y2Y, as_of)

    def get_credit_spread(
        self,
        as_of: Optional[pd.Timestamp] = None,
        high_yield: bool = True,
    ) -> Optional[float]:
        """Get credit spread as-of date.

        Args:
            as_of: Point-in-time date (uses latest if None)
            high_yield: If True, return HY spread; else IG spread

        Returns:
            Credit spread in basis points or None
        """
        indicator = (
            MacroIndicator.HIGH_YIELD_SPREAD
            if high_yield
            else MacroIndicator.INVESTMENT_GRADE_SPREAD
        )
        if as_of is None:
            df = self._load_dataframe(indicator.value)
            if df is None:
                return None
            return float(df.iloc[-1, 0])
        return self.get(indicator, as_of)

    def get_fed_funds(self, as_of: Optional[pd.Timestamp] = None) -> Optional[float]:
        """Get Federal Funds rate as-of date.
        
        Args:
            as_of: Point-in-time date (uses latest if None)
            
        Returns:
            Fed Funds rate or None
        """
        if as_of is None:
            df = self._load_dataframe(MacroIndicator.FED_FUNDS.value)
            if df is None:
                return None
            return float(df.iloc[-1, 0])
        return self.get(MacroIndicator.FED_FUNDS, as_of)

    def get_treasury_rate(
        self,
        as_of: Optional[pd.Timestamp] = None,
        maturity: str = "10Y",
    ) -> Optional[float]:
        """Get Treasury yield as-of date.

        Args:
            as_of: Point-in-time date (uses latest if None)
            maturity: "3M", "2Y", or "10Y"

        Returns:
            Treasury yield or None
            
        Raises:
            ValueError: If invalid maturity
        """
        indicator_map = {
            "3M": MacroIndicator.TREASURY_3M,
            "2Y": MacroIndicator.TREASURY_2Y,
            "10Y": MacroIndicator.TREASURY_10Y,
        }
        if maturity not in indicator_map:
            raise ValueError(f"Invalid maturity: {maturity}. Use 3M, 2Y, or 10Y")
        
        indicator = indicator_map[maturity]
        if as_of is None:
            df = self._load_dataframe(indicator.value)
            if df is None:
                return None
            return float(df.iloc[-1, 0])
        return self.get(indicator, as_of)

    def get_cpi(self, as_of: Optional[pd.Timestamp] = None) -> Optional[float]:
        """Get Consumer Price Index as-of date.
        
        Args:
            as_of: Point-in-time date (uses latest if None)
            
        Returns:
            CPI value or None
        """
        if as_of is None:
            df = self._load_dataframe(MacroIndicator.CPI.value)
            if df is None:
                return None
            return float(df.iloc[-1, 0])
        return self.get(MacroIndicator.CPI, as_of)

    def get_unemployment(
        self,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Optional[float]:
        """Get Unemployment rate as-of date.
        
        Args:
            as_of: Point-in-time date (uses latest if None)
            
        Returns:
            Unemployment rate (%) or None
        """
        if as_of is None:
            df = self._load_dataframe(MacroIndicator.UNEMPLOYMENT.value)
            if df is None:
                return None
            return float(df.iloc[-1, 0])
        return self.get(MacroIndicator.UNEMPLOYMENT, as_of)

    # ==================== Statistical Methods ====================

    def get_zscore(
        self,
        indicator: MacroIndicator,
        as_of: pd.Timestamp,
        lookback_days: int = 252,
    ) -> Optional[float]:
        """Get z-score of current value relative to lookback period.

        Args:
            indicator: Which indicator
            as_of: Point-in-time date
            lookback_days: Days for calculating mean/std

        Returns:
            Z-score of current value, or None if insufficient data
            
        Example:
            >>> loader = MacroDataLoader()
            >>> vix_zscore = loader.get_zscore(
            ...     MacroIndicator.VIX,
            ...     pd.Timestamp("2024-01-15"),
            ...     lookback_days=252
            ... )
        """
        series = self.get_lookback(indicator, as_of, lookback_days)
        
        if len(series) < 20:  # Minimum observations
            return None
        
        current = series.iloc[-1]
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return None
        
        return float((current - mean) / std)

    def get_percentile(
        self,
        indicator: MacroIndicator,
        as_of: pd.Timestamp,
        lookback_days: int = 252,
    ) -> Optional[float]:
        """Get percentile rank of current value in lookback period.

        Args:
            indicator: Which indicator
            as_of: Point-in-time date
            lookback_days: Days for calculating percentile

        Returns:
            Percentile (0-100) or None if insufficient data
        """
        series = self.get_lookback(indicator, as_of, lookback_days)
        
        if len(series) < 5:
            return None
        
        current = series.iloc[-1]
        percentile = (series <= current).sum() / len(series) * 100
        return float(percentile)

    # ==================== Regime Signal Helpers ====================

    def is_high_vol_regime(
        self,
        as_of: Optional[pd.Timestamp] = None,
        threshold: float = 25.0,
    ) -> bool:
        """Check if VIX indicates high volatility.

        Args:
            as_of: Date to check (uses latest if None)
            threshold: VIX threshold (default 25)

        Returns:
            True if VIX > threshold
        """
        vix = self.get_vix(as_of)
        return vix is not None and vix > threshold

    def is_yield_curve_inverted(self, as_of: Optional[pd.Timestamp] = None) -> bool:
        """Check if yield curve is inverted (recession signal).

        Args:
            as_of: Date to check (uses latest if None)

        Returns:
            True if 10Y-2Y spread is negative
        """
        spread = self.get_yield_curve_spread(as_of)
        return spread is not None and spread < 0

    def get_macro_snapshot(self, as_of: pd.Timestamp) -> Dict[str, Optional[float]]:
        """Get snapshot of all major macro indicators as-of date.
        
        Args:
            as_of: Point-in-time date
            
        Returns:
            Dictionary of indicator_name -> value
        """
        return {
            "vix": self.get_vix(as_of),
            "yield_curve_10y2y": self.get_yield_curve_spread(as_of),
            "treasury_10y": self.get_treasury_rate(as_of, "10Y"),
            "treasury_2y": self.get_treasury_rate(as_of, "2Y"),
            "fed_funds": self.get_fed_funds(as_of),
            "hy_spread": self.get_credit_spread(as_of, high_yield=True),
            "ig_spread": self.get_credit_spread(as_of, high_yield=False),
            "cpi": self.get_cpi(as_of),
            "unemployment": self.get_unemployment(as_of),
        }

    def get_available_indicators(self) -> List[MacroIndicator]:
        """Return list of indicators with available data.
        
        Returns:
            List of MacroIndicator enums with loaded data
        """
        available = []
        for indicator in MacroIndicator:
            if self._load_dataframe(indicator.value) is not None:
                available.append(indicator)
    def get_yield_curve_spread(self, date: Optional[str] = None) -> float:
        """Get 10Y-2Y Treasury spread.

        Args:
            date: Optional date

        Returns:
            Spread in percentage points
        """
        spread = self.load_indicator("TREASURY_SPREAD_10Y2Y")
        if date:
            spread = spread.loc[:date]
        return float(spread.iloc[-1])

    def is_high_vol_regime(
        self,
        date: Optional[str] = None,
        threshold: float = 25.0,
    ) -> bool:
        """Check if VIX indicates high volatility.

        Args:
            date: Date to check
            threshold: VIX threshold (default 25)

        Returns:
            True if VIX > threshold
        """
        return self.get_vix(date) > threshold

    def is_yield_curve_inverted(self, date: Optional[str] = None) -> bool:
        """Check if yield curve is inverted (recession signal).

        Args:
            date: Date to check

        Returns:
            True if 10Y-2Y spread is negative
        """
        return self.get_yield_curve_spread(date) < 0


class RegimeDetector:
    """Detect market regimes using macro data.

    Combines multiple signals to classify regime.
    """

    def __init__(self, macro_loader: MacroDataLoader) -> None:
        self.macro = macro_loader

    def detect_regime(self, date: str) -> str:
        """Detect regime for a given date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Regime string: 'RISK_ON', 'RISK_OFF', 'HIGH_VOL', 'RECESSION_WARNING'
        """
        try:
            vix = self.macro.get_vix(date)
            spread = self.macro.get_yield_curve_spread(date)
        except Exception:
            return "UNKNOWN"

        # Inverted yield curve = recession warning
        if spread < 0:
            return "RECESSION_WARNING"

        # High VIX = high volatility regime
        if vix > 30:
            return "HIGH_VOL"
        elif vix > 20:
            return "ELEVATED_VOL"

        # Normal conditions
        return "RISK_ON"
