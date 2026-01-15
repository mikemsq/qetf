"""Macro data loader for regime detection.

Loads FRED data and provides regime signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class MacroDataLoader:
    """Load and process macroeconomic data.

    Attributes:
        data_dir: Directory containing macro parquet files
    """

    data_dir: Path = field(default_factory=lambda: Path("data/raw/macro"))

    def __post_init__(self) -> None:
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

    def load_indicator(self, name: str) -> pd.Series:
        """Load a single indicator.

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

    def get_vix(self, date: Optional[str] = None) -> float:
        """Get VIX value for date.

        Args:
            date: Optional date (returns latest if None)

        Returns:
            VIX value
        """
        vix = self.load_indicator("VIX")
        if date:
            vix = vix.loc[:date]
        return float(vix.iloc[-1])

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
