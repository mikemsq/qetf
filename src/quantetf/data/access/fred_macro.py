"""FREDMacroAccessor - macro data access with regime detection."""

from typing import Optional
import pandas as pd

from .abstract import MacroDataAccessor
from .types import Regime


class FREDMacroAccessor(MacroDataAccessor):
    """Access macro data and detect market regimes.
    
    Wraps existing MacroDataLoader to provide clean DAL interface.
    Adds regime detection logic based on macro indicators.
    
    Usage:
        from quantetf.data.macro_loader import MacroDataLoader
        
        loader = MacroDataLoader()
        accessor = FREDMacroAccessor(loader)
        
        # Get macro indicator
        vix = accessor.read_macro_indicator("VIX", pd.Timestamp("2024-01-31"))
        
        # Get regime
        regime = accessor.get_regime(pd.Timestamp("2024-01-31"))
    """
    
    def __init__(self, macro_loader):
        """Initialize with MacroDataLoader instance.
        
        Args:
            macro_loader: Existing MacroDataLoader from quantetf.data.macro_loader
        """
        self.macro_loader = macro_loader
    
    def read_macro_indicator(
        self,
        indicator: str,
        as_of: pd.Timestamp,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return time series for a macro indicator.
        
        Returns data up to as_of (point-in-time).
        
        Args:
            indicator: Indicator name (e.g., "VIX", "T10Y2Y", "UNRATE")
            as_of: Cutoff date (inclusive)
            lookback_days: Optional lookback window
            
        Returns:
            DataFrame with DatetimeIndex and indicator values
            
        Raises:
            ValueError: If indicator not available
        """
        from quantetf.data.macro_loader import MacroIndicator
        
        try:
            # Try to get the indicator as an enum member
            indicator_enum = MacroIndicator[indicator] if indicator in MacroIndicator.__members__ else None
        except (KeyError, AttributeError):
            indicator_enum = None
        
        # If not found as enum member, try matching by value
        if indicator_enum is None:
            for member in MacroIndicator:
                if member.value == indicator:
                    indicator_enum = member
                    break
        
        if indicator_enum is None:
            raise ValueError(f"Indicator not found: {indicator}")
        
        # Get the data using lookback or date range
        if lookback_days is not None:
            series = self.macro_loader.get_lookback(
                indicator_enum,
                as_of,
                lookback_days
            )
        else:
            # Get all data up to as_of
            # First load the full indicator to get start date
            df = self.macro_loader._load_dataframe(indicator_enum.value)
            if df is None:
                raise ValueError(f"No data available for indicator: {indicator}")
            
            start = df.index.min()
            series = self.macro_loader.get_series(indicator_enum, start, as_of)
        
        if series.empty:
            raise ValueError(f"No data available for indicator {indicator} before {as_of}")
        
        # Convert series to DataFrame for consistency
        result = pd.DataFrame(series)
        result.columns = [indicator]
        return result
    
    def get_regime(self, as_of: pd.Timestamp) -> Regime:
        """Detect current market regime as of date.
        
        Regime logic:
        1. If VIX > 30 AND YIELD_SPREAD < -0.5: RECESSION_WARNING
        2. Else if VIX > 30: HIGH_VOL
        3. Else if VIX > 20: ELEVATED_VOL
        4. Else if YIELD_SPREAD < 0: ELEVATED_VOL
        5. Else: RISK_ON
        6. If any indicator missing: UNKNOWN
        
        Args:
            as_of: Date for regime detection
            
        Returns:
            One of RISK_ON, ELEVATED_VOL, HIGH_VOL, RECESSION_WARNING, UNKNOWN
        """
        try:
            # Read VIX and yield curve spread
            vix_data = self.read_macro_indicator("VIX", as_of, lookback_days=1)
            spread_data = self.read_macro_indicator("T10Y2Y", as_of, lookback_days=1)
            
            # Get latest values
            vix = float(vix_data.iloc[-1, 0]) if not vix_data.empty else None
            spread = float(spread_data.iloc[-1, 0]) if not spread_data.empty else None
            
            if vix is None or spread is None:
                return Regime.UNKNOWN
            
            # Apply regime logic
            if vix > 30 and spread < -0.5:
                return Regime.RECESSION_WARNING
            elif vix > 30:
                return Regime.HIGH_VOL
            elif vix > 20:
                return Regime.ELEVATED_VOL
            elif spread < 0:
                return Regime.ELEVATED_VOL
            else:
                return Regime.RISK_ON
        except Exception:
            return Regime.UNKNOWN
    
    def get_available_indicators(self) -> list[str]:
        """Return list of available macro indicators.
        
        Returns:
            List of indicator names (enum values)
        """
        from quantetf.data.macro_loader import MacroIndicator
        
        # Return all indicator enum values
        return [member.value for member in MacroIndicator]
