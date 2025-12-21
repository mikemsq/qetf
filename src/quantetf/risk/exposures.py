from __future__ import annotations

import pandas as pd


def one_hot_exposures(instrument_master: pd.DataFrame, *, column: str) -> pd.DataFrame:
    """Create one-hot exposures for a categorical column (sector, country, etc.)."""
    df = instrument_master.copy()
    if "ticker" in df.columns:
        df = df.set_index("ticker")
    if column not in df.columns:
        raise KeyError(f"Missing column '{column}' in instrument_master")
    return pd.get_dummies(df[column].astype("string"), prefix=column)
