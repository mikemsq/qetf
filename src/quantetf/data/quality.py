from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityIssue:
    severity: str  # "info", "warning", "error"
    message: str
    context: dict


def check_missingness(df: pd.DataFrame, *, max_missing_frac: float = 0.05) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    missing = df.isna().mean()
    bad = missing[missing > max_missing_frac]
    for col, frac in bad.items():
        issues.append(
            DataQualityIssue(
                severity="warning",
                message="Column exceeds missingness threshold",
                context={"column": col, "missing_frac": float(frac), "threshold": max_missing_frac},
            )
        )
    return issues


def winsorize_returns(r: pd.DataFrame, *, lower_q: float = 0.001, upper_q: float = 0.999) -> pd.DataFrame:
    """Optional helper for robust statistics; do not use to hide data issues."""
    lo = r.quantile(lower_q)
    hi = r.quantile(upper_q)
    return r.clip(lower=lo, upper=hi, axis=1)
