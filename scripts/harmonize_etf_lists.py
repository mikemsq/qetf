#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9_]+", "_", str(c).strip().lower()).strip("_") for c in df.columns
    ]
    return df


def _clean_ticker(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
        .replace({"NAN": pd.NA, "NONE": pd.NA, "": pd.NA})
    )


def _parse_expense_ratio_pct(series: pd.Series) -> pd.Series:
    # Returns numeric percent values, e.g. "0.35%" -> 0.35
    s = series.astype(str).str.strip()
    s = s.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    def _one(x: str):
        if x is None or x is pd.NA:
            return pd.NA
        x = str(x).strip()
        if x == "" or x.lower() in {"nan", "none"}:
            return pd.NA
        x = x.replace(",", "")
        if x.endswith("%"):
            x = x[:-1].strip()
        try:
            return float(x)
        except ValueError:
            return pd.NA

    return s.map(_one)


def _infer_issuer_family(provider: str | None, name: str | None) -> str | None:
    p = (provider or "").strip().lower()
    n = (name or "").strip().lower()

    if "ishares" in p or n.startswith("ishares "):
        return "iShares"
    if "vanguard" in p or n.startswith("vanguard "):
        return "Vanguard"
    if "state street" in p or "spdr" in p or n.startswith("spdr "):
        return "SPDR"
    if "invesco" in p or "powershares" in p:
        return "Invesco"
    if "schwab" in p:
        return "Schwab"
    if "fidelity" in p:
        return "Fidelity"
    if "wisdomtree" in p:
        return "WisdomTree"
    if "proshares" in p:
        return "ProShares"
    if "van eck" in p or "vaneck" in p:
        return "VanEck"
    if "global x" in p:
        return "Global X"

    # Fallback: if we at least have a provider string, keep it as the "family".
    if provider and provider.strip():
        return provider.strip()

    return None


def _read_etf_list(path: Path) -> pd.DataFrame:
    # etf_list.csv has many trailing empty columns; pandas will create unnamed columns.
    df = pd.read_csv(path, dtype=str, engine="python")
    df = _normalize_columns(df)

    # Drop unnamed columns that are fully empty.
    unnamed = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Some columns can be entirely empty; drop them.
    df = df.dropna(axis=1, how="all")

    out = pd.DataFrame(
        {
            "ticker": _clean_ticker(df.get("symbol")),
            "name": df.get("name"),
            "provider": df.get("provider"),
            "inception_date": df.get("inception"),
            "expense_ratio_pct": _parse_expense_ratio_pct(df.get("expense_ratio")),
            "asset_class": df.get("asset_class"),
            "region": df.get("region"),
            "category": df.get("category"),
            "structure": df.get("structure"),
            "source_etf_list": True,
        }
    )

    # Clean obvious placeholders.
    out["name"] = out["name"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    out["provider"] = out["provider"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    out["asset_class"] = out["asset_class"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    out["region"] = out["region"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    out["category"] = out["category"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    out["structure"] = out["structure"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="first")
    return out


def _read_ishares_list(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = _normalize_columns(df)

    out = pd.DataFrame(
        {
            "ticker": _clean_ticker(df.get("ticker")),
            "name": df.get("name"),
            "provider": "iShares",
            "inception_date": df.get("inception_date"),
            "expense_ratio_pct": _parse_expense_ratio_pct(df.get("expense_ratio")),
            "asset_class": df.get("asset_class"),
            "sub_asset_class": df.get("sub_asset_class"),
            "region": df.get("region"),
            "market": df.get("market"),
            "location": df.get("location"),
            "investment_style": df.get("investment_style"),
            "isin": df.get("isin"),
            "cusip": df.get("cusip"),
            "source_ishares_list": True,
        }
    )

    out["name"] = out["name"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="first")
    return out


def harmonize(etf_list_path: Path, ishares_list_path: Path) -> pd.DataFrame:
    etf = _read_etf_list(etf_list_path)
    ish = _read_ishares_list(ishares_list_path)

    merged = etf.merge(ish, on="ticker", how="outer", suffixes=("_etf", "_ish"))

    def choose(col: str) -> pd.Series:
        c_ish = f"{col}_ish" if f"{col}_ish" in merged.columns else None
        c_etf = f"{col}_etf" if f"{col}_etf" in merged.columns else None

        if c_ish and c_etf:
            return merged[c_ish].combine_first(merged[c_etf])
        if c_ish:
            return merged[c_ish]
        if c_etf:
            return merged[c_etf]
        return pd.Series([pd.NA] * len(merged))

    out = pd.DataFrame(
        {
            "ticker": merged["ticker"],
            "name": choose("name"),
            "provider": choose("provider"),
            "issuer_family": None,
            "inception_date": choose("inception_date"),
            "expense_ratio_pct": choose("expense_ratio_pct"),
            "asset_class": choose("asset_class"),
            "sub_asset_class": choose("sub_asset_class"),
            "region": choose("region"),
            "market": choose("market"),
            "location": choose("location"),
            "investment_style": choose("investment_style"),
            "category": choose("category"),
            "structure": choose("structure"),
            "isin": choose("isin"),
            "cusip": choose("cusip"),
            "source": None,
        }
    )

    out["issuer_family"] = [
        _infer_issuer_family(provider=p, name=n) for p, n in zip(out["provider"], out["name"])
    ]

    src_etf = merged.get("source_etf_list")
    src_ish = merged.get("source_ishares_list")

    # These source columns are True where present and NaN where absent.
    # Using `.notna()` avoids pandas' future downcasting warning for object-dtype `.fillna(False)`.
    src_etf_flags = src_etf.notna() if src_etf is not None else pd.Series([False] * len(merged))
    src_ish_flags = src_ish.notna() if src_ish is not None else pd.Series([False] * len(merged))

    def _src(a, b):
        a = bool(a) if a is not None and a is not pd.NA else False
        b = bool(b) if b is not None and b is not pd.NA else False
        if a and b:
            return "both"
        if b:
            return "ishares_list"
        if a:
            return "etf_list"
        return "unknown"

    out["source"] = [_src(a, b) for a, b in zip(src_etf_flags, src_ish_flags)]

    # Final cleanup: stable sort and drop empty tickers.
    out = out.dropna(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)

    # Coerce expense ratio to numeric where possible.
    out["expense_ratio_pct"] = pd.to_numeric(out["expense_ratio_pct"], errors="coerce")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Harmonize ETF lists from data/raw into one CSV.")
    parser.add_argument(
        "--etf-list",
        type=Path,
        default=Path("data/raw/etf_list.csv"),
        help="Path to etf_list.csv",
    )
    parser.add_argument(
        "--ishares-list",
        type=Path,
        default=Path("data/raw/ishares_list.csv"),
        help="Path to ishares_list.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/curated/etf_universe.csv"),
        help="Output CSV path",
    )

    args = parser.parse_args()

    df = harmonize(args.etf_list, args.ishares_list)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Small console summary
    total = len(df)
    families = (
        df["issuer_family"].fillna("(unknown)").value_counts().head(15).to_dict()
        if total
        else {}
    )
    both = int((df["source"] == "both").sum())

    print(f"Wrote {total} ETFs to {args.out}")
    print(f"Tickers present in both lists: {both}")
    if families:
        print("Top issuer_family counts (top 15):")
        for k, v in families.items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
