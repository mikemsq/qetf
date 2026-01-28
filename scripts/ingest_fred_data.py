#!/usr/bin/env python3
"""Ingest macroeconomic data from FRED.

Downloads key economic indicators for regime detection and macro analysis.

Usage:
    python scripts/ingest_fred_data.py --start-date 2015-01-01
    python scripts/ingest_fred_data.py --api-key YOUR_KEY --indicators VIX,DGS10

Environment:
    FRED_API_KEY: Your FRED API key (get free at https://fred.stlouisfed.org/docs/api/api_key.html)
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml


# Default indicators to download
DEFAULT_INDICATORS: Dict[str, str] = {
    # Volatility
    "VIX": "VIXCLS",
    # Interest Rates
    "TREASURY_10Y": "DGS10",
    "TREASURY_2Y": "DGS2",
    "TREASURY_3M": "DTB3",
    "TREASURY_SPREAD_10Y2Y": "T10Y2Y",
    "FED_FUNDS": "FEDFUNDS",
    # Credit
    "HY_SPREAD": "BAMLH0A0HYM2",  # ICE BofA High Yield Spread
    "IG_SPREAD": "BAMLC0A0CM",  # Investment Grade Spread
    # Economic
    "UNEMPLOYMENT": "UNRATE",
    "CPI": "CPIAUCSL",
    "INDUSTRIAL_PROD": "INDPRO",
    # Market
    "SP500": "SP500",  # S&P 500 index
}


def get_fred_api_key() -> str:
    """Get FRED API key from environment or prompt."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED_API_KEY not found in environment. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set FRED_API_KEY environment variable."
        )
    return key


def download_fred_series(
    api_key: str,
    series_id: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.Series:
    """Download a single FRED series.

    Args:
        api_key: FRED API key
        series_id: FRED series code (e.g., 'VIXCLS')
        start_date: Start date (YYYY-MM-DD)
        end_date: Optional end date

    Returns:
        pandas Series with data
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Please install fredapi: pip install fredapi")

    fred = Fred(api_key=api_key)

    kwargs = {"observation_start": start_date}
    if end_date:
        kwargs["observation_end"] = end_date

    data = fred.get_series(series_id, **kwargs)
    data.name = series_id

    return data


def download_all_indicators(
    api_key: str,
    indicators: Dict[str, str],
    start_date: str,
    end_date: Optional[str] = None,
    output_dir: Path = Path("data/raw/macro"),
) -> Dict[str, pd.Series]:
    """Download all specified indicators.

    Args:
        api_key: FRED API key
        indicators: Dict mapping name -> FRED code
        start_date: Start date
        end_date: Optional end date
        output_dir: Directory to save files

    Returns:
        Dict of downloaded series
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    failed = []

    for name, code in indicators.items():
        print(f"Downloading {name} ({code})...", end=" ")
        try:
            data = download_fred_series(api_key, code, start_date, end_date)
            results[name] = data

            # Save to parquet
            df = data.to_frame()
            df.to_parquet(output_dir / f"{name}.parquet")
            print(f"OK ({len(data)} observations)")

        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(name)

    # Save metadata
    metadata = {
        "download_date": datetime.now().isoformat(),
        "start_date": start_date,
        "end_date": end_date or "latest",
        "indicators": {name: code for name, code in indicators.items()},
        "successful": list(results.keys()),
        "failed": failed,
    }

    with open(output_dir / "macro.metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"\nDownloaded {len(results)}/{len(indicators)} indicators")
    print(f"Saved to {output_dir}")

    return results


def create_combined_macro_dataset(
    data_dir: Path = Path("data/raw/macro"),
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Combine all macro indicators into single DataFrame.

    Args:
        data_dir: Directory with individual parquet files
        output_path: Optional path to save combined file

    Returns:
        Combined DataFrame with all indicators
    """
    dfs = []
    for parquet_file in data_dir.glob("*.parquet"):
        if parquet_file.name == "macro.parquet":
            continue
        df = pd.read_parquet(parquet_file)
        if len(df.columns) == 1:
            df.columns = [parquet_file.stem]
        else:
            df.columns = [f"{parquet_file.stem}_{col}" for col in df.columns]
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No parquet files found in {data_dir}")

    combined = pd.concat(dfs, axis=1)
    combined = combined.sort_index()

    if output_dir:
        combined.to_parquet(output_dir / "macro.parquet")
        src = data_dir / "macro.metadata.yaml"
        dst = output_dir / "macro.metadata.yaml"
        shutil.copy2(src, dst)
        print(f"Combined dataset saved to {output_dir / 'macro.parquet'}")

    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download FRED economic indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all default indicators
    python scripts/ingest_fred_data.py --start-date 2015-01-01

    # Download specific indicators
    python scripts/ingest_fred_data.py --start-date 2015-01-01 --indicators VIX,DGS10,T10Y2Y

    # Use specific API key
    FRED_API_KEY=your_key python scripts/ingest_fred_data.py --start-date 2015-01-01
        """,
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), default is latest",
    )
    parser.add_argument(
        "--indicators",
        type=str,
        default=None,
        help="Comma-separated list of indicator names to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/macro",
        help="Output directory",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="FRED API key (or set FRED_API_KEY env var)",
    )
    parser.add_argument(
        "--list-indicators",
        action="store_true",
        help="List available indicators and exit",
    )

    args = parser.parse_args()

    if args.list_indicators:
        print("Available indicators:")
        print("-" * 50)
        for name, code in DEFAULT_INDICATORS.items():
            print(f"  {name:25s} -> {code}")
        return 0

    # Get API key
    api_key = args.api_key or get_fred_api_key()

    # Select indicators
    if args.indicators:
        selected = args.indicators.split(",")
        indicators = {k: v for k, v in DEFAULT_INDICATORS.items() if k in selected}
        if not indicators:
            print(f"Warning: None of {selected} found in default indicators")
            # Try treating as FRED codes directly
            indicators = {s: s for s in selected}
    else:
        indicators = DEFAULT_INDICATORS

    # Download
    output_dir = Path(args.output_dir)
    results = download_all_indicators(
        api_key=api_key,
        indicators=indicators,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=output_dir,
    )

    # Optionally combine
    if results:
        create_combined_macro_dataset(
            data_dir=output_dir,
            output_dir=Path("data/snapshots"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
