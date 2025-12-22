#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import pandas as pd


STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
DEFAULT_START = "1990-01-01"


@dataclass(frozen=True)
class DownloadStats:
    tickers_total: int
    tickers_ok: int
    tickers_no_data: int
    tickers_failed: int
    rows_written: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_tickers(tickers: Iterable[str]) -> list[str]:
    out: list[str] = []
    for t in tickers:
        t = str(t).strip().upper()
        if not t or t in {"NAN", "NONE"}:
            continue
        out.append(t)
    # stable de-dupe
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _read_universe_tickers(universe_csv: Path) -> list[str]:
    df = pd.read_csv(universe_csv, dtype=str)
    if "ticker" not in df.columns:
        raise ValueError(f"Expected a 'ticker' column in {universe_csv}")
    return _clean_tickers(df["ticker"].tolist())


def _normalize_stooq_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Expected Stooq columns: Date, Open, High, Low, Close, Volume
    if df.empty:
        return df

    cols = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    if not all(k in cols for k in required):
        # Sometimes stooq returns HTML or different columns on error.
        return pd.DataFrame()

    out = df[[cols[k] for k in required]].copy()
    out.columns = required

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        # Stooq may include commas or other formatting.
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out


def _read_existing_last_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        existing = pd.read_parquet(path)
    except Exception:
        return None
    if existing.empty or "date" not in existing.columns:
        return None
    d = pd.to_datetime(existing["date"], errors="coerce").dropna()
    if d.empty:
        return None
    return pd.Timestamp(d.max())


def _append_parquet(path: Path, new_df: pd.DataFrame) -> int:
    if new_df.empty:
        return 0
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        combined = new_df

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return len(new_df)


def _fetch_csv_from_url(url: str, *, timeout: float) -> pd.DataFrame:
    with urlopen(url, timeout=timeout) as resp:
        # Stooq returns plain CSV text.
        text = resp.read().decode("utf-8", errors="replace")
    return pd.read_csv(StringIO(text))


def download_ticker(
    *,
    ticker: str,
    suffix: str,
    out_dir: Path,
    start: pd.Timestamp | None,
    timeout: float,
) -> tuple[str, int, str]:
    """Returns (ticker, rows_written, status). status in {'ok','no_data','failed'}."""

    symbol = f"{ticker.lower()}{suffix}"
    url = STOOQ_DAILY_URL.format(symbol=symbol)

    out_path = out_dir / f"{ticker}.parquet"

    last_date = _read_existing_last_date(out_path)

    try:
        raw = _fetch_csv_from_url(url, timeout=timeout)
    except Exception:
        return ticker, 0, "failed"

    df = _normalize_stooq_frame(raw)
    if df.empty:
        return ticker, 0, "no_data"

    if start is not None:
        df = df[df["date"] >= start]

    if last_date is not None:
        df = df[df["date"] > last_date]

    rows_written = _append_parquet(out_path, df)
    return ticker, rows_written, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Free daily OHLCV downloader for ETFs via Stooq.")
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("data/curated/etf_universe.csv"),
        help="CSV with a 'ticker' column (default: data/curated/etf_universe.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/prices/stooq"),
        help="Output directory for per-ticker parquet files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/prices/stooq/manifest.json"),
        help="Path to write a small JSON manifest (includes last_full_download_at)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".us",
        help="Stooq symbol suffix (default: .us). For non-US you may need other suffixes.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START,
        help="Start date YYYY-MM-DD (default: 1990-01-01)",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional limit for testing",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Sleep seconds between requests (politeness / throttling)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout seconds per ticker (default: 20)",
    )

    args = parser.parse_args()

    start_ts = pd.Timestamp(args.start)

    tickers = _read_universe_tickers(args.universe)
    if args.max_tickers is not None:
        tickers = tickers[: args.max_tickers]

    ok = no_data = failed = 0
    rows_written = 0

    for i, t in enumerate(tickers, start=1):
        ticker, wrote, status = download_ticker(
            ticker=t,
            suffix=args.suffix,
            out_dir=args.out_dir,
            start=start_ts,
            timeout=args.timeout,
        )

        rows_written += wrote
        if status == "ok":
            ok += 1
        elif status == "no_data":
            no_data += 1
        else:
            failed += 1

        if i % 50 == 0:
            print(
                f"Progress {i}/{len(tickers)} | ok={ok} no_data={no_data} failed={failed} | rows_written={rows_written}"
            )

        time.sleep(max(args.sleep, 0.0))

    stats = DownloadStats(
        tickers_total=len(tickers),
        tickers_ok=ok,
        tickers_no_data=no_data,
        tickers_failed=failed,
        rows_written=rows_written,
    )

    # Write/update manifest.
    manifest = _load_json(args.manifest)
    run = {
        "run_at": _utc_now_iso(),
        "universe": str(args.universe),
        "out_dir": str(args.out_dir),
        "suffix": args.suffix,
        "start": str(start_ts.date()),
        "max_tickers": args.max_tickers,
        "tickers_total": stats.tickers_total,
        "tickers_ok": stats.tickers_ok,
        "tickers_no_data": stats.tickers_no_data,
        "tickers_failed": stats.tickers_failed,
        "rows_written": stats.rows_written,
    }
    manifest["last_run"] = run

    # Only set last_full_download_at when we attempted the full universe (no max_tickers limit)
    # and the run completed without hard failures.
    if args.max_tickers is None and stats.tickers_failed == 0:
        manifest["last_full_download_at"] = run["run_at"]
        manifest["last_full_download"] = run

    _write_json(args.manifest, manifest)

    print(
        "Done | "
        f"tickers={stats.tickers_total} ok={stats.tickers_ok} no_data={stats.tickers_no_data} failed={stats.tickers_failed} "
        f"rows_written={stats.rows_written} out_dir={args.out_dir} manifest={args.manifest}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
