#!/usr/bin/env python3
"""
Show Current Regime Status

Quick script to display current regime state without updating.

Usage:
    python scripts/show_regime_status.py
    python scripts/show_regime_status.py --json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Show current regime status")
    parser.add_argument(
        "--state-dir",
        type=str,
        default="data/state",
        help="State directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=0,
        help="Show last N days of history",
    )

    args = parser.parse_args()

    state_file = Path(args.state_dir) / "current_regime.json"

    if not state_file.exists():
        print("No regime state found. Run 'python scripts/run_daily_monitor.py' first.")
        return 1

    with open(state_file) as f:
        state = json.load(f)

    if args.json:
        print(json.dumps(state, indent=2))
    else:
        print("\n" + "=" * 40)
        print("CURRENT REGIME STATUS")
        print("=" * 40)
        print(f"Regime:     {state['regime']}")
        print(f"Trend:      {state['trend']}")
        print(f"Volatility: {state['vol']}")
        print(f"As of:      {state['as_of']}")
        print("-" * 40)
        indicators = state.get("indicators", {})
        if indicators:
            print(f"SPY Price:  ${indicators.get('spy_price', 0):.2f}")
            print(f"SPY 200MA:  ${indicators.get('spy_200ma', 0):.2f}")
            print(f"VIX:        {indicators.get('vix', 0):.2f}")
        print("=" * 40 + "\n")

    # Show history if requested
    if args.history > 0:
        history_file = Path(args.state_dir) / "regime_history.parquet"
        if history_file.exists():
            import pandas as pd
            history = pd.read_parquet(history_file)
            recent = history.tail(args.history)

            print(f"\nLast {args.history} days:")
            print("-" * 60)
            for _, row in recent.iterrows():
                changed = " *" if row.get("regime_changed") else ""
                print(f"  {row['date'].date()}  {row['regime']}{changed}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
