#!/usr/bin/env python
"""Data health check script for snapshot validation.

Usage:
    python scripts/data_health_check.py --snapshot data/snapshots/snapshot_5yr_20etfs
    python scripts/data_health_check.py --snapshot data/snapshots/snapshot_5yr_20etfs --output report.txt
"""

import argparse
from pathlib import Path
import pandas as pd

from quantetf.data.quality import generate_quality_report_text


def main():
    """Run data quality checks on a snapshot."""
    parser = argparse.ArgumentParser(
        description='Run data quality checks on snapshot data'
    )
    parser.add_argument(
        '--snapshot',
        required=True,
        help='Path to snapshot directory'
    )
    parser.add_argument(
        '--output',
        help='Optional output file path (default: print to console)'
    )

    args = parser.parse_args()

    # Load snapshot
    snapshot_path = Path(args.snapshot)
    data_file = snapshot_path / 'data.parquet'

    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return 1

    print(f"Loading snapshot from: {snapshot_path}")
    df = pd.read_parquet(data_file)

    # Generate report
    print("Analyzing data quality...")
    report = generate_quality_report_text(df)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    else:
        print("\n" + report)

    return 0


if __name__ == '__main__':
    exit(main())
