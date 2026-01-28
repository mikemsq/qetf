#!/usr/bin/env python3
"""Create Versioned Data Snapshot

This script creates an immutable, versioned snapshot from curated data for
reproducible backtesting. Snapshots include the data file, manifest with
metadata, and validation results.

Usage:
    python scripts/create_snapshot.py --input data/curated/initial_20_etfs_2021-01-10_2026-01-09_20260109_022324.parquet
    python scripts/create_snapshot.py --universe initial_20_etfs --lookback-years 5
"""

import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml
import subprocess
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_for_yaml(obj):
    """Convert numpy types to native Python types for YAML serialization.

    Args:
        obj: Object to convert (dict, list, numpy type, etc.)

    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {k: clean_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_yaml(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_git_commit_hash() -> str:
    """Get current git commit hash for reproducibility.

    Returns:
        Git commit hash string or 'unknown' if not in a git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def create_snapshot(
    curated_path: Path,
    metadata_path: Path,
    snapshot_name: str = None
) -> Path:
    """Create a versioned snapshot from curated data.

    Args:
        curated_path: Path to curated parquet file
        metadata_path: Path to metadata YAML file
        snapshot_name: Optional custom snapshot name (defaults to timestamp)

    Returns:
        Path to created snapshot directory
    """
    # Generate snapshot name
    if snapshot_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"snapshot_{timestamp}"

    # Create snapshot directory (replace if exists)
    snapshots_dir = Path(__file__).parent.parent / "data" / "snapshots"
    snapshot_dir = snapshots_dir / snapshot_name
    if snapshot_dir.exists():
        logger.info(f"Replacing existing snapshot: {snapshot_name}")
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating snapshot: {snapshot_name}")

    # Copy data file
    data_dest = snapshot_dir / "data.parquet"
    logger.info(f"Copying data: {curated_path} -> {data_dest}")
    shutil.copy2(curated_path, data_dest)

    # Load and verify data
    df = pd.read_parquet(data_dest)
    logger.info(f"Snapshot contains {len(df)} rows, {len(df.columns.get_level_values('Ticker').unique())} tickers")

    # Load curated metadata (use unsafe loader since we control the source and need numpy types)
    with open(metadata_path, 'r') as f:
        curated_metadata = yaml.unsafe_load(f)

    # Create snapshot manifest (clean numpy types)
    manifest = {
        'snapshot_name': snapshot_name,
        'created_at': datetime.now().isoformat(),
        'git_commit': get_git_commit_hash(),
        'source_files': {
            'curated_data': str(curated_path.name),
            'curated_metadata': str(metadata_path.name) if metadata_path else 'none'
        },
        'data_summary': {
            'universe': curated_metadata.get('universe_name', 'unknown'),
            'num_tickers': len(df.columns.get_level_values('Ticker').unique()),
            'tickers': sorted(df.columns.get_level_values('Ticker').unique().tolist()),
            'num_rows': len(df),
            'date_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            },
            'columns_format': 'MultiIndex (Ticker, Price)',
            'price_fields': sorted(df.columns.get_level_values('Price').unique().tolist())
        },
        'validation_summary': clean_for_yaml(curated_metadata.get('validation_summary', {})),
        'notes': 'Immutable snapshot for reproducible backtesting'
    }

    manifest_path = snapshot_dir / "manifest.yaml"
    logger.info(f"Writing manifest: {manifest_path}")
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    # Extract validation details if available
    if 'validation_details' in curated_metadata:
        validation_path = snapshot_dir / "validation.yaml"
        logger.info(f"Writing validation details: {validation_path}")

        validation_data = {
            'validation_timestamp': curated_metadata.get('created_at'),
            'tickers_validated': len(curated_metadata['validation_details']),
            'details': clean_for_yaml(curated_metadata['validation_details'])
        }

        with open(validation_path, 'w') as f:
            yaml.dump(validation_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Snapshot created successfully: {snapshot_dir}")
    return snapshot_dir


def find_latest_curated(universe_name: str) -> tuple:
    """Find the most recent curated data file for a universe.

    Args:
        universe_name: Name of the universe to search for

    Returns:
        Tuple of (parquet_path, metadata_path) or (None, None) if not found
    """
    curated_dir = Path(__file__).parent.parent / "data" / "curated"

    # First check for simple filename (used by daily ingestion)
    simple_path = curated_dir / f"{universe_name}.parquet"
    if simple_path.exists():
        metadata = simple_path.with_suffix('.metadata.yaml')
        if not metadata.exists():
            logger.warning(f"Metadata file not found: {metadata}")
            return simple_path, None
        return simple_path, metadata

    # Fall back to timestamped files for historical data
    pattern = f"{universe_name}_*.parquet"
    parquet_files = sorted(curated_dir.glob(pattern), reverse=True)

    if not parquet_files:
        return None, None

    # Get most recent file (sorted by timestamp in filename)
    latest = parquet_files[0]
    metadata = latest.with_suffix('.metadata.yaml')

    if not metadata.exists():
        logger.warning(f"Metadata file not found: {metadata}")
        return latest, None

    return latest, metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create versioned data snapshot for reproducible backtesting'
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=Path,
        help='Path to curated parquet file'
    )
    input_group.add_argument(
        '--universe',
        help='Universe name (uses most recent curated data)'
    )

    parser.add_argument(
        '--name',
        help='Custom snapshot name (defaults to timestamp)'
    )

    args = parser.parse_args()

    # Determine input files
    if args.input:
        curated_path = args.input
        metadata_path = curated_path.with_suffix('.metadata.yaml')

        if not curated_path.exists():
            logger.error(f"Curated data file not found: {curated_path}")
            return 1
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            metadata_path = None
    else:
        curated_path, metadata_path = find_latest_curated(args.universe)

        if curated_path is None:
            logger.error(f"No curated data found for universe: {args.universe}")
            return 1

        logger.info(f"Using latest curated data: {curated_path}")

    # Create snapshot
    try:
        snapshot_dir = create_snapshot(curated_path, metadata_path, args.name)

        logger.info("=" * 80)
        logger.info(f"SUCCESS: Snapshot created at {snapshot_dir}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Failed to create snapshot: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
