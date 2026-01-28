#!/usr/bin/env python3
"""ETF Data Ingestion Script

This script fetches historical price data for ETFs defined in a universe configuration
and stores it in the curated data directory with proper validation and metadata.

Usage:
    python scripts/ingest_etf_data.py --universe initial_20_etfs --start-date 2018-01-01 --end-date 2024-12-31
    python scripts/ingest_etf_data.py --universe initial_20_etfs --lookback-years 5
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import yaml

from quantetf.data.providers.yfinance_provider import YFinanceProvider
from quantetf.utils.validation import validate_ohlcv_dataframe, detect_price_anomalies


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Helper to compute output file path
def get_output_filepath(universe_name: str) -> Path:
    """Return the output file path for the given universe name."""
    curated_dir = Path(__file__).parent.parent / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{universe_name}.parquet"
    return curated_dir / filename


def load_universe_config(universe_name: str) -> dict:
    """Load universe configuration from YAML file.

    Args:
        universe_name: Name of the universe config file (without .yaml extension)

    Returns:
        Dictionary containing universe configuration

    Raises:
        FileNotFoundError: If universe config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(__file__).parent.parent / "configs" / "universes" / f"{universe_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Universe config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config structure
    if 'source' not in config or 'tickers' not in config['source']:
        raise ValueError(f"Invalid universe config: missing 'source.tickers' field")

    logger.info(f"Loaded universe '{config['name']}' with {len(config['source']['tickers'])} tickers")
    return config


def fetch_etf_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    provider: Optional[YFinanceProvider] = None
) -> Optional[pd.DataFrame]:
    """Fetch ETF price data using the configured provider.

    Args:
        tickers: List of ETF ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        provider: Optional YFinanceProvider instance (creates new one if None)

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price_Field) or None if failed
    """
    if provider is None:
        provider = YFinanceProvider(auto_adjust=True, progress=True)

    logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")

    # Download tickers in batches, logging and skipping failures
    batch_size = 10
    all_data = []
    failed_tickers = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        data = provider.fetch_prices(batch, start_date, end_date)
        if data is None or data.empty:
            logger.warning(f"Failed to fetch data for tickers: {batch}")
            failed_tickers.extend(batch)
            continue
        # Remove tickers with all-NaN columns (yfinance returns them for failed tickers)
        valid_tickers = [t for t in batch if not data[t].dropna(how='all').empty]
        for t in batch:
            if t not in valid_tickers:
                logger.warning(f"No data for ticker: {t}")
                failed_tickers.append(t)
        # Keep only valid tickers
        if valid_tickers:
            all_data.append(data[valid_tickers])
    if not all_data:
        logger.error("Failed to fetch data for all tickers.")
        return None
    # Concatenate all valid data
    result = pd.concat(all_data, axis=1)
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers}")
        # Log to file
        with open("failed_tickers.log", "w") as f:
            for t in failed_tickers:
                f.write(f"{t}\n")
    logger.info(f"Successfully fetched {len(result)} rows of data for {len(result.columns.get_level_values('Ticker').unique())} tickers")
    return result


def validate_ticker_data(df: pd.DataFrame, ticker: str) -> dict:
    """Validate data for a single ticker and detect anomalies.

    Args:
        df: DataFrame with MultiIndex columns (Ticker, Price_Field)
        ticker: Ticker symbol to validate

    Returns:
        Dictionary with validation results:
        - valid: bool indicating if data passed validation
        - error: str with error message if validation failed
        - anomalies: dict with anomaly detection results
    """
    result = {
        'ticker': ticker,
        'valid': False,
        'error': None,
        'anomalies': None
    }

    try:
        # Validate OHLCV data structure and quality
        validate_ohlcv_dataframe(df, ticker)
        result['valid'] = True

        # Detect anomalies (doesn't fail validation, just warns)
        anomalies = detect_price_anomalies(df, ticker)
        result['anomalies'] = anomalies

        # Log warnings for anomalies
        if anomalies['suspicious_days'] > 0:
            logger.warning(f"{ticker}: Found {anomalies['suspicious_days']} suspicious days (OHLC all equal)")
        if anomalies['extreme_changes'] > 0:
            logger.warning(f"{ticker}: Found {anomalies['extreme_changes']} extreme price changes")
        if anomalies['zero_volume'] > 0:
            logger.warning(f"{ticker}: Found {anomalies['zero_volume']} zero volume days")

    except ValueError as e:
        result['error'] = str(e)
        logger.error(f"Validation failed for {ticker}: {e}")

    return result


def save_to_curated(
    df: pd.DataFrame,
    universe_name: str,
    start_date: str,
    end_date: str,
    validation_results: List[dict]
) -> Path:
    """Save curated data and metadata to disk.

    Merges new data with existing data if the file exists, avoiding duplicates.

    Args:
        df: DataFrame with MultiIndex columns (Ticker, Price_Field)
        universe_name: Name of the universe
        start_date: Start date of data
        end_date: End date of data
        validation_results: List of validation result dictionaries

    Returns:
        Path to the saved data file
    """

    # Compute output file path in one place
    filepath = get_output_filepath(universe_name)

    # Merge with existing data if file exists
    if filepath.exists():
        logger.info(f"Merging with existing data from {filepath}")
        existing_df = pd.read_parquet(filepath)
        # Concatenate and remove duplicate dates (keep new data)
        df = pd.concat([existing_df, df])
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        # Update start_date to reflect full range
        start_date = df.index.min().strftime('%Y-%m-%d')

    # Save data as Parquet (efficient columnar format)
    logger.info(f"Saving curated data to {filepath}")
    df.to_parquet(filepath, compression='snappy', index=True)

    # Create metadata file
    metadata = {
        'universe_name': universe_name,
        'start_date': start_date,
        'end_date': end_date,
        'created_at': datetime.now().isoformat(),
        'num_tickers': len(df.columns.get_level_values('Ticker').unique()),
        'num_rows': len(df),
        'date_range': {
            'first_date': df.index.min().isoformat(),
            'last_date': df.index.max().isoformat()
        },
        'validation_summary': {
            'total_tickers': len(validation_results),
            'valid_tickers': sum(1 for r in validation_results if r['valid']),
            'failed_tickers': sum(1 for r in validation_results if not r['valid']),
            'total_anomalies': sum(
                r['anomalies']['suspicious_days'] +
                r['anomalies']['extreme_changes'] +
                r['anomalies']['zero_volume']
                for r in validation_results if r['anomalies']
            )
        },
        'validation_details': validation_results
    }

    metadata_path = filepath.with_suffix('.metadata.yaml')
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Successfully saved curated data: {filepath}")
    return filepath


def main():
    """Main entry point for data ingestion script."""
    parser = argparse.ArgumentParser(
        description='Ingest ETF price data from Yahoo Finance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch specific date range
  python scripts/ingest_etf_data.py --universe initial_20_etfs --start-date 2018-01-01 --end-date 2024-12-31

  # Fetch last N years
  python scripts/ingest_etf_data.py --universe initial_20_etfs --lookback-years 5

  # Fetch with custom output
  python scripts/ingest_etf_data.py --universe initial_20_etfs --lookback-years 3 --output-dir ./custom_data
        """
    )

    parser.add_argument(
        '--universe',
        required=True,
        #default='tier4_broad_200',
        help='Name of universe config file (without .yaml extension). Default: tier4_broad_200'
    )

    # Date range options (all optional)
    parser.add_argument(
        '--lookback-years',
        type=int,
        help='Number of years of history to fetch (from today)'
    )
    parser.add_argument(
        '--start-date',
        help='Start date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--end-date',
        help='End date in YYYY-MM-DD format (requires --start-date)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate tickers without fetching full history'
    )

    args = parser.parse_args()

    # Validate date arguments
    #if args.start_date and not args.end_date:
    #    parser.error("--start-date requires --end-date")
    #if args.end_date and not args.start_date:
    #    parser.error("--end-date requires --start-date")

    # Calculate date range
    def get_latest_end_date(universe_name):
            """Get the latest end date from the data in the output Parquet file for the given universe."""
            try:
                filepath = get_output_filepath(universe_name)
                if not filepath.exists():
                    return None
                df = pd.read_parquet(filepath)
                if df.empty:
                    return None
                # Assume index is datetime or convertible to string
                last_date = df.index.max()
                if hasattr(last_date, 'strftime'):
                    return last_date.strftime('%Y-%m-%d')
                return str(last_date)
            except Exception as e:
                logger.warning(f"Could not determine latest end date from file: {e}")
                return None

    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    if args.lookback_years:
        end_date = today
        start_date = (datetime.now() - timedelta(days=args.lookback_years * 365)).strftime('%Y-%m-%d')
    else:
        # Use yesterday as default end date since today's data may not be available yet
        end_date = args.end_date if args.end_date else yesterday
        if args.start_date:
            start_date = args.start_date
        else:
            # Try to get the latest available date in the snapshot and add one day
            latest_end = get_latest_end_date(args.universe)
            if latest_end:
                try:
                    next_day = (datetime.strptime(latest_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                    start_date = next_day
                except Exception:
                    start_date = None
            else:
                start_date = None
        if not start_date:
            parser.error('Could not determine a default start date. Please specify --start-date.')

    # Check if data is already up to date
    if start_date > end_date:
        logger.info("=" * 80)
        logger.info("Data is already up to date. Nothing to fetch.")
        logger.info("=" * 80)
        return 0

    logger.info("=" * 80)
    logger.info("ETF Data Ingestion")
    logger.info("=" * 80)

    # Load universe configuration
    try:
        universe_config = load_universe_config(args.universe)
        tickers = universe_config['source']['tickers']
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load universe config: {e}")
        return 1

    # Initialize provider
    provider = YFinanceProvider(auto_adjust=True, progress=True)

    # Validate tickers first
    if args.validate_only:
        logger.info("Validating tickers only (not fetching full history)")
        for ticker in tickers:
            is_valid = provider.validate_ticker(ticker)
            status = "✓" if is_valid else "✗"
            logger.info(f"{status} {ticker}")
        return 0

    # Fetch data
    logger.info(f"Date range: {start_date} to {end_date}")
    data = fetch_etf_data(tickers, start_date, end_date, provider)

    if data is None:
        logger.error("Failed to fetch data")
        return 1

    # Validate each ticker
    logger.info("Validating data quality...")
    validation_results = []

    available_tickers = data.columns.get_level_values('Ticker').unique()
    for ticker in available_tickers:
        result = validate_ticker_data(data, ticker)
        validation_results.append(result)

    # Summary
    valid_count = sum(1 for r in validation_results if r['valid'])
    failed_count = len(validation_results) - valid_count

    logger.info("=" * 80)
    logger.info(f"Validation Summary: {valid_count}/{len(validation_results)} tickers passed")
    if failed_count > 0:
        logger.warning(f"Failed tickers: {[r['ticker'] for r in validation_results if not r['valid']]}")
    logger.info("=" * 80)

    # Save curated data
    output_path = save_to_curated(
        data,
        args.universe,
        start_date,
        end_date,
        validation_results
    )

    logger.info("=" * 80)
    logger.info(f"SUCCESS: Data saved to {output_path}")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
