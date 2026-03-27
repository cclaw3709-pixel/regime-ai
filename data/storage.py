"""Storage layer for CSV and Parquet data."""

import os
import pandas as pd
from datetime import datetime
from typing import Optional
import config


def get_storage_path(symbol: str, interval: str, file_format: str = "parquet") -> str:
    """Get storage file path for symbol and interval.

    Args:
        symbol: Trading symbol
        interval: Data interval
        file_format: "csv" or "parquet"

    Returns:
        Full file path
    """
    filename = f"{symbol}_{interval}.{file_format}"
    return os.path.join(config.STORAGE_DIR, filename)


def save_data(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    file_format: str = "parquet",
    append: bool = True,
) -> str:
    """Save data to storage.

    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol
        interval: Data interval
        file_format: "csv" or "parquet"
        append: Whether to append to existing file

    Returns:
        Path to saved file
    """
    path = get_storage_path(symbol, interval, file_format)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if append and os.path.exists(path):
        existing = load_data(symbol, interval, file_format)
        if existing is not None:
            # Merge and deduplicate
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            df = combined

    if file_format == "csv":
        df.to_csv(path)
    else:
        df.to_parquet(path, index=True)

    return path


def load_data(
    symbol: str,
    interval: str,
    file_format: str = "parquet",
) -> Optional[pd.DataFrame]:
    """Load data from storage.

    Args:
        symbol: Trading symbol
        interval: Data interval
        file_format: "csv" or "parquet"

    Returns:
        DataFrame or None if file doesn't exist
    """
    path = get_storage_path(symbol, interval, file_format)

    if not os.path.exists(path):
        return None

    if file_format == "csv":
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    else:
        df = pd.read_parquet(path)

    return df


def load_or_fetch(
    symbol: str,
    interval: str,
    period: str = "30d",
    file_format: str = "parquet",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load from cache or fetch fresh data.

    Args:
        symbol: Trading symbol
        interval: Data interval
        period: Data period
        file_format: "csv" or "parquet"
        force_refresh: Force fetch even if cache exists

    Returns:
        DataFrame with OHLCV data
    """
    if not force_refresh:
        df = load_data(symbol, interval, file_format)

        if df is not None:
            # Check if cache is stale
            age = datetime.now() - df.index[-1]
            if age.total_seconds() < config.CACHE_DURATION * 60:
                print(f"Using cached data for {symbol} ({age:.1f} minutes old)")
                return df

    # Fetch fresh data
    from fetchers import fetch_data
    df = fetch_data(symbol, interval, period)

    # Save to cache
    if df is not None and len(df) > 0:
        save_data(df, symbol, interval, file_format)

    return df


if __name__ == "__main__":
    # Test storage
    from fetchers import fetch_data
    df = fetch_data("BTCUSD", period="5d")
    path = save_data(df, "BTCUSD", "1h")
    print(f"Saved to {path}")

    loaded = load_data("BTCUSD", "1h")
    print(f"Loaded {len(loaded)} rows")
