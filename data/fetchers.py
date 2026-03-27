"""Data fetching from Yahoo Finance or Binance."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import config


def _convert_symbol(symbol: str) -> str:
    """Convert exchange symbol format to Yahoo Finance format."""
    # BTCUSD -> BTC-USD, ETHUSD -> ETH-USD, etc.
    if symbol.endswith("USD") and not symbol.startswith("BTC"):
        return symbol[:-3] + "-USD"
    if symbol.endswith("USD"):
        return symbol.replace("USD", "-USD")
    return symbol


def fetch_yahoo_data(
    symbol: str,
    interval: str = "1h",
    period: str = "30d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Fetch data from Yahoo Finance.

    Args:
        symbol: Trading symbol (e.g., BTCUSD, ETHUSD)
        interval: Data interval (1m, 5m, 15m, 1h, 1d, etc.)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, etc.)
        start: Start datetime (overrides period if provided)
        end: End datetime (overrides period if provided)

    Returns:
        DataFrame with OHLCV data
    """
    yahoo_symbol = _convert_symbol(symbol)
    ticker = yf.Ticker(yahoo_symbol)
    if start and end:
        df = ticker.history(start=start, end=end, interval=interval)
    else:
        df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for symbol {symbol} (Yahoo: {yahoo_symbol})")

    # Flatten MultiIndex columns if present (yfinance returns multi-level columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]

    # Handle timezone - newer pandas returns Index without tz
    try:
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    df.index.name = "timestamp"
    return df


def fetch_binance_data(
    symbol: str,
    interval: str = "1h",
    period: str = "30d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Fetch data from Binance.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT, ETHUSDT)
        interval: Data interval (1m, 5m, 15m, 1h, 4h, 1d)
        period: Lookback period as string (e.g., "30 days")
        start: Start datetime
        end: End datetime

    Returns:
        DataFrame with OHLCV data
    """
    # Map interval names
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m",
        "1h": "1h", "4h": "4h", "1d": "1d"
    }
    binance_interval = interval_map.get(interval, "1h")

    # Map symbol to Binance format
    binance_symbol = symbol
    if not symbol.endswith("USDT"):
        binance_symbol = symbol + "USDT"

    # Calculate dates
    if start and end:
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
    else:
        days = int(period.rstrip("d")) if period.endswith("d") else 30
        end_str = datetime.now().strftime("%Y-%m-%d")
        start_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Build URL
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={binance_symbol}&interval={binance_interval}"
        f"&startDate={start_str}&endDate={end_str}&limit=1000"
    )

    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        print(f"Binance fetch failed: {e}, falling back to Yahoo Finance")
        return fetch_yahoo_data(symbol, interval, period, start, end)


def fetch_data(
    symbol: str,
    interval: str = None,
    period: str = "30d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Fetch data from configured source.

    Args:
        symbol: Trading symbol
        interval: Data interval
        period: Data period
        start: Start datetime
        end: End datetime

    Returns:
        DataFrame with OHLCV data
    """
    if interval is None:
        interval = config.DEFAULT_INTERVAL

    source = config.DATA_SOURCE.lower()

    if source == "binance":
        return fetch_binance_data(symbol, interval, period, start, end)
    else:
        return fetch_yahoo_data(symbol, interval, period, start, end)


if __name__ == "__main__":
    # Test fetch
    df = fetch_data("BTCUSD", period="5d")
    print(df.tail())
