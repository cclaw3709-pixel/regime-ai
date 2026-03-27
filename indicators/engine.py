"""Technical indicator engine using pandas_ta."""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any
import config


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all configured technical indicators.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicator columns
    """
    params = config.INDICATOR_PARAMS
    result = df.copy()

    # RSI
    rsi_period = params.get("rsi", {}).get("period", 14)
    rsi = ta.rsi(result["close"], length=rsi_period)
    result["rsi"] = rsi.values if hasattr(rsi, "values") else rsi

    # MACD
    macd_params = params.get("macd", {})
    macd_df = ta.macd(
        result["close"],
        fast=macd_params.get("fast", 12),
        slow=macd_params.get("slow", 26),
        signal=macd_params.get("signal", 9)
    )
    # Handle both old and new pandas_ta column name formats
    macd_cols = macd_df.columns.tolist()
    result["macd"] = macd_df[macd_cols[0]].values if len(macd_cols) >= 1 else None
    result["macd_signal"] = macd_df[macd_cols[2]].values if len(macd_cols) >= 3 else None
    result["macd_hist"] = macd_df[macd_cols[1]].values if len(macd_cols) >= 2 else None

    # Bollinger Bands
    bb_params = params.get("bb", {})
    bb_period = bb_params.get("period", 20)
    bb_std = bb_params.get("std_dev", 2)
    bb_df = ta.bbands(result["close"], length=bb_period, std=bb_std)
    bb_cols = bb_df.columns.tolist()
    result["bb_lower"] = bb_df[bb_cols[0]].values
    result["bb_mid"] = bb_df[bb_cols[1]].values
    result["bb_upper"] = bb_df[bb_cols[2]].values
    result["bb_width"] = bb_df[bb_cols[3]].values if len(bb_cols) > 3 else None
    result["bb_percent"] = bb_df[bb_cols[4]].values if len(bb_cols) > 4 else None

    # SMA
    sma_short = params.get("sma_short", 50)
    sma_long = params.get("sma_long", 200)
    result["sma_short"] = ta.sma(result["close"], length=sma_short)
    result["sma_long"] = ta.sma(result["close"], length=sma_long)

    # EMA
    ema_short = params.get("ema_short", 12)
    ema_long = params.get("ema_long", 26)
    result["ema_short"] = ta.ema(result["close"], length=ema_short)
    result["ema_long"] = ta.ema(result["close"], length=ema_long)

    # ATR
    atr_period = params.get("atr", {}).get("period", 14)
    result["atr"] = ta.atr(result["high"], result["low"], result["close"], length=atr_period)

    # VWAP (requires volume)
    if "volume" in result.columns:
        result["vwap"] = ta.vwap(result["high"], result["low"], result["close"], result["volume"])

    return result


def get_latest_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Extract latest indicator values.

    Args:
        df: DataFrame with computed indicators

    Returns:
        Dictionary of latest indicator values
    """
    latest = df.iloc[-1]
    indicators = {}

    indicator_names = [
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_lower", "bb_mid", "bb_upper", "bb_width", "bb_percent",
        "sma_short", "sma_long", "ema_short", "ema_long",
        "atr", "vwap", "close"
    ]

    for name in indicator_names:
        if name in df.columns:
            val = latest[name]
            indicators[name] = float(val) if pd.notna(val) else None

    return indicators


def detect_crossovers(df: pd.DataFrame) -> Dict[str, bool]:
    """Detect MA and MACD crossovers.

    Args:
        df: DataFrame with computed indicators

    Returns:
        Dictionary of crossover signals
    """
    if len(df) < 2:
        return {}

    current = df.iloc[-1]
    previous = df.iloc[-2]

    return {
        "sma_cross_above": (
            previous["sma_short"] < previous["sma_long"] and
            current["sma_short"] > current["sma_long"]
        ) if "sma_short" in df.columns and "sma_long" in df.columns else False,
        "sma_cross_below": (
            previous["sma_short"] > previous["sma_long"] and
            current["sma_short"] < current["sma_long"]
        ) if "sma_short" in df.columns and "sma_long" in df.columns else False,
        "ema_cross_above": (
            previous["ema_short"] < previous["ema_long"] and
            current["ema_short"] > current["ema_long"]
        ) if "ema_short" in df.columns and "ema_long" in df.columns else False,
        "ema_cross_below": (
            previous["ema_short"] > previous["ema_long"] and
            current["ema_short"] < current["ema_long"]
        ) if "ema_short" in df.columns and "ema_long" in df.columns else False,
        "macd_cross_above": (
            previous["macd"] < previous["macd_signal"] and
            current["macd"] > current["macd_signal"]
        ) if "macd" in df.columns and "macd_signal" in df.columns else False,
        "macd_cross_below": (
            previous["macd"] > previous["macd_signal"] and
            current["macd"] < current["macd_signal"]
        ) if "macd" in df.columns and "macd_signal" in df.columns else False,
    }


if __name__ == "__main__":
    from data.fetchers import fetch_yahoo_data

    df = fetch_yahoo_data("BTCUSD", "1h", period="30d")
    df = compute_all_indicators(df)
    indicators = get_latest_indicators(df)
    crossovers = detect_crossovers(df)
    print("Latest indicators:", indicators)
    print("Crossovers:", crossovers)
