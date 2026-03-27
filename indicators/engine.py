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
    if "rsi" in params:
        rsi_period = params["rsi"]["period"]
        result["rsi"] = ta.rsi(result["close"], length=rsi_period)

    # MACD
    macd_params = params.get("macd", {})
    macd = ta.macd(
        result["close"],
        fast=macd_params.get("fast", 12),
        slow=macd_params.get("slow", 26),
        signal=macd_params.get("signal", 9)
    )
    result["macd"] = macd["MACD_12_26_9"]
    result["macd_signal"] = macd["MACDs_12_26_9"]
    result["macd_hist"] = macd["MACDh_12_26_9"]

    # Bollinger Bands
    bb_params = params.get("bb", {})
    bb = ta.bbands(
        result["close"],
        length=bb_params.get("period", 20),
        std=bb_params.get("std_dev", 2)
    )
    result["bb_lower"] = bb["BBL_20_2.0"]
    result["bb_mid"] = bb["BBM_20_2.0"]
    result["bb_upper"] = bb["BBU_20_2.0"]
    result["bb_width"] = bb["BBB_20_2.0"]
    result["bb_percent"] = bb["BBP_20_2.0"]

    # SMA
    result["sma_short"] = ta.sma(result["close"], length=params.get("sma_short", 50))
    result["sma_long"] = ta.sma(result["close"], length=params.get("sma_long", 200))

    # EMA
    result["ema_short"] = ta.ema(result["close"], length=params.get("ema_short", 12))
    result["ema_long"] = ta.ema(result["close"], length=params.get("ema_long", 26))

    # ATR
    atr_params = params.get("atr", {})
    result["atr"] = ta.atr(
        result["high"], result["low"], result["close"],
        length=atr_params.get("period", 14)
    )

    # VWAP
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
    from data.fetchers import fetch_data
    from data.storage import load_or_fetch

    df = load_or_fetch("BTCUSD", "1h", period="30d")
    df = compute_all_indicators(df)
    indicators = get_latest_indicators(df)
    crossovers = detect_crossovers(df)
    print("Latest indicators:", indicators)
    print("Crossovers:", crossovers)
