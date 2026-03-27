"""Configuration for trading signals program."""

import os

# Symbols to trade
SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]

# Data fetcher settings
DATA_SOURCE = "yfinance"  # "yfinance" or "binance"
DEFAULT_INTERVAL = "1h"

# Indicator parameters
INDICATOR_PARAMS = {
    "rsi": {"period": 14, "overbought": 70, "oversold": 30},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bb": {"period": 20, "std_dev": 2},
    "sma_short": 50,
    "sma_long": 200,
    "ema_short": 12,
    "ema_long": 26,
    "atr": {"period": 14},
    "vwap": {},
}

# Signal generation parameters
SIGNAL_PARAMS = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_crossover_confirmation": True,
    "bb_lower_touch": True,
    "bb_upper_touch": True,
}

# ML model settings
ML_PARAMS = {
    "labeling": {
        "future_periods": 24,  # 2h ahead for 1h interval
        "buy_threshold": 0.01,  # 1% return
        "sell_threshold": -0.01,
    },
    "model": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
    },
}

# Alert thresholds
ALERT_THRESHOLDS = {
    "min_confidence_buy": 0.6,
    "min_confidence_sell": 0.6,
}

# Telegram settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Storage settings
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "data", "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Default data cache duration (minutes)
CACHE_DURATION = 60
