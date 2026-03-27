"""Multi-indicator signal generator."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import config
from indicators.engine import compute_all_indicators, get_latest_indicators, detect_crossovers
from data.storage import load_or_fetch


@dataclass
class Signal:
    """Trading signal data class."""
    timestamp: datetime
    symbol: str
    direction: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    indicators: Dict[str, float]
    reason: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "entry_price": round(self.entry_price, 4),
            "indicators": {k: round(v, 4) if v else None for k, v in self.indicators.items()},
            "reason": self.reason,
        }


class SignalGenerator:
    """Multi-indicator signal generator combining RSI, MACD, BB, and MA crossovers."""

    def __init__(self, symbol: str, interval: str = "1h"):
        self.symbol = symbol
        self.interval = interval
        self.params = config.SIGNAL_PARAMS
        self.df: Optional[pd.DataFrame] = None

    def load_data(self, period: str = "30d") -> pd.DataFrame:
        """Load market data."""
        self.df = load_or_fetch(self.symbol, self.interval, period)
        self.df = compute_all_indicators(self.df)
        return self.df

    def generate_signal(self) -> Signal:
        """Generate trading signal based on multiple indicators.

        Returns:
            Signal object with direction, confidence, and indicators
        """
        if self.df is None:
            self.load_data()

        indicators = get_latest_indicators(self.df)
        crossovers = detect_crossovers(self.df)
        timestamp = self.df.index[-1]
        close_price = indicators.get("close", 0)

        buy_score = 0.0
        sell_score = 0.0
        reasons = []

        # RSI analysis
        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi < self.params["rsi_oversold"]:
                buy_score += 0.25
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > self.params["rsi_overbought"]:
                sell_score += 0.25
                reasons.append(f"RSI overbought ({rsi:.1f})")

        # MACD analysis
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        macd_hist = indicators.get("macd_hist", 0)

        if crossovers.get("macd_cross_above"):
            buy_score += 0.3
            reasons.append("MACD crossed above signal")
        elif crossovers.get("macd_cross_below"):
            sell_score += 0.3
            reasons.append("MACD crossed below signal")
        elif macd_hist > 0:
            buy_score += 0.15
        elif macd_hist < 0:
            sell_score += 0.15

        # Bollinger Bands analysis
        bb_percent = indicators.get("bb_percent")
        if bb_percent is not None:
            if bb_percent < 0.1:  # Near lower band
                buy_score += 0.2
                reasons.append("Near Bollinger lower band")
            elif bb_percent > 0.9:  # Near upper band
                sell_score += 0.2
                reasons.append("Near Bollinger upper band")

        # MA crossover analysis
        if crossovers.get("sma_cross_above") or crossovers.get("ema_cross_above"):
            buy_score += 0.25
            reasons.append("Short MA crossed above long MA")
        elif crossovers.get("sma_cross_below") or crossovers.get("ema_cross_below"):
            sell_score += 0.25
            reasons.append("Short MA crossed below long MA")

        # VWAP analysis
        vwap = indicators.get("vwap")
        if vwap and close_price > vwap:
            buy_score += 0.1
        elif vwap and close_price < vwap:
            sell_score += 0.1

        # Determine direction and confidence
        total_score = buy_score + sell_score
        if total_score > 0:
            if buy_score > sell_score:
                direction = "BUY"
                confidence = buy_score / total_score * min(buy_score, 1.0)
            else:
                direction = "SELL"
                confidence = sell_score / total_score * min(sell_score, 1.0)
        else:
            direction = "HOLD"
            confidence = 0.5

        # Apply minimum confidence threshold
        min_threshold = config.ALERT_THRESHOLDS.get("min_confidence_buy", 0.6)
        if direction == "BUY" and confidence < min_threshold:
            direction = "HOLD"
        elif direction == "SELL" and confidence < config.ALERT_THRESHOLDS.get("min_confidence_sell", 0.6):
            direction = "HOLD"

        reason = ", ".join(reasons) if reasons else "No strong signals"

        return Signal(
            timestamp=timestamp,
            symbol=self.symbol,
            direction=direction,
            confidence=confidence,
            entry_price=close_price,
            indicators=indicators,
            reason=reason,
        )


def generate_signals(symbols: List[str], interval: str = "1h") -> List[Signal]:
    """Generate signals for multiple symbols.

    Args:
        symbols: List of trading symbols
        interval: Data interval

    Returns:
        List of Signal objects
    """
    signals = []
    for symbol in symbols:
        try:
            generator = SignalGenerator(symbol, interval)
            signal = generator.generate_signal()
            signals.append(signal)
        except Exception as e:
            print(f"Error generating signal for {symbol}: {e}")
    return signals


if __name__ == "__main__":
    signals = generate_signals(["BTCUSD"], "1h")
    for sig in signals:
        print(sig.to_dict())
