"""Binance WebSocket real-time data streaming."""

import json
import time
import threading
from datetime import datetime
from typing import Callable, Optional, List, Dict
import config
from indicators.engine import compute_all_indicators, get_latest_indicators
from signals.generator import SignalGenerator
from alerts.notifier import TelegramNotifier
import pandas as pd


class BinanceWebSocket:
    """Connects to Binance WebSocket streams for real-time kline/candlestick data.

    Supports: 1m, 5m, 15m, 1h, 4h, 1d intervals
    """

    def __init__(self, symbols: List[str], intervals: List[str] = ["1h"],
                 on_signal: Optional[Callable] = None,
                 on_tick: Optional[Callable] = None):
        self.symbols = [s.upper().replace("USD", "USDT") for s in symbols]  # BTCUSD → BTCUSDT
        self.intervals = intervals
        self.on_signal = on_signal
        self.on_tick = on_tick
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.latest_data: Dict[str, Dict] = {}  # symbol → {klines, signal}
        self.signal_generator = SignalGenerator()
        self.notifier = TelegramNotifier()

    def _build_stream_url(self) -> str:
        """Build combined streams URL for multiple symbols/intervals."""
        streams = []
        for sym in self.symbols:
            for interval in self.intervals:
                streams.append(f"{sym.lower()}@kline_{interval}")
        return "wss://stream.binance.com:9443/stream?streams=" + "/".join(streams)

    def _parse_kline(self, msg: dict) -> Optional[dict]:
        """Parse a kline (candlestick) message from WebSocket."""
        try:
            data = msg["data"]["k"]
            return {
                "symbol": data["s"],
                "interval": data["i"],
                "open_time": datetime.fromtimestamp(data["t"] / 1000),
                "open": float(data["o"]),
                "high": float(data["h"]),
                "low": float(data["l"]),
                "close": float(data["c"]),
                "volume": float(data["v"]),
                "closed": data["x"],  # Is this kline closed?
            }
        except (KeyError, ValueError):
            return None

    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            msg = json.loads(message)
            kline = self._parse_kline(msg)
            if not kline:
                return

            symbol = kline["symbol"].replace("USDT", "USD")
            interval = kline["interval"]

            # Store latest kline
            if symbol not in self.latest_data:
                self.latest_data[symbol] = {"klines": [], "signal": None}

            self.latest_data[symbol]["klines"].append(kline)

            # Keep last 100 klines per symbol
            if len(self.latest_data[symbol]["klines"]) > 100:
                self.latest_data[symbol]["klines"] = self.latest_data[symbol]["klines"][-100:]

            # Run signal check on closed candles only
            if kline["closed"]:
                self._check_signal(symbol, interval)

            if self.on_tick:
                self.on_tick(symbol, kline)

        except Exception as e:
            print(f"[WS] Message error: {e}")

    def _check_signal(self, symbol: str, interval: str):
        """Check for trading signal when candle closes."""
        try:
            klines = self.latest_data[symbol]["klines"]
            if len(klines) < 30:
                return

            # Build DataFrame
            df = pd.DataFrame(klines)
            df.set_index("open_time", inplace=True)
            df = compute_all_indicators(df)

            indicators = get_latest_indicators(df)
            self.latest_data[symbol]["indicators"] = indicators

            # Generate signal
            signal = self.signal_generator.generate_signal()
            self.latest_data[symbol]["signal"] = signal

            # Fire alert
            if signal.direction != "HOLD" and self.notifier.api_url:
                self.notifier.send_signal_alert(
                    symbol=symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    confidence=signal.confidence,
                    indicators=indicators,
                    reason=signal.reason,
                )

            if self.on_signal:
                self.on_signal(symbol, signal, indicators)

        except Exception as e:
            print(f"[WS] Signal check error for {symbol}: {e}")

    def start(self):
        """Start the WebSocket in a background thread."""
        import websocket

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[WS] Started Binance WebSocket for {self.symbols}")

    def _run(self):
        """Run WebSocket loop."""
        import websocket

        url = self._build_stream_url()
        ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=lambda ws, err: print(f"[WS] Error: {err}"),
            on_close=lambda ws, code, msg: print(f"[WS] Closed: {code} {msg}"),
        )

        while self.running:
            try:
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"[WS] Connection error: {e}, reconnecting in 5s...")
                time.sleep(5)

    def stop(self):
        """Stop the WebSocket."""
        self.running = False
        print("[WS] Stopped")


def run_live(symbols: List[str], intervals: List[str] = ["1h"]):
    """Run live streaming with periodic signal reports."""
    def on_signal(symbol, signal, indicators):
        direction_emoji = "🟢" if signal.direction == "BUY" else "🔴" if signal.direction == "SELL" else "⚪"
        print(f"\n{direction_emoji} [{symbol}] {signal.direction} @ ${signal.entry_price:,.4f} "
              f"(confidence: {signal.confidence:.1%})")
        print(f"   Reason: {signal.reason}")

    def on_tick(symbol, kline):
        if kline["closed"]:
            print(f"[{symbol}] {kline['interval']} closed: O={kline['open']:.4f} H={kline['high']:.4f} "
                  f"L={kline['low']:.4f} C={kline['close']:.4f}")

    ws = BinanceWebSocket(symbols, intervals, on_signal=on_signal, on_tick=on_tick)
    ws.start()

    print("\nStreaming live Binance data. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ws.stop()


if __name__ == "__main__":
    import sys
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["BTCUSD", "ETHUSD"]
    run_live(symbols, intervals=["1h", "4h"])