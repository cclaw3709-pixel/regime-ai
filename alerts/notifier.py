"""Telegram notifier for trading alerts."""

import requests
from typing import Optional, List, Dict
import config


class TelegramNotifier:
    """Sends alerts via Telegram Bot API."""

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send message to Telegram.

        Args:
            text: Message text
            parse_mode: "Markdown" or "HTML"

        Returns:
            True if successful, False otherwise
        """
        if not self.api_url or not self.chat_id:
            print(f"Telegram not configured. Would send: {text}")
            return False

        url = f"{self.api_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Failed to send Telegram message: {e}")
            return False

    def format_signal_alert(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        confidence: float,
        indicators: Dict[str, float],
        shap_features: Optional[List[tuple]] = None,
        reason: str = "",
    ) -> str:
        """Format signal alert message.

        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            confidence: Confidence score
            indicators: Indicator values
            shap_features: List of (feature_name, shap_value) tuples
            reason: Signal reason

        Returns:
            Formatted message string
        """
        emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
        confidence_pct = confidence * 100

        msg = f"{emoji} *{direction} Signal* {emoji}\n\n"
        msg += f"*Symbol:* {symbol}\n"
        msg += f"*Direction:* {direction}\n"
        msg += f"*Entry Price:* ${entry_price:,.4f}\n"
        msg += f"*Confidence:* {confidence_pct:.1f}%\n"

        if reason:
            msg += f"*Reason:* {reason}\n"

        # Add key indicators
        if indicators:
            msg += "\n*Key Indicators:*\n"
            if "rsi" in indicators and indicators["rsi"] is not None:
                msg += f"  RSI: {indicators['rsi']:.1f}\n"
            if "macd" in indicators and indicators["macd"] is not None:
                msg += f"  MACD: {indicators['macd']:.4f}\n"
            if "macd_hist" in indicators and indicators["macd_hist"] is not None:
                msg += f"  MACD Hist: {indicators['macd_hist']:.4f}\n"

        # Add SHAP features
        if shap_features:
            msg += "\n*Top SHAP Features:*\n"
            for name, value in shap_features:
                sign = "+" if value > 0 else ""
                msg += f"  {name}: {sign}{value:.4f}\n"

        return msg

    def send_signal_alert(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        confidence: float,
        indicators: Dict[str, float],
        shap_features: Optional[List[tuple]] = None,
        reason: str = "",
    ) -> bool:
        """Send signal alert to Telegram.

        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            confidence: Confidence score
            indicators: Indicator values
            shap_features: List of (feature_name, shap_value) tuples
            reason: Signal reason

        Returns:
            True if successful, False otherwise
        """
        # Only send for BUY or SELL
        if direction == "HOLD":
            return False

        # Check confidence threshold
        threshold = config.ALERT_THRESHOLDS.get(f"min_confidence_{direction.lower()}", 0.6)
        if confidence < threshold:
            return False

        message = self.format_signal_alert(
            symbol, direction, entry_price, confidence,
            indicators, shap_features, reason
        )

        return self.send_message(message)


if __name__ == "__main__":
    notifier = TelegramNotifier()

    # Test message
    test_msg = "🟢 *BUY Signal* 🟢\n\n*Symbol:* BTCUSD\n*Direction:* BUY\n*Entry Price:* $45,000.00\n*Confidence:* 75.0%"
    notifier.send_message(test_msg)
