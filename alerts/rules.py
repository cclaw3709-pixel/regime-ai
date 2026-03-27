"""Alert rules engine for trading signals."""

from typing import Dict, List, Callable
from dataclasses import dataclass
import config


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: Callable[[Dict], bool]
    message_template: str
    direction: str  # BUY or SELL


class AlertRulesEngine:
    """Rules engine for alert generation."""

    def __init__(self):
        self.rules: List[AlertRule] = self._load_default_rules()

    def _load_default_rules(self) -> List[AlertRule]:
        """Load default alert rules."""
        rules = []

        # RSI oversold + MACD crossover
        def rsi_macd_buy(signal: Dict) -> bool:
            ind = signal.get("indicators", {})
            rsi = ind.get("rsi")
            macd_hist = ind.get("macd_hist")
            return (
                rsi is not None and
                rsi < config.SIGNAL_PARAMS["rsi_oversold"] and
                macd_hist is not None and
                macd_hist > 0
            )

        rules.append(AlertRule(
            name="RSI Oversold + MACD Positive",
            condition=rsi_macd_buy,
            message_template="BUY signal: RSI oversold ({rsi:.1f}) + MACD histogram positive ({macd_hist:.4f})",
            direction="BUY",
        ))

        # RSI overbought + MACD negative
        def rsi_macd_sell(signal: Dict) -> bool:
            ind = signal.get("indicators", {})
            rsi = ind.get("rsi")
            macd_hist = ind.get("macd_hist")
            return (
                rsi is not None and
                rsi > config.SIGNAL_PARAMS["rsi_overbought"] and
                macd_hist is not None and
                macd_hist < 0
            )

        rules.append(AlertRule(
            name="RSI Overbought + MACD Negative",
            condition=rsi_macd_sell,
            message_template="SELL signal: RSI overbought ({rsi:.1f}) + MACD histogram negative ({macd_hist:.4f})",
            direction="SELL",
        ))

        # Bollinger lower touch
        def bb_lower_touch(signal: Dict) -> bool:
            ind = signal.get("indicators", {})
            bb_percent = ind.get("bb_percent")
            return (
                bb_percent is not None and
                bb_percent < 0.05
            )

        rules.append(AlertRule(
            name="Bollinger Lower Touch",
            condition=bb_lower_touch,
            message_template="BUY signal: Price at Bollinger lower band (BB% = {bb_percent:.2f})",
            direction="BUY",
        ))

        # Bollinger upper touch
        def bb_upper_touch(signal: Dict) -> bool:
            ind = signal.get("indicators", {})
            bb_percent = ind.get("bb_percent")
            return (
                bb_percent is not None and
                bb_percent > 0.95
            )

        rules.append(AlertRule(
            name="Bollinger Upper Touch",
            condition=bb_upper_touch,
            message_template="SELL signal: Price at Bollinger upper band (BB% = {bb_percent:.2f})",
            direction="SELL",
        ))

        return rules

    def evaluate_rules(self, signal: Dict) -> List[AlertRule]:
        """Evaluate all rules against a signal.

        Args:
            signal: Signal dictionary

        Returns:
            List of triggered rules
        """
        triggered = []
        for rule in self.rules:
            try:
                if rule.condition(signal):
                    triggered.append(rule)
            except Exception:
                pass
        return triggered

    def format_alert_message(self, rule: AlertRule, signal: Dict) -> str:
        """Format alert message with signal data.

        Args:
            rule: Triggered rule
            signal: Signal dictionary

        Returns:
            Formatted message string
        """
        msg = rule.message_template
        ind = signal.get("indicators", {})

        # Replace placeholders with actual values
        replacements = {
            "{rsi}": ind.get("rsi", 0),
            "{macd_hist}": ind.get("macd_hist", 0),
            "{bb_percent}": ind.get("bb_percent", 0),
            "{symbol}": signal.get("symbol", ""),
            "{direction}": rule.direction,
            "{confidence}": signal.get("confidence", 0),
            "{entry_price}": signal.get("entry_price", 0),
        }

        for key, value in replacements.items():
            if isinstance(value, float):
                msg = msg.replace(key, f"{value:.4f}")
            else:
                msg = msg.replace(key, str(value))

        return msg


if __name__ == "__main__":
    # Test rules engine
    engine = AlertRulesEngine()

    test_signal = {
        "symbol": "BTCUSD",
        "direction": "BUY",
        "confidence": 0.75,
        "entry_price": 45000.0,
        "indicators": {
            "rsi": 28.5,
            "macd_hist": 0.0012,
            "bb_percent": 0.03,
        },
    }

    triggered = engine.evaluate_rules(test_signal)
    for rule in triggered:
        msg = engine.format_alert_message(rule, test_signal)
        print(f"Rule: {rule.name}")
        print(f"Message: {msg}")
