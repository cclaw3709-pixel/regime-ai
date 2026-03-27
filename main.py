#!/usr/bin/env python3
"""CLI entrypoint for trading signals program."""

import argparse
import sys
from typing import List
import json
from datetime import datetime

import config
from data.storage import load_or_fetch
from indicators.engine import compute_all_indicators
from signals.generator import SignalGenerator, generate_signals
from signals.ml_model import MLSignalGenerator, XGB_AVAILABLE
from alerts.rules import AlertRulesEngine
from alerts.notifier import TelegramNotifier


def run_indicator_analysis(symbol: str, interval: str) -> dict:
    """Run pure technical indicator analysis.

    Args:
        symbol: Trading symbol
        interval: Data interval

    Returns:
        Analysis results
    """
    print(f"\n{'='*60}")
    print(f"Running multi-indicator analysis for {symbol} ({interval})")
    print(f"{'='*60}\n")

    # Load and process data
    df = load_or_fetch(symbol, interval, period="30d")
    df = compute_all_indicators(df)

    # Generate signal
    generator = SignalGenerator(symbol, interval)
    generator.df = df
    signal = generator.generate_signal()

    # Display results
    print(f"\nSignal: {signal.direction}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Entry Price: ${signal.entry_price:,.4f}")
    print(f"Reason: {signal.reason}")
    print("\nIndicator Values:")
    for key, value in signal.indicators.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")

    return signal.to_dict()


def run_ml_analysis(symbol: str, interval: str) -> dict:
    """Run ML-based analysis with SHAP explanations.

    Args:
        symbol: Trading symbol
        interval: Data interval

    Returns:
        Analysis results
    """
    print(f"\n{'='*60}")
    print(f"Running ML analysis for {symbol} ({interval})")
    print(f"{'='*60}\n")

    if not XGB_AVAILABLE:
        print("xgboost or shap not installed. Falling back to indicator analysis.")
        return run_indicator_analysis(symbol, interval)

    # Train and predict
    ml = MLSignalGenerator(symbol, interval)

    try:
        metrics = ml.train()
        print(f"Model trained. Accuracy: {metrics.get('train_accuracy', 0):.2%}")

        signal = ml.generate_signal()

        print(f"\nML Signal: {signal['direction']}")
        print(f"Confidence: {signal['confidence']:.2%}")
        print(f"Entry Price: ${signal['entry_price']:,.4f}")
        print(f"Reason: {signal['reason']}")
        print("\nTop SHAP Features:")
        for name, value in signal.get("top_shap_features", []):
            sign = "+" if value > 0 else ""
            print(f"  {name}: {sign}{value:.4f}")

        return signal

    except Exception as e:
        print(f"ML analysis failed: {e}")
        print("Falling back to indicator analysis.")
        return run_indicator_analysis(symbol, interval)


def run_rules_analysis(symbol: str, interval: str) -> dict:
    """Run rules-based analysis.

    Args:
        symbol: Trading symbol
        interval: Data interval

    Returns:
        Analysis results
    """
    print(f"\n{'='*60}")
    print(f"Running rules-based analysis for {symbol} ({interval})")
    print(f"{'='*60}\n")

    # Get indicator signal
    signal_data = run_indicator_analysis(symbol, interval)
    signal_dict = {
        "symbol": symbol,
        "direction": signal_data["direction"],
        "confidence": signal_data["confidence"],
        "entry_price": signal_data["entry_price"],
        "indicators": signal_data["indicators"],
        "reason": signal_data["reason"],
    }

    # Evaluate rules
    engine = AlertRulesEngine()
    triggered = engine.evaluate_rules(signal_dict)

    if triggered:
        print(f"\n{len(triggered)} rules triggered:")
        for rule in triggered:
            msg = engine.format_alert_message(rule, signal_dict)
            print(f"  - {rule.name}: {msg}")
            signal_dict["reason"] = msg
    else:
        print("\nNo rules triggered.")

    return signal_dict


def send_alert(signal_data: dict, shap_features: List[tuple] = None):
    """Send Telegram alert if configured.

    Args:
        signal_data: Signal data dictionary
        shap_features: SHAP features for ML signals
    """
    notifier = TelegramNotifier()

    if not notifier.api_url:
        print("\n[Telegram] Bot token not configured. Skipping alert.")
        return

    if not notifier.chat_id:
        print("\n[Telegram] Chat ID not configured. Skipping alert.")
        return

    success = notifier.send_signal_alert(
        symbol=signal_data["symbol"],
        direction=signal_data["direction"],
        entry_price=signal_data["entry_price"],
        confidence=signal_data["confidence"],
        indicators=signal_data.get("indicators", {}),
        shap_features=shap_features,
        reason=signal_data.get("reason", ""),
    )

    if success:
        print("\n[Telegram] Alert sent successfully!")
    else:
        print("\n[Telegram] Alert not sent (threshold not met or HOLD signal).")


def main():
    parser = argparse.ArgumentParser(description="Trading Signal Generator")
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Trading symbol")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="multi-indicator",
        choices=["multi-indicator", "ml", "rules"],
        help="Signal generation strategy",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (overrides --symbol)",
    )
    parser.add_argument("--no-alert", action="store_true", help="Disable Telegram alerts")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    # Get symbols to process
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = [args.symbol]

    all_results = []

    for symbol in symbols:
        # Run selected strategy
        if args.strategy == "ml":
            result = run_ml_analysis(symbol, args.interval)
            shap_features = result.get("top_shap_features")
        elif args.strategy == "rules":
            result = run_rules_analysis(symbol, args.interval)
            shap_features = None
        else:
            result = run_indicator_analysis(symbol, args.interval)
            shap_features = None

        all_results.append(result)

        # Send alert if not disabled
        if not args.no_alert and result["direction"] != "HOLD":
            send_alert(result, shap_features)

    # Save output if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
