"""Trading visualization dashboard with professional charting."""

import os
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.markers import MarkerStyle
import mplfinance as mpf
import pandas as pd
import numpy as np

# Professional dark theme colors
COLORS = {
    "background": "#1a1a2e",
    "grid": "#2d2d44",
    "text": "#e0e0e0",
    "buy": "#00ff88",
    "sell": "#ff4757",
    "hold": "#ffa502",
    "rsi_ob": "#ff4757",
    "rsi_os": "#00ff88",
    "rsi_line": "#e0e0e0",
    "macd_line": "#00aaff",
    "macd_signal": "#ff6b00",
    "macd_hist_pos": "#00ff88",
    "macd_hist_neg": "#ff4757",
    "bb_line": "#87ceeb",
    "price": "#ffffff",
    "volume": "#555555",
    "shap_pos": "#00ff88",
    "shap_neg": "#ff4757",
    "equity_line": "#00aaff",
}


def plot_price_with_indicators(df: pd.DataFrame, symbol: str, save_path: str) -> str:
    """Create candlestick chart with RSI, MACD, and Bollinger Bands.

    Args:
        df: DataFrame with OHLCV data and indicators
        symbol: Trading symbol
        save_path: Path to save PNG file

    Returns:
        Path to saved file
    """
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12), facecolor=COLORS["background"])

    # Create gridspec for layout
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.1)

    # Price subplot with Bollinger Bands
    ax_price = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
    ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
    ax_volume = fig.add_subplot(gs[3], sharex=ax_price)

    # Prepare data - use last 100 candles for clarity
    plot_df = df.tail(100).copy()

    # Candlestick chart colors
    candle_colors = [
        COLORS["buy"] if row["close"] >= row["open"] else COLORS["sell"]
        for _, row in plot_df.iterrows()
    ]

    # Plot candlesticks
    width = 0.6
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = COLORS["buy"] if row["close"] >= row["open"] else COLORS["sell"]
        # High-Low line (wick)
        ax_price.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
        # Open-Close body
        rect = plt.Rectangle(
            [i - width/2, row["open"]],
            width,
            row["close"] - row["open"] if row["close"] != row["open"] else 0.001,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5
        )
        ax_price.add_patch(rect)

    # Bollinger Bands
    if "bb_upper" in plot_df.columns and "bb_lower" in plot_df.columns:
        ax_price.plot(
            range(len(plot_df)),
            plot_df["bb_upper"].values,
            color=COLORS["bb_line"],
            linestyle="--",
            linewidth=1,
            label="BB Upper"
        )
        ax_price.plot(
            range(len(plot_df)),
            plot_df["bb_mid"].values,
            color=COLORS["bb_line"],
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
            label="BB Mid"
        )
        ax_price.plot(
            range(len(plot_df)),
            plot_df["bb_lower"].values,
            color=COLORS["bb_line"],
            linestyle="--",
            linewidth=1,
            label="BB Lower"
        )
        # Fill between bands
        ax_price.fill_between(
            range(len(plot_df)),
            plot_df["bb_upper"].values,
            plot_df["bb_lower"].values,
            alpha=0.1,
            color=COLORS["bb_line"]
        )

    # Price line overlay
    ax_price.plot(
        range(len(plot_df)),
        plot_df["close"].values,
        color=COLORS["price"],
        linewidth=1.5,
        alpha=0.5
    )

    ax_price.set_ylabel("Price (USD)", color=COLORS["text"])
    ax_price.tick_params(colors=COLORS["text"])
    ax_price.set_xlim(-1, len(plot_df))
    ax_price.set_xticklabels([])
    ax_price.grid(True, alpha=0.3, color=COLORS["grid"])
    ax_price.set_facecolor(COLORS["background"])
    ax_price.legend(loc="upper left", facecolor=COLORS["background"], edgecolor=COLORS["grid"])

    # RSI subplot
    if "rsi" in plot_df.columns:
        rsi = plot_df["rsi"].values
        ax_rsi.plot(range(len(plot_df)), rsi, color=COLORS["rsi_line"], linewidth=1.5)
        ax_rsi.axhline(y=70, color=COLORS["rsi_ob"], linestyle="--", linewidth=1, alpha=0.7)
        ax_rsi.axhline(y=30, color=COLORS["rsi_os"], linestyle="--", linewidth=1, alpha=0.7)
        ax_rsi.fill_between(range(len(plot_df)), rsi, 70, where=(rsi >= 70), color=COLORS["rsi_ob"], alpha=0.3)
        ax_rsi.fill_between(range(len(plot_df)), rsi, 30, where=(rsi <= 30), color=COLORS["rsi_os"], alpha=0.3)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", color=COLORS["text"])
        ax_rsi.tick_params(colors=COLORS["text"])
        ax_rsi.set_xticklabels([])
        ax_rsi.grid(True, alpha=0.3, color=COLORS["grid"])
        ax_rsi.set_facecolor(COLORS["background"])

    # MACD subplot
    if "macd" in plot_df.columns and "macd_signal" in plot_df.columns:
        macd = plot_df["macd"].values
        macd_signal = plot_df["macd_signal"].values
        macd_hist = plot_df["macd_hist"].values if "macd_hist" in plot_df.columns else np.zeros(len(plot_df))

        ax_macd.plot(range(len(plot_df)), macd, color=COLORS["macd_line"], linewidth=1.5, label="MACD")
        ax_macd.plot(range(len(plot_df)), macd_signal, color=COLORS["macd_signal"], linewidth=1.5, label="Signal")

        # Histogram
        hist_colors = [COLORS["macd_hist_pos"] if h >= 0 else COLORS["macd_hist_neg"] for h in macd_hist]
        ax_macd.bar(range(len(plot_df)), macd_hist, color=hist_colors, alpha=0.6, width=0.8)

        ax_macd.axhline(y=0, color=COLORS["text"], linestyle="-", linewidth=0.5, alpha=0.3)
        ax_macd.set_ylabel("MACD", color=COLORS["text"])
        ax_macd.tick_params(colors=COLORS["text"])
        ax_macd.set_xticklabels([])
        ax_macd.grid(True, alpha=0.3, color=COLORS["grid"])
        ax_macd.set_facecolor(COLORS["background"])
        ax_macd.legend(loc="upper left", facecolor=COLORS["background"], edgecolor=COLORS["grid"])

    # Volume subplot
    if "volume" in plot_df.columns:
        volumes = plot_df["volume"].values
        vol_colors = [COLORS["buy"] if plot_df.iloc[i]["close"] >= plot_df.iloc[i]["open"] else COLORS["sell"]
                      for i in range(len(plot_df))]
        ax_volume.bar(range(len(plot_df)), volumes, color=vol_colors, alpha=0.5, width=0.8)
        ax_volume.set_ylabel("Volume", color=COLORS["text"])
        ax_volume.tick_params(colors=COLORS["text"])
        ax_volume.set_xticklabels([])
        ax_volume.grid(True, alpha=0.3, color=COLORS["grid"])
        ax_volume.set_facecolor(COLORS["background"])

    # Format x-axis with dates
    tick_positions = range(0, len(plot_df), max(1, len(plot_df) // 8))
    tick_labels = [plot_df.index[i].strftime("%m/%d %H:%M") for i in tick_positions]
    ax_volume.set_xticks(list(tick_positions))
    ax_volume.set_xticklabels(tick_labels, rotation=45, ha="right", color=COLORS["text"])

    ax_volume.set_xlabel("Time", color=COLORS["text"])

    # Title
    fig.suptitle(f"{symbol} - Price Chart with Indicators", fontsize=16, color=COLORS["text"], y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=COLORS["background"], edgecolor="none", bbox_inches="tight")
    plt.close()

    return save_path


def plot_signal_signals(df: pd.DataFrame, signals: List[Dict], symbol: str, save_path: str) -> str:
    """Create price chart with BUY/SELL signal markers.

    Args:
        df: DataFrame with OHLCV data
        signals: List of signal dictionaries with timestamp, direction, entry_price
        symbol: Trading symbol
        save_path: Path to save PNG file

    Returns:
        Path to saved file
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 8), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Use last 100 candles
    plot_df = df.tail(100).copy()

    # Plot price line
    ax.plot(
        range(len(plot_df)),
        plot_df["close"].values,
        color=COLORS["price"],
        linewidth=2,
        label="Close Price"
    )

    # Plot candlesticks
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = COLORS["buy"] if row["close"] >= row["open"] else COLORS["sell"]
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8, alpha=0.7)
        rect = plt.Rectangle(
            [i - 0.3, row["open"]],
            0.6,
            row["close"] - row["open"] if abs(row["close"] - row["open"]) > 0.001 else 0.001,
            facecolor=color,
            edgecolor=color,
            alpha=0.5
        )
        ax.add_patch(rect)

    # Plot signals
    buy_signals = [s for s in signals if s.get("direction") == "BUY"]
    sell_signals = [s for s in signals if s.get("direction") == "SELL"]

    # Map timestamps to indices
    signal_indices = {str(ts): i for i, ts in enumerate(plot_df.index)}

    for sig in buy_signals:
        ts = sig.get("timestamp")
        if ts is None:
            continue
        # Try to find matching timestamp
        for i, idx in enumerate(plot_df.index):
            if str(idx) == str(ts) or abs((idx - ts).total_seconds()) < 3600:
                ax.scatter(
                    i,
                    sig.get("entry_price", plot_df.iloc[i]["close"]),
                    marker="^",
                    color=COLORS["buy"],
                    s=200,
                    zorder=10,
                    edgecolors="white",
                    linewidths=1.5
                )
                ax.annotate(
                    f"BUY\n{sig.get('confidence', 0):.0%}",
                    (i, sig.get("entry_price", plot_df.iloc[i]["close"])),
                    xytext=(0, 30),
                    textcoords="offset points",
                    ha="center",
                    color=COLORS["buy"],
                    fontsize=9,
                    fontweight="bold"
                )
                break

    for sig in sell_signals:
        ts = sig.get("timestamp")
        if ts is None:
            continue
        for i, idx in enumerate(plot_df.index):
            if str(idx) == str(ts) or abs((idx - ts).total_seconds()) < 3600:
                ax.scatter(
                    i,
                    sig.get("entry_price", plot_df.iloc[i]["close"]),
                    marker="v",
                    color=COLORS["sell"],
                    s=200,
                    zorder=10,
                    edgecolors="white",
                    linewidths=1.5
                )
                ax.annotate(
                    f"SELL\n{sig.get('confidence', 0):.0%}",
                    (i, sig.get("entry_price", plot_df.iloc[i]["close"])),
                    xytext=(0, -40),
                    textcoords="offset points",
                    ha="center",
                    color=COLORS["sell"],
                    fontsize=9,
                    fontweight="bold"
                )
                break

    # Formatting
    ax.set_xlabel("Time", color=COLORS["text"])
    ax.set_ylabel("Price (USD)", color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    ax.grid(True, alpha=0.3, color=COLORS["grid"])

    # X-axis labels
    tick_positions = range(0, len(plot_df), max(1, len(plot_df) // 8))
    tick_labels = [plot_df.index[i].strftime("%m/%d %H:%M") for i in tick_positions]
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", color=COLORS["text"])

    # Legend
    buy_patch = mpatches.Patch(color=COLORS["buy"], label="BUY Signal")
    sell_patch = mpatches.Patch(color=COLORS["sell"], label="SELL Signal")
    ax.legend(handles=[buy_patch, sell_patch], loc="upper left", facecolor=COLORS["background"],
              edgecolor=COLORS["grid"])

    fig.suptitle(f"{symbol} - Trading Signals", fontsize=16, color=COLORS["text"], y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=COLORS["background"], edgecolor="none", bbox_inches="tight")
    plt.close()

    return save_path


def plot_shap_features(shap_features: List[Tuple[str, float]], symbol: str, save_path: str) -> str:
    """Create horizontal bar chart of SHAP feature importances.

    Args:
        shap_features: List of (feature_name, shap_value) tuples
        symbol: Trading symbol
        save_path: Path to save PNG file

    Returns:
        Path to saved file
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, max(6, len(shap_features) * 0.5 + 2)), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    if not shap_features:
        ax.text(0.5, 0.5, "No SHAP features available", ha="center", va="center",
                color=COLORS["text"], transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Sort by absolute value
        sorted_features = sorted(shap_features, key=lambda x: abs(x[1]), reverse=True)
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        y_pos = range(len(features))
        colors = [COLORS["shap_pos"] if v >= 0 else COLORS["shap_neg"] for v in values]

        bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor="white", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            label_x = width + 0.01 if width >= 0 else width - 0.01
            ha = "left" if width >= 0 else "right"
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f"{val:+.4f}",
                    va="center", ha=ha, color=COLORS["text"], fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, color=COLORS["text"])
        ax.invert_yaxis()

        ax.set_xlabel("SHAP Value (impact on model output)", color=COLORS["text"])
        ax.axvline(x=0, color=COLORS["text"], linewidth=1, linestyle="-")
        ax.grid(True, axis="x", alpha=0.3, color=COLORS["grid"])
        ax.tick_params(colors=COLORS["text"])

        # Legend
        pos_patch = mpatches.Patch(color=COLORS["shap_pos"], label="Positive (BUY)")
        neg_patch = mpatches.Patch(color=COLORS["shap_neg"], label="Negative (SELL)")
        ax.legend(handles=[pos_patch, neg_patch], loc="lower right", facecolor=COLORS["background"],
                  edgecolor=COLORS["grid"])

    fig.suptitle(f"{symbol} - SHAP Feature Importance", fontsize=16, color=COLORS["text"], y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=COLORS["background"], edgecolor="none", bbox_inches="tight")
    plt.close()

    return save_path


def plot_equity_curve(trades: List[Dict], symbol: str, save_path: str) -> str:
    """Create equity curve from trade history.

    Args:
        trades: List of trade dictionaries with entry_price, exit_price, direction, size
        symbol: Trading symbol
        save_path: Path to save PNG file

    Returns:
        Path to saved file
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    if not trades:
        ax.text(0.5, 0.5, "No trade history available", ha="center", va="center",
                color=COLORS["text"], transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Calculate equity curve
        equity = [1.0]  # Start at 1.0 (100%)
        trade_returns = []

        for trade in trades:
            direction = trade.get("direction", "BUY")
            entry = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", entry)
            size = trade.get("size", 1.0)

            if entry == 0:
                continue

            if direction.upper() == "BUY":
                ret = (exit_price - entry) / entry * size
            else:  # SELL
                ret = (entry - exit_price) / entry * size

            trade_returns.append(ret)
            equity.append(equity[-1] * (1 + ret))

        # Plot equity curve
        x = range(len(equity))
        ax.plot(x, equity, color=COLORS["equity_line"], linewidth=2.5, label="Portfolio Value")

        # Fill under curve
        ax.fill_between(x, equity, alpha=0.3, color=COLORS["equity_line"])

        # Mark trades
        for i in range(1, len(equity)):
            color = COLORS["buy"] if trade_returns[i-1] >= 0 else COLORS["sell"]
            ax.scatter(i, equity[i], color=color, s=50, zorder=5, edgecolors="white", linewidths=0.5)

        # Add horizontal line at 1.0 (starting value)
        ax.axhline(y=1.0, color=COLORS["text"], linestyle="--", linewidth=1, alpha=0.5)

        # Calculate stats
        total_return = (equity[-1] - 1) * 100
        max_dd = min(equity) - max(equity) if equity else 0
        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0

        # Stats annotation
        stats_text = f"Total Return: {total_return:+.2f}%\n"
        stats_text += f"Win Rate: {win_rate:.1%}\n"
        stats_text += f"Trades: {len(trades)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment="top", color=COLORS["text"],
                bbox=dict(boxstyle="round", facecolor=COLORS["background"], alpha=0.8, edgecolor=COLORS["grid"]))

        ax.set_xlabel("Trade #", color=COLORS["text"])
        ax.set_ylabel("Portfolio Value", color=COLORS["text"])
        ax.tick_params(colors=COLORS["text"])
        ax.grid(True, alpha=0.3, color=COLORS["grid"])
        ax.legend(loc="lower right", facecolor=COLORS["background"], edgecolor=COLORS["grid"])

    fig.suptitle(f"{symbol} - Equity Curve", fontsize=16, color=COLORS["text"], y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=COLORS["background"], edgecolor="none", bbox_inches="tight")
    plt.close()

    return save_path


def generate_dashboard(symbol: str, interval: str, output_dir: str = "output",
                       df: Optional[pd.DataFrame] = None,
                       signals: Optional[List[Dict]] = None,
                       shap_features: Optional[List[Tuple[str, float]]] = None,
                       trades: Optional[List[Dict]] = None) -> str:
    """Generate complete HTML dashboard with all charts.

    Args:
        symbol: Trading symbol
        interval: Data interval
        output_dir: Directory to save outputs
        df: DataFrame with OHLCV data and indicators
        signals: List of signal dictionaries
        shap_features: List of (feature_name, shap_value) tuples
        trades: List of trade dictionaries

    Returns:
        Path to generated HTML file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Default data if not provided
    if df is None:
        from data.storage import load_or_fetch
        from indicators.engine import compute_all_indicators
        df = load_or_fetch(symbol, interval, period="30d")
        df = compute_all_indicators(df)

    if signals is None:
        from signals.generator import SignalGenerator
        gen = SignalGenerator(symbol, interval)
        gen.df = df
        sig = gen.generate_signal()
        signals = [sig.to_dict()]

    # Generate chart paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{symbol}_{interval}_{timestamp}"

    chart_paths = {}

    # 1. Price with indicators
    chart_paths["price_indicators"] = os.path.join(output_dir, f"{prefix}_price_indicators.png")
    plot_price_with_indicators(df, symbol, chart_paths["price_indicators"])

    # 2. Signal chart
    chart_paths["signals"] = os.path.join(output_dir, f"{prefix}_signals.png")
    plot_signal_signals(df, signals, symbol, chart_paths["signals"])

    # 3. SHAP features
    chart_paths["shap"] = os.path.join(output_dir, f"{prefix}_shap.png")
    if shap_features is None and signals:
        # Try to get SHAP from last signal
        for sig in reversed(signals):
            if "top_shap_features" in sig:
                shap_features = sig["top_shap_features"]
                break
    if shap_features:
        plot_shap_features(shap_features, symbol, chart_paths["shap"])

    # 4. Equity curve
    chart_paths["equity"] = os.path.join(output_dir, f"{prefix}_equity.png")
    if trades:
        plot_equity_curve(trades, symbol, chart_paths["equity"])

    # Calculate stats
    buy_count = len([s for s in signals if s.get("direction") == "BUY"])
    sell_count = len([s for s in signals if s.get("direction") == "SELL"])
    hold_count = len([s for s in signals if s.get("direction") == "HOLD"])
    avg_confidence = np.mean([s.get("confidence", 0) for s in signals]) if signals else 0

    # Get latest signal
    latest_signal = signals[-1] if signals else None

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} Trading Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00ff88, #00aaff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .header .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(0,0,0,0.4);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }}
        .stat-card.buy {{ border-top: 3px solid #00ff88; }}
        .stat-card.sell {{ border-top: 3px solid #ff4757; }}
        .stat-card.hold {{ border-top: 3px solid #ffa502; }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-value.buy {{ color: #00ff88; }}
        .stat-value.sell {{ color: #ff4757; }}
        .stat-value.hold {{ color: #ffa502; }}
        .stat-label {{
            color: #888;
            font-size: 0.85em;
            margin-top: 5px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 25px;
        }}
        .chart-container {{
            background: rgba(0,0,0,0.4);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .chart-container h2 {{
            margin-bottom: 15px;
            color: #fff;
            font-size: 1.2em;
        }}
        .chart-container img {{
            width: 100%;
            border-radius: 8px;
        }}
        .signal-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .signal-table th, .signal-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .signal-table th {{
            color: #888;
            font-weight: 600;
        }}
        .signal-table tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .direction-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
        }}
        .direction-badge.buy {{
            background: rgba(0,255,136,0.2);
            color: #00ff88;
        }}
        .direction-badge.sell {{
            background: rgba(255,71,87,0.2);
            color: #ff4757;
        }}
        .direction-badge.hold {{
            background: rgba(255,165,2,0.2);
            color: #ffa502;
        }}
        .latest-signal {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .latest-signal .item {{
            background: rgba(255,255,255,0.05);
            padding: 12px;
            border-radius: 8px;
        }}
        .latest-signal .item-label {{
            color: #888;
            font-size: 0.75em;
            text-transform: uppercase;
        }}
        .latest-signal .item-value {{
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 4px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{symbol} Trading Dashboard</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Interval: {interval}</div>
    </div>

    <div class="stats-grid">
        <div class="stat-card buy">
            <div class="stat-value buy">{buy_count}</div>
            <div class="stat-label">BUY Signals</div>
        </div>
        <div class="stat-card sell">
            <div class="stat-value sell">{sell_count}</div>
            <div class="stat-label">SELL Signals</div>
        </div>
        <div class="stat-card hold">
            <div class="stat-value hold">{hold_count}</div>
            <div class="stat-label">HOLD Signals</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_confidence:.1%}</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-container">
            <h2>Price Chart with RSI, MACD & Bollinger Bands</h2>
            <img src="{os.path.basename(chart_paths['price_indicators'])}" alt="Price with Indicators">
        </div>

        <div class="chart-container">
            <h2>Trading Signals</h2>
            <img src="{os.path.basename(chart_paths['signals'])}" alt="Trading Signals">
        </div>
"""

    if "shap" in chart_paths and shap_features:
        html += f"""
        <div class="chart-container">
            <h2>SHAP Feature Importance</h2>
            <img src="{os.path.basename(chart_paths['shap'])}" alt="SHAP Features">
        </div>
"""

    if "equity" in chart_paths and trades:
        html += f"""
        <div class="chart-container">
            <h2>Equity Curve</h2>
            <img src="{os.path.basename(chart_paths['equity'])}" alt="Equity Curve">
        </div>
"""

    # Latest Signal Table
    if latest_signal:
        html += f"""
        <div class="chart-container">
            <h2>Latest Signal Summary</h2>
            <span class="direction-badge {latest_signal.get('direction', 'hold').lower()}">
                {latest_signal.get('direction', 'N/A')}
            </span>
            <div class="latest-signal">
                <div class="item">
                    <div class="item-label">Entry Price</div>
                    <div class="item-value">${latest_signal.get('entry_price', 0):,.4f}</div>
                </div>
                <div class="item">
                    <div class="item-label">Confidence</div>
                    <div class="item-value">{latest_signal.get('confidence', 0):.1%}</div>
                </div>
                <div class="item">
                    <div class="item-label">Reason</div>
                    <div class="item-value" style="font-size: 0.9em;">{latest_signal.get('reason', 'N/A')}</div>
                </div>
                <div class="item">
                    <div class="item-label">Timestamp</div>
                    <div class="item-value">{latest_signal.get('timestamp', 'N/A')}</div>
                </div>
            </div>
        </div>
"""

    html += """
    <div class="footer">
        Trading Signals Dashboard | Auto-generated
    </div>
</body>
</html>
"""

    # Save HTML
    html_path = os.path.join(output_dir, f"{prefix}_dashboard.html")
    with open(html_path, "w") as f:
        f.write(html)

    print(f"[Dashboard] Generated: {html_path}")
    return html_path


def load_image_base64(path: str) -> str:
    """Load image and convert to base64 for embedding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


if __name__ == "__main__":
    # Test the dashboard
    from data.storage import load_or_fetch
    from indicators.engine import compute_all_indicators
    from signals.generator import SignalGenerator

    symbol = "BTCUSD"
    interval = "1h"

    print(f"Loading data for {symbol}...")
    df = load_or_fetch(symbol, interval, period="30d")
    df = compute_all_indicators(df)

    print("Generating signals...")
    gen = SignalGenerator(symbol, interval)
    gen.df = df
    signal = gen.generate_signal()
    signals = [signal.to_dict()]

    print("Creating dashboard...")
    html_path = generate_dashboard(symbol, interval, "output", df, signals)
    print(f"Done! Dashboard saved to: {html_path}")