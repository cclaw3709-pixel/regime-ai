"""Visualization package for trading signals dashboard."""

from visualization.dashboard import (
    plot_price_with_indicators,
    plot_signal_signals,
    plot_shap_features,
    plot_equity_curve,
    generate_dashboard,
)

__all__ = [
    "plot_price_with_indicators",
    "plot_signal_signals",
    "plot_shap_features",
    "plot_equity_curve",
    "generate_dashboard",
]