"""XGBoost ML classifier for signal generation with SHAP explainability."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import config
from indicators.engine import compute_all_indicators
from data.storage import load_or_fetch

try:
    import xgboost as xgb
    import shap
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def create_labels(df: pd.DataFrame) -> pd.Series:
    """Create labels based on future returns.

    Labeling strategy:
    - Future return > buy_threshold -> BUY (2)
    - Future return < sell_threshold -> SELL (0)
    - Otherwise -> HOLD (1)

    Args:
        df: DataFrame with close prices

    Returns:
        Series with labels
    """
    labeling_cfg = config.ML_PARAMS["labeling"]
    future_periods = labeling_cfg["future_periods"]
    buy_threshold = labeling_cfg["buy_threshold"]
    sell_threshold = labeling_cfg["sell_threshold"]

    # Calculate future returns
    future_returns = df["close"].shift(-future_periods) / df["close"] - 1

    labels = pd.Series(1, index=df.index)  # Default HOLD
    labels[future_returns > buy_threshold] = 2  # BUY
    labels[future_returns < sell_threshold] = 0  # SELL

    return labels


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for ML model.

    Args:
        df: DataFrame with computed indicators

    Returns:
        Feature DataFrame
    """
    feature_cols = [
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_percent", "bb_width",
        "sma_short", "sma_long", "ema_short", "ema_long",
        "atr", "vwap",
        "close"
    ]

    # Price-based features
    df = df.copy()
    df["price_vs_sma_short"] = df["close"] / df["sma_short"] - 1
    df["price_vs_sma_long"] = df["close"] / df["sma_long"] - 1
    df["price_vs_ema_short"] = df["close"] / df["ema_short"] - 1
    df["price_vs_ema_long"] = df["close"] / df["ema_long"] - 1
    df["price_vs_bb_upper"] = df["close"] / df["bb_upper"] - 1
    df["price_vs_bb_lower"] = df["close"] / df["bb_lower"] - 1
    df["atr_percent"] = df["atr"] / df["close"]

    # Volume feature
    if "volume" in df.columns:
        df["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    additional_features = [
        "price_vs_sma_short", "price_vs_sma_long",
        "price_vs_ema_short", "price_vs_ema_long",
        "price_vs_bb_upper", "price_vs_bb_lower",
        "atr_percent", "volume_ma_ratio"
    ]

    all_features = feature_cols + additional_features
    available_features = [c for c in all_features if c in df.columns]

    return df[available_features].dropna()


class MLSignalGenerator:
    """XGBoost classifier with SHAP explainability."""

    def __init__(self, symbol: str, interval: str = "1h"):
        self.symbol = symbol
        self.interval = interval
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[Dict[str, float]] = None

    def prepare_data(self, period: str = "60d") -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data.

        Args:
            period: Historical data period

        Returns:
            Tuple of (features, labels)
        """
        df = load_or_fetch(self.symbol, self.interval, period)
        df = compute_all_indicators(df)

        labels = create_labels(df)
        features = prepare_features(df)

        # Align features and labels
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        self.feature_names = features.columns.tolist()

        return features, labels

    def train(self, period: str = "60d") -> Dict[str, float]:
        """Train XGBoost model.

        Args:
            period: Training data period

        Returns:
            Training metrics
        """
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost or shap not installed")

        X, y = self.prepare_data(period)

        model_cfg = config.ML_PARAMS["model"]
        self.model = xgb.XGBClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 6),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )

        self.model.fit(X, y)

        # Train explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Feature importances
        importances = self.model.feature_importances_
        self.feature_importances_ = dict(zip(self.feature_names, importances))

        # Return training accuracy
        train_score = self.model.score(X, y)
        return {"train_accuracy": train_score}

    def predict_with_explanation(self, df: pd.DataFrame) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Generate prediction with SHAP explanations.

        Args:
            df: DataFrame with latest data

        Returns:
            Tuple of (direction, confidence, top_features)
        """
        if self.model is None:
            self.train()

        # Prepare features
        features = prepare_features(df)
        latest_features = features.iloc[[-1]]

        # Predict
        pred = self.model.predict(latest_features)[0]
        proba = self.model.predict_proba(latest_features)[0]

        direction_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        direction = direction_map.get(pred, "HOLD")
        confidence = float(max(proba))

        # SHAP explanation
        top_features = []
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(latest_features)
            # Get SHAP values for predicted class
            shap_vals = shap_values[0][pred] if len(shap_values.shape) > 2 else shap_values[0]

            feature_shap = list(zip(self.feature_names, shap_vals))
            feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = [
                (name, float(val)) for name, val in feature_shap[:3]
            ]

        return direction, confidence, top_features

    def generate_signal(self) -> Dict:
        """Generate ML-based signal with explanation.

        Returns:
            Signal dictionary with direction, confidence, and SHAP features
        """
        df = load_or_fetch(self.symbol, self.interval, period="30d")
        df = compute_all_indicators(df)

        direction, confidence, top_features = self.predict_with_explanation(df)
        latest = df.iloc[-1]
        indicators = {
            "close": float(latest["close"]),
            "rsi": float(latest["rsi"]) if "rsi" in df.columns else None,
            "macd": float(latest["macd"]) if "macd" in df.columns else None,
        }

        return {
            "timestamp": df.index[-1],
            "symbol": self.symbol,
            "direction": direction,
            "confidence": confidence,
            "entry_price": float(latest["close"]),
            "top_shap_features": top_features,
            "indicators": indicators,
            "reason": f"ML model: {direction} with top features: {', '.join([f[0] for f in top_features])}",
        }


if __name__ == "__main__":
    if XGB_AVAILABLE:
        ml = MLSignalGenerator("BTCUSD", "1h")
        try:
            metrics = ml.train()
            print("Training metrics:", metrics)

            signal = ml.generate_signal()
            print("ML Signal:", signal)
        except Exception as e:
            print(f"ML training failed: {e}")
    else:
        print("xgboost/shap not available")
