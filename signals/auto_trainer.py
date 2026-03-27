"""Auto-retraining scheduler for the ML model."""

import os
import time
import threading
import schedule
from datetime import datetime
from typing import Optional, List
import config
from signals.ml_model import MLSignalGenerator
import logging

logger = logging.getLogger(__name__)


def _get_visualization_module():
    """Lazy-load visualization to avoid circular imports."""
    try:
        from visualization.dashboard import plot_shap_features, generate_dashboard
        return plot_shap_features, generate_dashboard
    except ImportError:
        return None, None

logger = logging.getLogger(__name__)


class AutoTrainer:
    """Automatically retrains the ML model on a schedule.

    Retrains:
    - Every N hours (configurable, default: every 6 hours)
    - On market close (16:00 EST for US markets)
    - When drift is detected (optional)
    """

    def __init__(self, symbols: List[str], intervals: List[str] = ["1h"],
                 retrain_interval_hours: int = 6,
                 retrain_on_close: bool = True):
        self.symbols = symbols
        self.intervals = intervals
        self.retrain_interval_hours = retrain_interval_hours
        self.retrain_on_close = retrain_on_close
        self.models: dict = {}  # (symbol, interval) → MLSignalGenerator
        self.last_train_time: dict = {}  # (symbol, interval) → datetime
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._results: dict = {}  # (symbol, interval) → metrics

    def _retrain(self, symbol: str, interval: str) -> dict:
        """Retrain model for a symbol/interval pair."""
        key = (symbol, interval)
        print(f"[AutoTrain] Retraining {symbol} ({interval})...")

        try:
            ml = MLSignalGenerator(symbol, interval)
            metrics = ml.train(period="90d")
            self.models[key] = ml
            self.last_train_time[key] = datetime.now()
            self._results[key] = metrics
            print(f"[AutoTrain] ✓ {symbol} ({interval}) trained. "
                  f"Accuracy: {metrics.get('train_accuracy', 0):.2%}")

            # Save SHAP visualization after successful retrain
            self.save_visualization(symbol, interval)

            return metrics
        except Exception as e:
            print(f"[AutoTrain] ✗ {symbol} ({interval}) failed: {e}")
            return {"error": str(e)}

    def _schedule_retrain(self):
        """Schedule all retraining jobs."""
        # Periodic retraining
        schedule.every(self.retrain_interval_hours).hours.do(
            lambda: [self._retrain(s, i) for s in self.symbols for i in self.intervals]
        )

        if self.retrain_on_close:
            # Daily at US market close (16:00 EST = 20:00 UTC)
            schedule.every().day.at("20:00").do(
                lambda: [self._retrain(s, i) for s in self.symbols for i in self.intervals]
            )

    def _run_scheduler(self):
        """Run the schedule loop in background."""
        self._schedule_retrain()
        print(f"[AutoTrain] Scheduler started. Retraining every {self.retrain_interval_hours}h"
              f"{' and daily at 20:00 UTC' if self.retrain_on_close else ''}")
        while self.running:
            schedule.run_pending()
            time.sleep(60)

    def start(self):
        """Start the auto-trainer in a background thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        print(f"[AutoTrain] Started for {self.symbols}")

    def stop(self):
        """Stop the auto-trainer."""
        self.running = False
        print("[AutoTrain] Stopped")

    def get_model(self, symbol: str, interval: str) -> Optional[MLSignalGenerator]:
        """Get trained model for symbol/interval."""
        return self.models.get((symbol, interval))

    def get_last_train_time(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get last training time for symbol/interval."""
        return self.last_train_time.get((symbol, interval))

    def get_results(self, symbol: str, interval: str) -> Optional[dict]:
        """Get training results for symbol/interval."""
        return self._results.get((symbol, interval))

    def save_visualization(self, symbol: str, interval: str, output_dir: str = "output") -> Optional[str]:
        """Plot and save SHAP features visualization after retraining.

        Args:
            symbol: Trading symbol
            interval: Data interval
            output_dir: Directory to save visualization

        Returns:
            Path to saved chart or None if visualization unavailable
        """
        plot_shap_features, _ = _get_visualization_module()
        if plot_shap_features is None:
            logger.warning("Visualization module not available")
            return None

        try:
            model = self.get_model(symbol, interval)
            if model is None or not hasattr(model, "feature_importances_"):
                logger.warning(f"No trained model available for {symbol}/{interval}")
                return None

            # Get feature importances as SHAP-like format
            shap_features = [
                (name, float(imp))
                for name, imp in sorted(
                    model.feature_importances_.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            ][:10]  # Top 10 features

            if not shap_features:
                logger.warning(f"No feature importances for {symbol}/{interval}")
                return None

            save_path = f"{output_dir}/{symbol}_{interval}_shap.png"
            os.makedirs(output_dir, exist_ok=True)

            plot_shap_features(shap_features, symbol, save_path)
            print(f"[AutoTrain] Saved SHAP visualization: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save visualization for {symbol}/{interval}: {e}")
            return None


if __name__ == "__main__":
    import sys

    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["BTCUSD", "ETHUSD"]
    trainer = AutoTrainer(symbols, intervals=["1h", "4h"], retrain_interval_hours=6)
    trainer.start()

    print("\nAuto-trainer running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        trainer.stop()