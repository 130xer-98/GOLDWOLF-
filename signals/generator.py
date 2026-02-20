"""
GOLDWOLF — Signal Generator
Loads the trained XGBoost model and generates 3-tier trade signals.

3-Tier Signal System:
  Tier 1: 60-69% confidence → small position signal
  Tier 2: 70-79% confidence → normal position signal
  Tier 3: 80%+  confidence → sniper entry signal

Filters:
  - Cooldown: minimum SIGNAL_COOLDOWN_CANDLES M15 bars between signals
  - Session filter: configurable, default = all sessions
  - Daily loss limit: if SIGNAL_DAILY_LOSS_LIMIT signals hit SL in a day → stop
  - Minimum confidence: configurable, default 60%
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    MODEL_OUTPUT_PATH,
    SIGNAL_MIN_CONFIDENCE,
    SIGNAL_TIER1_MIN,
    SIGNAL_TIER2_MIN,
    SIGNAL_TIER3_MIN,
    SIGNAL_COOLDOWN_CANDLES,
    SIGNAL_DAILY_LOSS_LIMIT,
    LABEL_TP_PIPS,
    LABEL_SL_PIPS,
    PIP_SIZE,
)
from utils.helpers import get_logger

logger = get_logger(__name__)

# Candle DNA type names for human-readable signal reasons
DNA_NAMES = {
    -1: "Unclassified",
    0: "Clean Push",
    1: "Absorption",
    2: "Exhaustion",
    3: "Trap",
    4: "Dead Zone",
    5: "Breakout",
    6: "Fakeout",
    7: "Accumulation",
}

# Class encoding matches model/evaluator.py: 0=NO_TRADE, 1=BUY, 2=SELL
_CLASS_TO_DIR = {1: "BUY", 2: "SELL"}


class SignalGenerator:
    """
    Generates trade signals from a trained XGBoost model.

    Parameters
    ----------
    model_path            : Path to saved XGBoost model (.json).
    min_confidence        : Minimum confidence (%) to emit a signal.
    cooldown_candles      : Minimum M15 bars between signals.
    daily_loss_limit      : Max SL hits in one day before stopping.
    session_filter        : Set of session labels to trade (None = all).
    """

    def __init__(
        self,
        model_path: str = MODEL_OUTPUT_PATH,
        min_confidence: float = SIGNAL_MIN_CONFIDENCE,
        cooldown_candles: int = SIGNAL_COOLDOWN_CANDLES,
        daily_loss_limit: int = SIGNAL_DAILY_LOSS_LIMIT,
        session_filter: set[int] | None = None,
    ) -> None:
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.cooldown_candles = cooldown_candles
        self.daily_loss_limit = daily_loss_limit
        self.session_filter = session_filter  # None = all sessions

        self._model = None
        self._feature_names: list[str] | None = None
        self._last_signal_bar: int = -999  # bar index of last signal
        self._daily_losses: dict[str, int] = {}  # date → loss count
        self._bars_since_last_signal: int = 999

    def _load_model(self) -> None:
        """Lazy-load the XGBoost model from disk."""
        if self._model is not None:
            return
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(self.model_path)
            self._model = model
            # Try to get feature names from model
            try:
                self._feature_names = model.feature_names
            except Exception:
                self._feature_names = None
            logger.info("Model loaded from %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load model from %s: %s", self.model_path, e)
            raise

    def _get_tier(self, confidence: float) -> int:
        """Map confidence % to tier number."""
        if confidence >= SIGNAL_TIER3_MIN:
            return 3
        if confidence >= SIGNAL_TIER2_MIN:
            return 2
        return 1

    def _get_top_reasons(
        self,
        features: pd.Series,
        pred_class: int,
    ) -> list[str]:
        """
        Return top 3 feature names contributing to the prediction.
        Uses absolute feature values as a simple proxy (no SHAP needed).
        """
        # Map known features to human-readable reasons
        reason_map = {
            "l4_whale_footprint": "Whale footprint detected",
            "l4_trap_score": "High trap probability",
            "l4_candle_dna": f"Candle type: {DNA_NAMES.get(int(features.get('l4_candle_dna', -1)), 'Unknown')}",
            "l2_kill_zone": "Kill zone active",
            "l3_liquidity_sweep": "Liquidity sweep detected",
            "l4_multi_layer_confluence": "Multi-layer confluence",
            "l3_bos_direction": "Structure break (BOS)",
            "l3_demand_ob_distance": "Near demand order block",
            "l3_supply_ob_distance": "Near supply order block",
            "l4_volume_climax": "Institutional climax volume",
            "l4_momentum_divergence": "Momentum divergence",
            "l3_choch_flag": "Change of character (CHoCH)",
        }

        reasons = []
        if features.get("l4_whale_footprint", 0) >= 2:
            reasons.append(reason_map["l4_whale_footprint"])
        if features.get("l2_kill_zone", 0) > 0:
            reasons.append(reason_map["l2_kill_zone"])
        if features.get("l3_liquidity_sweep", 0) != 0:
            reasons.append(reason_map["l3_liquidity_sweep"])
        if features.get("l4_volume_climax", 0) == 1:
            reasons.append(reason_map["l4_volume_climax"])
        if abs(features.get("l4_multi_layer_confluence", 0)) >= 3:
            reasons.append(reason_map["l4_multi_layer_confluence"])
        if features.get("l3_bos_direction", 0) != 0:
            reasons.append(reason_map["l3_bos_direction"])
        if features.get("l3_choch_flag", 0) != 0:
            reasons.append(reason_map["l3_choch_flag"])
        if abs(features.get("l3_demand_ob_distance", 999)) < 10:
            reasons.append(reason_map["l3_demand_ob_distance"])
        if abs(features.get("l3_supply_ob_distance", 999)) < 10:
            reasons.append(reason_map["l3_supply_ob_distance"])
        if features.get("l4_momentum_divergence", 0) != 0:
            reasons.append(reason_map["l4_momentum_divergence"])

        # Return top 3 (or fewer if not enough)
        return reasons[:3] if reasons else ["Model prediction"]

    def _is_cooldown_active(self, bar_index: int) -> bool:
        """Return True if we're within the cooldown period after the last signal."""
        return (bar_index - self._last_signal_bar) < self.cooldown_candles

    def _is_daily_limit_hit(self, timestamp: pd.Timestamp) -> bool:
        """Return True if the daily loss limit has been reached for today."""
        date_str = str(timestamp.date())
        return self._daily_losses.get(date_str, 0) >= self.daily_loss_limit

    def record_sl_hit(self, timestamp: pd.Timestamp) -> None:
        """Record a stop-loss hit for the daily limit tracker."""
        date_str = str(timestamp.date())
        self._daily_losses[date_str] = self._daily_losses.get(date_str, 0) + 1

    def generate_signal(
        self,
        df_row: pd.Series,
        bar_index: int = 0,
    ) -> dict[str, Any] | None:
        """
        Generate a signal for a single M15 bar.

        Parameters
        ----------
        df_row    : pd.Series — one row from the feature DataFrame (all L1-L4 cols).
        bar_index : int — position of this bar in the current session.

        Returns
        -------
        Signal dict or None if no signal.
        """
        self._load_model()

        timestamp = df_row.name if isinstance(df_row.name, pd.Timestamp) else pd.Timestamp.now()

        # --- Session filter ---
        if self.session_filter is not None:
            sess = df_row.get("l2_session", None)
            if sess is not None and int(sess) not in self.session_filter:
                return None

        # --- Cooldown filter ---
        if self._is_cooldown_active(bar_index):
            return None

        # --- Daily loss limit ---
        if self._is_daily_limit_hit(timestamp):
            logger.debug("Daily loss limit reached — skipping signal for %s", timestamp)
            return None

        # --- Build feature vector ---
        import xgboost as xgb

        if self._feature_names:
            feat_values = np.array(
                [float(df_row.get(f, 0)) for f in self._feature_names],
                dtype=np.float32,
            )
            dmat = xgb.DMatrix(
                feat_values.reshape(1, -1),
                feature_names=self._feature_names,
            )
        else:
            # Use all numeric values as features
            feat_values = np.array(
                [float(v) for v in df_row.values if isinstance(v, (int, float, np.number))],
                dtype=np.float32,
            )
            dmat = xgb.DMatrix(feat_values.reshape(1, -1))

        # --- Predict ---
        probs = self._model.predict(dmat)[0]  # shape (3,): [P_NO_TRADE, P_BUY, P_SELL]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100

        # --- Filter: NO_TRADE or insufficient confidence ---
        if pred_class == 0:
            return None
        if confidence < self.min_confidence:
            return None

        # --- Build signal object ---
        direction = _CLASS_TO_DIR[pred_class]
        tier = self._get_tier(confidence)
        entry_price = float(df_row.get("m15_close", 0.0))
        tp_dist = LABEL_TP_PIPS * PIP_SIZE
        sl_dist = LABEL_SL_PIPS * PIP_SIZE

        if direction == "BUY":
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + tp_dist
        else:
            stop_loss = entry_price + sl_dist
            take_profit = entry_price - tp_dist

        rr = tp_dist / sl_dist if sl_dist > 0 else 0.0
        dna_val = int(df_row.get("l4_candle_dna", -1))

        signal: dict[str, Any] = {
            "timestamp": timestamp,
            "direction": direction,
            "confidence": round(confidence, 2),
            "tier": tier,
            "entry_price": round(entry_price, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "risk_reward": round(rr, 2),
            "top_reasons": self._get_top_reasons(df_row, pred_class),
            "candle_dna": DNA_NAMES.get(dna_val, "Unknown"),
        }

        # Update cooldown tracker
        self._last_signal_bar = bar_index

        logger.info(
            "Signal: %s @ %.5f | Confidence: %.1f%% (Tier %d)",
            direction, entry_price, confidence, tier,
        )
        return signal

    def process_dataframe(
        self,
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Generate signals for all bars in *df*.

        Parameters
        ----------
        df : pd.DataFrame — full feature DataFrame.

        Returns
        -------
        list of signal dicts.
        """
        signals = []
        for i, (_, row) in enumerate(df.iterrows()):
            sig = self.generate_signal(row, bar_index=i)
            if sig is not None:
                signals.append(sig)
        logger.info("Generated %d signals from %d bars", len(signals), len(df))
        return signals
