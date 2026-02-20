"""
GOLDWOLF — Signal Generator Tests
Tests for signal generation, tier classification, cooldown, and Telegram formatting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.generator import SignalGenerator, DNA_NAMES
from config.settings import (
    SIGNAL_TIER1_MIN,
    SIGNAL_TIER2_MIN,
    SIGNAL_TIER3_MIN,
    SIGNAL_COOLDOWN_CANDLES,
    SIGNAL_DAILY_LOSS_LIMIT,
)


# ---------------------------------------------------------------------------
# Helper: build mock feature row
# ---------------------------------------------------------------------------

def make_feature_row(
    direction: str = "BUY",
    confidence: float = 0.75,
    timestamp: str = "2024-01-15 10:00",
    kill_zone: int = 1,
    session: int = 1,
    whale_footprint: int = 2,
) -> pd.Series:
    """
    Build a minimal feature Series that mimics one row of the full feature DataFrame.
    """
    ts = pd.Timestamp(timestamp)
    data = {
        # OHLCV
        "m15_close": 2000.0,
        "m15_open": 1999.0,
        "m15_high": 2001.0,
        "m15_low": 1998.0,
        # L1 features
        "l1_custom_volume": 12.0,
        "l1_volatility_energy": 3.0,
        "l1_price_velocity": 0.2,
        "l1_reversal_count": 2.0,
        "l1_early_late_ratio": 0.5,
        "l1_price_acceleration": 0.05,
        "l1_absorption_count": 3.0,
        "l1_absorption_intensity": 0.35,
        # L2 features
        "l2_session": float(session),
        "l2_kill_zone": float(kill_zone),
        # L3 features
        "l3_liquidity_sweep": 0.0,
        "l3_demand_ob_distance": 5.0,
        "l3_supply_ob_distance": 15.0,
        "l3_bos_direction": 1.0,
        "l3_choch_flag": 0.0,
        "l3_premium_discount": -1.0,
        # L4 features
        "l4_whale_footprint": float(whale_footprint),
        "l4_trap_score": 0.0,
        "l4_candle_dna": 0.0,
        "l4_momentum_divergence": 0.0,
        "l4_consecutive_bias": 3.0,
        "l4_volume_climax": 0.0,
        "l4_range_compression": 1.0,
        "l4_session_continuation": 0.6,
        "l4_multi_layer_confluence": 3.0,
        "l4_time_volatility_regime": 1.0,
    }
    return pd.Series(data, name=ts)


class MockModel:
    """Mock XGBoost Booster that returns predetermined probabilities."""

    def __init__(self, probs: list[float]) -> None:
        """probs = [P_NO_TRADE, P_BUY, P_SELL]"""
        self._probs = np.array(probs, dtype=np.float32)
        self.feature_names = None

    def predict(self, dmat: object) -> np.ndarray:
        return self._probs.reshape(1, -1)

    def load_model(self, path: str) -> None:
        pass


def make_generator_with_mock(probs: list[float], **kwargs) -> SignalGenerator:
    """Create a SignalGenerator with a mock model injected."""
    gen = SignalGenerator(**kwargs)
    gen._model = MockModel(probs)
    gen._feature_names = None
    return gen


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

class TestTierClassification:
    def test_tier1_range(self):
        """60-69% confidence → Tier 1."""
        gen = SignalGenerator.__new__(SignalGenerator)
        assert gen._get_tier.__func__(gen, 60.0) == 1
        assert gen._get_tier.__func__(gen, 69.9) == 1

    def test_tier2_range(self):
        """70-79% confidence → Tier 2."""
        gen = SignalGenerator.__new__(SignalGenerator)
        assert gen._get_tier.__func__(gen, 70.0) == 2
        assert gen._get_tier.__func__(gen, 79.9) == 2

    def test_tier3_range(self):
        """80%+ confidence → Tier 3."""
        gen = SignalGenerator.__new__(SignalGenerator)
        assert gen._get_tier.__func__(gen, 80.0) == 3
        assert gen._get_tier.__func__(gen, 99.9) == 3


# ---------------------------------------------------------------------------
# Signal generation with mock model
# ---------------------------------------------------------------------------

class TestSignalGeneration:
    def test_buy_signal_generated(self):
        """Mock model predicting BUY at 82% → signal with direction=BUY, tier=3."""
        gen = make_generator_with_mock([0.05, 0.82, 0.13])
        row = make_feature_row()
        sig = gen.generate_signal(row, bar_index=10)
        assert sig is not None
        assert sig["direction"] == "BUY"
        assert sig["tier"] == 3
        assert sig["confidence"] == pytest.approx(82.0, abs=0.1)

    def test_sell_signal_generated(self):
        """Mock model predicting SELL at 75% → signal with direction=SELL, tier=2."""
        gen = make_generator_with_mock([0.05, 0.20, 0.75])
        row = make_feature_row()
        sig = gen.generate_signal(row, bar_index=10)
        assert sig is not None
        assert sig["direction"] == "SELL"
        assert sig["tier"] == 2

    def test_no_signal_when_no_trade_dominant(self):
        """Mock model predicting NO_TRADE → no signal."""
        gen = make_generator_with_mock([0.90, 0.05, 0.05])
        row = make_feature_row()
        sig = gen.generate_signal(row, bar_index=10)
        assert sig is None

    def test_no_signal_below_confidence_threshold(self):
        """Prediction below min_confidence → no signal."""
        gen = make_generator_with_mock([0.45, 0.35, 0.20], min_confidence=60.0)
        row = make_feature_row()
        sig = gen.generate_signal(row, bar_index=10)
        assert sig is None

    def test_signal_has_required_keys(self):
        """Signal dict must contain all required keys."""
        required_keys = [
            "timestamp", "direction", "confidence", "tier",
            "entry_price", "stop_loss", "take_profit", "risk_reward",
            "top_reasons", "candle_dna",
        ]
        gen = make_generator_with_mock([0.05, 0.82, 0.13])
        row = make_feature_row()
        sig = gen.generate_signal(row, bar_index=10)
        assert sig is not None
        for key in required_keys:
            assert key in sig, f"Missing key: {key}"

    def test_stop_loss_take_profit_direction_correct(self):
        """For BUY: SL < entry < TP. For SELL: TP < entry < SL."""
        gen_buy = make_generator_with_mock([0.05, 0.85, 0.10])
        row = make_feature_row()
        sig_buy = gen_buy.generate_signal(row, bar_index=10)
        assert sig_buy is not None
        assert sig_buy["stop_loss"] < sig_buy["entry_price"] < sig_buy["take_profit"]

        gen_sell = make_generator_with_mock([0.05, 0.10, 0.85])
        sig_sell = gen_sell.generate_signal(row, bar_index=20)
        assert sig_sell is not None
        assert sig_sell["take_profit"] < sig_sell["entry_price"] < sig_sell["stop_loss"]


# ---------------------------------------------------------------------------
# Cooldown filter
# ---------------------------------------------------------------------------

class TestCooldownFilter:
    def test_cooldown_blocks_signal(self):
        """Signal should be blocked within cooldown period."""
        gen = make_generator_with_mock([0.05, 0.85, 0.10], cooldown_candles=3)
        row = make_feature_row()

        sig1 = gen.generate_signal(row, bar_index=10)
        assert sig1 is not None  # first signal passes

        sig2 = gen.generate_signal(row, bar_index=11)
        assert sig2 is None  # blocked by cooldown

        sig3 = gen.generate_signal(row, bar_index=12)
        assert sig3 is None  # still in cooldown

    def test_signal_allowed_after_cooldown(self):
        """Signal should be allowed after cooldown period expires."""
        cooldown = 2
        gen = make_generator_with_mock([0.05, 0.85, 0.10], cooldown_candles=cooldown)
        row = make_feature_row()

        sig1 = gen.generate_signal(row, bar_index=10)
        assert sig1 is not None

        # Bar 10 + cooldown = bar 12 is the first allowed bar
        sig2 = gen.generate_signal(row, bar_index=10 + cooldown)
        assert sig2 is not None


# ---------------------------------------------------------------------------
# Daily loss limit filter
# ---------------------------------------------------------------------------

class TestDailyLossLimit:
    def test_daily_loss_limit_blocks_signals(self):
        """After SIGNAL_DAILY_LOSS_LIMIT SL hits, no more signals for the day."""
        limit = 3
        gen = make_generator_with_mock([0.05, 0.85, 0.10], daily_loss_limit=limit)
        row = make_feature_row(timestamp="2024-01-15 10:00")

        # Record the maximum allowed losses
        ts = row.name
        for _ in range(limit):
            gen.record_sl_hit(ts)

        # Now the daily limit should be reached
        sig = gen.generate_signal(row, bar_index=999)
        assert sig is None

    def test_different_days_not_blocked(self):
        """SL hits on one day don't block signals on the next day."""
        limit = 1
        gen = make_generator_with_mock([0.05, 0.85, 0.10], daily_loss_limit=limit)

        day1_ts = pd.Timestamp("2024-01-15 10:00")
        day2_ts = pd.Timestamp("2024-01-16 10:00")

        gen.record_sl_hit(day1_ts)  # max out day 1

        row = make_feature_row(timestamp="2024-01-16 10:00")
        sig = gen.generate_signal(row, bar_index=999)
        assert sig is not None  # day 2 should not be blocked


# ---------------------------------------------------------------------------
# Telegram message formatting
# ---------------------------------------------------------------------------

class TestTelegramFormatting:
    def test_fancy_format_contains_key_fields(self):
        """Fancy Telegram format should contain direction, confidence, entry, SL, TP."""
        from live.telegram_bot import _format_signal_fancy

        sig = {
            "timestamp": pd.Timestamp("2024-01-15 10:15"),
            "direction": "BUY",
            "confidence": 82.0,
            "tier": 3,
            "entry_price": 2000.50,
            "stop_loss": 1990.50,
            "take_profit": 2015.50,
            "risk_reward": 1.5,
            "top_reasons": ["Whale footprint detected", "Kill zone active"],
            "candle_dna": "Clean Push",
        }

        message = _format_signal_fancy(sig)
        assert "BUY" in message
        assert "82" in message
        assert "2000" in message
        assert "TIER 3" in message.upper() or "Tier 3" in message or "tier 3" in message.lower()

    def test_plain_format_contains_key_fields(self):
        """Plain Telegram format should contain direction and price."""
        from live.telegram_bot import _format_signal_plain

        sig = {
            "timestamp": pd.Timestamp("2024-01-15 10:15"),
            "direction": "SELL",
            "confidence": 75.0,
            "tier": 2,
            "entry_price": 2000.50,
            "stop_loss": 2010.50,
            "take_profit": 1985.50,
        }

        message = _format_signal_plain(sig)
        assert "SELL" in message
        assert "2000" in message

    def test_send_signal_logs_to_file_when_unconfigured(self, tmp_path):
        """send_signal should log to file when Telegram is not configured."""
        import os
        from live.telegram_bot import send_signal, _log_to_file
        from config import settings as s

        # Temporarily override credentials to empty
        original_token = s.TELEGRAM_BOT_TOKEN
        original_chat = s.TELEGRAM_CHAT_ID
        s.TELEGRAM_BOT_TOKEN = ""
        s.TELEGRAM_CHAT_ID = ""

        sig = {
            "timestamp": pd.Timestamp("2024-01-15 10:15"),
            "direction": "BUY",
            "confidence": 82.0,
            "tier": 3,
            "entry_price": 2000.50,
            "stop_loss": 1990.50,
            "take_profit": 2015.50,
            "risk_reward": 1.5,
            "top_reasons": ["Test"],
            "candle_dna": "Clean Push",
        }

        result = send_signal(sig)
        # Should return False (logged to file, not sent via Telegram)
        assert result is False

        # Restore
        s.TELEGRAM_BOT_TOKEN = original_token
        s.TELEGRAM_CHAT_ID = original_chat
