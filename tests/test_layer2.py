"""
GOLDWOLF — Layer 2 (Time DNA) Feature Tests
Uses synthetic M15 DataFrames with known timestamps and prices to verify
each Time DNA feature independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.layer2 import compute_layer2_features
from config.settings import (
    EPSILON,
    PIP_SIZE,
    VOL_SPIKE_CAP,
    SESSION_VOL_RANK_WINDOW,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal M15 DataFrame
# ---------------------------------------------------------------------------

def make_m15_df(
    timestamps: list[str],
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
    vol_energy: list[float] | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame that mimics the Phase 1 output structure.

    Parameters
    ----------
    timestamps  : ISO-format strings, e.g. ["2023-01-02 00:00", ...]
    closes      : m15_close values
    highs       : m15_high  (defaults to close + 1)
    lows        : m15_low   (defaults to close - 1)
    opens       : m15_open  (defaults to close)
    vol_energy  : l1_volatility_energy (defaults to 1.0)
    """
    n = len(timestamps)
    closes = list(closes)
    highs = highs if highs is not None else [c + 1.0 for c in closes]
    lows = lows if lows is not None else [c - 1.0 for c in closes]
    opens = opens if opens is not None else list(closes)
    ve = vol_energy if vol_energy is not None else [1.0] * n

    idx = pd.to_datetime(timestamps)
    return pd.DataFrame(
        {
            "m15_open": np.array(opens, dtype=np.float64),
            "m15_high": np.array(highs, dtype=np.float64),
            "m15_low": np.array(lows, dtype=np.float64),
            "m15_close": np.array(closes, dtype=np.float64),
            "m15_volume": np.zeros(n, dtype=np.float64),
            "l1_volatility_energy": np.array(ve, dtype=np.float64),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Feature 1: l2_session
# ---------------------------------------------------------------------------

class TestSessionMapping:
    def test_asian_hours(self):
        """00:00–07:45 GMT → session 0 (Asian)."""
        ts = [f"2023-01-02 {h:02d}:{m:02d}" for h in range(0, 8) for m in [0, 15, 30, 45]]
        df = make_m15_df(ts, [1800.0] * len(ts))
        out = compute_layer2_features(df)
        assert (out["l2_session"] == 0).all()

    def test_london_hours(self):
        """08:00–15:45 GMT → session 1 (London)."""
        ts = [f"2023-01-02 {h:02d}:{m:02d}" for h in range(8, 16) for m in [0, 15, 30, 45]]
        df = make_m15_df(ts, [1800.0] * len(ts))
        out = compute_layer2_features(df)
        assert (out["l2_session"] == 1).all()

    def test_ny_hours(self):
        """16:00–23:45 GMT → session 2 (NY)."""
        ts = [f"2023-01-02 {h:02d}:{m:02d}" for h in range(16, 24) for m in [0, 15, 30, 45]]
        df = make_m15_df(ts, [1800.0] * len(ts))
        out = compute_layer2_features(df)
        assert (out["l2_session"] == 2).all()

    def test_session_boundary(self):
        """Boundary bars get correct session."""
        ts = ["2023-01-02 07:45", "2023-01-02 08:00", "2023-01-02 15:45", "2023-01-02 16:00"]
        df = make_m15_df(ts, [1800.0] * 4)
        out = compute_layer2_features(df)
        expected = [0, 1, 1, 2]
        assert list(out["l2_session"]) == expected


# ---------------------------------------------------------------------------
# Feature 2: l2_session_overlap
# ---------------------------------------------------------------------------

class TestSessionOverlap:
    def test_inside_overlap(self):
        """13:00–15:45 → overlap = 1."""
        ts = [f"2023-01-02 {h:02d}:{m:02d}" for h in range(13, 16) for m in [0, 15, 30, 45]]
        df = make_m15_df(ts, [1800.0] * len(ts))
        out = compute_layer2_features(df)
        assert (out["l2_session_overlap"] == 1).all()

    def test_outside_overlap(self):
        """08:00–12:45 → overlap = 0."""
        ts = [f"2023-01-02 {h:02d}:{m:02d}" for h in range(8, 13) for m in [0, 15, 30, 45]]
        df = make_m15_df(ts, [1800.0] * len(ts))
        out = compute_layer2_features(df)
        assert (out["l2_session_overlap"] == 0).all()

    def test_boundary_16(self):
        """16:00 GMT is NOT in overlap window."""
        ts = ["2023-01-02 15:45", "2023-01-02 16:00"]
        df = make_m15_df(ts, [1800.0, 1800.0])
        out = compute_layer2_features(df)
        assert out["l2_session_overlap"].iloc[0] == 1  # 15:45 in overlap
        assert out["l2_session_overlap"].iloc[1] == 0  # 16:00 NOT in overlap


# ---------------------------------------------------------------------------
# Feature 3: l2_kill_zone
# ---------------------------------------------------------------------------

class TestKillZone:
    def test_london_open_kill_zone(self):
        ts = ["2023-01-02 08:00", "2023-01-02 08:30", "2023-01-02 08:45"]
        df = make_m15_df(ts, [1800.0] * 3)
        out = compute_layer2_features(df)
        assert (out["l2_kill_zone"] == 1).all()

    def test_ny_open_kill_zone(self):
        ts = ["2023-01-02 13:00", "2023-01-02 13:30", "2023-01-02 13:45"]
        df = make_m15_df(ts, [1800.0] * 3)
        out = compute_layer2_features(df)
        assert (out["l2_kill_zone"] == 2).all()

    def test_london_close_kill_zone(self):
        ts = ["2023-01-02 15:30", "2023-01-02 15:45", "2023-01-02 16:00", "2023-01-02 16:15"]
        df = make_m15_df(ts, [1800.0] * 4)
        out = compute_layer2_features(df)
        assert (out["l2_kill_zone"] == 3).all()

    def test_no_kill_zone(self):
        ts = ["2023-01-02 10:00", "2023-01-02 11:00", "2023-01-02 17:00"]
        df = make_m15_df(ts, [1800.0] * 3)
        out = compute_layer2_features(df)
        assert (out["l2_kill_zone"] == 0).all()


# ---------------------------------------------------------------------------
# Feature 4: l2_day_of_week
# ---------------------------------------------------------------------------

class TestDayOfWeek:
    def test_known_weekdays(self):
        # 2023-01-02 = Monday, 2023-01-06 = Friday
        ts = [
            "2023-01-02 10:00",  # Monday
            "2023-01-03 10:00",  # Tuesday
            "2023-01-04 10:00",  # Wednesday
            "2023-01-05 10:00",  # Thursday
            "2023-01-06 10:00",  # Friday
        ]
        df = make_m15_df(ts, [1800.0] * 5)
        out = compute_layer2_features(df)
        assert list(out["l2_day_of_week"]) == [0, 1, 2, 3, 4]

    def test_hour_values(self):
        ts = ["2023-01-02 00:00", "2023-01-02 12:00", "2023-01-02 23:45"]
        df = make_m15_df(ts, [1800.0] * 3)
        out = compute_layer2_features(df)
        assert list(out["l2_hour"]) == [0, 12, 23]


# ---------------------------------------------------------------------------
# Feature 6: l2_distance_from_session_open
# ---------------------------------------------------------------------------

class TestDistanceFromSessionOpen:
    def test_first_candle_of_session_is_zero(self):
        """First candle of a session: distance = 0 (open == open)."""
        ts = ["2023-01-02 08:00"]  # London session start
        opens = [1800.0]
        closes = [1800.0]
        df = make_m15_df(ts, closes, opens=opens)
        out = compute_layer2_features(df)
        assert out["l2_distance_from_session_open"].iloc[0] == pytest.approx(0.0, abs=1e-3)

    def test_positive_when_close_above_session_open(self):
        """Session open = 1800; later close = 1802 → +20 pips."""
        ts = [
            "2023-01-02 08:00",
            "2023-01-02 08:15",
            "2023-01-02 08:30",
        ]
        opens = [1800.0, 1800.5, 1801.0]
        closes = [1800.0, 1800.5, 1802.0]
        df = make_m15_df(ts, closes, opens=opens)
        out = compute_layer2_features(df)
        # Third bar: (close - session_open) / PIP_SIZE = (1802 - 1800) / 0.1 = 20
        assert out["l2_distance_from_session_open"].iloc[2] == pytest.approx(20.0, abs=1e-3)

    def test_negative_when_close_below_session_open(self):
        """Session bearish: close drops below session open → negative pips."""
        ts = [
            "2023-01-02 08:00",
            "2023-01-02 08:15",
        ]
        opens = [1800.0, 1799.0]
        closes = [1800.0, 1799.0]
        df = make_m15_df(ts, closes, opens=opens)
        out = compute_layer2_features(df)
        assert out["l2_distance_from_session_open"].iloc[1] < 0

    def test_resets_at_session_boundary(self):
        """Distance should reset to ~0 at the start of a new session."""
        ts = [
            "2023-01-02 07:45",  # Asian (last bar)
            "2023-01-02 08:00",  # London opens — new session
        ]
        opens = [1800.0, 1810.0]
        closes = [1805.0, 1810.0]
        df = make_m15_df(ts, closes, opens=opens)
        out = compute_layer2_features(df)
        # London open bar: distance = (close - m15_open) / PIP_SIZE = (1810-1810)/0.1 = 0
        assert out["l2_distance_from_session_open"].iloc[1] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Feature 7: l2_session_position
# ---------------------------------------------------------------------------

class TestSessionPosition:
    def test_at_session_high(self):
        """Close == session high → position ≈ 1."""
        ts = [
            "2023-01-02 08:00",
            "2023-01-02 08:15",
            "2023-01-02 08:30",
        ]
        opens = [1800.0] * 3
        closes = [1800.0, 1802.0, 1804.0]
        highs = [1801.0, 1803.0, 1804.0]
        lows = [1799.0, 1799.0, 1799.0]
        df = make_m15_df(ts, closes, highs=highs, lows=lows, opens=opens)
        out = compute_layer2_features(df)
        # Session high = 1804, low = 1799; pos = (1804-1799)/(1804-1799+ε) ≈ 1
        assert out["l2_session_position"].iloc[2] == pytest.approx(1.0, abs=1e-3)

    def test_at_session_low(self):
        """Close == session low → position ≈ 0."""
        ts = [
            "2023-01-02 08:00",
            "2023-01-02 08:15",
        ]
        opens = [1805.0, 1804.0]
        closes = [1804.0, 1800.0]
        highs = [1806.0, 1805.0]
        lows = [1803.0, 1800.0]
        df = make_m15_df(ts, closes, highs=highs, lows=lows, opens=opens)
        out = compute_layer2_features(df)
        assert out["l2_session_position"].iloc[1] == pytest.approx(0.0, abs=1e-3)

    def test_first_candle_is_zero_or_one(self):
        """Single candle session: high == low == close → position = 0 (or 1)."""
        ts = ["2023-01-02 08:00"]
        df = make_m15_df(ts, [1800.0], highs=[1800.0], lows=[1800.0])
        out = compute_layer2_features(df)
        pos = out["l2_session_position"].iloc[0]
        assert 0.0 <= float(pos) <= 1.0

    def test_range_is_clipped(self):
        """All values must lie in [0, 1]."""
        ts = [
            f"2023-01-02 08:{m:02d}" for m in [0, 15, 30, 45]
        ]
        closes = [1800.0, 1810.0, 1795.0, 1805.0]
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]
        df = make_m15_df(ts, closes, highs=highs, lows=lows)
        out = compute_layer2_features(df)
        assert (out["l2_session_position"] >= 0).all()
        assert (out["l2_session_position"] <= 1).all()


# ---------------------------------------------------------------------------
# Feature 8: l2_time_since_vol_spike
# ---------------------------------------------------------------------------

class TestTimeSinceVolSpike:
    def test_capped_at_vsc(self):
        """With no spikes at all, value should be capped at VOL_SPIKE_CAP."""
        ts = [f"2023-01-02 08:{m:02d}" for m in range(0, 60, 15)]
        df = make_m15_df(ts, [1800.0] * 4, vol_energy=[1.0] * 4)
        out = compute_layer2_features(df)
        assert (out["l2_time_since_vol_spike"] <= VOL_SPIKE_CAP).all()

    def test_resets_after_spike(self):
        """Value resets near 0 immediately after a spike."""
        # Build 25 bars of baseline energy, then a huge spike on bar 25
        n_base = 25
        base = [1.0] * n_base
        spike = [1000.0]
        ve = base + spike
        ts = [f"2023-01-02 {8 + i // 4:02d}:{(i % 4) * 15:02d}" for i in range(n_base + 1)]
        df = make_m15_df(ts, [1800.0] * (n_base + 1), vol_energy=ve)
        out = compute_layer2_features(df)
        # Right after the spike
        assert out["l2_time_since_vol_spike"].iloc[-1] == pytest.approx(0.0, abs=1.0)

    def test_counts_up_after_spike(self):
        """After a spike, time_since_vol_spike increases by 1 each bar."""
        n_base = 22
        base = [1.0] * n_base
        spike = [1000.0]
        post = [1.0, 1.0, 1.0]
        ve = base + spike + post
        n = len(ve)
        ts = [f"2023-01-02 {8 + i // 4:02d}:{(i % 4) * 15:02d}" for i in range(n)]
        df = make_m15_df(ts, [1800.0] * n, vol_energy=ve)
        out = compute_layer2_features(df)
        vals = out["l2_time_since_vol_spike"].values
        # After spike: positions spike_idx+1, +2, +3 should increase by 1
        spike_idx = n_base
        assert vals[spike_idx + 1] == pytest.approx(vals[spike_idx] + 1, abs=1.0)
        assert vals[spike_idx + 2] == pytest.approx(vals[spike_idx + 1] + 1, abs=1.0)


# ---------------------------------------------------------------------------
# Feature 9: l2_session_volatility_rank
# ---------------------------------------------------------------------------

class TestSessionVolatilityRank:
    def test_high_energy_gets_high_rank(self):
        """A very high energy bar after low-energy history should rank near 1."""
        n = 20
        ts = [f"2023-01-02 08:{(i % 4) * 15:02d}" if i < 4 else
              f"2023-01-02 {8 + i // 4:02d}:{(i % 4) * 15:02d}"
              for i in range(n + 1)]
        # Build 20 low-energy London bars, then one high-energy bar
        ve = [1.0] * n + [100.0]
        ts = [f"2023-01-02 {8 + i // 4:02d}:{(i % 4) * 15:02d}" for i in range(n + 1)]
        df = make_m15_df(ts, [1800.0] * (n + 1), vol_energy=ve)
        out = compute_layer2_features(df)
        assert out["l2_session_volatility_rank"].iloc[-1] == pytest.approx(1.0, abs=0.1)

    def test_first_bar_rank_is_zero(self):
        """No history → rank = 0."""
        ts = ["2023-01-02 08:00"]
        df = make_m15_df(ts, [1800.0], vol_energy=[5.0])
        out = compute_layer2_features(df)
        assert out["l2_session_volatility_rank"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_rank_in_unit_interval(self):
        """Rank must always lie in [0, 1]."""
        import random
        random.seed(42)
        ve = [random.uniform(0.5, 5.0) for _ in range(50)]
        ts = [f"2023-01-02 {8 + i // 4:02d}:{(i % 4) * 15:02d}" for i in range(50)]
        df = make_m15_df(ts, [1800.0] * 50, vol_energy=ve)
        out = compute_layer2_features(df)
        assert (out["l2_session_volatility_rank"] >= 0).all()
        assert (out["l2_session_volatility_rank"] <= 1).all()


# ---------------------------------------------------------------------------
# Feature 10: l2_session_trend
# ---------------------------------------------------------------------------

class TestSessionTrend:
    def test_positive_for_rising_session(self):
        """Monotonically rising closes within a session → positive slope."""
        ts = pd.date_range("2023-01-02 08:00", periods=6, freq="15min")
        closes = [1800.0 + i * 2 for i in range(6)]
        df = make_m15_df([str(t) for t in ts], closes)
        out = compute_layer2_features(df)
        # From bar 2 onward there's enough data for a positive slope
        assert out["l2_session_trend"].iloc[-1] > 0

    def test_negative_for_falling_session(self):
        """Monotonically falling closes within a session → negative slope."""
        ts = pd.date_range("2023-01-02 08:00", periods=6, freq="15min")
        closes = [1810.0 - i * 2 for i in range(6)]
        df = make_m15_df([str(t) for t in ts], closes)
        out = compute_layer2_features(df)
        assert out["l2_session_trend"].iloc[-1] < 0

    def test_first_candle_is_zero(self):
        """Only one candle in session → slope = 0 (undefined)."""
        ts = ["2023-01-02 08:00"]
        df = make_m15_df(ts, [1800.0])
        out = compute_layer2_features(df)
        assert out["l2_session_trend"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_resets_at_new_session(self):
        """After session change, trend should reflect only the new session data."""
        ts = [
            "2023-01-02 07:30",  # Asian bar (going up)
            "2023-01-02 07:45",
            "2023-01-02 08:00",  # London opens — trend reset
        ]
        closes = [1800.0, 1810.0, 1800.0]
        df = make_m15_df(ts, closes)
        out = compute_layer2_features(df)
        # London bar is the first of its session → slope = 0
        assert out["l2_session_trend"].iloc[2] == pytest.approx(0.0, abs=1e-6)
