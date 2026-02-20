"""
GOLDWOLF — Layer 1 Feature Tests
Tests each feature function with synthetic M1 data of known expected values.
Covers normal cases, edge cases (dead candles, single candle, etc.).
"""

import pytest

from features.layer1 import (
    custom_volume,
    volatility_energy,
    price_velocity,
    intra_bar_reversal_count,
    early_late_ratio,
    price_acceleration,
    absorption_detection,
)
from config.settings import EPSILON


# ---------------------------------------------------------------------------
# Helpers to build synthetic M1 candle dicts
# ---------------------------------------------------------------------------

def make_candle(open_: float, high: float, low: float, close: float) -> dict:
    return {"open": open_, "high": high, "low": low, "close": close}


def dead_candle(price: float = 1800.0) -> dict:
    return make_candle(price, price, price, price)


def up_candle(base: float = 1800.0, move: float = 1.0, wick: float = 0.2) -> dict:
    """Bullish candle: closes above open, with equal wicks."""
    return make_candle(base, base + move + wick, base - wick, base + move)


def down_candle(base: float = 1801.0, move: float = 1.0, wick: float = 0.2) -> dict:
    """Bearish candle: closes below open, with equal wicks."""
    return make_candle(base, base + wick, base - move - wick, base - move)


def doji_candle(base: float = 1800.0, wick: float = 0.5) -> dict:
    """Doji: open == close, non-zero range."""
    return make_candle(base, base + wick, base - wick, base)


def absorption_bar(base: float = 1800.0, range_: float = 2.0, body: float = 0.1) -> dict:
    """Absorption candle: large range, tiny body."""
    # Place body near mid-range to keep high/low consistent
    mid = base + range_ / 2
    return make_candle(mid, base + range_, base, mid + body)


# ---------------------------------------------------------------------------
# Feature 1: Custom Volume
# ---------------------------------------------------------------------------

class TestCustomVolume:
    def test_all_active(self):
        candles = [up_candle() for _ in range(15)]
        assert custom_volume(candles) == 15

    def test_all_dead(self):
        candles = [dead_candle() for _ in range(15)]
        assert custom_volume(candles) == 0

    def test_mixed(self):
        candles = [up_candle()] * 10 + [dead_candle()] * 5
        assert custom_volume(candles) == 10

    def test_empty(self):
        assert custom_volume([]) == 0

    def test_single_active(self):
        assert custom_volume([up_candle()]) == 1

    def test_single_dead(self):
        assert custom_volume([dead_candle()]) == 0


# ---------------------------------------------------------------------------
# Feature 2: Volatility Energy
# ---------------------------------------------------------------------------

class TestVolatilityEnergy:
    def test_known_range(self):
        # Each up_candle with move=1.0, wick=0.2 → range = (1.0+0.2+0.2) = 1.4
        candles = [up_candle(move=1.0, wick=0.2) for _ in range(15)]
        expected = 15 * 1.4
        assert abs(volatility_energy(candles) - expected) < 1e-4

    def test_all_dead(self):
        candles = [dead_candle() for _ in range(15)]
        assert volatility_energy(candles) == pytest.approx(0.0)

    def test_empty(self):
        assert volatility_energy([]) == pytest.approx(0.0)

    def test_single_candle(self):
        c = make_candle(1800, 1802, 1799, 1801)
        assert volatility_energy([c]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Feature 3: Price Velocity
# ---------------------------------------------------------------------------

class TestPriceVelocity:
    def test_normal(self):
        # 10 active candles, M15 moved +5
        candles = [up_candle()] * 10 + [dead_candle()] * 5
        vel = price_velocity(candles, m15_open=1800.0, m15_close=1805.0)
        assert vel == pytest.approx(5.0 / 10)

    def test_no_active_candles(self):
        candles = [dead_candle() for _ in range(15)]
        vel = price_velocity(candles, m15_open=1800.0, m15_close=1805.0)
        assert vel == pytest.approx(0.0)

    def test_negative_move(self):
        candles = [down_candle()] * 15
        vel = price_velocity(candles, m15_open=1815.0, m15_close=1800.0)
        assert vel == pytest.approx(-15.0 / 15)

    def test_empty_candles(self):
        vel = price_velocity([], m15_open=1800.0, m15_close=1801.0)
        assert vel == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Feature 4: Intra-Bar Reversal Count
# ---------------------------------------------------------------------------

class TestIntraBarReversalCount:
    def test_no_reversals_all_up(self):
        candles = [up_candle(base=1800 + i) for i in range(15)]
        assert intra_bar_reversal_count(candles) == 0

    def test_no_reversals_all_down(self):
        candles = [down_candle(base=1815 - i) for i in range(15)]
        assert intra_bar_reversal_count(candles) == 0

    def test_alternating(self):
        # up, down, up, down … 15 candles → 14 direction changes
        candles = []
        for i in range(15):
            if i % 2 == 0:
                candles.append(up_candle())
            else:
                candles.append(down_candle())
        assert intra_bar_reversal_count(candles) == 14

    def test_single_reversal(self):
        candles = [up_candle()] * 7 + [down_candle()] * 8
        assert intra_bar_reversal_count(candles) == 1

    def test_single_candle(self):
        assert intra_bar_reversal_count([up_candle()]) == 0

    def test_empty(self):
        assert intra_bar_reversal_count([]) == 0

    def test_doji_does_not_flip(self):
        # up, doji, down — doji should not count as reversal itself,
        # but the subsequent down after doji IS a flip from up
        candles = [up_candle(), doji_candle(), down_candle()]
        # up→doji: doji skipped; then doji's last known = up, down flips → 1
        assert intra_bar_reversal_count(candles) == 1

    def test_all_doji(self):
        candles = [doji_candle() for _ in range(15)]
        assert intra_bar_reversal_count(candles) == 0


# ---------------------------------------------------------------------------
# Feature 5: Early vs Late Ratio
# ---------------------------------------------------------------------------

class TestEarlyLateRatio:
    def test_all_energy_in_late(self):
        # 7 dead early, 8 active late
        early = [dead_candle() for _ in range(7)]
        late = [up_candle(move=1.0, wick=0.0) for _ in range(8)]
        candles = early + late
        ratio = early_late_ratio(candles)
        # late_energy > 0, early_energy == 0 → ratio ≈ 1.0
        assert ratio == pytest.approx(1.0, abs=1e-6)

    def test_all_energy_in_early(self):
        early = [up_candle(move=1.0, wick=0.0) for _ in range(7)]
        late = [dead_candle() for _ in range(8)]
        candles = early + late
        ratio = early_late_ratio(candles)
        # late_energy == 0 → ratio ≈ 0.0
        assert ratio == pytest.approx(0.0, abs=1e-6)

    def test_equal_energy(self):
        # Both halves have same energy per candle, but late has 8 and early 7
        # so ratio > 0.5
        candles = [up_candle(move=1.0, wick=0.0) for _ in range(15)]
        ratio = early_late_ratio(candles)
        assert ratio > 0.5

    def test_empty(self):
        ratio = early_late_ratio([])
        assert ratio == pytest.approx(0.0, abs=1e-6)

    def test_no_division_by_zero(self):
        candles = [dead_candle() for _ in range(15)]
        ratio = early_late_ratio(candles)
        assert ratio == pytest.approx(0.0, abs=1e-6)

    def test_only_one_candle(self):
        # 1 candle goes into early (index 0); late is empty → ratio ≈ 0
        ratio = early_late_ratio([up_candle()])
        assert ratio == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Feature 6: Price Acceleration
# ---------------------------------------------------------------------------

class TestPriceAcceleration:
    def test_constant_velocity(self):
        # Each candle moves exactly +1 pip → no acceleration
        candles = [make_candle(1800 + i, 1801 + i, 1799 + i, 1801 + i) for i in range(15)]
        acc = price_acceleration(candles)
        # velocity_early  = (candle6_close - candle0_open) / 7
        #                 = (1807 - 1800) / 7 = 1.0
        # velocity_late   = (candle14_close - candle7_open) / 8
        #                 = (1815 - 1807) / 8 = 1.0
        # acceleration = 0.0
        assert acc == pytest.approx(0.0, abs=1e-4)

    def test_accelerating(self):
        # First half: small move; second half: big move
        early = [make_candle(1800, 1800.1, 1799.9, 1800.1) for _ in range(7)]
        late = [make_candle(1810, 1820, 1809, 1820) for _ in range(8)]
        candles = early + late
        acc = price_acceleration(candles)
        assert acc > 0

    def test_decelerating(self):
        early = [make_candle(1800, 1810, 1799, 1810) for _ in range(7)]
        late = [make_candle(1810, 1810.1, 1809.9, 1810.1) for _ in range(8)]
        candles = early + late
        acc = price_acceleration(candles)
        assert acc < 0

    def test_empty(self):
        assert price_acceleration([]) == pytest.approx(0.0)

    def test_single_candle(self):
        # 1 candle → goes into early; late is empty → vel_late = 0
        c = make_candle(1800, 1802, 1799, 1801)
        acc = price_acceleration([c])
        # vel_early = (1801-1800)/1 = 1.0; vel_late = 0.0 → acc = -1.0
        assert acc == pytest.approx(-1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Feature 7: Absorption Detection
# ---------------------------------------------------------------------------

class TestAbsorptionDetection:
    def test_no_absorption(self):
        # Strong body candles — body_ratio ≈ 1.0 → not absorption
        candles = [make_candle(1800, 1802, 1800, 1802) for _ in range(15)]  # 100% body
        count, intensity = absorption_detection(candles)
        assert count == 0
        assert intensity == pytest.approx(0.0, abs=1e-6)

    def test_all_absorption(self):
        # Tiny body, large range
        candles = [absorption_bar() for _ in range(15)]
        count, intensity = absorption_detection(candles)
        assert count == 15
        assert intensity == pytest.approx(1.0, abs=1e-4)

    def test_mixed(self):
        strong = [make_candle(1800, 1802, 1800, 1802) for _ in range(10)]
        absorb = [absorption_bar() for _ in range(5)]
        candles = strong + absorb
        count, _ = absorption_detection(candles)
        assert count == 5

    def test_dead_candles_excluded(self):
        # Dead candles have range 0 and must not be counted
        candles = [dead_candle() for _ in range(15)]
        count, intensity = absorption_detection(candles)
        assert count == 0
        assert intensity == pytest.approx(0.0, abs=1e-6)

    def test_empty(self):
        count, intensity = absorption_detection([])
        assert count == 0
        assert intensity == pytest.approx(0.0, abs=1e-6)

    def test_intensity_partial(self):
        strong = [make_candle(1800, 1802, 1800, 1802) for _ in range(10)]  # range 2 each
        absorb = [absorption_bar(range_=4.0) for _ in range(5)]             # range 4 each
        candles = strong + absorb
        count, intensity = absorption_detection(candles)
        assert count == 5
        # total_energy = 10*2 + 5*4 = 40; absorption_range = 5*4 = 20
        assert intensity == pytest.approx(20.0 / 40.0, abs=1e-4)
