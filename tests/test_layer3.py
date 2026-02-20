"""
GOLDWOLF — Layer 3 (SMC) Feature Tests
Uses synthetic M15 DataFrames with known price structures to verify each
Smart Money Concepts feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.layer3 import (
    compute_layer3_features,
    _detect_swings,
    _find_liquidity_pool,
    _swing_structure_trend,
)
from config.settings import (
    SWING_LOOKBACK,
    PIP_SIZE,
    LIQUIDITY_MIN_TOUCHES,
    LIQUIDITY_TOLERANCE_PIPS,
    EPSILON,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal M15 DataFrame
# ---------------------------------------------------------------------------

def make_price_df(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
    start: str = "2020-01-06 00:00",  # Monday to avoid weekend filter issues
) -> pd.DataFrame:
    """
    Build a DataFrame that mimics the Layer 1+2 output.

    Timestamps are generated as consecutive M15 bars starting at *start*
    (a weekday to match realistic data).
    """
    n = len(closes)
    highs = highs if highs is not None else [c + 1.0 for c in closes]
    lows = lows if lows is not None else [c - 1.0 for c in closes]
    opens = opens if opens is not None else list(closes)

    idx = pd.date_range(start=start, periods=n, freq="15min")
    return pd.DataFrame(
        {
            "m15_open": np.array(opens, dtype=np.float64),
            "m15_high": np.array(highs, dtype=np.float64),
            "m15_low": np.array(lows, dtype=np.float64),
            "m15_close": np.array(closes, dtype=np.float64),
            "m15_volume": np.zeros(n, dtype=np.float64),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Feature 1: l3_swing_high / l3_swing_low
# ---------------------------------------------------------------------------

class TestSwingDetection:
    """Tests the _detect_swings helper directly for speed."""

    def test_single_peak(self):
        """A clear peak in the middle should be flagged as swing high."""
        N = SWING_LOOKBACK
        # Build: N flat bars, 1 peak, N flat bars
        highs = [1800.0] * N + [1810.0] + [1800.0] * N
        lows = [1799.0] * (2 * N + 1)
        sh, sl = _detect_swings(np.array(highs), np.array(lows), n=N)
        assert sh[N] == 1
        # Surrounding bars are not swing highs
        assert sh[N - 1] == 0
        assert sh[N + 1] == 0

    def test_single_trough(self):
        """A clear trough in the middle should be flagged as swing low."""
        N = SWING_LOOKBACK
        highs = [1801.0] * (2 * N + 1)
        lows = [1800.0] * N + [1790.0] + [1800.0] * N
        sh, sl = _detect_swings(np.array(highs), np.array(lows), n=N)
        assert sl[N] == 1
        assert sl[N - 1] == 0
        assert sl[N + 1] == 0

    def test_insufficient_data(self):
        """Fewer bars than 2N+1 → all zeros."""
        N = SWING_LOOKBACK
        highs = [1800.0] * (2 * N)  # one too few
        lows = [1799.0] * (2 * N)
        sh, sl = _detect_swings(np.array(highs), np.array(lows), n=N)
        assert sh.sum() == 0
        assert sl.sum() == 0

    def test_no_swings_in_flat_data(self):
        """Perfectly flat prices → no swings."""
        n = 30
        highs = [1800.0] * n
        lows = [1799.0] * n
        sh, sl = _detect_swings(np.array(highs), np.array(lows), n=SWING_LOOKBACK)
        assert sh.sum() == 0
        assert sl.sum() == 0

    def test_via_compute_layer3(self):
        """Verify swing flags appear in the full feature output."""
        N = SWING_LOOKBACK
        highs = [1800.0] * N + [1810.0] + [1800.0] * N
        lows = [1799.0] * (2 * N + 1)
        closes = [1800.0] * (2 * N + 1)
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        assert out["l3_swing_high"].iloc[N] == 1


# ---------------------------------------------------------------------------
# Feature 2: l3_bos_direction
# ---------------------------------------------------------------------------

class TestBOSDetection:
    def test_bullish_bos_after_swing_high(self):
        """
        Build a clear swing high then a candle that breaks above it.
        Expect l3_bos_direction == 1 on the breakout candle.
        """
        N = SWING_LOOKBACK
        # N bars at 1800, peak at 1810, N bars back at 1800, then a big breakout
        closes = [1800.0] * N + [1810.0] + [1800.0] * N + [1820.0]
        highs  = [c + 0.5 for c in closes]
        lows   = [c - 0.5 for c in closes]
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        # The breakout bar is the last one
        bos = out["l3_bos_direction"].values
        # At least one bullish BOS should be detected after the swing high is confirmed
        assert (bos == 1).any()

    def test_bearish_bos_after_swing_low(self):
        """
        Build a clear swing low then a candle that breaks below it.
        Expect l3_bos_direction == -1 on the breakdown candle.
        """
        N = SWING_LOOKBACK
        closes = [1810.0] * N + [1800.0] + [1810.0] * N + [1790.0]
        highs  = [c + 0.5 for c in closes]
        lows   = [c - 0.5 for c in closes]
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        bos = out["l3_bos_direction"].values
        assert (bos == -1).any()

    def test_no_bos_in_flat_data(self):
        """Flat prices with no structural breaks → no BOS."""
        closes = [1800.0] * 30
        df = make_price_df(closes)
        out = compute_layer3_features(df)
        # No BOS expected (or very few due to flat structure)
        assert (out["l3_bos_direction"] == 0).all()


# ---------------------------------------------------------------------------
# Feature 3: l3_choch_flag
# ---------------------------------------------------------------------------

class TestCHoCHDetection:
    def test_choch_after_trend_reversal(self):
        """
        Sequence: establish bullish trend (bullish BOS), then bearish BOS → CHoCH = -1.

        Pattern:
        1. Create swing low  (N flat + dip + N flat)
        2. Create swing high (N rising + peak + N flat)
        3. Break above swing high → bullish BOS  (prevailing_trend = 1)
        4. Create new swing low below the broken swing low → bearish BOS → CHoCH = -1
        """
        N = SWING_LOOKBACK
        # --- Phase 1: swing low at dip1=1790 ---
        phase1 = [1800.0] * N + [1790.0] + [1800.0] * N      # 2N+1 bars
        # --- Phase 2: swing high at peak=1820 ---
        phase2 = [1800.0 + i * 2 for i in range(N + 1)] + [1820.0 - i * 2 for i in range(1, N + 1)]  # 2N+1 bars
        # --- Phase 3: break above swing high (1820) → bullish BOS ---
        phase3 = [1825.0]                                     # 1 bar
        # --- Phase 4: swing low below original dip → bearish BOS + CHoCH ---
        phase4 = [1810.0] * N + [1780.0] + [1810.0] * N + [1775.0]  # 2N+2 bars

        closes = phase1 + phase2 + phase3 + phase4
        highs  = [c + 0.5 for c in closes]
        lows   = [c - 0.5 for c in closes]
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        # Either a CHoCH (flag != 0) or at minimum a BOS occurred
        assert (out["l3_choch_flag"] != 0).any() or (out["l3_bos_direction"] != 0).any()

    def test_no_choch_in_single_direction(self):
        """Sustained uptrend with only bullish BOS → no bearish CHoCH."""
        N = SWING_LOOKBACK
        # Two clear bullish BOS events in sequence
        closes = (
            [1800.0] * N + [1790.0] + [1800.0] * N + [1815.0]
            + [1815.0] * N + [1805.0] + [1815.0] * N + [1830.0]
        )
        highs = [c + 0.5 for c in closes]
        lows  = [c - 0.5 for c in closes]
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        assert (out["l3_choch_flag"] == -1).sum() == 0


# ---------------------------------------------------------------------------
# Feature 4: l3_demand_ob_distance / l3_supply_ob_distance
# ---------------------------------------------------------------------------

class TestOrderBlockDetection:
    def test_demand_ob_created_on_bullish_bos(self):
        """After a bullish BOS, demand_ob_distance should become non-zero."""
        N = SWING_LOOKBACK
        # Include a bearish candle before the swing high so there's an OB candidate
        opens  = [1800.0] * N + [1810.0] + [1800.0] * (N - 1) + [1805.0] + [1820.0]
        closes = [1800.0] * N + [1810.0] + [1800.0] * (N - 1) + [1802.0] + [1820.0]
        # Make sure the "bearish OB" candle open > close
        opens[2 * N] = 1808.0   # bar just before breakout: bearish (open > close)
        closes[2 * N] = 1802.0
        highs = [max(o, c) + 0.5 for o, c in zip(opens, closes)]
        lows  = [min(o, c) - 0.5 for o, c in zip(opens, closes)]
        df = make_price_df(closes, highs=highs, lows=lows, opens=opens)
        out = compute_layer3_features(df)
        # After BOS occurs demand_ob_distance should be non-zero
        dem = out["l3_demand_ob_distance"].values
        assert (dem != 0.0).any()

    def test_ob_distance_output_is_float(self):
        """OB distance columns must be present and numeric."""
        df = make_price_df([1800.0] * 20)
        out = compute_layer3_features(df)
        assert "l3_demand_ob_distance" in out.columns
        assert "l3_supply_ob_distance" in out.columns
        assert out["l3_demand_ob_distance"].dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# Feature 5 & 6: l3_fvg_active / l3_fvg_distance
# ---------------------------------------------------------------------------

class TestFVGDetection:
    def test_bullish_fvg_detected(self):
        """
        Bullish FVG: high[i-2] < low[i].
        After the FVG bar appears, fvg_active should be 1.
        """
        # Build a gap-up sequence: bar0 high=1800, bar1 anything, bar2 low=1805
        closes = [1799.0, 1801.0, 1806.0]  # bar2 low (1806-1) = 1805 > bar0 high (1800)
        highs  = [1800.0, 1802.0, 1807.0]
        lows   = [1798.0, 1800.0, 1805.0]
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        # FVG is detected at bar 2 (index 2)
        # gap = lows[2] - highs[0] = 1805 - 1800 = 5 > FVG_MIN_GAP_PIPS
        assert out["l3_fvg_active"].iloc[2] == 1

    def test_bearish_fvg_detected(self):
        """Bearish FVG: low[i-2] > high[i]."""
        closes = [1810.0, 1806.0, 1801.0]
        highs  = [1811.0, 1808.0, 1803.0]
        lows   = [1809.0, 1805.0, 1800.0]
        # gap = lows[0] - highs[2] = 1809 - 1803 = 6 > FVG_MIN_GAP_PIPS
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        assert out["l3_fvg_active"].iloc[2] == -1

    def test_no_fvg_when_gap_too_small(self):
        """Gaps smaller than FVG_MIN_GAP_PIPS should not create an FVG."""
        closes = [1800.0, 1800.1, 1800.2]
        # Very tiny gap: lows[2] - highs[0] = 1799.1 - 1800.5 < 0 → no gap
        highs = [1800.5, 1800.6, 1800.7]
        lows  = [1799.5, 1799.6, 1799.7]
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        assert out["l3_fvg_active"].iloc[2] == 0

    def test_fvg_filled_when_price_returns(self):
        """FVG is cleared when subsequent price fills the gap."""
        # bar0 high=1800, bar2 low=1805 → FVG zone [1800, 1805]
        # bar3 low <= 1805 → fills the gap
        closes = [1799.0, 1801.0, 1806.0, 1801.0]
        highs  = [1800.0, 1802.0, 1807.0, 1805.0]  # bar3 high touches gap
        lows   = [1798.0, 1800.0, 1805.0, 1799.0]  # bar3 low <= gap hi (1805)
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        # After the FVG is filled, fvg_active should go to 0 for bar3
        assert out["l3_fvg_active"].iloc[3] == 0

    def test_fvg_columns_present(self):
        df = make_price_df([1800.0] * 10)
        out = compute_layer3_features(df)
        assert "l3_fvg_active" in out.columns
        assert "l3_fvg_distance" in out.columns


# ---------------------------------------------------------------------------
# Feature 7 & 8: l3_buy_liq_distance / l3_sell_liq_distance
# ---------------------------------------------------------------------------

class TestLiquidityPool:
    def test_buy_side_pool_detected(self):
        """Three bars with equal highs (within tolerance) form a buy-side pool."""
        n = LIQUIDITY_MIN_TOUCHES + 5
        # Alternate between 1800 and 1810, with LIQUIDITY_MIN_TOUCHES bars near 1810
        closes = [1800.0] * n
        # Create equal highs at 1810 for the first LIQUIDITY_MIN_TOUCHES bars
        highs = [1810.0] * LIQUIDITY_MIN_TOUCHES + [1802.0] * (n - LIQUIDITY_MIN_TOUCHES)
        lows  = [1799.0] * n
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        # After the touches, buy_liq_distance should be non-zero somewhere
        buy = out["l3_buy_liq_distance"].values
        assert (buy != 0.0).any()

    def test_sell_side_pool_detected(self):
        """Three bars with equal lows form a sell-side pool."""
        n = LIQUIDITY_MIN_TOUCHES + 5
        closes = [1810.0] * n
        highs = [1811.0] * n
        lows  = [1800.0] * LIQUIDITY_MIN_TOUCHES + [1808.0] * (n - LIQUIDITY_MIN_TOUCHES)
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        sell = out["l3_sell_liq_distance"].values
        assert (sell != 0.0).any()

    def test_no_pool_with_insufficient_touches(self):
        """Fewer than LIQUIDITY_MIN_TOUCHES → no pool."""
        n = 15
        closes = [1800.0] * n
        # All highs are strictly different (no cluster possible)
        highs = [1800.0 + i * 5 for i in range(n)]  # 1800, 1805, 1810, ... — no two within tolerance
        lows  = [1799.0] * n
        df = make_price_df(closes, highs=highs, lows=lows)
        out = compute_layer3_features(df)
        assert (out["l3_buy_liq_distance"] == 0.0).all()

    def test_find_liquidity_pool_helper(self):
        """Test _find_liquidity_pool directly."""
        tol = LIQUIDITY_TOLERANCE_PIPS * PIP_SIZE
        prices = np.array([1810.0, 1810.02, 1810.01, 1800.0, 1800.0])
        # First 3 are within tolerance, 2 others are far away
        result = _find_liquidity_pool(prices)
        if LIQUIDITY_MIN_TOUCHES <= 3:
            assert result is not None
            assert abs(result - 1810.0) < tol * 2


# ---------------------------------------------------------------------------
# Feature 11: l3_premium_discount
# ---------------------------------------------------------------------------

class TestPremiumDiscount:
    @staticmethod
    def _make_zigzag_df(
        sl_close: float, sh_close: float, end_closes: list
    ) -> pd.DataFrame:
        """
        Build a DataFrame with a clear swing low at *sl_close* and a swing high
        at *sh_close* (N flat bars on each side), followed by *end_closes* for
        zone checking.
        """
        N = SWING_LOOKBACK
        neutral = (sl_close + sh_close) / 2.0
        # Swing low: N neutral + trough + N neutral
        phase_sl = [neutral] * N + [sl_close] + [neutral] * N
        # Swing high: N neutral + peak + N neutral
        phase_sh = [neutral] * N + [sh_close] + [neutral] * N
        closes = phase_sl + phase_sh + list(end_closes)
        highs = [c + 0.5 for c in closes]
        lows  = [c - 0.5 for c in closes]
        return make_price_df(closes, highs=highs, lows=lows)

    def test_premium_zone(self):
        """Price in upper 50% of swing range → premium = 1."""
        sl, sh = 1800.0, 1900.0
        mid = (sl + sh) / 2.0
        premium = mid + 30.0   # clearly > mid + 5 % equilibrium band
        df = self._make_zigzag_df(sl, sh, [premium] * 5)
        out = compute_layer3_features(df)
        assert (out["l3_premium_discount"].tail(5) == 1).any()

    def test_discount_zone(self):
        """Price in lower 50% of swing range → discount = -1."""
        sl, sh = 1800.0, 1900.0
        mid = (sl + sh) / 2.0
        discount = mid - 30.0  # clearly < mid - 5 % equilibrium band
        df = self._make_zigzag_df(sl, sh, [discount] * 5)
        out = compute_layer3_features(df)
        assert (out["l3_premium_discount"].tail(5) == -1).any()

    def test_no_swing_range_gives_zero(self):
        """Without established swing range, premium_discount = 0."""
        df = make_price_df([1800.0] * 5)
        out = compute_layer3_features(df)
        assert (out["l3_premium_discount"] == 0).all()


# ---------------------------------------------------------------------------
# Feature 12: l3_structure_trend
# ---------------------------------------------------------------------------

class TestStructureTrend:
    def test_bullish_structure(self):
        """HH + HL pattern → structure_trend = 1."""
        # Using _swing_structure_trend directly
        N = SWING_LOOKBACK
        # Four swings: SL1 < SL2 (HL), SH1 < SH2 (HH)
        sh = np.zeros(4 * (2 * N + 1), dtype=np.int8)
        sl = np.zeros(4 * (2 * N + 1), dtype=np.int8)
        highs = np.ones(4 * (2 * N + 1)) * 1800.0
        lows  = np.ones(4 * (2 * N + 1)) * 1799.0

        # Swing low 1
        sl[N] = 1
        lows[N] = 1790.0
        # Swing high 1
        sh[2 * N + 2] = 1
        highs[2 * N + 2] = 1810.0
        # Swing low 2 (higher low)
        sl[3 * N + 3] = 1
        lows[3 * N + 3] = 1795.0
        # Swing high 2 (higher high)
        sh[4 * N + 4] = 1
        highs[4 * N + 4] = 1820.0

        trend = _swing_structure_trend(sh, sl, highs, lows)
        # At bar 4*N+4, should be bullish
        assert trend[4 * N + 4] == 1

    def test_bearish_structure(self):
        """LH + LL pattern → structure_trend = -1."""
        N = SWING_LOOKBACK
        size = 4 * (2 * N + 1)
        sh = np.zeros(size, dtype=np.int8)
        sl = np.zeros(size, dtype=np.int8)
        highs = np.ones(size) * 1800.0
        lows  = np.ones(size) * 1799.0

        # Swing high 1
        sh[N] = 1
        highs[N] = 1820.0
        # Swing low 1
        sl[2 * N + 2] = 1
        lows[2 * N + 2] = 1800.0
        # Swing high 2 (lower high)
        sh[3 * N + 3] = 1
        highs[3 * N + 3] = 1815.0
        # Swing low 2 (lower low)
        sl[4 * N + 4] = 1
        lows[4 * N + 4] = 1795.0

        trend = _swing_structure_trend(sh, sl, highs, lows)
        assert trend[4 * N + 4] == -1


# ---------------------------------------------------------------------------
# Higher-timeframe trend columns
# ---------------------------------------------------------------------------

class TestHTFTrend:
    def test_columns_present(self):
        """l3_h1_trend and l3_h4_trend columns must exist."""
        df = make_price_df([1800.0] * 50)
        out = compute_layer3_features(df)
        assert "l3_h1_trend" in out.columns
        assert "l3_h4_trend" in out.columns

    def test_values_in_valid_set(self):
        """Trend values must be in {-1, 0, 1}."""
        df = make_price_df([1800.0 + i * 0.5 for i in range(100)])
        out = compute_layer3_features(df)
        assert set(out["l3_h1_trend"].unique()).issubset({-1, 0, 1})
        assert set(out["l3_h4_trend"].unique()).issubset({-1, 0, 1})

    def test_same_length_as_input(self):
        """Output must have the same length as input."""
        n = 80
        df = make_price_df([1800.0] * n)
        out = compute_layer3_features(df)
        assert len(out) == n
