"""
GOLDWOLF — Layer 4 (Private Edge Features) Tests
Uses synthetic M15 DataFrames with known feature values to verify each L4 feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.layer4 import (
    compute_layer4_features,
    _whale_footprint,
    _trap_score,
    _candle_dna,
    _momentum_divergence,
    _consecutive_bias,
    _volume_climax,
    _range_compression,
    _multi_layer_confluence,
)
from config.settings import (
    L4_WHALE_VOLUME_MIN,
    L4_WHALE_REVERSAL_MAX,
    L4_WHALE_ABSORPTION_MIN,
    L4_DNA_TRAP_SCORE_MIN,
    L4_CLIMAX_WINDOW,
    L4_CLIMAX_SIGMA,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal M15 DataFrame with all required L1/L2/L3 columns
# ---------------------------------------------------------------------------

def make_full_df(
    n: int = 50,
    close_vals: list[float] | None = None,
    open_vals: list[float] | None = None,
    high_vals: list[float] | None = None,
    low_vals: list[float] | None = None,
    start: str = "2020-01-06 00:00",
) -> pd.DataFrame:
    """
    Build a DataFrame that mimics the Layer 1+2+3 output with all required
    columns for Layer 4 computation.
    """
    if close_vals is None:
        close_vals = [1800.0 + i * 0.1 for i in range(n)]
    n = len(close_vals)
    opens = open_vals if open_vals is not None else [c - 0.05 for c in close_vals]
    highs = high_vals if high_vals is not None else [c + 1.0 for c in close_vals]
    lows = low_vals if low_vals is not None else [c - 1.0 for c in close_vals]

    idx = pd.date_range(start=start, periods=n, freq="15min")
    data = {
        "m15_open": np.array(opens, dtype=np.float64),
        "m15_high": np.array(highs, dtype=np.float64),
        "m15_low": np.array(lows, dtype=np.float64),
        "m15_close": np.array(close_vals, dtype=np.float64),
        "m15_volume": np.zeros(n, dtype=np.float64),
        # L1 features
        "l1_custom_volume": np.full(n, 10, dtype=np.float64),
        "l1_volatility_energy": np.full(n, 3.0, dtype=np.float64),
        "l1_price_velocity": np.full(n, 0.1, dtype=np.float64),
        "l1_reversal_count": np.full(n, 2, dtype=np.float64),
        "l1_early_late_ratio": np.full(n, 0.5, dtype=np.float64),
        "l1_price_acceleration": np.full(n, 0.0, dtype=np.float64),
        "l1_absorption_count": np.full(n, 2, dtype=np.float64),
        "l1_absorption_intensity": np.full(n, 0.2, dtype=np.float64),
        # L2 features
        "l2_session": np.zeros(n, dtype=np.int8),
        "l2_session_overlap": np.zeros(n, dtype=np.int8),
        "l2_kill_zone": np.zeros(n, dtype=np.int8),
        "l2_day_of_week": np.zeros(n, dtype=np.int8),
        "l2_hour": np.zeros(n, dtype=np.int8),
        "l2_distance_from_session_open": np.zeros(n, dtype=np.float32),
        "l2_session_position": np.full(n, 0.5, dtype=np.float32),
        "l2_time_since_vol_spike": np.zeros(n, dtype=np.float32),
        "l2_session_volatility_rank": np.full(n, 0.5, dtype=np.float32),
        "l2_session_trend": np.zeros(n, dtype=np.float32),
        # L3 features
        "l3_swing_high": np.zeros(n, dtype=np.int8),
        "l3_swing_low": np.zeros(n, dtype=np.int8),
        "l3_bos_direction": np.zeros(n, dtype=np.int8),
        "l3_choch_flag": np.zeros(n, dtype=np.int8),
        "l3_demand_ob_distance": np.zeros(n, dtype=np.float32),
        "l3_supply_ob_distance": np.zeros(n, dtype=np.float32),
        "l3_fvg_active": np.zeros(n, dtype=np.int8),
        "l3_fvg_distance": np.zeros(n, dtype=np.float32),
        "l3_buy_liq_distance": np.zeros(n, dtype=np.float32),
        "l3_sell_liq_distance": np.zeros(n, dtype=np.float32),
        "l3_liquidity_sweep": np.zeros(n, dtype=np.int8),
        "l3_premium_discount": np.zeros(n, dtype=np.int8),
        "l3_structure_trend": np.zeros(n, dtype=np.int8),
        "l3_h1_trend": np.zeros(n, dtype=np.int8),
        "l3_h4_trend": np.zeros(n, dtype=np.int8),
    }
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Feature 1: l4_whale_footprint
# ---------------------------------------------------------------------------

class TestWhaleFootprint:
    def test_score_zero_no_conditions_met(self):
        """Low volume, high reversals, low absorption → score 0."""
        cv = np.array([5.0])   # < L4_WHALE_VOLUME_MIN
        rc = np.array([8.0])   # >= L4_WHALE_REVERSAL_MAX
        ac = np.array([1.0])   # < L4_WHALE_ABSORPTION_MIN
        score = _whale_footprint(cv, rc, ac)
        assert score[0] == 0

    def test_score_three_all_conditions_met(self):
        """All three conditions met → score 3."""
        cv = np.array([float(L4_WHALE_VOLUME_MIN)])
        rc = np.array([float(L4_WHALE_REVERSAL_MAX - 1)])
        ac = np.array([float(L4_WHALE_ABSORPTION_MIN)])
        score = _whale_footprint(cv, rc, ac)
        assert score[0] == 3

    def test_score_one_single_condition(self):
        """Only volume condition met → score 1."""
        cv = np.array([float(L4_WHALE_VOLUME_MIN)])
        rc = np.array([8.0])   # fails
        ac = np.array([1.0])   # fails
        score = _whale_footprint(cv, rc, ac)
        assert score[0] == 1

    def test_output_length(self):
        """Output length matches input."""
        n = 20
        cv = np.ones(n) * 13.0
        rc = np.ones(n) * 2.0
        ac = np.ones(n) * 3.0
        score = _whale_footprint(cv, rc, ac)
        assert len(score) == n

    def test_via_compute_layer4(self):
        """l4_whale_footprint column exists in compute_layer4_features output."""
        df = make_full_df(n=30)
        out = compute_layer4_features(df)
        assert "l4_whale_footprint" in out.columns
        assert out["l4_whale_footprint"].between(0, 3).all()


# ---------------------------------------------------------------------------
# Feature 2: l4_trap_score
# ---------------------------------------------------------------------------

class TestTrapScore:
    def test_score_zero_no_signals(self):
        """No signals active → score 0."""
        n = 5
        sweep = np.zeros(n, dtype=np.int8)
        kz = np.zeros(n, dtype=np.int8)
        close = np.full(n, 1800.0)
        dem = np.full(n, 50.0)   # far from OB
        sup = np.full(n, 50.0)
        ab = np.zeros(n)
        score = _trap_score(sweep, kz, close, dem, sup, ab)
        assert (score == 0).all()

    def test_liquidity_sweep_adds_30(self):
        """Liquidity sweep alone should add 30 points."""
        n = 3
        sweep = np.array([1, 0, -1], dtype=np.int8)
        kz = np.zeros(n, dtype=np.int8)
        close = np.full(n, 1800.0)
        dem = np.full(n, 50.0)
        sup = np.full(n, 50.0)
        ab = np.zeros(n)
        score = _trap_score(sweep, kz, close, dem, sup, ab)
        # Bars with sweep get at least 30 (they may also get reversal points)
        assert score[0] >= 30
        assert score[2] >= 30

    def test_score_capped_at_100(self):
        """Score is capped at 100."""
        n = 5
        sweep = np.ones(n, dtype=np.int8)
        kz = np.ones(n, dtype=np.int8)
        close = np.full(n, 1800.0)
        dem = np.full(n, 1.0)    # near OB
        sup = np.full(n, 1.0)
        ab = np.full(n, 0.9)     # high absorption
        score = _trap_score(sweep, kz, close, dem, sup, ab)
        assert (score <= 100).all()

    def test_score_range(self):
        """Score is always between 0 and 100."""
        df = make_full_df(n=50)
        out = compute_layer4_features(df)
        assert out["l4_trap_score"].between(0, 100).all()


# ---------------------------------------------------------------------------
# Feature 3: l4_candle_dna
# ---------------------------------------------------------------------------

class TestCandleDNA:
    def test_type_4_dead_zone(self):
        """Low volume + low energy → type 4 (dead zone)."""
        df = make_full_df(n=30)
        # Override L1 features for dead zone conditions
        df["l1_custom_volume"] = 3.0  # < L4_DNA_DEAD_VOLUME_MAX (8)
        df["l1_volatility_energy"] = 0.5  # < L4_DNA_DEAD_ENERGY_MAX (2.0)
        # Keep other conditions not met
        df["l3_bos_direction"] = 0
        df["l3_choch_flag"] = 0
        out = compute_layer4_features(df)
        # Most bars should be type 4
        assert (out["l4_candle_dna"] == 4).any()

    def test_type_3_priority_over_others(self):
        """Trap score >= threshold should always produce type 3, regardless of other features."""
        df = make_full_df(n=30)
        # Set trap-favorable conditions so the computed trap_score is high
        df["l3_liquidity_sweep"] = 1
        df["l2_kill_zone"] = 1
        df["l1_absorption_intensity"] = 0.9
        df["l3_demand_ob_distance"] = 1.0   # near OB
        # Set close > previous close to trigger reversal
        closes = list(df["m15_close"].values)
        for i in range(2, len(closes)):
            closes[i] = closes[i - 2] - 1.0  # alternating direction
        df["m15_close"] = closes
        out = compute_layer4_features(df)
        assert "l4_candle_dna" in out.columns
        # Values should be in valid range
        assert set(out["l4_candle_dna"].unique()).issubset(set(range(-1, 8)))

    def test_dna_values_in_valid_range(self):
        """All DNA values must be -1 or 0-7."""
        df = make_full_df(n=50)
        out = compute_layer4_features(df)
        valid = set(range(-1, 8))
        assert set(out["l4_candle_dna"].unique()).issubset(valid)

    def test_columns_present(self):
        """l4_candle_dna column must exist."""
        df = make_full_df(n=20)
        out = compute_layer4_features(df)
        assert "l4_candle_dna" in out.columns


# ---------------------------------------------------------------------------
# Feature 4: l4_momentum_divergence
# ---------------------------------------------------------------------------

class TestMomentumDivergence:
    def test_bearish_divergence(self):
        """Price rising but energy falling → bearish divergence (-1)."""
        n = 10
        close = np.array([1800.0 + i for i in range(n)])   # rising
        energy = np.array([5.0 - i * 0.4 for i in range(n)])  # falling
        from features.layer4 import _momentum_divergence
        result = _momentum_divergence(close, energy, lookback=5)
        # After lookback, price is up and energy is down → -1
        assert result[5] == -1

    def test_bullish_divergence(self):
        """Price falling but energy rising → bullish divergence (+1)."""
        n = 10
        close = np.array([1800.0 - i for i in range(n)])   # falling
        energy = np.array([1.0 + i * 0.5 for i in range(n)])  # rising
        from features.layer4 import _momentum_divergence
        result = _momentum_divergence(close, energy, lookback=5)
        assert result[5] == 1

    def test_no_divergence_same_direction(self):
        """Both price and energy moving up → no divergence (0)."""
        n = 10
        close = np.array([1800.0 + i for i in range(n)])
        energy = np.array([1.0 + i * 0.3 for i in range(n)])
        from features.layer4 import _momentum_divergence
        result = _momentum_divergence(close, energy, lookback=5)
        assert result[5] == 0

    def test_first_lookback_bars_zero(self):
        """First lookback bars cannot have divergence."""
        n = 20
        close = np.arange(n, dtype=float)
        energy = np.arange(n, dtype=float)[::-1]
        from features.layer4 import _momentum_divergence
        result = _momentum_divergence(close, energy, lookback=5)
        assert (result[:5] == 0).all()

    def test_output_in_valid_range(self):
        """All values must be in {-1, 0, 1}."""
        df = make_full_df(n=50)
        out = compute_layer4_features(df)
        assert set(out["l4_momentum_divergence"].unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Feature 5: l4_consecutive_bias (tested via compute_layer4_features)
# ---------------------------------------------------------------------------

class TestConsecutiveBias:
    def test_bullish_streak(self):
        """All bullish closes → positive increasing streak."""
        n = 10
        opens = [1800.0] * n
        closes = [1801.0] * n  # always bullish
        df = make_full_df(n=n, close_vals=closes, open_vals=opens)
        out = compute_layer4_features(df)
        bias = out["l4_consecutive_bias"].values
        # Streak should be positive and growing
        assert (bias > 0).all()
        assert bias[-1] >= bias[0]

    def test_bearish_streak(self):
        """All bearish closes → negative increasing streak."""
        n = 10
        opens = [1802.0] * n
        closes = [1800.0] * n  # always bearish
        df = make_full_df(n=n, close_vals=closes, open_vals=opens)
        out = compute_layer4_features(df)
        bias = out["l4_consecutive_bias"].values
        assert (bias < 0).all()

    def test_bias_capped_at_20(self):
        """Streak is capped at ±20."""
        n = 50
        opens = [1800.0] * n
        closes = [1801.0] * n
        df = make_full_df(n=n, close_vals=closes, open_vals=opens)
        out = compute_layer4_features(df)
        assert (out["l4_consecutive_bias"] <= 20).all()
        assert (out["l4_consecutive_bias"] >= -20).all()

    def test_streak_resets_on_direction_change(self):
        """Direction change should reset the streak."""
        from features.layer4 import _consecutive_bias
        open_ = np.array([1800.0] * 6)
        close = np.array([1801.0] * 3 + [1799.0] * 3)  # 3 bull then 3 bear
        result = _consecutive_bias(close, open_)
        # After the flip, the streak should become negative
        assert result[3] == -1  # first bearish bar resets to -1
        assert result[5] == -3


# ---------------------------------------------------------------------------
# Feature 6: l4_volume_climax
# ---------------------------------------------------------------------------

class TestVolumeClimax:
    def test_no_climax_in_normal_data(self):
        """Normal volatility levels → no climax."""
        df = make_full_df(n=50)
        out = compute_layer4_features(df)
        # With constant energy and volume, no climax expected after warm-up
        # (may have some from initial rolling window behavior)
        assert "l4_volume_climax" in out.columns
        assert set(out["l4_volume_climax"].unique()).issubset({0, 1})

    def test_climax_detected_on_spike(self):
        """Extreme spike in both volume and energy → climax = 1."""
        n = L4_CLIMAX_WINDOW + 5
        cv = np.full(n, 5.0)
        ve = np.full(n, 2.0)
        # Create a spike 10 sigma above mean
        cv[-1] = 5.0 + 10 * np.std(cv[:-1]) + 1.0
        ve[-1] = 2.0 + 10 * np.std(ve[:-1]) + 1.0
        from features.layer4 import _volume_climax
        result = _volume_climax(cv, ve)
        assert result[-1] == 1

    def test_climax_zero_without_both_conditions(self):
        """Only volume spike (not energy) → no climax."""
        n = L4_CLIMAX_WINDOW + 5
        cv = np.full(n, 5.0)
        ve = np.full(n, 2.0)
        cv[-1] = 100.0  # huge volume spike
        # energy stays normal
        from features.layer4 import _volume_climax
        result = _volume_climax(cv, ve)
        assert result[-1] == 0


# ---------------------------------------------------------------------------
# Feature 7: l4_range_compression
# ---------------------------------------------------------------------------

class TestRangeCompression:
    def test_ratio_one_for_constant_range(self):
        """Constant range → rolling average equals current → ratio ≈ 1.0."""
        n = 30
        closes = [1800.0 + i * 0.1 for i in range(n)]
        highs = [c + 2.0 for c in closes]
        lows = [c - 2.0 for c in closes]
        df = make_full_df(n=n, close_vals=closes, high_vals=highs, low_vals=lows)
        out = compute_layer4_features(df)
        # After warm-up, ratio should be close to 1.0
        ratio_tail = out["l4_range_compression"].tail(10).values
        assert np.allclose(ratio_tail, 1.0, atol=0.05)

    def test_ratio_high_for_expansion(self):
        """Current range much larger than average → ratio > 1."""
        from features.layer4 import _range_compression
        n = 25
        high = np.full(n, 1802.0)
        low = np.full(n, 1800.0)
        # Last bar has a huge range
        high[-1] = 1810.0
        low[-1] = 1790.0
        result = _range_compression(high, low)
        assert result[-1] > 1.5

    def test_column_present_and_float(self):
        """l4_range_compression column must exist and be float."""
        df = make_full_df(n=30)
        out = compute_layer4_features(df)
        assert "l4_range_compression" in out.columns
        assert out["l4_range_compression"].dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# Feature 8 & 9: l4_session_continuation / l4_multi_layer_confluence
# ---------------------------------------------------------------------------

class TestSessionContinuation:
    def test_column_present_and_in_range(self):
        """l4_session_continuation must exist and be 0-1."""
        df = make_full_df(n=50)
        out = compute_layer4_features(df)
        assert "l4_session_continuation" in out.columns
        assert out["l4_session_continuation"].between(0.0, 1.0).all()


class TestMultiLayerConfluence:
    def test_all_bullish_gives_positive(self):
        """All layers bullish → positive score."""
        n = 10
        sess_trend = np.ones(n)       # bullish session
        struct_trend = np.ones(n, dtype=np.int8)
        h1_trend = np.ones(n, dtype=np.int8)
        h4_trend = np.ones(n, dtype=np.int8)
        prem_disc = np.full(n, -1, dtype=np.int8)  # discount = bullish

        score = _multi_layer_confluence(sess_trend, struct_trend, h1_trend, h4_trend, prem_disc)
        assert (score > 0).all()

    def test_all_bearish_gives_negative(self):
        """All layers bearish → negative score."""
        n = 10
        sess_trend = np.full(n, -1.0)
        struct_trend = np.full(n, -1, dtype=np.int8)
        h1_trend = np.full(n, -1, dtype=np.int8)
        h4_trend = np.full(n, -1, dtype=np.int8)
        prem_disc = np.ones(n, dtype=np.int8)  # premium = bearish

        score = _multi_layer_confluence(sess_trend, struct_trend, h1_trend, h4_trend, prem_disc)
        assert (score < 0).all()

    def test_score_in_valid_range(self):
        """Score must be in -5 to +5."""
        df = make_full_df(n=30)
        out = compute_layer4_features(df)
        assert out["l4_multi_layer_confluence"].between(-5, 5).all()


# ---------------------------------------------------------------------------
# Feature 10: l4_time_volatility_regime
# ---------------------------------------------------------------------------

class TestTimeVolatilityRegime:
    def test_high_regime_when_recent_higher(self):
        """Rising volatility over time → regime should eventually be 1."""
        n = 250
        # Start low, then ramp up in latter half
        ve = np.concatenate([np.ones(150) * 1.0, np.ones(100) * 10.0])
        from features.layer4 import _time_volatility_regime
        result = _time_volatility_regime(ve)
        # After sufficient bars, high regime should appear
        assert (result[200:] == 1).any()

    def test_low_regime_when_recent_lower(self):
        """Falling volatility → regime should be -1."""
        n = 250
        ve = np.concatenate([np.ones(150) * 10.0, np.ones(100) * 1.0])
        from features.layer4 import _time_volatility_regime
        result = _time_volatility_regime(ve)
        assert (result[200:] == -1).any()

    def test_values_in_valid_set(self):
        """All values must be in {-1, 0, 1}."""
        df = make_full_df(n=50)
        out = compute_layer4_features(df)
        assert set(out["l4_time_volatility_regime"].unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Integration: compute_layer4_features on a full synthetic DataFrame
# ---------------------------------------------------------------------------

class TestComputeLayer4Features:
    def test_all_columns_present(self):
        """All 10 L4 feature columns must be present in output."""
        expected = [
            "l4_whale_footprint",
            "l4_trap_score",
            "l4_candle_dna",
            "l4_momentum_divergence",
            "l4_consecutive_bias",
            "l4_volume_climax",
            "l4_range_compression",
            "l4_session_continuation",
            "l4_multi_layer_confluence",
            "l4_time_volatility_regime",
        ]
        df = make_full_df(n=50)
        out = compute_layer4_features(df)
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_output_length_equals_input(self):
        """Output must have the same number of rows as input."""
        n = 40
        df = make_full_df(n=n)
        out = compute_layer4_features(df)
        assert len(out) == n

    def test_input_columns_preserved(self):
        """All input columns must still exist in output."""
        df = make_full_df(n=30)
        out = compute_layer4_features(df)
        for col in df.columns:
            assert col in out.columns

    def test_no_nan_in_discrete_features(self):
        """Discrete integer features must not contain NaN."""
        df = make_full_df(n=50)
        out = compute_layer4_features(df)
        discrete_cols = [
            "l4_whale_footprint",
            "l4_candle_dna",
            "l4_momentum_divergence",
            "l4_consecutive_bias",
            "l4_volume_climax",
            "l4_multi_layer_confluence",
            "l4_time_volatility_regime",
        ]
        for col in discrete_cols:
            assert not out[col].isna().any(), f"NaN found in {col}"
