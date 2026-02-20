"""
GOLDWOLF — Layer 4 Private Edge Features
Computes 10 meta-features from the existing L1, L2, L3 features + raw OHLC.

NO-LOOKAHEAD RULE
-----------------
All features look backward only.  No feature at time T uses data from T+1
or later.  Labels (forward-looking) live in model/labeler.py.

Output columns (all prefixed ``l4_``)
--------------------------------------
l4_whale_footprint       0-3 institutional activity score
l4_trap_score            0-100 retail trap probability
l4_candle_dna            -1..7 candle archetype classification
l4_momentum_divergence   -1/0/1 price vs energy divergence
l4_consecutive_bias      ±20 streak of bullish/bearish closes
l4_volume_climax         0/1 institutional climax event flag
l4_range_compression     ratio of current range to rolling 20-bar average
l4_session_continuation  0-1 probability of session trend continuation
l4_multi_layer_confluence -5..+5 layer agreement score
l4_time_volatility_regime -1/0/1 volatility environment classification
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    EPSILON,
    PIP_SIZE,
    L4_WHALE_VOLUME_MIN,
    L4_WHALE_REVERSAL_MAX,
    L4_WHALE_ABSORPTION_MIN,
    L4_TRAP_OB_DISTANCE_PIPS,
    L4_TRAP_ABSORPTION_THRESHOLD,
    L4_DNA_ABSORPTION_BODY_MAX,
    L4_DNA_EXHAUSTION_RATIO_MAX,
    L4_DNA_EXHAUSTION_ACCEL_MAX,
    L4_DNA_TRAP_SCORE_MIN,
    L4_DNA_DEAD_VOLUME_MAX,
    L4_DNA_DEAD_ENERGY_MAX,
    L4_DNA_ACCUM_OB_PIPS,
    L4_DNA_ACCUM_ABSORPTION_MIN,
    L4_CLIMAX_WINDOW,
    L4_CLIMAX_SIGMA,
    L4_RANGE_WINDOW,
    L4_SESSION_CONTINUATION_WINDOW,
    L4_REGIME_SHORT_WINDOW,
    L4_REGIME_LONG_WINDOW,
    L4_REGIME_TRANSITION_BAND,
)
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal column aliases
# ---------------------------------------------------------------------------
_M15_OPEN = "m15_open"
_M15_HIGH = "m15_high"
_M15_LOW = "m15_low"
_M15_CLOSE = "m15_close"


# ---------------------------------------------------------------------------
# Feature 1: l4_whale_footprint
# ---------------------------------------------------------------------------


def _whale_footprint(
    custom_volume: np.ndarray,
    reversal_count: np.ndarray,
    absorption_count: np.ndarray,
) -> np.ndarray:
    """
    Score 0-3: count of conditions met for institutional candle presence.

    Conditions:
      - l1_custom_volume >= L4_WHALE_VOLUME_MIN
      - l1_reversal_count < L4_WHALE_REVERSAL_MAX  (clean directional move)
      - l1_absorption_count >= L4_WHALE_ABSORPTION_MIN
    """
    cond1 = (custom_volume >= L4_WHALE_VOLUME_MIN).astype(np.int8)
    cond2 = (reversal_count < L4_WHALE_REVERSAL_MAX).astype(np.int8)
    cond3 = (absorption_count >= L4_WHALE_ABSORPTION_MIN).astype(np.int8)
    return (cond1 + cond2 + cond3).astype(np.int8)


# ---------------------------------------------------------------------------
# Feature 2: l4_trap_score
# ---------------------------------------------------------------------------


def _trap_score(
    liquidity_sweep: np.ndarray,
    kill_zone: np.ndarray,
    close: np.ndarray,
    demand_ob_dist: np.ndarray,
    supply_ob_dist: np.ndarray,
    absorption_intensity: np.ndarray,
) -> np.ndarray:
    """
    Combine signals into a 0-100 trap probability score.

    Points awarded:
      +30  liquidity sweep detected
      +20  in kill zone
      +20  reversal candle (close opposite to previous close direction)
      +15  at order block (< L4_TRAP_OB_DISTANCE_PIPS away)
      +15  high absorption intensity (> L4_TRAP_ABSORPTION_THRESHOLD)
    """
    n = len(close)
    score = np.zeros(n, dtype=np.int32)

    score += (liquidity_sweep != 0).astype(np.int32) * 30
    score += (kill_zone > 0).astype(np.int32) * 20

    # Reversal candle: current close opposite to previous bar's direction
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_open = np.roll(close, 2)
    prev_open[0] = close[0]
    prev_open[1] = close[0]
    # Previous bar was bullish if close[i-1] > close[i-2]; current bar is
    # bearish if close[i] < close[i-1] — proxy for reversal
    prev_bull = prev_close > prev_open
    curr_bear = close < prev_close
    prev_bear = prev_close < prev_open
    curr_bull = close > prev_close
    reversal = (prev_bull & curr_bear) | (prev_bear & curr_bull)
    score += reversal.astype(np.int32) * 20

    # At order block
    near_ob = (np.abs(demand_ob_dist) < L4_TRAP_OB_DISTANCE_PIPS) | (
        np.abs(supply_ob_dist) < L4_TRAP_OB_DISTANCE_PIPS
    )
    score += near_ob.astype(np.int32) * 15

    score += (absorption_intensity > L4_TRAP_ABSORPTION_THRESHOLD).astype(np.int32) * 15

    return np.clip(score, 0, 100).astype(np.int32)


# ---------------------------------------------------------------------------
# Feature 3: l4_candle_dna
# ---------------------------------------------------------------------------


def _candle_dna(
    price_velocity: np.ndarray,
    reversal_count: np.ndarray,
    custom_volume: np.ndarray,
    absorption_intensity: np.ndarray,
    early_late_ratio: np.ndarray,
    price_acceleration: np.ndarray,
    trap_score: np.ndarray,
    volatility_energy: np.ndarray,
    bos_direction: np.ndarray,
    choch_flag: np.ndarray,
    demand_ob_dist: np.ndarray,
    supply_ob_dist: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Classify each candle into archetypes (-1 = unclassified, 0-7).

    Priority: 3 > 5 > 6 > 0 > 1 > 2 > 7 > 4
    """
    n = len(close)
    result = np.full(n, -1, dtype=np.int8)

    # Pre-compute rolling percentiles for velocity and range
    vel_series = pd.Series(price_velocity)
    vel_75 = vel_series.rolling(200, min_periods=10).quantile(0.75).values
    vel_95 = vel_series.rolling(200, min_periods=10).quantile(0.95).values

    range_arr = high - low
    range_series = pd.Series(range_arr)
    range_25 = range_series.rolling(200, min_periods=10).quantile(0.25).values

    # Body ratio for absorption check
    bar_range = high - low
    body = np.abs(close - open_)
    body_ratio = body / (bar_range + EPSILON)

    # Previous candle values for Type 6 (fakeout)
    prev_bos = np.roll(bos_direction, 1)
    prev_bos[0] = 0

    for i in range(n):
        vth = vel_75[i] if not np.isnan(vel_75[i]) else np.inf
        v95 = vel_95[i] if not np.isnan(vel_95[i]) else np.inf
        r25 = range_25[i] if not np.isnan(range_25[i]) else 0.0
        pv = float(price_velocity[i])

        # Type 3 — Trap
        if trap_score[i] >= L4_DNA_TRAP_SCORE_MIN:
            result[i] = 3
            continue

        # Type 5 — Breakout
        if pv > v95 and bos_direction[i] != 0:
            result[i] = 5
            continue

        # Type 6 — Fakeout
        if prev_bos[i] != 0 and choch_flag[i] != 0:
            result[i] = 6
            continue

        # Type 0 — Clean Push
        if (
            pv > vth
            and reversal_count[i] < L4_WHALE_REVERSAL_MAX
            and custom_volume[i] >= L4_WHALE_VOLUME_MIN
        ):
            result[i] = 0
            continue

        # Type 1 — Absorption
        if (
            absorption_intensity[i] > L4_TRAP_ABSORPTION_THRESHOLD
            and body_ratio[i] < L4_DNA_ABSORPTION_BODY_MAX
        ):
            result[i] = 1
            continue

        # Type 2 — Exhaustion
        if (
            early_late_ratio[i] < L4_DNA_EXHAUSTION_RATIO_MAX
            and price_acceleration[i] < L4_DNA_EXHAUSTION_ACCEL_MAX
        ):
            result[i] = 2
            continue

        # Type 7 — Accumulation
        near_ob = (np.abs(demand_ob_dist[i]) < L4_DNA_ACCUM_OB_PIPS) or (
            np.abs(supply_ob_dist[i]) < L4_DNA_ACCUM_OB_PIPS
        )
        if (
            range_arr[i] < r25
            and custom_volume[i] >= L4_DNA_ACCUM_ABSORPTION_MIN
            and near_ob
        ):
            result[i] = 7
            continue

        # Type 4 — Dead Zone
        if (
            custom_volume[i] < L4_DNA_DEAD_VOLUME_MAX
            and volatility_energy[i] < L4_DNA_DEAD_ENERGY_MAX
        ):
            result[i] = 4
            continue

    return result


# ---------------------------------------------------------------------------
# Feature 4: l4_momentum_divergence
# ---------------------------------------------------------------------------


def _momentum_divergence(
    close: np.ndarray,
    volatility_energy: np.ndarray,
    lookback: int = 5,
) -> np.ndarray:
    """
    Compare price direction vs energy direction over last *lookback* candles.

    -1 = bearish divergence (price up, energy down)
     0 = no divergence
     1 = bullish divergence (price down, energy up)
    """
    n = len(close)
    result = np.zeros(n, dtype=np.int8)

    for i in range(lookback, n):
        price_up = close[i] > close[i - lookback]
        energy_up = volatility_energy[i] > volatility_energy[i - lookback]

        if price_up and not energy_up:
            result[i] = -1  # bearish divergence
        elif not price_up and energy_up:
            result[i] = 1   # bullish divergence

    return result


# ---------------------------------------------------------------------------
# Feature 5: l4_consecutive_bias
# ---------------------------------------------------------------------------


def _consecutive_bias(
    close: np.ndarray,
    open_: np.ndarray,
    cap: int = 20,
) -> np.ndarray:
    """
    Count consecutive candles with the same directional bias.

    Positive value = bullish streak length.
    Negative value = bearish streak length.
    Capped at ±cap.
    """
    n = len(close)
    result = np.zeros(n, dtype=np.int8)
    streak = 0

    for i in range(n):
        bullish = close[i] > open_[i]
        bearish = close[i] < open_[i]

        if bullish:
            streak = max(1, streak + 1) if streak >= 0 else 1
        elif bearish:
            streak = min(-1, streak - 1) if streak <= 0 else -1
        # Doji (close == open) — keep previous streak

        result[i] = np.int8(np.clip(streak, -cap, cap))

    return result


# ---------------------------------------------------------------------------
# Feature 6: l4_volume_climax
# ---------------------------------------------------------------------------


def _volume_climax(
    custom_volume: np.ndarray,
    volatility_energy: np.ndarray,
) -> np.ndarray:
    """
    Flag = 1 if BOTH custom_volume AND volatility_energy are more than
    L4_CLIMAX_SIGMA standard deviations above their rolling L4_CLIMAX_WINDOW
    period means.
    """
    cv_series = pd.Series(custom_volume.astype(np.float64))
    ve_series = pd.Series(volatility_energy.astype(np.float64))

    roll_cv = cv_series.rolling(L4_CLIMAX_WINDOW, min_periods=L4_CLIMAX_WINDOW)
    roll_ve = ve_series.rolling(L4_CLIMAX_WINDOW, min_periods=L4_CLIMAX_WINDOW)

    cv_thresh = (roll_cv.mean() + L4_CLIMAX_SIGMA * roll_cv.std()).values
    ve_thresh = (roll_ve.mean() + L4_CLIMAX_SIGMA * roll_ve.std()).values

    cv_spike = ~np.isnan(cv_thresh) & (custom_volume > cv_thresh)
    ve_spike = ~np.isnan(ve_thresh) & (volatility_energy > ve_thresh)

    return (cv_spike & ve_spike).astype(np.int8)


# ---------------------------------------------------------------------------
# Feature 7: l4_range_compression
# ---------------------------------------------------------------------------


def _range_compression(
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    """
    Ratio of current M15 range (high - low) to rolling L4_RANGE_WINDOW-period
    average range.

    Returns raw ratio (not capped).  NaN for first window bars filled with 1.0.
    """
    bar_range = (high - low).astype(np.float64)
    series = pd.Series(bar_range)
    rolling_mean = series.rolling(L4_RANGE_WINDOW, min_periods=1).mean().values

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rolling_mean > EPSILON, bar_range / rolling_mean, 1.0)

    return ratio.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature 8: l4_session_continuation
# ---------------------------------------------------------------------------


def _session_continuation(
    session_trend: np.ndarray,
    session: np.ndarray,
) -> np.ndarray:
    """
    Rolling probability of session trend continuation from London→NY.

    Uses last L4_SESSION_CONTINUATION_WINDOW London sessions to estimate
    the probability that NY continues in the same direction as London close.

    Value 0-1 at the start of each NY session, carried forward throughout.
    Asian→London transition also supported (using Asian trend → London).
    """
    n = len(session_trend)
    result = np.zeros(n, dtype=np.float32)

    # Track London closing trend and whether NY continued it
    london_closing_trend: list[float] = []
    ny_continued: list[int] = []   # 1 if NY continued London, 0 otherwise
    ny_prob = 0.5  # default prior

    asian_closing_trend: list[float] = []
    london_continued: list[int] = []
    london_prob = 0.5

    prev_sess = int(session[0])
    current_prob = 0.5

    # Running trend at end of each session
    last_london_trend: float | None = None
    last_asian_trend: float | None = None

    # Track the trend at each session boundary
    sess_end_trend: float = 0.0

    for i in range(n):
        curr_sess = int(session[i])

        if i > 0 and curr_sess != prev_sess:
            # Session transition
            if prev_sess == 1 and curr_sess == 2:
                # London → NY
                last_london_trend = sess_end_trend
                # Check continuation from Asian→London
                if last_asian_trend is not None:
                    continued = int(
                        (sess_end_trend > 0) == (last_asian_trend > 0)
                        and last_asian_trend != 0
                    )
                    london_continued.append(continued)
                    asian_closing_trend.append(last_asian_trend)
                    if len(london_continued) > L4_SESSION_CONTINUATION_WINDOW:
                        london_continued.pop(0)
                        asian_closing_trend.pop(0)
                    if london_continued:
                        london_prob = sum(london_continued) / len(london_continued)

                # Compute NY continuation probability from London trend
                if last_london_trend is not None and last_london_trend != 0:
                    if len(ny_continued) > 0:
                        ny_prob = sum(ny_continued) / len(ny_continued)
                    current_prob = ny_prob
                else:
                    current_prob = 0.5

            elif prev_sess == 0 and curr_sess == 1:
                # Asian → London
                last_asian_trend = sess_end_trend
                if last_asian_trend is not None and last_asian_trend != 0:
                    current_prob = london_prob
                else:
                    current_prob = 0.5

            elif prev_sess == 2 and curr_sess == 0:
                # NY end → Asian: record NY continuation
                if last_london_trend is not None and last_london_trend != 0:
                    continued = int(
                        (sess_end_trend > 0) == (last_london_trend > 0)
                        and sess_end_trend != 0
                    )
                    ny_continued.append(continued)
                    london_closing_trend.append(last_london_trend)
                    if len(ny_continued) > L4_SESSION_CONTINUATION_WINDOW:
                        ny_continued.pop(0)
                        london_closing_trend.pop(0)
                    if ny_continued:
                        ny_prob = sum(ny_continued) / len(ny_continued)
                current_prob = 0.5

        sess_end_trend = float(session_trend[i])
        result[i] = np.float32(current_prob)
        prev_sess = curr_sess

    return result


# ---------------------------------------------------------------------------
# Feature 9: l4_multi_layer_confluence
# ---------------------------------------------------------------------------


def _multi_layer_confluence(
    session_trend: np.ndarray,
    structure_trend: np.ndarray,
    h1_trend: np.ndarray,
    h4_trend: np.ndarray,
    premium_discount: np.ndarray,
) -> np.ndarray:
    """
    Count how many layers agree on direction (-5 to +5).

    Each layer contributes +1 (bullish) or -1 (bearish) if it has a clear
    direction, 0 if neutral.  Premium/discount alignment:
      - premium zone (-1) contributes -1 (expect sell from premium)
      - discount zone (+1) contributes +1 (expect buy from discount)
    """
    # Normalise session_trend to -1/0/1
    sess_dir = np.sign(session_trend).astype(np.int8)

    # Premium/discount: flip sign so discount→buy, premium→sell
    pd_dir = (-premium_discount).astype(np.int8)

    score = (
        sess_dir
        + structure_trend.astype(np.int8)
        + h1_trend.astype(np.int8)
        + h4_trend.astype(np.int8)
        + pd_dir
    )
    return np.clip(score, -5, 5).astype(np.int8)


# ---------------------------------------------------------------------------
# Feature 10: l4_time_volatility_regime
# ---------------------------------------------------------------------------


def _time_volatility_regime(
    volatility_energy: np.ndarray,
) -> np.ndarray:
    """
    Classify current volatility environment based on short vs long rolling means.

     1 = High volatility regime (50-bar MA > 200-bar MA)
    -1 = Low volatility regime  (50-bar MA < 200-bar MA)
     0 = Transitioning (within 10 % of each other)
    """
    ve_series = pd.Series(volatility_energy.astype(np.float64))
    ma_short = ve_series.rolling(L4_REGIME_SHORT_WINDOW, min_periods=1).mean().values
    ma_long = ve_series.rolling(L4_REGIME_LONG_WINDOW, min_periods=1).mean().values

    result = np.zeros(len(volatility_energy), dtype=np.int8)

    band = np.abs(ma_long) * L4_REGIME_TRANSITION_BAND
    high_vol = ma_short > ma_long + band
    low_vol = ma_short < ma_long - band

    result[high_vol] = 1
    result[low_vol] = -1

    return result


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------


def compute_layer4_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Layer 4 private-edge features for every M15 bar in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``compute_layer3_features`` — one row per M15 bar with
        a DatetimeIndex and all L1, L2, L3 feature columns.

    Returns
    -------
    pd.DataFrame
        *df* with all ``l4_*`` feature columns appended.
    """
    logger.info("Computing Layer 4 features for %d M15 bars …", len(df))

    with Timer("layer4 features") as t:
        # Raw price arrays
        high = df[_M15_HIGH].values.astype(np.float64)
        low = df[_M15_LOW].values.astype(np.float64)
        close = df[_M15_CLOSE].values.astype(np.float64)
        open_ = df[_M15_OPEN].values.astype(np.float64)

        # L1 features
        custom_vol = df["l1_custom_volume"].values.astype(np.float64)
        vol_energy = df["l1_volatility_energy"].values.astype(np.float64)
        price_vel = df["l1_price_velocity"].values.astype(np.float64)
        rev_count = df["l1_reversal_count"].values.astype(np.float64)
        el_ratio = df["l1_early_late_ratio"].values.astype(np.float64)
        price_accel = df["l1_price_acceleration"].values.astype(np.float64)
        abs_count = df["l1_absorption_count"].values.astype(np.float64)
        abs_intensity = df["l1_absorption_intensity"].values.astype(np.float64)

        # L2 features
        kill_zone = df["l2_kill_zone"].values.astype(np.int8)
        session = df["l2_session"].values.astype(np.int8)
        session_trend = df["l2_session_trend"].values.astype(np.float64)

        # L3 features
        liq_sweep = df["l3_liquidity_sweep"].values.astype(np.int8)
        demand_ob_dist = df["l3_demand_ob_distance"].values.astype(np.float64)
        supply_ob_dist = df["l3_supply_ob_distance"].values.astype(np.float64)
        bos_dir = df["l3_bos_direction"].values.astype(np.int8)
        choch_flag = df["l3_choch_flag"].values.astype(np.int8)
        premium_disc = df["l3_premium_discount"].values.astype(np.int8)
        struct_trend = df["l3_structure_trend"].values.astype(np.int8)
        h1_trend = df["l3_h1_trend"].values.astype(np.int8)
        h4_trend = df["l3_h4_trend"].values.astype(np.int8)

        # --- Compute features ---
        whale = _whale_footprint(custom_vol, rev_count, abs_count)
        trap = _trap_score(
            liq_sweep, kill_zone, close, demand_ob_dist, supply_ob_dist, abs_intensity
        )
        dna = _candle_dna(
            price_vel, rev_count, custom_vol, abs_intensity, el_ratio, price_accel,
            trap, vol_energy, bos_dir, choch_flag, demand_ob_dist, supply_ob_dist,
            high, low, open_, close,
        )
        mom_div = _momentum_divergence(close, vol_energy)
        consec_bias = _consecutive_bias(close, open_)
        vol_climax = _volume_climax(custom_vol, vol_energy)
        range_comp = _range_compression(high, low)
        sess_cont = _session_continuation(session_trend, session)
        confluence = _multi_layer_confluence(
            session_trend, struct_trend, h1_trend, h4_trend, premium_disc
        )
        regime = _time_volatility_regime(vol_energy)

    logger.info("Layer 4 features computed in %s", t.elapsed_str)

    features = pd.DataFrame(
        {
            "l4_whale_footprint": whale,
            "l4_trap_score": trap,
            "l4_candle_dna": dna,
            "l4_momentum_divergence": mom_div,
            "l4_consecutive_bias": consec_bias,
            "l4_volume_climax": vol_climax,
            "l4_range_compression": range_comp,
            "l4_session_continuation": sess_cont,
            "l4_multi_layer_confluence": confluence,
            "l4_time_volatility_regime": regime,
        },
        index=df.index,
    )
    return df.join(features)
