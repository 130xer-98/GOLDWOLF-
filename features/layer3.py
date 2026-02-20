"""
GOLDWOLF — Layer 3 SMC (Smart Money Concepts) Features
Computes structural market features from M15 price action.

NO-LOOKAHEAD RULE
-----------------
* ``l3_swing_high`` / ``l3_swing_low`` flags are computed using N candles on
  *both* sides (standard definition) and therefore contain an N-bar embedded
  look-ahead in the flag label itself.
* All *trading-decision* features (BOS, CHoCH, order blocks, etc.) are
  computed with strict no-look-ahead: only swing points confirmed at least
  N bars ago (i.e. swing_idx <= current_idx - N) are used.

Output columns (all prefixed ``l3_``)
--------------------------------------
l3_swing_high            1 = confirmed swing high
l3_swing_low             1 = confirmed swing low
l3_bos_direction         1=bull BOS, -1=bear BOS, 0=none
l3_choch_flag            1=bull CHoCH, -1=bear CHoCH, 0=none
l3_demand_ob_distance    pips from close to active demand order block
l3_supply_ob_distance    pips from close to active supply order block
l3_fvg_active            1=bullish FVG below, -1=bearish FVG above, 0=none
l3_fvg_distance          pips to nearest unfilled FVG midpoint
l3_buy_liq_distance      pips to nearest buy-side liquidity pool
l3_sell_liq_distance     pips to nearest sell-side liquidity pool
l3_liquidity_sweep       1=buy-side sweep, -1=sell-side sweep, 0=none
l3_premium_discount      1=premium, -1=discount, 0=equilibrium
l3_structure_trend       1=bullish, -1=bearish, 0=ranging
l3_h1_trend              H1 timeframe trend (1/-1/0), forward-filled to M15
l3_h4_trend              H4 timeframe trend (1/-1/0), forward-filled to M15
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from config.settings import (
    EPSILON,
    PIP_SIZE,
    SWING_LOOKBACK,
    LIQUIDITY_TOLERANCE_PIPS,
    LIQUIDITY_MIN_TOUCHES,
    LIQUIDITY_LOOKBACK,
    FVG_MIN_GAP_PIPS,
    OB_MAX_AGE,
)
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

_M15_OPEN = "m15_open"
_M15_HIGH = "m15_high"
_M15_LOW = "m15_low"
_M15_CLOSE = "m15_close"

# Tolerance in price units (converted from pips once)
_LIQ_TOL = LIQUIDITY_TOLERANCE_PIPS * PIP_SIZE
_FVG_MIN = FVG_MIN_GAP_PIPS * PIP_SIZE

# Premium/discount equilibrium band (10 % of range on each side of mid)
_EQ_HALF = 0.05


# ---------------------------------------------------------------------------
# Data container for an active order block
# ---------------------------------------------------------------------------
class _OrderBlock(NamedTuple):
    idx: int        # bar index where the OB was identified
    level: float    # representative price level (mid of OB candle body)
    ob_low: float   # low of the OB candle body  (demand OB mitigation check)
    ob_high: float  # high of the OB candle body (supply OB mitigation check)
    kind: str       # "demand" or "supply"


# ---------------------------------------------------------------------------
# Data container for an active Fair Value Gap
# ---------------------------------------------------------------------------
class _FVG(NamedTuple):
    idx: int        # bar index at which the FVG was created
    lo: float       # lower bound of the gap
    hi: float       # upper bound of the gap
    kind: str       # "bull" or "bear"


# ---------------------------------------------------------------------------
# Swing high / low detection (vectorised)
# ---------------------------------------------------------------------------


def _detect_swings(
    high: np.ndarray,
    low: np.ndarray,
    n: int = SWING_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return boolean arrays marking swing highs and swing lows.

    A swing high at bar *i* is confirmed when high[i] is strictly greater
    than the highest high in the *n* bars before *and* the *n* bars after.
    The *n* edge bars at each end cannot be confirmed and are set to 0.

    Parameters
    ----------
    high, low : np.ndarray  shape (m,)
    n         : int         lookback / look-ahead bars

    Returns
    -------
    (swing_high, swing_low) — both int8, shape (m,).
    """
    length = len(high)
    sh = np.zeros(length, dtype=np.int8)
    sl = np.zeros(length, dtype=np.int8)

    if length < 2 * n + 1:
        return sh, sl

    highs = pd.Series(high)
    lows = pd.Series(low)

    # Rolling max of left N bars (exclusive of current)
    left_max_h = highs.rolling(n, min_periods=n).max().shift(1).values
    left_min_l = lows.rolling(n, min_periods=n).min().shift(1).values

    # Rolling max of right N bars (exclusive of current) — reverse trick
    right_max_h = highs[::-1].rolling(n, min_periods=n).max().shift(1).values[::-1]
    right_min_l = lows[::-1].rolling(n, min_periods=n).min().shift(1).values[::-1]

    valid = ~(
        np.isnan(left_max_h)
        | np.isnan(right_max_h)
        | np.isnan(left_min_l)
        | np.isnan(right_min_l)
    )
    sh[valid & (high > left_max_h) & (high > right_max_h)] = 1
    sl[valid & (low < left_min_l) & (low < right_min_l)] = 1

    return sh, sl


# ---------------------------------------------------------------------------
# Higher-timeframe trend helper
# ---------------------------------------------------------------------------


def _htf_trend(
    df_m15: pd.DataFrame,
    resample_rule: str,
) -> pd.Series:
    """
    Resample M15 OHLC to a higher timeframe, detect swing-based structure,
    and return a series of trend labels (1/-1/0) indexed by M15 timestamp
    (forward-filled).

    Parameters
    ----------
    df_m15       : pd.DataFrame  M15 bars with m15_open/high/low/close
    resample_rule: str           pandas resample rule, e.g. "1h" or "4h"

    Returns
    -------
    pd.Series  int8, indexed like *df_m15*.
    """
    # Aggregate OHLC
    htf = df_m15[[_M15_OPEN, _M15_HIGH, _M15_LOW, _M15_CLOSE]].resample(
        resample_rule
    ).agg(
        {
            _M15_OPEN: "first",
            _M15_HIGH: "max",
            _M15_LOW: "min",
            _M15_CLOSE: "last",
        }
    ).dropna()

    if len(htf) < 2 * SWING_LOOKBACK + 1:
        return pd.Series(
            np.zeros(len(df_m15), dtype=np.int8), index=df_m15.index
        )

    high = htf[_M15_HIGH].values
    low = htf[_M15_LOW].values

    sh, sl = _detect_swings(high, low, n=SWING_LOOKBACK)

    trend = _swing_structure_trend(sh, sl, high, low)

    htf_series = pd.Series(trend, index=htf.index, dtype=np.int8)
    # Reindex to M15 timestamps and forward-fill
    result = (
        htf_series
        .reindex(htf_series.index.union(df_m15.index))
        .ffill()
        .reindex(df_m15.index)
        .fillna(0)
        .astype(np.int8)
    )
    return result


# ---------------------------------------------------------------------------
# Structure trend from last 4 swing points
# ---------------------------------------------------------------------------


def _swing_structure_trend(
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    """
    Determine market structure trend at every bar based on the last 4
    confirmed swing points (alternating SH and SL).

    Higher Highs + Higher Lows = Bullish  (1)
    Lower  Highs + Lower  Lows = Bearish  (-1)
    Mixed                       = Ranging  (0)

    Returns
    -------
    np.ndarray, int8, shape (n,).
    """
    n = len(high)
    trend_arr = np.zeros(n, dtype=np.int8)

    swing_pts: list[tuple[int, float, str]] = []  # (idx, price, 'H'|'L')

    for i in range(n):
        if swing_high[i]:
            swing_pts.append((i, float(high[i]), "H"))
        if swing_low[i]:
            swing_pts.append((i, float(low[i]), "L"))

        # Need at least 4 swing points
        if len(swing_pts) < 4:
            continue

        last4 = swing_pts[-4:]
        # Filter to last 2 highs and last 2 lows
        highs_4 = [p[1] for p in last4 if p[2] == "H"]
        lows_4 = [p[1] for p in last4 if p[2] == "L"]

        if len(highs_4) >= 2 and len(lows_4) >= 2:
            hh = highs_4[-1] > highs_4[-2]
            hl = lows_4[-1] > lows_4[-2]
            lh = highs_4[-1] < highs_4[-2]
            ll = lows_4[-1] < lows_4[-2]

            if hh and hl:
                trend_arr[i] = 1
            elif lh and ll:
                trend_arr[i] = -1
            else:
                trend_arr[i] = 0
        else:
            trend_arr[i] = 0

    return trend_arr


# ---------------------------------------------------------------------------
# Main iterative SMC engine
# ---------------------------------------------------------------------------


def _compute_smc(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    n_swing: int = SWING_LOOKBACK,
) -> dict[str, np.ndarray]:
    """
    Compute all iterative SMC features in a single forward pass.

    Parameters
    ----------
    open_, high, low, close : np.ndarray  shape (m,)
    swing_high, swing_low   : np.ndarray  int8, shape (m,)
    n_swing                 : int         SWING_LOOKBACK

    Returns
    -------
    dict mapping column name → np.ndarray.
    """
    m = len(close)

    bos_dir = np.zeros(m, dtype=np.int8)
    choch_flag = np.zeros(m, dtype=np.int8)
    demand_ob_dist = np.zeros(m, dtype=np.float32)
    supply_ob_dist = np.zeros(m, dtype=np.float32)
    fvg_active = np.zeros(m, dtype=np.int8)
    fvg_dist = np.zeros(m, dtype=np.float32)
    buy_liq_dist = np.zeros(m, dtype=np.float32)
    sell_liq_dist = np.zeros(m, dtype=np.float32)
    liq_sweep = np.zeros(m, dtype=np.int8)
    prem_disc = np.zeros(m, dtype=np.int8)

    # --- BOS / CHoCH state ---
    last_sh_price: float | None = None   # most recent confirmed swing-high price
    last_sl_price: float | None = None   # most recent confirmed swing-low price
    last_sh_idx: int = -9999
    last_sl_idx: int = -9999
    prevailing_trend: int = 0            # +1 bull, -1 bear, 0 neutral

    # --- Order block state ---
    active_demand_obs: list[_OrderBlock] = []
    active_supply_obs: list[_OrderBlock] = []

    # --- FVG state ---
    active_bull_fvgs: list[_FVG] = []
    active_bear_fvgs: list[_FVG] = []

    # --- Premium/discount: last confirmed swing range ---
    last_major_sh: float | None = None
    last_major_sl: float | None = None

    for i in range(m):
        # ----------------------------------------------------------------
        # 1.  Register swing points as they become confirmed.
        #     A swing at position j is confirmed when we reach bar j + n_swing.
        # ----------------------------------------------------------------
        conf_idx = i - n_swing  # confirmed-swing position for this bar
        if conf_idx >= 0:
            if swing_high[conf_idx]:
                # Update to most recent confirmed swing high (used as BOS target)
                last_sh_price = float(high[conf_idx])
                last_sh_idx = conf_idx
                last_major_sh = last_sh_price

            if swing_low[conf_idx]:
                # Update to most recent confirmed swing low (used as BOS target)
                last_sl_price = float(low[conf_idx])
                last_sl_idx = conf_idx
                last_major_sl = last_sl_price

        cl = float(close[i])
        hi = float(high[i])
        lo = float(low[i])
        op = float(open_[i])

        # ----------------------------------------------------------------
        # 2.  BOS detection
        # ----------------------------------------------------------------
        bos = 0
        if last_sh_price is not None and cl > last_sh_price:
            bos = 1
            # Identify demand OB: last bearish candle *before* the BOS
            for j in range(i - 1, max(last_sh_idx, 0) - 1, -1):
                if float(close[j]) < float(open_[j]):  # bearish candle
                    ob_lo = min(float(open_[j]), float(close[j]))
                    ob_hi = max(float(open_[j]), float(close[j]))
                    ob_mid = (ob_lo + ob_hi) / 2.0
                    active_demand_obs.append(
                        _OrderBlock(i, ob_mid, ob_lo, ob_hi, "demand")
                    )
                    break
            last_sh_price = None  # consumed by BOS
        elif last_sl_price is not None and cl < last_sl_price:
            bos = -1
            # Identify supply OB: last bullish candle before the BOS
            for j in range(i - 1, max(last_sl_idx, 0) - 1, -1):
                if float(close[j]) > float(open_[j]):  # bullish candle
                    ob_lo = min(float(open_[j]), float(close[j]))
                    ob_hi = max(float(open_[j]), float(close[j]))
                    ob_mid = (ob_lo + ob_hi) / 2.0
                    active_supply_obs.append(
                        _OrderBlock(i, ob_mid, ob_lo, ob_hi, "supply")
                    )
                    break
            last_sl_price = None  # consumed by BOS

        bos_dir[i] = bos

        # ----------------------------------------------------------------
        # 3.  CHoCH: first BOS in the opposite direction of prevailing trend
        # ----------------------------------------------------------------
        choch = 0
        if bos != 0:
            if prevailing_trend == 1 and bos == -1:
                choch = -1
            elif prevailing_trend == -1 and bos == 1:
                choch = 1
            prevailing_trend = bos  # update trend
        choch_flag[i] = choch

        # ----------------------------------------------------------------
        # 4.  Order block distances + mitigation
        # ----------------------------------------------------------------
        # Demand OBs
        nearest_dem = np.inf
        survived: list[_OrderBlock] = []
        for ob in active_demand_obs:
            # Mitigated if close goes below OB low
            if cl < ob.ob_low:
                continue
            # Expired
            if (i - ob.idx) > OB_MAX_AGE:
                continue
            dist = (cl - ob.level) / PIP_SIZE
            if abs(dist) < abs(nearest_dem):
                nearest_dem = dist
            survived.append(ob)
        active_demand_obs = survived
        demand_ob_dist[i] = 0.0 if np.isinf(nearest_dem) else float(nearest_dem)

        # Supply OBs
        nearest_sup = np.inf
        survived2: list[_OrderBlock] = []
        for ob in active_supply_obs:
            # Mitigated if close goes above OB high
            if cl > ob.ob_high:
                continue
            if (i - ob.idx) > OB_MAX_AGE:
                continue
            dist = (ob.level - cl) / PIP_SIZE
            if abs(dist) < abs(nearest_sup):
                nearest_sup = dist
            survived2.append(ob)
        active_supply_obs = survived2
        supply_ob_dist[i] = 0.0 if np.isinf(nearest_sup) else float(nearest_sup)

        # ----------------------------------------------------------------
        # 5.  FVG detection + mitigation
        # ----------------------------------------------------------------
        # Detect new FVGs at current bar (requires bar i-2)
        if i >= 2:
            gap_bull = float(low[i]) - float(high[i - 2])
            gap_bear = float(low[i - 2]) - float(high[i])
            if gap_bull >= _FVG_MIN:
                active_bull_fvgs.append(
                    _FVG(i, float(high[i - 2]), float(low[i]), "bull")
                )
            if gap_bear >= _FVG_MIN:
                active_bear_fvgs.append(
                    _FVG(i, float(high[i]), float(low[i - 2]), "bear")
                )

        # Check active bull FVGs (unfilled gap *below* current price)
        nearest_bull_fvg_dist = np.inf
        nearest_bull_fvg_val = 0.0
        kept_bull: list[_FVG] = []
        for fvg in active_bull_fvgs:
            mid = (fvg.lo + fvg.hi) / 2.0
            # Filled when a subsequent bar's low enters the gap zone.
            # Skip the fill check on the detection bar itself (lo == fvg.hi by
            # definition, which would incorrectly mark it as immediately filled).
            if i > fvg.idx and lo <= fvg.hi:
                continue
            dist_pips = (cl - mid) / PIP_SIZE
            if abs(dist_pips) < nearest_bull_fvg_dist:
                nearest_bull_fvg_dist = abs(dist_pips)
                nearest_bull_fvg_val = dist_pips
            kept_bull.append(fvg)
        active_bull_fvgs = kept_bull

        # Check active bear FVGs (unfilled gap *above* current price)
        nearest_bear_fvg_dist = np.inf
        nearest_bear_fvg_val = 0.0
        kept_bear: list[_FVG] = []
        for fvg in active_bear_fvgs:
            mid = (fvg.lo + fvg.hi) / 2.0
            # Filled when a subsequent bar's high enters the gap zone.
            if i > fvg.idx and hi >= fvg.lo:
                continue
            dist_pips = (mid - cl) / PIP_SIZE
            if abs(dist_pips) < nearest_bear_fvg_dist:
                nearest_bear_fvg_dist = abs(dist_pips)
                nearest_bear_fvg_val = dist_pips
            kept_bear.append(fvg)
        active_bear_fvgs = kept_bear

        if not np.isinf(nearest_bull_fvg_dist):
            fvg_active[i] = 1
            fvg_dist[i] = float(nearest_bull_fvg_val)
        elif not np.isinf(nearest_bear_fvg_dist):
            fvg_active[i] = -1
            fvg_dist[i] = float(nearest_bear_fvg_val)

        # ----------------------------------------------------------------
        # 6.  Liquidity pools (sliding window)
        # ----------------------------------------------------------------
        start = max(0, i - LIQUIDITY_LOOKBACK + 1)

        # Buy-side (equal highs)
        window_highs = high[start : i + 1]
        buy_pool = _find_liquidity_pool(window_highs)
        if buy_pool is not None:
            buy_liq_dist[i] = float((buy_pool - cl) / PIP_SIZE)

        # Sell-side (equal lows)
        window_lows = low[start : i + 1]
        sell_pool = _find_liquidity_pool(-window_lows)  # flip to use same fn
        if sell_pool is not None:
            sell_pool = -sell_pool  # restore
            sell_liq_dist[i] = float((cl - sell_pool) / PIP_SIZE)

        # ----------------------------------------------------------------
        # 7.  Liquidity sweep detection
        # ----------------------------------------------------------------
        sweep = 0
        # Buy-side sweep: high exceeds pool but close below pool
        if buy_pool is not None and hi > buy_pool and cl < buy_pool:
            sweep = 1
        # Sell-side sweep: low undercuts pool but close above pool
        sell_pool2 = None
        if sell_pool is None:
            window_lows2 = low[start : i + 1]
            sell_pool2_raw = _find_liquidity_pool(-window_lows2)
            if sell_pool2_raw is not None:
                sell_pool2 = -sell_pool2_raw
        else:
            sell_pool2 = sell_pool

        if sell_pool2 is not None and lo < sell_pool2 and cl > sell_pool2:
            sweep = -1
        liq_sweep[i] = sweep

        # ----------------------------------------------------------------
        # 8.  Premium / discount zones
        # ----------------------------------------------------------------
        if last_major_sh is not None and last_major_sl is not None:
            rng = last_major_sh - last_major_sl
            if rng > EPSILON:
                pos = (cl - last_major_sl) / rng
                if pos > 0.5 + _EQ_HALF:
                    prem_disc[i] = 1
                elif pos < 0.5 - _EQ_HALF:
                    prem_disc[i] = -1
                else:
                    prem_disc[i] = 0

    return {
        "l3_bos_direction": bos_dir,
        "l3_choch_flag": choch_flag,
        "l3_demand_ob_distance": demand_ob_dist,
        "l3_supply_ob_distance": supply_ob_dist,
        "l3_fvg_active": fvg_active,
        "l3_fvg_distance": fvg_dist,
        "l3_buy_liq_distance": buy_liq_dist,
        "l3_sell_liq_distance": sell_liq_dist,
        "l3_liquidity_sweep": liq_sweep,
        "l3_premium_discount": prem_disc,
    }


def _find_liquidity_pool(prices: np.ndarray) -> float | None:
    """
    Find the modal cluster in *prices* (interpreted as highs or lows)
    within a tolerance band of _LIQ_TOL.

    Returns the mean of the cluster if at least LIQUIDITY_MIN_TOUCHES
    touches are found, else None.

    Parameters
    ----------
    prices : np.ndarray   1-D array of high or (negated) low prices.

    Returns
    -------
    float | None
    """
    if len(prices) < LIQUIDITY_MIN_TOUCHES:
        return None

    best_level: float | None = None
    best_count = LIQUIDITY_MIN_TOUCHES - 1

    # Check each unique price level as a cluster centre
    for anchor in prices:
        mask = np.abs(prices - anchor) <= _LIQ_TOL
        cnt = int(mask.sum())
        if cnt > best_count:
            best_count = cnt
            best_level = float(prices[mask].mean())

    return best_level


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------


def compute_layer3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Layer 3 SMC features for every M15 bar in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``compute_layer2_features`` — one row per M15 bar with a
        DatetimeIndex and all Layer 1 + Layer 2 feature columns.

    Returns
    -------
    pd.DataFrame
        *df* with all ``l3_*`` feature columns appended.
    """
    logger.info("Computing Layer 3 (SMC) features for %d M15 bars …", len(df))

    with Timer("layer3 features") as t:
        open_ = df[_M15_OPEN].values.astype(np.float64)
        high = df[_M15_HIGH].values.astype(np.float64)
        low = df[_M15_LOW].values.astype(np.float64)
        close = df[_M15_CLOSE].values.astype(np.float64)

        # --- Swing detection ---
        swing_high, swing_low = _detect_swings(high, low, n=SWING_LOOKBACK)

        # --- Structure trend ---
        struct_trend = _swing_structure_trend(swing_high, swing_low, high, low)

        # --- Main iterative SMC engine ---
        smc_cols = _compute_smc(open_, high, low, close, swing_high, swing_low)

        # --- Higher-timeframe trends ---
        h1_trend = _htf_trend(df, "1h")
        h4_trend = _htf_trend(df, "4h")

    logger.info("Layer 3 features computed in %s", t.elapsed_str)

    features = pd.DataFrame(
        {
            "l3_swing_high": swing_high,
            "l3_swing_low": swing_low,
            **smc_cols,
            "l3_structure_trend": struct_trend,
            "l3_h1_trend": h1_trend.values,
            "l3_h4_trend": h4_trend.values,
        },
        index=df.index,
    )
    return df.join(features)
