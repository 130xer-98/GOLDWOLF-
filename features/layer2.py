"""
GOLDWOLF — Layer 2 Time DNA Features
Computes 10 time-based features for each M15 bar.

All features are computed using only past + current data (no lookahead).
Features are prefixed with ``l2_``.

Output columns
--------------
l2_session               Asian=0, London=1, NY=2
l2_session_overlap       1 if London+NY overlap window (13:00–16:00 GMT)
l2_kill_zone             Institutional kill zone (0=none,1=LO,2=NYO,3=LC)
l2_day_of_week           Mon=0 … Fri=4
l2_hour                  Hour of day (0–23)
l2_distance_from_session_open  (close − session_open) in pips
l2_session_position      (close − session_low) / (session_range + ε)
l2_time_since_vol_spike  M15 bars since last vol spike, capped at 50
l2_session_volatility_rank  Percentile rank of vol within same-session window
l2_session_trend         Normalised linear-regression slope over session
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from config.settings import (
    EPSILON,
    PIP_SIZE,
    SESSION_ASIAN_END,
    SESSION_LONDON_END,
    SESSION_OVERLAP_START,
    SESSION_OVERLAP_END,
    KZ_LONDON_OPEN,
    KZ_NY_OPEN,
    KZ_LONDON_CLOSE,
    VOL_SPIKE_WINDOW,
    VOL_SPIKE_SIGMA,
    VOL_SPIKE_CAP,
    SESSION_VOL_RANK_WINDOW,
    COL_VOLATILITY_ENERGY,
)
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal M15 column aliases
# ---------------------------------------------------------------------------
_M15_OPEN = "m15_open"
_M15_HIGH = "m15_high"
_M15_LOW = "m15_low"
_M15_CLOSE = "m15_close"

# ATR calculation window (bars) used in l2_session_trend
_ATR_WINDOW = 14


# ---------------------------------------------------------------------------
# Individual feature helpers
# ---------------------------------------------------------------------------


def _session_labels(hour: np.ndarray) -> np.ndarray:
    """Map GMT hour → session label: 0=Asian, 1=London, 2=NY."""
    session = np.zeros(len(hour), dtype=np.int8)
    session[hour >= SESSION_ASIAN_END] = 1    # London
    session[hour >= SESSION_LONDON_END] = 2   # New York
    return session


def _session_overlap(hour: np.ndarray) -> np.ndarray:
    """1 if 13:00–16:00 GMT (London+NY overlap), else 0."""
    return (
        (hour >= SESSION_OVERLAP_START) & (hour < SESSION_OVERLAP_END)
    ).astype(np.int8)


def _kill_zone(hour: np.ndarray, minute: np.ndarray) -> np.ndarray:
    """
    Kill zone encoding:
      1 = London Open  (08:00–09:00)
      2 = NY Open      (13:00–14:00)
      3 = London Close (15:30–16:30)
      0 = None
    """
    total_min = (hour * 60 + minute).astype(np.int32)
    kz = np.zeros(len(hour), dtype=np.int8)

    lo_s = KZ_LONDON_OPEN[0] * 60 + KZ_LONDON_OPEN[1]
    lo_e = KZ_LONDON_OPEN[2] * 60 + KZ_LONDON_OPEN[3]
    ny_s = KZ_NY_OPEN[0] * 60 + KZ_NY_OPEN[1]
    ny_e = KZ_NY_OPEN[2] * 60 + KZ_NY_OPEN[3]
    lc_s = KZ_LONDON_CLOSE[0] * 60 + KZ_LONDON_CLOSE[1]
    lc_e = KZ_LONDON_CLOSE[2] * 60 + KZ_LONDON_CLOSE[3]

    kz[(total_min >= lo_s) & (total_min < lo_e)] = 1
    kz[(total_min >= ny_s) & (total_min < ny_e)] = 2
    kz[(total_min >= lc_s) & (total_min < lc_e)] = 3
    return kz


def _session_running_stats(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    session: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute running per-session statistics without lookahead.

    For each bar the function returns:
    - ``session_open``: M15 open price of the first candle in the current session.
    - ``session_high``: running maximum of high prices since session start.
    - ``session_low``:  running minimum of low prices since session start.

    A new session begins whenever the session label changes from the
    previous bar.

    Parameters
    ----------
    open_, high, low, close : np.ndarray
        M15 price arrays (length n).
    session : np.ndarray
        Session labels (0/1/2) for each bar, length n.

    Returns
    -------
    (session_open, session_high, session_low) — each shape (n,), float64.
    """
    n = len(close)
    sess_open = np.empty(n, dtype=np.float64)
    sess_high = np.empty(n, dtype=np.float64)
    sess_low = np.empty(n, dtype=np.float64)

    cur_sess = int(session[0])
    cur_open = float(open_[0])
    cur_high = float(high[0])
    cur_low = float(low[0])

    for i in range(n):
        if i > 0 and int(session[i]) != cur_sess:
            # New session — reset all trackers to current bar values
            cur_sess = int(session[i])
            cur_open = float(open_[i])
            cur_high = float(high[i])
            cur_low = float(low[i])
        else:
            # Extend running high/low within the current session
            if float(high[i]) > cur_high:
                cur_high = float(high[i])
            if float(low[i]) < cur_low:
                cur_low = float(low[i])

        sess_open[i] = cur_open
        sess_high[i] = cur_high
        sess_low[i] = cur_low

    return sess_open, sess_high, sess_low


def _time_since_vol_spike(vol_energy: np.ndarray) -> np.ndarray:
    """
    Count M15 bars since the last volatility spike, capped at VOL_SPIKE_CAP.

    A spike is when ``l1_volatility_energy`` > rolling_mean + VOL_SPIKE_SIGMA
    * rolling_std over a VOL_SPIKE_WINDOW-bar past window.  Bars with
    insufficient history are assigned VOL_SPIKE_CAP (treat as long since spike).

    Parameters
    ----------
    vol_energy : np.ndarray
        l1_volatility_energy values, shape (n,).

    Returns
    -------
    np.ndarray, shape (n,), dtype float32.
    """
    n = len(vol_energy)
    result = np.full(n, VOL_SPIKE_CAP, dtype=np.float32)

    series = pd.Series(vol_energy)
    roll = series.rolling(VOL_SPIKE_WINDOW, min_periods=VOL_SPIKE_WINDOW)
    threshold = (roll.mean() + VOL_SPIKE_SIGMA * roll.std()).values

    last_spike_idx = -VOL_SPIKE_CAP  # pretend far-past spike

    for i in range(n):
        if not np.isnan(threshold[i]) and vol_energy[i] > threshold[i]:
            last_spike_idx = i
        result[i] = min(float(i - last_spike_idx), float(VOL_SPIKE_CAP))

    return result


def _session_vol_rank(vol_energy: np.ndarray, session: np.ndarray) -> np.ndarray:
    """
    Rolling percentile rank of ``l1_volatility_energy`` within the last
    SESSION_VOL_RANK_WINDOW same-session-type candles.

    Rank = fraction of historical same-session values ≤ current value.
    Range 0–1.  High value = unusually volatile for this session type.

    No lookahead: each bar only looks at past bars.

    Parameters
    ----------
    vol_energy : np.ndarray  shape (n,)
    session    : np.ndarray  shape (n,), int8

    Returns
    -------
    np.ndarray, shape (n,), dtype float32.
    """
    n = len(vol_energy)
    result = np.zeros(n, dtype=np.float32)
    # Per-session-type rolling history buffers (capped at SESSION_VOL_RANK_WINDOW)
    history: dict[int, deque] = {0: deque(), 1: deque(), 2: deque()}

    for i in range(n):
        sess = int(session[i])
        buf = history[sess]
        ve = float(vol_energy[i])

        if buf:
            count_le = sum(1 for v in buf if v <= ve)
            result[i] = count_le / len(buf)

        buf.append(ve)
        if len(buf) > SESSION_VOL_RANK_WINDOW:
            buf.popleft()

    return result


def _session_trend(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    session: np.ndarray,
) -> np.ndarray:
    """
    Linear regression slope of M15 close prices from the session start to
    the current bar, normalised by a rolling ATR.

    Uses an incremental (O(1) per bar) regression update to avoid O(n²)
    cost.  The slope is normalised by the rolling ``_ATR_WINDOW``-bar mean
    of (high − low) so that values are comparable across time.

    Returns 0.0 for the first bar of a session (only one point).

    Parameters
    ----------
    close, high, low : np.ndarray  shape (n,)
    session          : np.ndarray  shape (n,), int8

    Returns
    -------
    np.ndarray, shape (n,), dtype float32.
    """
    n = len(close)
    result = np.zeros(n, dtype=np.float32)

    # Rolling ATR: mean of (high − low) over _ATR_WINDOW bars
    hl_range = (high - low).astype(np.float64)
    atr = pd.Series(hl_range).rolling(_ATR_WINDOW, min_periods=1).mean().values

    cur_sess = int(session[0])
    # Incremental regression state
    # Using Welford-style running sums for y = close, x = 0,1,2,...
    sess_start = 0
    n_pts = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0

    for i in range(n):
        if i > 0 and int(session[i]) != cur_sess:
            # New session — reset
            cur_sess = int(session[i])
            sess_start = i
            n_pts = 0
            sum_x = sum_y = sum_xx = sum_xy = 0.0

        x = float(i - sess_start)  # position within session (0, 1, 2, …)
        y = float(close[i])

        n_pts += 1
        sum_x += x
        sum_y += y
        sum_xx += x * x
        sum_xy += x * y

        if n_pts < 2:
            result[i] = 0.0
            continue

        # OLS slope
        denom = n_pts * sum_xx - sum_x * sum_x
        if abs(denom) < EPSILON:
            result[i] = 0.0
            continue

        slope = (n_pts * sum_xy - sum_x * sum_y) / denom

        curr_atr = float(atr[i]) if float(atr[i]) > 0 else EPSILON
        result[i] = float(slope / curr_atr)

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------


def compute_layer2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Layer 2 (Time DNA) features for every M15 bar in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``compute_layer1_features`` — one row per M15 bar with
        a DatetimeIndex (UTC-naive) and all Layer 1 feature columns.

    Returns
    -------
    pd.DataFrame
        *df* with ten ``l2_*`` feature columns appended.
    """
    logger.info("Computing Layer 2 (Time DNA) features for %d M15 bars …", len(df))

    with Timer("layer2 features") as t:
        idx = df.index  # DatetimeIndex
        hour = idx.hour.values.astype(np.int32)
        minute = idx.minute.values.astype(np.int32)

        open_ = df[_M15_OPEN].values.astype(np.float64)
        high = df[_M15_HIGH].values.astype(np.float64)
        low = df[_M15_LOW].values.astype(np.float64)
        close = df[_M15_CLOSE].values.astype(np.float64)

        session = _session_labels(hour)

        # Session running stats (no lookahead)
        sess_open, sess_high, sess_low = _session_running_stats(
            open_, high, low, close, session
        )

        # l2_distance_from_session_open (pips)
        dist_from_open = (close - sess_open) / PIP_SIZE

        # l2_session_position (0–1)
        sess_range = sess_high - sess_low
        sess_pos = (close - sess_low) / (sess_range + EPSILON)
        sess_pos = np.clip(sess_pos, 0.0, 1.0)

        # Volatility energy column
        if COL_VOLATILITY_ENERGY in df.columns:
            vol_energy = df[COL_VOLATILITY_ENERGY].values.astype(np.float64)
        else:
            logger.warning(
                "Column '%s' not found; vol-related features set to 0.",
                COL_VOLATILITY_ENERGY,
            )
            vol_energy = np.zeros(len(df), dtype=np.float64)

        time_since_spike = _time_since_vol_spike(vol_energy)
        sess_vol_rank = _session_vol_rank(vol_energy, session)
        sess_trend = _session_trend(close, high, low, session)

    logger.info("Layer 2 features computed in %s", t.elapsed_str)

    features = pd.DataFrame(
        {
            "l2_session": session,
            "l2_session_overlap": _session_overlap(hour),
            "l2_kill_zone": _kill_zone(hour, minute),
            "l2_day_of_week": idx.dayofweek.values.astype(np.int8),
            "l2_hour": hour.astype(np.int8),
            "l2_distance_from_session_open": dist_from_open.astype(np.float32),
            "l2_session_position": sess_pos.astype(np.float32),
            "l2_time_since_vol_spike": time_since_spike,
            "l2_session_volatility_rank": sess_vol_rank,
            "l2_session_trend": sess_trend,
        },
        index=df.index,
    )
    return df.join(features)
