"""
GOLDWOLF — Layer 1 Microstructure Features
Computes 7 custom features from the M1 candles inside each M15 bar.
All functions operate on a list of M1 OHLC dicts as returned by the processor.

Output column names follow the ``l1_`` prefix convention defined in
config/settings.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    EPSILON,
    EARLY_SPLIT,
    ABSORPTION_BODY_RATIO_THRESHOLD,
    COL_CUSTOM_VOLUME,
    COL_VOLATILITY_ENERGY,
    COL_PRICE_VELOCITY,
    COL_REVERSAL_COUNT,
    COL_EARLY_LATE_RATIO,
    COL_PRICE_ACCELERATION,
    COL_ABSORPTION_COUNT,
    COL_ABSORPTION_INTENSITY,
    COL_M1_COUNT,
    CSV_OPEN_COL,
    CSV_HIGH_COL,
    CSV_LOW_COL,
    CSV_CLOSE_COL,
)
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

# Type alias for a single M1 candle represented as a dict
M1Candle = dict[str, float]


# ---------------------------------------------------------------------------
# Individual feature functions
# Each accepts a list of M1Candle dicts and returns a scalar (or tuple).
# ---------------------------------------------------------------------------


def custom_volume(m1_candles: list[M1Candle]) -> int:
    """
    Feature 1 — Custom Volume
    Count of M1 candles with actual price movement (high != low).
    Range: 0–15.
    """
    return sum(1 for c in m1_candles if c[CSV_HIGH_COL] != c[CSV_LOW_COL])


def volatility_energy(m1_candles: list[M1Candle]) -> float:
    """
    Feature 2 — Volatility Energy
    Sum of (high − low) for all M1 candles within the M15 period.
    Measures total intra-bar energy regardless of direction.
    """
    return float(sum(c[CSV_HIGH_COL] - c[CSV_LOW_COL] for c in m1_candles))


def price_velocity(
    m1_candles: list[M1Candle],
    m15_open: float,
    m15_close: float,
) -> float:
    """
    Feature 3 — Price Velocity
    (M15_close − M15_open) / number_of_active_M1_candles.
    Returns 0.0 when there are no active candles to avoid division by zero.
    """
    active = custom_volume(m1_candles)
    if active == 0:
        return 0.0
    return (m15_close - m15_open) / active


def intra_bar_reversal_count(m1_candles: list[M1Candle]) -> int:
    """
    Feature 4 — Intra-Bar Reversal Count
    Count how many times consecutive M1 bars flipped direction
    (up-close → down-close or vice versa).
    A candle is "up" when close > open, "down" when close < open.
    Doji candles (close == open) do not trigger a reversal.
    """
    if len(m1_candles) < 2:
        return 0

    reversals = 0
    prev_dir: int | None = None

    for c in m1_candles:
        diff = c[CSV_CLOSE_COL] - c[CSV_OPEN_COL]
        if diff > 0:
            curr_dir = 1
        elif diff < 0:
            curr_dir = -1
        else:
            # Doji — keep previous direction, no flip counted
            continue

        if prev_dir is not None and curr_dir != prev_dir:
            reversals += 1
        prev_dir = curr_dir

    return reversals


def early_late_ratio(m1_candles: list[M1Candle]) -> float:
    """
    Feature 5 — Early vs Late Volume Split
    early = first EARLY_SPLIT candles, late = remaining candles.
    ratio = late_energy / (early_energy + late_energy + ε)
    > 0.5  → energy building (continuation likely)
    < 0.5  → energy dying   (exhaustion/reversal likely)
    """
    early = m1_candles[:EARLY_SPLIT]
    late = m1_candles[EARLY_SPLIT:]
    e_energy = volatility_energy(early)
    l_energy = volatility_energy(late)
    return l_energy / (e_energy + l_energy + EPSILON)


def price_acceleration(m1_candles: list[M1Candle]) -> float:
    """
    Feature 6 — Price Acceleration
    velocity_first_half  = Δprice over first EARLY_SPLIT candles / EARLY_SPLIT
    velocity_second_half = Δprice over remaining candles / len(remaining)
    acceleration = velocity_second_half − velocity_first_half
    Positive → speeding up (momentum).  Negative → slowing down (exhaustion).
    """
    early = m1_candles[:EARLY_SPLIT]
    late = m1_candles[EARLY_SPLIT:]

    def _velocity(candles: list[M1Candle]) -> float:
        if not candles:
            return 0.0
        delta = candles[-1][CSV_CLOSE_COL] - candles[0][CSV_OPEN_COL]
        return delta / len(candles)

    return _velocity(late) - _velocity(early)


def absorption_detection(
    m1_candles: list[M1Candle],
) -> tuple[int, float]:
    """
    Feature 7 — Absorption Detection
    An M1 candle is an "absorption bar" when:
      - (high − low) is large (non-zero range)
      - body_ratio = |close − open| / (high − low + ε) < THRESHOLD
    This signals big range but price went nowhere — institutional absorption.

    Returns
    -------
    absorption_count : int
        Number of absorption bars in the M15 period.
    absorption_intensity : float
        Sum of ranges of absorption bars / total volatility energy.
        0.0 when total energy is zero.
    """
    total_energy = volatility_energy(m1_candles)
    absorption_range_sum = 0.0
    count = 0

    for c in m1_candles:
        bar_range = c[CSV_HIGH_COL] - c[CSV_LOW_COL]
        if bar_range == 0:
            continue  # dead candle — cannot be an absorption bar
        body_ratio = abs(c[CSV_CLOSE_COL] - c[CSV_OPEN_COL]) / (bar_range + EPSILON)
        if body_ratio < ABSORPTION_BODY_RATIO_THRESHOLD:
            count += 1
            absorption_range_sum += bar_range

    intensity = absorption_range_sum / (total_energy + EPSILON)
    return count, intensity


# ---------------------------------------------------------------------------
# Batch computation — apply all features to the grouped DataFrame
# ---------------------------------------------------------------------------


def compute_layer1_features(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Layer 1 features for every M15 bar in *grouped_df*.

    Parameters
    ----------
    grouped_df : pd.DataFrame
        Output of ``processor.group_m1_by_m15`` — one row per M15 bar with
        an ``m1_candles`` column.

    Returns
    -------
    pd.DataFrame
        *grouped_df* with all Layer 1 feature columns appended.  The
        ``m1_candles`` column is dropped from the output to keep the CSV/
        parquet lightweight.
    """
    logger.info("Computing Layer 1 features for %d M15 bars …", len(grouped_df))

    with Timer("layer1 features") as t:
        results: list[dict[str, Any]] = []
        for idx, row in grouped_df.iterrows():
            candles: list[M1Candle] = row["m1_candles"]
            m15_open: float = row.get("m15_open", float("nan"))
            m15_close: float = row.get("m15_close", float("nan"))

            cv = custom_volume(candles)
            ve = volatility_energy(candles)
            pv = price_velocity(candles, m15_open, m15_close)
            rc = intra_bar_reversal_count(candles)
            elr = early_late_ratio(candles)
            pa = price_acceleration(candles)
            ab_count, ab_intensity = absorption_detection(candles)

            results.append(
                {
                    COL_CUSTOM_VOLUME: cv,
                    COL_VOLATILITY_ENERGY: ve,
                    COL_PRICE_VELOCITY: pv,
                    COL_REVERSAL_COUNT: rc,
                    COL_EARLY_LATE_RATIO: elr,
                    COL_PRICE_ACCELERATION: pa,
                    COL_ABSORPTION_COUNT: ab_count,
                    COL_ABSORPTION_INTENSITY: ab_intensity,
                }
            )

    logger.info("Layer 1 features computed in %s", t.elapsed_str)

    features_df = pd.DataFrame(results, index=grouped_df.index)

    # Drop raw M1 candle lists — not needed in output
    output = grouped_df.drop(columns=["m1_candles"]).join(features_df)
    return output
