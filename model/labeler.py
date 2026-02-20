"""
GOLDWOLF — Model Labeler
Creates forward-looking trade labels for each M15 candle.

Label Creation (NO lookahead in features — labels look forward intentionally):
  For each M15 candle, scan future candles until TP or SL is hit.
    - Label  1 (BUY)      : price goes up and hits +TP pips before -SL pips
    - Label -1 (SELL)     : price goes down and hits -TP pips before +SL pips
    - Label  0 (NO_TRADE) : neither TP nor SL hit within max_horizon candles

Pip size for XAUUSD: 1 pip = 0.1 (so 150 pips = $15 move).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    LABEL_TP_PIPS,
    LABEL_SL_PIPS,
    LABEL_MAX_HORIZON,
    PIP_SIZE,
)
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

# Internal column aliases
_M15_HIGH = "m15_high"
_M15_LOW = "m15_low"
_M15_CLOSE = "m15_close"


def create_labels(
    df: pd.DataFrame,
    tp_pips: int = LABEL_TP_PIPS,
    sl_pips: int = LABEL_SL_PIPS,
    max_horizon: int = LABEL_MAX_HORIZON,
) -> pd.Series:
    """
    Create forward-looking trade labels for each M15 bar.

    For each bar at index *i*, scan bars i+1 … i+max_horizon:
      - If high[j] >= entry + tp_price before low[j] <= entry - sl_price → Label 1 (BUY)
      - If low[j]  <= entry - sl_price before high[j] >= entry + tp_price → Label -1 (SELL)
      - If neither hit within max_horizon → Label 0 (NO_TRADE)

    Entry price = close[i] (next bar entry approximation).

    Parameters
    ----------
    df          : pd.DataFrame with m15_close, m15_high, m15_low columns.
    tp_pips     : Take-profit distance in pips.
    sl_pips     : Stop-loss distance in pips.
    max_horizon : Maximum candles to scan forward.

    Returns
    -------
    pd.Series of int8 labels (-1, 0, 1) indexed like *df*.
    """
    logger.info(
        "Creating labels: TP=%d pips, SL=%d pips, horizon=%d bars …",
        tp_pips, sl_pips, max_horizon,
    )

    with Timer("label creation") as t:
        tp_price = tp_pips * PIP_SIZE
        sl_price = sl_pips * PIP_SIZE

        close = df[_M15_CLOSE].values.astype(np.float64)
        high = df[_M15_HIGH].values.astype(np.float64)
        low = df[_M15_LOW].values.astype(np.float64)
        n = len(close)

        labels = np.zeros(n, dtype=np.int8)

        for i in range(n):
            entry = close[i]
            tp_level = entry + tp_price
            sl_level = entry - sl_price

            label = 0
            for j in range(i + 1, min(i + max_horizon + 1, n)):
                hit_tp = high[j] >= tp_level
                hit_sl = low[j] <= sl_level

                if hit_tp and hit_sl:
                    # Both hit on same candle: check which direction is more likely
                    # by comparing distance from entry to each
                    dist_to_tp = tp_level - entry
                    dist_to_sl = entry - sl_level
                    if dist_to_tp <= dist_to_sl:
                        label = 1
                    else:
                        label = -1
                    break
                elif hit_tp:
                    label = 1
                    break
                elif hit_sl:
                    label = -1
                    break

            labels[i] = label

    logger.info("Label creation completed in %s", t.elapsed_str)

    # Log class distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    total = n
    logger.info(
        "Label distribution — BUY: %d (%.1f%%), SELL: %d (%.1f%%), NO_TRADE: %d (%.1f%%)",
        dist.get(1, 0), dist.get(1, 0) / total * 100,
        dist.get(-1, 0), dist.get(-1, 0) / total * 100,
        dist.get(0, 0), dist.get(0, 0) / total * 100,
    )

    return pd.Series(labels, index=df.index, name="label", dtype=np.int8)
