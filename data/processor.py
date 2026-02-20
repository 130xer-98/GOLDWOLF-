"""
GOLDWOLF — M1-inside-M15 Grouping Engine
Core processor: aligns each M1 candle to its parent M15 window and
returns merged data ready for feature extraction.
"""

import numpy as np
import pandas as pd

from config.settings import (
    CSV_OPEN_COL,
    CSV_HIGH_COL,
    CSV_LOW_COL,
    CSV_CLOSE_COL,
    CSV_VOLUME_COL,
    M15_PERIOD_MINUTES,
    COL_M1_COUNT,
)
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)


def assign_m15_bucket(m1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``m15_datetime`` column to a M1 DataFrame that indicates the
    start timestamp of the parent M15 bar for each M1 row.

    An M15 bar starting at HH:MM covers M1 bars from HH:MM to HH:MM+14
    (i.e. the M15 bar timestamp is floor(M1_time, 15 minutes)).

    Parameters
    ----------
    m1_df : pd.DataFrame
        M1 data with a DatetimeIndex, as produced by ``load_csv``.

    Returns
    -------
    pd.DataFrame
        Copy of *m1_df* with an additional ``m15_datetime`` column
        (dtype: datetime64[ns]).
    """
    df = m1_df.copy()
    # Floor each M1 timestamp to the nearest 15-minute boundary
    df["m15_datetime"] = df.index.floor(f"{M15_PERIOD_MINUTES}min")
    return df


def group_m1_by_m15(
    m1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every M15 bar present in *m15_df*, collect its corresponding M1
    candles from *m1_df* and return a merged DataFrame.

    The output has one row per M15 bar.  Each row contains:
    - All M15 OHLCV columns (prefixed with ``m15_``).
    - An ``m1_candles`` column containing a list of per-M1 dicts with keys
      ``open``, ``high``, ``low``, ``close``.
    - ``l1_m1_count`` — how many M1 candles were found in this M15 window.

    Parameters
    ----------
    m1_df : pd.DataFrame
        Clean M1 data (DatetimeIndex), output of ``load_csv``.
    m15_df : pd.DataFrame
        Clean M15 data (DatetimeIndex), output of ``load_csv``.

    Returns
    -------
    pd.DataFrame
        One row per M15 bar that has at least one matching M1 candle,
        indexed by the M15 bar's datetime.
    """
    logger.info(
        "Grouping %d M1 rows into M15 buckets (%d M15 bars) …",
        len(m1_df),
        len(m15_df),
    )

    with Timer("assign_m15_bucket") as t:
        m1_bucketed = assign_m15_bucket(m1_df)

    logger.info("Bucket assignment done in %s", t.elapsed_str)

    with Timer("groupby + agg") as t:
        # Build a lightweight representation of each M1 candle (avoid heavy
        # object storage — store as structured numpy arrays per group).
        m1_bucketed["_idx"] = np.arange(len(m1_bucketed))

        # Group m1 by its parent M15 timestamp
        grouped = m1_bucketed.groupby("m15_datetime", sort=True)

        # Collect arrays for each M15 bucket
        records = []
        for m15_ts, grp in grouped:
            m1_candles = grp[[CSV_OPEN_COL, CSV_HIGH_COL, CSV_LOW_COL, CSV_CLOSE_COL]].to_dict("records")
            records.append(
                {
                    "m15_datetime": m15_ts,
                    "m1_candles": m1_candles,
                    COL_M1_COUNT: len(m1_candles),
                }
            )

    logger.info("Groupby done in %s  (%d groups)", t.elapsed_str, len(records))

    # Build result DataFrame
    result = pd.DataFrame(records).set_index("m15_datetime")

    # Merge M15 OHLCV data onto result
    m15_renamed = m15_df.copy()
    m15_renamed.index.name = "m15_datetime"
    m15_renamed.columns = [f"m15_{c}" for c in m15_renamed.columns]

    result = result.join(m15_renamed, how="inner")

    logger.info(
        "Final grouped DataFrame: %d M15 bars with M1 data.",
        len(result),
    )

    # Report how many M15 bars had fewer than 15 M1 candles
    incomplete = (result[COL_M1_COUNT] < 15).sum()
    logger.info(
        "M15 bars with < 15 M1 candles: %d  (%.1f%%)",
        incomplete,
        100 * incomplete / len(result) if len(result) else 0,
    )

    return result
