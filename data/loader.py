"""
GOLDWOLF — Data Loader
Reads M1 and M15 CSV files, validates them, and returns clean DataFrames
ready for the processor.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    CSV_DATE_COL,
    CSV_TIME_COL,
    CSV_DATETIME_COL,
    CSV_DATE_FORMAT,
    CSV_OPEN_COL,
    CSV_HIGH_COL,
    CSV_LOW_COL,
    CSV_CLOSE_COL,
    CSV_VOLUME_COL,
)
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

# Price columns that must be positive
PRICE_COLS = [CSV_OPEN_COL, CSV_HIGH_COL, CSV_LOW_COL, CSV_CLOSE_COL]


def load_csv(filepath: str, timeframe_label: str = "data") -> pd.DataFrame:
    """
    Load a GOLDWOLF OHLCV CSV file and return a clean, validated DataFrame.

    Processing steps
    ----------------
    1. Read CSV with explicit dtypes for memory efficiency.
    2. Parse date + time columns into a single ``datetime`` column.
    3. Validate: nulls, positive prices, sorted dates.
    4. Filter dead candles (open == high == low == close).
    5. Filter weekend rows.
    6. Log statistics.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.
    timeframe_label : str
        Human-readable label used in log messages (e.g. "M1", "M15").

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a ``datetime`` index (UTC-naive) sorted
        ascending.  Original ``date`` and ``time`` columns are dropped.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"[{timeframe_label}] CSV file not found: {filepath}"
        )

    logger.info("[%s] Loading: %s", timeframe_label, filepath)

    with Timer(f"read_csv {timeframe_label}") as t:
        # Read with explicit dtypes to reduce memory usage
        df = pd.read_csv(
            filepath,
            dtype={
                CSV_DATE_COL: str,
                CSV_TIME_COL: str,
                CSV_OPEN_COL: np.float32,
                CSV_HIGH_COL: np.float32,
                CSV_LOW_COL: np.float32,
                CSV_CLOSE_COL: np.float32,
                CSV_VOLUME_COL: np.float32,
            },
            engine="c",
        )

    logger.info(
        "[%s] Raw rows: %d  (read in %s)",
        timeframe_label,
        len(df),
        t.elapsed_str,
    )

    rows_raw = len(df)

    # ------------------------------------------------------------------
    # 1. Parse datetime
    # ------------------------------------------------------------------
    df[CSV_DATETIME_COL] = pd.to_datetime(
        df[CSV_DATE_COL].str.strip() + " " + df[CSV_TIME_COL].str.strip(),
        format=CSV_DATE_FORMAT,
    )
    df.drop(columns=[CSV_DATE_COL, CSV_TIME_COL], inplace=True)
    df.set_index(CSV_DATETIME_COL, inplace=True)

    # ------------------------------------------------------------------
    # 2. Validate: nulls
    # ------------------------------------------------------------------
    null_counts = df[PRICE_COLS].isnull().sum()
    if null_counts.any():
        logger.warning(
            "[%s] Null values found:\n%s", timeframe_label, null_counts[null_counts > 0]
        )
        df.dropna(subset=PRICE_COLS, inplace=True)
        logger.info("[%s] Rows after null drop: %d", timeframe_label, len(df))

    # ------------------------------------------------------------------
    # 3. Validate: positive prices
    # ------------------------------------------------------------------
    neg_mask = (df[PRICE_COLS] <= 0).any(axis=1)
    if neg_mask.any():
        logger.warning(
            "[%s] %d rows with non-positive prices removed.",
            timeframe_label,
            neg_mask.sum(),
        )
        df = df[~neg_mask]

    # ------------------------------------------------------------------
    # 4. Validate: sorted ascending
    # ------------------------------------------------------------------
    df.sort_index(inplace=True)

    # ------------------------------------------------------------------
    # 5. Filter dead candles (open == high == low == close)
    # ------------------------------------------------------------------
    dead_mask = (
        (df[CSV_OPEN_COL] == df[CSV_HIGH_COL])
        & (df[CSV_HIGH_COL] == df[CSV_LOW_COL])
        & (df[CSV_LOW_COL] == df[CSV_CLOSE_COL])
    )
    rows_before_dead = len(df)
    df = df[~dead_mask]
    rows_after_dead = len(df)
    logger.info(
        "[%s] Dead candles removed: %d  (remaining: %d)",
        timeframe_label,
        rows_before_dead - rows_after_dead,
        rows_after_dead,
    )

    # ------------------------------------------------------------------
    # 6. Filter weekends (Saturday = 5, Sunday = 6)
    # ------------------------------------------------------------------
    rows_before_weekend = len(df)
    df = df[df.index.dayofweek < 5]
    rows_after_weekend = len(df)
    logger.info(
        "[%s] Weekend rows removed: %d  (remaining: %d)",
        timeframe_label,
        rows_before_weekend - rows_after_weekend,
        rows_after_weekend,
    )

    # ------------------------------------------------------------------
    # 7. Log summary statistics
    # ------------------------------------------------------------------
    rows_filtered = rows_raw - len(df)
    logger.info(
        "[%s] Final rows: %d  |  filtered: %d  |  date range: %s → %s",
        timeframe_label,
        len(df),
        rows_filtered,
        df.index.min(),
        df.index.max(),
    )

    # Detect obvious time gaps (more than 2× the expected period)
    _log_gaps(df, timeframe_label)

    return df


def _log_gaps(df: pd.DataFrame, label: str) -> None:
    """Log any large time gaps in the data (for diagnostic purposes)."""
    if len(df) < 2:
        return
    deltas = df.index.to_series().diff().dropna()
    median_delta = deltas.median()
    # Flag gaps larger than 10× the median step
    gap_threshold = median_delta * 10
    gaps = deltas[deltas > gap_threshold]
    if not gaps.empty:
        logger.info(
            "[%s] Detected %d large time gaps (> %s).  Largest: %s at %s",
            label,
            len(gaps),
            gap_threshold,
            gaps.max(),
            gaps.idxmax(),
        )
    else:
        logger.info("[%s] No large time gaps detected.", label)
