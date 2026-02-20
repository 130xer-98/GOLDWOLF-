"""
GOLDWOLF — Data Bridge
Fills the gap between historical CSV data and live MT5 data.

On first run:
  - Historical CSV ends at some past date
  - Fetch M1 data from MT5 from that date to now
  - Merge with historical data
  - Save to LIVE_CACHE_PATH for persistence

On subsequent runs:
  - Only fetch from last cached timestamp to now
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import LIVE_CACHE_PATH, MT5_SYMBOL, PIP_SIZE
from utils.helpers import get_logger

logger = get_logger(__name__)

# M15 grouping window in minutes
_M15_MINUTES = 15


def fill_gap(
    historical_df: pd.DataFrame,
    symbol: str = MT5_SYMBOL,
) -> pd.DataFrame:
    """
    Fetch M1 data from MT5 to fill the gap between the end of *historical_df*
    and the current time.  Merge and return the combined DataFrame.

    Parameters
    ----------
    historical_df : pd.DataFrame — existing historical M1 data.
    symbol        : MT5 symbol name.

    Returns
    -------
    pd.DataFrame — historical data extended with live data (if available).
    """
    from live.mt5_connector import get_latest_m1_candles

    cache_path = Path(LIVE_CACHE_PATH)

    # Determine the last known timestamp
    if historical_df is not None and len(historical_df) > 0:
        last_ts = historical_df.index.max()
    else:
        last_ts = pd.Timestamp("2000-01-01")

    logger.info("Data bridge: last historical timestamp = %s", last_ts)

    # Check cache
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(cached) > 0:
                cached_last = cached.index.max()
                if cached_last > last_ts:
                    logger.info("Using cached data up to %s", cached_last)
                    historical_df = pd.concat([historical_df, cached]).drop_duplicates()
                    last_ts = cached_last
        except Exception as exc:
            logger.warning("Failed to read live cache: %s", exc)

    # Fetch latest data from MT5
    now = pd.Timestamp.utcnow().replace(tzinfo=None)
    if last_ts >= now - pd.Timedelta(minutes=1):
        logger.info("Data is already up to date.")
        return historical_df

    # Fetch enough recent M1 bars to cover the gap
    gap_minutes = int((now - last_ts).total_seconds() / 60) + 100
    count = min(gap_minutes, 50000)  # MT5 limit

    live_df = get_latest_m1_candles(symbol, count=count)
    if live_df is None or len(live_df) == 0:
        logger.warning("MT5 returned no data — returning historical only.")
        return historical_df

    # Standardise column names
    live_df = live_df.rename(
        columns={c: c.replace("open", "open").replace("volume", "volume") for c in live_df.columns}
    )

    # Merge
    combined = pd.concat([historical_df, live_df])
    # Deduplicate by index (keep last, i.e. most recent data wins)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        live_df.to_csv(cache_path)
        logger.info("Live cache updated: %d rows saved to %s", len(live_df), cache_path)
    except Exception as exc:
        logger.warning("Failed to save live cache: %s", exc)

    logger.info(
        "Gap filled: %d live bars added (total: %d)",
        len(live_df), len(combined),
    )
    return combined


def get_live_m15_bar(symbol: str = MT5_SYMBOL) -> pd.DataFrame | None:
    """
    Get the latest complete M15 bar with its M1 sub-candles.

    Returns a single-row DataFrame ready for feature computation,
    or None if MT5 is not available.
    """
    from live.mt5_connector import get_latest_m15_candles, get_latest_m1_candles

    m15_df = get_latest_m15_candles(symbol, count=2)
    m1_df = get_latest_m1_candles(symbol, count=_M15_MINUTES + 5)

    if m15_df is None or m1_df is None:
        return None

    # Return the second-to-last M15 bar (the most recently completed one)
    if len(m15_df) < 2:
        return None

    completed_bar = m15_df.iloc[-2]
    bar_ts = m15_df.index[-2]

    # Get the M1 candles that belong to this M15 bar
    bar_end = bar_ts + pd.Timedelta(minutes=_M15_MINUTES)
    m1_sub = m1_df.loc[bar_ts:bar_end].head(_M15_MINUTES)

    result = pd.DataFrame(
        {
            "m15_open": [float(completed_bar["open"])],
            "m15_high": [float(completed_bar["high"])],
            "m15_low": [float(completed_bar["low"])],
            "m15_close": [float(completed_bar["close"])],
            "m15_volume": [float(completed_bar.get("volume", 0))],
            "m1_candles": [[
                {
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                }
                for _, r in m1_sub.iterrows()
            ]],
        },
        index=[bar_ts],
    )
    return result
