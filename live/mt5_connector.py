"""
GOLDWOLF — MT5 Connector
Connects to MetaTrader 5 and fetches live price data.

Credentials are loaded from .env (never hardcoded):
  MT5_LOGIN    = account number
  MT5_PASSWORD = password
  MT5_SERVER   = broker server
  MT5_PATH     = terminal executable path (optional)

Falls back to offline/backtest mode if MT5 is not installed or unreachable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH, MT5_SYMBOL
from utils.helpers import get_logger

logger = get_logger(__name__)

# Try to import MetaTrader5 — optional dependency (Windows only)
try:
    import MetaTrader5 as _mt5
    _MT5_AVAILABLE = True
except ImportError:
    _mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False
    logger.warning(
        "MetaTrader5 library not installed — running in offline/backtest mode."
    )

_MAX_RETRIES = 3


def _mt5_timeframe(minutes: int) -> Any:
    """Convert minutes to MT5 TIMEFRAME constant."""
    if not _MT5_AVAILABLE:
        return None
    mapping = {
        1: _mt5.TIMEFRAME_M1,
        5: _mt5.TIMEFRAME_M5,
        15: _mt5.TIMEFRAME_M15,
        30: _mt5.TIMEFRAME_M30,
        60: _mt5.TIMEFRAME_H1,
        240: _mt5.TIMEFRAME_H4,
        1440: _mt5.TIMEFRAME_D1,
    }
    return mapping.get(minutes)


def connect() -> bool:
    """
    Initialize and login to MetaTrader 5.

    Returns True on success, False on failure (offline mode).
    """
    if not _MT5_AVAILABLE:
        logger.warning("MT5 not available — cannot connect.")
        return False

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            kwargs: dict[str, Any] = {}
            if MT5_PATH:
                kwargs["path"] = MT5_PATH
            if not _mt5.initialize(**kwargs):
                logger.warning(
                    "MT5 initialize failed (attempt %d/%d): %s",
                    attempt, _MAX_RETRIES, _mt5.last_error(),
                )
                continue

            login_kwargs: dict[str, Any] = {}
            if MT5_LOGIN:
                login_kwargs["login"] = int(MT5_LOGIN)
            if MT5_PASSWORD:
                login_kwargs["password"] = MT5_PASSWORD
            if MT5_SERVER:
                login_kwargs["server"] = MT5_SERVER

            if login_kwargs and not _mt5.login(**login_kwargs):
                logger.warning(
                    "MT5 login failed (attempt %d/%d): %s",
                    attempt, _MAX_RETRIES, _mt5.last_error(),
                )
                _mt5.shutdown()
                continue

            logger.info("MT5 connected successfully.")
            return True

        except Exception as exc:
            logger.warning("MT5 connection error (attempt %d): %s", attempt, exc)

    logger.error("MT5 connection failed after %d attempts — offline mode.", _MAX_RETRIES)
    return False


def disconnect() -> None:
    """Shutdown the MT5 connection."""
    if _MT5_AVAILABLE:
        try:
            _mt5.shutdown()
            logger.info("MT5 disconnected.")
        except Exception as exc:
            logger.warning("MT5 disconnect error: %s", exc)


def get_latest_m1_candles(symbol: str = MT5_SYMBOL, count: int = 15) -> pd.DataFrame | None:
    """
    Fetch the latest *count* M1 candles from MT5.

    Returns a DataFrame with columns: datetime, open, high, low, close, volume.
    Returns None if MT5 is not connected.
    """
    return _get_candles(symbol, 1, count)


def get_latest_m15_candles(symbol: str = MT5_SYMBOL, count: int = 100) -> pd.DataFrame | None:
    """
    Fetch the latest *count* M15 candles from MT5.

    Returns a DataFrame indexed by datetime with OHLCV columns.
    Returns None if MT5 is not connected.
    """
    return _get_candles(symbol, 15, count)


def get_current_price(symbol: str = MT5_SYMBOL) -> dict[str, float] | None:
    """
    Get current bid/ask for *symbol*.

    Returns {'bid': float, 'ask': float} or None.
    """
    if not _MT5_AVAILABLE:
        return None
    try:
        tick = _mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {"bid": float(tick.bid), "ask": float(tick.ask)}
    except Exception as exc:
        logger.warning("Failed to get current price: %s", exc)
        return None


def _get_candles(
    symbol: str,
    timeframe_minutes: int,
    count: int,
) -> pd.DataFrame | None:
    """Internal: fetch candles from MT5."""
    if not _MT5_AVAILABLE:
        return None

    tf = _mt5_timeframe(timeframe_minutes)
    if tf is None:
        logger.warning("Unsupported timeframe: %d minutes", timeframe_minutes)
        return None

    try:
        rates = _mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning("No data returned for %s M%d", symbol, timeframe_minutes)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(
            columns={
                "time": "datetime",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "volume",
            }
        )
        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        df = df.set_index("datetime")
        return df

    except Exception as exc:
        logger.warning("Failed to fetch %s M%d candles: %s", symbol, timeframe_minutes, exc)
        return None
