"""
GOLDWOLF — Live Trading Runner
Main live loop that runs continuously, generating signals on each M15 close.

Usage:
  python main.py --live

Loop:
  1. Connect to MT5
  2. Fill any data gaps
  3. Every 15 minutes (on M15 candle close):
     a. Fetch latest 15 M1 candles
     b. Compute all features (L1 → L4)
     c. Run model prediction
     d. If signal passes filters → send to Telegram
     e. Log signal to output/signal_log.csv
  4. Sleep until next M15 close
"""

from __future__ import annotations

import csv
import datetime
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import SIGNAL_LOG_PATH, MT5_SYMBOL
from utils.helpers import get_logger

logger = get_logger(__name__)

# Flag to stop the loop gracefully on Ctrl+C
_RUNNING = True


def _handle_sigint(signum: int, frame: Any) -> None:
    global _RUNNING
    logger.info("Shutdown signal received — stopping after current cycle.")
    _RUNNING = False


def _seconds_until_next_m15() -> float:
    """Return seconds until the next M15 candle close."""
    now = datetime.datetime.utcnow()
    minutes = now.minute
    seconds = now.second
    microseconds = now.microsecond

    # Next M15 boundary
    next_boundary = ((minutes // 15) + 1) * 15
    if next_boundary >= 60:
        next_boundary = 0
        delta_mins = 60 - minutes - 1
    else:
        delta_mins = next_boundary - minutes - 1

    remaining = (
        delta_mins * 60
        + (60 - seconds)
        - microseconds / 1_000_000
        + 5  # small buffer to ensure candle is closed
    )
    return max(remaining, 0.0)


def _log_signal_to_csv(signal: dict[str, Any]) -> None:
    """Append a signal to the signal log CSV file."""
    log_path = Path(SIGNAL_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp", "direction", "confidence", "tier",
        "entry_price", "stop_loss", "take_profit", "risk_reward",
        "candle_dna", "top_reasons",
    ]

    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {k: signal.get(k, "") for k in fieldnames}
        row["top_reasons"] = " | ".join(signal.get("top_reasons", []))
        writer.writerow(row)


def _build_feature_row(
    m15_bar: pd.DataFrame,
    historical_context: pd.DataFrame,
) -> pd.Series | None:
    """
    Compute all features (L1 → L4) for a single live M15 bar.

    Parameters
    ----------
    m15_bar           : Single-row DataFrame with m1_candles column.
    historical_context: Historical feature DataFrame for rolling computations.

    Returns
    -------
    pd.Series — one row of all features, or None on error.
    """
    try:
        from features.layer1 import compute_layer1_features
        from features.layer2 import compute_layer2_features
        from features.layer3 import compute_layer3_features
        from features.layer4 import compute_layer4_features

        # We need rolling context — append new bar to tail of historical data
        # For live, use the last N bars of history + the new bar
        context_tail = historical_context.tail(500).copy()

        # Drop existing feature columns to avoid overlap on recomputation
        l2_cols = [c for c in context_tail.columns if c.startswith("l2_")]
        l3_cols = [c for c in context_tail.columns if c.startswith("l3_")]
        l4_cols = [c for c in context_tail.columns if c.startswith("l4_")]
        context_tail = context_tail.drop(columns=l2_cols + l3_cols + l4_cols, errors="ignore")

        # Compute features on the full context window
        out = compute_layer1_features(context_tail) if "m1_candles" in context_tail.columns else context_tail
        out = compute_layer2_features(out)
        out = compute_layer3_features(out)
        out = compute_layer4_features(out)

        # Return the last row (the live bar)
        return out.iloc[-1]

    except Exception as exc:
        logger.error("Feature computation failed: %s", exc)
        return None


def run_live(
    symbol: str = MT5_SYMBOL,
    offline_mode: bool = False,
) -> None:
    """
    Main live trading loop.

    Parameters
    ----------
    symbol       : MT5 symbol to trade.
    offline_mode : If True, skip MT5 connection (for testing).
    """
    global _RUNNING

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    logger.info("=" * 60)
    logger.info("GOLDWOLF — Live Signal Runner")
    logger.info("Symbol: %s | Mode: %s", symbol, "OFFLINE" if offline_mode else "LIVE")
    logger.info("=" * 60)

    from signals.generator import SignalGenerator
    from live.telegram_bot import send_signal, send_alert
    from live.dashboard import GoldwolfDashboard
    from rich.live import Live

    gen = SignalGenerator()
    dashboard = GoldwolfDashboard()

    # --- Connect to MT5 ---
    connected = False
    if not offline_mode:
        from live import mt5_connector
        connected = mt5_connector.connect()
        if not connected:
            send_alert("MT5 connection failed — running in offline mode.")

    # --- Load historical context ---
    historical_df: pd.DataFrame | None = None
    try:
        from config.settings import OUTPUT_PATH_PHASE2_3
        p = Path(OUTPUT_PATH_PHASE2_3)
        if p.exists():
            historical_df = pd.read_parquet(p)
            logger.info("Loaded historical context: %d rows", len(historical_df))
    except Exception as exc:
        logger.warning("Could not load historical context: %s", exc)

    # --- Fill data gap if connected ---
    if connected and historical_df is not None:
        from live.data_bridge import fill_gap
        historical_df = fill_gap(historical_df, symbol)

    # --- Main loop ---
    cycle = 0
    with Live(dashboard.render(), refresh_per_second=1, screen=True) as live:
        while _RUNNING:
            cycle += 1

            try:
                # Fetch latest M15 bar
                if connected:
                    from live.data_bridge import get_live_m15_bar
                    live_bar = get_live_m15_bar(symbol)
                else:
                    live_bar = None

                if live_bar is None:
                    if historical_df is not None and len(historical_df) > 0:
                        # Offline: use the last bar of historical data for demo
                        live_bar = historical_df.tail(1)
                    else:
                        logger.warning("No data available — skipping cycle.")
                        time.sleep(60)
                        continue

                # Compute features
                if historical_df is not None:
                    feature_row = _build_feature_row(live_bar, historical_df)
                else:
                    feature_row = live_bar.iloc[-1] if len(live_bar) > 0 else None

                if feature_row is None:
                    logger.warning("Feature computation returned None — skipping.")
                    time.sleep(60)
                    continue

                # Update market panel
                dashboard.update_market(feature_row)
                live.update(dashboard.render())

                # Get raw prediction for dashboard (before filters)
                try:
                    raw = gen.predict_raw(feature_row)
                    raw_confidence = raw["confidence"]
                    raw_direction = raw["direction"]
                except Exception as raw_exc:
                    logger.debug("predict_raw failed: %s", raw_exc)
                    raw_confidence = 0.0
                    raw_direction = "NO_TRADE"

                # Generate filtered signal
                sig = gen.generate_signal(feature_row, bar_index=cycle)

                # Update prediction panel
                dashboard.update_prediction(sig, raw_confidence, raw_direction)
                live.update(dashboard.render())

                if sig is not None:
                    logger.info("Signal generated: %s", sig)
                    _log_signal_to_csv(sig)
                    send_signal(sig)
                    dashboard.add_signal(sig)
                else:
                    logger.debug("No signal this cycle.")

            except Exception as exc:
                logger.error("Error in live loop cycle %d: %s", cycle, exc, exc_info=True)
                try:
                    send_alert(f"Error in live loop: {exc}")
                except Exception:
                    pass

            if not _RUNNING:
                break

            # Sleep until next M15 close, updating the countdown every second
            deadline = time.monotonic() + _seconds_until_next_m15()
            while _RUNNING:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                dashboard.update_countdown(remaining)
                live.update(dashboard.render())
                time.sleep(min(1.0, remaining))

    # --- Cleanup ---
    if connected:
        from live import mt5_connector
        mt5_connector.disconnect()
    logger.info("Live runner stopped.")
