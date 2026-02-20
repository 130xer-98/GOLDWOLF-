"""
GOLDWOLF — Live Trading Runner
Main live loop that runs continuously, generating signals on each M15 close.

Usage:
  python main.py --live

Architecture:
  1. QApplication + GoldwolfGUI run on the main thread.
  2. TradingWorker (QThread) handles MT5, feature computation, and prediction.
  3. Worker emits Qt signals; main thread slots update the GUI.
  4. QTimer drives 1-second countdown updates without blocking the GUI.
"""

from __future__ import annotations

import csv
import datetime
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import SIGNAL_LOG_PATH, MT5_SYMBOL
from utils.helpers import get_logger

logger = get_logger(__name__)

# Kept for backward compatibility with tests that reference runner._RUNNING
_RUNNING = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seconds_until_next_m15() -> float:
    """Return seconds until the next M15 candle close."""
    now = datetime.datetime.utcnow()
    minutes = now.minute
    seconds = now.second
    microseconds = now.microsecond

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
        context_tail = historical_context.tail(500).copy()

        # Drop existing feature columns to avoid overlap on recomputation
        l2_cols = [c for c in context_tail.columns if c.startswith("l2_")]
        l3_cols = [c for c in context_tail.columns if c.startswith("l3_")]
        l4_cols = [c for c in context_tail.columns if c.startswith("l4_")]
        context_tail = context_tail.drop(columns=l2_cols + l3_cols + l4_cols, errors="ignore")

        out = (
            compute_layer1_features(context_tail)
            if "m1_candles" in context_tail.columns
            else context_tail
        )
        out = compute_layer2_features(out)
        out = compute_layer3_features(out)
        out = compute_layer4_features(out)

        return out.iloc[-1]

    except Exception as exc:
        logger.error("Feature computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Worker thread factory — deferred so PySide6 is not imported at module level
# ---------------------------------------------------------------------------

def _make_trading_worker_class() -> type:
    """Return TradingWorker class (lazy import keeps PySide6 out of module scope)."""
    from PySide6.QtCore import QObject, QThread, Signal  # noqa: PLC0415

    class TradingWorker(QThread):
        """
        Background thread: connects MT5, computes features, generates signals.
        Emits Qt signals so the GUI (main thread) can update safely.
        """

        market_updated = Signal(object)                    # pd.Series feature_row
        prediction_ready = Signal(object, float, str)      # signal | None, confidence, dir
        signal_fired = Signal(object)                      # signal dict
        cycle_done = Signal(int)                           # cycle number
        mt5_status = Signal(bool)                          # connected?
        chart_ready = Signal(object, object)               # historical_df, signal | None

        def __init__(
            self,
            symbol: str = MT5_SYMBOL,
            offline_mode: bool = False,
            parent: "QObject | None" = None,
        ) -> None:
            super().__init__(parent)
            self._symbol = symbol
            self._offline_mode = offline_mode
            self._running = True

        def stop(self) -> None:
            self._running = False

        def run(self) -> None:  # noqa: C901
            from signals.generator import SignalGenerator
            from live.telegram_bot import send_signal, send_alert

            gen = SignalGenerator()

            # Connect MT5
            connected = False
            if not self._offline_mode:
                try:
                    from live import mt5_connector
                    connected = mt5_connector.connect()
                except Exception as exc:
                    logger.warning("MT5 connection error: %s", exc)
                if not connected:
                    try:
                        send_alert("MT5 connection failed — running in offline mode.")
                    except Exception:
                        pass
            self.mt5_status.emit(connected)

            # Load historical context
            historical_df: pd.DataFrame | None = None
            try:
                from config.settings import OUTPUT_PATH_PHASE2_3
                p = Path(OUTPUT_PATH_PHASE2_3)
                if p.exists():
                    historical_df = pd.read_parquet(p)
                    logger.info("Loaded historical context: %d rows", len(historical_df))
            except Exception as exc:
                logger.warning("Could not load historical context: %s", exc)

            if connected and historical_df is not None:
                try:
                    from live.data_bridge import fill_gap
                    historical_df = fill_gap(historical_df, self._symbol)
                except Exception as exc:
                    logger.warning("fill_gap failed: %s", exc)

            # Fetch last 100 M15 candles from MT5 for the chart
            chart_df: pd.DataFrame | None = None
            if connected:
                try:
                    import MetaTrader5 as mt5  # noqa: PLC0415
                    rates = mt5.copy_rates_from_pos(
                        self._symbol, mt5.TIMEFRAME_M15, 0, 100
                    )
                    if rates is not None and len(rates) > 0:
                        chart_df = pd.DataFrame(rates)
                        logger.info("Fetched %d M15 candles from MT5", len(chart_df))
                except Exception as exc:
                    logger.warning("Failed to fetch M15 candles from MT5: %s", exc)

            cycle = 0
            last_signal: "dict | None" = None
            while self._running:
                cycle += 1

                try:
                    # Fetch latest M15 bar
                    live_bar = None
                    if connected:
                        try:
                            from live.data_bridge import get_live_m15_bar
                            live_bar = get_live_m15_bar(self._symbol)
                        except Exception as exc:
                            logger.warning("get_live_m15_bar failed: %s", exc)

                    if live_bar is None:
                        if historical_df is not None and len(historical_df) > 0:
                            live_bar = historical_df.tail(1)
                        else:
                            logger.warning("No data available — skipping cycle.")
                            self._sleep_interruptible(60)
                            continue

                    # Refresh MT5 chart candles each cycle
                    if connected:
                        try:
                            import MetaTrader5 as mt5  # noqa: PLC0415
                            rates = mt5.copy_rates_from_pos(
                                self._symbol, mt5.TIMEFRAME_M15, 0, 100
                            )
                            if rates is not None and len(rates) > 0:
                                chart_df = pd.DataFrame(rates)
                        except Exception:
                            pass

                    # Compute features
                    if historical_df is not None:
                        feature_row = _build_feature_row(live_bar, historical_df)
                    else:
                        feature_row = live_bar.iloc[-1] if len(live_bar) > 0 else None

                    if feature_row is None:
                        logger.warning("Feature computation returned None — skipping.")
                        self._sleep_interruptible(60)
                        continue

                    # CRITICAL FIX: Append new live bar to historical context
                    # so next cycle sees updated data and produces different predictions
                    if historical_df is not None and live_bar is not None:
                        try:
                            last_ts = historical_df.index[-1] if isinstance(historical_df.index, pd.DatetimeIndex) else None
                            live_ts = live_bar.index[-1] if isinstance(live_bar.index, pd.DatetimeIndex) else None
                            if last_ts is None or live_ts is None or live_ts > last_ts:
                                historical_df = pd.concat([historical_df, live_bar]).drop_duplicates().tail(10000)
                                logger.debug("Appended live bar to historical context — now %d rows", len(historical_df))
                        except Exception as append_exc:
                            logger.debug("Could not append live bar: %s", append_exc)

                    self.market_updated.emit(feature_row)

                    # Emit chart: prefer MT5 real-time candles, fall back to historical_df
                    emit_df = chart_df if chart_df is not None else historical_df
                    if emit_df is not None:
                        self.chart_ready.emit(emit_df, None)

                    # Raw prediction (before filters)
                    try:
                        raw = gen.predict_raw(feature_row)
                        raw_confidence = float(raw["confidence"])
                        raw_direction = str(raw["direction"])
                        logger.info("Raw prediction: %s %.1f%% | probs: %s", raw_direction, raw_confidence, raw["probabilities"])
                    except Exception as raw_exc:
                        logger.debug("predict_raw failed: %s", raw_exc)
                        raw_confidence = 0.0
                        raw_direction = "NO_TRADE"

                    # Filtered signal
                    sig = gen.generate_signal(feature_row, bar_index=cycle)
                    last_signal = sig
                    self.prediction_ready.emit(sig, raw_confidence, raw_direction)

                    if sig is not None:
                        logger.info("Signal generated: %s", sig)
                        _log_signal_to_csv(sig)
                        try:
                            send_signal(sig)
                        except Exception:
                            pass
                        self.signal_fired.emit(sig)
                        emit_df = chart_df if chart_df is not None else historical_df
                        if emit_df is not None:
                            self.chart_ready.emit(emit_df, sig)

                    self.cycle_done.emit(cycle)

                except Exception as exc:
                    logger.error("Error in live loop cycle %d: %s", cycle, exc, exc_info=True)
                    try:
                        from live.telegram_bot import send_alert as _alert
                        _alert(f"Error in live loop: {exc}")
                    except Exception:
                        pass

                # Wait until next M15 close — 1-second ticks with live candle updates
                deadline = time.monotonic() + _seconds_until_next_m15()
                # Track last-candle overrides separately to avoid a full DataFrame copy per tick
                tick_high: float | None = None
                tick_low: float | None = None
                while self._running:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    # Fetch latest tick and update the last (forming) candle in real-time
                    if connected and chart_df is not None:
                        try:
                            import MetaTrader5 as mt5  # noqa: PLC0415
                            tick = mt5.symbol_info_tick(self._symbol)
                            if tick is not None:
                                current_price = tick.bid
                                last_idx = len(chart_df) - 1
                                orig_high = float(chart_df.loc[last_idx, "high"])
                                orig_low = float(chart_df.loc[last_idx, "low"])
                                tick_high = max(tick_high or orig_high, current_price)
                                tick_low = min(tick_low or orig_low, current_price)
                                # Only copy once we have updated tick values to emit
                                chart_df_copy = chart_df.copy()
                                chart_df_copy.loc[last_idx, "close"] = current_price
                                chart_df_copy.loc[last_idx, "high"] = tick_high
                                chart_df_copy.loc[last_idx, "low"] = tick_low
                                self.chart_ready.emit(chart_df_copy, last_signal)
                        except Exception:
                            pass
                    time.sleep(min(1.0, remaining))

            # Cleanup
            if connected:
                try:
                    from live import mt5_connector
                    mt5_connector.disconnect()
                except Exception:
                    pass
            logger.info("TradingWorker stopped.")

        def _sleep_interruptible(self, seconds: float) -> None:
            end = time.monotonic() + seconds
            while self._running and time.monotonic() < end:
                time.sleep(0.5)

    return TradingWorker


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_live(
    symbol: str = MT5_SYMBOL,
    offline_mode: bool = False,
) -> None:
    """
    Main live trading entry point.

    Creates the QApplication + GoldwolfGUI, starts TradingWorker in a
    background thread, and drives countdown updates via QTimer.

    Parameters
    ----------
    symbol       : MT5 symbol to trade.
    offline_mode : If True, skip MT5 connection (for testing).
    """
    from PySide6.QtCore import QTimer              # noqa: PLC0415
    from PySide6.QtWidgets import QApplication     # noqa: PLC0415

    logger.info("=" * 60)
    logger.info("GOLDWOLF — Live Signal Runner (PySide6 GUI)")
    logger.info("Symbol: %s | Mode: %s", symbol, "OFFLINE" if offline_mode else "LIVE")
    logger.info("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)

    from live.gui import GoldwolfGUI               # noqa: PLC0415
    gui = GoldwolfGUI()
    gui.show()

    # Build and instantiate the worker
    TradingWorker = _make_trading_worker_class()
    worker = TradingWorker(symbol=symbol, offline_mode=offline_mode)

    # Connect worker signals → GUI slots (all on main thread)
    worker.market_updated.connect(gui.update_market)
    worker.market_updated.connect(gui.update_candle_dna)
    worker.prediction_ready.connect(gui.update_prediction)
    worker.signal_fired.connect(gui.add_signal_to_history)
    worker.signal_fired.connect(lambda _sig: gui.update_performance())
    worker.cycle_done.connect(gui.set_cycle_status)
    worker.mt5_status.connect(gui.set_mt5_status)
    worker.chart_ready.connect(gui.update_chart)

    # Countdown QTimer — fires every second, updates GUI on main thread
    _deadline: list[float] = [time.monotonic() + _seconds_until_next_m15()]

    def _tick_countdown() -> None:
        remaining = _deadline[0] - time.monotonic()
        if remaining < 0:
            _deadline[0] = time.monotonic() + _seconds_until_next_m15()
            remaining = _deadline[0] - time.monotonic()
        gui.update_countdown(max(0.0, remaining))

    countdown_timer = QTimer()
    countdown_timer.setInterval(1000)
    countdown_timer.timeout.connect(_tick_countdown)
    countdown_timer.start()

    # Graceful shutdown: stop worker when window closes
    def _on_close() -> None:
        worker.stop()
        countdown_timer.stop()
        worker.wait(5000)

    app.aboutToQuit.connect(_on_close)

    worker.start()
    sys.exit(app.exec())
