"""
GOLDWOLF — PySide6 Professional Trading Desk GUI
Full multi-panel trading desk GUI using QMainWindow + QDockWidget.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QGridLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import qdarkstyle
    _HAS_QDARKSTYLE = True
except ImportError:
    _HAS_QDARKSTYLE = False

try:
    import pyqtgraph as pg
    _HAS_PYQTGRAPH = True
except ImportError:
    _HAS_PYQTGRAPH = False

from config.settings import SIGNAL_LOG_PATH, SIGNAL_MIN_CONFIDENCE, PIP_SIZE
from utils.helpers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DNA_NAMES: dict[int, str] = {
    -1: "Unclassified",
    0: "Clean Push",
    1: "Absorption",
    2: "Exhaustion",
    3: "Trap",
    4: "Dead Zone",
    5: "Breakout",
    6: "Fakeout",
    7: "Accumulation",
}
SESSION_NAMES: dict[int, str] = {0: "Asian", 1: "London", 2: "New York"}
KILL_ZONE_NAMES: dict[int, str] = {0: "None", 1: "London Open", 2: "NY Open", 3: "London Close"}

COLOR_BG = "#1a1a2e"
COLOR_PANEL = "#16213e"
COLOR_ACCENT = "#0f3460"
COLOR_TEXT = "#e0e0e0"
COLOR_BUY = "#00ff88"
COLOR_SELL = "#ff4444"
COLOR_WARN = "#ffaa00"
COLOR_INFO = "#00bcd4"
COLOR_GOLD = "#ffd700"

GLOBAL_QSS = f"""
QMainWindow {{ background-color: {COLOR_BG}; }}
QWidget {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
QDockWidget {{ color: {COLOR_TEXT}; font-weight: bold; }}
QDockWidget::title {{ background: {COLOR_ACCENT}; padding: 6px; }}
QLabel {{ color: {COLOR_TEXT}; }}
QTableWidget {{ background-color: {COLOR_PANEL}; color: {COLOR_TEXT};
                gridline-color: {COLOR_ACCENT}; }}
QTableWidget::item {{ padding: 4px; }}
QHeaderView::section {{ background-color: {COLOR_ACCENT}; color: {COLOR_TEXT};
                        padding: 4px; border: none; }}
QProgressBar {{ background-color: {COLOR_ACCENT}; border: 1px solid {COLOR_BG};
                border-radius: 4px; text-align: center; color: {COLOR_TEXT}; }}
QProgressBar::chunk {{ background-color: {COLOR_INFO}; border-radius: 4px; }}
QStatusBar {{ background-color: {COLOR_ACCENT}; color: {COLOR_TEXT}; }}
"""

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _styled_label(
    text: str = "",
    font_size: int = 10,
    bold: bool = False,
    color: str = COLOR_TEXT,
    align: Qt.AlignmentFlag = Qt.AlignLeft,
) -> QLabel:
    lbl = QLabel(text)
    style = f"color: {color}; font-size: {font_size}pt;"
    if bold:
        style += " font-weight: bold;"
    lbl.setStyleSheet(style)
    lbl.setAlignment(align)
    return lbl


def _progress_bar(value: int = 0, max_val: int = 100, color: str = COLOR_INFO) -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, max_val)
    bar.setValue(value)
    bar.setStyleSheet(
        f"QProgressBar::chunk {{ background-color: {color}; border-radius: 4px; }}"
    )
    bar.setFixedHeight(16)
    return bar


# ---------------------------------------------------------------------------
# Panel: Market Status
# ---------------------------------------------------------------------------

class MarketPanel(QWidget):
    """Panel 2 — Market Status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setSpacing(6)
        grid.setContentsMargins(8, 8, 8, 8)

        self._price_lbl = _styled_label("—", font_size=18, bold=True, color=COLOR_INFO,
                                        align=Qt.AlignLeft)
        self._session_lbl = _styled_label("—")
        self._killzone_lbl = _styled_label("—")
        self._vol_lbl = _styled_label("—")
        self._h4_lbl = _styled_label("—")
        self._h1_lbl = _styled_label("—")

        rows = [
            ("💰 Price", self._price_lbl),
            ("🌍 Session", self._session_lbl),
            ("🔥 Kill Zone", self._killzone_lbl),
            ("⚡ Volatility", self._vol_lbl),
            ("📈 H4 Trend", self._h4_lbl),
            ("📉 H1 Trend", self._h1_lbl),
        ]
        for i, (label, widget) in enumerate(rows):
            grid.addWidget(_styled_label(label, bold=True), i, 0)
            grid.addWidget(widget, i, 1)

    def update_market(self, feature_row: pd.Series) -> None:
        price = _safe_float(feature_row.get("m15_close", 0.0))
        session = int(_safe_float(feature_row.get("l2_session", 0)))
        kill_zone = int(_safe_float(feature_row.get("l2_kill_zone", 0)))
        vol_rank = _safe_float(feature_row.get("l2_session_volatility_rank", 0.0))
        h4 = _safe_float(feature_row.get("l3_h4_trend", 0.0))
        h1 = _safe_float(feature_row.get("l3_h1_trend", 0.0))

        self._price_lbl.setText(f"{price:.5f}")

        sess_name = SESSION_NAMES.get(session, f"S{session}")
        self._session_lbl.setText(sess_name)

        kz_name = KILL_ZONE_NAMES.get(kill_zone, "None")
        kz_color = COLOR_SELL if kill_zone > 0 else COLOR_TEXT
        self._killzone_lbl.setText(kz_name)
        self._killzone_lbl.setStyleSheet(f"color: {kz_color};")

        if vol_rank >= 0.66:
            vol_str, vol_color = "HIGH", COLOR_SELL
        elif vol_rank >= 0.33:
            vol_str, vol_color = "MED", COLOR_WARN
        else:
            vol_str, vol_color = "LOW", COLOR_INFO
        self._vol_lbl.setText(vol_str)
        self._vol_lbl.setStyleSheet(f"color: {vol_color}; font-weight: bold;")

        self._h4_lbl.setText(self._trend_text(h4))
        self._h4_lbl.setStyleSheet(f"color: {self._trend_color(h4)}; font-weight: bold;")
        self._h1_lbl.setText(self._trend_text(h1))
        self._h1_lbl.setStyleSheet(f"color: {self._trend_color(h1)}; font-weight: bold;")

    @staticmethod
    def _trend_text(val: float) -> str:
        if val > 0:
            return "BULL ▲"
        if val < 0:
            return "BEAR ▼"
        return "NEUTRAL ─"

    @staticmethod
    def _trend_color(val: float) -> str:
        if val > 0:
            return COLOR_BUY
        if val < 0:
            return COLOR_SELL
        return "#888888"


# ---------------------------------------------------------------------------
# Panel: Candle DNA
# ---------------------------------------------------------------------------

class CandleDnaPanel(QWidget):
    """Panel 3 — Candle DNA."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setSpacing(6)
        grid.setContentsMargins(8, 8, 8, 8)

        self._type_lbl = _styled_label("—", bold=True)
        self._whale_bar = _progress_bar(0, 3, COLOR_WARN)
        self._trap_bar = _progress_bar(0, 100, COLOR_INFO)
        self._confluence_lbl = _styled_label("—")
        self._momentum_lbl = _styled_label("—")
        self._liq_lbl = _styled_label("—")

        rows = [
            ("🧬 Type", self._type_lbl),
            ("🐋 Whale Footprint", self._whale_bar),
            ("⚠️ Trap Score", self._trap_bar),
            ("🎯 Confluence", self._confluence_lbl),
            ("📊 Momentum", self._momentum_lbl),
            ("🔄 Liquidity Sweep", self._liq_lbl),
        ]
        for i, (label, widget) in enumerate(rows):
            grid.addWidget(_styled_label(label, bold=True), i, 0)
            grid.addWidget(widget, i, 1)

    def update_candle_dna(self, feature_row: pd.Series) -> None:
        dna = int(_safe_float(feature_row.get("l4_candle_dna", -1)))
        whale = _safe_float(feature_row.get("l4_whale_footprint", 0.0))
        trap = _safe_float(feature_row.get("l4_trap_score", 0.0))
        conf = _safe_float(feature_row.get("l4_multi_layer_confluence", 0.0))
        mom = _safe_float(feature_row.get("l4_momentum_divergence", 0.0))
        liq = _safe_float(feature_row.get("l3_liquidity_sweep", 0.0))

        self._type_lbl.setText(DNA_NAMES.get(dna, "Unknown"))

        self._whale_bar.setValue(int(min(3, max(0, round(whale)))))

        trap_int = int(min(100, max(0, trap)))
        self._trap_bar.setValue(trap_int)
        trap_color = COLOR_SELL if trap > 60 else COLOR_INFO
        self._trap_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {trap_color}; border-radius: 4px; }}"
        )

        conf_color = COLOR_BUY if conf > 0 else (COLOR_SELL if conf < 0 else "#888888")
        self._confluence_lbl.setText(f"{conf:+.0f}")
        self._confluence_lbl.setStyleSheet(f"color: {conf_color}; font-weight: bold;")

        mom_str = "▲" if mom > 0 else ("▼" if mom < 0 else "─")
        mom_color = COLOR_BUY if mom > 0 else (COLOR_SELL if mom < 0 else "#888888")
        self._momentum_lbl.setText(mom_str)
        self._momentum_lbl.setStyleSheet(f"color: {mom_color}; font-size: 12pt; font-weight: bold;")

        liq_str = "▲" if liq > 0 else ("▼" if liq < 0 else "None")
        liq_color = COLOR_WARN if liq != 0 else "#888888"
        self._liq_lbl.setText(liq_str)
        self._liq_lbl.setStyleSheet(f"color: {liq_color};")


# ---------------------------------------------------------------------------
# Panel: Countdown + Signal
# ---------------------------------------------------------------------------

class CountdownSignalPanel(QWidget):
    """Panel 4 — Countdown + Signal."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Countdown section
        self._countdown_bar = _progress_bar(0, 900, COLOR_INFO)
        self._countdown_bar.setFixedHeight(20)
        self._countdown_lbl = _styled_label("15:00", font_size=16, bold=True,
                                            color=COLOR_INFO, align=Qt.AlignCenter)

        layout.addWidget(_styled_label("⏳ Next M15 Candle", bold=True))
        layout.addWidget(self._countdown_bar)
        layout.addWidget(self._countdown_lbl)

        # Divider
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background-color: {COLOR_ACCENT};")
        layout.addWidget(line)

        # Signal section
        self._dir_lbl = _styled_label("⚪ NO TRADE", font_size=20, bold=True,
                                       color="#888888", align=Qt.AlignCenter)
        self._conf_bar = _progress_bar(0, 100, "#888888")
        self._conf_lbl = _styled_label("Confidence: 0.0%", align=Qt.AlignCenter)
        self._tier_lbl = _styled_label("", align=Qt.AlignCenter)
        self._entry_lbl = _styled_label("")
        self._sl_lbl = _styled_label("")
        self._tp_lbl = _styled_label("")
        self._rr_lbl = _styled_label("")
        self._reasons_lbl = _styled_label("")
        self._standby_lbl = _styled_label(
            "🐺 Standing by... waiting for high-probability setup",
            color="#888888", align=Qt.AlignCenter
        )
        self._standby_lbl.setWordWrap(True)

        layout.addWidget(self._dir_lbl)
        layout.addWidget(self._conf_bar)
        layout.addWidget(self._conf_lbl)
        layout.addWidget(self._tier_lbl)
        layout.addWidget(self._entry_lbl)
        layout.addWidget(self._sl_lbl)
        layout.addWidget(self._tp_lbl)
        layout.addWidget(self._rr_lbl)
        layout.addWidget(self._reasons_lbl)
        layout.addWidget(self._standby_lbl)
        layout.addStretch()

    def update_countdown(self, seconds_remaining: float) -> None:
        secs = max(0.0, seconds_remaining)
        elapsed = max(0, 900 - int(secs))
        self._countdown_bar.setValue(elapsed)
        mins = int(secs) // 60
        sec = int(secs) % 60
        self._countdown_lbl.setText(f"{mins:02d}:{sec:02d}")

    def update_prediction(
        self,
        signal: dict[str, Any] | None,
        confidence: float,
        direction: str,
    ) -> None:
        conf_int = int(min(100, max(0, confidence)))
        self._conf_bar.setValue(conf_int)
        self._conf_lbl.setText(f"Confidence: {confidence:.1f}%")

        if signal is not None and confidence >= SIGNAL_MIN_CONFIDENCE:
            dir_color = COLOR_BUY if direction == "BUY" else COLOR_SELL
            dir_icon = "🟢 BUY" if direction == "BUY" else "🔴 SELL"
            self._dir_lbl.setText(dir_icon)
            self._dir_lbl.setStyleSheet(
                f"color: {dir_color}; font-size: 20pt; font-weight: bold;"
            )
            self._conf_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background-color: {dir_color}; border-radius: 4px; }}"
            )

            tier = signal.get("tier", 1)
            tier_map = {1: ("🥉 TIER 1", COLOR_TEXT), 2: ("🥈 TIER 2", COLOR_TEXT),
                        3: ("🥇 TIER 3", COLOR_GOLD)}
            tier_str, tier_color = tier_map.get(tier, (f"TIER {tier}", COLOR_TEXT))
            self._tier_lbl.setText(tier_str)
            self._tier_lbl.setStyleSheet(f"color: {tier_color}; font-weight: bold;")

            entry = _safe_float(signal.get("entry_price", 0.0))
            sl = _safe_float(signal.get("stop_loss", 0.0))
            tp = _safe_float(signal.get("take_profit", 0.0))
            rr = _safe_float(signal.get("risk_reward", 0.0))
            sl_pips = abs(entry - sl) / PIP_SIZE
            tp_pips = abs(tp - entry) / PIP_SIZE

            self._entry_lbl.setText(f"Entry: {entry:.5f}")
            self._entry_lbl.setStyleSheet(f"color: {COLOR_TEXT};")
            self._sl_lbl.setText(f"SL: {sl:.5f}  ({sl_pips:.0f} pips)")
            self._sl_lbl.setStyleSheet(f"color: {COLOR_SELL};")
            self._tp_lbl.setText(f"TP: {tp:.5f}  ({tp_pips:.0f} pips)")
            self._tp_lbl.setStyleSheet(f"color: {COLOR_BUY};")
            self._rr_lbl.setText(f"R:R  {rr:.2f}")
            self._rr_lbl.setStyleSheet(f"color: {COLOR_TEXT};")

            reasons = signal.get("top_reasons", [])
            self._reasons_lbl.setText(" | ".join(str(r) for r in reasons[:3]))
            self._reasons_lbl.setStyleSheet(f"color: #888888;")
            self._standby_lbl.hide()

        else:
            self._dir_lbl.setText("⚪ NO TRADE")
            self._dir_lbl.setStyleSheet("color: #888888; font-size: 20pt; font-weight: bold;")
            self._conf_bar.setStyleSheet(
                "QProgressBar::chunk { background-color: #888888; border-radius: 4px; }"
            )
            self._tier_lbl.setText("")
            self._entry_lbl.setText("")
            self._sl_lbl.setText("")
            self._tp_lbl.setText("")
            self._rr_lbl.setText("")
            self._reasons_lbl.setText("")
            self._standby_lbl.show()


# ---------------------------------------------------------------------------
# Panel: Signal History
# ---------------------------------------------------------------------------

class SignalHistoryPanel(QWidget):
    """Panel 5 — Signal History table."""

    COLUMNS = ["TIME", "DIR", "CONF", "TIER", "RESULT", "PIPS"]
    MAX_ROWS = 50

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(False)

        layout.addWidget(self._table)
        self._load_from_csv()

    def _load_from_csv(self) -> None:
        path = Path(SIGNAL_LOG_PATH)
        if not path.exists():
            return
        try:
            rows: list[dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
            for row in rows[-self.MAX_ROWS:]:
                self._append_csv_row(row)
        except Exception as exc:
            logger.debug("Could not load signal log: %s", exc)

    def _append_csv_row(self, row: dict[str, Any]) -> None:
        ts = str(row.get("timestamp", ""))[:16]
        direction = str(row.get("direction", "?"))
        conf = f"{_safe_float(row.get('confidence', 0)):.0f}%"
        tier = f"T{row.get('tier', '?')}"
        result = str(row.get("result", "⏳ OPEN"))
        pips = str(row.get("pips", "-"))
        self._add_row(ts, direction, conf, tier, result, pips)

    def add_signal_to_history(self, signal: dict[str, Any]) -> None:
        ts = str(signal.get("timestamp", ""))[:16]
        direction = str(signal.get("direction", "?"))
        conf = f"{_safe_float(signal.get('confidence', 0)):.0f}%"
        tier = f"T{signal.get('tier', 1)}"
        result = "⏳ OPEN"
        pips = "-"
        self._add_row(ts, direction, conf, tier, result, pips)
        # Trim to MAX_ROWS
        while self._table.rowCount() > self.MAX_ROWS:
            self._table.removeRow(0)

    def _add_row(
        self,
        ts: str,
        direction: str,
        conf: str,
        tier: str,
        result: str,
        pips: str,
    ) -> None:
        row_idx = self._table.rowCount()
        self._table.insertRow(row_idx)

        dir_color = QColor(COLOR_BUY) if direction == "BUY" else QColor(COLOR_SELL)

        result_color = QColor(COLOR_TEXT)
        if "WIN" in result or result.lower() == "win":
            result_color = QColor(COLOR_BUY)
            result = "✅ WIN"
        elif "LOSS" in result or result.lower() == "loss":
            result_color = QColor(COLOR_SELL)
            result = "❌ LOSS"
        elif "OPEN" in result:
            result_color = QColor(COLOR_WARN)
            result = "⏳ OPEN"

        values = [ts, direction, conf, tier, result, pips]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setForeground(dir_color if col == 1 else
                               result_color if col == 4 else
                               QColor(COLOR_TEXT))
            self._table.setItem(row_idx, col, item)

        self._table.scrollToBottom()


# ---------------------------------------------------------------------------
# Panel: Performance + Equity Curve
# ---------------------------------------------------------------------------

class PerformancePanel(QWidget):
    """Panel 6 — Performance stats + equity curve."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(4)

        self._total_lbl = _styled_label("0")
        self._wl_lbl = _styled_label("—")
        self._wr_bar = _progress_bar(0, 100, COLOR_BUY)
        self._wr_lbl = _styled_label("0.0%")
        self._pips_lbl = _styled_label("0")
        self._pf_lbl = _styled_label("—")

        stats = [
            ("Total Signals", self._total_lbl),
            ("Wins / Losses / Open", self._wl_lbl),
            ("Win Rate", self._wr_bar),
            ("", self._wr_lbl),
            ("Total Pips", self._pips_lbl),
            ("Profit Factor", self._pf_lbl),
        ]
        for i, (label, widget) in enumerate(stats):
            if label:
                grid.addWidget(_styled_label(label, bold=True), i, 0)
            grid.addWidget(widget, i, 1)

        layout.addLayout(grid)

        # Equity curve
        self._plot: "pg.PlotWidget | None"
        self._curve: "pg.PlotDataItem | None"
        if _HAS_PYQTGRAPH:
            pg.setConfigOption("background", COLOR_BG)
            pg.setConfigOption("foreground", COLOR_TEXT)
            self._plot = pg.PlotWidget(title="Equity Curve (Cumulative Pips)")
            self._plot.setLabel("left", "Pips")
            self._plot.setLabel("bottom", "Trade #")
            self._curve = self._plot.plot(pen=pg.mkPen(COLOR_BUY, width=2))
            layout.addWidget(self._plot)
        else:
            self._plot = None
            self._curve = None
            layout.addWidget(_styled_label("Install pyqtgraph for equity curve", color="#888888"))

    def update_performance(self) -> None:
        path = Path(SIGNAL_LOG_PATH)
        rows: list[dict[str, Any]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = [dict(r) for r in reader]
            except Exception as exc:
                logger.debug("Could not load signal log: %s", exc)

        total = len(rows)
        wins = sum(1 for r in rows if str(r.get("result", "")).strip().lower() == "win")
        losses = sum(1 for r in rows if str(r.get("result", "")).strip().lower() == "loss")
        open_c = total - wins - losses
        win_rate = (wins / total * 100) if total > 0 else 0.0
        pips_list = [_safe_float(r.get("pips", 0)) for r in rows]
        total_pips = sum(pips_list)

        self._total_lbl.setText(str(total))
        self._wl_lbl.setText(f"✅ {wins}  ❌ {losses}  ⏳ {open_c}")
        self._wr_bar.setValue(int(win_rate))
        self._wr_lbl.setText(f"{win_rate:.1f}%")

        pips_color = COLOR_BUY if total_pips >= 0 else COLOR_SELL
        self._pips_lbl.setText(f"{total_pips:+.0f}")
        self._pips_lbl.setStyleSheet(f"color: {pips_color}; font-weight: bold;")

        # Profit factor
        gross_win = sum(p for p in pips_list if p > 0)
        gross_loss = abs(sum(p for p in pips_list if p < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        self._pf_lbl.setText(f"{pf:.2f}" if pf != float("inf") else "∞")

        # Update equity curve
        if self._curve is not None and pips_list:
            cumulative = [sum(pips_list[:i + 1]) for i in range(len(pips_list))]
            self._curve.setData(list(range(1, len(cumulative) + 1)), cumulative)


# ---------------------------------------------------------------------------
# Panel: Candlestick Chart (Central Widget)
# ---------------------------------------------------------------------------

class ChartPanel(QWidget):
    """Panel 1 — Candlestick Chart."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._plot: "pg.PlotWidget | None"
        if _HAS_PYQTGRAPH:
            pg.setConfigOption("background", COLOR_BG)
            pg.setConfigOption("foreground", COLOR_TEXT)
            self._plot = pg.PlotWidget(title="XAUUSD M15 — Candlestick")
            self._plot.setLabel("left", "Price")
            self._plot.setLabel("bottom", "Candle #")
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            layout.addWidget(self._plot)
            self._has_chart = True
        else:
            self._plot = None
            self._has_chart = False
            lbl = _styled_label(
                "Install pyqtgraph to display the candlestick chart.",
                color="#888888", align=Qt.AlignCenter
            )
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        self._signal_lines: list[Any] = []

    def update_chart(
        self, historical_df: pd.DataFrame, signal: dict[str, Any] | None = None
    ) -> None:
        if not self._has_chart or self._plot is None:
            return

        df = historical_df.tail(100).copy().reset_index(drop=True)
        if df.empty or "m15_close" not in df.columns:
            return

        self._plot.clear()
        self._signal_lines = []

        # Draw close-price line as fallback candlestick representation
        close_vals = df["m15_close"].astype(float).tolist()
        x = list(range(len(close_vals)))
        self._plot.plot(x, close_vals, pen=pg.mkPen(COLOR_INFO, width=1.5))

        # Attempt to draw OHLC bars using pyqtgraph CandlestickItem equivalent
        if all(c in df.columns for c in ("m15_open", "m15_high", "m15_low", "m15_close")):
            for i, row in df.iterrows():
                o = float(row["m15_open"])
                h = float(row["m15_high"])
                lo = float(row["m15_low"])
                c = float(row["m15_close"])
                color = COLOR_BUY if c >= o else COLOR_SELL
                # Wick
                self._plot.plot(
                    [i, i], [lo, h],
                    pen=pg.mkPen(color, width=1)
                )
                # Body
                self._plot.plot(
                    [i - 0.3, i + 0.3, i + 0.3, i - 0.3, i - 0.3],
                    [o, o, c, c, o],
                    pen=pg.mkPen(color, width=1),
                    fillLevel=None,
                )

        # Draw signal lines
        if signal is not None:
            n = len(df)
            entry = _safe_float(signal.get("entry_price", 0.0))
            sl = _safe_float(signal.get("stop_loss", 0.0))
            tp = _safe_float(signal.get("take_profit", 0.0))
            if entry:
                self._plot.addLine(y=entry, pen=pg.mkPen("white", width=1, style=Qt.SolidLine))
            if sl:
                self._plot.addLine(y=sl, pen=pg.mkPen(COLOR_SELL, width=1,
                                                        style=Qt.DashLine))
            if tp:
                self._plot.addLine(y=tp, pen=pg.mkPen(COLOR_BUY, width=1,
                                                        style=Qt.DashLine))


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class GoldwolfGUI(QMainWindow):
    """
    GOLDWOLF Professional Trading Desk GUI.

    All public update methods are safe to call from the main thread via Qt signals.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("🐺 GOLDWOLF by 130xer — XAUUSD | M15 | LIVE")
        self.resize(1400, 900)

        # Apply stylesheet
        if _HAS_QDARKSTYLE:
            self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6") + GLOBAL_QSS)
        else:
            self.setStyleSheet(GLOBAL_QSS)

        # --- Central widget: Chart ---
        self._chart = ChartPanel()
        self.setCentralWidget(self._chart)

        # --- Panel 2: Market Status (Right Top) ---
        self._market_panel = MarketPanel()
        self._add_dock("📈 MARKET STATUS", self._market_panel, Qt.RightDockWidgetArea)

        # --- Panel 3: Candle DNA (Right Middle) ---
        self._dna_panel = CandleDnaPanel()
        self._add_dock("🧬 CANDLE DNA", self._dna_panel, Qt.RightDockWidgetArea)

        # --- Panel 4: Countdown + Signal (Right Bottom) ---
        self._countdown_panel = CountdownSignalPanel()
        self._add_dock("⏳ COUNTDOWN & SIGNAL", self._countdown_panel, Qt.RightDockWidgetArea)

        # --- Panel 5: Signal History (Bottom Left) ---
        self._history_panel = SignalHistoryPanel()
        self._add_dock("📋 SIGNAL HISTORY", self._history_panel, Qt.BottomDockWidgetArea)

        # --- Panel 6: Performance + Equity Curve (Bottom Right) ---
        self._perf_panel = PerformancePanel()
        self._add_dock("📊 PERFORMANCE", self._perf_panel, Qt.BottomDockWidgetArea)

        # --- Status bar ---
        self._status_mt5 = QLabel("🔴 MT5 Disconnected")
        self._status_cycle = QLabel("Cycle: 0 | Last update: —")
        self._status_brand = QLabel("🐺 GOLDWOLF by 130xer")

        sb = self.statusBar()
        sb.addPermanentWidget(self._status_mt5)
        sb.addPermanentWidget(self._status_cycle, 1)
        sb.addPermanentWidget(self._status_brand)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea
            | Qt.RightDockWidgetArea
            | Qt.TopDockWidgetArea
            | Qt.BottomDockWidgetArea
        )
        self.addDockWidget(area, dock)
        return dock

    # ------------------------------------------------------------------
    # Public API — called from worker thread via Qt signals
    # ------------------------------------------------------------------

    @Slot(object)
    def update_market(self, feature_row: pd.Series) -> None:
        """Update market status panel from latest features."""
        self._market_panel.update_market(feature_row)

    @Slot(object)
    def update_candle_dna(self, feature_row: pd.Series) -> None:
        """Update candle DNA panel."""
        self._dna_panel.update_candle_dna(feature_row)

    @Slot(object, float, str)
    def update_prediction(
        self,
        signal: dict[str, Any] | None,
        confidence: float,
        direction: str,
    ) -> None:
        """Update signal/prediction panel."""
        self._countdown_panel.update_prediction(signal, confidence, direction)
        if signal is not None and signal.get("tier", 1) >= 2:
            QApplication.beep()

    @Slot(float)
    def update_countdown(self, seconds_remaining: float) -> None:
        """Update countdown timer."""
        self._countdown_panel.update_countdown(seconds_remaining)

    @Slot(object, object)
    def update_chart(
        self, historical_df: pd.DataFrame, signal: dict[str, Any] | None = None
    ) -> None:
        """Update candlestick chart with new data and trade markers."""
        self._chart.update_chart(historical_df, signal)

    @Slot(object)
    def add_signal_to_history(self, signal: dict[str, Any]) -> None:
        """Add signal to history table."""
        self._history_panel.add_signal_to_history(signal)

    @Slot()
    def update_performance(self) -> None:
        """Recalculate and update performance stats + equity curve."""
        self._perf_panel.update_performance()

    def set_mt5_status(self, connected: bool) -> None:
        """Update the MT5 connection status in the status bar."""
        if connected:
            self._status_mt5.setText("🟢 MT5 Connected")
            self._status_mt5.setStyleSheet(f"color: {COLOR_BUY};")
        else:
            self._status_mt5.setText("🔴 MT5 Disconnected")
            self._status_mt5.setStyleSheet(f"color: {COLOR_SELL};")

    def set_cycle_status(self, cycle: int) -> None:
        """Update cycle counter in the status bar."""
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
        self._status_cycle.setText(f"Cycle: {cycle} | Last update: {now} GMT")
