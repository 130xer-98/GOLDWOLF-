"""
GOLDWOLF — Rich Terminal Dashboard
Professional real-time terminal dashboard using the Rich library.

Displays a sticky, full-screen dashboard (screen=True) with:
  1. Header panel
  2. Market panel + Candle DNA panel (side by side)
  3. Countdown panel
  4. Prediction panel
  5. Signal history panel
  6. Today's performance panel
"""

from __future__ import annotations

import csv
import datetime
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import SIGNAL_LOG_PATH, SIGNAL_MIN_CONFIDENCE, PIP_SIZE
from utils.helpers import get_logger

logger = get_logger(__name__)

# Candle DNA type names (matches signals/generator.py)
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


def _make_bar(value: float, max_val: float, width: int = 20) -> str:
    """Create a simple block-character progress bar."""
    pct = max(0.0, min(1.0, value / max_val if max_val > 0 else 0.0))
    filled = int(pct * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _safe_float(value: object) -> float:
    """Convert *value* to float, returning 0.0 on failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


class GoldwolfDashboard:
    """
    Professional Rich terminal dashboard for the GOLDWOLF live runner.

    Usage
    -----
    dashboard = GoldwolfDashboard()
    with Live(dashboard.render(), refresh_per_second=1, screen=True) as live:
        # in your loop:
        dashboard.update_market(feature_row)
        dashboard.update_prediction(signal, confidence, direction)
        dashboard.update_countdown(seconds_remaining)
        live.update(dashboard.render())
    """

    def __init__(self) -> None:
        """Initialise the dashboard with empty / zero state."""
        self._lock = threading.Lock()

        # Market panel state
        self._price: float = 0.0
        self._session: int = 0
        self._kill_zone: int = 0
        self._vol_rank: float = 0.0
        self._h4_trend: float = 0.0
        self._h1_trend: float = 0.0

        # Candle DNA panel state
        self._candle_dna: int = -1
        self._whale_footprint: float = 0.0
        self._trap_score: float = 0.0
        self._confluence: float = 0.0
        self._momentum: float = 0.0
        self._liq_sweep: float = 0.0

        # Prediction panel state
        self._signal: dict[str, Any] | None = None
        self._confidence: float = 0.0
        self._direction: str = "NO_TRADE"

        # Countdown state
        self._countdown_secs: float = 900.0

        # In-memory signal history (current session)
        self._signals: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update_market(self, feature_row: pd.Series) -> None:
        """Update market data from the latest feature row."""
        with self._lock:
            self._price = float(feature_row.get("m15_close", 0.0))
            self._session = int(feature_row.get("l2_session", 0))
            self._kill_zone = int(feature_row.get("l2_kill_zone", 0))
            self._vol_rank = float(feature_row.get("l2_session_volatility_rank", 0.0))
            self._h4_trend = float(feature_row.get("l3_h4_trend", 0.0))
            self._h1_trend = float(feature_row.get("l3_h1_trend", 0.0))
            self._candle_dna = int(feature_row.get("l4_candle_dna", -1))
            self._whale_footprint = float(feature_row.get("l4_whale_footprint", 0.0))
            self._trap_score = float(feature_row.get("l4_trap_score", 0.0))
            self._confluence = float(feature_row.get("l4_multi_layer_confluence", 0.0))
            self._momentum = float(feature_row.get("l4_momentum_divergence", 0.0))
            self._liq_sweep = float(feature_row.get("l3_liquidity_sweep", 0.0))

    def update_prediction(
        self,
        signal: dict[str, Any] | None,
        confidence: float,
        direction: str,
    ) -> None:
        """Update the prediction panel.  signal=None means NO TRADE."""
        with self._lock:
            self._signal = signal
            self._confidence = confidence
            self._direction = direction

    def update_countdown(self, seconds_remaining: float) -> None:
        """Update the countdown timer (seconds until next M15 candle close)."""
        with self._lock:
            self._countdown_secs = max(0.0, seconds_remaining)

    def add_signal(self, signal: dict[str, Any]) -> None:
        """Add a fired signal to the in-memory history list."""
        with self._lock:
            entry: dict[str, Any] = {
                "time": str(signal.get("timestamp", ""))[:16],
                "dir": signal.get("direction", "?"),
                "conf": f"{signal.get('confidence', 0):.0f}%",
                "tier": f"T{signal.get('tier', 1)}",
                "result": "⏳",
                "pips": "-",
            }
            self._signals.append(entry)

    # ------------------------------------------------------------------
    # Render entry point
    # ------------------------------------------------------------------

    def render(self) -> "Layout":  # type: ignore[name-defined]
        """Build and return the full Rich Layout for the dashboard."""
        from rich.layout import Layout

        with self._lock:
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=4),
                Layout(name="market_row", size=10),
                Layout(name="countdown", size=3),
                Layout(name="prediction", size=12),
                Layout(name="history", size=14),
                Layout(name="performance", size=6),
            )
            layout["market_row"].split_row(
                Layout(name="market"),
                Layout(name="dna"),
            )

            layout["header"].update(self._build_header())
            layout["market_row"]["market"].update(self._build_market_panel())
            layout["market_row"]["dna"].update(self._build_dna_panel())
            layout["countdown"].update(self._build_countdown_panel())
            layout["prediction"].update(self._build_prediction_panel())
            layout["history"].update(self._build_history_panel())
            layout["performance"].update(self._build_performance_panel())

        return layout

    # ------------------------------------------------------------------
    # Private panel builders (called under self._lock)
    # ------------------------------------------------------------------

    def _build_header(self) -> "Panel":  # type: ignore[name-defined]
        from rich.align import Align
        from rich.panel import Panel
        from rich.text import Text

        text = Text()
        text.append("🐺  G O L D W O L F  v1.0\n", style="bold magenta")
        text.append("XAUUSD  •  M15  •  🟢 LIVE", style="bold cyan")
        return Panel(Align.center(text), style="bold cyan", height=4)

    def _build_market_panel(self) -> "Panel":  # type: ignore[name-defined]
        from rich.panel import Panel
        from rich.table import Table

        t = Table.grid(padding=(0, 1))
        t.add_column(style="cyan", no_wrap=True)
        t.add_column()

        t.add_row("💰 Price", f"[bold white]{self._price:.5f}[/bold white]")

        sess_name = SESSION_NAMES.get(self._session, f"S{self._session}")
        t.add_row("🌍 Session", f"[yellow]{sess_name}[/yellow]")

        kz_name = KILL_ZONE_NAMES.get(self._kill_zone, "None")
        kz_style = "bold red" if self._kill_zone > 0 else "dim"
        t.add_row("🔥 Kill Zone", f"[{kz_style}]{kz_name}[/{kz_style}]")

        t.add_row("⚡ Volatility", self._vol_label(self._vol_rank))
        t.add_row("📊 H4 Trend", self._trend_label(self._h4_trend))
        t.add_row("📊 H1 Trend", self._trend_label(self._h1_trend))

        return Panel(t, title="[bold cyan]📈 MARKET[/bold cyan]", style="bold cyan")

    def _build_dna_panel(self) -> "Panel":  # type: ignore[name-defined]
        from rich.panel import Panel
        from rich.table import Table

        t = Table.grid(padding=(0, 1))
        t.add_column(style="cyan", no_wrap=True)
        t.add_column()

        dna_name = DNA_NAMES.get(self._candle_dna, "Unknown")
        t.add_row("🧬 Type", f"[bold white]{dna_name}[/bold white]")

        whale_bar = _make_bar(self._whale_footprint, 3.0, width=12)
        t.add_row(
            "🐋 Whale",
            f"[yellow]{whale_bar}[/yellow] {self._whale_footprint:.0f}/3",
        )

        trap_bar = _make_bar(self._trap_score, 100.0, width=12)
        trap_color = "bold red" if self._trap_score > 60 else "yellow"
        t.add_row(
            "⚠️ Trap",
            f"[{trap_color}]{trap_bar}[/{trap_color}] {self._trap_score:.0f}",
        )

        conf_color = (
            "bold green"
            if self._confluence > 0
            else ("bold red" if self._confluence < 0 else "dim")
        )
        t.add_row(
            "🎯 Confluence",
            f"[{conf_color}]{self._confluence:+.0f}[/{conf_color}]",
        )

        mom_str = (
            "▲" if self._momentum > 0 else ("▼" if self._momentum < 0 else "─")
        )
        mom_color = (
            "green" if self._momentum > 0 else ("red" if self._momentum < 0 else "dim")
        )
        t.add_row("📊 Momentum", f"[{mom_color}]{mom_str}[/{mom_color}]")

        liq_str = (
            "▲" if self._liq_sweep > 0 else ("▼" if self._liq_sweep < 0 else "None")
        )
        liq_color = "yellow" if self._liq_sweep != 0 else "dim"
        t.add_row("🔄 Liq Sweep", f"[{liq_color}]{liq_str}[/{liq_color}]")

        return Panel(
            t,
            title="[bold cyan]🧬 CANDLE DNA[/bold cyan]",
            style="bold cyan",
        )

    def _build_countdown_panel(self) -> "Panel":  # type: ignore[name-defined]
        from rich.panel import Panel
        from rich.text import Text

        secs = self._countdown_secs
        elapsed = 900.0 - secs
        bar = _make_bar(elapsed, 900.0, width=30)
        mins = int(secs) // 60
        sec = int(secs) % 60

        text = Text()
        text.append(f"{bar}  ", style="cyan")
        text.append(f"{mins:02d}:{sec:02d}", style="bold cyan")

        return Panel(
            text,
            title="[bold cyan]⏱ NEXT CANDLE[/bold cyan]",
            style="bold cyan",
            height=3,
        )

    def _build_prediction_panel(self) -> "Panel":  # type: ignore[name-defined]
        from rich.panel import Panel
        from rich.table import Table

        t = Table.grid(padding=(0, 1))
        t.add_column(style="cyan", no_wrap=True)
        t.add_column()

        if self._signal is not None and self._confidence >= SIGNAL_MIN_CONFIDENCE:
            dir_color = "bold green" if self._direction == "BUY" else "bold red"
            dir_icon = "🟢" if self._direction == "BUY" else "🔴"
            t.add_row(
                "Direction",
                f"[{dir_color}]{dir_icon} {self._direction}[/{dir_color}]",
            )

            conf_bar = _make_bar(self._confidence, 100.0, width=20)
            t.add_row(
                "Confidence",
                f"[{dir_color}]{conf_bar}[/{dir_color}] {self._confidence:.1f}%",
            )

            tier = self._signal.get("tier", 1)
            tier_labels = {1: "🥉 TIER 1", 2: "🥈 TIER 2", 3: "🥇 TIER 3"}
            tier_colors = {1: "white", 2: "white", 3: "bold yellow"}
            tier_str = tier_labels.get(tier, f"TIER {tier}")
            tier_color = tier_colors.get(tier, "white")
            t.add_row("Tier", f"[{tier_color}]{tier_str}[/{tier_color}]")

            entry = self._signal.get("entry_price", 0.0)
            sl = self._signal.get("stop_loss", 0.0)
            tp = self._signal.get("take_profit", 0.0)
            rr = self._signal.get("risk_reward", 0.0)
            sl_pips = abs(float(entry) - float(sl)) / PIP_SIZE
            tp_pips = abs(float(tp) - float(entry)) / PIP_SIZE

            t.add_row("Entry", f"[white]{entry:.5f}[/white]")
            t.add_row("Stop Loss", f"[red]{sl:.5f}[/red] ({sl_pips:.0f} pips)")
            t.add_row("Take Profit", f"[green]{tp:.5f}[/green] ({tp_pips:.0f} pips)")
            t.add_row("R:R", f"[white]{rr:.2f}[/white]")

            reasons = self._signal.get("top_reasons", [])
            if reasons:
                t.add_row("Reasons", f"[dim]{' | '.join(str(r) for r in reasons[:3])}[/dim]")
        else:
            t.add_row("Direction", "[dim white]⚪ NO TRADE[/dim white]")
            conf_bar = _make_bar(self._confidence, 100.0, width=20)
            t.add_row(
                "Confidence",
                f"[dim]{conf_bar}[/dim] {self._confidence:.1f}%",
            )
            t.add_row(
                "",
                "[dim]🐺 Standing by... waiting for high-probability setup[/dim]",
            )

        return Panel(
            t,
            title="[bold cyan]🎯 PREDICTION[/bold cyan]",
            style="bold cyan",
        )

    def _build_history_panel(self) -> "Panel":  # type: ignore[name-defined]
        from rich.panel import Panel
        from rich.table import Table

        table = Table(
            show_header=True,
            header_style="bold cyan",
            style="bold cyan",
            show_lines=False,
        )
        table.add_column("TIME", style="dim", width=16)
        table.add_column("DIR", width=6)
        table.add_column("CONF", width=6)
        table.add_column("TIER", width=5)
        table.add_column("RESULT", width=10)
        table.add_column("PIPS", width=6)

        csv_rows = self._load_today_signals()
        rows_to_show = csv_rows[-10:] if csv_rows else self._signals[-10:]

        for row in rows_to_show:
            # CSV rows use string keys; in-memory entries use abbreviated keys
            if "direction" in row:
                ts = str(row.get("timestamp", ""))[:16]
                direction = str(row.get("direction", "?"))
                conf_str = f"{float(row.get('confidence', 0) or 0):.0f}%"
                tier_str = f"T{row.get('tier', '?')}"
                result_str = "⏳ OPEN"
                pips_str = "-"
            else:
                ts = str(row.get("time", ""))
                direction = str(row.get("dir", "?"))
                conf_str = str(row.get("conf", "-"))
                tier_str = str(row.get("tier", "-"))
                result_str = str(row.get("result", "⏳"))
                pips_str = str(row.get("pips", "-"))

            dir_color = "bold green" if direction == "BUY" else "bold red"
            table.add_row(
                ts,
                f"[{dir_color}]{direction}[/{dir_color}]",
                conf_str,
                tier_str,
                result_str,
                pips_str,
            )

        if not rows_to_show:
            table.add_row("[dim]No signals yet[/dim]", "", "", "", "", "")

        return Panel(
            table,
            title="[bold cyan]📋 SIGNAL HISTORY (Today)[/bold cyan]",
            style="bold cyan",
        )

    def _build_performance_panel(self) -> "Panel":  # type: ignore[name-defined]
        from rich.panel import Panel
        from rich.table import Table

        csv_rows = self._load_today_signals()
        total = len(csv_rows)
        wins = sum(
            1 for r in csv_rows if str(r.get("result", "")).strip().lower() == "win"
        )
        losses = sum(
            1 for r in csv_rows if str(r.get("result", "")).strip().lower() == "loss"
        )
        open_c = total - wins - losses
        win_rate = (wins / total * 100) if total > 0 else 0.0
        total_pips = sum(
            _safe_float(r.get("pips", 0)) for r in csv_rows
        )

        t = Table.grid(padding=(0, 2))
        t.add_column(style="cyan", no_wrap=True)
        t.add_column()

        t.add_row("Signals Today", f"[white]{total}[/white]")
        t.add_row(
            "W / L / Open",
            f"[green]{wins}[/green] / [red]{losses}[/red] / [yellow]{open_c}[/yellow]",
        )
        wr_bar = _make_bar(win_rate, 100.0, width=15)
        t.add_row("Win Rate", f"[green]{wr_bar}[/green] {win_rate:.1f}%")
        pips_color = "green" if total_pips >= 0 else "red"
        t.add_row("Total Pips", f"[{pips_color}]{total_pips:+.0f}[/{pips_color}]")

        return Panel(
            t,
            title="[bold cyan]📊 TODAY'S PERFORMANCE[/bold cyan]",
            style="bold cyan",
            height=6,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _trend_label(self, val: float) -> str:
        if val > 0:
            return "[bold green]BULL ▲[/bold green]"
        if val < 0:
            return "[bold red]BEAR ▼[/bold red]"
        return "[dim]NEUTRAL ─[/dim]"

    def _vol_label(self, rank: float) -> str:
        if rank >= 0.66:
            return "[bold red]HIGH[/bold red]"
        if rank >= 0.33:
            return "[bold yellow]MED[/bold yellow]"
        return "[cyan]LOW[/cyan]"

    def _load_today_signals(self) -> list[dict[str, Any]]:
        """Load today's signals from the signal log CSV (read-only, tolerant)."""
        path = Path(SIGNAL_LOG_PATH)
        today = str(datetime.date.today())
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = str(row.get("timestamp", ""))
                    if ts.startswith(today):
                        rows.append(dict(row))
        except Exception as exc:
            logger.debug("Could not read signal log: %s", exc)
        return rows
