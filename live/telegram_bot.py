"""
GOLDWOLF — Telegram Bot
Sends trade signals and alerts via Telegram.

Credentials from .env:
  TELEGRAM_BOT_TOKEN = bot token from @BotFather
  TELEGRAM_CHAT_ID   = target chat or channel ID

Falls back to file logging if credentials are not configured.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import requests

from config.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_FANCY_FORMAT,
    SIGNAL_LOG_PATH,
)
from utils.helpers import get_logger

logger = get_logger(__name__)

_TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_SECONDS = 10


def _is_configured() -> bool:
    """Return True if Telegram credentials are set."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _send_message(text: str) -> bool:
    """
    Send a text message via Telegram API.

    Returns True on success, False on failure.
    """
    if not _is_configured():
        return False

    url = _TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("Telegram send failed: %s", exc)
        return False


def _log_to_file(message: str) -> None:
    """Log message to SIGNAL_LOG_PATH when Telegram is not configured."""
    log_path = Path(SIGNAL_LOG_PATH).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def _format_signal_fancy(signal: dict[str, Any]) -> str:
    """Format signal as a fancy Telegram message with emoji."""
    ts = signal.get("timestamp", "")
    if hasattr(ts, "strftime"):
        ts_str = ts.strftime("%Y-%m-%d %H:%M GMT")
    else:
        ts_str = str(ts)

    direction = signal.get("direction", "?")
    dir_emoji = "📈" if direction == "BUY" else "📉"
    confidence = signal.get("confidence", 0)
    tier = signal.get("tier", 1)
    entry = signal.get("entry_price", 0)
    sl = signal.get("stop_loss", 0)
    tp = signal.get("take_profit", 0)
    rr = signal.get("risk_reward", 0)
    reasons = signal.get("top_reasons", [])
    dna = signal.get("candle_dna", "Unknown")

    sl_pips = abs(entry - sl) / 0.1
    tp_pips = abs(tp - entry) / 0.1

    sl_str = f"{sl:.2f} (-{sl_pips:.0f} pips)"
    tp_str = f"{tp:.2f} (+{tp_pips:.0f} pips)"

    reasons_text = "\n".join(
        f"{i + 1}. {r}" for i, r in enumerate(reasons[:3])
    )

    return (
        f"🐺 <b>GOLDWOLF SIGNAL</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"{dir_emoji} <b>XAUUSD | {direction}</b>\n"
        f"💪 Confidence: {confidence:.0f}% (TIER {tier})\n"
        f"🕯 Candle: {dna}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💰 Entry: {entry:.2f}\n"
        f"🛑 SL: {sl_str}\n"
        f"✅ TP: {tp_str}\n"
        f"📐 R:R = 1:{rr:.1f}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🔑 <b>Reasons:</b>\n"
        f"{reasons_text}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"⏰ {ts_str}"
    )


def _format_signal_plain(signal: dict[str, Any]) -> str:
    """Format signal as plain text."""
    ts = signal.get("timestamp", "")
    if hasattr(ts, "strftime"):
        ts_str = ts.strftime("%Y-%m-%d %H:%M GMT")
    else:
        ts_str = str(ts)

    return (
        f"GOLDWOLF SIGNAL | {signal.get('direction', '?')} XAUUSD\n"
        f"Confidence: {signal.get('confidence', 0):.0f}% Tier {signal.get('tier', 1)}\n"
        f"Entry: {signal.get('entry_price', 0):.2f} | "
        f"SL: {signal.get('stop_loss', 0):.2f} | "
        f"TP: {signal.get('take_profit', 0):.2f}\n"
        f"Time: {ts_str}"
    )


def send_signal(signal: dict[str, Any]) -> bool:
    """
    Format and send a trade signal via Telegram.

    Falls back to file logging if credentials are not set.

    Parameters
    ----------
    signal : Signal dict from signals/generator.py.

    Returns
    -------
    True if sent via Telegram, False if logged to file.
    """
    if TELEGRAM_FANCY_FORMAT:
        message = _format_signal_fancy(signal)
    else:
        message = _format_signal_plain(signal)

    if _is_configured():
        success = _send_message(message)
        if success:
            logger.info("Signal sent via Telegram.")
            return True
        logger.warning("Telegram send failed — logging to file.")

    _log_to_file(message)
    return False


def send_alert(message: str) -> bool:
    """
    Send a plain error/info alert via Telegram.

    Falls back to file logging if credentials are not set.
    """
    full_msg = f"⚠️ <b>GOLDWOLF ALERT</b>\n{message}"

    if _is_configured():
        success = _send_message(full_msg)
        if success:
            return True

    _log_to_file(f"ALERT: {message}")
    return False


def send_daily_report(stats: dict[str, Any]) -> bool:
    """
    Send end-of-day performance summary via Telegram.

    Parameters
    ----------
    stats : dict with keys: date, total_signals, wins, losses, pips_net.
    """
    date = stats.get("date", str(datetime.date.today()))
    total = stats.get("total_signals", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pips = stats.get("pips_net", 0.0)
    win_rate = wins / total * 100 if total > 0 else 0

    message = (
        f"📊 <b>GOLDWOLF Daily Report — {date}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Signals: {total} | Wins: {wins} | Losses: {losses}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Net P&L: {pips:+.0f} pips\n"
        f"━━━━━━━━━━━━━━━━"
    )

    if _is_configured():
        success = _send_message(message)
        if success:
            return True

    _log_to_file(f"DAILY REPORT: {message}")
    return False
