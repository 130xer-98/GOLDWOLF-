"""
GOLDWOLF — Configuration & Settings
All configurable paths, parameters, and thresholds for Phase 1.
Values can be overridden via environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

# Load .env file if present (does not override existing env vars)
load_dotenv()

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
M1_DATA_PATH: str = os.getenv("M1_DATA_PATH", "D:/XAUUSD_M1_COMBINED.csv")
M15_DATA_PATH: str = os.getenv("M15_DATA_PATH", "D:/XAUUSD_M15_COMBINED.csv")

# Output file path (parquet preferred for size/speed)
OUTPUT_PATH: str = os.getenv("OUTPUT_PATH", "output/goldwolf_phase1.parquet")

# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
# Date and time columns in source CSVs (M1 format)
CSV_DATE_COL: str = "date"
CSV_TIME_COL: str = "time"
# Timestamp column in M15 format CSVs (single combined datetime column)
CSV_TIMESTAMP_COL: str = "timestamp"
# Combined datetime column name after parsing
CSV_DATETIME_COL: str = "datetime"
# Format used in the raw M1 CSV files
CSV_DATE_FORMAT: str = "%Y.%m.%d %H:%M"

# OHLCV column names
CSV_OPEN_COL: str = "open"
CSV_HIGH_COL: str = "high"
CSV_LOW_COL: str = "low"
CSV_CLOSE_COL: str = "close"
CSV_VOLUME_COL: str = "volume"

# ---------------------------------------------------------------------------
# Data filtering
# ---------------------------------------------------------------------------
# Minimum number of M1 candles required inside an M15 period to compute
# features.  Periods with fewer candles are still kept but flagged.
MIN_M1_CANDLES_PER_M15: int = 1

# ---------------------------------------------------------------------------
# M15 grouping
# ---------------------------------------------------------------------------
# Duration of the higher timeframe bar in minutes
M15_PERIOD_MINUTES: int = 15
# Expected number of M1 bars inside each M15 bar
M1_PER_M15: int = 15
# Early/Late split index (first 7 → indices 0-6, last 8 → indices 7-14)
EARLY_SPLIT: int = 7

# ---------------------------------------------------------------------------
# Feature thresholds
# ---------------------------------------------------------------------------
# Small constant to avoid division by zero in ratio features
EPSILON: float = 1e-10
# Body-to-range ratio below which an M1 candle is considered an absorption bar
ABSORPTION_BODY_RATIO_THRESHOLD: float = 0.3

# ---------------------------------------------------------------------------
# Output column names  (Layer 1)
# ---------------------------------------------------------------------------
COL_CUSTOM_VOLUME: str = "l1_custom_volume"
COL_VOLATILITY_ENERGY: str = "l1_volatility_energy"
COL_PRICE_VELOCITY: str = "l1_price_velocity"
COL_REVERSAL_COUNT: str = "l1_reversal_count"
COL_EARLY_LATE_RATIO: str = "l1_early_late_ratio"
COL_PRICE_ACCELERATION: str = "l1_price_acceleration"
COL_ABSORPTION_COUNT: str = "l1_absorption_count"
COL_ABSORPTION_INTENSITY: str = "l1_absorption_intensity"
COL_M1_COUNT: str = "l1_m1_count"

# ---------------------------------------------------------------------------
# Output path — Phase 2 + 3
# ---------------------------------------------------------------------------
OUTPUT_PATH_PHASE2_3: str = os.getenv(
    "OUTPUT_PATH_PHASE2_3", "output/goldwolf_phase2_3.parquet"
)

# ---------------------------------------------------------------------------
# Phase 2 — Time DNA settings
# ---------------------------------------------------------------------------
# XAUUSD pip size (1 pip = 0.1 price units)
PIP_SIZE: float = float(os.getenv("PIP_SIZE", "0.1"))

# Session boundaries (GMT hour, start inclusive, end exclusive)
# Asian: 00:00–08:00 → label 0
# London: 08:00–16:00 → label 1
# New York: 16:00–24:00 → label 2
SESSION_ASIAN_END: int = 8
SESSION_LONDON_END: int = 16

# London + NY overlap: 13:00–16:00 GMT
SESSION_OVERLAP_START: int = 13
SESSION_OVERLAP_END: int = 16

# Kill zones: (start_hour, start_minute, end_hour, end_minute)
KZ_LONDON_OPEN: tuple = (8, 0, 9, 0)    # → value 1
KZ_NY_OPEN: tuple = (13, 0, 14, 0)      # → value 2
KZ_LONDON_CLOSE: tuple = (15, 30, 16, 30)  # → value 3

# Volatility spike detection (l2_time_since_vol_spike)
VOL_SPIKE_WINDOW: int = int(os.getenv("VOL_SPIKE_WINDOW", "20"))
VOL_SPIKE_SIGMA: float = float(os.getenv("VOL_SPIKE_SIGMA", "2.0"))
VOL_SPIKE_CAP: int = int(os.getenv("VOL_SPIKE_CAP", "50"))

# Session volatility rank: rolling window of same-session candles
SESSION_VOL_RANK_WINDOW: int = int(os.getenv("SESSION_VOL_RANK_WINDOW", "100"))

# ---------------------------------------------------------------------------
# Phase 3 — SMC settings
# ---------------------------------------------------------------------------
# Swing detection lookback (N candles on each side)
SWING_LOOKBACK: int = int(os.getenv("SWING_LOOKBACK", "5"))

# Liquidity pool detection
LIQUIDITY_TOLERANCE_PIPS: float = float(os.getenv("LIQUIDITY_TOLERANCE_PIPS", "0.5"))
LIQUIDITY_MIN_TOUCHES: int = int(os.getenv("LIQUIDITY_MIN_TOUCHES", "3"))
LIQUIDITY_LOOKBACK: int = int(os.getenv("LIQUIDITY_LOOKBACK", "100"))

# Fair Value Gap minimum size (pips)
FVG_MIN_GAP_PIPS: float = float(os.getenv("FVG_MIN_GAP_PIPS", "0.5"))

# Order block maximum age (candles before expiry)
OB_MAX_AGE: int = int(os.getenv("OB_MAX_AGE", "500"))
