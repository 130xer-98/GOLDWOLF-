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
