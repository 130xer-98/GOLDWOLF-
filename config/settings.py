"""
GOLDWOLF — Configuration & Settings
All configurable paths, parameters, and thresholds for Phases 1-7.
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

# ---------------------------------------------------------------------------
# Phase 4 — L4 feature thresholds
# ---------------------------------------------------------------------------
# l4_whale_footprint
L4_WHALE_VOLUME_MIN: int = int(os.getenv("L4_WHALE_VOLUME_MIN", "13"))
L4_WHALE_REVERSAL_MAX: int = int(os.getenv("L4_WHALE_REVERSAL_MAX", "4"))
L4_WHALE_ABSORPTION_MIN: int = int(os.getenv("L4_WHALE_ABSORPTION_MIN", "3"))

# l4_trap_score — distance threshold for order block proximity (pips)
L4_TRAP_OB_DISTANCE_PIPS: float = float(os.getenv("L4_TRAP_OB_DISTANCE_PIPS", "10"))
L4_TRAP_ABSORPTION_THRESHOLD: float = float(os.getenv("L4_TRAP_ABSORPTION_THRESHOLD", "0.4"))

# l4_candle_dna
L4_DNA_ABSORPTION_BODY_MAX: float = float(os.getenv("L4_DNA_ABSORPTION_BODY_MAX", "0.3"))
L4_DNA_EXHAUSTION_RATIO_MAX: float = float(os.getenv("L4_DNA_EXHAUSTION_RATIO_MAX", "0.35"))
L4_DNA_EXHAUSTION_ACCEL_MAX: float = float(os.getenv("L4_DNA_EXHAUSTION_ACCEL_MAX", "-0.1"))
L4_DNA_TRAP_SCORE_MIN: int = int(os.getenv("L4_DNA_TRAP_SCORE_MIN", "60"))
L4_DNA_DEAD_VOLUME_MAX: int = int(os.getenv("L4_DNA_DEAD_VOLUME_MAX", "8"))
L4_DNA_DEAD_ENERGY_MAX: float = float(os.getenv("L4_DNA_DEAD_ENERGY_MAX", "2.0"))
L4_DNA_ACCUM_OB_PIPS: float = float(os.getenv("L4_DNA_ACCUM_OB_PIPS", "15"))
L4_DNA_ACCUM_ABSORPTION_MIN: int = int(os.getenv("L4_DNA_ACCUM_ABSORPTION_MIN", "5"))

# l4_volume_climax
L4_CLIMAX_WINDOW: int = int(os.getenv("L4_CLIMAX_WINDOW", "20"))
L4_CLIMAX_SIGMA: float = float(os.getenv("L4_CLIMAX_SIGMA", "2.0"))

# l4_range_compression
L4_RANGE_WINDOW: int = int(os.getenv("L4_RANGE_WINDOW", "20"))

# l4_session_continuation
L4_SESSION_CONTINUATION_WINDOW: int = int(os.getenv("L4_SESSION_CONTINUATION_WINDOW", "200"))

# l4_time_volatility_regime
L4_REGIME_SHORT_WINDOW: int = int(os.getenv("L4_REGIME_SHORT_WINDOW", "50"))
L4_REGIME_LONG_WINDOW: int = int(os.getenv("L4_REGIME_LONG_WINDOW", "200"))
L4_REGIME_TRANSITION_BAND: float = float(os.getenv("L4_REGIME_TRANSITION_BAND", "0.1"))

# ---------------------------------------------------------------------------
# Phase 5 — Model settings
# ---------------------------------------------------------------------------
# Labeler
LABEL_TP_PIPS: int = int(os.getenv("LABEL_TP_PIPS", "150"))
LABEL_SL_PIPS: int = int(os.getenv("LABEL_SL_PIPS", "100"))
LABEL_MAX_HORIZON: int = int(os.getenv("LABEL_MAX_HORIZON", "96"))

# Train / validation / test date boundaries
TRAIN_START: str = os.getenv("TRAIN_START", "2013-01-01")
TRAIN_END: str = os.getenv("TRAIN_END", "2022-12-31")
VAL_START: str = os.getenv("VAL_START", "2023-01-01")
VAL_END: str = os.getenv("VAL_END", "2023-12-31")
TEST_START: str = os.getenv("TEST_START", "2024-01-01")
TEST_END: str = os.getenv("TEST_END", "2025-12-31")

# Walk-forward cross-validation
WF_MIN_TRAIN_YEARS: int = int(os.getenv("WF_MIN_TRAIN_YEARS", "2"))
WF_FOLD_MONTHS: int = int(os.getenv("WF_FOLD_MONTHS", "3"))
WF_MIN_FOLDS: int = int(os.getenv("WF_MIN_FOLDS", "8"))

# XGBoost hyperparameters
XGB_MAX_DEPTH: int = int(os.getenv("XGB_MAX_DEPTH", "6"))
XGB_LEARNING_RATE: float = float(os.getenv("XGB_LEARNING_RATE", "0.05"))
XGB_N_ESTIMATORS: int = int(os.getenv("XGB_N_ESTIMATORS", "1500"))
XGB_SUBSAMPLE: float = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
XGB_COLSAMPLE_BYTREE: float = float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8"))
XGB_MIN_CHILD_WEIGHT: int = int(os.getenv("XGB_MIN_CHILD_WEIGHT", "5"))
XGB_GAMMA: float = float(os.getenv("XGB_GAMMA", "1.0"))
XGB_REG_ALPHA: float = float(os.getenv("XGB_REG_ALPHA", "0.1"))
XGB_REG_LAMBDA: float = float(os.getenv("XGB_REG_LAMBDA", "1.0"))
XGB_EARLY_STOPPING: int = int(os.getenv("XGB_EARLY_STOPPING", "50"))

# Model output paths
MODEL_OUTPUT_PATH: str = os.getenv("MODEL_OUTPUT_PATH", "output/goldwolf_model.json")
FEATURE_IMPORTANCE_PATH: str = os.getenv(
    "FEATURE_IMPORTANCE_PATH", "output/feature_importance.csv"
)
TRAINING_REPORT_PATH: str = os.getenv(
    "TRAINING_REPORT_PATH", "output/training_report.txt"
)
WALKFORWARD_RESULTS_PATH: str = os.getenv(
    "WALKFORWARD_RESULTS_PATH", "output/walkforward_results.csv"
)

# Phase 4 output path
OUTPUT_PATH_PHASE4: str = os.getenv(
    "OUTPUT_PATH_PHASE4", "output/goldwolf_phase4.parquet"
)

# ---------------------------------------------------------------------------
# Phase 6+7 — Signal system settings
# ---------------------------------------------------------------------------
# MT5 symbol
MT5_SYMBOL: str = os.getenv("MT5_SYMBOL", "XAUUSD")

# Signal confidence tiers (%)
SIGNAL_TIER1_MIN: float = float(os.getenv("SIGNAL_TIER1_MIN", "60"))
SIGNAL_TIER2_MIN: float = float(os.getenv("SIGNAL_TIER2_MIN", "70"))
SIGNAL_TIER3_MIN: float = float(os.getenv("SIGNAL_TIER3_MIN", "80"))
SIGNAL_MIN_CONFIDENCE: float = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "60"))

# Cooldown period in candles (15-min bars)
SIGNAL_COOLDOWN_CANDLES: int = int(os.getenv("SIGNAL_COOLDOWN_CANDLES", "2"))

# Daily loss limit (number of SL hits before stopping for the day)
SIGNAL_DAILY_LOSS_LIMIT: int = int(os.getenv("SIGNAL_DAILY_LOSS_LIMIT", "3"))

# Signal log path
SIGNAL_LOG_PATH: str = os.getenv("SIGNAL_LOG_PATH", "output/signal_log.csv")

# Live data cache path
LIVE_CACHE_PATH: str = os.getenv("LIVE_CACHE_PATH", "data/live_cache.csv")

# Telegram message format toggle (True = fancy emoji format, False = plain)
TELEGRAM_FANCY_FORMAT: bool = os.getenv("TELEGRAM_FANCY_FORMAT", "true").lower() == "true"

# Retrain day (0=Monday … 6=Sunday)
RETRAIN_DAY: int = int(os.getenv("RETRAIN_DAY", "6"))

# MT5 credentials (from .env — never hardcoded)
MT5_LOGIN: str = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
MT5_SERVER: str = os.getenv("MT5_SERVER", "")
MT5_PATH: str = os.getenv("MT5_PATH", "")

# Telegram credentials (from .env — never hardcoded)
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
