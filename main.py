"""
GOLDWOLF — Phase 1 + 2 + 3 Pipeline Entry Point
Loads M1 and M15 data (or re-uses an existing Phase 1 parquet), computes
Layer 1, Layer 2 (Time DNA), and Layer 3 (SMC) features, saves the combined
output, and prints summary statistics.
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the Python path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    M1_DATA_PATH,
    M15_DATA_PATH,
    OUTPUT_PATH,
    OUTPUT_PATH_PHASE2_3,
)
from data.loader import load_csv
from data.processor import group_m1_by_m15
from features.layer1 import compute_layer1_features
from features.layer2 import compute_layer2_features
from features.layer3 import compute_layer3_features
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

# All expected feature column names (L1 + L2 + L3)
FEATURE_COLS = [
    # Layer 1
    "l1_custom_volume",
    "l1_volatility_energy",
    "l1_price_velocity",
    "l1_reversal_count",
    "l1_early_late_ratio",
    "l1_price_acceleration",
    "l1_absorption_count",
    "l1_absorption_intensity",
    # Layer 2
    "l2_session",
    "l2_session_overlap",
    "l2_kill_zone",
    "l2_day_of_week",
    "l2_hour",
    "l2_distance_from_session_open",
    "l2_session_position",
    "l2_time_since_vol_spike",
    "l2_session_volatility_rank",
    "l2_session_trend",
    # Layer 3
    "l3_swing_high",
    "l3_swing_low",
    "l3_bos_direction",
    "l3_choch_flag",
    "l3_demand_ob_distance",
    "l3_supply_ob_distance",
    "l3_fvg_active",
    "l3_fvg_distance",
    "l3_buy_liq_distance",
    "l3_sell_liq_distance",
    "l3_liquidity_sweep",
    "l3_premium_discount",
    "l3_structure_trend",
    "l3_h1_trend",
    "l3_h4_trend",
]


def _load_or_compute_phase1() -> pd.DataFrame:
    """
    Return the Phase 1 DataFrame.

    If the Phase 1 parquet already exists, load it directly.  Otherwise
    run the full Phase 1 pipeline (load CSVs → group → compute features).
    """
    phase1_path = Path(OUTPUT_PATH)
    if phase1_path.exists():
        logger.info("Phase 1 parquet found at %s — loading directly.", phase1_path)
        return pd.read_parquet(phase1_path)

    logger.info("Phase 1 parquet not found — running Phase 1 pipeline …")
    m1_df = load_csv(M1_DATA_PATH, timeframe_label="M1")
    m15_df = load_csv(M15_DATA_PATH, timeframe_label="M15")
    grouped = group_m1_by_m15(m1_df, m15_df)
    return compute_layer1_features(grouped)


def main() -> None:
    logger.info("=" * 60)
    logger.info("GOLDWOLF — Phase 2 + 3 Pipeline")
    logger.info("=" * 60)

    with Timer("total pipeline") as total_timer:
        # ------------------------------------------------------------------
        # 1. Load / compute Phase 1 output
        # ------------------------------------------------------------------
        result = _load_or_compute_phase1()
        logger.info("Phase 1 data: %d M15 bars, %d columns", len(result), len(result.columns))

        # ------------------------------------------------------------------
        # 2. Compute Layer 2 (Time DNA) features
        # ------------------------------------------------------------------
        result = compute_layer2_features(result)

        # ------------------------------------------------------------------
        # 3. Compute Layer 3 (SMC) features
        # ------------------------------------------------------------------
        result = compute_layer3_features(result)

        # ------------------------------------------------------------------
        # 4. Save combined output
        # ------------------------------------------------------------------
        output_path = Path(OUTPUT_PATH_PHASE2_3)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".parquet":
            result.to_parquet(output_path, index=True)
        else:
            result.to_csv(output_path, index=True)

        logger.info("Output saved to: %s  (%d rows)", output_path, len(result))

        # ------------------------------------------------------------------
        # 5. Summary statistics for ALL features (L1 + L2 + L3)
        # ------------------------------------------------------------------
        logger.info("\n%s", "=" * 60)
        logger.info("Feature Summary Statistics (L1 + L2 + L3)")
        logger.info("%s", "=" * 60)

        available_cols = [c for c in FEATURE_COLS if c in result.columns]
        if available_cols:
            stats: pd.DataFrame = result[available_cols].describe().T[
                ["mean", "std", "min", "max"]
            ]
            logger.info("\n%s", stats.to_string())

    logger.info("Total pipeline time: %s", total_timer.elapsed_str)


if __name__ == "__main__":
    main()
