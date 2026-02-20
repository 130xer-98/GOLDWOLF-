"""
GOLDWOLF — Phase 1 Pipeline Entry Point
Loads M1 and M15 data, groups M1 bars into M15 windows, computes Layer 1
features, saves the result, and prints summary statistics.
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the Python path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import M1_DATA_PATH, M15_DATA_PATH, OUTPUT_PATH
from data.loader import load_csv
from data.processor import group_m1_by_m15
from features.layer1 import compute_layer1_features
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

FEATURE_COLS = [
    "l1_custom_volume",
    "l1_volatility_energy",
    "l1_price_velocity",
    "l1_reversal_count",
    "l1_early_late_ratio",
    "l1_price_acceleration",
    "l1_absorption_count",
    "l1_absorption_intensity",
]


def main() -> None:
    logger.info("=" * 60)
    logger.info("GOLDWOLF — Phase 1 Data Engine")
    logger.info("=" * 60)

    with Timer("total pipeline") as total_timer:
        # ------------------------------------------------------------------
        # 1. Load data
        # ------------------------------------------------------------------
        m1_df = load_csv(M1_DATA_PATH, timeframe_label="M1")
        m15_df = load_csv(M15_DATA_PATH, timeframe_label="M15")

        # ------------------------------------------------------------------
        # 2. Group M1 candles into M15 windows
        # ------------------------------------------------------------------
        grouped = group_m1_by_m15(m1_df, m15_df)

        # ------------------------------------------------------------------
        # 3. Compute Layer 1 features
        # ------------------------------------------------------------------
        result = compute_layer1_features(grouped)

        # ------------------------------------------------------------------
        # 4. Save output
        # ------------------------------------------------------------------
        output_path = Path(OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".parquet":
            result.to_parquet(output_path, index=True)
        else:
            result.to_csv(output_path, index=True)

        logger.info("Output saved to: %s  (%d rows)", output_path, len(result))

        # ------------------------------------------------------------------
        # 5. Summary statistics
        # ------------------------------------------------------------------
        logger.info("\n%s", "=" * 60)
        logger.info("Layer 1 Feature Summary Statistics")
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
