"""
GOLDWOLF — Pipeline Entry Point
Supports multiple modes via command-line arguments:

  python main.py            — run full feature pipeline (L1 → L4)
  python main.py --train    — features + train XGBoost model
  python main.py --live     — run live signal loop
  python main.py --backtest — run backtest on test period
  python main.py --retrain  — retrain model with latest data
"""

import argparse
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
    OUTPUT_PATH_PHASE4,
)
from data.loader import load_csv
from data.processor import group_m1_by_m15
from features.layer1 import compute_layer1_features
from features.layer2 import compute_layer2_features
from features.layer3 import compute_layer3_features
from features.layer4 import compute_layer4_features
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

# All expected feature column names (L1 + L2 + L3 + L4)
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
    # Layer 4
    "l4_whale_footprint",
    "l4_trap_score",
    "l4_candle_dna",
    "l4_momentum_divergence",
    "l4_consecutive_bias",
    "l4_volume_climax",
    "l4_range_compression",
    "l4_session_continuation",
    "l4_multi_layer_confluence",
    "l4_time_volatility_regime",
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


def run_pipeline() -> pd.DataFrame:
    """Run full feature pipeline (L1 → L4) and save output."""
    logger.info("=" * 60)
    logger.info("GOLDWOLF — Feature Pipeline (Phases 1-4)")
    logger.info("=" * 60)

    with Timer("total pipeline") as total_timer:
        # Phase 1
        result = _load_or_compute_phase1()
        logger.info("Phase 1 data: %d M15 bars, %d columns", len(result), len(result.columns))

        # Phase 2 — Time DNA
        result = compute_layer2_features(result)

        # Phase 3 — SMC
        result = compute_layer3_features(result)

        # Save Phase 2+3 output
        output_path_23 = Path(OUTPUT_PATH_PHASE2_3)
        output_path_23.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path_23, index=True)
        logger.info("Phase 2+3 output saved to: %s  (%d rows)", output_path_23, len(result))

        # Phase 4 — Private Edge Features
        result = compute_layer4_features(result)

        # Save Phase 4 output
        output_path_4 = Path(OUTPUT_PATH_PHASE4)
        output_path_4.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path_4, index=True)
        logger.info("Phase 4 output saved to: %s  (%d rows)", output_path_4, len(result))

        # Summary statistics
        logger.info("\n%s", "=" * 60)
        logger.info("Feature Summary Statistics (L1-L4)")
        logger.info("%s", "=" * 60)
        available_cols = [c for c in FEATURE_COLS if c in result.columns]
        if available_cols:
            stats: pd.DataFrame = result[available_cols].describe().T[
                ["mean", "std", "min", "max"]
            ]
            logger.info("\n%s", stats.to_string())

    logger.info("Total pipeline time: %s", total_timer.elapsed_str)
    return result


def run_train() -> None:
    """Run feature pipeline + train XGBoost model."""
    result = run_pipeline()
    from model.trainer import train_model
    train_model(df=result)


def run_live() -> None:
    """Start the live trading signal loop."""
    from live.runner import run_live as _run_live
    _run_live()


def run_backtest() -> None:
    """Run backtest on the test period."""
    from config.settings import TEST_START, TEST_END, LABEL_TP_PIPS, LABEL_SL_PIPS
    import xgboost as xgb
    from model.labeler import create_labels
    from model.evaluator import (
        trading_simulation,
        compute_confusion_matrix,
        compute_classification_report,
        tier_analysis,
        LABEL_TO_CLASS,
    )
    from model.trainer import _get_feature_cols

    logger.info("=" * 60)
    logger.info("GOLDWOLF — Backtest")
    logger.info("=" * 60)

    # Load Phase 4 output
    path = Path(OUTPUT_PATH_PHASE4)
    if not path.exists():
        path = Path(OUTPUT_PATH_PHASE2_3)
    logger.info("Loading data from %s …", path)
    df = pd.read_parquet(path)

    if "l4_whale_footprint" not in df.columns:
        df = compute_layer4_features(df)

    # Create labels
    labels = create_labels(df, LABEL_TP_PIPS, LABEL_SL_PIPS)
    df = df.copy()
    df["label"] = labels
    df["label_class"] = df["label"].map(LABEL_TO_CLASS)

    # Test period
    test_df = df.loc[TEST_START:TEST_END]
    logger.info("Test period: %d rows", len(test_df))

    # Load model
    from config.settings import MODEL_OUTPUT_PATH
    model = xgb.Booster()
    model.load_model(MODEL_OUTPUT_PATH)

    feature_cols = _get_feature_cols(df)
    X_test = test_df[feature_cols].fillna(0).values
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    probs = model.predict(dtest)

    import numpy as np
    y_pred = np.argmax(probs, axis=1)
    y_test = test_df["label_class"].values

    cm = compute_confusion_matrix(y_test, y_pred)
    report = compute_classification_report(y_test, y_pred)
    logger.info("\nConfusion Matrix:\n%s", cm.to_string())
    logger.info("\nClassification Report:\n%s", report)

    sim = trading_simulation(test_df, probs, LABEL_TP_PIPS, LABEL_SL_PIPS)
    logger.info("Trading simulation: %s", sim)


def run_retrain() -> None:
    """Retrain the model with latest data."""
    from live.retrainer import retrain
    retrain()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GOLDWOLF Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  (no args)   Run full feature pipeline (L1-L4)\n"
            "  --train     Feature pipeline + train XGBoost model\n"
            "  --live      Live signal loop (requires MT5)\n"
            "  --backtest  Backtest on test period (requires trained model)\n"
            "  --retrain   Retrain model with latest data\n"
        ),
    )
    parser.add_argument("--train", action="store_true", help="Run features + train model")
    parser.add_argument("--live", action="store_true", help="Run live signal loop")
    parser.add_argument("--backtest", action="store_true", help="Backtest on test period")
    parser.add_argument("--retrain", action="store_true", help="Retrain model with latest data")

    args = parser.parse_args()

    if args.train:
        run_train()
    elif args.live:
        run_live()
    elif args.backtest:
        run_backtest()
    elif args.retrain:
        run_retrain()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
