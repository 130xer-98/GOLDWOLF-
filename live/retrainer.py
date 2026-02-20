"""
GOLDWOLF — Weekly Retrainer
Appends new data, retrains the model, and only replaces the existing model
if the new model has a higher profit factor on the last month's data.

Usage:
  python main.py --retrain
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    MODEL_OUTPUT_PATH,
    OUTPUT_PATH_PHASE2_3,
    LABEL_TP_PIPS,
    LABEL_SL_PIPS,
    LABEL_MAX_HORIZON,
    SIGNAL_MIN_CONFIDENCE,
)
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)


def _backup_model(model_path: str) -> str | None:
    """Create a timestamped backup of the existing model."""
    p = Path(model_path)
    if not p.exists():
        return None
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = p.with_stem(f"{p.stem}_backup_{ts}")
    shutil.copy2(p, backup_path)
    logger.info("Model backup saved to: %s", backup_path)
    return str(backup_path)


def _evaluate_model_profit_factor(
    model_path: str,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
) -> float:
    """
    Evaluate a saved model on *df_eval* and return its profit factor.

    Returns 0.0 if the model cannot be loaded or evaluation fails.
    """
    try:
        import xgboost as xgb
        from model.evaluator import trading_simulation

        model = xgb.Booster()
        model.load_model(model_path)

        X = df_eval[feature_cols].fillna(0).values
        dmat = xgb.DMatrix(X, feature_names=feature_cols)
        probs = model.predict(dmat)

        result = trading_simulation(
            df_eval, probs, LABEL_TP_PIPS, LABEL_SL_PIPS,
            confidence_threshold=SIGNAL_MIN_CONFIDENCE,
        )
        return float(result.get("profit_factor", 0.0))

    except Exception as exc:
        logger.warning("Failed to evaluate model at %s: %s", model_path, exc)
        return 0.0


def retrain(
    new_data_path: str | None = None,
) -> None:
    """
    Retrain the model on expanded dataset.

    Steps:
      1. Load existing Phase 4 data (or Phase 2+3 fallback).
      2. Append new data if provided.
      3. Retrain using model/trainer.train_model().
      4. Compare new model vs old model on last month's data.
      5. Replace old model only if new model's profit factor is higher.
      6. Log comparison results.

    Parameters
    ----------
    new_data_path : Optional path to new data parquet to append.
    """
    from model.trainer import train_model, _get_feature_cols

    logger.info("=" * 60)
    logger.info("GOLDWOLF — Weekly Retrainer")
    logger.info("=" * 60)

    with Timer("retraining pipeline") as t:
        # --- Load base data ---
        from config.settings import OUTPUT_PATH_PHASE4
        base_path = OUTPUT_PATH_PHASE4
        if not Path(base_path).exists():
            base_path = OUTPUT_PATH_PHASE2_3

        logger.info("Loading base data from %s …", base_path)
        df = pd.read_parquet(base_path)

        # Compute L4 features if not present
        if "l4_whale_footprint" not in df.columns:
            from features.layer4 import compute_layer4_features
            df = compute_layer4_features(df)

        # --- Append new data if provided ---
        if new_data_path and Path(new_data_path).exists():
            logger.info("Appending new data from %s …", new_data_path)
            new_df = pd.read_parquet(new_data_path)
            if "l4_whale_footprint" not in new_df.columns:
                from features.layer4 import compute_layer4_features
                new_df = compute_layer4_features(new_df)
            df = pd.concat([df, new_df]).drop_duplicates()
            df = df.sort_index()
            logger.info("Combined data: %d rows", len(df))

        # --- Create labels ---
        from model.labeler import create_labels
        labels = create_labels(df, LABEL_TP_PIPS, LABEL_SL_PIPS, LABEL_MAX_HORIZON)
        df = df.copy()
        df["label"] = labels

        # --- Identify evaluation window (last month) ---
        last_ts = df.index.max()
        eval_start = last_ts - pd.DateOffset(months=1)
        df_eval = df.loc[eval_start:]
        logger.info(
            "Evaluation window: %s → %s (%d rows)",
            eval_start.date(), last_ts.date(), len(df_eval),
        )

        # --- Evaluate old model ---
        feature_cols = _get_feature_cols(df)
        old_pf = 0.0
        if Path(MODEL_OUTPUT_PATH).exists():
            old_pf = _evaluate_model_profit_factor(MODEL_OUTPUT_PATH, df_eval, feature_cols)
            logger.info("Old model profit factor (last month): %.4f", old_pf)

        # --- Backup old model ---
        backup_path = _backup_model(MODEL_OUTPUT_PATH)

        # --- Train new model ---
        new_model_path = MODEL_OUTPUT_PATH + ".new"
        import os
        os.environ["_RETRAIN_OUTPUT"] = new_model_path  # temporary override
        train_model(df=df)

        # --- Evaluate new model ---
        new_pf = _evaluate_model_profit_factor(MODEL_OUTPUT_PATH, df_eval, feature_cols)
        logger.info("New model profit factor (last month): %.4f", new_pf)

        # --- Compare and decide ---
        if new_pf >= old_pf:
            logger.info(
                "New model is better (%.4f >= %.4f) — keeping new model.",
                new_pf, old_pf,
            )
        else:
            logger.warning(
                "New model is worse (%.4f < %.4f) — restoring backup.",
                new_pf, old_pf,
            )
            if backup_path and Path(backup_path).exists():
                shutil.copy2(backup_path, MODEL_OUTPUT_PATH)
                logger.info("Old model restored from backup.")

        # --- Log comparison ---
        comparison = pd.DataFrame([
            {"model": "old", "profit_factor": old_pf},
            {"model": "new", "profit_factor": new_pf},
        ])
        logger.info("\nModel comparison:\n%s", comparison.to_string(index=False))

    logger.info("Retraining completed in %s", t.elapsed_str)
