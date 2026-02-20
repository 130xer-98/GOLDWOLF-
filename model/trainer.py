"""
GOLDWOLF — XGBoost Model Trainer
Full training pipeline: feature loading, labeling, walk-forward CV, training,
evaluation, and model persistence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    OUTPUT_PATH_PHASE2_3,
    OUTPUT_PATH_PHASE4,
    MODEL_OUTPUT_PATH,
    FEATURE_IMPORTANCE_PATH,
    TRAINING_REPORT_PATH,
    WALKFORWARD_RESULTS_PATH,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
    WF_MIN_TRAIN_YEARS,
    WF_FOLD_MONTHS,
    WF_MIN_FOLDS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_MIN_CHILD_WEIGHT,
    XGB_GAMMA,
    XGB_REG_ALPHA,
    XGB_REG_LAMBDA,
    XGB_EARLY_STOPPING,
    LABEL_TP_PIPS,
    LABEL_SL_PIPS,
    LABEL_MAX_HORIZON,
)
from model.labeler import create_labels
from model.evaluator import (
    compute_confusion_matrix,
    compute_classification_report,
    trading_simulation,
    tier_analysis,
    LABEL_TO_CLASS,
)
from utils.helpers import get_logger, Timer

logger = get_logger(__name__)

# Feature column prefixes to include in training
FEATURE_PREFIXES = ("l1_", "l2_", "l3_", "l4_")

# Exclude the m1_count meta-column (not a tradeable feature)
EXCLUDE_COLS = {"l1_m1_count"}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (l1_* l2_* l3_* l4_*) from *df*."""
    return [
        c for c in df.columns
        if any(c.startswith(p) for p in FEATURE_PREFIXES) and c not in EXCLUDE_COLS
    ]


def _remove_highly_correlated(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """
    Remove features with Pearson correlation > threshold.
    Returns the filtered list of feature columns.
    """
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    if to_drop:
        logger.info("Removing %d highly correlated features: %s", len(to_drop), to_drop)
    return [c for c in feature_cols if c not in to_drop]


def _detect_gpu() -> str:
    """
    Detect if GPU is available for XGBoost.
    Returns 'hist' (GPU) or 'hist' (CPU fallback) tree method string.
    On modern XGBoost (>=2.0), device='cuda' is preferred; falls back gracefully.
    """
    try:
        import xgboost as xgb
        # Create a tiny test to check GPU availability
        test_data = xgb.DMatrix(np.zeros((10, 2)), label=np.zeros(10))
        params = {"tree_method": "hist", "device": "cuda", "verbosity": 0}
        xgb.train(params, test_data, num_boost_round=1, verbose_eval=False)
        logger.info("GPU detected — using device=cuda")
        return "cuda"
    except Exception:
        logger.info("GPU not available — using CPU (device=cpu)")
        return "cpu"


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute sample weights inversely proportional to class frequency."""
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes.tolist(), counts.tolist()))
    total = len(y)
    weights = np.array([total / (len(classes) * freq[c]) for c in y])
    return weights


def _time_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/val/test by date (time-based, no data leakage)."""
    train = df.loc[train_start:train_end]
    val = df.loc[val_start:val_end]
    test = df.loc[test_start:test_end]
    logger.info(
        "Split — Train: %d rows, Val: %d rows, Test: %d rows",
        len(train), len(val), len(test),
    )
    return train, val, test


def _walk_forward_cv(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    xgb_params: dict,
    min_train_years: int = WF_MIN_TRAIN_YEARS,
    fold_months: int = WF_FOLD_MONTHS,
    min_folds: int = WF_MIN_FOLDS,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward cross-validation.

    Start with *min_train_years* of training data, validate on next
    *fold_months* months, then slide forward.

    Returns a DataFrame with per-fold results.
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score

    results = []
    start_date = df_train.index.min()
    end_date = df_train.index.max()

    # Generate fold boundaries
    fold_start = start_date + pd.DateOffset(years=min_train_years)
    fold_num = 0

    while fold_start < end_date:
        fold_end = fold_start + pd.DateOffset(months=fold_months)

        train_fold = df_train.loc[:fold_start - pd.Timedelta(minutes=15)]
        val_fold = df_train.loc[fold_start:fold_end]

        if len(train_fold) < 100 or len(val_fold) < 10:
            fold_start = fold_end
            continue

        X_train = train_fold[feature_cols].fillna(0).values
        y_train = train_fold[label_col].values
        X_val = val_fold[feature_cols].fillna(0).values
        y_val = val_fold[label_col].values

        sw = _compute_sample_weights(y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals_result: dict = {}
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=xgb_params.get("n_estimators", XGB_N_ESTIMATORS),
            evals=[(dval, "val")],
            early_stopping_rounds=XGB_EARLY_STOPPING,
            evals_result=evals_result,
            verbose_eval=False,
        )

        y_pred = np.argmax(model.predict(dval), axis=1)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        val_loss = evals_result.get("val", {}).get("mlogloss", [None])[-1]

        results.append({
            "fold": fold_num,
            "train_start": str(train_fold.index.min().date()),
            "train_end": str(train_fold.index.max().date()),
            "val_start": str(val_fold.index.min().date()),
            "val_end": str(val_fold.index.max().date()),
            "train_rows": len(train_fold),
            "val_rows": len(val_fold),
            "val_accuracy": round(acc, 4),
            "val_f1_weighted": round(f1, 4),
            "val_mlogloss": round(val_loss, 6) if val_loss is not None else None,
            "best_iteration": model.best_iteration,
        })
        logger.info(
            "Fold %d: acc=%.4f, f1=%.4f, loss=%.4f, best_iter=%d",
            fold_num, acc, f1,
            val_loss if val_loss is not None else 0,
            model.best_iteration,
        )

        fold_start = fold_end
        fold_num += 1

        if fold_num >= min_folds and fold_end >= end_date:
            break

    logger.info("Walk-forward CV: %d folds completed", len(results))
    return pd.DataFrame(results)


def train_model(
    df: pd.DataFrame | None = None,
    phase4_output_path: str | None = None,
) -> None:
    """
    Full training pipeline.

    1. Load or use provided DataFrame (Phase 4 output).
    2. Create labels.
    3. Remove highly correlated features.
    4. Time-based train/val/test split.
    5. Walk-forward cross-validation.
    6. Train final model on full training set.
    7. Evaluate on test set.
    8. Save model, feature importance, training report.

    Parameters
    ----------
    df                 : Pre-loaded DataFrame (Phase 4 output). If None, loads from disk.
    phase4_output_path : Path to Phase 4 parquet. Uses OUTPUT_PATH_PHASE4 if None.
    """
    import xgboost as xgb

    logger.info("=" * 60)
    logger.info("GOLDWOLF — XGBoost Training Pipeline")
    logger.info("=" * 60)

    with Timer("full training pipeline") as total_timer:
        # -------------------------------------------------------------------
        # 1. Load data
        # -------------------------------------------------------------------
        if df is None:
            path = phase4_output_path or OUTPUT_PATH_PHASE4
            if not Path(path).exists():
                # Fall back to Phase 2+3 output
                path = OUTPUT_PATH_PHASE2_3
            logger.info("Loading data from %s …", path)
            df = pd.read_parquet(path)

        logger.info("Data loaded: %d rows, %d columns", len(df), len(df.columns))

        # -------------------------------------------------------------------
        # 2. Compute Layer 4 if not already present
        # -------------------------------------------------------------------
        if "l4_whale_footprint" not in df.columns:
            from features.layer4 import compute_layer4_features
            df = compute_layer4_features(df)

        # -------------------------------------------------------------------
        # 3. Create labels
        # -------------------------------------------------------------------
        labels = create_labels(df, LABEL_TP_PIPS, LABEL_SL_PIPS, LABEL_MAX_HORIZON)
        df = df.copy()
        df["label"] = labels

        # -------------------------------------------------------------------
        # 4. Feature selection
        # -------------------------------------------------------------------
        feature_cols = _get_feature_cols(df)
        logger.info("Initial feature count: %d", len(feature_cols))

        feature_cols = _remove_highly_correlated(df, feature_cols)
        logger.info("Features after correlation removal: %d", len(feature_cols))

        # Map labels to XGBoost classes (0, 1, 2)
        df["label_class"] = df["label"].map(LABEL_TO_CLASS)

        # -------------------------------------------------------------------
        # 5. Time-based split
        # -------------------------------------------------------------------
        train_df, val_df, test_df = _time_split(
            df, TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END
        )

        # -------------------------------------------------------------------
        # 6. Detect GPU
        # -------------------------------------------------------------------
        device = _detect_gpu()

        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "device": device,
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "n_estimators": XGB_N_ESTIMATORS,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "min_child_weight": XGB_MIN_CHILD_WEIGHT,
            "gamma": XGB_GAMMA,
            "reg_alpha": XGB_REG_ALPHA,
            "reg_lambda": XGB_REG_LAMBDA,
            "eval_metric": "mlogloss",
            "verbosity": 1,
        }

        # -------------------------------------------------------------------
        # 7. Walk-forward cross-validation
        # -------------------------------------------------------------------
        logger.info("Running walk-forward cross-validation …")
        wf_results = _walk_forward_cv(
            train_df, feature_cols, "label_class", xgb_params
        )

        # -------------------------------------------------------------------
        # 8. Train final model on full train set, evaluate on val set
        # -------------------------------------------------------------------
        logger.info("Training final model …")
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["label_class"].values
        X_val = val_df[feature_cols].fillna(0).values
        y_val = val_df["label_class"].values

        sw_train = _compute_sample_weights(y_train)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train,
                              feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        evals_result: dict = {}
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=XGB_N_ESTIMATORS,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=XGB_EARLY_STOPPING,
            evals_result=evals_result,
            verbose_eval=100,
        )
        logger.info("Best iteration: %d", model.best_iteration)

        # -------------------------------------------------------------------
        # 9. Evaluate on test set
        # -------------------------------------------------------------------
        X_test = test_df[feature_cols].fillna(0).values
        dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
        probs = model.predict(dtest)  # shape (n, 3)
        y_pred = np.argmax(probs, axis=1)
        y_test = test_df["label_class"].values

        cm = compute_confusion_matrix(y_test, y_pred)
        report = compute_classification_report(y_test, y_pred)
        logger.info("\nConfusion Matrix:\n%s", cm.to_string())
        logger.info("\nClassification Report:\n%s", report)

        # Trading simulation
        test_df_with_labels = test_df.copy()
        test_df_with_labels["label"] = test_df["label"].values
        sim_results = trading_simulation(
            test_df_with_labels, probs, LABEL_TP_PIPS, LABEL_SL_PIPS
        )
        logger.info("Trading Simulation: %s", sim_results)

        # Tier analysis
        tiers = tier_analysis(
            test_df_with_labels, probs, LABEL_TP_PIPS, LABEL_SL_PIPS
        )
        logger.info("Tier Analysis:\n%s", tiers.to_string())

        # Feature importance (top 20)
        fi = model.get_score(importance_type="gain")
        fi_series = pd.Series(fi).sort_values(ascending=False)
        logger.info("Top 20 features by gain:\n%s", fi_series.head(20).to_string())

        # -------------------------------------------------------------------
        # 10. Save outputs
        # -------------------------------------------------------------------
        output_dir = Path(MODEL_OUTPUT_PATH).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        model.save_model(MODEL_OUTPUT_PATH)
        logger.info("Model saved to: %s", MODEL_OUTPUT_PATH)

        fi_df = fi_series.reset_index()
        fi_df.columns = ["feature", "gain"]
        fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        logger.info("Feature importance saved to: %s", FEATURE_IMPORTANCE_PATH)

        wf_results.to_csv(WALKFORWARD_RESULTS_PATH, index=False)
        logger.info("Walk-forward results saved to: %s", WALKFORWARD_RESULTS_PATH)

        # Training report
        report_lines = [
            "GOLDWOLF — Training Report",
            "=" * 60,
            f"Data rows: {len(df)}",
            f"Feature count: {len(feature_cols)}",
            f"Train rows: {len(train_df)}",
            f"Val rows: {len(val_df)}",
            f"Test rows: {len(test_df)}",
            f"Best iteration: {model.best_iteration}",
            "",
            "Classification Report:",
            report,
            "",
            "Confusion Matrix:",
            cm.to_string(),
            "",
            "Trading Simulation:",
            str(sim_results),
            "",
            "Tier Analysis:",
            tiers.to_string(),
            "",
            "Walk-forward CV summary:",
            wf_results.to_string(),
        ]
        with open(TRAINING_REPORT_PATH, "w") as f:
            f.write("\n".join(report_lines))
        logger.info("Training report saved to: %s", TRAINING_REPORT_PATH)

    logger.info("Total training pipeline time: %s", total_timer.elapsed_str)
