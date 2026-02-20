"""
GOLDWOLF — Model Evaluator
Confusion matrix, trading simulation, and performance metrics for backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

# Class label mapping for display
LABEL_NAMES = {0: "NO_TRADE", 1: "BUY", 2: "SELL"}
# Internal mapping: label (-1, 0, 1) → XGBoost class (0, 1, 2)
LABEL_TO_CLASS = {0: 0, 1: 1, -1: 2}
CLASS_TO_LABEL = {0: 0, 1: 1, 2: -1}


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Compute and return a confusion matrix as a DataFrame.

    Parameters
    ----------
    y_true : np.ndarray  True labels (XGBoost class encoding 0/1/2).
    y_pred : np.ndarray  Predicted classes (XGBoost class encoding 0/1/2).

    Returns
    -------
    pd.DataFrame  3×3 confusion matrix.
    """
    from sklearn.metrics import confusion_matrix
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    names = [LABEL_NAMES[c] for c in labels]
    return pd.DataFrame(cm, index=names, columns=names)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """
    Return sklearn classification report as a string.
    """
    from sklearn.metrics import classification_report
    target_names = [LABEL_NAMES[c] for c in [0, 1, 2]]
    return classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=target_names, zero_division=0
    )


def trading_simulation(
    df_test: pd.DataFrame,
    probabilities: np.ndarray,
    tp_pips: int,
    sl_pips: int,
    confidence_threshold: float = 60.0,
) -> dict[str, float | int]:
    """
    Simulate trades on the test set.

    Parameters
    ----------
    df_test           : pd.DataFrame with 'label' column (true labels, -1/0/1).
    probabilities     : np.ndarray shape (n, 3) — P(NO_TRADE), P(BUY), P(SELL).
    tp_pips           : Take-profit in pips.
    sl_pips           : Stop-loss in pips.
    confidence_threshold : Minimum confidence (%) to take a signal.

    Returns
    -------
    dict with metrics: total_trades, win_rate, total_pips_won, total_pips_lost,
                       profit_factor, max_drawdown, sharpe_ratio.
    """
    true_labels = df_test["label"].values if "label" in df_test.columns else None

    preds = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(preds)), preds] * 100

    # Only act on BUY (class 1) or SELL (class 2) with sufficient confidence
    act_mask = (preds != 0) & (confidence >= confidence_threshold)

    if not act_mask.any():
        logger.info("No signals met confidence threshold of %.0f%%", confidence_threshold)
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pips_won": 0.0,
            "total_pips_lost": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    # Map predictions back to trade direction (-1/1)
    pred_dirs = np.where(preds == 1, 1, -1)  # 1=BUY, 2=SELL(-1)

    pnl_series: list[float] = []
    wins = 0
    total_won = 0.0
    total_lost = 0.0

    for i in range(len(df_test)):
        if not act_mask[i]:
            continue
        if true_labels is None:
            continue

        direction = pred_dirs[i]
        true_label = int(true_labels[i])

        if true_label == 0:
            # No trade hit — neutral outcome, slight negative (spread cost)
            pnl = -1.0
        elif true_label == direction:
            pnl = float(tp_pips)
            wins += 1
            total_won += pnl
        else:
            pnl = -float(sl_pips)
            total_lost += abs(pnl)

        pnl_series.append(pnl)

    total_trades = len(pnl_series)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pips_won": total_won,
            "total_pips_lost": total_lost,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    win_rate = wins / total_trades
    profit_factor = total_won / max(total_lost, 1e-10)

    # Max drawdown
    cumulative = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    # Sharpe ratio (annualised, assuming ~96 M15 bars per day)
    pnl_arr = np.array(pnl_series)
    bars_per_year = 96 * 252
    sharpe = 0.0
    if pnl_arr.std() > 0:
        sharpe = float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(bars_per_year))

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "total_pips_won": round(total_won, 2),
        "total_pips_lost": round(total_lost, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 4),
    }


def tier_analysis(
    df_test: pd.DataFrame,
    probabilities: np.ndarray,
    tp_pips: int,
    sl_pips: int,
    tier_thresholds: tuple[float, float, float] = (60.0, 70.0, 80.0),
) -> pd.DataFrame:
    """
    Analyse performance broken down by confidence tier.

    Parameters
    ----------
    df_test          : pd.DataFrame with 'label' column.
    probabilities    : np.ndarray shape (n, 3).
    tp_pips          : Take-profit pips.
    sl_pips          : Stop-loss pips.
    tier_thresholds  : (T1_min, T2_min, T3_min) confidence thresholds.

    Returns
    -------
    pd.DataFrame with one row per tier showing performance metrics.
    """
    t1, t2, t3 = tier_thresholds
    tiers = [
        (f"Tier1 ({t1:.0f}-{t2 - 1:.0f}%)", t1, t2),
        (f"Tier2 ({t2:.0f}-{t3 - 1:.0f}%)", t2, t3),
        (f"Tier3 ({t3:.0f}%+)", t3, 101.0),
    ]

    rows = []
    for name, lo, hi in tiers:
        preds = np.argmax(probabilities, axis=1)
        confidence = probabilities[np.arange(len(preds)), preds] * 100
        mask = (preds != 0) & (confidence >= lo) & (confidence < hi)
        tier_probs = np.zeros_like(probabilities)
        tier_probs[mask] = probabilities[mask]
        # Create a sub-dataframe for this tier
        tier_df = df_test.copy()
        tier_df_mask = pd.Series(mask, index=df_test.index)
        # Zero out non-tier predictions
        tier_probs_full = probabilities.copy()
        tier_probs_full[~mask] = [1.0, 0.0, 0.0]  # force NO_TRADE for non-tier
        stats = trading_simulation(tier_df, tier_probs_full, tp_pips, sl_pips, lo)
        rows.append({"tier": name, **stats})

    return pd.DataFrame(rows).set_index("tier")
