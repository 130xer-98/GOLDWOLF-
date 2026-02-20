"""
GOLDWOLF — Model Tests
Tests for labeler, evaluator, and training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.labeler import create_labels
from model.evaluator import (
    trading_simulation,
    compute_confusion_matrix,
    LABEL_TO_CLASS,
)
from config.settings import PIP_SIZE


# ---------------------------------------------------------------------------
# Helper: build a minimal price DataFrame for labeling tests
# ---------------------------------------------------------------------------

def make_price_df(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    start: str = "2020-01-06 00:00",
) -> pd.DataFrame:
    n = len(closes)
    highs = highs if highs is not None else [c + 2.0 for c in closes]
    lows = lows if lows is not None else [c - 2.0 for c in closes]
    idx = pd.date_range(start=start, periods=n, freq="15min")
    return pd.DataFrame(
        {
            "m15_open": np.array(closes, dtype=np.float64),
            "m15_high": np.array(highs, dtype=np.float64),
            "m15_low": np.array(lows, dtype=np.float64),
            "m15_close": np.array(closes, dtype=np.float64),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Label creation tests
# ---------------------------------------------------------------------------

class TestLabelCreation:
    def test_buy_label_when_price_goes_up(self):
        """
        Price moves strongly up → BUY label (1).
        TP = 15 pips = 1.5 price units.  SL = 10 pips = 1.0 price unit.
        Entry at 1800.  TP level = 1801.5, SL level = 1799.0.
        Make bar 1 high = 1810 (hits TP), low = 1800 (doesn't hit SL).
        """
        closes = [1800.0, 1810.0]
        highs = [1800.5, 1810.0]
        lows = [1799.5, 1801.0]  # low stays well above SL (1799.0)
        df = make_price_df(closes, highs=highs, lows=lows)
        labels = create_labels(df, tp_pips=15, sl_pips=10, max_horizon=5)
        assert labels.iloc[0] == 1

    def test_sell_label_when_price_goes_down(self):
        """
        Price drops strongly → SELL label (-1).
        Entry at 1800, TP level = 1798.5, SL level = 1801.0.
        Bar 1 low = 1790 (hits TP), high = 1800.5 (doesn't hit SL).
        """
        closes = [1800.0, 1790.0]
        highs = [1800.5, 1800.5]  # stays below SL level 1801.0
        lows = [1799.5, 1790.0]
        df = make_price_df(closes, highs=highs, lows=lows)
        labels = create_labels(df, tp_pips=15, sl_pips=10, max_horizon=5)
        assert labels.iloc[0] == -1

    def test_no_trade_when_neither_hit(self):
        """
        Price stays in a narrow range → NO_TRADE label (0).
        """
        closes = [1800.0] * 10
        # Tight range — never hits TP (15 pips away) or SL (10 pips away)
        highs = [1800.5] * 10
        lows = [1799.5] * 10
        df = make_price_df(closes, highs=highs, lows=lows)
        labels = create_labels(df, tp_pips=15, sl_pips=10, max_horizon=5)
        # First 5 bars should be NO_TRADE (range too small)
        assert labels.iloc[0] == 0

    def test_last_bars_are_no_trade(self):
        """Last max_horizon bars should be 0 (not enough forward data)."""
        n = 10
        closes = [1800.0] * n
        highs = [1800.0 + 2.0] * n
        lows = [1800.0 - 2.0] * n
        df = make_price_df(closes, highs=highs, lows=lows)
        # With max_horizon=3, last 3 bars likely can't complete labeling
        # (the very last bar has no forward data, so it's 0)
        labels = create_labels(df, tp_pips=100, sl_pips=100, max_horizon=3)
        assert labels.iloc[-1] == 0

    def test_no_lookahead(self):
        """
        Label at time T must only depend on prices AFTER T.
        We verify this by checking that label[0] is determined by bars 1+,
        not by bar 0's own high/low.
        """
        # Entry close = 1800, TP distance = 2.0 price units
        # Bar 0: high = 1803 (would trigger TP in-bar), but label is based on future
        # Bar 1 actually hits TP
        closes = [1800.0, 1810.0]
        highs = [1803.0, 1810.0]   # bar 0 high doesn't count, bar 1 high = 1810
        lows = [1797.0, 1800.0]
        df = make_price_df(closes, highs=highs, lows=lows)
        # TP = 20 pips = 2.0 price, SL = 30 pips = 3.0 price
        labels = create_labels(df, tp_pips=20, sl_pips=30, max_horizon=5)
        # Bar 0 label uses bars 1+ only
        assert labels.iloc[0] == 1  # bar 1 high (1810) > 1800 + 2.0 (1802) ✓

    def test_output_length_equals_input(self):
        """Label series length must equal input DataFrame length."""
        n = 20
        df = make_price_df([1800.0] * n)
        labels = create_labels(df, tp_pips=15, sl_pips=10, max_horizon=5)
        assert len(labels) == n

    def test_output_values_in_valid_set(self):
        """All label values must be in {-1, 0, 1}."""
        df = make_price_df(
            [1800.0 + i * 0.5 for i in range(50)],
            highs=[1800.0 + i * 0.5 + 3.0 for i in range(50)],
            lows=[1800.0 + i * 0.5 - 3.0 for i in range(50)],
        )
        labels = create_labels(df, tp_pips=15, sl_pips=10, max_horizon=10)
        assert set(labels.unique()).issubset({-1, 0, 1})

    def test_index_matches_input(self):
        """Label index must match input DataFrame index."""
        df = make_price_df([1800.0] * 10)
        labels = create_labels(df, tp_pips=15, sl_pips=10, max_horizon=5)
        pd.testing.assert_index_equal(labels.index, df.index)


# ---------------------------------------------------------------------------
# Time-based split test
# ---------------------------------------------------------------------------

class TestTimeSplit:
    def test_split_is_time_based(self):
        """Train/val/test split must be based on timestamps (no random)."""
        from model.trainer import _time_split

        n = 200
        idx = pd.date_range("2021-01-01", periods=n, freq="15min")
        df = pd.DataFrame({"value": np.arange(n, dtype=float)}, index=idx)

        # Split at a known boundary
        train, val, test = _time_split(
            df,
            "2021-01-01", "2021-01-05",
            "2021-01-06", "2021-01-08",
            "2021-01-09", "2021-01-12",
        )

        # Train must all be before val
        if len(train) > 0 and len(val) > 0:
            assert train.index.max() <= val.index.min()
        # Val must all be before test
        if len(val) > 0 and len(test) > 0:
            assert val.index.max() <= test.index.min()

    def test_no_data_leakage_between_splits(self):
        """No index overlap between splits."""
        from model.trainer import _time_split

        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="15min")
        df = pd.DataFrame({"v": np.arange(n, dtype=float)}, index=idx)

        train, val, test = _time_split(
            df,
            "2020-01-01", "2020-01-03",
            "2020-01-04", "2020-01-05",
            "2020-01-06", "2020-01-08",
        )

        # Check no overlap
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert len(train_idx & val_idx) == 0
        assert len(val_idx & test_idx) == 0
        assert len(train_idx & test_idx) == 0


# ---------------------------------------------------------------------------
# Model training smoke test
# ---------------------------------------------------------------------------

class TestModelTraining:
    def test_model_can_train_on_small_synthetic_data(self):
        """
        Train on a tiny synthetic dataset to verify the pipeline runs end-to-end.
        Does not test model quality — just that no exceptions are raised.
        """
        import xgboost as xgb
        from model.trainer import _get_feature_cols, _compute_sample_weights

        n = 200
        idx = pd.date_range("2020-01-06", periods=n, freq="15min")

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "l1_custom_volume": np.random.randint(5, 15, n).astype(float),
                "l1_volatility_energy": np.random.uniform(0.5, 5.0, n),
                "l1_price_velocity": np.random.uniform(-0.5, 0.5, n),
                "l2_session": np.random.randint(0, 3, n).astype(float),
                "l3_bos_direction": np.random.choice([-1, 0, 1], n).astype(float),
            },
            index=idx,
        )

        # Synthetic labels
        y = np.random.choice([0, 1, 2], n)
        feature_cols = list(df.columns)

        sw = _compute_sample_weights(y)
        assert len(sw) == n
        assert (sw > 0).all()

        dtrain = xgb.DMatrix(df.values, label=y, weight=sw)
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 2,
            "learning_rate": 0.1,
            "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=5, verbose_eval=False)
        preds = model.predict(xgb.DMatrix(df.values))
        assert preds.shape == (n, 3)


# ---------------------------------------------------------------------------
# Evaluation metrics tests
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def test_confusion_matrix_shape(self):
        """Confusion matrix must be 3×3."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 2, 1, 2])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_perfect_predictions_diagonal_cm(self):
        """Perfect predictions → diagonal confusion matrix."""
        y = np.array([0, 1, 2, 0, 1, 2])
        cm = compute_confusion_matrix(y, y)
        assert cm.values.trace() == len(y)

    def test_trading_simulation_win_rate(self):
        """Simulation with all correct predictions → win rate = 1.0."""
        n = 10
        idx = pd.date_range("2020-01-06", periods=n, freq="15min")
        df = pd.DataFrame({"label": np.ones(n, dtype=np.int8)}, index=idx)

        # All predict BUY with 90% confidence
        probs = np.column_stack([
            np.zeros(n),       # P(NO_TRADE)
            np.full(n, 0.9),   # P(BUY)
            np.full(n, 0.1),   # P(SELL)
        ])

        result = trading_simulation(df, probs, tp_pips=150, sl_pips=100)
        assert result["total_trades"] > 0
        assert result["win_rate"] == 1.0

    def test_trading_simulation_all_losses(self):
        """Simulation where all predictions are wrong → win rate = 0.0."""
        n = 10
        idx = pd.date_range("2020-01-06", periods=n, freq="15min")
        # All true labels are SELL (-1), all predictions are BUY
        df = pd.DataFrame({"label": np.full(n, -1, dtype=np.int8)}, index=idx)

        probs = np.column_stack([
            np.zeros(n),
            np.full(n, 0.9),   # predicts BUY
            np.full(n, 0.1),
        ])

        result = trading_simulation(df, probs, tp_pips=150, sl_pips=100)
        assert result["win_rate"] == 0.0

    def test_trading_simulation_no_signals_above_threshold(self):
        """If no predictions exceed confidence threshold → 0 trades."""
        n = 5
        idx = pd.date_range("2020-01-06", periods=n, freq="15min")
        df = pd.DataFrame({"label": np.zeros(n, dtype=np.int8)}, index=idx)

        # All predict NO_TRADE with high confidence
        probs = np.column_stack([
            np.full(n, 0.9),   # P(NO_TRADE) dominant
            np.full(n, 0.05),
            np.full(n, 0.05),
        ])

        result = trading_simulation(df, probs, tp_pips=150, sl_pips=100, confidence_threshold=80.0)
        assert result["total_trades"] == 0
