"""
GOLDWOLF — Data Loader Tests
Tests load_csv() with both M1 (date+time+volume) and M15 (timestamp, no volume) CSV formats.
"""

import io
import tempfile
import os

import numpy as np
import pandas as pd
import pytest

from data.loader import load_csv
from config.settings import (
    CSV_OPEN_COL,
    CSV_HIGH_COL,
    CSV_LOW_COL,
    CSV_CLOSE_COL,
    CSV_VOLUME_COL,
)

# ---------------------------------------------------------------------------
# CSV fixtures
# ---------------------------------------------------------------------------

M1_CSV = """\
date,time,open,high,low,close,volume
2024.01.02,09:00,1900.0,1902.0,1898.0,1901.0,100
2024.01.02,09:01,1901.0,1903.0,1900.0,1902.0,120
2024.01.02,09:02,1902.0,1905.0,1901.0,1904.0,80
"""

M15_CSV = """\
timestamp,open,high,low,close
2024-01-02 09:00:00,1900.0,1902.0,1898.0,1901.0
2024-01-02 09:15:00,1901.0,1903.0,1900.0,1902.0
2024-01-02 09:30:00,1902.0,1905.0,1901.0,1904.0
"""


def _write_tmp(content: str) -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# M1 format tests
# ---------------------------------------------------------------------------

class TestLoadCsvM1:
    def setup_method(self):
        self.path = _write_tmp(M1_CSV)

    def teardown_method(self):
        os.unlink(self.path)

    def test_returns_dataframe(self):
        df = load_csv(self.path, "M1")
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = load_csv(self.path, "M1")
        assert len(df) == 3

    def test_index_is_datetime(self):
        df = load_csv(self.path, "M1")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_expected_columns(self):
        df = load_csv(self.path, "M1")
        for col in [CSV_OPEN_COL, CSV_HIGH_COL, CSV_LOW_COL, CSV_CLOSE_COL, CSV_VOLUME_COL]:
            assert col in df.columns

    def test_volume_dtype(self):
        df = load_csv(self.path, "M1")
        assert df[CSV_VOLUME_COL].dtype == np.float32

    def test_no_date_time_cols_in_output(self):
        df = load_csv(self.path, "M1")
        assert "date" not in df.columns
        assert "time" not in df.columns

    def test_sorted_ascending(self):
        df = load_csv(self.path, "M1")
        assert df.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# M15 format tests
# ---------------------------------------------------------------------------

class TestLoadCsvM15:
    def setup_method(self):
        self.path = _write_tmp(M15_CSV)

    def teardown_method(self):
        os.unlink(self.path)

    def test_returns_dataframe(self):
        df = load_csv(self.path, "M15")
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = load_csv(self.path, "M15")
        assert len(df) == 3

    def test_index_is_datetime(self):
        df = load_csv(self.path, "M15")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_expected_columns(self):
        df = load_csv(self.path, "M15")
        for col in [CSV_OPEN_COL, CSV_HIGH_COL, CSV_LOW_COL, CSV_CLOSE_COL, CSV_VOLUME_COL]:
            assert col in df.columns

    def test_volume_is_zero(self):
        """M15 has no volume column — loader must add it filled with 0."""
        df = load_csv(self.path, "M15")
        assert (df[CSV_VOLUME_COL] == 0).all()

    def test_volume_dtype(self):
        df = load_csv(self.path, "M15")
        assert df[CSV_VOLUME_COL].dtype == np.float32

    def test_no_timestamp_col_in_output(self):
        df = load_csv(self.path, "M15")
        assert "timestamp" not in df.columns

    def test_sorted_ascending(self):
        df = load_csv(self.path, "M15")
        assert df.index.is_monotonic_increasing

    def test_ohlc_values_correct(self):
        df = load_csv(self.path, "M15")
        assert df[CSV_OPEN_COL].iloc[0] == pytest.approx(1900.0, abs=1e-4)
        assert df[CSV_CLOSE_COL].iloc[-1] == pytest.approx(1904.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestLoadCsvErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/tmp/nonexistent_goldwolf.csv", "M1")

    def test_unrecognized_format(self):
        bad_csv = "foo,bar,baz\n1,2,3\n"
        path = _write_tmp(bad_csv)
        try:
            with pytest.raises(ValueError, match="Unrecognized CSV format"):
                load_csv(path, "TEST")
        finally:
            os.unlink(path)
