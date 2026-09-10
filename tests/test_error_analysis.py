"""Tests for src/error_analysis.py"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from error_analysis import (
    error_before_failure,
    error_failure_correlation,
    error_frequency,
    predictor_summary,
)


def _errors():
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2015-01-05 08:00:00",
                    "2015-01-06 09:00:00",
                    "2015-01-10 10:00:00",
                    "2015-01-15 11:00:00",
                    "2015-01-20 12:00:00",
                ]
            ),
            "machineID": [1, 1, 1, 2, 2],
            "errorID": ["error1", "error1", "error2", "error3", "error1"],
        }
    )


def _failures():
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2015-01-12 08:00:00",
                    "2015-01-25 08:00:00",
                    "2015-01-20 08:00:00",
                ]
            ),
            "machineID": [1, 2, 3],
            "failure": ["comp1", "comp2", "comp1"],
        }
    )


def test_error_frequency_basic():
    df = _errors()
    result = error_frequency(df)
    assert len(result) == 3
    assert set(result.columns) == {"errorID", "n_errors", "pct"}
    assert result["n_errors"].sum() == 5
    assert result["pct"].sum() == pytest.approx(1.0)
    row1 = result[result["errorID"] == "error1"].iloc[0]
    assert row1["n_errors"] == 3
    assert row1["pct"] == pytest.approx(0.6)


def test_error_frequency_empty():
    result = error_frequency(pd.DataFrame(columns=["datetime", "machineID", "errorID"]))
    assert len(result) == 0


def test_error_before_failure_window():
    errors = _errors()
    failures = _failures()
    result = error_before_failure(errors, failures, window_days=14)
    # machine 1: failure 2015-01-12, errors on 01-05 (7d before, in window), 01-06 (6d, in), 01-10 (2d, in) -> 3
    r1 = result[(result["machineID"] == 1) & (result["failure_datetime"] == pd.Timestamp("2015-01-12 08:00:00"))]
    assert r1.iloc[0]["n_errors_before"] == 3
    assert "error1" in r1.iloc[0]["error_ids"]
    assert "error2" in r1.iloc[0]["error_ids"]

    # machine 3: no errors, 0 expected
    r3 = result[result["machineID"] == 3].iloc[0]
    assert r3["n_errors_before"] == 0


def test_error_before_failure_outside_window():
    errors = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-01-01 08:00:00"]),
            "machineID": [1],
            "errorID": ["error1"],
        }
    )
    failures = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-01-20 08:00:00"]),
            "machineID": [1],
            "failure": ["comp1"],
        }
    )
    result = error_before_failure(errors, failures, window_days=14)
    # 19 days before -> outside 14-day window
    assert result.iloc[0]["n_errors_before"] == 0


def test_error_before_failure_empty():
    result = error_before_failure(
        pd.DataFrame(columns=["datetime", "machineID", "errorID"]),
        pd.DataFrame(columns=["datetime", "machineID", "failure"]),
    )
    assert len(result) == 0


def test_error_failure_correlation():
    errors = _errors()
    failures = _failures()
    dist = error_before_failure(errors, failures, window_days=14)
    corr = error_failure_correlation(dist)
    assert len(corr) == 3
    assert set(corr.columns) == {"errorID", "total_occurrences", "n_preceding_failure", "hit_rate"}
    row1 = corr[corr["errorID"] == "error1"].iloc[0]
    assert row1["total_occurrences"] == 3
    assert row1["n_preceding_failure"] == 2
    assert row1["hit_rate"] == pytest.approx(2.0 / 3.0)
    row2 = corr[corr["errorID"] == "error2"].iloc[0]
    assert row2["total_occurrences"] == 1
    assert row2["n_preceding_failure"] == 1
    assert row2["hit_rate"] == pytest.approx(1.0)


def test_error_failure_correlation_empty():
    dist = pd.DataFrame(
        columns=["machineID", "failure_datetime", "n_errors_before", "error_ids"]
    )
    result = error_failure_correlation(dist)
    assert len(result) == 0


def test_predictor_summary():
    errors = _errors()
    failures = _failures()
    summary = predictor_summary(errors, failures, window_days=14)
    assert isinstance(summary, dict)
    assert "top_errors" in summary
    assert "coverage" in summary
    assert summary["coverage"] >= 0.0
    # 2 out of 3 failures have preceding errors -> coverage ~0.667
    assert summary["coverage"] == pytest.approx(2.0 / 3.0)
    # top_errors should have error1 (highest hit_rate)
    top_ids = [e["errorID"] for e in summary["top_errors"]]
    assert "error1" in top_ids


def test_predictor_summary_empty():
    summary = predictor_summary(
        pd.DataFrame(columns=["datetime", "machineID", "errorID"]),
        pd.DataFrame(columns=["datetime", "machineID", "failure"]),
    )
    assert summary["coverage"] == 0.0
    assert len(summary["top_errors"]) == 0
