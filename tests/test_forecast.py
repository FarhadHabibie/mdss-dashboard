"""Tests for src/forecast.py"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from forecast import project_cost_trend, project_failures, forecast_metric


def _failures_df() -> pd.DataFrame:
    # machine 1 fails 1,2,3,4 times across Q1..Q4 2015 -> quarterly slope 1
    return pd.DataFrame({
        "datetime": pd.to_datetime([
            "2015-01-15",                      # Q1: 1
            "2015-04-10", "2015-05-20",        # Q2: 2
            "2015-07-05", "2015-08-10", "2015-09-15",  # Q3: 3
            "2015-10-05", "2015-11-10", "2015-12-05", "2015-12-25",  # Q4: 4
        ]),
        "machineID": [1] * 10,
        "failure": ["comp1"] * 10,
    })


def _machines_df() -> pd.DataFrame:
    return pd.DataFrame({"machineID": [1, 2], "model": ["A", "B"], "age": [5, 8]})


def _cost_df() -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.to_datetime([
            "2015-01-15", "2015-04-20", "2015-07-10", "2015-10-05",
        ]),
        "machineID": [1, 1, 1, 1],
        "component": ["comp1"] * 4,
        "total_repair_cost_Jt": [100.0, 200.0, 300.0, 400.0],
    })


def _empty_failures() -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.Series(dtype="datetime64[ns]"),
        "machineID": pd.Series(dtype="int"),
        "failure": pd.Series(dtype="object"),
    })


def _empty_cost() -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.Series(dtype="datetime64[ns]"),
        "machineID": pd.Series(dtype="int"),
        "total_repair_cost_Jt": pd.Series(dtype="float"),
    })


# ---------------------------------------------------------------------------
# forecast_metric
# ---------------------------------------------------------------------------

def test_forecast_perfect_linear():
    res = forecast_metric(pd.Series([1.0, 2.0, 3.0, 4.0]), periods_ahead=2)
    assert res["slope"] == pytest.approx(1.0)
    assert res["intercept"] == pytest.approx(1.0)
    assert res["r2"] == pytest.approx(1.0)
    assert res["next_values"] == pytest.approx([5.0, 6.0])
    assert res["last_value"] == pytest.approx(4.0)


def test_forecast_flat_series():
    res = forecast_metric(pd.Series([5.0, 5.0, 5.0, 5.0]))
    assert res["slope"] == pytest.approx(0.0, abs=1e-9)
    assert res["r2"] == pytest.approx(1.0)
    assert res["next_values"] == pytest.approx([5.0] * 4)


def test_forecast_less_than_two_points():
    res = forecast_metric(pd.Series([7.0]))
    assert res["slope"] is None
    assert res["r2"] is None
    assert res["next_values"] == [0.0] * 4
    assert res["last_value"] == pytest.approx(7.0)


def test_forecast_empty_series():
    res = forecast_metric(pd.Series(dtype="float"))
    assert res["slope"] is None
    assert res["r2"] is None
    assert res["std_err"] is None
    assert res["last_value"] == 0.0


def test_forecast_matches_manual_extrapolation():
    series = pd.Series([10.0, 13.0, 16.0, 19.0])
    slope, intercept = np.polyfit(np.arange(4), series, 1)
    periods_ahead = 3
    res = forecast_metric(series, periods_ahead=periods_ahead)
    expected = [slope * (4 + j) + intercept for j in range(periods_ahead)]
    assert res["next_values"] == pytest.approx(expected)
    assert res["slope"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# project_failures
# ---------------------------------------------------------------------------

def test_project_failures_linear():
    result = project_failures(_failures_df(), _machines_df(), periods_ahead=4)
    row = result[result["machineID"] == 1].iloc[0]
    assert row["slope"] == pytest.approx(1.0)  # 1 failure each quarter
    assert row["r2"] == pytest.approx(1.0)
    # next 4 quarters at slope 1: quarters 5..8 -> 5+6+7+8
    assert row["forecast_sum_ahead"] == pytest.approx(26.0)
    assert row["trend_direction"] == "increasing"


def test_project_failures_flat_direction():
    dates = pd.to_datetime(["2015-01-15", "2015-04-20", "2015-07-10", "2015-10-05"])
    flat = pd.DataFrame({
        "datetime": dates, "machineID": [1, 1, 1, 1], "failure": ["comp1"] * 4,
    })
    result = project_failures(flat, _machines_df())
    assert result.iloc[0]["trend_direction"] == "flat"
    assert result.iloc[0]["slope"] == pytest.approx(0.0, abs=1e-9)


def test_project_failures_single_quarter_none():
    one = _failures_df().head(1)
    result = project_failures(one, _machines_df())
    assert result.iloc[0]["slope"] is None
    assert result.iloc[0]["forecast_sum_ahead"] is None
    assert result.iloc[0]["trend_direction"] == "flat"


def test_project_failures_empty():
    result = project_failures(_empty_failures(), _machines_df())
    assert len(result) == 0
    assert list(result.columns) == [
        "machineID", "slope", "r2", "forecast_sum_ahead", "trend_direction",
    ]


# ---------------------------------------------------------------------------
# project_cost_trend
# ---------------------------------------------------------------------------

def test_project_cost_trend_linear():
    result = project_cost_trend(_cost_df(), _machines_df(), periods_ahead=4)
    row = result.iloc[0]
    assert row["slope"] == pytest.approx(100.0)
    assert row["r2"] == pytest.approx(1.0)
    # quarters 5..8 at slope 100: 500+600+700+800
    assert row["forecast_sum_ahead"] == pytest.approx(2600.0)


def test_project_cost_trend_empty():
    result = project_cost_trend(_empty_cost(), _machines_df())
    assert len(result) == 0


def test_project_cost_trend_none_on_single_point():
    one = _cost_df().head(1)
    result = project_cost_trend(one, _machines_df())
    assert result.iloc[0]["slope"] is None
    assert result.iloc[0]["forecast_sum_ahead"] is None