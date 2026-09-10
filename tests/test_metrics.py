"""Tests for src/metrics.py"""
import math
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (
    calculate_availability,
    calculate_failure_frequency,
    calculate_maintenance_cost,
    calculate_mtbf,
    calculate_mttr,
    cost_per_downtime_hour,
)


def _machines():
    return pd.DataFrame(
        {"machineID": [1, 2], "model": ["model1", "model2"], "age": [2, 1]}
    )


def test_mtbf_known_intervals():
    failures = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2015-01-01", "2015-01-03", "2015-01-07"]
            ),
            "machineID": [1, 1, 1],
            "failure": ["c1", "c2", "c3"],
        }
    )
    result = calculate_mtbf(
        failures, _machines(), "2014-01-01", "2016-01-01"
    )
    row = result[result["machineID"] == 1].iloc[0]
    assert row["n_failures"] == 3
    assert row["total_operating_hours"] == 24 * 730.0
    assert row["mtbf_hours"] == pytest.approx(72.0)


def test_mtbf_zero_failures_nan():
    failures = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-01-01"]),
            "machineID": [9],
            "failure": ["c1"],
        }
    )
    result = calculate_mtbf(
        failures, _machines(), "2014-01-01", "2016-01-01"
    )
    row = result[result["machineID"] == 2].iloc[0]
    assert row["n_failures"] == 0
    assert math.isnan(row["mtbf_hours"]) or row["mtbf_hours"] in (float("inf"),)


def test_mtbf_single_failure_nan():
    failures = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-06-01"]),
            "machineID": [1],
            "failure": ["c1"],
        }
    )
    result = calculate_mtbf(
        failures, _machines(), "2014-01-01", "2016-01-01"
    )
    row = result[result["machineID"] == 1].iloc[0]
    assert row["n_failures"] == 1
    assert math.isnan(row["mtbf_hours"])


def test_mttr_known():
    cost = pd.DataFrame(
        {
            "machineID": [1, 1, 1],
            "downtime_hours": [10, 20, 30],
            "total_repair_cost_Jt": [100, 200, 300],
        }
    )
    result = calculate_mttr(cost)
    row = result[result["machineID"] == 1].iloc[0]
    assert row["n_repairs"] == 3
    assert row["mttr_hours"] == pytest.approx(20.0)
    assert row["total_downtime_hours"] == pytest.approx(60.0)


def test_availability_formula():
    mtbf = pd.DataFrame({"machineID": [1], "mtbf_hours": [100.0], "n_failures": [3]})
    mttr = pd.DataFrame({"machineID": [1], "mttr_hours": [10.0], "n_repairs": [3]})
    result = calculate_availability(mtbf, mttr)
    assert result.loc[0, "availability_pct"] == pytest.approx(90.909, rel=1e-3)


def test_failure_frequency():
    failures = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-01-01", "2015-02-01", "2015-06-01"]),
            "machineID": [1, 1, 1],
            "failure": ["c1", "c2", "c3"],
        }
    )
    result = calculate_failure_frequency(failures, _machines())
    row = result[result["machineID"] == 1].iloc[0]
    assert row["total_failures"] == 3
    assert row["machine_age_months"] == pytest.approx(24.0)
    assert row["failures_per_month"] == pytest.approx(3.0 / 24.0)


def test_maintenance_cost():
    cost = pd.DataFrame(
        {
            "machineID": [1, 1, 1],
            "total_repair_cost_Jt": [100.0, 200.0, 300.0],
            "downtime_hours": [10, 20, 30],
        }
    )
    result = calculate_maintenance_cost(cost)
    row = result[result["machineID"] == 1].iloc[0]
    assert row["total_cost_Jt"] == pytest.approx(600.0)
    assert row["avg_cost_Jt"] == pytest.approx(200.0)
    assert row["max_cost_Jt"] == pytest.approx(300.0)
    assert row["n_incidents"] == 3


def test_cost_per_downtime_hour():
    cost = pd.DataFrame(
        {
            "machineID": [1, 1],
            "total_repair_cost_Jt": [100.0, 300.0],
            "downtime_hours": [20.0, 40.0],
        }
    )
    result = cost_per_downtime_hour(cost)
    row = result[result["machineID"] == 1].iloc[0]
    assert row["cost_per_hour_Jt"] == pytest.approx(400.0 / 60.0)
