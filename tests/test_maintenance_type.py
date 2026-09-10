"""Tests for src/maintenance_type.py"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maintenance_type import (
    maintenance_type_ratio,
    preventive_maturity,
    unscheduled_cost_penalty,
)


def _maint():
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2015-01-05 08:00:00",
                    "2015-01-12 08:00:00",
                    "2015-02-02 08:00:00",
                    "2015-02-10 08:00:00",
                    "2015-01-20 08:00:00",
                ]
            ),
            "machineID": [1, 1, 1, 2, 2],
            "type": ["scheduled", "scheduled", "scheduled", "unscheduled", "unscheduled"],
        }
    )


def _machines():
    return pd.DataFrame({"machineID": [1, 2]})


def test_maintenance_type_ratio():
    df = _maint()
    result = maintenance_type_ratio(df)
    assert set(result.columns) == {"type", "n", "pct"}
    assert result["n"].sum() == 5
    assert result["pct"].sum() == pytest.approx(1.0)
    sched = result[result["type"] == "scheduled"].iloc[0]
    assert sched["n"] == 3
    assert sched["pct"] == pytest.approx(0.6)


def test_maintenance_type_ratio_empty():
    result = maintenance_type_ratio(
        pd.DataFrame(columns=["datetime", "machineID", "type"])
    )
    assert len(result) == 0


def test_preventive_maturity_thresholds():
    result = preventive_maturity(_maint(), _machines())
    assert len(result) == 2
    assert set(result.columns) == {
        "machineID",
        "n_maintenance",
        "n_scheduled",
        "n_unscheduled",
        "schedule_ratio",
        "maturity",
    }
    r1 = result[result["machineID"] == 1].iloc[0]
    assert r1["n_scheduled"] == 3
    assert r1["n_unscheduled"] == 0
    assert r1["schedule_ratio"] == pytest.approx(1.0)
    assert r1["maturity"] == "HIGH"
    r2 = result[result["machineID"] == 2].iloc[0]
    assert r2["schedule_ratio"] == pytest.approx(0.0)
    assert r2["maturity"] == "LOW"


def test_preventive_maturity_medium():
    maint = _maint().iloc[:1]
    rows = maint.to_dict("records")
    rows.append(
        {
            "datetime": pd.Timestamp("2015-01-15 08:00:00"),
            "machineID": 2,
            "type": "scheduled",
        }
    )
    rows.append(
        {
            "datetime": pd.Timestamp("2015-01-22 08:00:00"),
            "machineID": 2,
            "type": "unscheduled",
        }
    )
    df = pd.DataFrame(rows)
    result = preventive_maturity(df, _machines())
    r2 = result[result["machineID"] == 2].iloc[0]
    assert r2["schedule_ratio"] == pytest.approx(0.5)
    assert r2["maturity"] == "MEDIUM"


def test_preventive_maturity_period_month():
    result = preventive_maturity(_maint(), _machines(), period="M")
    # group by machine, ratio already 2 rows
    assert len(result[result["machineID"] == 1]) == 1


def test_preventive_maturity_empty():
    result = preventive_maturity(
        pd.DataFrame(columns=["datetime", "machineID", "type"]), _machines()
    )
    assert len(result) == 0


def test_unscheduled_cost_penalty():
    maint = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2015-01-05 08:00:00",
                    "2015-01-12 08:00:00",
                    "2015-02-08 08:00:00",
                    "2015-02-10 08:00:00",
                ]
            ),
            "machineID": [1, 1, 2, 2],
            "type": ["scheduled", "scheduled", "scheduled", "unscheduled"],
        }
    )
    cost = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2015-01-05 08:00:00",
                    "2015-01-12 08:00:00",
                    "2015-02-08 08:00:00",
                    "2015-02-10 08:00:00",
                ]
            ),
            "machineID": [1, 1, 2, 2],
            "total_repair_cost_Jt": [100.0, 100.0, 100.0, 300.0],
        }
    )
    result = unscheduled_cost_penalty(cost, maint)
    assert len(result) == 2
    assert set(result.columns) == {
        "machineID",
        "avg_cost_scheduled_Jt",
        "avg_cost_unscheduled_Jt",
        "penalty_pct",
    }
    r1 = result[result["machineID"] == 1].iloc[0]
    assert r1["avg_cost_scheduled_Jt"] == pytest.approx(100.0)
    assert r1["penalty_pct"] == pytest.approx(0.0)
    r2 = result[result["machineID"] == 2].iloc[0]
    assert r2["avg_cost_unscheduled_Jt"] == pytest.approx(300.0)
    assert r2["penalty_pct"] == pytest.approx(200.0)


def test_unscheduled_cost_penalty_empty():
    result = unscheduled_cost_penalty(
        pd.DataFrame(columns=["datetime", "machineID", "total_repair_cost_Jt"]),
        pd.DataFrame(columns=["datetime", "machineID", "type"]),
    )
    assert len(result) == 0