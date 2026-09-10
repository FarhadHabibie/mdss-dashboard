"""Tests for src/components.py"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from components import (
    component_cost_summary,
    component_failure_summary,
    component_mtbf,
    component_risk_score,
)


def _machines_df() -> pd.DataFrame:
    return pd.DataFrame({"machineID": [1, 2], "model": ["A", "B"], "age": [5, 9]})


def _failures_df() -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.to_datetime([
            "2015-01-10", "2015-02-10", "2015-03-10",
            "2015-01-20", "2015-01-25",
            "2015-06-01", "2015-06-02", "2015-06-03", "2015-06-04",
        ]),
        "machineID": [1, 1, 1, 2, 2, 2, 2, 1, 1],
        "failure": ["comp1", "comp2", "comp1", "comp1", "comp2", "comp2",
                    "comp2", "comp2", "comp2"],
    })


def _cost_df() -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.to_datetime(["2015-01-10", "2015-02-10", "2015-01-20"]),
        "machineID": [1, 1, 2],
        "component": ["comp1", "comp2", "comp1"],
        "total_repair_cost_Jt": [100.0, 500.0, 300.0],
    })


def _empty_failures() -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.Series(dtype="datetime64[ns]"),
        "machineID": pd.Series(dtype="int"),
        "failure": pd.Series(dtype="object"),
    })


# ---------------------------------------------------------------------------
# component_failure_summary
# ---------------------------------------------------------------------------

def test_failure_summary_counts_and_pct():
    result = component_failure_summary(_failures_df())
    total = result["n_failures"].sum()
    assert total == 9
    comp2 = result[result["component"] == "comp2"].iloc[0]
    assert comp2["n_failures"] == 6
    assert comp2["pct"] == pytest.approx(6 / 9 * 100)
    assert result["pct"].sum() == pytest.approx(100.0)


def test_failure_summary_first_last_seen():
    result = component_failure_summary(_failures_df())
    comp1 = result[result["component"] == "comp1"].iloc[0]
    assert comp1["first_seen"] == pd.Timestamp("2015-01-10")
    assert comp1["last_seen"] == pd.Timestamp("2015-03-10")


def test_failure_summary_empty():
    result = component_failure_summary(_empty_failures())
    assert len(result) == 0


# ---------------------------------------------------------------------------
# component_cost_summary
# ---------------------------------------------------------------------------

def test_cost_summary_sums():
    result = component_cost_summary(_cost_df())
    comp1 = result[result["component"] == "comp1"].iloc[0]
    assert comp1["total_cost_Jt"] == 400.0
    assert comp1["avg_cost_Jt"] == 200.0
    assert comp1["max_cost_Jt"] == 300.0
    assert comp1["n_incidents"] == 2
    comp2 = result[result["component"] == "comp2"].iloc[0]
    assert comp2["total_cost_Jt"] == 500.0


def test_cost_summary_empty():
    result = component_cost_summary(_cost_df().head(0))
    assert len(result) == 0


# ---------------------------------------------------------------------------
# component_mtbf
# ---------------------------------------------------------------------------

def test_component_mtbf_real_intervals():
    # machine 1: comp1 at 2015-01-10 and 2015-03-10 -> 1448h gap (60 days * 24 = 1440)
    # machine 2: comp1 at 2015-01-20 only -> no interval
    machine1_gap_h = (pd.Timestamp("2015-03-10") - pd.Timestamp("2015-01-10")).total_seconds() / 3600
    result = component_mtbf(_failures_df(), _machines_df())
    comp1 = result[result["component"] == "comp1"].iloc[0]
    assert comp1["avg_mtbf_hours"] == machine1_gap_h
    assert comp1["n_machines_affected"] == 2  # both machines saw comp1


def test_component_mtbf_no_intervals_nan():
    # each machine has exactly 1 failure of comp1 -> no TBF
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-01-10", "2015-02-10"]),
        "machineID": [1, 2],
        "failure": ["comp1", "comp1"],
    })
    result = component_mtbf(df, _machines_df())
    comp1 = result[result["component"] == "comp1"].iloc[0]
    assert np.isnan(comp1["avg_mtbf_hours"])
    assert comp1["n_machines_affected"] == 2


def test_component_mtbf_empty():
    result = component_mtbf(_empty_failures(), _machines_df())
    assert len(result) == 0


# ---------------------------------------------------------------------------
# component_risk_score
# ---------------------------------------------------------------------------

def test_risk_score_ordering():
    fail = pd.DataFrame({
        "component": ["comp1", "comp2", "comp3"],
        "n_failures": [10, 5, 1],
    })
    cost = pd.DataFrame({
        "component": ["comp1", "comp2", "comp3"],
        "total_cost_Jt": [1000.0, 500.0, 100.0],
    })
    result = component_risk_score(fail, cost)
    assert list(result["component"]) == ["comp1", "comp2", "comp3"]
    assert result["risk_score"].is_monotonic_decreasing
    assert result.iloc[0]["risk_level"] == "HIGH"


def test_risk_score_high_thresholds():
    fail = pd.DataFrame({
        "component": ["a", "b", "c"],
        "n_failures": [100, 50, 1],
    })
    cost = pd.DataFrame({
        "component": ["a", "b", "c"],
        "total_cost_Jt": [10000.0, 100.0, 10.0],
    })
    result = component_risk_score(fail, cost)
    levels = dict(zip(result["component"], result["risk_level"]))
    assert levels["a"] == "HIGH"
    assert levels["c"] == "LOW"


def test_risk_score_empty():
    result = component_risk_score(pd.DataFrame(), pd.DataFrame())
    assert len(result) == 0