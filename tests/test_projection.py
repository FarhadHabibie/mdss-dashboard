"""Tests for src/projection.py"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from projection import (
    trend_mtbf_over_time,
    trend_cost_over_time,
    calculate_lifecycle_stage,
    project_replacement_timeline,
    cost_projection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _machines_df() -> pd.DataFrame:
    return pd.DataFrame({
        "machineID": [1, 2, 3, 4, 5],
        "model": ["A", "B", "C", "D", "E"],
        "age": [1, 5, 12, 16, 20],
    })


def _failures_df() -> pd.DataFrame:
    """5 machines, quarterly failures over 2015. Machine 1: few failures (new).
    Machine 2: moderate. Machine 3: increasing (aging).
    Machine 4: high failure rate (degrading). Machine 5: very high (old)."""
    rows = [
        # Machine 1: new, low failures
        {"datetime": "2015-01-15", "machineID": 1, "failure": "comp1"},
        {"datetime": "2015-06-20", "machineID": 1, "failure": "comp1"},
        # Machine 2: moderate
        {"datetime": "2015-02-10", "machineID": 2, "failure": "comp2"},
        {"datetime": "2015-05-15", "machineID": 2, "failure": "comp2"},
        {"datetime": "2015-08-20", "machineID": 2, "failure": "comp2"},
        {"datetime": "2015-11-10", "machineID": 2, "failure": "comp2"},
        # Machine 3: increasing failures (1 in Q1, 1 in Q2, 2 in Q3, 3 in Q4)
        {"datetime": "2015-01-20", "machineID": 3, "failure": "comp3"},
        {"datetime": "2015-04-15", "machineID": 3, "failure": "comp3"},
        {"datetime": "2015-07-10", "machineID": 3, "failure": "comp3"},
        {"datetime": "2015-07-25", "machineID": 3, "failure": "comp3"},
        {"datetime": "2015-10-05", "machineID": 3, "failure": "comp3"},
        {"datetime": "2015-10-20", "machineID": 3, "failure": "comp3"},
        {"datetime": "2015-11-15", "machineID": 3, "failure": "comp3"},
        # Machine 4: high failure rate throughout
        {"datetime": "2015-01-10", "machineID": 4, "failure": "comp4"},
        {"datetime": "2015-02-15", "machineID": 4, "failure": "comp4"},
        {"datetime": "2015-03-20", "machineID": 4, "failure": "comp4"},
        {"datetime": "2015-04-10", "machineID": 4, "failure": "comp4"},
        {"datetime": "2015-05-15", "machineID": 4, "failure": "comp4"},
        {"datetime": "2015-06-20", "machineID": 4, "failure": "comp4"},
        # Machine 5: old, many failures
        {"datetime": "2015-01-05", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-02-10", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-03-15", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-04-20", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-05-25", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-06-30", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-07-05", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-08-10", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-09-15", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-10-20", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-11-25", "machineID": 5, "failure": "comp5"},
        {"datetime": "2015-12-30", "machineID": 5, "failure": "comp5"},
    ]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _cost_df() -> pd.DataFrame:
    rows = [
        # Machine 1: low cost
        {"datetime": "2015-01-15", "machineID": 1, "total_repair_cost_Jt": 100},
        {"datetime": "2015-06-20", "machineID": 1, "total_repair_cost_Jt": 150},
        # Machine 2: moderate
        {"datetime": "2015-02-10", "machineID": 2, "total_repair_cost_Jt": 200},
        {"datetime": "2015-05-15", "machineID": 2, "total_repair_cost_Jt": 250},
        {"datetime": "2015-08-20", "machineID": 2, "total_repair_cost_Jt": 300},
        {"datetime": "2015-11-10", "machineID": 2, "total_repair_cost_Jt": 350},
        # Machine 3: increasing cost
        {"datetime": "2015-01-20", "machineID": 3, "total_repair_cost_Jt": 100},
        {"datetime": "2015-04-15", "machineID": 3, "total_repair_cost_Jt": 200},
        {"datetime": "2015-07-10", "machineID": 3, "total_repair_cost_Jt": 400},
        {"datetime": "2015-07-25", "machineID": 3, "total_repair_cost_Jt": 300},
        {"datetime": "2015-10-05", "machineID": 3, "total_repair_cost_Jt": 500},
        {"datetime": "2015-10-20", "machineID": 3, "total_repair_cost_Jt": 600},
        {"datetime": "2015-11-15", "machineID": 3, "total_repair_cost_Jt": 700},
        # Machine 4: high cost
        {"datetime": "2015-01-10", "machineID": 4, "total_repair_cost_Jt": 500},
        {"datetime": "2015-02-15", "machineID": 4, "total_repair_cost_Jt": 550},
        {"datetime": "2015-03-20", "machineID": 4, "total_repair_cost_Jt": 600},
        {"datetime": "2015-04-10", "machineID": 4, "total_repair_cost_Jt": 650},
        {"datetime": "2015-05-15", "machineID": 4, "total_repair_cost_Jt": 700},
        {"datetime": "2015-06-20", "machineID": 4, "total_repair_cost_Jt": 750},
        # Machine 5: very high cost
        {"datetime": "2015-01-05", "machineID": 5, "total_repair_cost_Jt": 400},
        {"datetime": "2015-02-10", "machineID": 5, "total_repair_cost_Jt": 450},
        {"datetime": "2015-03-15", "machineID": 5, "total_repair_cost_Jt": 500},
        {"datetime": "2015-04-20", "machineID": 5, "total_repair_cost_Jt": 550},
        {"datetime": "2015-05-25", "machineID": 5, "total_repair_cost_Jt": 600},
        {"datetime": "2015-06-30", "machineID": 5, "total_repair_cost_Jt": 650},
        {"datetime": "2015-07-05", "machineID": 5, "total_repair_cost_Jt": 700},
        {"datetime": "2015-08-10", "machineID": 5, "total_repair_cost_Jt": 750},
        {"datetime": "2015-09-15", "machineID": 5, "total_repair_cost_Jt": 800},
        {"datetime": "2015-10-20", "machineID": 5, "total_repair_cost_Jt": 850},
        {"datetime": "2015-11-25", "machineID": 5, "total_repair_cost_Jt": 900},
        {"datetime": "2015-12-30", "machineID": 5, "total_repair_cost_Jt": 950},
    ]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ---------------------------------------------------------------------------
# trend_mtbf_over_time
# ---------------------------------------------------------------------------

def test_trend_mtbf_over_time():
    failures = _failures_df()
    machines = _machines_df()
    result = trend_mtbf_over_time(failures, machines, period="Q")
    assert "machineID" in result.columns
    assert len(result) == 5
    period_cols = [c for c in result.columns if c != "machineID"]
    assert len(period_cols) >= 1


def test_trend_mtbf_over_time_monthly():
    failures = _failures_df()
    machines = _machines_df()
    result = trend_mtbf_over_time(failures, machines, period="M")
    assert "machineID" in result.columns
    assert len(result) == 5
    period_cols = [c for c in result.columns if c != "machineID"]
    assert len(period_cols) >= 1


# ---------------------------------------------------------------------------
# trend_cost_over_time
# ---------------------------------------------------------------------------

def test_trend_cost_over_time():
    cost = _cost_df()
    result = trend_cost_over_time(cost, period="Q")
    assert "machineID" in result.columns
    assert len(result) == 5
    period_cols = [c for c in result.columns if c != "machineID"]
    assert len(period_cols) >= 1
    # Verify quarterly sums are non-negative
    for col in period_cols:
        assert (result[col].fillna(0) >= 0).all()


# ---------------------------------------------------------------------------
# lifecycle_stage
# ---------------------------------------------------------------------------

def test_lifecycle_stage_new_machine():
    machines = pd.DataFrame({
        "machineID": [1], "model": ["A"], "age": [1],
    })
    failures = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-06-15"]),
        "machineID": [1], "failure": ["comp1"],
    })
    cost = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-06-15"]),
        "machineID": [1], "total_repair_cost_Jt": [100],
    })
    result = calculate_lifecycle_stage(machines, failures, cost)
    assert len(result) == 1
    assert result.iloc[0]["stage"] == "NEW"


def test_lifecycle_stage_old_degrading():
    machines = pd.DataFrame({
        "machineID": [1], "model": ["A"], "age": [18],
    })
    # High failure rate: many failures spread across quarters with increasing trend
    dates = pd.to_datetime([
        "2015-01-10", "2015-02-15", "2015-03-20", "2015-04-10",
        "2015-05-15", "2015-06-20", "2015-07-10", "2015-08-15",
        "2015-09-20", "2015-10-10", "2015-11-15", "2015-12-20",
    ])
    failures = pd.DataFrame({
        "datetime": dates,
        "machineID": [1] * 12,
        "failure": ["comp1"] * 12,
    })
    cost = pd.DataFrame({
        "datetime": dates,
        "machineID": [1] * 12,
        "total_repair_cost_Jt": [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050],
    })
    result = calculate_lifecycle_stage(machines, failures, cost)
    assert len(result) == 1
    stage = result.iloc[0]["stage"]
    assert stage in ("CRITICAL", "END_OF_LIFE", "DEGRADING")


def test_lifecycle_stage_output_columns():
    result = calculate_lifecycle_stage(_machines_df(), _failures_df(), _cost_df())
    expected_cols = {
        "machineID", "model", "age", "avg_mtbf", "avg_cost_per_period",
        "mtbf_slope", "cost_slope", "stage",
    }
    assert expected_cols.issubset(set(result.columns))


def test_lifecycle_stage_insufficient_data():
    machines = pd.DataFrame({
        "machineID": [1], "model": ["A"], "age": [5],
    })
    failures = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-06-15"]),
        "machineID": [1], "failure": ["comp1"],
    })
    cost = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-06-15"]),
        "machineID": [1], "total_repair_cost_Jt": [100],
    })
    result = calculate_lifecycle_stage(machines, failures, cost)
    assert len(result) == 1
    stage = result.iloc[0]["stage"]
    assert stage == "STABLE"  # age 5, insufficient data for regression


# ---------------------------------------------------------------------------
# project_replacement_timeline
# ---------------------------------------------------------------------------

def test_project_replacement_timeline():
    result = calculate_lifecycle_stage(_machines_df(), _failures_df(), _cost_df())
    timeline = project_replacement_timeline(result)
    assert "remaining_life_months" in timeline.columns
    assert "urgency" in timeline.columns
    critical = timeline[timeline["stage"].isin(("CRITICAL", "DEGRADING", "END_OF_LIFE"))]
    if len(critical) > 0:
        assert critical["remaining_life_months"].min() >= 0
        assert (critical["urgency"].isin(["Immediate", "Near-term", "Medium", "Long-term"])).all()


def test_project_replacement_timeline_end_of_life():
    machines = pd.DataFrame({
        "machineID": [1], "model": ["A"], "age": [20],
    })
    failures = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-01-10", "2015-02-10", "2015-03-10",
                                     "2015-04-10", "2015-05-10", "2015-06-10",
                                     "2015-07-10", "2015-08-10", "2015-09-10",
                                     "2015-10-10", "2015-11-10", "2015-12-10"]),
        "machineID": [1] * 12,
        "failure": ["comp1"] * 12,
    })
    cost = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-01-10", "2015-02-10", "2015-03-10",
                                     "2015-04-10", "2015-05-10", "2015-06-10",
                                     "2015-07-10", "2015-08-10", "2015-09-10",
                                     "2015-10-10", "2015-11-10", "2015-12-10"]),
        "machineID": [1] * 12,
        "total_repair_cost_Jt": [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050],
    })
    lifecycle = calculate_lifecycle_stage(machines, failures, cost)
    timeline = project_replacement_timeline(lifecycle)
    eol = timeline[timeline["stage"] == "END_OF_LIFE"]
    if len(eol) > 0:
        assert (eol["remaining_life_months"] == 0).all()
        assert (eol["urgency"] == "Immediate").all()


# ---------------------------------------------------------------------------
# cost_projection
# ---------------------------------------------------------------------------

def test_cost_projection():
    cost = _cost_df()
    machines = _machines_df()
    result = cost_projection(cost, machines, periods_ahead=6)
    assert "machineID" in result.columns
    assert "avg_quarterly_cost_Jt" in result.columns
    assert "projected_6q_cost_Jt" in result.columns
    assert "n_quarters_data" in result.columns
    assert len(result) == 5
    # projected = avg * periods
    for _, row in result.iterrows():
        expected = row["avg_quarterly_cost_Jt"] * 6
        assert row["projected_6q_cost_Jt"] == pytest.approx(expected, rel=1e-3)


def test_cost_projection_empty():
    cost = pd.DataFrame({
        "datetime": pd.Series(dtype="datetime64[ns]"),
        "machineID": pd.Series(dtype="int"),
        "total_repair_cost_Jt": pd.Series(dtype="float"),
    })
    machines = pd.DataFrame({
        "machineID": [1], "model": ["A"], "age": [5],
    })
    result = cost_projection(cost, machines, periods_ahead=6)
    assert len(result) == 0
