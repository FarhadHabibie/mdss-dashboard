"""Tests for src/data_loader.py"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import clean_data, load_all, merge_failure_cost

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_load_all_returns_7_keys():
    data = load_all(DATA_DIR)
    expected = {
        "machines",
        "telemetry",
        "errors",
        "failures",
        "maintenance",
        "cost",
        "engineering",
    }
    assert set(data.keys()) == expected
    for df in data.values():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


def test_load_all_columns():
    data = load_all(DATA_DIR)
    assert list(data["machines"].columns) == ["machineID", "model", "age"]
    assert list(data["telemetry"].columns) == [
        "datetime",
        "machineID",
        "volt",
        "rot",
        "press",
        "vib",
    ]
    assert list(data["errors"].columns) == ["datetime", "machineID", "errorID"]
    assert list(data["failures"].columns) == ["datetime", "machineID", "failure"]
    assert list(data["maintenance"].columns) == [
        "datetime",
        "machineID",
        "comp",
        "type",
    ]
    assert list(data["cost"].columns) == [
        "datetime",
        "machineID",
        "component",
        "parts_cost_Jt",
        "labor_hours",
        "labor_cost_Jt",
        "total_repair_cost_Jt",
        "downtime_hours",
        "replacement_cost_Jt",
        "is_replacement",
    ]
    assert list(data["engineering"].columns) == [
        "job_type",
        "complexity",
        "estimated_days",
        "actual_days",
        "crew_size",
        "skill_required",
        "tools",
        "spare_parts_needed",
        "potential_delay_hours",
        "parallel_allowed",
        "buffer_pct",
    ]


def test_load_all_datetime_parsed():
    data = load_all(DATA_DIR)
    for key in ("telemetry", "errors", "failures", "maintenance", "cost"):
        assert pd.api.types.is_datetime64_any_dtype(data[key]["datetime"])


def test_clean_data_telemetry_drops_nan():
    telemetry = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-01-01", "2015-01-02"]),
            "machineID": [1, 1],
            "volt": [1.0, None],
            "rot": [2.0, 3.0],
            "press": [3.0, 4.0],
            "vib": [4.0, 5.0],
        }
    )
    data = {"telemetry": telemetry}
    cleaned = clean_data(data)
    assert len(cleaned["telemetry"]) == 1


def test_clean_data_failures_dedup():
    failures = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-01-02", "2015-01-01"]),
            "machineID": [1, 1],
            "failure": ["comp1", "comp1"],
        }
    )
    dup = pd.concat([failures, failures], ignore_index=True)
    data = {"failures": dup}
    cleaned = clean_data(data)
    assert len(cleaned["failures"]) == 2
    assert cleaned["failures"]["datetime"].is_monotonic_increasing


def test_clean_data_cost_float_and_fill():
    cost = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-01-01"]),
            "machineID": [1],
            "component": ["comp1"],
            "parts_cost_Jt": ["1.5"],
            "labor_hours": [2],
            "labor_cost_Jt": [3],
            "total_repair_cost_Jt": [None],
            "downtime_hours": [4],
            "replacement_cost_Jt": [5],
            "is_replacement": [0],
        }
    )
    cleaned = clean_data({"cost": cost})["cost"]
    assert cleaned["parts_cost_Jt"].dtype == float
    assert cleaned["total_repair_cost_Jt"].iloc[0] == 0.0


def test_merge_failure_cost_joins():
    failures = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2015-01-01", "2015-01-02"]),
            "machineID": [1, 2],
            "failure": ["comp1", "comp2"],
        }
    )
    cost_columns = [
        "datetime",
        "machineID",
        "component",
        "parts_cost_Jt",
        "labor_hours",
        "labor_cost_Jt",
        "total_repair_cost_Jt",
        "downtime_hours",
        "replacement_cost_Jt",
        "is_replacement",
    ]
    cost = pd.DataFrame(
        [
            [pd.Timestamp("2015-01-01"), 1, "comp1", 1.0, 1, 1.0, 10.0, 5.0, 9.0, 0],
            [pd.Timestamp("2015-01-02"), 2, "comp2", 2.0, 2, 2.0, 20.0, 6.0, 9.0, 0],
        ],
        columns=cost_columns,
    )
    machines = pd.DataFrame({"machineID": [1, 2], "model": ["a", "b"], "age": [4, 9]})

    merged = merge_failure_cost(
        {"failures": failures, "cost": cost, "machines": machines}
    )
    assert len(merged) == 2
    assert "total_repair_cost_Jt" in merged.columns
    assert "age" in merged.columns
    assert merged.loc[0, "model"] == "a"
