"""Tests for src/decision.py"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision import (
    calculate_repair_replacement_score,
    get_replacement_candidates,
    get_review_candidates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cost_df(
    machine_id: int,
    total_repair: float,
    replacement: float,
    downtime: float,
    n_incidents: int = 1,
) -> pd.DataFrame:
    rows = []
    for _ in range(n_incidents):
        rows.append({
            "machineID": machine_id,
            "total_repair_cost_Jt": total_repair,
            "replacement_cost_Jt": replacement,
            "downtime_hours": downtime,
        })
    return pd.DataFrame(rows)


def _cheap_repair_cost() -> pd.DataFrame:
    """cost_ratio = 3000/10000 = 0.3 → REPAIR."""
    return _make_cost_df(1, total_repair=3000, replacement=10000, downtime=10, n_incidents=3)


def _expensive_repair_cost() -> pd.DataFrame:
    """cost_ratio = 6000/5000 = 1.2 → REPLACE."""
    return _make_cost_df(2, total_repair=6000, replacement=5000, downtime=25, n_incidents=4)


def _borderline_cost() -> pd.DataFrame:
    """cost_ratio = 4000/7500 ≈ 0.533 → REVIEW."""
    return _make_cost_df(3, total_repair=4000, replacement=7500, downtime=15, n_incidents=2)


def _machines_df() -> pd.DataFrame:
    return pd.DataFrame({
        "machineID": [1, 2, 3],
        "model": ["A", "B", "C"],
        "age": [5, 12, 16],
    })


def _freq_df() -> pd.DataFrame:
    return pd.DataFrame({
        "machineID": [1, 2, 3],
        "failures_per_month": [1.0, 4.0, 2.0],
    })


# ---------------------------------------------------------------------------
# Scoring decisions
# ---------------------------------------------------------------------------

def test_decision_repair_when_cheap():
    cost = _cheap_repair_cost()
    machines = _machines_df()
    freq = _freq_df()
    df = calculate_repair_replacement_score(cost, machines, freq)
    row = df[df["machineID"] == 1].iloc[0]
    assert row["decision"] == "REPAIR"


def test_decision_replace_when_expensive():
    cost = _expensive_repair_cost()
    machines = _machines_df()
    freq = _freq_df()
    df = calculate_repair_replacement_score(cost, machines, freq)
    row = df[df["machineID"] == 2].iloc[0]
    assert row["decision"] == "REPLACE"


def test_decision_review_borderline():
    cost = _borderline_cost()
    machines = _machines_df()
    freq = _freq_df()
    df = calculate_repair_replacement_score(cost, machines, freq)
    row = df[df["machineID"] == 3].iloc[0]
    assert row["decision"] == "REVIEW"


def test_age_boost():
    cost = _borderline_cost()
    freq = _freq_df()
    young = pd.DataFrame({
        "machineID": [3], "model": ["C"], "age": [5],
    })
    old = pd.DataFrame({
        "machineID": [3], "model": ["C"], "age": [20],
    })
    df_young = calculate_repair_replacement_score(cost, young, freq)
    df_old = calculate_repair_replacement_score(cost, old, freq)
    assert df_old.iloc[0]["total_score"] > df_young.iloc[0]["total_score"]


def test_critical_boost():
    cost = _borderline_cost()
    machines = _machines_df()
    freq = _freq_df()
    base = calculate_repair_replacement_score(cost, machines, freq)
    boosted = calculate_repair_replacement_score(
        cost, machines, freq, critical_machines=[3],
    )
    assert boosted.iloc[0]["total_score"] > base.iloc[0]["total_score"]


def test_roi_years_calculation():
    cost = _borderline_cost()
    machines = _machines_df()
    freq = _freq_df()
    df = calculate_repair_replacement_score(cost, machines, freq)
    row = df[df["machineID"] == 3].iloc[0]
    expected_roi = 7500.0 / (4000.0 * 12)
    assert row["roi_years"] == pytest.approx(expected_roi, rel=1e-3)


def test_get_replacement_candidates():
    cost_all = pd.concat([
        _expensive_repair_cost(), _cheap_repair_cost(), _borderline_cost(),
    ], ignore_index=True)
    machines = _machines_df()
    freq = _freq_df()
    df = calculate_repair_replacement_score(cost_all, machines, freq)
    candidates = get_replacement_candidates(df)
    assert (candidates["decision"] == "REPLACE").all()
    if len(candidates) > 1:
        assert candidates["roi_years"].is_monotonic_increasing


def test_get_review_candidates():
    cost_all = pd.concat([
        _expensive_repair_cost(), _cheap_repair_cost(), _borderline_cost(),
    ], ignore_index=True)
    machines = _machines_df()
    freq = _freq_df()
    df = calculate_repair_replacement_score(cost_all, machines, freq)
    reviews = get_review_candidates(df)
    assert (reviews["decision"] == "REVIEW").all()
