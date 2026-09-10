"""Tests for src/priority.py"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from priority import (
    calculate_priority_score,
    get_by_category,
    get_top_priority,
    normalize_series,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _availability_df() -> pd.DataFrame:
    return pd.DataFrame({
        "machineID": [1, 2, 3, 4],
        "mtbf_hours": [100.0, 500.0, 2000.0, 30.0],
        "mttr_hours": [2.0, 10.0, 1.0, 40.0],
        "availability_pct": [98.0, 98.0, 99.5, 42.8],
    })


def _frequency_df() -> pd.DataFrame:
    return pd.DataFrame({
        "machineID": [1, 2, 3, 4],
        "failures_per_month": [2.0, 0.5, 0.1, 5.0],
    })


def _cost_df() -> pd.DataFrame:
    return pd.DataFrame({
        "machineID": [1, 2, 3, 4],
        "total_cost_Jt": [5000.0, 2000.0, 500.0, 12000.0],
        "avg_cost_Jt": [1000.0, 500.0, 100.0, 3000.0],
    })


def _machines_df() -> pd.DataFrame:
    return pd.DataFrame({
        "machineID": [1, 2, 3, 4],
        "model": ["A", "B", "C", "D"],
        "age": [5, 10, 2, 15],
    })


# ---------------------------------------------------------------------------
# normalize_series
# ---------------------------------------------------------------------------

def test_normalize_series_basic():
    s = pd.Series([0.0, 50.0, 100.0])
    result = normalize_series(s)
    expected = pd.Series([0.0, 0.5, 1.0])
    pd.testing.assert_series_equal(result, expected)


def test_normalize_series_all_same():
    s = pd.Series([5.0, 5.0, 5.0])
    result = normalize_series(s)
    expected = pd.Series([0.5, 0.5, 0.5])
    pd.testing.assert_series_equal(result, expected)


def test_normalize_series_with_nan():
    s = pd.Series([np.nan, 0.0, 100.0])
    result = normalize_series(s)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# calculate_priority_score
# ---------------------------------------------------------------------------

def test_priority_score_range():
    df = calculate_priority_score(
        _availability_df(), _frequency_df(), _cost_df(), _machines_df(),
    )
    assert "score" in df.columns
    assert df["score"].between(0, 100).all()


def test_priority_urgent_category():
    df = calculate_priority_score(
        _availability_df(), _frequency_df(), _cost_df(), _machines_df(),
    )
    worst = df.loc[df["machineID"] == 4]
    assert worst["category"].iloc[0] == "URGENT"


def test_priority_low_category():
    df = calculate_priority_score(
        _availability_df(), _frequency_df(), _cost_df(), _machines_df(),
    )
    best = df.loc[df["machineID"] == 3]
    assert best["category"].iloc[0] == "LOW"


def test_critical_machine_boost():
    base = calculate_priority_score(
        _availability_df(), _frequency_df(), _cost_df(), _machines_df(),
    )
    boosted = calculate_priority_score(
        _availability_df(), _frequency_df(), _cost_df(), _machines_df(),
        critical_machines=[1],
    )
    score_base = base.loc[base["machineID"] == 1, "score"].iloc[0]
    score_boosted = boosted.loc[boosted["machineID"] == 1, "score"].iloc[0]
    assert score_boosted > score_base


def test_get_top_priority():
    df = calculate_priority_score(
        _availability_df(), _frequency_df(), _cost_df(), _machines_df(),
    )
    top2 = get_top_priority(df, n=2)
    assert len(top2) == 2
    assert top2.iloc[0]["score"] >= top2.iloc[1]["score"]


def test_get_by_category():
    df = calculate_priority_score(
        _availability_df(), _frequency_df(), _cost_df(), _machines_df(),
    )
    urgent = get_by_category(df, "URGENT")
    assert (urgent["category"] == "URGENT").all()
    assert len(urgent) >= 1
