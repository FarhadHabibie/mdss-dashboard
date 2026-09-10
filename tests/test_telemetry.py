"""Tests for src/telemetry.py"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from telemetry import anomaly_scores, early_warning, sensor_trend


def _machines_df() -> pd.DataFrame:
    return pd.DataFrame({"machineID": [1, 2], "model": ["A", "B"], "age": [5, 9]})


def _hourly_telemetry(pid, n_days, rot_base, rot_step, vib_base, vib_step) -> pd.DataFrame:
    """Hourly readings for `pid` where rot/vib drift by step per day."""
    start = pd.Timestamp("2015-06-01")
    days = [start + pd.Timedelta(days=d) for d in range(n_days)]
    rows = []
    for d, day in enumerate(days):
        for h in range(24):
            rows.append({
                "datetime": day + pd.Timedelta(hours=h),
                "machineID": pid,
                "volt": 170.0,
                "rot": rot_base + rot_step * d,
                "press": 100.0,
                "vib": vib_base + vib_step * d,
            })
    return pd.DataFrame(rows)


def _telemetry_df() -> pd.DataFrame:
    t1 = _hourly_telemetry(1, n_days=10, rot_base=300, rot_step=2.0,
                           vib_base=40, vib_step=0.8)
    t2 = _hourly_telemetry(2, n_days=10, rot_base=400, rot_step=0.0,
                           vib_base=50, vib_step=0.0)
    return pd.concat([t1, t2], ignore_index=True)


# ---------------------------------------------------------------------------
# sensor_trend
# ---------------------------------------------------------------------------

def test_sensor_trend_slopes_and_age():
    result = sensor_trend(_telemetry_df(), _machines_df())
    drift = result[result["machineID"] == 1].iloc[0]
    assert drift["rot_slope"] == pytest.approx(2.0)
    assert drift["vib_slope"] == pytest.approx(0.8)
    assert drift["age"] == pytest.approx(5)
    stable = result[result["machineID"] == 2].iloc[0]
    assert stable["rot_slope"] == pytest.approx(0.0, abs=1e-9)
    assert stable["sensor_stability"] is not None


def test_sensor_trend_empty():
    t = _telemetry_df().head(0)
    result = sensor_trend(t, _machines_df())
    assert len(result) == 0


# ---------------------------------------------------------------------------
# anomaly_scores
# ---------------------------------------------------------------------------

def test_anomaly_scores_at_risk():
    result = anomaly_scores(_telemetry_df(), pd.DataFrame(), _machines_df())
    r1 = result[result["machineID"] == 1].iloc[0]
    r2 = result[result["machineID"] == 2].iloc[0]
    assert r1["at_risk"] == True       # rot_slope 2.0 > 1.0
    assert r2["at_risk"] == False
    assert 0.0 <= r1["anomaly_score"] <= 1.0
    assert r1["total_readings"] == 10 * 24 * 4


def test_anomaly_scores_empty():
    t = _telemetry_df().head(0)
    result = anomaly_scores(t, pd.DataFrame(), _machines_df())
    assert len(result) == 0


# ---------------------------------------------------------------------------
# early_warning
# ---------------------------------------------------------------------------

def test_early_warning_window_delta():
    # vib high in the 7 days before failure, low baseline before that
    start = pd.Timestamp("2015-06-01")
    rows = []
    for d in range(20):
        for h in range(24):
            vib = 100.0 if d >= 13 else 10.0
            rows.append({
                "datetime": start + pd.Timedelta(days=d, hours=h),
                "machineID": 1,
                "volt": 170.0, "rot": 400.0, "press": 100.0, "vib": vib,
            })
    telemetry = pd.DataFrame(rows)
    failures = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-06-21 12:00:00"]),
        "machineID": [1],
        "failure": ["comp1"],
    })
    result = early_warning(failures, telemetry, window_days=7)
    row = result.iloc[0]
    assert row["machineID"] == 1
    assert row["pre_failure_vib"] == pytest.approx(100.0)
    assert row["delta_vib"] == pytest.approx(100.0 - row["baseline_vib"], rel=1e-6)
    assert row["pre_failure_volt"] == pytest.approx(170.0)
    assert row["delta_volt"] == pytest.approx(0.0, abs=1e-6)


def test_early_warning_no_telemetry_none():
    failures = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-06-21 12:00:00"]),
        "machineID": [99],
        "failure": ["comp1"],
    })
    telemetry = _telemetry_df().head(0)
    result = early_warning(failures, telemetry, window_days=7)
    row = result.iloc[0]
    assert row["pre_failure_vib"] is None
    assert row["baseline_vib"] is None
    assert row["delta_vib"] is None


def test_early_warning_empty_failures():
    failures = pd.DataFrame({
        "datetime": pd.Series(dtype="datetime64[ns]"),
        "machineID": pd.Series(dtype="int"),
        "failure": pd.Series(dtype="object"),
    })
    result = early_warning(failures, _telemetry_df())
    assert len(result) == 0