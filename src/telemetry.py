"""Sensor regression trend, anomaly scoring, and early-failure warning."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

SENSOR_COLS: List[str] = ["volt", "rot", "press", "vib"]


def _slope_deg1(values: pd.Series) -> float:
    """Linear-regression slope of a numeric series. NaN if < 2 points."""
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if vals.size < 2:
        return np.nan
    x = np.arange(vals.size, dtype=float)
    return float(np.polyfit(x, vals, deg=1)[0])


def _empty_sensor_trend() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "machineID", "age", *[f"{s}_slope" for s in SENSOR_COLS], "sensor_stability",
    ])


def sensor_trend(telemetry_df: pd.DataFrame, machines_df: pd.DataFrame) -> pd.DataFrame:
    """Slope of each sensor's daily-mean over time, plus machine age.

    Per machine, aggregates hourly readings to daily means and fits a degree-1
    line over the day index for volt/rot/press/vib. sensor_stability is the
    std of the four slopes (low = stable)."""
    if telemetry_df.empty:
        return _empty_sensor_trend()
    t = telemetry_df.copy()
    t["day"] = pd.to_datetime(t["datetime"]).dt.floor("D")
    daily = t.groupby(["machineID", "day"])[SENSOR_COLS].mean().reset_index()
    daily = daily.sort_values(["machineID", "day"])

    rows = []
    for mid, grp in daily.groupby("machineID"):
        slopes = {s: _slope_deg1(grp[s]) for s in SENSOR_COLS}
        finite = [v for v in slopes.values() if not np.isnan(v)]
        rows.append({
            "machineID": mid,
            **{f"{s}_slope": slopes[s] for s in SENSOR_COLS},
            "sensor_stability": float(np.std(finite)) if finite else np.nan,
        })
    out = pd.DataFrame(rows)
    if "age" in machines_df.columns:
        out = out.merge(machines_df[["machineID", "age"]], on="machineID", how="left")
    else:
        out["age"] = np.nan
    return out[["machineID", "age", *[f"{s}_slope" for s in SENSOR_COLS], "sensor_stability"]]


def _empty_anomaly() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "machineID", "anomaly_score", "high_anomaly_count", "total_readings",
        "at_risk", *[f"{s}_slope" for s in SENSOR_COLS],
    ])


def anomaly_scores(
    telemetry_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    machines_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-machine anomaly score from per-sensor z-scores plus sensor slopes.

    Each hourly reading is z-scored against that machine's own sensor mean/std;
    readings beyond 3 std count as anomalies. failures_df is accepted for
    signature parity (unused)."""
    if telemetry_df.empty:
        return _empty_anomaly()
    t = telemetry_df.copy()
    rows = []
    for mid, grp in t.groupby("machineID"):
        total = len(grp) * len(SENSOR_COLS)
        hi = 0
        for s in SENSOR_COLS:
            vals = pd.to_numeric(grp[s], errors="coerce")
            std = vals.std()
            if std == 0 or np.isnan(std):
                continue
            z = ((vals - vals.mean()) / std).abs()
            hi += int((z > 3).sum())
        rows.append({
            "machineID": mid,
            "anomaly_score": hi / total if total else 0.0,
            "high_anomaly_count": hi,
            "total_readings": total,
        })
    out = pd.DataFrame(rows)

    slopes = sensor_trend(t, machines_df)[
        ["machineID", *[f"{s}_slope" for s in SENSOR_COLS]]
    ]
    out = out.merge(slopes, on="machineID", how="left")

    # At-risk via anomaly ratio OR slope outliers vs fleet distribution.
    # Sensors drift slowly (daily means), so a machine whose vib/rot slope is
    # a z-score outlier vs all machines flags degradation even when the raw
    # slope stays small in absolute terms.
    def _slope_outlier(col: str, z_thresh: float = 2.5) -> pd.Series:
        vals = pd.to_numeric(out[col], errors="coerce")
        mu, sd = vals.mean(), vals.std()
        if not sd or np.isnan(sd):
            return pd.Series(False, index=out.index)
        return (vals - mu).abs() > z_thresh * sd

    outlier_vib = _slope_outlier("vib_slope")
    outlier_rot = _slope_outlier("rot_slope")
    outlier_sensor = _slope_outlier("sensor_stability") if "sensor_stability" in out.columns else pd.Series(False, index=out.index)

    is_risk = (
        (out["anomaly_score"] > 0.2)
        | (out["vib_slope"].abs() > 0.5)
        | (out["rot_slope"].abs() > 1.0)
        | outlier_vib
        | outlier_rot
        | outlier_sensor
    )
    out["at_risk"] = is_risk.fillna(False).astype(bool)
    return out[["machineID", "anomaly_score", "high_anomaly_count", "total_readings",
                "at_risk", *[f"{s}_slope" for s in SENSOR_COLS]]]


_EARLY_WARNING_COLS = [
    "machineID", "failure_datetime",
    *[f"pre_failure_{s}" for s in SENSOR_COLS],
    *[f"baseline_{s}" for s in SENSOR_COLS],
    *[f"delta_{s}" for s in SENSOR_COLS],
]


def early_warning(
    failures_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """Compare sensor means in the pre-failure window vs machine baseline.

    For every failure, the mean of each sensor over the window_days before the
    failure (per machine) is compared against the machine's overall mean. No
    telemetry for a machine yields None values."""
    if failures_df.empty:
        return pd.DataFrame(columns=_EARLY_WARNING_COLS)

    t = pd.DataFrame({
        "machineID": telemetry_df["machineID"],
        "datetime": pd.to_datetime(telemetry_df["datetime"]),
    })
    for s in SENSOR_COLS:
        t[s] = pd.to_numeric(telemetry_df[s], errors="coerce")

    rows = []
    for _, f in failures_df.iterrows():
        mid, ft = f["machineID"], f["datetime"]
        m = t[t["machineID"] == mid]
        if m.empty:
            baseline: Dict[str, float | None] = {s: None for s in SENSOR_COLS}
            pre: Dict[str, float | None] = {s: None for s in SENSOR_COLS}
        else:
            baseline = {s: float(m[s].mean()) for s in SENSOR_COLS}
            w = m[m["datetime"] >= ft - pd.Timedelta(days=window_days)]
            w = w[w["datetime"] < ft]
            pre = {s: (float(w[s].mean()) if not w.empty else None) for s in SENSOR_COLS}
        row: Dict = {"machineID": mid, "failure_datetime": ft}
        for s in SENSOR_COLS:
            d = None if pre[s] is None or baseline[s] is None else pre[s] - baseline[s]
            row[f"pre_failure_{s}"] = pre[s]
            row[f"baseline_{s}"] = baseline[s]
            row[f"delta_{s}"] = d
        rows.append(row)
    return pd.DataFrame(rows)[_EARLY_WARNING_COLS]