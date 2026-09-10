"""Linear-regression forecasting for failures and cost."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # imported as src.forecast (app.py) vs top-level (tests)
    from src.projection import trend_cost_over_time, trend_mtbf_over_time
except ImportError:
    from projection import trend_cost_over_time, trend_mtbf_over_time


def forecast_metric(series: pd.Series, periods_ahead: int = 4) -> dict:
    """Fit a linear regression (np.polyfit deg=1) to a numeric series.

    Returns slope, intercept, r2, last_value, next_values (len periods_ahead)
    and std_err of the slope. With fewer than 2 points all model fields are
    None and next_values/last_value fall back to 0 / the single observation.
    """
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()
    if vals.size < 2:
        last = float(vals[-1]) if vals.size == 1 else 0.0
        return {
            "slope": None,
            "intercept": None,
            "r2": None,
            "last_value": last,
            "next_values": [0.0] * periods_ahead,
            "std_err": None,
        }

    x = np.arange(vals.size, dtype=float)
    slope, intercept = np.polyfit(x, vals, deg=1)
    pred = slope * x + intercept
    ss_res = float(np.sum((vals - pred) ** 2))
    ss_tot = float(np.sum((vals - vals.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    sigma = float(np.sqrt(ss_res / (vals.size - 2))) if vals.size > 2 else 0.0
    sxx = float(np.sum((x - x.mean()) ** 2))
    std_err = sigma / np.sqrt(sxx) if sxx > 0 else np.nan
    next_values = [float(slope * (vals.size + j) + intercept) for j in range(periods_ahead)]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "last_value": float(vals[-1]),
        "next_values": next_values,
        "std_err": std_err,
    }


def _direction(slope: float | None) -> str:
    if slope is None or np.isnan(slope):
        return "flat"
    if slope > 0.2:
        return "increasing"
    if slope < -0.2:
        return "decreasing"
    return "flat"


def _projections_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "machineID", "slope", "r2", "forecast_sum_ahead", "trend_direction",
    ])


def project_failures(
    failures_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    periods_ahead: int = 4,
) -> pd.DataFrame:
    """Forecast quarterly failure counts per machine from the failure pivot."""
    if failures_df.empty:
        return _projections_df()
    pivot = trend_mtbf_over_time(failures_df, machines_df, period="Q")
    rows = []
    for _, row in pivot.iterrows():
        fc = forecast_metric(row.drop("machineID"), periods_ahead=periods_ahead)
        slope = fc["slope"]
        forecast_sum = None if slope is None or np.isnan(slope) else float(np.sum(fc["next_values"]))
        rows.append({
            "machineID": row["machineID"],
            "slope": slope,
            "r2": fc["r2"],
            "forecast_sum_ahead": forecast_sum,
            "trend_direction": _direction(slope),
        })
    return pd.DataFrame(rows)


def project_cost_trend(
    cost_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    periods_ahead: int = 4,
) -> pd.DataFrame:
    """Forecast quarterly cost sums per machine from the cost pivot."""
    if cost_df.empty:
        return pd.DataFrame(columns=["machineID", "slope", "r2", "forecast_sum_ahead"])
    pivot = trend_cost_over_time(cost_df, period="Q")
    rows = []
    for _, row in pivot.iterrows():
        fc = forecast_metric(row.drop("machineID"), periods_ahead=periods_ahead)
        slope = fc["slope"]
        forecast_sum = None if slope is None or np.isnan(slope) else float(np.sum(fc["next_values"]))
        rows.append({
            "machineID": row["machineID"],
            "slope": slope,
            "r2": fc["r2"],
            "forecast_sum_ahead": forecast_sum,
        })
    return pd.DataFrame(rows)