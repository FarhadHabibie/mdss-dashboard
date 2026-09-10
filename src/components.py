"""Component-level failure and cost analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

_FAIL_COLS = ["component", "n_failures", "pct", "first_seen", "last_seen"]
_COST_COLS = ["component", "total_cost_Jt", "avg_cost_Jt", "max_cost_Jt", "n_incidents"]
_MTBF_COLS = ["component", "avg_mtbf_hours", "n_machines_affected"]
_RISK_COLS = ["component", "n_failures", "total_cost_Jt", "risk_score", "risk_level"]


def component_failure_summary(failures_df: pd.DataFrame) -> pd.DataFrame:
    """Count failures per component with share of total and first/last seen."""
    if failures_df.empty:
        return pd.DataFrame(columns=_FAIL_COLS)
    gb = failures_df.groupby("failure")
    out = gb.agg(
        n_failures=("failure", "size"),
        first_seen=("datetime", "min"),
        last_seen=("datetime", "max"),
    ).reset_index()
    out = out.rename(columns={"failure": "component"})
    out["pct"] = out["n_failures"] / out["n_failures"].sum() * 100
    out = out.sort_values("n_failures", ascending=False).reset_index(drop=True)
    return out[_FAIL_COLS]


def component_cost_summary(cost_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repair cost per component."""
    if cost_df.empty or "component" not in cost_df.columns:
        return pd.DataFrame(columns=_COST_COLS)
    out = cost_df.groupby("component")["total_repair_cost_Jt"].agg(
        total_cost_Jt="sum",
        avg_cost_Jt="mean",
        max_cost_Jt="max",
        n_incidents="size",
    ).reset_index()
    return out[_COST_COLS]


def component_mtbf(failures_df: pd.DataFrame, machines_df: pd.DataFrame) -> pd.DataFrame:
    """Per-component mean time between failures.

    For each machine, computes the mean gap (hours) between consecutive
    failures of the component, then averages across machines. Machines with a
    single occurrence contribute to n_machines_affected but not to the mean."""
    if failures_df.empty:
        return pd.DataFrame(columns=_MTBF_COLS)
    df = failures_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    rows = []
    for (mid, comp), grp in df.groupby(["machineID", "failure"]):
        times = grp.sort_values("datetime")["datetime"].astype("int64").to_numpy() / 1_000_000_000
        diffs = np.diff(times) / 3600
        rows.append({
            "machineID": mid,
            "component": comp,
            "mtbf_hours": float(diffs.mean()) if diffs.size else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=_MTBF_COLS)
    per = pd.DataFrame(rows)
    out = per.groupby("component").agg(
        avg_mtbf_hours=("mtbf_hours", "mean"),
        n_machines_affected=("machineID", "nunique"),
    ).reset_index()
    out["avg_mtbf_hours"] = out["avg_mtbf_hours"].astype(float)
    return out.sort_values("avg_mtbf_hours", na_position="last").reset_index(drop=True)[_MTBF_COLS]


def _minmax(values: pd.Series) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def component_risk_score(
    failure_summary_df: pd.DataFrame,
    cost_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Risk score = 0.6*norm(n_failures) + 0.4*norm(total_cost), ranked desc."""
    has_f = "component" in failure_summary_df.columns and not failure_summary_df.empty
    has_c = "component" in cost_summary_df.columns and not cost_summary_df.empty
    if not has_f and not has_c:
        return pd.DataFrame(columns=_RISK_COLS)
    fs = pd.DataFrame(columns=["component", "n_failures"])
    if has_f:
        fs = failure_summary_df[["component", "n_failures"]]
    cs = pd.DataFrame(columns=["component", "total_cost_Jt"])
    if has_c:
        cs = cost_summary_df[["component", "total_cost_Jt"]]
    merged = fs.merge(cs, on="component", how="outer")
    merged["n_failures"] = merged["n_failures"].fillna(0).astype(int)
    merged["total_cost_Jt"] = merged["total_cost_Jt"].fillna(0.0)
    merged["risk_score"] = 0.6 * _minmax(merged["n_failures"]) + 0.4 * _minmax(
        merged["total_cost_Jt"]
    )
    merged["risk_level"] = merged["risk_score"].apply(
        lambda r: "HIGH" if r >= 0.66 else ("MEDIUM" if r >= 0.33 else "LOW")
    )
    merged = merged.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return merged[_RISK_COLS]