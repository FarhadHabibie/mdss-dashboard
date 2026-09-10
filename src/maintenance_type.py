"""Scheduled vs unscheduled maintenance analysis and cost penalty."""
from __future__ import annotations

import pandas as pd

HIGH_MATURITY = 0.7
MEDIUM_MATURITY = 0.4


def maintenance_type_ratio(maint_df: pd.DataFrame) -> pd.DataFrame:
    """Share of each maintenance type over all maintenance events."""
    if maint_df.empty:
        return pd.DataFrame(columns=["type", "n", "pct"])
    counts = maint_df.groupby("type").size().reset_index(name="n")
    counts["pct"] = counts["n"] / counts["n"].sum()
    return counts


def _maturity_label(ratio: float) -> str:
    if ratio >= HIGH_MATURITY:
        return "HIGH"
    if ratio >= MEDIUM_MATURITY:
        return "MEDIUM"
    return "LOW"


def preventive_maturity(
    maint_df: pd.DataFrame, machines_df: pd.DataFrame, period: str = "M"
) -> pd.DataFrame:
    """Per-machine scheduled-maintenance ratio and maturity tier."""
    cols = [
        "machineID",
        "n_maintenance",
        "n_scheduled",
        "n_unscheduled",
        "schedule_ratio",
        "maturity",
    ]
    if maint_df.empty:
        return pd.DataFrame(columns=cols)
    grouped = maint_df.groupby("machineID")["type"].agg(
        n_maintenance="count",
        n_scheduled=lambda s: (s == "scheduled").sum(),
        n_unscheduled=lambda s: (s == "unscheduled").sum(),
    ).reset_index()
    grouped["schedule_ratio"] = grouped["n_scheduled"] / grouped["n_maintenance"]
    grouped["maturity"] = [
        _maturity_label(r) for r in grouped["schedule_ratio"]
    ]
    return grouped[cols]


def unscheduled_cost_penalty(cost_df: pd.DataFrame, maint_df: pd.DataFrame) -> pd.DataFrame:
    """Extra cost of unscheduled vs scheduled repair, per machine."""
    cols = [
        "machineID",
        "avg_cost_scheduled_Jt",
        "avg_cost_unscheduled_Jt",
        "penalty_pct",
    ]
    if cost_df.empty or maint_df.empty or "type" not in maint_df.columns:
        return pd.DataFrame(columns=cols)
    joined = maint_df[["datetime", "machineID", "type"]].merge(
        cost_df, on=["datetime", "machineID"], how="inner"
    )
    means = joined.groupby(["machineID", "type"])["total_repair_cost_Jt"].mean().unstack("type")
    sched = means["scheduled"] if "scheduled" in means.columns else pd.Series(0.0, dtype=float)
    unsched = means["unscheduled"] if "unscheduled" in means.columns else pd.Series(0.0, dtype=float)
    result = pd.DataFrame(
        {
            "machineID": means.index,
            "avg_cost_scheduled_Jt": sched.reindex(means.index).values,
            "avg_cost_unscheduled_Jt": unsched.reindex(means.index).values,
        }
    )
    result["penalty_pct"] = (
        (result["avg_cost_unscheduled_Jt"] - result["avg_cost_scheduled_Jt"])
        / result["avg_cost_scheduled_Jt"]
        * 100
    ).where(result["avg_cost_scheduled_Jt"] != 0).fillna(0.0)
    return result[cols]