"""Maintenance reliability and cost metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_MONTH = 730.0


def calculate_mtbf(
    failures_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Per-machine Mean Time Between Failures (hours)."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    window = (end - start).total_seconds() / 3600.0

    rows = []
    for mid in machines_df["machineID"]:
        times = failures_df[
            (failures_df["machineID"] == mid) & (failures_df["datetime"].between(start, end))
        ]["datetime"].sort_values().to_numpy()
        n = len(times)
        if n >= 2:
            intervals = np.diff(times).astype("timedelta64[h]").astype(float)
            mtbf = float(np.mean(intervals))
        else:
            mtbf = float("nan")
        rows.append({"machineID": mid, "n_failures": n, "mtbf_hours": mtbf,
                     "total_operating_hours": window})
    return pd.DataFrame(rows)


def calculate_mttr(cost_df: pd.DataFrame) -> pd.DataFrame:
    """Per-machine Mean Time To Repair (hours)."""
    grouped = cost_df.groupby("machineID")["downtime_hours"]
    return pd.DataFrame(
        {
            "machineID": grouped.count().index,
            "n_repairs": grouped.count().values,
            "mttr_hours": grouped.mean().values,
            "total_downtime_hours": grouped.sum().values,
        }
    )


def calculate_availability(mtbf_df: pd.DataFrame, mttr_df: pd.DataFrame) -> pd.DataFrame:
    """Availability (%) = mtbf / (mtbf + mttr) * 100."""
    merged = mtbf_df.merge(mttr_df, on="machineID", how="outer")
    merged["availability_pct"] = (
        merged["mtbf_hours"] / (merged["mtbf_hours"] + merged["mttr_hours"]) * 100
    )
    return merged[["machineID", "mtbf_hours", "mttr_hours", "availability_pct"]]


def calculate_failure_frequency(
    failures_df: pd.DataFrame, machines_df: pd.DataFrame
) -> pd.DataFrame:
    """Per-machine failures per month of age."""
    counts = failures_df.groupby("machineID").size()
    rows = []
    for mid, age in zip(machines_df["machineID"], machines_df["age"]):
        total = int(counts.get(mid, 0))
        age_months = float(age) * 12.0
        rows.append(
            {
                "machineID": mid,
                "total_failures": total,
                "machine_age_months": age_months,
                "failures_per_month": total / age_months if age_months else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calculate_maintenance_cost(cost_df: pd.DataFrame) -> pd.DataFrame:
    """Per-machine cost summary."""
    grouped = cost_df.groupby("machineID")["total_repair_cost_Jt"]
    return pd.DataFrame(
        {
            "machineID": grouped.sum().index,
            "total_cost_Jt": grouped.sum().values,
            "avg_cost_Jt": grouped.mean().values,
            "max_cost_Jt": grouped.max().values,
            "n_incidents": grouped.count().values,
        }
    )


def cost_per_downtime_hour(cost_df: pd.DataFrame) -> pd.DataFrame:
    """Per-machine total cost / total downtime hours."""
    grouped = cost_df.groupby("machineID")
    total_cost = grouped["total_repair_cost_Jt"].sum()
    total_downtime = grouped["downtime_hours"].sum()
    return pd.DataFrame(
        {
            "machineID": total_cost.index,
            "cost_per_hour_Jt": (total_cost / total_downtime).values,
        }
    )
