"""Equipment lifecycle projection module."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

EXPECTED_USEFUL_LIFE = 15


def trend_mtbf_over_time(
    failures_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    period: Literal["Q", "M"] = "Q",
) -> pd.DataFrame:
    """Pivot quarterly/monthly failure counts per machine."""
    df = failures_df.copy()
    df["period"] = df["datetime"].dt.to_period(period)
    counts = df.groupby(["machineID", "period"]).size().reset_index(name="failures")
    pivot = counts.pivot(index="machineID", columns="period", values="failures").fillna(0)
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot.reset_index()


def trend_cost_over_time(
    cost_df: pd.DataFrame,
    period: Literal["Q", "M"] = "Q",
) -> pd.DataFrame:
    """Pivot quarterly/monthly total cost per machine."""
    df = cost_df.copy()
    df["period"] = df["datetime"].dt.to_period(period)
    sums = df.groupby(["machineID", "period"])["total_repair_cost_Jt"].sum().reset_index()
    pivot = sums.pivot(index="machineID", columns="period", values="total_repair_cost_Jt").fillna(0)
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot.reset_index()


def _slope(series: pd.Series) -> float:
    """Linear regression slope via np.polyfit. Returns 0 if < 2 points."""
    vals = series.dropna().values.astype(float)
    if len(vals) < 2:
        return np.nan
    x = np.arange(len(vals), dtype=float)
    coeffs = np.polyfit(x, vals, deg=1)
    return float(coeffs[0])


def calculate_lifecycle_stage(
    machines_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    expected_useful_life: int = EXPECTED_USEFUL_LIFE,
) -> pd.DataFrame:
    """Determine lifecycle stage per machine."""
    machines = machines_df.copy()

    # Quarterly failure trend per machine
    ft = trend_mtbf_over_time(failures_df, machines_df, period="Q")
    # Quarterly cost trend per machine
    ct = trend_cost_over_time(cost_df, period="Q")

    # Compute slopes
    mtbf_slopes = {}
    cost_slopes = {}
    avg_mtbf = {}
    avg_cost = {}
    for _, row in ft.iterrows():
        mid = row["machineID"]
        period_vals = row.drop("machineID")
        mtbf_slopes[mid] = _slope(period_vals)
        vals = period_vals.dropna()
        avg_mtbf[mid] = float(vals.mean()) if len(vals) > 0 else np.nan

    for _, row in ct.iterrows():
        mid = row["machineID"]
        period_vals = row.drop("machineID")
        cost_slopes[mid] = _slope(period_vals)
        vals = period_vals.dropna()
        avg_cost[mid] = float(vals.mean()) if len(vals) > 0 else np.nan

    def _classify(row: pd.Series) -> str:
        mid = row["machineID"]
        age = row["age"]
        n_failures = len(failures_df[failures_df["machineID"] == mid])
        n_cost = len(cost_df[cost_df["machineID"] == mid])
        insufficient = n_failures < 3 or n_cost < 3

        mtbf_sl = mtbf_slopes.get(mid, np.nan)
        cost_sl = cost_slopes.get(mid, np.nan)

        def _is_degrading() -> bool:
            if not np.isnan(mtbf_sl) and mtbf_sl > 0.3:
                return True
            if not np.isnan(cost_sl) and cost_sl > 200:
                return True
            return False

        if insufficient:
            if age < 3:
                return "NEW"
            if 3 <= age < 10:
                return "STABLE"
            if 10 <= age < expected_useful_life:
                return "AGING"
            return "OLD"

        degrading = _is_degrading()
        very_high = n_failures >= 10 and age >= expected_useful_life

        if age > expected_useful_life and degrading:
            return "END_OF_LIFE"
        if very_high or (degrading and age >= expected_useful_life):
            return "CRITICAL"
        if degrading:
            return "DEGRADING"
        if age < 3:
            return "NEW"
        if 3 <= age < 10:
            return "STABLE"
        if 10 <= age < expected_useful_life:
            return "AGING"
        return "OLD"

    machines["avg_mtbf"] = machines["machineID"].map(avg_mtbf)
    machines["avg_cost_per_period"] = machines["machineID"].map(avg_cost)
    machines["mtbf_slope"] = machines["machineID"].map(mtbf_slopes)
    machines["cost_slope"] = machines["machineID"].map(cost_slopes)
    machines["mtbf_slope"] = machines["mtbf_slope"].fillna(0)
    machines["cost_slope"] = machines["cost_slope"].fillna(0)
    machines["avg_mtbf"] = machines["avg_mtbf"].fillna(0)
    machines["avg_cost_per_period"] = machines["avg_cost_per_period"].fillna(0)
    machines["stage"] = machines.apply(_classify, axis=1)

    return machines[[
        "machineID", "model", "age", "avg_mtbf", "avg_cost_per_period",
        "mtbf_slope", "cost_slope", "stage",
    ]]


def project_replacement_timeline(lifecycle_df: pd.DataFrame) -> pd.DataFrame:
    """Project months until END_OF_LIFE for CRITICAL/DEGRADING machines."""
    df = lifecycle_df.copy()
    df["remaining_life_months"] = df.apply(
        lambda r: 0 if r["stage"] == "END_OF_LIFE"
        else max(0, (EXPECTED_USEFUL_LIFE - r["age"]) * 12),
        axis=1,
    )

    def _urgency(months: int) -> str:
        if months <= 3:
            return "Immediate"
        if months <= 12:
            return "Near-term"
        if months <= 24:
            return "Medium"
        return "Long-term"

    df["urgency"] = df["remaining_life_months"].apply(_urgency)
    return df.sort_values("remaining_life_months").reset_index(drop=True)


def cost_projection(
    cost_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    periods_ahead: int = 6,
) -> pd.DataFrame:
    """Project future cost per machine based on average quarterly cost."""
    df = cost_df.copy()
    df["period"] = df["datetime"].dt.to_period("Q")
    grouped = df.groupby(["machineID", "period"])["total_repair_cost_Jt"].sum().reset_index()
    stats = grouped.groupby("machineID").agg(
        avg_quarterly_cost_Jt=("total_repair_cost_Jt", "mean"),
        n_quarters_data=("total_repair_cost_Jt", "count"),
    ).reset_index()
    stats["projected_cost"] = stats["avg_quarterly_cost_Jt"] * periods_ahead

    col_name = f"projected_{periods_ahead}q_cost_Jt"
    stats[col_name] = stats["projected_cost"]
    stats = stats.drop(columns=["projected_cost"])

    result = machines_df[["machineID"]].merge(stats, on="machineID", how="inner")
    return result[["machineID", "avg_quarterly_cost_Jt", col_name, "n_quarters_data"]]
