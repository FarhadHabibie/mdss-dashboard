"""Repair vs. replacement decision engine."""
from __future__ import annotations

from typing import List, Optional

import pandas as pd


def calculate_repair_replacement_score(
    cost_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    failure_frequency_df: pd.DataFrame,
    critical_machines: Optional[List[int]] = None,
    expected_useful_life: int = 15,
) -> pd.DataFrame:
    """Score each machine on whether to repair or replace.

    Scoring rules (start at 0):
        cost_ratio < 0.5      → -30  (repair is cheap)
        0.5 ≤ ratio < 0.8     → -10
        0.8 ≤ ratio < 1.0     → +10
        ratio ≥ 1.0            → +30  (repair costs more)
        failures_per_month > 3 → +25
        age > useful_life      → +30
        is_critical            → +20
        avg_downtime > 20 hrs  → +15

    Decision thresholds:
        total_score ≥ 40      → REPLACE
        0 ≤ score < 40        → REVIEW
        score < 0             → REPAIR

    roi_years = avg_replacement_cost / (avg_repair_cost × 12)

    Returns DataFrame with columns:
        machineID, model, age, avg_repair_cost_Jt, avg_replacement_cost_Jt,
        cost_ratio, failures_per_month, avg_downtime_hours, total_score,
        decision, roi_years
    """
    critical_set = set(critical_machines or [])

    grouped = cost_df.groupby("machineID")
    summary = pd.DataFrame({
        "avg_repair_cost_Jt": grouped["total_repair_cost_Jt"].mean(),
        "avg_replacement_cost_Jt": grouped["replacement_cost_Jt"].mean(),
        "avg_downtime_hours": grouped["downtime_hours"].mean(),
    }).reset_index()

    summary["cost_ratio"] = summary["avg_repair_cost_Jt"] / summary["avg_replacement_cost_Jt"]

    df = (
        summary
        .merge(machines_df, on="machineID", how="inner")
        .merge(failure_frequency_df[["machineID", "failures_per_month"]], on="machineID", how="left")
    )
    df["failures_per_month"] = df["failures_per_month"].fillna(0.0)

    def _score(row: pd.Series) -> int:
        score = 0
        cr = row["cost_ratio"]
        if cr < 0.5:
            score -= 30
        elif cr < 0.8:
            score -= 10
        elif cr < 1.0:
            score += 10
        else:
            score += 30

        if row["failures_per_month"] > 3:
            score += 25
        if row["age"] > expected_useful_life:
            score += 30
        if row["machineID"] in critical_set:
            score += 20
        if row["avg_downtime_hours"] > 20:
            score += 15
        return score

    df["total_score"] = df.apply(_score, axis=1)

    def _decision(s: int) -> str:
        if s >= 40:
            return "REPLACE"
        if s >= 0:
            return "REVIEW"
        return "REPAIR"

    df["decision"] = df["total_score"].apply(_decision)

    df["roi_years"] = df["avg_replacement_cost_Jt"] / (df["avg_repair_cost_Jt"] * 12)

    return df[[
        "machineID", "model", "age", "avg_repair_cost_Jt",
        "avg_replacement_cost_Jt", "cost_ratio", "failures_per_month",
        "avg_downtime_hours", "total_score", "decision", "roi_years",
    ]].reset_index(drop=True)


def get_replacement_candidates(decision_df: pd.DataFrame) -> pd.DataFrame:
    """Filter REPLACE decisions, sorted by roi_years ascending (best ROI first)."""
    return (
        decision_df[decision_df["decision"] == "REPLACE"]
        .sort_values("roi_years")
        .reset_index(drop=True)
    )


def get_review_candidates(decision_df: pd.DataFrame) -> pd.DataFrame:
    """Filter REVIEW decisions."""
    return decision_df[decision_df["decision"] == "REVIEW"].reset_index(drop=True)
