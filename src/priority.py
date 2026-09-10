"""Priority scoring engine for maintenance decision support."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series to [0, 1].

    Edge cases:
    - All values identical → returns 0.5 for every element.
    - NaN values are preserved as NaN in the output.

    Formula: norm(x) = (x - min) / (max - min)
    """
    lo = series.min()
    hi = series.max()
    if lo == hi:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def calculate_priority_score(
    availability_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    critical_machines: Optional[List[int]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Calculate a weighted priority score (0-100) per machine.

    Default weights:
        w_mtbf_inv  = 0.25  (lower MTBF → higher priority, inverted before norm)
        w_mttr      = 0.20  (higher MTTR → higher priority)
        w_freq      = 0.20  (higher failure frequency → higher priority)
        w_cost      = 0.20  (higher cost → higher priority)
        w_critical  = 0.15  (critical machine flag)

    Algorithm:
        1. Inner-join all four input DataFrames on machineID.
        2. Invert MTBF: mtbf_inv = 1 / mtbf_hours (no-failure machines get 0).
        3. Normalize each metric to [0, 1].
        4. Score = Σ(weight_i × normalized_metric_i).
        5. Scale to 0-100.
        6. Assign category: ≥75 URGENT, 50-74 HIGH, 25-49 MEDIUM, <25 LOW.

    Returns DataFrame with columns:
        machineID, model, age, mtbf_hours, mttr_hours, availability_pct,
        failures_per_month, total_cost_Jt, score, category
    """
    default_weights = {
        "w_mtbf_inv": 0.25,
        "w_mttr": 0.20,
        "w_freq": 0.20,
        "w_cost": 0.20,
        "w_critical": 0.15,
    }
    w = {**default_weights, **(weights or {})}

    critical_set = set(critical_machines or [])

    df = (
        availability_df
        .merge(frequency_df, on="machineID", how="inner")
        .merge(cost_df, on="machineID", how="inner")
        .merge(machines_df, on="machineID", how="inner")
    )

    # Invert MTBF: machines with no failures (NaN mtbf) → mtbf_inv = 0
    df["mtbf_inv"] = df["mtbf_hours"].apply(
        lambda x: 1.0 / x if pd.notna(x) and x > 0 else 0.0
    )

    df["norm_mtbf_inv"] = normalize_series(df["mtbf_inv"])
    df["norm_mttr"] = normalize_series(df["mttr_hours"])
    df["norm_freq"] = normalize_series(df["failures_per_month"])
    df["norm_cost"] = normalize_series(df["total_cost_Jt"])
    df["is_critical"] = df["machineID"].apply(lambda x: 1.0 if x in critical_set else 0.0)

    df["score"] = (
        w["w_mtbf_inv"] * df["norm_mtbf_inv"]
        + w["w_mttr"] * df["norm_mttr"]
        + w["w_freq"] * df["norm_freq"]
        + w["w_cost"] * df["norm_cost"]
        + w["w_critical"] * df["is_critical"]
    ) * 100.0

    df["category"] = pd.cut(
        df["score"],
        bins=[-float("inf"), 25, 50, 75, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH", "URGENT"],
    )

    return df[[
        "machineID", "model", "age", "mtbf_hours", "mttr_hours",
        "availability_pct", "failures_per_month", "total_cost_Jt",
        "score", "category",
    ]].reset_index(drop=True)


def get_top_priority(priority_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top N machines sorted by score descending."""
    return priority_df.nlargest(n, "score").reset_index(drop=True)


def get_by_category(priority_df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Filter machines by category string (LOW/MEDIUM/HIGH/URGENT)."""
    return priority_df[priority_df["category"] == category].reset_index(drop=True)
