"""Engineering duration and resource estimation module."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def estimate_duration(
    engineering_df: pd.DataFrame,
    job_type: str,
    complexity: str | None = None,
) -> dict[str, Any]:
    """Estimate duration stats for a given job type (optionally filtered by complexity)."""
    mask = engineering_df["job_type"] == job_type
    if complexity is not None:
        mask = mask & (engineering_df["complexity"] == complexity)
    subset = engineering_df[mask]

    n = len(subset)
    if n == 0:
        return {
            "job_type": job_type,
            "complexity": complexity,
            "est_days": 0.0,
            "act_days": 0.0,
            "accuracy_ratio": 0.0,
            "std_days": 0.0,
            "buffer_recommendation": 0.0,
            "n_samples": 0,
        }

    est_days = float(subset["estimated_days"].mean())
    act_days = float(subset["actual_days"].mean())
    accuracy_ratio = act_days / est_days if est_days > 0 else 0.0
    std_days = float(subset["actual_days"].std()) if n > 1 else 0.0
    buffer_recommendation = std_days * 1.5

    return {
        "job_type": job_type,
        "complexity": complexity,
        "est_days": est_days,
        "act_days": act_days,
        "accuracy_ratio": accuracy_ratio,
        "std_days": std_days,
        "buffer_recommendation": buffer_recommendation,
        "n_samples": n,
    }


def estimate_resources(
    engineering_df: pd.DataFrame,
    job_type: str,
) -> dict[str, Any]:
    """Estimate typical resources for a job type."""
    subset = engineering_df[engineering_df["job_type"] == job_type]
    n = len(subset)

    if n == 0:
        return {
            "job_type": job_type,
            "crew_size": 0,
            "skill_required": "",
            "tools": "",
            "spare_parts_pct": 0.0,
            "avg_delay_hours": 0.0,
            "parallel_pct": 0.0,
        }

    crew_size = math.ceil(float(subset["crew_size"].mean()))
    skill_required = subset["skill_required"].mode().iloc[0] if not subset["skill_required"].mode().empty else ""
    tools = subset["tools"].mode().iloc[0] if not subset["tools"].mode().empty else ""
    spare_parts_pct = float((subset["spare_parts_needed"] == "Ya").mean())
    avg_delay_hours = float(subset["potential_delay_hours"].mean())
    parallel_pct = float((subset["parallel_allowed"] == "Ya").mean())

    return {
        "job_type": job_type,
        "crew_size": crew_size,
        "skill_required": skill_required,
        "tools": tools,
        "spare_parts_pct": spare_parts_pct,
        "avg_delay_hours": avg_delay_hours,
        "parallel_pct": parallel_pct,
    }


def build_timeline(
    engineering_df: pd.DataFrame,
    job_types: list[dict[str, Any]],
) -> tuple[pd.DataFrame, int, int]:
    """Build Gantt-style timeline. Returns (timeline_df, total_project_days, total_crew_peak)."""
    if not job_types:
        return pd.DataFrame(), 0, 0

    rows = []
    current_day = 0
    peak_crew = 0

    for jt in job_types:
        jtype = jt["type"]
        qty = jt["quantity"]
        res = estimate_resources(engineering_df, jtype)
        dur = estimate_duration(engineering_df, jtype)
        est_each = dur["est_days"]
        crew = res["crew_size"]
        parallel = res["parallel_pct"] > 0.5

        if parallel:
            start = current_day
            end = start + est_each
            total_days = est_each
        else:
            start = current_day
            end = start + est_each * qty
            total_days = est_each * qty

        current_day = end
        peak_crew = max(peak_crew, crew * (qty if parallel else 1))

        rows.append({
            "job_type": jtype,
            "quantity": qty,
            "est_days_each": est_each,
            "total_days": total_days,
            "start_day": start,
            "end_day": end,
            "parallel": parallel,
            "crew_needed": crew * (qty if parallel else 1),
        })

    timeline = pd.DataFrame(rows)
    return timeline, int(current_day), peak_crew


def summarize_engineering_cost(
    engineering_df: pd.DataFrame,
    job_types: list[dict[str, Any]],
    labor_rate: float = 0.8,
) -> dict[str, Any]:
    """Summarize labor cost for planned jobs."""
    if not job_types:
        return {"per_job_breakdown": [], "total_labor_cost_Jt": 0.0, "total_days": 0}

    breakdown = []
    total_cost = 0.0
    total_days = 0

    for jt in job_types:
        jtype = jt["type"]
        qty = jt["quantity"]
        dur = estimate_duration(engineering_df, jtype)
        res = estimate_resources(engineering_df, jtype)
        crew = res["crew_size"]
        est_each = dur["est_days"]

        labor_cost = est_each * 8 * crew * labor_rate * qty
        total_cost += labor_cost
        total_days += est_each * qty

        breakdown.append({
            "job_type": jtype,
            "quantity": qty,
            "est_days": est_each,
            "crew_size": crew,
            "labor_cost_Jt": labor_cost,
        })

    return {
        "per_job_breakdown": breakdown,
        "total_labor_cost_Jt": total_cost,
        "total_days": total_days,
    }
