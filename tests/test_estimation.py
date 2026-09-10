"""Tests for src/estimation.py"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimation import (
    estimate_duration,
    estimate_resources,
    build_timeline,
    summarize_engineering_cost,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _engineering_df() -> pd.DataFrame:
    rows = [
        {"job_type": "Overhaul Kompressor", "complexity": "Tinggi", "estimated_days": 10, "actual_days": 12,
         "crew_size": 3, "skill_required": "Senior", "tools": "Kunci Inggris", "spare_parts_needed": "Ya",
         "potential_delay_hours": 4, "parallel_allowed": "Ya", "buffer_pct": 0.2},
        {"job_type": "Overhaul Kompressor", "complexity": "Tinggi", "estimated_days": 10, "actual_days": 14,
         "crew_size": 4, "skill_required": "Senior", "tools": "Kunci Inggris", "spare_parts_needed": "Ya",
         "potential_delay_hours": 6, "parallel_allowed": "Ya", "buffer_pct": 0.2},
        {"job_type": "Overhaul Kompressor", "complexity": "Sedang", "estimated_days": 7, "actual_days": 8,
         "crew_size": 2, "skill_required": "Mid", "tools": "Obeng", "spare_parts_needed": "Tidak",
         "potential_delay_hours": 2, "parallel_allowed": "Ya", "buffer_pct": 0.15},
        {"job_type": "Ganti Bearing Motor", "complexity": "Sedang", "estimated_days": 3, "actual_days": 4,
         "crew_size": 2, "skill_required": "Mid", "tools": "Bearing Puller", "spare_parts_needed": "Ya",
         "potential_delay_hours": 2, "parallel_allowed": "Tidak", "buffer_pct": 0.1},
        {"job_type": "Ganti Bearing Motor", "complexity": "Sedang", "estimated_days": 3, "actual_days": 3,
         "crew_size": 2, "skill_required": "Mid", "tools": "Bearing Puller", "spare_parts_needed": "Ya",
         "potential_delay_hours": 1, "parallel_allowed": "Tidak", "buffer_pct": 0.1},
        {"job_type": "Ganti Bearing Motor", "complexity": "Rendah", "estimated_days": 2, "actual_days": 2,
         "crew_size": 1, "skill_required": "Junior", "tools": "Obeng", "spare_parts_needed": "Tidak",
         "potential_delay_hours": 0, "parallel_allowed": "Tidak", "buffer_pct": 0.05},
        {"job_type": "Inspeksi Visual", "complexity": "Rendah", "estimated_days": 1, "actual_days": 1,
         "crew_size": 1, "skill_required": "Junior", "tools": "Checklist", "spare_parts_needed": "Tidak",
         "potential_delay_hours": 0, "parallel_allowed": "Ya", "buffer_pct": 0.0},
        {"job_type": "Inspeksi Visual", "complexity": "Rendah", "estimated_days": 1, "actual_days": 1,
         "crew_size": 1, "skill_required": "Junior", "tools": "Checklist", "spare_parts_needed": "Tidak",
         "potential_delay_hours": 0, "parallel_allowed": "Ya", "buffer_pct": 0.0},
        {"job_type": "Kalibrasi Sensor", "complexity": "Sedang", "estimated_days": 2, "actual_days": 3,
         "crew_size": 2, "skill_required": "Mid", "tools": "Multimeter", "spare_parts_needed": "Tidak",
         "potential_delay_hours": 1, "parallel_allowed": "Ya", "buffer_pct": 0.1},
        {"job_type": "Kalibrasi Sensor", "complexity": "Sedang", "estimated_days": 2, "actual_days": 2,
         "crew_size": 2, "skill_required": "Mid", "tools": "Multimeter", "spare_parts_needed": "Ya",
         "potential_delay_hours": 1, "parallel_allowed": "Ya", "buffer_pct": 0.1},
        {"job_type": "Pengecekan Belt", "complexity": "Rendah", "estimated_days": 1, "actual_days": 1,
         "crew_size": 1, "skill_required": "Junior", "tools": "Checklist", "spare_parts_needed": "Tidak",
         "potential_delay_hours": 0, "parallel_allowed": "Ya", "buffer_pct": 0.0},
        {"job_type": "Pembersihan Filter", "complexity": "Rendah", "estimated_days": 1, "actual_days": 1,
         "crew_size": 1, "skill_required": "Junior", "tools": "Obeng", "spare_parts_needed": "Tidak",
         "potential_delay_hours": 0, "parallel_allowed": "Ya", "buffer_pct": 0.0},
        {"job_type": "Penggantian Fuse", "complexity": "Rendah", "estimated_days": 1, "actual_days": 1,
         "crew_size": 1, "skill_required": "Junior", "tools": "Obeng", "spare_parts_needed": "Ya",
         "potential_delay_hours": 0, "parallel_allowed": "Ya", "buffer_pct": 0.0},
        {"job_type": "Penggantian Filter", "complexity": "Rendah", "estimated_days": 1, "actual_days": 1,
         "crew_size": 1, "skill_required": "Junior", "tools": "Obeng", "spare_parts_needed": "Ya",
         "potential_delay_hours": 0, "parallel_allowed": "Ya", "buffer_pct": 0.0},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# estimate_duration
# ---------------------------------------------------------------------------

def test_estimate_duration_basic():
    df = _engineering_df()
    result = estimate_duration(df, "Overhaul Kompressor")
    assert result["job_type"] == "Overhaul Kompressor"
    assert result["n_samples"] == 3
    assert result["est_days"] == pytest.approx(9.0, rel=1e-3)
    assert result["act_days"] == pytest.approx(11.333, rel=1e-2)
    assert result["accuracy_ratio"] == pytest.approx(11.333 / 9.0, rel=1e-2)
    assert result["std_days"] > 0
    assert result["buffer_recommendation"] == pytest.approx(result["std_days"] * 1.5, rel=1e-3)


def test_estimate_duration_with_complexity_filter():
    df = _engineering_df()
    result = estimate_duration(df, "Overhaul Kompressor", complexity="Tinggi")
    assert result["complexity"] == "Tinggi"
    assert result["n_samples"] == 2
    assert result["est_days"] == pytest.approx(10.0, rel=1e-3)
    assert result["act_days"] == pytest.approx(13.0, rel=1e-3)


def test_estimate_duration_no_match():
    df = _engineering_df()
    result = estimate_duration(df, "Nonexistent Job")
    assert result["n_samples"] == 0
    assert result["est_days"] == 0
    assert result["act_days"] == 0


# ---------------------------------------------------------------------------
# estimate_resources
# ---------------------------------------------------------------------------

def test_estimate_resources():
    df = _engineering_df()
    result = estimate_resources(df, "Overhaul Kompressor")
    assert result["job_type"] == "Overhaul Kompressor"
    assert result["crew_size"] >= 1
    assert result["skill_required"] in ("Senior", "Mid", "Junior")
    assert 0 <= result["spare_parts_pct"] <= 1
    assert result["avg_delay_hours"] >= 0
    assert 0 <= result["parallel_pct"] <= 1


def test_estimate_resources_all_parallel():
    df = _engineering_df()
    result = estimate_resources(df, "Inspeksi Visual")
    assert result["parallel_pct"] == pytest.approx(1.0, rel=1e-3)


def test_estimate_resources_no_match():
    df = _engineering_df()
    result = estimate_resources(df, "Nonexistent Job")
    assert result["crew_size"] == 0


# ---------------------------------------------------------------------------
# build_timeline
# ---------------------------------------------------------------------------

def test_build_timeline_parallel():
    df = _engineering_df()
    job_types = [
        {"type": "Overhaul Kompressor", "quantity": 3},
    ]
    timeline, total_days, peak_crew = build_timeline(df, job_types)
    assert len(timeline) == 1
    row = timeline.iloc[0]
    assert row["parallel"] == True
    assert row["start_day"] == 0
    assert row["end_day"] > 0
    assert row["quantity"] == 3
    assert peak_crew >= 3


def test_build_timeline_sequential():
    df = _engineering_df()
    job_types = [
        {"type": "Ganti Bearing Motor", "quantity": 3},
    ]
    timeline, total_days, peak_crew = build_timeline(df, job_types)
    assert len(timeline) == 1
    row = timeline.iloc[0]
    assert row["parallel"] == False
    assert row["start_day"] == 0
    assert row["end_day"] == pytest.approx(row["est_days_each"] * 3, rel=1e-3)


def test_build_timeline_mixed():
    df = _engineering_df()
    job_types = [
        {"type": "Overhaul Kompressor", "quantity": 2},
        {"type": "Ganti Bearing Motor", "quantity": 3},
    ]
    timeline, total_days, peak_crew = build_timeline(df, job_types)
    assert len(timeline) == 2
    assert total_days > 0
    assert peak_crew >= 2


def test_build_timeline_empty():
    df = _engineering_df()
    timeline, total_days, peak_crew = build_timeline(df, [])
    assert len(timeline) == 0
    assert total_days == 0
    assert peak_crew == 0


# ---------------------------------------------------------------------------
# summarize_engineering_cost
# ---------------------------------------------------------------------------

def test_summarize_engineering_cost():
    df = _engineering_df()
    job_types = [
        {"type": "Overhaul Kompressor", "quantity": 2},
        {"type": "Ganti Bearing Motor", "quantity": 3},
    ]
    result = summarize_engineering_cost(df, job_types, labor_rate=0.8)
    assert "per_job_breakdown" in result
    assert "total_labor_cost_Jt" in result
    assert "total_days" in result
    assert len(result["per_job_breakdown"]) == 2
    assert result["total_labor_cost_Jt"] > 0
    # Verify formula: est_days * 8 * crew_size * labor_rate * quantity
    for job in result["per_job_breakdown"]:
        expected = job["est_days"] * 8 * job["crew_size"] * 0.8 * job["quantity"]
        assert job["labor_cost_Jt"] == pytest.approx(expected, rel=1e-3)


def test_summarize_engineering_cost_custom_rate():
    df = _engineering_df()
    job_types = [{"type": "Inspeksi Visual", "quantity": 5}]
    r1 = summarize_engineering_cost(df, job_types, labor_rate=0.5)
    r2 = summarize_engineering_cost(df, job_types, labor_rate=1.0)
    assert r2["total_labor_cost_Jt"] == pytest.approx(r1["total_labor_cost_Jt"] * 2, rel=1e-3)


def test_summarize_engineering_cost_empty():
    df = _engineering_df()
    result = summarize_engineering_cost(df, [], labor_rate=0.8)
    assert result["total_labor_cost_Jt"] == 0
    assert result["total_days"] == 0
