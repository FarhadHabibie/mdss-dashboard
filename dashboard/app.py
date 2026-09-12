#!/usr/bin/env python3
"""Phase 4 Maintenance Decision Support System — Web Dashboard.

Single-file stdlib HTTP server. No Flask, no external deps.
Usage:
    python dashboard/app.py --port 8323 --data-dir data --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_all, clean_data
from src.metrics import (
    calculate_mtbf,
    calculate_mttr,
    calculate_availability,
    calculate_failure_frequency,
    calculate_maintenance_cost,
)
from src.priority import calculate_priority_score, get_top_priority, get_by_category
from src.decision import calculate_repair_replacement_score
from src.projection import (
    calculate_lifecycle_stage,
    project_replacement_timeline,
    trend_cost_over_time,
)
from src.estimation import estimate_duration, estimate_resources
from src.forecast import project_failures, project_cost_trend
from src.components import (
    component_failure_summary,
    component_cost_summary,
    component_mtbf,
    component_risk_score,
)
from src.error_analysis import error_before_failure, predictor_summary
from src.maintenance_type import maintenance_type_ratio, preventive_maturity
from src.telemetry import sensor_trend, anomaly_scores, early_warning


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, coercing NaN to None."""
    return json.loads(df.to_json(orient="records"))


def _safe_float(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return v


def build_payload(data_dir: str) -> dict:
    """Compute all metrics once and return the full API payload."""
    raw = load_all(data_dir)
    data = clean_data(raw)
    machines = data["machines"]
    failures = data["failures"]
    cost = data["cost"]
    engineering = data["engineering"]
    errors = data["errors"]
    maintenance = data["maintenance"]
    telemetry = data["telemetry"]

    date_min = str(failures["datetime"].min().date())
    date_max = str(failures["datetime"].max().date())

    mtbf = calculate_mtbf(failures, machines, date_min, date_max)
    mttr = calculate_mttr(cost)
    availability = calculate_availability(mtbf, mttr)
    frequency = calculate_failure_frequency(failures, machines)
    maint_cost = calculate_maintenance_cost(cost)
    priority = calculate_priority_score(availability, frequency, maint_cost, machines)
    decision = calculate_repair_replacement_score(cost, machines, frequency)
    lifecycle = calculate_lifecycle_stage(machines, failures, cost)
    timeline = project_replacement_timeline(lifecycle)

    # --- new analysis modules ---
    component_fail = component_failure_summary(failures)
    component_cost = component_cost_summary(cost)
    component_mtbf_df = component_mtbf(failures, machines)
    component_risk = component_risk_score(component_fail, component_cost)

    forecast_fail = project_failures(failures, machines)
    forecast_cost = project_cost_trend(cost, machines)

    err_before = error_before_failure(errors, failures)
    pred_summary = predictor_summary(errors, failures)

    maint_ratio = maintenance_type_ratio(maintenance)
    preventive = preventive_maturity(maintenance, machines)

    sensor_trend_df = sensor_trend(telemetry, machines)
    anomaly = anomaly_scores(telemetry, failures, machines)
    warning = early_warning(failures, telemetry)

    priority_sorted = priority.sort_values("score", ascending=False).reset_index(drop=True)

    avg_mtbf = _safe_float(availability["mtbf_hours"].mean())
    avg_mttr = _safe_float(availability["mttr_hours"].mean())
    avg_avail = _safe_float(availability["availability_pct"].mean())
    total_cost_val = _safe_float(maint_cost["total_cost_Jt"].sum())
    total_failures_val = int(frequency["total_failures"].sum())

    urgent_count = int((priority["category"] == "URGENT").sum())
    high_count = int((priority["category"] == "HIGH").sum())
    replace_count = int((decision["decision"] == "REPLACE").sum())
    review_count = int((decision["decision"] == "REVIEW").sum())
    repair_count = int((decision["decision"] == "REPAIR").sum())

    job_types = sorted(engineering["job_type"].unique().tolist())

    at_risk_count = int((anomaly["at_risk"] == True).sum()) if len(anomaly) else 0
    unsched_pct = 0.0
    _mratio = maint_ratio.set_index("type")["pct"] if len(maint_ratio) else None
    if _mratio is not None and "unscheduled" in _mratio.index:
        unsched_pct = float(_mratio.get("unscheduled", 0.0))
    error_coverage = pred_summary.get("coverage", 0.0) if isinstance(pred_summary, dict) else 0.0

    return {
        "summary": {
            "total_machines": int(len(machines)),
            "avg_mtbf": avg_mtbf,
            "avg_mttr": avg_mttr,
            "avg_availability": avg_avail,
            "total_failures": total_failures_val,
            "total_maintenance_cost": total_cost_val,
            "urgent_count": urgent_count,
            "high_count": high_count,
            "replace_count": replace_count,
            "review_count": review_count,
            "repair_count": repair_count,
            "at_risk_count": at_risk_count,
            "unscheduled_pct": unsched_pct,
            "error_coverage": error_coverage,
        },
        "priority": _df_to_records(priority_sorted),
        "decision": _df_to_records(decision),
        "lifecycle": _df_to_records(lifecycle),
        "replacement_timeline": _df_to_records(timeline),
        "machines": _df_to_records(machines[["machineID", "model", "age"]].sort_values("machineID")),
        "engineering_jobs": job_types,
        # new modules
        "components": _df_to_records(component_risk),
        "component_mtbf": _df_to_records(component_mtbf_df),
        "forecast_failures": _df_to_records(forecast_fail),
        "forecast_cost": _df_to_records(forecast_cost),
        "error_analysis": _df_to_records(err_before),
        "error_predictor": pred_summary,
        "maintenance_ratio": _df_to_records(maint_ratio),
        "preventive": _df_to_records(preventive),
        "telemetry_trend": _df_to_records(sensor_trend_df),
        "anomaly": _df_to_records(anomaly),
        "early_warning": _df_to_records(warning),
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MDSS — Maintenance Decision Support System</title>
 <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 32 32'%3E%3Ctext x='16' y='24' text-anchor='middle' font-family='system-ui' font-weight='800' font-size='20' fill='%234f46e5'%3EM%3C/text%3E%3C/svg%3E">
 <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
/* ============================================================
   MDSS — Enterprise Command Center (light premium palette)
   Design tokens, layers, spacing scale 8px
   ============================================================ */
:root{
  /* surfaces — light premium */
  --bg:#eef1f9;
  --surface-1:#ffffff;
  --surface-2:#f4f6fc;
  --surface-3:#e9edf7;
  --border:#e4e8f2;
  --border-strong:#cdd5e8;

/* text */
   --text-hi:#161d33;
   --text:#3f4759;
   --text-mid:#576178;
   --text-low:#6b7280;

  /* brand */
  --brand:#4f46e5;
  --brand-2:#7c3aed;
  --brand-3:#2563eb;

  /* status */
  --urgent:#d92d20;  --urgent-bg:rgba(217,45,32,.10);
  --high:#ea580c;    --high-bg:rgba(234,88,12,.12);
  --medium:#b45309;  --medium-bg:rgba(180,83,9,.12);
  --low:#056347;     --low-bg:rgba(4,120,87,.12);
  --new:#075985;     --new-bg:rgba(3,105,161,.12);
  --stable:#056347;  --stable-bg:rgba(4,120,87,.12);
  --aging:#a16207;   --aging-bg:rgba(180,83,9,.12);
  --degrading:#ea580c;--degrading-bg:rgba(234,88,12,.12);
  --critical:#d92d20;--critical-bg:rgba(217,45,32,.10);
  --eol:#b42318;     --eol-bg:rgba(180,35,24,.12);

  /* status — REPLACE is neutral (swap action), never red */
  --replace:#46569e; --replace-bg:rgba(70,86,158,.12);

  --r-sm:8px; --r-md:12px; --r-lg:18px;
  --shadow-sm:0 1px 2px rgba(22,29,51,.05);
  --shadow:0 6px 16px -4px rgba(22,29,51,.10),0 2px 4px rgba(22,29,51,.04);
  --shadow-lg:0 20px 40px -12px rgba(22,29,51,.18),0 8px 16px -6px rgba(22,29,51,.08);
  --teal:#0d9488;
}
*{margin:0;padding:0;box-sizing:border-box}
button{font-family:inherit}
html{scroll-behavior:smooth}
body{
  background:var(--bg);
  color:var(--text);font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
  line-height:1.5;font-size:14px;-webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;
}

/* --- brand bar --- */
.navbar{
  position:sticky;top:0;z-index:50;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 30px;height:64px;
  background:rgba(255,255,255,.82);backdrop-filter:blur(16px) saturate(1.4);
  -webkit-backdrop-filter:blur(16px) saturate(1.4);
  border-bottom:1px solid var(--border);
  box-shadow:0 1px 0 rgba(255,255,255,.6) inset,0 8px 24px -12px rgba(22,29,51,.10);
}
.brand{display:flex;align-items:center;gap:12px}
.brand-mark{
  width:34px;height:34px;border-radius:10px;
  background:var(--brand);
  display:flex;align-items:center;justify-content:center;
  color:#fff;font-weight:800;font-size:16px;letter-spacing:.5px;
}
.brand-name{font-weight:800;color:var(--text-hi);font-size:15px;letter-spacing:-.2px}
.brand-sub{color:var(--text-mid);font-size:11px;letter-spacing:.6px;text-transform:uppercase;font-weight:600}
.nav-status{display:flex;align-items:center;gap:8px}
.live-dot{width:8px;height:8px;border-radius:50%;background:var(--teal)}
.nav-status span{color:var(--text-mid);font-size:12px;font-weight:500}

.wrap{max-width:1440px;margin:0 auto;padding:24px 28px 48px}
.page-head{margin:14px 0 26px}
.page-head h1{
  font-size:27px;font-weight:800;color:var(--text-hi);letter-spacing:-.6px;
  text-wrap:balance;
}
.page-head p{color:var(--text-mid);font-size:13.5px;margin-top:6px;max-width:720px;line-height:1.6}
.text-low{color:var(--text-low)}

/* --- needs action banner --- */
.action-banner{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0 22px}
.action-banner[hidden]{display:none}
.action-card{
  flex:1;min-width:240px;display:flex;align-items:center;gap:12px;
  background:var(--surface-1);border:1px solid var(--border);border-radius:var(--r-md);
  padding:16px 18px;text-decoration:none;color:var(--text);box-shadow:var(--shadow);
  transition:box-shadow .15s,border-color .15s;
}
.action-card:hover{box-shadow:var(--shadow-lg);border-color:var(--border-strong)}
.action-card .icon{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:17px;flex:none}
.action-card.urgent .icon{background:var(--urgent-bg);color:var(--urgent)}
.action-card.warn .icon{background:var(--medium-bg);color:var(--medium)}
.action-card .title{font-size:13.5px;font-weight:700;color:var(--text-hi)}
.action-card .sub{font-size:12px;color:var(--text-mid);margin-top:2px}
.action-card .arrow{margin-left:auto;color:var(--text-mid);font-size:15px}

/* --- stat cards --- */
.stats-row{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:24px}
.stat-card{
   position:relative;overflow:hidden;
   background:#fff;
   border:1px solid var(--border);
   border-radius:var(--r-md);padding:20px 20px 18px;
   box-shadow:var(--shadow);transition:box-shadow .15s,border-color .15s;
 }
 .stat-card:hover{box-shadow:var(--shadow-lg);border-color:var(--border-strong)}
.stat-card::before{
  content:"";position:absolute;left:0;top:18px;bottom:18px;width:1px;border-radius:1px;
  background:var(--brand);opacity:.6;
}
.stat-card .label{font-size:11px;color:var(--text-mid);text-transform:uppercase;
  letter-spacing:.7px;font-weight:700}
.stat-card .value{font-variant-numeric:tabular-nums;font-size:28px;font-weight:800;
  color:var(--text-hi);margin-top:9px;letter-spacing:-.8px}
.stat-card .sub{font-size:11.5px;color:var(--text-low);margin-top:3px}
.stat-card.tone-teal::before{background:#0d9488}
.stat-card.tone-info::before{background:#0284c7}
.stat-card.tone-amber::before{background:#b45309}

/* --- sections --- */
.section{
  background:#fff;
  border:1px solid var(--border);border-radius:var(--r-lg);
  padding:24px 26px;margin-bottom:24px;box-shadow:var(--shadow);
  transition:box-shadow .15s;
}
.section:hover{box-shadow:var(--shadow-lg)}
.section-head{display:flex;align-items:flex-start;justify-content:space-between;
  margin-bottom:4px;flex-wrap:wrap;gap:10px}
.section-head h2{
  display:flex;align-items:center;gap:10px;
  font-size:16.5px;font-weight:750;color:var(--text-hi);letter-spacing:-.25px;
  text-wrap:balance;
}
.section-head h2::before{content:"";width:3px;height:16px;border-radius:2px;background:var(--brand)}
.section-head .hint{font-size:12px;color:var(--text-mid)}

/* unified filter component (pills + chips) */
.priority-bar,.chip-row{display:flex;gap:8px;margin:14px 0 16px;flex-wrap:wrap}
.pill,.chip{
  padding:7px 15px;border-radius:9px;font-size:12.5px;font-weight:650;
  cursor:pointer;border:1px solid var(--border);transition:color .18s,border-color .18s,box-shadow .18s,background .18s;user-select:none;
  background:#fff;color:var(--text-mid);box-shadow:var(--shadow-sm);
}
.pill:hover,.chip:hover{color:var(--text-hi);border-color:var(--border-strong);box-shadow:var(--shadow)}
.pill.active,.chip.active{box-shadow:0 0 0 1.5px var(--border-strong)}
.pill-urgent,.chip-urgent{background:var(--urgent-bg);color:var(--urgent)}
.pill-urgent.active,.chip-urgent.active{border-color:var(--urgent)}
.pill-high,.chip-high{background:var(--high-bg);color:var(--high)}
.pill-high.active,.chip-high.active{border-color:var(--high)}
.pill-medium,.chip-medium{background:var(--medium-bg);color:var(--medium)}
.pill-medium.active,.chip-medium.active{border-color:var(--medium)}
.pill-low,.chip-low{background:var(--low-bg);color:var(--low)}
.pill-low.active,.chip-low.active{border-color:var(--low)}
.pill-all,.chip-all{background:var(--surface-2);color:var(--text-hi)}
.pill-all.active,.chip-all.active{border-color:var(--brand);color:var(--brand)}
.reset-btn{
  padding:7px 15px;border-radius:9px;font-size:12.5px;font-weight:650;cursor:pointer;
  border:1px solid var(--border);background:#fff;color:var(--text-mid);
  transition:color .18s,border-color .18s,box-shadow .18s;user-select:none;
}
.reset-btn:hover{color:var(--urgent);border-color:var(--urgent);box-shadow:var(--shadow)}
.reset-btn:focus-visible,.pill:focus-visible,.chip:focus-visible{
  outline:2px solid var(--brand);outline-offset:2px;
}
table{width:100%;border-collapse:collapse;font-size:13px}
thead{position:sticky;top:0;z-index:2}
th{
  text-align:left;padding:12px 18px;color:var(--text-mid);font-weight:700;
  border-bottom:1px solid var(--border-strong);font-size:11px;
  text-transform:uppercase;letter-spacing:.6px;
  background:#f0f3fa;white-space:normal;line-height:1.35;
}
td{padding:12px 18px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums;
  color:var(--text)}
tr:last-child td{border-bottom:none}
tbody tr{transition:background .12s}
tbody tr:hover td{background:var(--surface-2)}

/* badges with dot */
.badge{display:inline-flex;align-items:center;gap:6px;
  padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;
  letter-spacing:.3px;text-transform:uppercase}
.badge::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.badge-urgent{background:var(--urgent-bg);color:var(--urgent)}
.badge-high{background:var(--high-bg);color:var(--high)}
.badge-medium{background:var(--medium-bg);color:var(--medium)}
.badge-low{background:var(--low-bg);color:var(--low)}
.badge-replace{background:var(--replace-bg);color:var(--replace)}
.badge-review{background:var(--medium-bg);color:var(--medium)}
.badge-repair{background:var(--low-bg);color:var(--low)}
.badge-new{background:var(--new-bg);color:var(--new)}
.badge-stable{background:var(--stable-bg);color:var(--stable)}
.badge-aging{background:var(--aging-bg);color:var(--aging)}
.badge-degrading{background:var(--degrading-bg);color:var(--degrading)}
.badge-critical{background:var(--critical-bg);color:var(--critical)}
.badge-end_of_life{background:var(--eol-bg);color:var(--eol)}
.badge-old{background:var(--surface-2);color:var(--text-mid)}

/* charts */
.charts-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:24px}
.chart-card{
  background:#fff;
  border:1px solid var(--border);border-radius:var(--r-lg);
  padding:20px 20px 14px;box-shadow:var(--shadow);transition:box-shadow .15s,transform .15s;
}
.chart-card:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px)}
.chart-card .chart-title{font-size:13px;font-weight:750;color:var(--text-hi);margin-bottom:16px;
  padding-left:11px;border-left:1px solid var(--brand);letter-spacing:.1px}
.chart-fallback{color:var(--text-mid);text-align:center;padding:48px 20px;font-size:13px}

/* footer */
.footer{text-align:center;padding:28px 0 10px;color:var(--text-mid);font-size:12px;
  border-top:1px solid var(--border);margin-top:8px;letter-spacing:.2px}
.footer .sep{padding:0 8px;color:var(--text-low)}

/* responsive */


/* cost trend chips */

/* --- entrance animations --- */
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.stats-row{animation:fadeUp .35s cubic-bezier(.2,.8,.2,1) both}
.charts-row{animation:fadeUp .4s cubic-bezier(.2,.8,.2,1) .08s both}
.section{animation:fadeUp .45s cubic-bezier(.2,.8,.2,1) .16s both}
.priority-bar,.chip-row{margin:14px 0 16px}

/* focus-visible for interactive elements */
.pill:focus-visible,.chip:focus-visible{
  outline:2px solid var(--brand);outline-offset:2px;
}
table{border-collapse:separate;border-spacing:0}
th,td{outline:none}
.table-wrap:focus-within{box-shadow:0 0 0 2px var(--brand)}

@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{animation-duration:.01ms !important;animation-iteration-count:1 !important;transition-duration:.01ms !important;scroll-behavior:auto !important}
}

@media(max-width:1100px){.stats-row{grid-template-columns:repeat(3,1fr)}.charts-row{grid-template-columns:1fr}}
@media(max-width:700px){.stats-row{grid-template-columns:repeat(2,1fr)}.navbar{padding:0 16px}.wrap{padding:16px}}
@media(max-width:480px){.stats-row{grid-template-columns:1fr}}
 .info-btn{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;border:1px solid var(--border-strong);background:#fff;color:var(--text-mid);font-size:12px;cursor:pointer;line-height:1;vertical-align:middle;margin-left:6px;padding:0}
 .info-btn:hover{border-color:var(--brand);color:var(--brand)}
 .info-panel{position:absolute;z-index:20;background:#fff;border:1px solid var(--border-strong);border-radius:var(--r-sm);padding:10px 14px;max-width:340px;font-size:12px;color:var(--text);box-shadow:var(--shadow-lg);line-height:1.5;display:none}
 .info-panel.open{display:block}
.section-head{position:relative}
 .trend{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:.2px}
.t-up{background:rgba(217,45,32,.10);color:#d92d20}
.t-down{background:rgba(5,99,71,.10);color:#056347}
.t-flat{background:rgba(107,113,140,.10);color:#67718c}
</style>
</head>
<body>
  <nav class="navbar">
<div class="brand">
       <div class="brand-mark">M</div>
       <div>
         <div class="brand-name">MDSS</div>
         <div class="brand-sub">Maintenance Decision Support</div>
       </div>
     </div>
    <div class="nav-status"><span class="live-dot"></span><span>Live Data &middot; 100 mesin</span></div>
  </nav>

  <div class="wrap">
    <div class="page-head">
      <h1>Command Center Maintenance</h1>
      <p>Analisis prioritas, keputusan repair vs replacement, proyeksi lifecycle, dan planning resource &mdash; berbasis data historis 100 mesin.</p>
    </div>

    <!-- action banner -->
    <div id="needs-action" class="action-banner" style="display:none;"></div>

    <div id="stats" class="stats-row"></div>
     <div class="data-timestamp" style="text-align:center;color:var(--text-mid);font-size:12px;margin-top:8px;">Data per 12 Sep 2026</div>

    <div class="charts-row">
<div class="chart-card"><h2 class="chart-title">Top 10 Priority</h2><canvas id="chart-priority" role="img" aria-label="Grafik batang 10 mesin prioritas tertinggi dengan skor 0-100"></canvas></div>
     <div class="chart-card"><h2 class="chart-title">Lifecycle Distribution</h2><canvas id="chart-lifecycle" role="img" aria-label="Grafik lingkaran distribusi lifecycle stage mesin (Baru, Stabil, Menua, Degradasi, Kritis, Akhir Umur)"></canvas></div>
     <div class="chart-card"><h2 class="chart-title">Decision Distribution</h2><canvas id="chart-decision" role="img" aria-label="Grafik donat keputusan repair vs replace (REPLACE, REVIEW, REPAIR)"></canvas></div>
    </div>

    <div id="priority-section" class="section">
      <div class="section-head">
        <h2>Priority Mesin <button type="button" class="info-btn" aria-expanded="false" aria-label="Info metodologi skor prioritas" onclick="toggleInfo(this)">&#9432;</button></h2>
        <span class="hint">Skor prioritas 0&ndash;100 &mdash; URGENT &ge;75 &middot; HIGH 50&ndash;74 &middot; MEDIUM 25&ndash;49 &middot; LOW &lt;25</span>
        <div class="info-panel">Skor = bobot biaya 40%, frekuensi 25%, umur 20%, criticality 15% (skala 0&ndash;100).</div>
      </div>
      <div id="priority-pills" class="priority-bar"></div>
      <div class="table-wrap">
        <table id="priority-table" aria-label="Prioritas mesin">
          <thead><tr>
<th>MachineID</th><th>Model</th><th>Age</th><th>MTBF (hr)</th>
             <th>MTTR (hr)</th><th>Avail %</th><th>Gagal/bln</th><th>Cost Trend</th><th>Biaya Total (Jt)</th>
             <th>Score</th><th>Category</th>
          </tr></thead>
          <tbody id="priority-body"></tbody>
        </table>
      </div>
    </div>

    <div id="decision-section" class="section">
      <div class="section-head">
        <h2>Keputusan Repair vs Replace <button type="button" class="info-btn" aria-expanded="false" aria-label="Info metodologi keputusan" onclick="toggleInfo(this)">&#9432;</button></h2>
        <span class="hint">Berdasarkan cost ratio, frekuensi, umur, criticality &amp; downtime</span>
        <div class="info-panel">Keputusan dari rasio biaya repair vs replacement, frekuensi breakdown, umur, dan criticality mesin.</div>
      </div>
      <div id="decision-chips" class="chip-row"></div>
      <div class="table-wrap">
        <table id="decision-table" aria-label="Keputusan repair vs replace">
          <thead><tr>
            <th>MachineID</th><th>Model</th><th>Age</th><th>Cost Ratio</th>
            <th>Score</th><th>Decision</th>
          </tr></thead>
          <tbody id="decision-body"></tbody>
        </table>
      </div>
    </div>

    <div id="lifecycle-section" class="section">
      <div class="section-head">
        <h2>Lifecycle Stage <button type="button" class="info-btn" aria-expanded="false" aria-label="Info metodologi lifecycle" onclick="toggleInfo(this)">&#9432;</button></h2>
        <span class="hint">Regresi tren failure &amp; biaya per kuartal + umur equipment</span>
        <div class="info-panel">Kategori dari tren MTBF, biaya, dan sisa umur (Baru, Stabil, Menua, Degradasi, Kritis, Akhir Umur).</div>
      </div>
      <div id="lifecycle-chips" class="chip-row"></div>
      <div class="table-wrap">
        <table id="lifecycle-table" aria-label="Lifecycle stage">
          <thead><tr>
            <th>MachineID</th><th>Model</th><th>Age</th><th>Stage</th>
          </tr></thead>
          <tbody id="lifecycle-body"></tbody>
        </table>
      </div>
    </div>

    <div id="timeline-section" class="section">
      <div class="section-head">
        <h2>Proyeksi Penggantian <button type="button" class="info-btn" aria-expanded="false" aria-label="Info metodologi proyeksi" onclick="toggleInfo(this)">&#9432;</button></h2>
        <span class="hint">Urut berdasar sisa umur &mdash; prioritas penggantian equipment</span>
        <div class="info-panel">Proyeksi linear dari tren biaya dan frekuensi failure historis.</div>
      </div>
      <div class="table-wrap">
        <table id="timeline-table" aria-label="Proyeksi penggantian">
          <thead><tr>
            <th>MachineID</th><th>Model</th><th>Stage</th>
            <th>Sisa Umur (bln)</th><th>Urgency</th>
          </tr></thead>
          <tbody id="timeline-body"></tbody>
        </table>
      </div>
    </div>

    <div id="components-section" class="section">
      <div class="section-head">
        <h2>Komponen &amp; Health <button type="button" class="info-btn" aria-expanded="false" aria-label="Info metodologi anomaly" onclick="toggleInfo(this)">&#9432;</button></h2>
        <span class="hint">Frekuensi failure per komponen, biaya, MTBF &mdash; sensor telemetry trend + anomaly</span>
        <div class="info-panel">Anomaly = deviasi sensor (voltase, rotasi, tekanan, vibrasi) dari baseline mesin.</div>
      </div>
      <div class="charts-row" style="grid-template-columns:1fr 1fr 1fr;margin-bottom:18px">
<div class="chart-card"><h2 class="chart-title">Komponen Risk</h2><canvas id="chart-component" role="img" aria-label="Grafik batang risk score tiap komponen"></canvas></div>
         <div class="chart-card"><h2 class="chart-title">Telemetry Anomaly</h2><canvas id="chart-anomaly" role="img" aria-label="Grafik batang anomaly score 20 mesin teratas"></canvas></div>
         <div class="chart-card"><h2 class="chart-title">Error hit-rate (14d)</h2><canvas id="chart-error" role="img" aria-label="Grafik batang hit rate error predictor 8 error teratas"></canvas></div>
      </div>
      <div class="table-wrap" style="margin-bottom:14px">
        <table id="component-table" aria-label="Analisa komponen">
          <thead><tr><th>Komponen</th><th>Failures</th><th>% Total</th><th>Biaya Total (Jt)</th><th>Risk Score</th><th>Risk Level</th></tr></thead>
          <tbody id="component-body"></tbody>
        </table>
      </div>
      <div class="table-wrap">
        <table id="telemetry-table" aria-label="Telemetry trend per mesin">
          <thead><tr><th>MachineID</th><th>Volt Δ</th><th>Rot Δ</th><th>Press Δ</th><th>Vib Δ</th><th>Anomaly</th><th>At Risk</th></tr></thead>
          <tbody id="telemetry-body"></tbody>
        </table>
      </div>
    </div>

    <div id="maint-section" class="section">
      <div class="section-head">
        <h2>Maintenance Type &amp; Error Leading Indicator <button type="button" class="info-btn" aria-expanded="false" aria-label="Info metodologi maintenance type" onclick="toggleInfo(this)">&#9432;</button></h2>
        <span class="hint">Rasio scheduled vs unscheduled &mdash; error codes sebelum failure (prediktor dini)</span>
        <div class="info-panel">Rasio maintenance terjadwal vs tidak terjadwal dari data maintenance; error codes yang sering muncul sebelum failure adalah prediktor dini.</div>
      </div>
      <div class="table-wrap" style="margin-bottom:14px">
        <table id="maint-table" aria-label="Maintenance type ratio">
          <thead><tr><th>Type</th><th>Jumlah</th><th>% Total</th></tr></thead>
          <tbody id="maint-body"></tbody>
        </table>
      </div>
      <div class="table-wrap">
        <table id="predictor-table" aria-label="Error predictor summary">
          <thead><tr><th>Error ID</th><th>Total</th><th>Preceding Failure</th><th>Hit Rate</th></tr></thead>
          <tbody id="predictor-body"></tbody>
        </table>
      </div>
    </div>

    <div class="footer">MDSS &middot; Enterprise Command Center<span class="sep">&middot;</span>Maintenance Decision Support System</div>
  </div>

<script id="app-data" type="application/json">__PAYLOAD__</script>
<script>
(function(){
"use strict";
var DATA;
try{ DATA = JSON.parse(document.getElementById("app-data").textContent); }
catch(e){ console.error("Failed to parse payload", e); return; }

var costSlopeMap = {};
try{
  (DATA.lifecycle||[]).forEach(function(r){ costSlopeMap[r.machineID] = r.cost_slope; });
}catch(e){}
var fmt = function(n, d){
  if(n==null) return "\u2014";
  return Number(n).toFixed(d===undefined?1:d);
};

// ---- info (ⓘ) methodology toggle ----
function toggleInfo(btn){
  var panel = btn.parentNode.parentNode.querySelector(".info-panel");
  if(!panel) return;
  var open = panel.classList.toggle("open");
  btn.setAttribute("aria-expanded", open ? "true" : "false");
}
document.addEventListener("click", function(e){
  if(!e.target.classList || !e.target.classList.contains("info-btn")){
    document.querySelectorAll(".info-panel.open").forEach(function(p){ p.classList.remove("open");
      var b = p.parentNode.querySelector(".info-btn"); if(b) b.setAttribute("aria-expanded","false"); });
  }
});

function emptyRow(cols){
  var tr=document.createElement("tr");
  var td=document.createElement("td");
  td.colSpan=cols;
  td.style.textAlign="center";
  td.style.padding="32px 16px";
  td.style.color="var(--text-mid)";
  td.style.fontSize="13px";
  td.style.cursor="pointer";
  td.textContent="Tidak ada data untuk filter ini — klik reset filter";
  td.onclick=function(){var cb=document.querySelector("[data-filter-reset]");if(cb)cb.click();};
  tr.appendChild(td);
  return tr;
}

function lcBadgeClass(s){
  return "badge-"+s.toLowerCase();
}

function el(tag, attrs, children){
  var e = document.createElement(tag);
  if(attrs) Object.keys(attrs).forEach(function(k){
    if(k==="className") e.className=attrs[k];
    else if(k==="textContent") e.textContent=attrs[k];
    else e.setAttribute(k,attrs[k]);
  });
  if(children) children.forEach(function(c){ e.appendChild(c); });
  return e;
}

// ---- STATS ----
try{
  // Needs Action banner (P0-1)
  var needsActionBanner = document.getElementById('needs-action');
  if(needsActionBanner){
    var s = DATA.summary;
    var urgentCount = s.urgent_count || 0;
    var atRiskCount = s.at_risk_count || 0;
    
    // Show banner only if there are actions needed
    if(urgentCount > 0 || atRiskCount > 0){
      needsActionBanner.style.display = 'flex';
      needsActionBanner.innerHTML = '';
      
      // Card 1: Urgent machines
      if(urgentCount > 0){
        var card1 = document.createElement('a');
        card1.href = '#priority-section';
        card1.className = 'action-card urgent';
        card1.innerHTML = `
          <div class="icon">⚠️</div>
          <div>
            <div class="title">${urgentCount} mesin URGENT</div>
            <div class="sub">lihat prioritas</div>
          </div>
          <div class="arrow">→</div>
        `;
        needsActionBanner.appendChild(card1);
      }
      
      // Card 2: At-risk machines
      if(atRiskCount > 0){
        var card2 = document.createElement('a');
        card2.href = '#components-section';
        card2.className = 'action-card warn';
        card2.innerHTML = `
          <div class="icon">📊</div>
          <div>
            <div class="title">${atRiskCount} mesin at-risk telemetry</div>
            <div class="sub">lihat komponen</div>
          </div>
          <div class="arrow">→</div>
        `;
        needsActionBanner.appendChild(card2);
      }
    }
  }
  
  var s = DATA.summary;
  var cards = [
    {label:"Total Mesin", value:s.total_machines, cls:"", sub:"URGENT "+s.urgent_count+" \u00b7 HIGH "+s.high_count},
    {label:"Rata MTBF", value:fmt(s.avg_mtbf,0)+" hr", cls:"tone-teal", sub:"Rata-rata antar failure"},
    {label:"Rata MTTR", value:fmt(s.avg_mttr,1)+" hr", cls:"", sub:"Rata-rata durasi repair"},
    {label:"Ketersediaan", value:fmt(s.avg_availability)+"%", cls:"tone-info", sub:"Uptime keseluruhan"},
    {label:"Biaya Total", value:fmt(s.total_maintenance_cost,2)+" Jt", cls:"tone-amber", sub:"Ganti "+s.replace_count+"\u00b7 Tak Terjadwal "+fmt(s.unscheduled_pct,0)+"%"+(s.replace_count===0?"\u00b7 0 mesin direkomendasikan ganti saat ini":"")},
    {label:"Total Gagal", value:s.total_failures, cls:"", sub:"Perbaiki "+s.repair_count+"\u00b7 Err coverage "+fmt(s.error_coverage*100,0)+"%"}
  ];
  var wrap = document.getElementById("stats");
  cards.forEach(function(c){
    var card = el("div",{className:"stat-card "+c.cls},[
      el("div",{className:"label",textContent:c.label}),
      el("div",{className:"value",textContent:String(c.value)}),
      el("div",{className:"sub",textContent:c.sub})
    ]);
    wrap.appendChild(card);
  });
}catch(e){console.error("stats",e)}

// ---- PRIORITY TABLE ----
var currentPriorityFilter = "ALL";
var _priorityFilterEl = null;
try{
  _priorityFilterEl = document.getElementById("priority-pills");
  _priorityFilterEl.setAttribute("role","radiogroup");
  _priorityFilterEl.setAttribute("aria-label","Filter prioritas");
  function renderPriorityPills(){
    _priorityFilterEl.innerHTML="";
    var pillData = [
      {key:"ALL",label:"ALL",cls:"pill-all",count:DATA.priority.length},
      {key:"URGENT",label:"URGENT",cls:"pill-urgent",count:DATA.summary.urgent_count},
      {key:"HIGH",label:"HIGH",cls:"pill-high",count:DATA.summary.high_count},
      {key:"MEDIUM",label:"MEDIUM",cls:"pill-medium",count:DATA.priority.filter(function(r){return r.category==="MEDIUM"}).length},
      {key:"LOW",label:"LOW",cls:"pill-low",count:DATA.priority.filter(function(r){return r.category==="LOW"}).length}
    ];
    pillData.forEach(function(p){
      var isActive = currentPriorityFilter===p.key;
      var pill = el("button",{type:"button",className:"pill "+p.cls+(isActive?" active":""),
        textContent:p.label+" ("+p.count+")","aria-pressed":String(isActive)});
      pill.onclick=function(){currentPriorityFilter=p.key;renderPriorityPills();renderPriorityTable();};
      _priorityFilterEl.appendChild(pill);
    });
    if(currentPriorityFilter!=="ALL"){
      var resetBtn = el("button",{type:"button",className:"reset-btn",textContent:"\u2715 Reset"});
      resetBtn.onclick=function(){currentPriorityFilter="ALL";renderPriorityPills();renderPriorityTable();};
      _priorityFilterEl.appendChild(resetBtn);
    }
  }
  function badgeClass(cat){
    var map={URGENT:"badge-urgent",HIGH:"badge-high",MEDIUM:"badge-medium",LOW:"badge-low"};
    return map[cat]||"badge-low";
  }
  function renderPriorityTable(){
    var rows = currentPriorityFilter==="ALL"?DATA.priority:DATA.priority.filter(function(r){return r.category===currentPriorityFilter});
    var tbody = document.getElementById("priority-body");
    tbody.innerHTML="";
    if(rows.length===0){var er=emptyRow(11);er.querySelector("td").onclick=function(){currentPriorityFilter="ALL";renderPriorityPills();renderPriorityTable();};tbody.appendChild(er);return;}
    rows.forEach(function(r){
      var tr = document.createElement("tr");
      tr.appendChild(el("td",{textContent:String(r.machineID)}));
      tr.appendChild(el("td",{textContent:r.model}));
      tr.appendChild(el("td",{textContent:String(r.age)}));
      tr.appendChild(el("td",{textContent:fmt(r.mtbf_hours,1)}));
      tr.appendChild(el("td",{textContent:fmt(r.mttr_hours,1)}));
      tr.appendChild(el("td",{textContent:fmt(r.availability_pct)}));
      tr.appendChild(el("td",{textContent:fmt(r.failures_per_month,2)}));

      var _slope = costSlopeMap[r.machineID];
      var _trend = "\u2014";
      var _tcls = "";
      if(_slope!=null && isFinite(_slope)){
        if(_slope > 1){_trend = "\u25b2 naik"; _tcls="t-up";}
        else if(_slope < -1){_trend = "\u25bc turun"; _tcls="t-down";}
        else {_trend = "\u2192 flat"; _tcls="t-flat";}
      }
      var _ttd = document.createElement("td");
      var _tspan = el("span",{className:"trend "+_tcls, textContent:_trend});
      _ttd.appendChild(_tspan);
      tr.appendChild(_ttd);
      tr.appendChild(el("td",{textContent:fmt(r.total_cost_Jt,2)}));
      tr.appendChild(el("td",{textContent:fmt(r.score,1)}));
      var catBadge = el("span",{className:"badge "+badgeClass(r.category),textContent:r.category});
      var td5 = document.createElement("td");
      td5.appendChild(catBadge);
      tr.appendChild(td5);
      tbody.appendChild(tr);
    });
  }
  renderPriorityPills();
  renderPriorityTable();
}catch(e){console.error("priority",e)}

// ---- DECISION TABLE ----
var currentDecisionFilter = "ALL";
try{
  var chipWrap = document.getElementById("decision-chips");
  chipWrap.setAttribute("role","radiogroup");
  chipWrap.setAttribute("aria-label","Filter keputusan");
  function decCount(d){ return DATA.decision.filter(function(r){return r.decision===d;}).length; }
  var decChips = [["ALL"],["REPLACE"],["REVIEW"],["REPAIR"]];
  function renderDecisionChips(){
    chipWrap.innerHTML="";
    decChips.forEach(function(pair){
      var c = pair[0];
      var cnt = c==="ALL" ? DATA.decision.length : decCount(c);
      var isActive = currentDecisionFilter===c;
      var chip = el("button",{type:"button",className:"chip"+(isActive?" active":""),textContent:c+" ("+cnt+")",
        "aria-pressed":String(isActive)});
      chip.onclick=function(){currentDecisionFilter=c;renderDecisionChips();renderDecisionTable();};
      chipWrap.appendChild(chip);
    });
  }
  function decBadgeClass(d){
    return {REPLACE:"badge-replace",REVIEW:"badge-review",REPAIR:"badge-repair"}[d]||"";
  }
  function renderDecisionTable(){
    var rows = currentDecisionFilter==="ALL"?DATA.decision:DATA.decision.filter(function(r){return r.decision===currentDecisionFilter});
    var tbody = document.getElementById("decision-body");
    tbody.innerHTML="";
    if(rows.length===0){tbody.appendChild(emptyRow(6));return;}
    rows.forEach(function(r){
      var tr = document.createElement("tr");
      tr.appendChild(el("td",{textContent:String(r.machineID)}));
      tr.appendChild(el("td",{textContent:r.model}));
      tr.appendChild(el("td",{textContent:String(r.age)}));
      tr.appendChild(el("td",{textContent:fmt(r.cost_ratio,2)}));
      tr.appendChild(el("td",{textContent:String(r.total_score)}));
      var badge = el("span",{className:"badge "+decBadgeClass(r.decision),textContent:r.decision});
      var td=document.createElement("td");td.appendChild(badge);tr.appendChild(td);
      tbody.appendChild(tr);
    });
  }
  renderDecisionChips();
  renderDecisionTable();
}catch(e){console.error("decision",e)}

// ---- LIFECYCLE TABLE ----
var currentLifecycleFilter = "ALL";
try{
  var lcChipWrap = document.getElementById("lifecycle-chips");
  lcChipWrap.setAttribute("role","radiogroup");
  lcChipWrap.setAttribute("aria-label","Filter lifecycle stage");
  function lcCount(s){ return DATA.lifecycle.filter(function(r){return r.stage===s;}).length; }
  var lcStages = ["ALL","NEW","STABLE","AGING","DEGRADING","CRITICAL","END_OF_LIFE"];
  var stageLabel = {NEW:"Baru",STABLE:"Stabil",AGING:"Menua",DEGRADING:"Degradasi",CRITICAL:"Kritis",END_OF_LIFE:"Akhir Umur"};
  function renderLifecycleChips(){
    lcChipWrap.innerHTML="";
    lcStages.forEach(function(c){
      var cnt = c==="ALL" ? DATA.lifecycle.length : lcCount(c);
      var isActive = currentLifecycleFilter===c;
      var chip = el("button",{type:"button",className:"chip"+(isActive?" active":""),textContent:(c==="ALL"?c:stageLabel[c]||c)+" ("+cnt+")",
        "aria-pressed":String(isActive)});
      chip.onclick=function(){currentLifecycleFilter=c;renderLifecycleChips();renderLifecycleTable();};
      lcChipWrap.appendChild(chip);
    });
  }
  function renderLifecycleTable(){
    var rows = currentLifecycleFilter==="ALL"?DATA.lifecycle:DATA.lifecycle.filter(function(r){return r.stage===currentLifecycleFilter});
    var tbody = document.getElementById("lifecycle-body");
    tbody.innerHTML="";
    if(rows.length===0){tbody.appendChild(emptyRow(4));return;}
    rows.forEach(function(r){
      var tr = document.createElement("tr");
      tr.appendChild(el("td",{textContent:String(r.machineID)}));
      tr.appendChild(el("td",{textContent:r.model}));
      tr.appendChild(el("td",{textContent:String(r.age)}));
      var badge = el("span",{className:"badge "+lcBadgeClass(r.stage),textContent:stageLabel[r.stage]||r.stage});
      var td=document.createElement("td");td.appendChild(badge);tr.appendChild(td);
      tbody.appendChild(tr);
    });
  }
  renderLifecycleChips();
  renderLifecycleTable();
}catch(e){console.error("lifecycle",e)}

// ---- REPLACEMENT TIMELINE ----
try{
  var tbody = document.getElementById("timeline-body");
  DATA.replacement_timeline.forEach(function(r){
    var tr = document.createElement("tr");
    tr.appendChild(el("td",{textContent:String(r.machineID)}));
    tr.appendChild(el("td",{textContent:r.model}));
    var sBadge = el("span",{className:"badge "+lcBadgeClass(r.stage),textContent:(typeof stageLabel!=="undefined"?stageLabel[r.stage]:r.stage)||r.stage});
    var td1=document.createElement("td");td1.appendChild(sBadge);tr.appendChild(td1);
    tr.appendChild(el("td",{textContent:String(r.remaining_life_months)}));
    tr.appendChild(el("td",{textContent:r.urgency}));
    tbody.appendChild(tr);
  });
}catch(e){console.error("timeline",e)}

// ---- CHARTS ----
try{
if(typeof Chart==="undefined"){
     document.querySelectorAll(".chart-card canvas").forEach(function(c){
       var p=document.createElement("p");p.className="chart-fallback";
       p.textContent="Chart.js tidak dimuat. Periksa koneksi internet.";
       c.parentNode.appendChild(p);c.style.display="none";
     });
   }else{
    var chartOpts={responsive:true,plugins:{legend:{labels:{color:"#576178",usePointStyle:true,boxWidth:8}}}};
    var scaleOpts={ticks:{color:"#576178"},grid:{color:"#e6eaf4"}};

    // Bar: top 10 priority
    var top10 = DATA.priority.slice(0,10);
    var thresholdPlugin = {
      id:"threshold75",
      afterDraw:function(chart){
        var y = chart.scales.y;
        if(!y) return;
        var y75 = y.getPixelForValue(75);
        var ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle="#dc2626"; ctx.setLineDash([6,4]); ctx.lineWidth=1.5;
        ctx.beginPath(); ctx.moveTo(chart.chartArea.left, y75); ctx.lineTo(chart.chartArea.right, y75); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle="#dc2626"; ctx.font="11px system-ui"; ctx.textAlign="left";
        ctx.fillText("Ambang URGENT (75)", chart.chartArea.left+4, y75-5);
        ctx.restore();
      }
    };
    new Chart(document.getElementById("chart-priority"),{
      type:"bar",
      data:{
        labels:top10.map(function(r){return "M"+r.machineID}),
        datasets:[{
          label:"Priority Score",
          data:top10.map(function(r){return r.score}),
          backgroundColor:top10.map(function(r){
            return {URGENT:"#dc2626",HIGH:"#ea580c",MEDIUM:"#ca8a04",LOW:"#059669"}[r.category]||"#4f46e5";
          }),
          borderRadius:4
        }]
      },
      options:{
        ...chartOpts,scales:{x:scaleOpts,y:{...scaleOpts,beginAtZero:true}},
        plugins:{legend:{...chartOpts.plugins.legend}},
        onClick:function(evt, items){
          if(!items || !items.length) return;
          var i = items[0].index;
          var mid = top10[i].machineID;
          var target = document.getElementById("priority-section");
          if(target) target.scrollIntoView({behavior:"smooth",block:"start"});
          var rows = document.querySelectorAll("#priority-body tr");
          rows.forEach(function(tr){
            var first = tr.querySelector("td");
            if(first && first.textContent===String(mid)){ tr.style.outline="2px solid var(--brand)"; tr.style.outlineOffset="-2px";
              setTimeout(function(){ tr.style.outline=""; },2200);
            }
          });
        }
      },
      plugins:[thresholdPlugin]
    });

    // Pie: lifecycle
    var lcCounts={};
    DATA.lifecycle.forEach(function(r){lcCounts[r.stage]=(lcCounts[r.stage]||0)+1;});
    var lcLabels=Object.keys(lcCounts);
    var lcColors={NEW:"#0284c7",STABLE:"#059669",AGING:"#ca8a04",DEGRADING:"#ea580c",CRITICAL:"#dc2626",END_OF_LIFE:"#b91c1c",OLD:"#94a3b8"};
    new Chart(document.getElementById("chart-lifecycle"),{
      type:"pie",
      data:{
        labels:lcLabels.map(function(l){return (typeof stageLabel!=="undefined"&&stageLabel[l])?stageLabel[l]:l; }),
        datasets:[{data:lcLabels.map(function(l){return lcCounts[l]}),
          backgroundColor:lcLabels.map(function(l){return lcColors[l]||"#4f46e5"})}]
      },
      options:{...chartOpts},
      plugins:[{
        id:"lifecycleClick",
        onClick:function(evt, items){
          if(!items || !items.length) return;
          var lab = lcLabels[items[0].index];
          var target = document.getElementById("lifecycle-section");
          if(target) target.scrollIntoView({behavior:"smooth",block:"start"});
          if(lab){ currentLifecycleFilter=lab; renderLifecycleChips(); renderLifecycleTable(); }
        }
      }]
    });

    // Doughnut: decision
    var dcCounts={};
    DATA.decision.forEach(function(r){dcCounts[r.decision]=(dcCounts[r.decision]||0)+1;});
    var dcLabels=Object.keys(dcCounts);
    var dcColors={REPLACE:"#dc2626",REVIEW:"#ca8a04",REPAIR:"#059669"};
    new Chart(document.getElementById("chart-decision"),{
      type:"doughnut",
      data:{
        labels:dcLabels,
        datasets:[{data:dcLabels.map(function(l){return dcCounts[l]}),
          backgroundColor:dcLabels.map(function(l){return dcColors[l]||"#4f46e5"})}]
      },
      options:{...chartOpts},
      plugins:[{
        id:"decisionClick",
        onClick:function(evt, items){
          if(!items || !items.length) return;
          var lab = dcLabels[items[0].index];
          var target = document.getElementById("decision-section");
          if(target) target.scrollIntoView({behavior:"smooth",block:"start"});
          if(lab && typeof currentDecisionFilter!=="undefined"){ currentDecisionFilter=lab; renderDecisionChips(); renderDecisionTable(); }
        }
      }]
    });
  }
}catch(e){console.error("charts",e)}

// ---- COMPONENT TABLE + CHART ----
try{
  var compBody = document.getElementById("component-body");
  (DATA.components||[]).forEach(function(r){
    var tr = document.createElement("tr");
    tr.appendChild(el("td",{textContent:r.component}));
    tr.appendChild(el("td",{textContent:String(r.n_failures)}));
    tr.appendChild(el("td",{textContent:fmt(r.pct,1)+"%"}));
    tr.appendChild(el("td",{textContent:fmt(r.total_cost_Jt)}));
    tr.appendChild(el("td",{textContent:fmt(r.risk_score,2)}));
    var lv = el("span",{className:"badge badge-"+(r.risk_level||"low").toLowerCase(),textContent:r.risk_level});
    var td = document.createElement("td");td.appendChild(lv);tr.appendChild(td);
    compBody.appendChild(tr);
  });
  var compChart = document.getElementById("chart-component");
  if(typeof Chart!=="undefined" && (DATA.components||[]).length){
    new Chart(compChart,{
      type:"bar",
      data:{labels:(DATA.components||[]).map(function(r){return r.component}),datasets:[{label:"Risk Score",data:(DATA.components||[]).map(function(r){return r.risk_score}),backgroundColor:"#4f46e5",borderRadius:4}]},
      options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{ticks:{color:"#576178"},grid:{color:"#e6eaf4"}},y:{beginAtZero:true,ticks:{color:"#576178"},grid:{color:"#e6eaf4"}}}}
    });
  }
}catch(e){console.error("component",e)}

// ---- TELEMETRY TABLE ----
try{
  var telBody = document.getElementById("telemetry-body");
  (DATA.telemetry_trend||[]).forEach(function(r){
    var tr = document.createElement("tr");
    tr.appendChild(el("td",{textContent:String(r.machineID)}));
    tr.appendChild(el("td",{textContent:fmt(r.volt_slope,3)}));
    tr.appendChild(el("td",{textContent:fmt(r.rot_slope,3)}));
    tr.appendChild(el("td",{textContent:fmt(r.press_slope,3)}));
    tr.appendChild(el("td",{textContent:fmt(r.vib_slope,3)}));
    tr.appendChild(el("td",{textContent:fmt(r.anomaly_score,3)}));
    var risk = r.at_risk ? el("span",{className:"badge badge-urgent",textContent:"RISK"}) : el("span",{className:"badge badge-low",textContent:"OK"});
    var td2=document.createElement("td");td2.appendChild(risk);tr.appendChild(td2);
    telBody.appendChild(tr);
  });
}catch(e){console.error("telemetry",e)}

// ---- ANOMALY CHART ----
try{
  var anChart = document.getElementById("chart-anomaly");
  if(typeof Chart!=="undefined" && (DATA.anomaly||[]).length){
    var an = (DATA.anomaly||[]).slice(0,20);
    new Chart(anChart,{
      type:"bar",
      data:{labels:an.map(function(r){return "M"+r.machineID}),datasets:[{label:"Anomaly Score",data:an.map(function(r){return r.anomaly_score}),backgroundColor:an.map(function(r){return r.at_risk?"#dc2626":"#059669"}),borderRadius:4}]},
      options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{ticks:{color:"#576178"},grid:{color:"#e6eaf4"}},y:{beginAtZero:true,ticks:{color:"#576178"},grid:{color:"#e6eaf4"}}}}
    });
  }
}catch(e){console.error("anomaly-chart",e)}

// ---- MAINT TYPE TABLE ----
try{
  var mtBody = document.getElementById("maint-body");
  (DATA.maintenance_ratio||[]).forEach(function(r){
    var tr = document.createElement("tr");
    tr.appendChild(el("td",{textContent:r.type}));
    tr.appendChild(el("td",{textContent:String(r.n)}));
    tr.appendChild(el("td",{textContent:fmt(r.pct,1)+"%"}));
    mtBody.appendChild(tr);
  });
}catch(e){console.error("maint-type",e)}

// ---- ERROR PREDICTOR ----
try{
  var prBody = document.getElementById("predictor-body");
  (DATA.error_predictor||{top_errors:[]}).top_errors.forEach(function(r){
    var tr = document.createElement("tr");
    tr.appendChild(el("td",{textContent:r.errorID}));
    tr.appendChild(el("td",{textContent:String(r.total_occurrences)}));
    tr.appendChild(el("td",{textContent:String(r.n_preceding_failure)}));
    tr.appendChild(el("td",{textContent:fmt(r.hit_rate,2)}));
    prBody.appendChild(tr);
  });
}catch(e){console.error("predictor",e)}

// ---- ERROR CHART ----
try{
  var errChart = document.getElementById("chart-error");
  if(typeof Chart!=="undefined" && (DATA.error_predictor||{}).top_errors && DATA.error_predictor.top_errors.length){
    var top = DATA.error_predictor.top_errors.slice(0,8);
    new Chart(errChart,{
      type:"bar",
      data:{labels:top.map(function(r){return r.errorID}),datasets:[{label:"Hit Rate",data:top.map(function(r){return r.hit_rate}),backgroundColor:"#ca8a04",borderRadius:4}]},
      options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{ticks:{color:"#576178"},grid:{color:"#e6eaf4"}},y:{beginAtZero:true,ticks:{color:"#576178"},grid:{color:"#e6eaf4"}}}}
    });
  }
}catch(e){console.error("error-chart",e)}

})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    _payload: dict = {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/data":
            self._send_json(self._payload)
        elif path == "/":
            raw = json.dumps(self._payload, default=str)
            safe = raw.replace("</", "<\\/")
            html = HTML_PAGE.replace("__PAYLOAD__", safe, 1)
            self._send_html(html)
        else:
            self.send_error(404)

    def _send_json(self, data: dict):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("[dashboard] %s\n" % (fmt % args))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MDSS Phase 4 Dashboard")
    parser.add_argument("--port", type=int, default=8323)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir} ...")
    payload = build_payload(args.data_dir)
    DashboardHandler._payload = payload
    print(f"Data loaded: {payload['summary']['total_machines']} machines")

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
