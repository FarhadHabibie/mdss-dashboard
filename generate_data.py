#!/usr/bin/env python3
"""
Generate synthetic maintenance data based on Microsoft Azure PdM schema
+ augmented tables for cost, engineering estimation, and replacement analysis.
"""
import csv, random, os
from datetime import datetime, timedelta

random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# === CONFIG ===
N_MACHINES = 100
START = datetime(2015, 1, 1)
END = datetime(2015, 12, 31, 23, 0, 0)
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
MODELS = ["model1", "model2", "model3", "model4"]
ERROR_TYPES = ["error1", "error2", "error3", "error4", "error5"]

# Component costs (Rp juta)
COMP_COST = {"comp1": (5, 15), "comp2": (12, 45), "comp3": (3, 10), "comp4": (8, 25)}
LABOR_COST_PER_HOUR = 0.8  # Rp juta

# Machine info
machines = []
for i in range(1, N_MACHINES + 1):
    age = random.randint(1, 20)
    # Higher age = higher failure rate
    machines.append({
        "machineID": i,
        "model": random.choice(MODELS),
        "age": age,
    })

# === 1. MACHINES ===
with open(f"{OUT}/PdM_machines.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["machineID", "model", "age"])
    w.writeheader()
    w.writerows(machines)
print(f"machines: {len(machines)} rows")

# === 2. TELEMETRY (hourly, 100 machines, ~8760 hours) ===
# Generate hourly readings with realistic patterns
telemetry_rows = []
hours = int((END - START).total_seconds() / 3600)

# Pre-generate failure times per machine for noise injection
failure_times = {}  # machineID -> sorted list of failure datetimes
for m in machines:
    mid = m["machineID"]
    n_failures = max(2, int(m["age"] * random.uniform(0.5, 2.5)))
    failure_times[mid] = sorted([
        START + timedelta(seconds=random.randint(0, int((END - START).total_seconds())))
        for _ in range(n_failures)
    ])

for h in range(hours):
    dt = START + timedelta(hours=h)
    ts = dt.strftime("%Y-%m-%d %H:%M:%S")
    for mid in range(1, N_MACHINES + 1):
        age = machines[mid - 1]["age"]
        # Base values with seasonal variation
        month = dt.month
        seasonal = 1 + 0.05 * (1 if month in [6, 7, 8] else 0)  # rainy season

        volt = random.gauss(170 + age * 0.5, 15) * seasonal
        rot = random.gauss(400 + age * 2, 30)
        press = random.gauss(100 + age * 0.3, 5)
        vib = random.gauss(40 + age * 1.5, 8)

        # Spike near failure times
        for ft in failure_times[mid]:
            if abs((dt - ft).total_seconds()) < 3600 * 24:  # within 24h of failure
                vib *= random.uniform(1.5, 3.0)
                volt *= random.uniform(0.7, 1.4)
                break

        telemetry_rows.append({
            "datetime": ts,
            "machineID": mid,
            "volt": round(max(120, volt), 2),
            "rot": round(max(100, rot), 2),
            "press": round(max(50, press), 2),
            "vib": round(max(5, vib), 2),
        })

with open(f"{OUT}/PdM_telemetry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["datetime", "machineID", "volt", "rot", "press", "vib"])
    w.writeheader()
    w.writerows(telemetry_rows)
print(f"telemetry: {len(telemetry_rows)} rows")

# === 3. ERRORS ===
error_rows = []
for m in machines:
    mid = m["machineID"]
    n_errors = max(1, int(m["age"] * random.uniform(0.3, 1.5)))
    for _ in range(n_errors):
        dt = START + timedelta(seconds=random.randint(0, int((END - START).total_seconds())))
        error_rows.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "machineID": mid,
            "errorID": random.choice(ERROR_TYPES),
        })

error_rows.sort(key=lambda x: x["datetime"])
with open(f"{OUT}/PdM_errors.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["datetime", "machineID", "errorID"])
    w.writeheader()
    w.writerows(error_rows)
print(f"errors: {len(error_rows)} rows")

# === 4. FAILURES ===
failure_rows = []
for mid, fts in failure_times.items():
    for dt in fts:
        failure_rows.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "machineID": mid,
            "failure": random.choice(COMPONENTS),
        })

failure_rows.sort(key=lambda x: x["datetime"])
with open(f"{OUT}/PdM_failures.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["datetime", "machineID", "failure"])
    w.writeheader()
    w.writerows(failure_rows)
print(f"failures: {len(failure_rows)} rows")

# === 5. MAINTENANCE (scheduled + unscheduled) ===
maint_rows = []
# Scheduled maintenance (proactive) — quarterly per machine
for m in machines:
    mid = m["machineID"]
    for quarter_start in [1, 91, 182, 273]:
        dt = START + timedelta(days=quarter_start, hours=random.randint(6, 10))
        maint_rows.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "machineID": mid,
            "comp": random.choice(COMPONENTS),
            "type": "scheduled",
        })

# Unscheduled maintenance (reactive) — from failures
for fr in failure_rows:
    maint_rows.append({
        "datetime": fr["datetime"],
        "machineID": fr["machineID"],
        "comp": fr["failure"],
        "type": "unscheduled",
    })

maint_rows.sort(key=lambda x: x["datetime"])
with open(f"{OUT}/PdM_maint.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["datetime", "machineID", "comp", "type"])
    w.writeheader()
    w.writerows(maint_rows)
print(f"maintenance: {len(maint_rows)} rows")

# === 6. COST DATA (augmented) ===
cost_rows = []
for fr in failure_rows:
    comp = fr["failure"]
    mid = fr["machineID"]
    age = machines[mid - 1]["age"]
    # Repair cost increases with age
    base_low, base_high = COMP_COST[comp]
    parts_cost = random.uniform(base_low, base_high) * (1 + age * 0.08)
    repair_hours = random.uniform(2, 16) * (1 + age * 0.05)
    labor_cost = repair_hours * LABOR_COST_PER_HOUR
    downtime_hours = repair_hours + random.uniform(0.5, 4)  # waiting + prep
    replacement_cost = random.uniform(30, 120) * (1 + age * 0.03)

    cost_rows.append({
        "datetime": fr["datetime"],
        "machineID": mid,
        "component": comp,
        "parts_cost_Jt": round(parts_cost, 2),
        "labor_hours": round(repair_hours, 1),
        "labor_cost_Jt": round(labor_cost, 2),
        "total_repair_cost_Jt": round(parts_cost + labor_cost, 2),
        "downtime_hours": round(downtime_hours, 1),
        "replacement_cost_Jt": round(replacement_cost, 2),
        "is_replacement": 0,
    })

with open(f"{OUT}/PdM_cost.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(cost_rows[0].keys()))
    w.writeheader()
    w.writerows(cost_rows)
print(f"cost: {len(cost_rows)} rows")

# === 7. ENGINEERING ESTIMATES ===
job_types = [
    {"type": "Overhaul Kompressor", "complexity": "high", "base_days": 5, "base_crew": 4, "skill": "Mekanik senior + Electrician"},
    {"type": "Ganti Bearing Motor", "complexity": "medium", "base_days": 1.5, "base_crew": 2, "skill": "Mekanik"},
    {"type": "Repair VFD Drive", "complexity": "medium", "base_days": 2, "base_crew": 2, "skill": "Electrician"},
    {"type": "Replace Pump Seal", "complexity": "low", "base_days": 0.5, "base_crew": 1, "skill": "Mekanik junior"},
    {"type": "Inspeksi Turbin", "complexity": "high", "base_days": 7, "base_crew": 5, "skill": "Insinyur + Mekanik + Safety officer"},
    {"type": "Calibrate Sensor", "complexity": "low", "base_days": 0.5, "base_crew": 1, "skill": "Instrumentation"},
    {"type": "Replace Conveyor Belt", "complexity": "medium", "base_days": 3, "base_crew": 3, "skill": "Mekanik + Welder"},
    {"type": "Repair Hydraulic Press", "complexity": "high", "base_days": 4, "base_crew": 3, "skill": "Mekanik senior + Hydraulik specialist"},
]

tools_list = [
    ["Hoist 2T", "Torque wrench", "Dial indicator"],
    ["Hydraulic puller", "Feeler gauge", "Dial indicator"],
    ["Multimeter", "Oscilloscope", "Soldering station"],
    ["Seal puller", "Gasket scraper", "Torque wrench"],
    ["Borescope", "Vibration analyzer", "Lifting rig"],
    ["Calibration kit", "Multimeter", "Software calibrator"],
    ["Belt tensioner", "Welder", "Measuring tape"],
    ["Hydraulic press", "Pressure gauge", "Alignment tools"],
]

eng_rows = []
for i, jt in enumerate(job_types):
    n_records = random.randint(5, 15)
    for _ in range(n_records):
        # Simulate historical estimates vs actuals
        est_days = jt["base_days"] * random.uniform(0.8, 1.3)
        act_days = est_days * random.uniform(0.85, 1.5)
        buffer_pct = max(0, (act_days - est_days) / est_days * 100)

        eng_rows.append({
            "job_type": jt["type"],
            "complexity": jt["complexity"],
            "estimated_days": round(est_days, 1),
            "actual_days": round(act_days, 1),
            "crew_size": jt["base_crew"] + random.randint(-1, 1),
            "skill_required": jt["skill"],
            "tools": " | ".join(tools_list[i]),
            "spare_parts_needed": random.choice(["Ya", "Ya", "Tidak"]),
            "potential_delay_hours": round(random.uniform(1, 20), 1),
            "parallel_allowed": random.choice(["Ya", "Ya", "Tidak"]),
            "buffer_pct": round(buffer_pct, 1),
        })

with open(f"{OUT}/PdM_engineering.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(eng_rows[0].keys()))
    w.writeheader()
    w.writerows(eng_rows)
print(f"engineering: {len(eng_rows)} rows")

print(f"\nAll data generated in {OUT}/")
