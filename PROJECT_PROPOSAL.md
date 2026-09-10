# PROYEK: SISTEM OPTIMASI MAINTENANCE & PERENCANAAN RESOURCES
## Maintenance Decision Support System (MDSS)

---

## 1. LATAR BELAKANG

Biaya maintenance industri manufaktur mencapai 15-40% dari biaya operasional.
Tanpa sistem keputusan berbasis data, organisasi sering terjebak dalam:
- Over-maintenance (biaya preventive terlalu tinggi)
- Under-maintenance (downtime tak terduga sangat mahal)
- Keputusan repair vs replacement berdasarkan intuisi
- Estimasi duration/resource yang meleset

**Tujuan proyek:** Membangun sistem pendukung keputusan yang mengintegrasikan
data historis maintenance untuk mengoptimalkan strategi maintenance, keputusan
repair/replacement, proyeksi equipment lifecycle, dan perencanaan resource engineering.

---

## 2. SCOPE & MODUL

### Modul 1: Analisis Preventive & Corrective Maintenance

**Input:** Data historis failure, maintenance, telemetry, error logs

**Output:**
| Metrik | Deskripsi | Rumus |
|--------|-----------|-------|
| MTBF | Rata-rata waktu operasi antar failure | Total operasi jam / Jumlah failure |
| MTTR | Rata-rata waktu perbaikan | Total downtime jam / Jumlah repair |
| Availability | Persentase waktu operasional | MTBF / (MTBF + MTTR) × 100% |
| Failure Frequency | Frekuensi breakdown per unit waktu | Jumlah failure / Bulan aktif |
| Repair Cost/Jt | Biaya rata-rata per incident | Total biaya repair / Jumlah incident |
| Total Downtime | Total jam tidak operasional | Sum downtime_hours |

**Scoring Prioritas (1-100):**
```
Priority Score = (w1 × normalized_MTBF_inv) +    # MTBF rendah → skor tinggi
                 (w2 × normalized_MTTR) +         # MTTR tinggi → skor tinggi
                 (w3 × normalized_freq) +         # Frekuensi tinggi → skor tinggi
                 (w4 × normalized_cost) +         # Biaya tinggi → skor tinggi
                 (w5 × criticality_flag)          # Mesin kritikal → bonus

Default weights: w1=0.25, w2=0.20, w3=0.20, w4=0.20, w5=0.15
```

**Threshold:**
- Score ≥ 75: URGENT (aintenance minggu ini)
- Score 50-74: HIGH (jadwalkan dalam 2 minggu)
- Score 25-49: MEDIUM (jadwalkan dalam bulan ini)
- Score < 25: LOW (monitoring)

---

### Modul 2: Repair vs Replacement Decision Engine

**Input:** Data cost, equipment age, failure history, replacement cost

**Keputusan berdasarkan scoring:**

| Kondisi | Repair (−) | Replacement (+) |
|---------|------------|-----------------|
| Biaya repair < 50% replacement | −30 | 0 |
| Biaya repair 50-80% replacement | −10 | +10 |
| Biaya repair > 80% replacement | 0 | +30 |
| Biaya repair > replacement | +30 (REPLACE) | +40 |
| Frekuensi breakdown > 3x/bulan | −10 | +25 |
| Umur > expected life | −10 | +30 |
| Equipment kritikal | −15 | +20 |
| Kondisi baik (vibration normal) | −20 | 0 |

**Threshold:**
- Total > 40: Ganti baru (REPLACE)
- Total 0-40: Repair dulu, review lagi 3 bulan
- Total < 0: Repair, optimize preventive schedule

---

### Modul 3: Proyeksi Equipment Lifecycle

**Input:** Time-series failure data, cost trend, equipment age

**Analisis:**
1. **Trend MTBF** — Plot MTBF vs umur equipment, identifikasi inflection point
2. **Trend Biaya** — Moving average repair cost per equipment, deteksi eskalasi
3. **Bathtub Curve** — Model kegagalan: early failure → useful life → wear-out
4. **Remaining Useful Life (RUL)** — Estimasi kapan equipment mencapai end-of-life
5. **Cost Projection** — Proyeksi total biaya maintenance untuk 1-3 tahun ke depan

**Proyeksi:**
```
Break-even Point = Umur di mana projected_maintenance_cost > replacement_investment
                   (termasuk present value dan downtime opportunity cost)
```

**Output per equipment:**
- Kategori lifecycle: [Baru] [Stabil] [Degradasi] [Critical] [End-of-Life]
- Rekomendasi: Lanjut operasi / Sched. replacement / Immediate replacement
- Confidence level: Berdasarkan jumlah data historis

---

### Modul 4: Estimasi Durasi & Resource Engineering

**Input:** Job types, historical duration, complexity, crew data

**Estimasi Duration:**
```
Estimated_Duration = Base_Days × Complexity_Factor × (1 + Historical_Variance%)
                   + Buffer_Hours × Risk_Factor

Complexity: Low=0.8, Medium=1.0, High=1.5
Risk_Factor: parallel=0.7, sequential=1.0
Buffer: 15-30% dari estimasi
```

**Resource Planning:**
| Resource | Algoritma |
|----------|-----------|
| Manpower | Base crew × complexity + skill matching |
| Equipment | Lookup table per job type |
| Material | BOM per job type + availability check |
| Timeline | Gantt chart: parallel vs sequential tasks |

**Output:**
- Work Breakdown Structure (WBS)
- Gantt chart dengan parallel/sequential tasks
- Resource histogram (manpower vs time)
- Total cost estimasi

---

## 3. DATA YANG DIGUNAKAN

Dataset di folder `/home/farhad/maintenance-project/data/`:

| File | Deskripsi | Baris |
|------|-----------|-------|
| `PdM_machines.csv` | 100 mesin: model, umur (tahun) | 100 |
| `PdM_telemetry.csv` | Sensor hourly: voltage, rotation, pressure, vibration | 875,900 |
| `PdM_errors.csv` | Log error non-fatal | 868 |
| `PdM_failures.csv` | Record replacement karena failure | 1,662 |
| `PdM_maint.csv` | Maintenance scheduled + unscheduled | 2,062 |
| `PdM_cost.csv` | Biaya repair, downtime, replacement cost | 1,662 |
| `PdM_engineering.csv` | Historical job estimates + actuals | 79 |

**Source schema:** Microsoft Azure Predictive Maintenance dataset
(augmented dengan cost dan engineering estimation data)

---

## 4. OUTPUT SISTEM

### Dashboard Web (Rekomendasi: Flask/Streamlit)

**Halaman 1 — Overview Dashboard**
- Total mesin, MTBF/MTTR rata-rata, availability rate
- Pie chart: distribusi failure per komponen
- Heatmap: mesin vs bulan (failure density)
- Top 10 mesin prioritas tertinggi

**Halaman 2 — Priority Matrix**
- Tabel sortable: semua mesin + priority score
- Filter: model, age range, component type
- Detail per mesin: trend MTBF, MTTR, cost per quarter
- Alert badge untuk mesin critical

**Halaman 3 — Repair vs Replacement**
- Scatter plot: repair cost vs replacement cost per mesin
- Decision table: rekomendasi repair/replace per mesin
- ROI calculator: "Jika ganti sekarang, break-even dalam X tahun"
- Filter by umur, frekuensi breakdown, total cost

**Halaman 4 — Lifecycle Projection**
- Line chart: projected MTBF & cost trend per equipment
- Bathtub curve per model
- Timeline equipment end-of-life
- Summary: equipment masuk kategori "critical" dalam 6 bulan

**Halaman 5 — Engineering Estimation**
- Input form: job type, complexity, scope
- Output: estimasi durasi, crew size, tools, material, total cost
- Gantt chart visual
- Historical accuracy: estimasi vs actual untuk job serupa

---

## 5. ARCHITECTURE

```
[CSV Data Sources]
       |
       v
[Data Layer — Pandas/SQL]
  - Load & clean
  - Compute MTBF, MTTR, MTTF
  - Aggregate per machine per period
       |
       v
[Logic Layer]
  - Module 1: Priority Scoring Engine
  - Module 2: Repair vs Replacement Decision
  - Module 3: Lifecycle Projection (regression + extrapolation)
  - Module 4: Duration & Resource Estimator
       |
       v
[Presentation Layer — Dashboard]
  - Streamlit / Flask + Chart.js
  - Interactive filters
  - Export to Excel/PDF
```

**Tech Stack:**
- Python 3.x
- Pandas, NumPy (data processing)
- SciPy/Scikit-learn (regression, projection)
- Streamlit (dashboard, cepat) atau Flask + Chart.js (full control)
- Openpyxl (export Excel)

---

## 6. FILE STRUCTURE

```
maintenance-project/
├── data/
│   ├── PdM_machines.csv          # 100 mesin
│   ├── PdM_telemetry.csv         # 875K sensor records
│   ├── PdM_errors.csv            # 868 error logs
│   ├── PdM_failures.csv          # 1,662 failure records
│   ├── PdM_maint.csv             # 2,062 maintenance records
│   ├── PdM_cost.csv              # 1,662 cost records
│   └── PdM_engineering.csv       # 79 engineering job records
├── generate_data.py              # Script generate dummy data
├── PROJECT_PROPOSAL.md           # Dokumen ini
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Load & clean data
│   ├── metrics.py                # Hitung MTBF, MTTR, availability
│   ├── priority.py               # Scoring engine
│   ├── decision.py               # Repair vs replacement
│   ├── projection.py             # Lifecycle projection
│   └── estimation.py             # Duration & resource estimator
├── dashboard/
│   ├── app.py                    # Streamlit/Flask main
│   ├── pages/
│   │   ├── overview.py
│   │   ├── priority.py
│   │   ├── decision.py
│   │   ├── projection.py
│   │   └── engineering.py
│   └── templates/                # (Flask only)
└── tests/
    ├── test_metrics.py
    ├── test_priority.py
    └── test_decision.py
```

---

## 7. IMPLEMENTASI — PHASED PLAN

### Phase 1: Data Processing & Metrics (1-2 minggu)
- [x] Generate/collect data
- [ ] Implementasi data_loader.py (clean, validate, join)
- [ ] Implementasi metrics.py (MTBF, MTTR, MTTF, availability)
- [ ] Unit tests

### Phase 2: Decision Logic (1 minggu)
- [ ] Priority scoring engine (Modul 1)
- [ ] Repair vs replacement decision (Modul 2)
- [ ] Unit tests

### Phase 3: Projection & Estimation (1-2 minggu)
- [ ] Lifecycle projection (Modul 3) — regression, bathtub curve
- [ ] Duration estimation (Modul 4) — historical average + variance
- [ ] Unit tests

### Phase 4: Dashboard (1-2 minggu)
- [ ] Setup Streamlit/Flask
- [ ] 5 halaman dashboard
- [ ] Interactive filters & charts
- [ ] Export functionality

### Phase 5: Validation & Tuning (3-5 hari)
- [ ] Validasi terhadap data historis
- [ ] Tuning weights scoring
- [ ] User acceptance testing

---

## 8. CATATAN PENTING

- **Dataset bersifat dummy/sintetis** berdasarkan schema Microsoft Azure PdM.
  Untuk implementasi production, ganti dengan data CMMS/EAM yang sesungguhnya
  (SAP PM, Maximo, Infor EAM, dll).

- **Bobot scoring (weights)** perlu di-tune berdasarkan domain spesifik.
  Bobot default di atas adalah starting point.

- **Modul 4 (Engineering Estimation)** paling dependent data historis perusahaan.
  Semakin banyak data pekerjaan sejenis, semakin akurat estimasi.

- **Biaya dalam Rp Juta** — sesuaikan currency dan skala sesuai kebutuhan.

- **Fungsi accessibility** untuk dashboard production (WCAG 2.2 compliance).
