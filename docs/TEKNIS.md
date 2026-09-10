================================================================================
MDSS — MAINTENANCE DECISION SUPPORT SYSTEM
Dokumentasi Teknis: Metode, Langkah, dan Cara Melacak Error
================================================================================

TANGGAL: 2026-09-10
CABANG: mdss-analysis (sudah di-merge ke main via PR #3)
LOKASI: /mnt/d/Orca-e2pay/orca-1-e2pay/maintenance-project


--------------------------------------------------------------------------------
1. RINGKASAN ARSITEKTUR
--------------------------------------------------------------------------------
Pipeline data -> analisa -> dashboard:

  data/*.csv (7 file raw)
    -> src/data_loader.py   (load + clean)
    -> src/*.py (10 modul analisa)
    -> dashboard/app.py     (build_payload -> JSON -> HTML render)
    -> docs/index.html      (static export utk GitHub Pages)

Alur data lengkap:
  1. data_loader.load_all()     membaca 7 CSV
  2. data_loader.clean_data()   bersihkan NaN/duplicate/outlier
  3. build_payload()            panggil semua modul analisa
  4. HTML_PAGE + __PAYLOAD__    dashboard render (Chart.js CDN)
  5. export_static.py           generate docs/index.html utk GitHub Pages


--------------------------------------------------------------------------------
2. STRUKTUR FILE
--------------------------------------------------------------------------------
  data/
    PdM_telemetry.csv   875k baris sensor per jam (volt/rot/press/vib)
    PdM_failures.csv    1662 baris failure (datetime/machineID/failure)
    PdM_maint.csv       2062 baris maintenance (type: scheduled/unscheduled)
    PdM_cost.csv        1662 baris biaya repair
    PdM_machines.csv    100 mesin (model/age)
    PdM_errors.csv      868 baris error code (error1-5)
    PdM_engineering.csv 79 baris job planning

  src/                     (modul analisa)
    data_loader.py         load 7 CSV + clean (NaN, dup, outlier sensor 3-sigma)
    metrics.py             MTBF, MTTR, availability, frekuensi, biaya
    priority.py            skor prioritas 0-100 (URGENT/HIGH/MEDIUM/LOW)
    decision.py            keputusan REPAIR/REVIEW/REPLACE (cost ratio)
    projection.py          lifecycle stage + regresi slope + timeline
    estimation.py          estimasi durasi/resource pekerjaan
    forecast.py            [BARU] regresi linear forecast failures & cost
    telemetry.py           [BARU] sensor trend + anomaly score + early warning
    components.py          [BARU] analisa per komponen (comp1-4)
    error_analysis.py      [BARU] error code -> failure predictor
    maintenance_type.py    [BARU] scheduled vs unscheduled ratio

  tests/                   (pytest, 103 test)
    test_*.py              test per modul

  dashboard/app.py         server HTTP + HTML/CSS/JS (single file)
  export_static.py         generate docs/index.html
  docs/index.html          GitHub Pages (static deployment)


--------------------------------------------------------------------------------
3. METODE ANALISA (5 MODUL BARU — INI YANG DITAMBAHKAN)
--------------------------------------------------------------------------------
Semua modul baru ikut pola yang sama:
  - input: DataFrame (raw dari data_loader)
  - proses: groupby + numpy
  - output: DataFrame / dict dengan kolom jelas
  - NaN-safe: gak pernah crash pada data kosong


3.1 FORECAST (src/forecast.py) — REGRESI LINEAR
  METODE: np.polyfit derajat 1 (least squares) pada jumlah failure/biaya
          per kuartal. Di-fit per mesin.
  LANGKAH:
    1. pivot failure/cost per kuartal (reuse trend_mtbf_over_time /
       trend_cost_over_time dari projection.py)
    2. np.polyfit(x=index kuartal, y=jumlah) -> slope + intercept
    3. R^2 = 1 - SS_res/SS_tot  (kualitas fit)
    4. forecast N kuartal ke depan = slope*(x_terakhir+j) + intercept
    5. trend_direction: slope>0.2 increasing, <-0.2 decreasing, else flat
  OUTPUT:
    project_failures  -> machineID, slope, r2, forecast_sum_ahead, trend_direction
    project_cost_trend-> machineID, slope, r2, forecast_sum_ahead
  ERROR TRACKING:
    - slope/r2 None jika <2 titik data (mesin jarang failure)
    - forecast_sum_ahead None jika slope NaN
    - data kosong -> DataFrame kosong dengan kolom benar

3.2 TELEMETRY (src/telemetry.py) — SENSOR TREND + ANOMALY
  METODE:
    a. sensor_trend: aggregasi daily-mean per sensor -> fit slope
       (np.polyfit) per mesin. sensor_stability = std dari 4 slope.
    b. anomaly_scores: z-score tiap pembacaan vs mean/std mesin sendiri;
       hitung pembacaan >3-sigma; anomaly_score = jumlah/total.
       at_risk = anomaly>0.2 ATAU slope outlier vs fleet (z>2.5) ATAU
       vib_slope>0.5 ATAU rot_slope>1.0.
    c. early_warning: mean sensor 7 hari SEBELUM failure vs baseline
       (mean keseluruhan mesin) -> delta.
  LANGKAH:
    1. sensor_trend: telemetry.groupby(machineID) -> resample harian
       -> polyfit slope per sensor
    2. anomaly_scores: tiap mesin, tiap sensor, z=(x-mean)/std, count >3
    3. merge slope + anomaly -> at_risk flag
    4. early_warning: per failure, filter telemetry dalam window_days
       sebelum tanggal failure, mean vs baseline
  PENTING (PITFALL YANG SUDAH DITEMUKAN):
    - Data drift sensor SANGAT halus (slope harian ~0.001-0.02), jadi
      threshold absolut (vib>0.5, rot>1.0) TIDAK cukup -> ditambah
      deteksi outlier slope vs seluruh armada (z-score >2.5). Tanpa ini
      at_risk = 0. Sekarang 3 mesin terdeteksi (38, 60, 90).
    - Sensor std bisa 0 (konstan) -> guard supaya tidak bagi nol.
  OUTPUT:
    sensor_trend  -> machineID, volt/rot/press/vib_slope, sensor_stability, age
    anomaly_scores-> machineID, anomaly_score, high_anomaly_count,
                     total_readings, at_risk, + slope cols
    early_warning -> machineID, failure_datetime, pre_failure_{sensor},
                     baseline_{sensor}, delta_{sensor}

3.3 COMPONENTS (src/components.py) — ANALISA PER KOMPONEN
  METODE: groupby kolom failure/component (comp1-4).
  LANGKAH:
    1. component_failure_summary: count per comp + % total + first/last seen
    2. component_cost_summary: sum/avg/max biaya per comp
    3. component_mtbf: interval antar failure per comp per mesin -> avg
    4. component_risk_score: merge fail + cost, normalize min-max,
       risk = 0.6*norm(n_failures) + 0.4*norm(total_cost),
       level HIGH>=0.66 / MEDIUM>=0.33 / LOW else
  OUTPUT: component, n_failures, total_cost_Jt, risk_score, risk_level
  ERROR TRACKING: NaN-safe; component_failure pakai kolom "failure",
       cost pakai kolom "component" (nama beda di 2 file!) — ini penting.

3.4 ERROR ANALYSIS (src/error_analysis.py) — ERROR -> FAILURE PREDICTOR
  METODE: hitung korelasi temporal antara error dan failure.
  LANGKAH:
    1. error_frequency: count per errorID + % total
    2. error_before_failure: per failure, count error SAMA mesin dalam
       window 14 hari SEBELUM (strict before). Left join -> failure
       tanpa error tetap masuk (0).
    3. error_failure_correlation: per errorID, hit_rate =
       n_preceding_failure / total_occurrences
    4. predictor_summary: top error by hit_rate + coverage =
       failures dengan >=1 error sebelum / total failures
  OUTPUT: errorID, total_occurrences, n_preceding_failure, hit_rate
  OUTPUT SUMMARY: top_errors + coverage (36% dari data)
  ERROR TRACKING: window strict "sebelum" (gak termasuk tanggal sama);

3.5 MAINTENANCE TYPE (src/maintenance_type.py)
  METODE: groupby kolom type (scheduled/unscheduled).
  LANGKAH:
    1. maintenance_type_ratio: count per type + % total
    2. preventive_maturity: per mesin, schedule_ratio =
       scheduled/total; maturity HIGH>=0.7 / MEDIUM>=0.4 / LOW
    3. unscheduled_cost_penalty: banding avg biaya repair scheduled vs
       unscheduled -> penalty_pct
  OUTPUT: type/n/pct | machineID/n_maintenance/n_scheduled/n_unscheduled/
          schedule_ratio/maturity
  TEMUAN: 80.6% maintenance = unscheduled (reaktif) — maturitas rendah.


--------------------------------------------------------------------------------
4. DASHBOARD WIRING (dashboard/app.py)
--------------------------------------------------------------------------------
  Modul baru dihubungkan di build_payload() (sekitar baris 65-170):
    1. import modul baru (bagian atas, baris ~20-42)
    2. panggil fungsi (blok "new analysis modules", sekitar baris 90-110)
    3. tambahkan hasil ke return dict (sekitar baris 150-180)
    4. HTML section baru (sekitar baris 560-650)
    5. JS render (sekitar baris 750-850)

  SECTION BARU DI DASHBOARD:
    - "Komponen & Health" (components-section):
        3 chart: chart-component, chart-anomaly, chart-error
        2 tabel: component-body, telemetry-body
    - "Maintenance Type & Error Leading Indicator" (maint-section):
        2 tabel: maint-body, predictor-body

  ERROR TRACKING di JS: tiap blok render di-bungkus try/catch dan
    console.error("nama-blok", e) — cek devtools console utk debug.


--------------------------------------------------------------------------------
5. CARA MENJALANKAN / VERIFIKASI
--------------------------------------------------------------------------------
  # 1. Test semua (103 test)
  cd /mnt/d/Orca-e2pay/orca-1-e2pay/maintenance-project
  python3 -m pytest -q

  # 2. Jalankan dashboard lokal
  python3 dashboard/app.py --port 8323 --data-dir data
  # buka: http://127.0.0.1:8323

  # 3. Regenerate static export (GITHUB PAGES)
  python3 export_static.py
  # menghasilkan docs/index.html — commit setelah perubahan

  # 4. Verifikasi API payload (debug cepat)
  curl -s http://127.0.0.1:8323/api/data | python3 -m json.tool


--------------------------------------------------------------------------------
6. CARA MELACAK ERROR (TROUBLESHOOTING)
--------------------------------------------------------------------------------
  GEJALA: dashboard tidak tampil / section kosong / angka salah.

  LANGKAH PENCARIAN (dari luar ke dalam):

  [1] CEK SERVER JALAN?
      curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8323/
      - 000 = server mati / belum up. Cek log server (lihat di terminal
        tempat start). Kalau pakai background, cek dengan:
          ps aux | grep app.py
      - 200 = server OK, lanjut ke [2]

  [2] CEK PAYLOAD API (data benar gak?)
      curl -s http://127.0.0.1:8323/api/data | python3 -m json.tool
      - Cek key: components, telemetry_trend, anomaly, error_predictor,
        maintenance_ratio, forecast_failures, forecast_cost
      - Key hilang = import/panggil gagal di build_payload. Cek error
        saat server start (ImportError / NameError di terminal).

  [3] CEK TEST (yang mana gagal?)
      python3 -m pytest tests/test_<modul>.py -q
      - test_forecast.py  -> forecast.py
      - test_telemetry.py -> telemetry.py
      - test_components.py-> components.py
      - test_error_analysis.py -> error_analysis.py
      - test_maintenance_type.py -> maintenance_type.py
      - Gagal 1 test = modul itu yang bermasalah.

  [4] CEK DATA (input salah?)
      python3 -c "import pandas as pd; print(pd.read_csv('data/PdM_failures.csv').head())"
      - Kolom harus sesuai yang diharapkan modul.
      - Modul components pakai kolom "failure" (failures csv) dan
        "component" (cost csv) — 2 nama beda, jangan ketuker.
      - Telemetry 875k baris — pastikan pandas cukup RAM (harusnya OK).

  [5] CEK JS (render gagal?) — BUKA BROWSER > F12 > CONSOLE
      - Error console: "component", "telemetry", "maint-type", "predictor",
        "error-chart" dst — nama blok yang catch-nya jalan.
      - Data hilang di tabel = DATA.<key> undefined/empty di JS.

  [6] GITHUB PAGES TIDAK UPDATE?
      - Regenerate: python3 export_static.py
      - Commit + push docs/index.html ke main (atau branch->PR->merge)
      - Tunggu 1-2 menit, hard-refresh (Ctrl+Shift+R)

  [7] ROLLBACK (kalau buntu)
      git log --oneline
      git checkout <commit_sebelumnya> -- src/ dashboard/ tests/
      # lalu test ulang


--------------------------------------------------------------------------------
7. GIT FLOW / WORKFLOW (ENTERPRISE)
--------------------------------------------------------------------------------
  main = selalu stabil (live di Pages).
  Fitur baru = branch -> PR -> merge -> delete branch.

  LANGKAH STANDAR:
    git checkout main && git pull
    git checkout -b mdss-<fitur-baru>
    # ... kerja + test ...
    python3 -m pytest -q          # harus green
    git add -A && git commit -m "<deskripsi>"
    git push -u origin mdss-<fitur-baru>
    # buat PR di GitHub -> review -> merge
    # (opsional) hapus branch setelah merge

  JANGAN: push langsung ke main. Semua lewat PR.


--------------------------------------------------------------------------------
8. KEPUTUSAN TEKNIS PENTING (SUPAYA TIDAK MENGULANG DEBUG)
--------------------------------------------------------------------------------
  1. Threshold at-risk telemetry: absolut + outlier z-score (kombinasi)
     -> karena drift halus, absolut saja = 0 terdeteksi.
  2. forecast.py import: pakai try/except dari src.projection ATAU
     projection (dua mode: app.py vs pytest) — jangan diubah ke satu saja.
  3. component "failure" vs "component" kolom: beda nama antar file CSV,
     jangan di-rename satu sisi.
  4. Semua modul NaN-safe: jika data kosong, return DataFrame kosong
     dengan kolom benar, bukan raise.
  5. test count 103 (56 lama + 47 baru). Angka ini harus selalu pass
     sebelum commit.
  6. export_static.py WAJIB dijalankan setelah ubah dashboard/app.py,
     supaya GitHub Pages ikut update.


--------------------------------------------------------------------------------
9. ANGKA KUNCI SAAT INI (BASELINE — BUAT BANDINGKAN KALAU BERUBAH)
--------------------------------------------------------------------------------
  - 100 mesin, 1662 failures, 1 tahun (2015)
  - MTBF avg 847.9 jam, MTTR avg 15.8 jam, availability 96.6%
  - Total biaya: 74.071,6 Jt
  - URGENT: 2 mesin, HIGH: 21
  - Decision: REPAIR 71, REVIEW 29, REPLACE 0
  - Maintenance: scheduled 400 (19.4%), unscheduled 1662 (80.6%)
  - Error coverage 14-hari: 36% (598/1662 failures didahului error)
  - At-risk telemetry: 3 mesin (38, 60, 90)
  - Trend: 80.6% maintenance reaktif -> opportunity preventif
================================================================================