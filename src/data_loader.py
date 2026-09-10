"""Data loading and cleaning for the Maintenance Decision Support System."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

DATETIME_FILES = ("telemetry", "errors", "failures", "maintenance", "cost")
FILE_MAP = {
    "machines": "PdM_machines",
    "telemetry": "PdM_telemetry",
    "errors": "PdM_errors",
    "failures": "PdM_failures",
    "maintenance": "PdM_maint",
    "cost": "PdM_cost",
    "engineering": "PdM_engineering",
}


def load_all(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """Load and parse all 7 CSV files into a dict of DataFrames."""
    data = {}
    for key, filename in FILE_MAP.items():
        df = pd.read_csv(f"{data_dir}/{filename}.csv")
        if key in DATETIME_FILES:
            df["datetime"] = pd.to_datetime(df["datetime"])
        data[key] = df
    return data


def clean_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Clean each DataFrame in place per-file rules and return the dict."""
    data = {k: df.copy() for k, df in data.items()}

    machines = data.get("machines")
    if machines is not None:
        data["machines"] = machines.drop_duplicates(subset="machineID")

    telemetry = data.get("telemetry")
    if telemetry is not None:
        sensor_cols = ["volt", "rot", "press", "vib"]
        telemetry = telemetry.dropna(subset=sensor_cols).copy()
        for col in sensor_cols:
            mean = telemetry[col].mean()
            std = telemetry[col].std()
            if std == 0 or np.isnan(std):
                continue
            lo, hi = mean - 3 * std, mean + 3 * std
            telemetry = telemetry[(telemetry[col] >= lo) & (telemetry[col] <= hi)]
        data["telemetry"] = telemetry

    failures = data.get("failures")
    if failures is not None:
        failures = failures.sort_values("datetime").copy()
        data["failures"] = failures.drop_duplicates(
            subset=["machineID", "datetime"]
        ).copy()

    cost = data.get("cost")
    if cost is not None:
        numeric_cols = [
            "parts_cost_Jt",
            "labor_hours",
            "labor_cost_Jt",
            "total_repair_cost_Jt",
            "downtime_hours",
            "replacement_cost_Jt",
            "is_replacement",
        ]
        for col in numeric_cols:
            cost[col] = pd.to_numeric(cost[col], errors="coerce")
        cost["total_repair_cost_Jt"] = cost["total_repair_cost_Jt"].fillna(0.0)
        data["cost"] = cost

    return data


def merge_failure_cost(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join failures with cost (datetime+machineID) then with machines."""
    failures = data["failures"]
    cost = data["cost"]
    machines = data["machines"]
    merged = failures.merge(
        cost, on=["datetime", "machineID"], how="left"
    ).merge(machines, on="machineID", how="left")
    return merged
