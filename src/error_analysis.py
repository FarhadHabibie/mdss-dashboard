"""Error-code frequency and failure-prediction analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd


def error_frequency(errors_df: pd.DataFrame) -> pd.DataFrame:
    """Error frequency per errorID: count and share of all errors."""
    if errors_df.empty:
        return pd.DataFrame(columns=["errorID", "n_errors", "pct"])
    counts = errors_df.groupby("errorID").size().reset_index(name="n_errors")
    counts["pct"] = counts["n_errors"] / counts["n_errors"].sum()
    return counts


def error_before_failure(
    errors_df: pd.DataFrame, failures_df: pd.DataFrame, window_days: int = 14
) -> pd.DataFrame:
    """Errors on the same machine within window_days before each failure."""
    cols = ["machineID", "failure_datetime", "n_errors_before", "error_ids"]
    if failures_df.empty:
        return pd.DataFrame(columns=cols)
    failures = failures_df.copy()
    failures["datetime"] = pd.to_datetime(failures["datetime"])
    if errors_df.empty:
        return failures.rename(columns={"datetime": "failure_datetime"}).assign(
            n_errors_before=0, error_ids=""
        )[cols]
    errors = errors_df.copy()
    errors["datetime"] = pd.to_datetime(errors["datetime"])
    window = pd.Timedelta(days=window_days)
    merged = errors.merge(failures, on="machineID", suffixes=("_err", ""))
    in_window = merged[
        (merged["datetime_err"] >= merged["datetime"] - window)
        & (merged["datetime_err"] < merged["datetime"])
    ]
    agg = (
        in_window.groupby(["machineID", "datetime"])
        .agg(n_errors_before=("errorID", "count"), error_ids=("errorID", lambda s: ",".join(s)))
        .reset_index()
    )
    result = failures.merge(agg, on=["machineID", "datetime"], how="left")
    result["n_errors_before"] = result["n_errors_before"].fillna(0).astype(int)
    result["error_ids"] = result["error_ids"].fillna("")
    return result.rename(columns={"datetime": "failure_datetime"})[cols]


def error_failure_correlation(dist_df: pd.DataFrame) -> pd.DataFrame:
    """Per errorID: occurrences in failure windows vs total window occurrences."""
    cols = ["errorID", "total_occurrences", "n_preceding_failure", "hit_rate"]
    if dist_df.empty or "error_ids" not in dist_df.columns:
        return pd.DataFrame(columns=cols)
    exploded = dist_df.loc[dist_df["error_ids"] != "", ["machineID", "failure_datetime", "error_ids"]]
    exploded["error_ids"] = exploded["error_ids"].str.split(",")
    rows = exploded.explode("error_ids").drop(columns="machineID")
    rows = rows[rows["error_ids"] != ""]
    total = rows.groupby("error_ids").size().rename("total_occurrences")
    preceding = (
        rows.drop_duplicates(["failure_datetime", "error_ids"])
        .groupby("error_ids")
        .size()
        .rename("n_preceding_failure")
    )
    result = pd.DataFrame({"total_occurrences": total, "n_preceding_failure": preceding})
    result["hit_rate"] = result["n_preceding_failure"] / result["total_occurrences"].where(
        result["total_occurrences"] > 0
    )
    return result.reset_index().rename(columns={"error_ids": "errorID"}).sort_values(
        "hit_rate", ascending=False, na_position="last"
    )[cols]


def predictor_summary(
    errors_df: pd.DataFrame, failures_df: pd.DataFrame, window_days: int = 14
) -> dict:
    """Leading-indicator summary: top error codes by hit_rate and failure coverage."""
    dist = error_before_failure(errors_df, failures_df, window_days)
    corr = error_failure_correlation(dist)
    total_failures = len(failures_df)
    covered = int(dist["n_errors_before"].gt(0).sum()) if not dist.empty else 0
    coverage = covered / total_failures if total_failures else 0.0
    top_errors = [
        {"errorID": row.errorID, "hit_rate": row.hit_rate} for row in corr.itertuples()
    ]
    return {"top_errors": top_errors, "coverage": float(coverage)}