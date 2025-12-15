"""Metric transformation utilities for rebuilding dashboard-ready data."""

import numpy as np
import pandas as pd

from data_pipeline import pipeline_config as config


def stored_metric_to_long(existing_table, metric_name):
    """Convert a stored metric file into long form (date, cbsa_code, value).

    Args:
        existing_table (pd.DataFrame): The existing metric dataframe.
        metric_name (str): The name of the metric column.

    Returns:
        pd.DataFrame: Long-format dataframe.
    """
    if existing_table is None or existing_table.empty:
        return pd.DataFrame(columns=["date", "cbsa_code", "value"])

    long_df = existing_table[["date", "cbsa_code", metric_name]].copy()
    long_df = long_df.rename(columns={metric_name: "value"})
    long_df = long_df.dropna(subset=["value"])
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["cbsa_code"] = long_df["cbsa_code"].astype("int32", copy=False)
    long_df["value"] = long_df["value"].astype("float64", copy=False)
    return long_df.dropna(subset=["date"]).reset_index(drop=True)


def build_metric_table(long_df, metric_name, is_ratio):
    """Create dashboard metric table (tidy format) from long metric data.

    Args:
        long_df (pd.DataFrame): Long-format metric data.
        metric_name (str): The metric name.
        is_ratio (bool): Whether the metric is a ratio (affects MOM/YOY calculation).

    Returns:
        pd.DataFrame: Tidy metric table with MOM and YOY columns.
    """
    if long_df.empty:
        return pd.DataFrame(columns=["date", "cbsa_code", metric_name, f"{metric_name}_mom", f"{metric_name}_yoy"])

    df = long_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["cbsa_code"] = df["cbsa_code"].astype("int32", copy=False)
    df["value"] = df["value"].astype("float64", copy=False)
    df = df.sort_values(["cbsa_code", "date"])  # type: ignore[arg-type]

    df["prev_month"] = df.groupby("cbsa_code", observed=True)["value"].shift(1)
    df["prev_year"] = df.groupby("cbsa_code", observed=True)["value"].shift(12)

    if is_ratio:
        df["mom"] = df["value"] - df["prev_month"]
        df["yoy"] = df["value"] - df["prev_year"]
    else:
        df["mom"] = ((df["value"] - df["prev_month"]) / df["prev_month"]) * 100
        df["yoy"] = ((df["value"] - df["prev_year"]) / df["prev_year"]) * 100

    df.loc[~np.isfinite(df["mom"]), "mom"] = np.nan
    df.loc[~np.isfinite(df["yoy"]), "yoy"] = np.nan

    # Round MOM/YOY to configured decimal places
    df["mom"] = df["mom"].round(config.MOM_YOY_DECIMAL_PLACES)
    df["yoy"] = df["yoy"].round(config.MOM_YOY_DECIMAL_PLACES)

    table = df[["date", "cbsa_code", "value", "mom", "yoy"]].copy()
    table = table.rename(
        columns={
            "value": metric_name,
            "mom": f"{metric_name}_mom",
            "yoy": f"{metric_name}_yoy",
        }
    )

    # Optimize dtypes for final output
    metric_cfg = config.get_metric_config(metric_name)
    table["date"] = pd.to_datetime(table["date"])
    table["cbsa_code"] = table["cbsa_code"].astype("category", copy=False)
    
    # Round and set metric value dtype based on config
    if metric_cfg.decimal_places is not None:
        table[metric_name] = table[metric_name].round(metric_cfg.decimal_places)
    table[metric_name] = table[metric_name].astype(metric_cfg.value_dtype)
    
    # Set MOM/YOY dtypes
    table[f"{metric_name}_mom"] = table[f"{metric_name}_mom"].astype(config.MOM_YOY_DTYPE)
    table[f"{metric_name}_yoy"] = table[f"{metric_name}_yoy"].astype(config.MOM_YOY_DTYPE)

    table = table.sort_values(["date", "cbsa_code"]).reset_index(drop=True)
    return table


def merge_existing_with_new(
    existing_table,
    new_data,
    metric_name,
    is_ratio,
):
    """Merge old and new data, rebuild metric table.

    Args:
        existing_table (pd.DataFrame): The existing metric data.
        new_data (pd.DataFrame): The new data to merge.
        metric_name (str): The metric name.
        is_ratio (bool): Whether the metric is a ratio.

    Returns:
        pd.DataFrame: Merged and rebuilt metric table.
    """
    base_long = stored_metric_to_long(existing_table, metric_name)
    combined = pd.concat([base_long, new_data], ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["date", "cbsa_code"], keep="last")
        .sort_values(["date", "cbsa_code"])
        .reset_index(drop=True)
    )
    return build_metric_table(combined, metric_name, is_ratio)
