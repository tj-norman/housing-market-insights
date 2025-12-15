"""Process latest FRED data into dashboard-ready parquet files."""

import argparse

import numpy as np
import pandas as pd

from data_pipeline import pipeline_config as config
from data_pipeline.utils import (
    date_utils,
    logging_utils,
    metric_transformers,
    path_utils,
)


def load_landing_zone_data(metric):
    """Load all CSV files for a metric from the landing zone.

    Args:
        metric (str): The metric name.

    Returns:
        pd.DataFrame: Combined dataframe of all CSVs for the metric.
    """
    landing_zone = path_utils.get_landing_zone_dir()
    frames = []
    for csv_path in sorted(landing_zone.glob(f"{metric}_*.csv")):
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "value", "cbsa_code"])
        df["cbsa_code"] = df["cbsa_code"].astype("int32")
        frames.append(df[["date", "cbsa_code", "value"]])
    if not frames:
        return pd.DataFrame(columns=["date", "cbsa_code", "value"])
    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_existing_metric(metric):
    """Load existing metric parquet file.

    Args:
        metric (str): The metric name.

    Returns:
        pd.DataFrame | None: The existing dataframe, or None if not found.
    """
    cfg = config.get_metric_config(metric)
    path = config.CBSA_DATA_DIR / cfg.app_filename
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_metric(metric, dataframe):
    """Save dataframe to parquet file.

    Args:
        metric (str): The metric name.
        dataframe (pd.DataFrame): The dataframe to save.
    """
    cfg = config.get_metric_config(metric)
    target_path = config.CBSA_DATA_DIR / cfg.app_filename
    dataframe.to_parquet(target_path, index=False)


def build_ratio_long(metric, earliest_new):
    """Calculate ratio metric from numerator and denominator data.

    Args:
        metric (str): The ratio metric name.
        earliest_new (pd.Timestamp): Earliest date to process.

    Returns:
        pd.DataFrame: Calculated ratio dataframe.

    Raises:
        FileNotFoundError: If dependency files are missing.
    """
    numerator_name, denominator_name = config.get_ratio_dependency(metric)
    numerator_cfg = config.get_metric_config(numerator_name)
    denominator_cfg = config.get_metric_config(denominator_name)

    numerator_path = config.CBSA_DATA_DIR / numerator_cfg.app_filename
    denominator_path = config.CBSA_DATA_DIR / denominator_cfg.app_filename
    if not numerator_path.exists() or not denominator_path.exists():
        raise FileNotFoundError(
            f"Missing dependency for ratio metric {metric}: {numerator_path.name}, {denominator_path.name}"
        )

    numerator_table = pd.read_parquet(numerator_path, columns=["date", "cbsa_code", numerator_name])
    denominator_table = pd.read_parquet(denominator_path, columns=["date", "cbsa_code", denominator_name])

    numerator_long = numerator_table.rename(columns={numerator_name: "numerator"})
    denominator_long = denominator_table.rename(columns={denominator_name: "denominator"})

    merged = numerator_long.merge(
        denominator_long,
        on=["date", "cbsa_code"],
        how="inner",
    )
    merged["value"] = (merged["numerator"] / merged["denominator"]) * 100
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged = merged.dropna(subset=["date"])
    merged = merged[merged["date"] >= earliest_new]
    return merged[["date", "cbsa_code", "value"]]


def process_metric(metric, earliest_new):
    """Process updates for a single metric.

    Args:
        metric (str): The metric name.
        earliest_new (pd.Timestamp): Earliest date to process.
    """
    logger = logging_utils.get_logger("process_latest_fred_data")
    cfg = config.get_metric_config(metric)

    if metric in config.RATIO_METRICS:
        fresh_data = build_ratio_long(metric, earliest_new)
        if fresh_data.empty:
            logger.info("No new ratio data for %s", metric)
            return
    else:
        fresh_data = load_landing_zone_data(metric)
        if fresh_data.empty:
            logger.info("No landing zone data found for %s", metric)
            return

        fresh_data = fresh_data[fresh_data["date"] >= earliest_new]
        if fresh_data.empty:
            logger.info("No new data for %s", metric)
            return

    existing_table = load_existing_metric(metric)
    updated_table = metric_transformers.merge_existing_with_new(
        existing_table=existing_table,
        new_data=fresh_data,
        metric_name=metric,
        is_ratio=cfg.is_ratio_metric,
    )

    save_metric(metric, updated_table)
    logger.info("Updated %s with %s records", metric, len(updated_table))


def clean_landing_zone(metrics):
    """Remove processed CSV files from the landing zone.

    Args:
        metrics (list[str]): List of metric names to clean up.
    """
    landing_zone = path_utils.get_landing_zone_dir()
    for metric in metrics:
        for csv_path in landing_zone.glob(f"{metric}_*.csv"):
            csv_path.unlink()


def main(selected_metrics=None):
    """Main entry point for processing data.

    Args:
        selected_metrics (list[str] | None): Optional list of metrics to process.

    Returns:
        int: Exit code.
    """
    logger = logging_utils.get_logger("process_latest_fred_data")

    metrics = selected_metrics or config.list_all_metrics()
    non_ratio_metrics = [m for m in metrics if m not in config.RATIO_METRICS]
    ratio_metrics = [m for m in metrics if m in config.RATIO_METRICS]
    ordered_metrics = non_ratio_metrics + ratio_metrics

    latest_date = path_utils.find_latest_global_date(metrics)
    if latest_date is None:
        logger.error("No existing app_data files found. Run full processing first.")
        return 1

    earliest_new = date_utils.month_start(latest_date) - pd.DateOffset(months=12)

    backup_path = path_utils.snapshot_app_data()
    logger.info("Backed up to %s", backup_path)

    for metric in ordered_metrics:
        process_metric(metric, earliest_new)

    clean_landing_zone(non_ratio_metrics)
    logger.info("Processed %s metrics", len(metrics))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process latest FRED data into app_data")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional list of metric names to process. Defaults to all metrics.",
    )
    args = parser.parse_args()
    metrics = args.metrics if args.metrics else None
    raise SystemExit(main(metrics))