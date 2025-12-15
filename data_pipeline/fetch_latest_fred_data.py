"""Fetch newly available FRED data for missing months."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from data_pipeline import pipeline_config as config
from data_pipeline.utils import cbsa_utils, date_utils, logging_utils, path_utils, retry_utils
from data_pipeline.utils.fred_api_client import FREDAPIClient


def determine_start_date(metrics):
    """Determine the start date for data fetching based on existing data.

    Args:
        metrics (list[str]): List of metric names to check.

    Returns:
        pd.Timestamp | None: The start date for fetching (next month after latest existing data), or None if no data exists.
    """
    latest = path_utils.find_latest_global_date(metrics)
    if latest is None:
        return None
    return date_utils.next_month(date_utils.month_start(latest))


def build_metric_plan(start_date):
    """Build a plan for which metrics to fetch and their configuration.

    Args:
        start_date (pd.Timestamp | None): The start date for fetching.

    Returns:
        dict: A dictionary mapping metric names to their fetch configuration (prefix, start_date).
    """
    plan = {}
    for metric in config.MAIN_DATASETS + config.SUPPLEMENTARY_ONLY_METRICS + config.ADDITIONAL_DATASETS:
        cfg = config.get_metric_config(metric)
        plan[metric] = {
            "fred_prefix": cfg.fred_prefix,
            "start_date": date_utils.format_date(start_date) if start_date else None,
        }
    return plan


def save_landing_zone_csv(metric, dataframe, suffix):
    """Save fetched data to a CSV in the landing zone.

    Args:
        metric (str): The metric name.
        dataframe (pd.DataFrame): The dataframe to save.
        suffix (str): Suffix for the filename (e.g., date).

    Returns:
        Path: The path to the saved file.
    """
    output_dir = path_utils.get_landing_zone_dir()
    filename = f"{metric}_{suffix}.csv"
    output_path = output_dir / filename
    dataframe.to_csv(output_path, index=False)
    return output_path


def fetch_metric(
    client,
    metric,
    fred_prefix,
    start_date,
    cbsa_codes,
):
    """Fetch data for a specific metric across multiple CBSAs.

    Args:
        client (FREDAPIClient): The FRED API client instance.
        metric (str): The metric name.
        fred_prefix (str): The FRED series prefix for the metric.
        start_date (str | None): The start date for fetching.
        cbsa_codes (list[int]): List of CBSA codes to fetch.

    Returns:
        pd.DataFrame: Combined dataframe of fetched data.
    """
    logger = logging_utils.get_logger("fetch_latest_fred_data")
    consecutive_state = retry_utils.ConsecutiveFailureState()

    frames = []
    for idx, cbsa_code in enumerate(cbsa_codes, start=1):
        series_id = f"{fred_prefix}{cbsa_code}"
        retry_state = retry_utils.MetricRetryState()

        while True:
            logger.info("Fetching %s (%s) attempt %s", series_id, metric, retry_state.attempts + 1)
            df = client.get_series_data(series_id, start_date)
            if df is not None and not df.empty:
                frames.append(df)
                consecutive_state.register_success()
                break

            retry_state.register_attempt()
            consecutive_state.register_failure()
            if retry_state.exhausted:
                logger.warning("Max retries reached for %s", series_id)
                break

            if consecutive_state.reached_threshold:
                raise RuntimeError(
                    f"Aborting collection: {config.CONSECUTIVE_CBSA_FAILURE_LIMIT} consecutive CBSA failures"
                )

        if idx % 100 == 0:
            logger.info("Processed %s/%s CBSA series for %s", idx, len(cbsa_codes), metric)

    if not frames:
        logger.warning("No data obtained for metric %s", metric)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["cbsa_code"] = combined["cbsa_code"].astype("int32")
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date", "value"])
    return combined


def main(group=None):
    """Main entry point for fetching FRED data.

    Args:
        group (str | None): Optional metric group to fetch.

    Returns:
        int: Exit code (0 for success).
    """
    logger = logging_utils.get_logger("fetch_latest_fred_data")
    metrics = config.get_metrics_for_group(group) if group else config.list_all_metrics()
    metrics = [m for m in metrics if config.get_metric_config(m).fred_prefix]

    start_date = determine_start_date(metrics)
    if start_date is None:
        logger.info("No existing data found - need full historical collection first")
        return 0

    logger.info("Fetching latest data from %s", date_utils.format_date(start_date))

    client = FREDAPIClient()

    base_cbsa_codes = cbsa_utils.get_base_cbsa_codes()
    supplementary_codes = cbsa_utils.get_supplementary_cbsa_codes()

    plan = build_metric_plan(start_date)

    for metric, details in plan.items():
        fred_prefix = details["fred_prefix"]
        if fred_prefix is None:
            continue  # Ratios computed locally

        if metric in config.SUPPLEMENTARY_ONLY_METRICS:
            codes = supplementary_codes
        else:
            codes = base_cbsa_codes

        logger.info("Fetching %s (%s CBSAs)", metric, len(codes))
        df = fetch_metric(client, metric, fred_prefix, details["start_date"], codes)
        if df.empty:
            logger.warning("Metric %s returned no data", metric)
            continue

        suffix = pd.Timestamp.now().strftime("%Y%m%d")
        output_path = save_landing_zone_csv(metric, df, suffix)
        logger.info("Saved %s rows to %s", len(df), output_path)

    logger.info("Fetch complete")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch latest FRED data for incremental updates")
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Optional metric group defined in pipeline_config.METRIC_GROUPS",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.group))
