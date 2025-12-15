"""Update aggregate median files for US and state levels."""
import argparse
from pathlib import Path

import pandas as pd

from data_pipeline import pipeline_config as config
from data_pipeline.utils import cbsa_utils, date_utils, logging_utils, path_utils


def load_metric(metric, max_date=None):
    """Load metric and convert to long format.

    Args:
        metric (str): The metric name.
        max_date (pd.Timestamp | None): Optional maximum date to include.

    Returns:
        pd.DataFrame: Long-format dataframe with columns [date, cbsa_code, metric, value].

    Raises:
        FileNotFoundError: If the metric file is missing.
        KeyError: If required columns are missing.
    """
    cfg = config.get_metric_config(metric)
    path = config.CBSA_DATA_DIR / cfg.app_filename
    if not path.exists():
        raise FileNotFoundError(f"Metric file missing: {path}")

    df = pd.read_parquet(path)

    required_columns = ["date", "cbsa_code"]
    value_columns = [col for col in [metric, f"{metric}_mom", f"{metric}_yoy"] if col in df.columns]
    if not value_columns:
        raise KeyError(f"Metric columns missing from {path.name} for metric {metric}")

    df = df[required_columns + value_columns]
    df["date"] = pd.to_datetime(df["date"])
    if max_date is not None:
        df = df[df["date"] <= max_date]

    df["cbsa_code"] = df["cbsa_code"].astype("int32")

    long_df = df.melt(
        id_vars=["date", "cbsa_code"],
        value_vars=value_columns,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"])
    long_df["metric"] = long_df["metric"].astype("string")
    return long_df[["date", "cbsa_code", "metric", "value"]]


def _expand_metadata_states(metadata):
    """Expand multi-state CBSAs into separate state rows for aggregation.

    Args:
        metadata (pd.DataFrame): CBSA metadata dataframe.

    Returns:
        pd.DataFrame: Expanded metadata mapping CBSA codes to state FIPS codes.
    """
    if "state" in metadata.columns:
        state_series = metadata["state"]
    else:
        state_series = metadata["primary_state"]
    
    meta = metadata[["cbsa_code"]].copy()
    meta["state_list"] = state_series.fillna("").str.split("-")
    meta = meta.explode("state_list")
    meta["state_list"] = meta["state_list"].str.strip().str.upper()
    meta = meta[meta["state_list"] != ""]
    
    # DC uses VA as its primary state
    meta = meta[meta["state_list"] != "DC"]

    state_fips_map = cbsa_utils.load_state_fips_map()
    lookup = dict(zip(state_fips_map["primary_state"], state_fips_map["state_fips_code"]))
    meta["state_fips_code"] = meta["state_list"].map(lookup)
    meta = meta.dropna(subset=["state_fips_code"])
    meta["state_fips_code"] = meta["state_fips_code"].astype("category")
    return meta[["cbsa_code", "state_fips_code"]].drop_duplicates()


def filter_month_range(dataframe, start=None):
    """Filter dataframe to dates on/after start date.

    Args:
        dataframe (pd.DataFrame): The dataframe to filter.
        start (pd.Timestamp | None): The start date (inclusive).

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    if start is None:
        return dataframe
    return dataframe[dataframe["date"] >= start]


def aggregate_metric(dataframe, metadata, metro_only=False):
    """Calculate US and state-level medians for a metric.

    Args:
        dataframe (pd.DataFrame): The metric data.
        metadata (pd.DataFrame): Enriched CBSA metadata.
        metro_only (bool): If True, filter to Metro CBSAs before aggregating.

    Returns:
        pd.DataFrame: Aggregated medians [metric, date, fipsstatecode, value].
    """
    metadata = metadata.copy()

    merged = dataframe.merge(metadata[["cbsa_code", "cbsa_type"]], on="cbsa_code", how="left")
    if metro_only:
        merged = merged[merged["cbsa_type"].str.contains("Metro", case=False, na=False)]

    merged = merged.dropna(subset=["value"])
    if merged.empty:
        return pd.DataFrame(columns=["metric", "date", "fipsstatecode", "value"])

    us_medians = (
        merged.groupby(["metric", "date"], observed=True)["value"]
        .median()
        .rename("value")
        .reset_index()
    )
    us_medians["fipsstatecode"] = config.US_MEDIAN_CODE
    expanded_states = _expand_metadata_states(metadata)
    expanded_states = expanded_states[expanded_states["cbsa_code"].isin(merged["cbsa_code"].unique())]

    state_combined = merged.merge(expanded_states, on="cbsa_code", how="inner")
    state_medians = (
        state_combined.groupby(["metric", "date", "state_fips_code"], observed=True)["value"]
        .median()
        .rename("value")
        .reset_index()
        .rename(columns={"state_fips_code": "fipsstatecode"})
    )
    result = pd.concat([us_medians, state_medians], ignore_index=True)
    result["fipsstatecode"] = result["fipsstatecode"].astype("category")
    result["value"] = result["value"].astype("float64")
    result["date"] = pd.to_datetime(result["date"])
    result["metric"] = result["metric"].astype("category")
    return result.sort_values(["metric", "date", "fipsstatecode"]).reset_index(drop=True)


def save_aggregates(dataframe, path):
    """Save aggregate dataframe to parquet.

    Args:
        dataframe (pd.DataFrame): The aggregate data to save.
        path (Path): Target file path.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(target_path, index=False)


def load_existing_aggregates(path):
    """Load existing aggregate parquet file.

    Args:
        path (Path): Path to the aggregate file.

    Returns:
        pd.DataFrame: Existing aggregates or empty DataFrame if missing.
    """
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main(rebuild=False, lookback_months=3, metrics=None):
    """Run aggregation pipeline.

    Args:
        rebuild (bool): If True, recompute from full history.
        lookback_months (int): Months of history to recompute if not rebuilding.
        metrics (list[str] | None): Optional list of metrics to process.

    Returns:
        int: Exit code.
    """
    logger = logging_utils.get_logger("update_aggregates")

    metadata = cbsa_utils.get_enriched_metadata()

    if rebuild:
        start = None # Build it from full history
    else:
        metric_list = metrics or config.list_all_metrics()
        latest_date = path_utils.find_latest_global_date(metric_list)
        if latest_date is None:
            logger.error("No metric files found")
            return 1
        start = date_utils.month_start(latest_date) - pd.DateOffset(months=lookback_months)

    frames_all: list[pd.DataFrame] = []
    frames_metro: list[pd.DataFrame] = []

    metric_iterable = metrics or config.list_all_metrics()
    for metric in metric_iterable:
        df = load_metric(metric)
        df = filter_month_range(df, start)
        if df.empty:
            logger.warning("Metric %s has no data in the selected range", metric)
            continue

        frames_all.append(aggregate_metric(df, metadata, metro_only=False))
        frames_metro.append(aggregate_metric(df, metadata, metro_only=True))

    if not frames_all:
        logger.warning("No new aggregate data generated.")
        return 0

    new_agg_all = pd.concat(frames_all, ignore_index=True)
    new_agg_metro = pd.concat(frames_metro, ignore_index=True)

    if not rebuild:
        # Load existing data
        old_agg_all = load_existing_aggregates(config.AGGREGATE_ALL_FILENAME)
        old_agg_metro = load_existing_aggregates(config.AGGREGATE_METRO_FILENAME)
        
        # Combine and deduplicate (keeping new values if overlap)
        if not old_agg_all.empty:
            agg_all = pd.concat([old_agg_all, new_agg_all], ignore_index=True)
            agg_all = agg_all.drop_duplicates(subset=["metric", "date", "fipsstatecode"], keep="last")
        else:
            agg_all = new_agg_all
            
        if not old_agg_metro.empty:
            agg_metro = pd.concat([old_agg_metro, new_agg_metro], ignore_index=True)
            agg_metro = agg_metro.drop_duplicates(subset=["metric", "date", "fipsstatecode"], keep="last")
        else:
            agg_metro = new_agg_metro
    else:
        agg_all = new_agg_all
        agg_metro = new_agg_metro
        
    # Ensure correct sorting
    agg_all = agg_all.sort_values(["metric", "date", "fipsstatecode"]).reset_index(drop=True)
    agg_metro = agg_metro.sort_values(["metric", "date", "fipsstatecode"]).reset_index(drop=True)

    save_aggregates(agg_all, config.AGGREGATE_ALL_FILENAME)
    save_aggregates(agg_metro, config.AGGREGATE_METRO_FILENAME)

    logger.info("Aggregates metrics updated: %s rows (all), %s rows (metro-only)", len(agg_all), len(agg_metro))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update aggregate metrics parquet files")
    parser.add_argument("--rebuild", action="store_true", help="Recalculate aggregates from full history")
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=3,
        help="Months of history to include when not rebuilding (default: 3)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional list of metrics to recompute (defaults to all)",
    )
    args = parser.parse_args()
    raise SystemExit(
        main(
            rebuild=args.rebuild,
            lookback_months=args.lookback_months,
            metrics=args.metrics,
        )
    )
