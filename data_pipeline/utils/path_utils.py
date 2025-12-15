"""Path and file helpers for the CBSA data pipeline."""

from datetime import datetime
from pathlib import Path
from shutil import copy2, rmtree

import pandas as pd

from data_pipeline import pipeline_config as config


def find_latest_app_data_date(metric_filename):
    """Get most recent date from a metric parquet file.

    Args:
        metric_filename (str): Name of the parquet file.

    Returns:
        pd.Timestamp | None: The latest date in the file, or None if empty/missing.
    """
    file_path = config.CBSA_DATA_DIR / metric_filename
    if not file_path.exists():
        return None

    df = pd.read_parquet(file_path, columns=["date"])
    if df.empty:
        return None

    return pd.to_datetime(df["date"].max())


def find_latest_global_date(metric_names):
    """Return latest intersection date across metrics.

    Args:
        metric_names (list[str]): List of metrics to check.

    Returns:
        pd.Timestamp | None: The minimum of the latest dates across all metrics.
    """
    latest_dates = []
    for metric in metric_names:
        cfg = config.get_metric_config(metric)
        dt = find_latest_app_data_date(cfg.app_filename)
        if dt is not None:
            latest_dates.append(dt)
    if not latest_dates:
        return None
    return min(latest_dates)


def get_landing_zone_dir():
    """Return landing zone directory for raw CSV files.

    Returns:
        Path: The landing zone directory path.
    """
    return config.LANDING_ZONE_DIR


def snapshot_app_data(backup_root=None, keep_count=1):
    """Backup current app_data files before processing and clean up old backups.

    Args:
        backup_root (Path | None): Root backup directory. Defaults to config.BACKUP_DIR.
        keep_count (int): Number of recent backups to keep. Defaults to 1 (overwriting style).

    Returns:
        Path: The directory where the backup was created.
    """
    backup_root = backup_root or config.BACKUP_DIR
    backup_root.mkdir(parents=True, exist_ok=True)
    timestamp_dir = backup_root / datetime.now().strftime(config.BACKUP_DATE_FORMAT)
    
    # If the directory exists (e.g. multiple runs in same day), clear it to ensure clean backup
    if timestamp_dir.exists():
        rmtree(timestamp_dir)
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in config.CBSA_DATA_DIR.glob("*.parquet"):
        target_path = timestamp_dir / parquet_path.name
        copy2(parquet_path, target_path)

    metadata_dir = config.APP_METADATA_DIR
    if metadata_dir.exists():
        backup_metadata_dir = timestamp_dir / "metadata"
        backup_metadata_dir.mkdir(exist_ok=True)
        for metadata_file in metadata_dir.glob("*.parquet"):
            metadata_target = backup_metadata_dir / metadata_file.name
            copy2(metadata_file, metadata_target)

    # Backup aggregate data
    agg_dir = config.AGGREGATE_DATA_DIR
    if agg_dir.exists():
        backup_agg_dir = timestamp_dir / "aggregate_data"
        backup_agg_dir.mkdir(exist_ok=True)
        for agg_file in agg_dir.glob("*.parquet"):
            agg_target = backup_agg_dir / agg_file.name
            copy2(agg_file, agg_target)

    # Cleanup old backups
    all_backups = sorted([d for d in backup_root.iterdir() if d.is_dir()])
    if len(all_backups) > keep_count:
        # Remove oldest backups, keeping the most recent 'keep_count'
        backups_to_remove = all_backups[:-keep_count]
        for backup_dir in backups_to_remove:
            rmtree(backup_dir)

    return timestamp_dir
