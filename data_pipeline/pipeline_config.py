"""Configuration settings for CBSA data pipeline."""

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PIPELINE_DIR = BASE_DIR / "data_pipeline"
APP_DATA_DIR = BASE_DIR / "app_data"
METADATA_DIR = BASE_DIR / "metadata"

CBSA_DATA_DIR = APP_DATA_DIR / "cbsa_data"
APP_METADATA_DIR = APP_DATA_DIR / "metadata"

LOG_DIR = DATA_PIPELINE_DIR / "logs"
BACKUP_DIR = DATA_PIPELINE_DIR / "backups"
LANDING_ZONE_DIR = DATA_PIPELINE_DIR / "landing_zone"

LOG_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
LANDING_ZONE_DIR.mkdir(parents=True, exist_ok=True)
APP_METADATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MetricConfig:
    """Configuration for an individual metric."""
    fred_prefix: str | None
    app_filename: str
    is_ratio_metric: bool = False
    value_dtype: str = "float32"
    decimal_places: int | None = None


METRICS = {
    "median_listing_price": MetricConfig(
        fred_prefix="MEDLISPRI",
        app_filename="median_listing_price.parquet",
        value_dtype="float64",
        decimal_places=0,
    ),
    "average_listing_price": MetricConfig(
        fred_prefix="AVELISPRI",
        app_filename="average_listing_price.parquet",
        value_dtype="float64",
        decimal_places=0,
    ),
    "active_listing_count": MetricConfig(
        fred_prefix="ACTLISCOU",
        app_filename="active_listing_count.parquet",
        value_dtype="int32",
        decimal_places=0,
    ),
    "median_days_on_market": MetricConfig(
        fred_prefix="MEDDAYONMAR",
        app_filename="median_days_on_market.parquet",
        value_dtype="int32",
        decimal_places=0,
    ),
    "new_listing_count": MetricConfig(
        fred_prefix="NEWLISCOU",
        app_filename="new_listing_count.parquet",
        value_dtype="int32",
        decimal_places=0,
    ),
    "pending_listing_count": MetricConfig(
        fred_prefix="PENLISCOU",
        app_filename="pending_listing_count.parquet",
        value_dtype="int32",
        decimal_places=0,
    ),
    "price_increase_count": MetricConfig(
        fred_prefix="PRIINCCOU",
        app_filename="price_increase_count.parquet",
        value_dtype="int32",
        decimal_places=0,
    ),
    "price_decrease_count": MetricConfig(
        fred_prefix="PRIREDCOU",
        app_filename="price_decrease_count.parquet",
        value_dtype="int32",
        decimal_places=0,
    ),
    "price_increase_ratio": MetricConfig(
        fred_prefix=None,
        app_filename="price_increase_ratio.parquet",
        is_ratio_metric=True,
        value_dtype="float32",
        decimal_places=1,
    ),
    "price_decrease_ratio": MetricConfig(
        fred_prefix=None,
        app_filename="price_decrease_ratio.parquet",
        is_ratio_metric=True,
        value_dtype="float32",
        decimal_places=1,
    ),
    "demand_score": MetricConfig(
        fred_prefix="DESCMSA",
        app_filename="demand_score.parquet",
        value_dtype="float32",
        decimal_places=1,
    ),
    "supply_score": MetricConfig(
        fred_prefix="SUSCMSA",
        app_filename="supply_score.parquet",
        value_dtype="float32",
        decimal_places=1,
    ),
    "median_square_feet": MetricConfig(
        fred_prefix="MEDSQUFEE",
        app_filename="median_square_feet.parquet",
        value_dtype="int32",
        decimal_places=0,
    ),
    "median_listing_price_per_sqft": MetricConfig(
        fred_prefix="MEDLISPRIPERSQUFEE",
        app_filename="median_listing_price_per_sqft.parquet",
        value_dtype="float32",
        decimal_places=1,
    ),
}


METRIC_GROUPS = {
    "core_metrics": [
        "median_listing_price",
        "active_listing_count",
        "new_listing_count",
        "pending_listing_count",
    ],
    "price_metrics": [
        "median_listing_price",
        "average_listing_price",
        "median_listing_price_per_sqft",
    ],
    "inventory_metrics": [
        "active_listing_count",
        "new_listing_count",
        "pending_listing_count",
        "price_decrease_count",
        "price_increase_count",
        "price_decrease_ratio",
        "price_increase_ratio",
    ],
    "market_health_metrics": [
        "demand_score",
        "supply_score",
        "median_days_on_market",
    ],
}

ALL_METRICS = list(METRICS.keys())
RATIO_METRICS = [name for name, cfg in METRICS.items() if cfg.is_ratio_metric]
SUPPLEMENTARY_ONLY_METRICS = [
    "demand_score",
    "supply_score",
]

MAIN_DATASETS = [
    "median_listing_price",
    "average_listing_price",
    "active_listing_count",
    "median_days_on_market",
    "new_listing_count",
    "pending_listing_count",
    "price_increase_count",
    "price_decrease_count",
]

ADDITIONAL_DATASETS = [
    "median_square_feet",
    "median_listing_price_per_sqft",
]

RATIO_DEPENDENCIES = {
    "price_decrease_ratio": ("price_decrease_count", "active_listing_count"),
    "price_increase_ratio": ("price_increase_count", "active_listing_count"),
}

MOM_YOY_DTYPE = "float32"
MOM_YOY_DECIMAL_PLACES = 2

FRED_RETRY_LIMIT = 3
CONSECUTIVE_CBSA_FAILURE_LIMIT = 5
REQUEST_TIMEOUT_SECONDS = 30
RATE_LIMIT_PER_MINUTE = 120

AGGREGATE_DATA_DIR = APP_DATA_DIR / "aggregate_data"
AGGREGATE_ALL_FILENAME = AGGREGATE_DATA_DIR / "aggregate_metrics_all.parquet"
AGGREGATE_METRO_FILENAME = AGGREGATE_DATA_DIR / "aggregate_metrics_metro.parquet"

US_MEDIAN_CODE = 0
US_MEDIAN_TITLE = "US Median"

DATE_FORMAT = "%Y-%m-%d"
BACKUP_DATE_FORMAT = "%Y%m%d"
LOG_FILE_BASENAME = "pipeline.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5


def get_metric_config(metric):
    return METRICS[metric]


def get_metrics_for_group(group_name):
    return METRIC_GROUPS.get(group_name, [])


def list_all_metrics():
    return ALL_METRICS


def get_ratio_dependency(metric):
    return RATIO_DEPENDENCIES[metric]
