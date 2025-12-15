"""Utility functions for working with CBSA metadata and code lists."""

from functools import lru_cache
from pathlib import Path
import pandas as pd

# Local imports
from data_pipeline import pipeline_config as config
from helpers import get_primary_state

BASE_CBSA_CODES_FILE = config.METADATA_DIR / "base_cbsa_codes.txt"
SUPPLEMENTARY_CBSA_CODES_FILE = config.METADATA_DIR / "supplementary_cbsa_codes.txt"
STATE_FIPS_CODES_FILE = config.APP_METADATA_DIR / "state_fips_codes.csv"


def _read_code_file(path):
    """Read and parse a code file (one code per line).

    Args:
        path (Path): Path to the code file.

    Returns:
        list[str]: List of codes.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"CBSA code file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


@lru_cache(maxsize=1)
def get_base_cbsa_codes():
    """Get list of base CBSA codes.

    Returns:
        list[str]: List of base CBSA codes.
    """
    return _read_code_file(BASE_CBSA_CODES_FILE)


@lru_cache(maxsize=1)
def get_supplementary_cbsa_codes():
    """Get list of supplementary CBSA codes.

    Returns:
        list[str]: List of supplementary CBSA codes.
    """
    return _read_code_file(SUPPLEMENTARY_CBSA_CODES_FILE)


@lru_cache(maxsize=1)
def load_cbsa_metadata():
    """Load CBSA metadata from parquet file.

    Returns:
        pd.DataFrame: DataFrame containing CBSA metadata with 'cbsa_code' as index.

    Raises:
        FileNotFoundError: If metadata file is missing.
        KeyError: If required columns are missing.
    """
    metadata_path = config.APP_METADATA_DIR / "cbsa_metadata.parquet"
    if metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
    else:
        raise FileNotFoundError("CBSA metadata parquet not found in app_data directory")

    metadata = metadata.copy()

    if "cbsa_code" not in metadata.columns:
        # Handle files that use cbsa_code as the index
        if metadata.index.name == "cbsa_code":
            metadata = metadata.reset_index()
        else:
            metadata = metadata.reset_index()
            if "cbsa_code" not in metadata.columns:
                raise KeyError("cbsa_code column missing from CBSA metadata")

    metadata["cbsa_code"] = metadata["cbsa_code"].astype("int32")
    metadata.set_index("cbsa_code", inplace=True, drop=False)
    return metadata


@lru_cache(maxsize=1)
def load_state_fips_map() -> pd.DataFrame:
    """Load state FIPS code mapping.

    Returns:
        pd.DataFrame: DataFrame mapping 'primary_state' to 'state_fips_code'.

    Raises:
        FileNotFoundError: If state FIPS file is missing.
    """
    if not STATE_FIPS_CODES_FILE.exists():
        raise FileNotFoundError(f"State FIPS mapping missing: {STATE_FIPS_CODES_FILE}")
    state_fips = pd.read_csv(STATE_FIPS_CODES_FILE, dtype={"fipsstatecode": "int16", "state": "string"})
    state_fips["state"] = state_fips["state"].str.upper()
    return state_fips.rename(columns={"state": "primary_state", "fipsstatecode": "state_fips_code"})


@lru_cache(maxsize=1)
def get_enriched_metadata() -> pd.DataFrame:
    """Return CBSA metadata with primary state and state FIPS codes.

    Returns:
        pd.DataFrame: Enriched metadata with 'primary_state' and 'state_fips_code'.
    """
    metadata = load_cbsa_metadata().reset_index(drop=True).copy()

    if "state" not in metadata.columns:
        metadata["state"] = metadata["cbsa_title"].str.extract(r", ([A-Z-]+)$")[0]

    metadata["state"] = metadata["state"].str.upper()
    metadata["primary_state"] = metadata["state"].apply(get_primary_state)

    state_fips = load_state_fips_map()
    metadata = metadata.merge(state_fips, on="primary_state", how="left")

    metadata["state_fips_code"] = metadata["state_fips_code"].astype("int32")
    if "cbsa_type" not in metadata.columns:
        metadata["cbsa_type"] = "Metro"

    metadata["cbsa_type"] = metadata["cbsa_type"].fillna("Metro")
    metadata["cbsa_code"] = metadata["cbsa_code"].astype("int32")
    return metadata
