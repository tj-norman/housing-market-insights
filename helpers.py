"""
Helper functions for the dashboard application.
"""

import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATHS = {
    'data': PROJECT_ROOT / 'processed_data',
    'geometries': PROJECT_ROOT / 'geometries',
    'app_data': PROJECT_ROOT / 'app_data',
    'agg_data': PROJECT_ROOT / 'app_data' / 'aggregate_data',
    'cbsa_data': PROJECT_ROOT / 'app_data' / 'cbsa_data',
    'metadata': PROJECT_ROOT / 'app_data' / 'metadata',
}

FILE_PATHS = {
    'geometries': DATA_PATHS['geometries'] / 'parquet' / 'cbsa_geometries.parquet',
    'state_boundaries': DATA_PATHS['geometries'] / 'state_geometries' / 'cb_2023_us_state_20m.json',
    'state_fips_codes': DATA_PATHS['metadata'] / 'state_fips_codes.csv',
    'metadata': DATA_PATHS['metadata'] / 'cbsa_metadata.parquet',
    'agg_data': DATA_PATHS['agg_data'] / 'aggregate_metrics_all.parquet',
    'agg_data_metro': DATA_PATHS['agg_data'] / 'aggregate_metrics_metro.parquet',
}


def get_data_path(key):
    """Fetches a pre-defined data directory path.

    Args:
        key (str): The key for the data path.

    Returns:
        Path: The requested directory path.

    Raises:
        KeyError: If key is not valid.
    """
    try:
        return DATA_PATHS[key]
    except KeyError:
        raise KeyError(
            f"Invalid data path key: '{key}'. "
            f"Available: {list(DATA_PATHS.keys())}"
        )


def get_file_path(key):
    """Fetches a pre-defined file path.

    Args:
        key (str): The key for the file path.

    Returns:
        Path: The requested file path.

    Raises:
        KeyError: If key is not valid.
    """
    try:
        return FILE_PATHS[key]
    except KeyError:
        raise KeyError(
            f"Invalid file key: '{key}'. "
            f"Available: {list(FILE_PATHS.keys())}"
        )


def human_format(num):
    """Format large numbers with K/M/G/T/P suffixes.
    
    Args:
        num (float | int): The number to format.
        
    Returns:
        str: Formatted string (e.g. "1.5M") or "N/A" if null.
    
    Examples:
        human_format(15500) returns "15.5K"
        human_format(1200000) returns "1.2M"
    """
    if pd.isnull(num):
        return "N/A"
    if abs(num) < 1000:
        return str(int(num))

    sign = "-" if num < 0 else ""
    num = abs(num)

    magnitude = int(np.log10(num) // 3)
    mantissa = num / (1000 ** magnitude)
    suffix = ["", "K", "M", "G", "T", "P"][magnitude]

    if mantissa % 1 == 0:
        return f"{sign}{int(mantissa)}{suffix}"
    else:
        return f"{sign}{mantissa:.1f}{suffix}"


def get_primary_state(state_string):
    """Extracts the primary state from a multi-state CBSA string.
    
    Args:
        state_string (str): State string like 'PA-NJ-DE-MD' or 'CA'
        
    Returns:
        str: The first state code (e.g., 'PA' from 'PA-NJ-DE-MD')
        
        Special case: For Washington DC, returns 'VA' instead.
    """
    if pd.isnull(state_string):
        return ''
    primary_state = state_string.split('-')[0].strip()
    # Special case for Washington DC - use VA instead
    if primary_state == 'DC':
        return 'VA'
    return primary_state


def get_shortened_cbsa_name(cbsa_title, cbsa_type):
    """Creates a shortened CBSA name for legend display.
    
    Extracts first city and first state, then adds cbsa_type.
    
    Args:
        cbsa_title (str): Full CBSA title (e.g., 'Augusta-Richmond County, GA-SC')
        cbsa_type (str): CBSA type ('Metro' or 'Micro')
    
    Returns:
        str: Shortened name (e.g., 'Augusta, GA Metro')
    
    Examples:
        'Augusta-Richmond County, GA-SC' -> 'Augusta, GA Metro'
        'Wisconsin Rapids-Marshfield, WI' -> 'Wisconsin Rapids, WI Micro'
    """
    # Split by comma to separate cities from states
    parts = cbsa_title.split(', ')
    if len(parts) < 2:
        return cbsa_title
    
    # Get first city (in multi-city CBSAs)
    cities_part = parts[0]
    first_city = cities_part.split('-')[0].strip()
    
    # Get first state (in multi-state CBSAs)
    states_part = parts[1]
    first_state = states_part.split('-')[0].strip()
    
    # Only append cbsa_type if it's provided and not empty
    if cbsa_type:
        return f"{first_city}, {first_state} {cbsa_type}"
    else:
        return f"{first_city}, {first_state}"


def get_color_indices(n_cbsas):
    """Calculate evenly spaced color indices for maximum contrast.
    
    This function distributes color indices across the MONO_COLORS array
    to maximize visual contrast between chart lines.
    
    Args:
        n_cbsas (int): Number of CBSAs to display (typically 3 or 5)
        
    Returns:
        list[int]: List of color indices to use from MONO_COLORS array
        
    Examples:
        get_color_indices(3) -> [0, 4, 8]
        get_color_indices(5) -> [0, 2, 4, 6, 8]
    """
    if n_cbsas == 3:
        return [0, 4, 8]
    elif n_cbsas == 5:
        return [0, 2, 4, 6, 8]
    else:
        return [0, 2, 4, 6, 8]


def format_metric_vectorized(series, decimals=1, suffix='%'):
    """Vectorized metric formatting for better performance.
    
    Formats a pandas Series of numeric metrics for display, handling non-numeric values.
    
    Args:
        series (pd.Series): The pandas Series of numeric values to format
        decimals (int): The number of decimal places to format to. Defaults to 1
        suffix (str): The string to append to the end. Defaults to '%'
        
    Returns:
        np.ndarray: Array of formatted strings (e.g., "+1.5%") or "N/A" for invalid input
    """
    # Create result array filled with N/A
    result = np.full(len(series), "N/A", dtype=object)
    
    # Mask for valid finite values
    valid_mask = pd.notna(series) & np.isfinite(series)
    valid_values = series[valid_mask].values
    
    if len(valid_values) == 0:
        return result
    
    # Format zero values without sign
    zero_mask = valid_mask & (series == 0)
    if zero_mask.any():
        result[zero_mask] = [f"{0:.{decimals}f}{suffix}" for _ in range(zero_mask.sum())]
    
    # Format non-zero values with sign
    nonzero_mask = valid_mask & (series != 0)
    if nonzero_mask.any():
        nonzero_values = series[nonzero_mask].values
        result[nonzero_mask] = [f"{v:+.{decimals}f}{suffix}" for v in nonzero_values]
    
    return result


def get_color_vectorized(series, dataset, reversed_datasets):
    """Vectorized version of get_color for better performance.
    
    Determines display colors for metrics based on their values.
    Standard logic is green for positive and red for negative. This is inverted
    if the dataset is in reversed_datasets.
    
    Args:
        series (pd.Series): The pandas Series of numeric values
        dataset (str): The name of the dataset this value belongs to
        reversed_datasets (set[str] | list[str]): A set of dataset names where color logic should be flipped
        
    Returns:
        np.ndarray: Array of color names ('green', 'red', 'black', or 'gray')
    """
    
    result = np.full(len(series), 'gray', dtype=object)
    
    valid_mask = pd.notna(series) & np.isfinite(series)
    
    if not valid_mask.any():
        return result
    
    zero_mask = valid_mask & (series == 0)
    result[zero_mask] = 'black'
    
    # Handle positive/negative values
    is_reversed = dataset in reversed_datasets
    positive_mask = valid_mask & (series > 0)
    negative_mask = valid_mask & (series < 0)
    
    if is_reversed:
        result[positive_mask] = 'red'
        result[negative_mask] = 'green'
    else:
        result[positive_mask] = 'green'
        result[negative_mask] = 'red'
    
    return result


def filter_cbsa_dataframe(df, state_selector, selected_states=None, metros_only=False):
    """Filters CBSA dataframe by state and metro criteria.
    
    Args:
        df (pd.DataFrame): DataFrame to filter (must have 'state' and optionally 'cbsa_type' columns)
        state_selector (str): "all", "active" (CONUS), or "custom"
        selected_states (list[str] | None): List of state codes for custom mode
        metros_only (bool): If True, filter to Metro CBSAs only
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered = df.copy()
    
    if state_selector == "custom" and selected_states:
        filtered = filtered[filtered['state'].isin(selected_states)]
    elif state_selector == "active":
        filtered = filtered[~filtered['state'].isin(['HI', 'AK'])]
    
    if metros_only and 'cbsa_type' in filtered.columns:
        filtered = filtered[filtered['cbsa_type'] == 'Metro']
    
    return filtered