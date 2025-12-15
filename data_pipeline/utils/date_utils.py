"""Date and time helpers for data collection and processing."""

from datetime import datetime
import pandas as pd

from data_pipeline import pipeline_config as config


def next_month(date_value):
    """Return the first day of the month after date_value.

    Args:
        date_value (pd.Timestamp): Input date.

    Returns:
        pd.Timestamp: First day of next month.
    """
    return (date_value + pd.offsets.MonthBegin(1)).normalize()


def month_start(date_value):
    """Return the first day of the month for a given timestamp.

    Args:
        date_value (pd.Timestamp): Input date.

    Returns:
        pd.Timestamp: First day of the month.
    """
    return pd.Timestamp(year=date_value.year, month=date_value.month, day=1)


def format_date(date_value):
    """Format a timestamp using pipeline configuration.

    Args:
        date_value (pd.Timestamp): Date to format.

    Returns:
        str: Formatted date string.
    """
    return date_value.strftime(config.DATE_FORMAT)