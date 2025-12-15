"""
FRED API Client for Real Estate Data Collection

This module handles data collection from the Federal Reserve Economic Data (FRED) API
for real estate metrics across CBSA (Core Based Statistical Area) codes.

Features:
- Rate limiting (120 requests/minute)
- Robust error handling and retry logic
- Progress tracking and logging
- Data validation and cleaning
- Automatic file organization
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import json
from dotenv import load_dotenv

from data_pipeline import pipeline_config as config

load_dotenv()

class FREDAPIClient:
    """Client for fetching real estate data from FRED API."""

    def __init__(self, api_key=None):
        """Initialize FRED API client.

        Args:
            api_key (str | None): FRED API key. If None, loads from environment variable.

        Raises:
            ValueError: If FRED API key is not found.
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key not found. Set FRED_API_KEY environment variable.")
        
        self.base_url = "https://api.stlouisfed.org/fred"
        self.rate_limit_delay = 60 / 120  # 120 requests per minute
        self.last_request_time = 0
        
        # Setup logging
        self._setup_logging()
        
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_DIR / f'fred_api_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint, params):
        """Make API request with error handling and rate limiting.

        Args:
            endpoint (str): API endpoint.
            params (dict): Request parameters.

        Returns:
            dict | None: API response data or None if failed.
        """
        self._rate_limit()
        
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed for {endpoint}: {e}")
            return None
    
    def get_series_data(self, series_id, start_date=None):
        """Fetch time series data for a specific series.

        Args:
            series_id (str): FRED series ID.
            start_date (str | None): Start date in YYYY-MM-DD format (optional).

        Returns:
            pd.DataFrame: DataFrame with date and metric values.
        """
        params = {
            'series_id': series_id
        }
        
        # Only add start date if specified
        if start_date:
            params['observation_start'] = start_date
        
        data = self._make_request('series/observations', params)
        
        if not data or 'observations' not in data:
            self.logger.warning(f"No data found for series {series_id}")
            return None
        
        observations = data['observations']
        
        # Convert to DataFrame
        df = pd.DataFrame(observations)
        
        if df.empty:
            self.logger.warning(f"Empty dataset for series {series_id}")
            return None
        
        # Clean and process data
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Remove rows with missing values
        df = df.dropna(subset=['value'])
        
        # Add metadata
        df['series_id'] = series_id
        df['cbsa_code'] = series_id[-5:]  # Extract CBSA code from series ID
        
        return df[['date', 'value', 'series_id', 'cbsa_code']]
    