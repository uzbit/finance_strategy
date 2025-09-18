"""
FRED (Federal Reserve Economic Data) API wrapper.
Handles fetching economic data series with caching and error handling.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import time
import os


class FredAPI:
    """FRED API client with caching and error handling."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Cache for recently fetched data
    _cache = {}
    _cache_timeout = 3600  # 1 hour cache timeout

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            print("[warn] No FRED API key provided. Some features may not work.")

    def fetch_series(self, series_id: str, start_date: str,
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch a FRED data series.

        Args:
            series_id: FRED series identifier
            start_date: Start date in YYYY-MM-DD format
            use_cache: Whether to use cached data

        Returns:
            DataFrame with datetime index and 'value' column
        """
        cache_key = f"{series_id}_{start_date}"

        # Check cache first
        if use_cache and cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            df = self._fetch_from_api(series_id, start_date)

            # Cache the result
            if use_cache:
                self._cache[cache_key] = (df.copy(), time.time())

            return df

        except Exception as e:
            print(f"[error] Failed to fetch {series_id}: {e}")
            return pd.DataFrame(columns=["value"], index=pd.to_datetime([]))

    def _fetch_from_api(self, series_id: str, start_date: str) -> pd.DataFrame:
        """Fetch data directly from FRED API."""
        params = {
            "series_id": series_id,
            "observation_start": start_date,
            "file_type": "json",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json().get("observations", [])
        if not data:
            return pd.DataFrame(columns=["value"], index=pd.to_datetime([]))

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])

        # Convert values to float, handling missing data
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        # Set datetime index and sort
        df = df.set_index("date")[["value"]].sort_index()

        return df

    def get_latest_value(self, series_id: str, start_date: str) -> Optional[float]:
        """Get the most recent value for a series."""
        try:
            df = self.fetch_series(series_id, start_date)
            if df.empty:
                return None
            return float(df["value"].iloc[-1])
        except Exception:
            return None

    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """Get metadata information about a series."""
        if not self.api_key:
            return {}

        try:
            url = "https://api.stlouisfed.org/fred/series"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json"
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if "seriess" in data and data["seriess"]:
                return data["seriess"][0]

        except Exception as e:
            print(f"[error] Failed to get series info for {series_id}: {e}")

        return {}

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()

    @classmethod
    def get_supported_series(cls) -> Dict[str, str]:
        """Get dictionary of supported FRED series and their descriptions."""
        return {
            "T10Y3M": "10Y minus 3M Treasury spread",
            "UNRATE": "Unemployment rate",
            "ICSA": "Initial jobless claims (weekly)",
            "NFCI": "Chicago Fed financial conditions",
            "BAMLH0A0HYM2": "High-yield corporate spread",
            "PERMIT": "Housing permits (SAAR)",
            "VIXCLS": "VIX Volatility Index",
            "UMCSENT": "University of Michigan Consumer Sentiment",
            "DFII10": "10-Year Treasury Inflation-Indexed (Real Rate)",
            "T10YIE": "10-Year Breakeven Inflation Rate",
            "USALOLITOAASTSAM": "OECD Leading Economic Indicators",
            "DCOILWTICO": "WTI Crude Oil Prices",
            "DGS10": "10-Year Treasury Constant Maturity Rate",
            "DGS2": "2-Year Treasury Constant Maturity Rate",
            "DGS1MO": "1-Month Treasury Constant Maturity Rate"
        }


# Note: This module provides the FredAPI class.
# Instantiation should be done by the caller with proper API key.