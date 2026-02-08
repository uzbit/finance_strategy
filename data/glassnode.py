"""
Glassnode API client for institutional crypto flows.
Supports US Spot ETF data for BTC and ETH.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import time


class GlassnodeAPI:
    """Glassnode data aggregator for US spot ETF flows."""

    BASE_URL = "https://api.glassnode.com/v1/metrics"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self._cache = {}
        self._cache_timeout = 900  # 15 minutes (Glassnode updates daily)
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Rate limiting: 1 req/sec

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a rate-limited request to Glassnode API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data
        """
        if not self.api_key:
            raise ValueError("[error] Glassnode API key required")

        self._rate_limit()

        params["api_key"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print("[warn] Glassnode API rate limit exceeded")
            elif response.status_code == 403:
                print("[warn] Glassnode API key invalid or insufficient permissions")
            else:
                print(f"[warn] Glassnode API HTTP error {response.status_code}: {e}")
            return []

        except Exception as e:
            print(f"[warn] Glassnode API request failed: {e}")
            return []

    def get_us_spot_etf_flows_net(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch daily net flows for US spot ETFs.

        Args:
            asset: Asset symbol (BTC or ETH)
            since: Start date (default: 90 days ago)
            until: End date (default: today)

        Returns:
            DataFrame with columns: [timestamp, net_flows_usd, net_flows_native]
        """
        cache_key = f"glassnode_etf_flows_{asset}"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        # Default date range
        if since is None:
            since = datetime.now() - timedelta(days=90)
        if until is None:
            until = datetime.now()

        params = {
            "a": asset,
            "s": int(since.timestamp()),
            "u": int(until.timestamp()),
            "f": "JSON",
            "timestamp_format": "humanized"
        }

        try:
            data = self._make_request("institutions/us_spot_etf_flows_net", params)

            if not data:
                return pd.DataFrame(columns=["timestamp", "net_flows_usd"])

            df = pd.DataFrame(data)
            df["t"] = pd.to_datetime(df["t"])
            df["v"] = pd.to_numeric(df["v"], errors="coerce")

            result = df.rename(columns={"t": "timestamp", "v": "net_flows_native"})[
                ["timestamp", "net_flows_native"]
            ].dropna()

            # Cache
            self._cache[cache_key] = (result.copy(), time.time())
            return result

        except Exception as e:
            print(f"[warn] Glassnode ETF flows failed for {asset}: {e}")
            return pd.DataFrame(columns=["timestamp", "net_flows_native"])

    def get_us_spot_etf_balances_latest(self, asset: str = "BTC") -> Dict[str, Any]:
        """
        Fetch latest US spot ETF balances snapshot.

        Args:
            asset: Asset symbol (BTC or ETH)

        Returns:
            Dict with latest balances and changes
        """
        cache_key = f"glassnode_etf_balances_{asset}"

        # Check cache (shorter TTL for current data)
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < 300:  # 5 min cache
                return cached_data.copy()

        params = {
            "a": asset,
            "f": "JSON",
            "timestamp_format": "humanized"
        }

        try:
            data = self._make_request("institutions/us_spot_etf_balances_latest", params)

            if not data or not isinstance(data, list) or len(data) == 0:
                return {"timestamp": None, "total_balance": None}

            # Latest record
            latest = data[-1] if isinstance(data, list) else data

            result = {
                "timestamp": pd.to_datetime(latest.get("t")) if latest.get("t") else None,
                "total_balance": float(latest.get("v", 0)) if latest.get("v") else None
            }

            # Cache
            self._cache[cache_key] = (result.copy(), time.time())
            return result

        except Exception as e:
            print(f"[warn] Glassnode ETF balances failed for {asset}: {e}")
            return {"timestamp": None, "total_balance": None}

    def get_us_spot_etf_flows_all(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch all US spot ETF flows (inflows + outflows separated).

        Args:
            asset: Asset symbol (BTC or ETH)
            since: Start date (default: 90 days ago)
            until: End date (default: today)

        Returns:
            DataFrame with columns: [timestamp, inflows, outflows, net]
        """
        cache_key = f"glassnode_etf_flows_all_{asset}"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        # Default date range
        if since is None:
            since = datetime.now() - timedelta(days=90)
        if until is None:
            until = datetime.now()

        params = {
            "a": asset,
            "s": int(since.timestamp()),
            "u": int(until.timestamp()),
            "f": "JSON",
            "timestamp_format": "humanized"
        }

        try:
            data = self._make_request("institutions/us_spot_etf_flows_all", params)

            if not data:
                return pd.DataFrame(columns=["timestamp", "inflows", "outflows", "net"])

            df = pd.DataFrame(data)
            df["t"] = pd.to_datetime(df["t"])

            # Value might be dict or single value, handle both
            if isinstance(df["v"].iloc[0], dict):
                df["inflows"] = df["v"].apply(lambda x: x.get("inflows", 0))
                df["outflows"] = df["v"].apply(lambda x: x.get("outflows", 0))
                df["net"] = df["v"].apply(lambda x: x.get("net", 0))
            else:
                df["net"] = pd.to_numeric(df["v"], errors="coerce")
                df["inflows"] = 0
                df["outflows"] = 0

            result = df.rename(columns={"t": "timestamp"})[
                ["timestamp", "inflows", "outflows", "net"]
            ].fillna(0)

            # Cache
            self._cache[cache_key] = (result.copy(), time.time())
            return result

        except Exception as e:
            print(f"[warn] Glassnode ETF flows (all) failed for {asset}: {e}")
            return pd.DataFrame(columns=["timestamp", "inflows", "outflows", "net"])

    def calculate_flow_metrics(self, asset: str = "BTC", lookback_days: int = 90) -> Dict[str, Any]:
        """
        Calculate ETF flow metrics and z-scores.

        Args:
            asset: Asset symbol (BTC or ETH)
            lookback_days: Days of history to analyze

        Returns:
            Dict with flow sums, z-scores, and latest values
        """
        since = datetime.now() - timedelta(days=lookback_days)
        flows_df = self.get_us_spot_etf_flows_net(asset, since=since)

        if flows_df.empty or len(flows_df) < 3:
            return {
                "latest_daily_flow": None,
                "flows_3d_sum": None,
                "flows_5d_sum": None,
                "flows_7d_sum": None,
                "z_90d": None,
                "mean_90d": None,
                "std_90d": None
            }

        flows = flows_df.set_index("timestamp")["net_flows_native"]

        # Latest values
        latest_daily_flow = float(flows.iloc[-1]) if len(flows) >= 1 else None

        # Rolling sums
        flows_3d_sum = float(flows.iloc[-3:].sum()) if len(flows) >= 3 else None
        flows_5d_sum = float(flows.iloc[-5:].sum()) if len(flows) >= 5 else None
        flows_7d_sum = float(flows.iloc[-7:].sum()) if len(flows) >= 7 else None

        # Z-score over full period
        mean_90d = float(flows.mean())
        std_90d = float(flows.std())

        if std_90d > 0 and latest_daily_flow is not None:
            z_90d = (latest_daily_flow - mean_90d) / std_90d
        else:
            z_90d = None

        return {
            "latest_daily_flow": latest_daily_flow,
            "flows_3d_sum": flows_3d_sum,
            "flows_5d_sum": flows_5d_sum,
            "flows_7d_sum": flows_7d_sum,
            "z_90d": z_90d,
            "mean_90d": mean_90d,
            "std_90d": std_90d
        }

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
