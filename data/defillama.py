"""
DeFiLlama API client for stablecoin issuance tracking.
Monitors USD liquidity entering/leaving crypto markets.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import time


class DeFiLlamaAPI:
    """DeFiLlama data aggregator for stablecoin supply."""

    BASE_URL = "https://stablecoins.llama.fi"

    def __init__(self):
        self._cache = {}
        self._cache_timeout = 1800  # 30 minutes (data updates slowly)

    def get_stablecoin_chart(self, stablecoin_id: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch circulating supply chart for a specific stablecoin or all stablecoins.

        Args:
            stablecoin_id: Specific stablecoin ID (None for aggregate)

        Returns:
            DataFrame with columns: [date, total_circulating_usd]
        """
        cache_key = f"defillama_chart_{stablecoin_id}"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            if stablecoin_id is None:
                # Aggregate all stablecoins - use stablecoincharts/all endpoint
                url = f"{self.BASE_URL}/stablecoincharts/all"
            else:
                # Specific stablecoin by ID
                url = f"{self.BASE_URL}/stablecoin/{stablecoin_id}"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse response structure
            if stablecoin_id is None:
                # Aggregate format - data is directly an array
                if not isinstance(data, list):
                    return pd.DataFrame(columns=["date", "total_circulating_usd"])
                records = data
            else:
                # Single stablecoin format
                records = data.get("data", []) if isinstance(data, dict) else []

            if not records:
                return pd.DataFrame(columns=["date", "total_circulating_usd"])

            df = pd.DataFrame(records)

            # Handle different response formats
            if "date" in df.columns and "totalCirculatingUSD" in df.columns:
                df["date"] = pd.to_datetime(pd.to_numeric(df["date"], errors="coerce"), unit="s")
                df["totalCirculatingUSD"] = pd.to_numeric(df["totalCirculatingUSD"], errors="coerce")

                result = df.rename(columns={"totalCirculatingUSD": "total_circulating_usd"})[
                    ["date", "total_circulating_usd"]
                ].dropna()

            elif "peggedUSD" in df.columns:
                # Alternative format
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(pd.to_numeric(df["date"], errors="coerce"), unit="s")
                df["peggedUSD"] = pd.to_numeric(df["peggedUSD"], errors="coerce")

                result = df.rename(columns={"peggedUSD": "total_circulating_usd"})[
                    ["date", "total_circulating_usd"]
                ].dropna()

            else:
                return pd.DataFrame(columns=["date", "total_circulating_usd"])

            # Cache
            self._cache[cache_key] = (result.copy(), time.time())
            return result

        except Exception as e:
            print(f"[warn] DeFiLlama chart API failed for stablecoin {stablecoin_id}: {e}")
            return pd.DataFrame(columns=["date", "total_circulating_usd"])

    def get_stablecoin_by_chain(self, chain: str) -> pd.DataFrame:
        """
        Fetch stablecoin supply by blockchain.

        Args:
            chain: Blockchain name (e.g., "Ethereum", "BSC", "Tron")

        Returns:
            DataFrame with circulating supply over time
        """
        cache_key = f"defillama_chain_{chain}"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            url = f"{self.BASE_URL}/stablecoincharts/{chain}"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data:
                return pd.DataFrame(columns=["date", "total_circulating_usd"])

            df = pd.DataFrame(data)

            if "date" in df.columns and "totalCirculatingUSD" in df.columns:
                df["date"] = pd.to_datetime(df["date"], unit="s")
                df["totalCirculatingUSD"] = pd.to_numeric(df["totalCirculatingUSD"], errors="coerce")

                result = df.rename(columns={"totalCirculatingUSD": "total_circulating_usd"})[
                    ["date", "total_circulating_usd"]
                ].dropna()

                # Cache
                self._cache[cache_key] = (result.copy(), time.time())
                return result

            return pd.DataFrame(columns=["date", "total_circulating_usd"])

        except Exception as e:
            print(f"[warn] DeFiLlama chain API failed for {chain}: {e}")
            return pd.DataFrame(columns=["date", "total_circulating_usd"])

    def get_all_stablecoins_info(self) -> List[Dict[str, Any]]:
        """
        Fetch metadata for all stablecoins.

        Returns:
            List of stablecoin info dicts with id, name, symbol
        """
        cache_key = "defillama_all_stablecoins"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < 3600:  # 1 hour cache
                return cached_data.copy()

        try:
            url = f"{self.BASE_URL}/stablecoins"
            params = {"includePrices": "true"}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            stablecoins = data.get("peggedAssets", [])

            # Cache
            self._cache[cache_key] = (stablecoins, time.time())
            return stablecoins

        except Exception as e:
            print(f"[warn] DeFiLlama all stablecoins API failed: {e}")
            return []

    def get_major_stablecoin_supplies(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch supply data for major stablecoins (USDT, USDC, DAI).

        Returns:
            Dict mapping symbol to DataFrame with supply history
        """
        # Get all stablecoins to find IDs
        all_stablecoins = self.get_all_stablecoins_info()

        major_coins = {
            "USDT": None,
            "USDC": None,
            "DAI": None
        }

        # Find IDs for majors
        for coin in all_stablecoins:
            symbol = coin.get("symbol", "").upper()
            if symbol in major_coins:
                major_coins[symbol] = coin.get("id")

        # Fetch charts for each
        result = {}
        for symbol, coin_id in major_coins.items():
            if coin_id is not None:
                df = self.get_stablecoin_chart(coin_id)
                if not df.empty:
                    result[symbol] = df

        return result

    def calculate_net_issuance(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Calculate net issuance (supply change) over specified period.

        Args:
            lookback_days: Number of days to analyze

        Returns:
            Dict with net issuance metrics for major stablecoins
        """
        # Get aggregate stablecoin supply
        aggregate_df = self.get_stablecoin_chart(stablecoin_id=None)

        if aggregate_df.empty:
            return {
                "net_issuance_7d_usd": None,
                "current_supply_usd": None,
                "supply_7d_ago_usd": None,
                "per_token": {}
            }

        # Sort by date
        aggregate_df = aggregate_df.sort_values("date").reset_index(drop=True)

        # Get current and lookback values
        current_supply = float(aggregate_df.iloc[-1]["total_circulating_usd"])
        current_date = aggregate_df.iloc[-1]["date"]

        # Find supply N days ago
        target_date = current_date - timedelta(days=lookback_days)
        closest_idx = (aggregate_df["date"] - target_date).abs().idxmin()
        supply_past = float(aggregate_df.iloc[closest_idx]["total_circulating_usd"])

        net_issuance = current_supply - supply_past

        # Get per-token breakdown
        per_token_data = {}
        major_supplies = self.get_major_stablecoin_supplies()

        for symbol, df in major_supplies.items():
            if df.empty or len(df) < 2:
                continue

            df = df.sort_values("date").reset_index(drop=True)
            current = float(df.iloc[-1]["total_circulating_usd"])

            # Find closest past date
            current_dt = df.iloc[-1]["date"]
            target_dt = current_dt - timedelta(days=lookback_days)
            past_idx = (df["date"] - target_dt).abs().idxmin()
            past = float(df.iloc[past_idx]["total_circulating_usd"])

            per_token_data[symbol] = {
                "current_supply": current,
                "supply_past": past,
                "net_issuance": current - past,
                "pct_change": ((current - past) / past * 100) if past > 0 else 0
            }

        return {
            "net_issuance_7d_usd": net_issuance,
            "net_issuance_pct": (net_issuance / supply_past * 100) if supply_past > 0 else 0,
            "current_supply_usd": current_supply,
            "supply_7d_ago_usd": supply_past,
            "per_token": per_token_data,
            "lookback_days": lookback_days
        }

    def get_stablecoin_dominance(self) -> Dict[str, float]:
        """
        Calculate market share of major stablecoins.

        Returns:
            Dict mapping symbol to market share percentage
        """
        all_stablecoins = self.get_all_stablecoins_info()

        # Get circulating amounts for majors
        total_supply = 0
        major_supplies = {}

        for coin in all_stablecoins:
            symbol = coin.get("symbol", "").upper()
            circulating = coin.get("circulating", {}).get("peggedUSD", 0)

            if circulating:
                total_supply += circulating

                if symbol in ["USDT", "USDC", "DAI", "BUSD", "USDD"]:
                    major_supplies[symbol] = circulating

        # Calculate percentages
        dominance = {}
        for symbol, supply in major_supplies.items():
            if total_supply > 0:
                dominance[symbol] = (supply / total_supply) * 100

        return dominance

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
