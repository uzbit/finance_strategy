"""
Derivatives data APIs for crypto futures/perps markets.
Supports Binance, Bybit, and OKX for funding rates and open interest.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import time


class DerivativesAPI:
    """Cryptocurrency derivatives data aggregator for funding & OI."""

    def __init__(self):
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes

    # ========== Binance APIs ==========

    def get_binance_funding_history(self, symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
        """
        Fetch funding rate history from Binance USDT-M Futures.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Number of records (max 1000)

        Returns:
            DataFrame with columns: [timestamp, funding_rate]
        """
        cache_key = f"binance_funding_{symbol}_{limit}"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {"symbol": symbol, "limit": limit}
            headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceStrategyBot/1.0)"}

            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()

            data = response.json()
            if not data:
                return pd.DataFrame(columns=["timestamp", "funding_rate"])

            df = pd.DataFrame(data)
            # Fix pandas FutureWarning by explicitly converting to numeric first
            df["fundingTime"] = pd.to_datetime(pd.to_numeric(df["fundingTime"], errors="coerce"), unit="ms")
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

            result = df.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate"})[
                ["timestamp", "funding_rate"]
            ].dropna()

            # Cache
            self._cache[cache_key] = (result.copy(), time.time())
            return result

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 451:
                print(f"[warn] Binance API unavailable (geo-restricted) for {symbol}")
            else:
                print(f"[warn] Binance funding API failed for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])
        except Exception as e:
            print(f"[warn] Binance funding API failed for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

    def get_binance_open_interest(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """
        Fetch current open interest from Binance.

        Args:
            symbol: Trading pair

        Returns:
            Current OI in base currency (e.g., BTC)
        """
        cache_key = f"binance_oi_{symbol}"

        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < 60:  # 1 min cache for current OI
                return cached_data

        try:
            url = "https://fapi.binance.com/fapi/v1/openInterest"
            params = {"symbol": symbol}
            headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceStrategyBot/1.0)"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            oi = float(data.get("openInterest", 0))

            self._cache[cache_key] = (oi, time.time())
            return oi

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 451:
                print(f"[warn] Binance API unavailable (geo-restricted)")
            else:
                print(f"[warn] Binance OI API failed for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"[warn] Binance OI API failed for {symbol}: {e}")
            return None

    def get_binance_oi_history(self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 500) -> pd.DataFrame:
        """
        Fetch historical open interest from Binance.

        Args:
            symbol: Trading pair
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of records (max 500)

        Returns:
            DataFrame with columns: [timestamp, open_interest, sum_open_interest_value]
        """
        cache_key = f"binance_oi_hist_{symbol}_{period}_{limit}"

        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            params = {"symbol": symbol, "period": period, "limit": limit}
            headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceStrategyBot/1.0)"}

            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()

            data = response.json()
            if not data:
                return pd.DataFrame(columns=["timestamp", "open_interest"])

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms")
            df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")

            result = df.rename(columns={"sumOpenInterest": "open_interest"})[
                ["timestamp", "open_interest"]
            ].dropna()

            self._cache[cache_key] = (result.copy(), time.time())
            return result

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 451:
                print(f"[warn] Binance API unavailable (geo-restricted)")
            else:
                print(f"[warn] Binance OI history API failed for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "open_interest"])
        except Exception as e:
            print(f"[warn] Binance OI history API failed for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "open_interest"])

    # ========== Bybit APIs ==========

    def get_bybit_funding_history(self, symbol: str = "BTCUSDT", limit: int = 200) -> pd.DataFrame:
        """
        Fetch funding rate history from Bybit.

        Args:
            symbol: Trading pair
            limit: Number of records (max 200)

        Returns:
            DataFrame with columns: [timestamp, funding_rate]
        """
        cache_key = f"bybit_funding_{symbol}_{limit}"

        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            url = "https://api.bybit.com/v5/market/funding/history"
            params = {"category": "linear", "symbol": symbol, "limit": limit}
            headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceStrategyBot/1.0)"}

            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()

            data = response.json()
            if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
                return pd.DataFrame(columns=["timestamp", "funding_rate"])

            records = data["result"]["list"]
            df = pd.DataFrame(records)
            df["fundingRateTimestamp"] = pd.to_datetime(pd.to_numeric(df["fundingRateTimestamp"], errors="coerce"), unit="ms")
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

            result = df.rename(columns={"fundingRateTimestamp": "timestamp", "fundingRate": "funding_rate"})[
                ["timestamp", "funding_rate"]
            ].dropna()

            self._cache[cache_key] = (result.copy(), time.time())
            return result

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"[warn] Bybit API access forbidden (check region/headers)")
            else:
                print(f"[warn] Bybit funding API failed for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])
        except Exception as e:
            print(f"[warn] Bybit funding API failed for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

    def get_bybit_open_interest(self, symbol: str = "BTCUSDT", interval_time: str = "1h", limit: int = 200) -> pd.DataFrame:
        """
        Fetch open interest timeseries from Bybit.

        Args:
            symbol: Trading pair
            interval_time: Time interval (5min, 15min, 30min, 1h, 4h, 1d)
            limit: Number of records (max 200)

        Returns:
            DataFrame with columns: [timestamp, open_interest]
        """
        cache_key = f"bybit_oi_{symbol}_{interval_time}_{limit}"

        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            url = "https://api.bybit.com/v5/market/open-interest"
            params = {"category": "linear", "symbol": symbol, "intervalTime": interval_time, "limit": limit}
            headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceStrategyBot/1.0)"}

            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()

            data = response.json()
            if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
                return pd.DataFrame(columns=["timestamp", "open_interest"])

            records = data["result"]["list"]
            df = pd.DataFrame(records)
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms")
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")

            result = df.rename(columns={"openInterest": "open_interest"})[
                ["timestamp", "open_interest"]
            ].dropna()

            self._cache[cache_key] = (result.copy(), time.time())
            return result

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"[warn] Bybit API access forbidden (check region/headers)")
            else:
                print(f"[warn] Bybit OI API failed for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "open_interest"])
        except Exception as e:
            print(f"[warn] Bybit OI API failed for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "open_interest"])

    # ========== OKX APIs ==========

    def get_okx_funding_history(self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> pd.DataFrame:
        """
        Fetch funding rate history from OKX.

        Args:
            inst_id: Instrument ID (e.g., "BTC-USDT-SWAP")
            limit: Number of records (max 100)

        Returns:
            DataFrame with columns: [timestamp, funding_rate]
        """
        cache_key = f"okx_funding_{inst_id}_{limit}"

        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            url = "https://www.okx.com/api/v5/public/funding-rate-history"
            params = {"instId": inst_id, "limit": limit}

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            if data.get("code") != "0" or not data.get("data"):
                return pd.DataFrame(columns=["timestamp", "funding_rate"])

            records = data["data"]
            df = pd.DataFrame(records)
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

            result = df.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate"})[
                ["timestamp", "funding_rate"]
            ].dropna()

            self._cache[cache_key] = (result.copy(), time.time())
            return result

        except Exception as e:
            print(f"[warn] OKX funding API failed for {inst_id}: {e}")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

    # ========== Aggregation & Analysis ==========

    def get_aggregate_funding_data(self, symbol: str = "BTC", lookback_periods: int = 90) -> Dict[str, Any]:
        """
        Aggregate funding rate data across Binance, Bybit, and OKX.

        Args:
            symbol: Base symbol (BTC, ETH)
            lookback_periods: Number of funding periods to fetch

        Returns:
            Dict with per-exchange data and OI weights
        """
        # Normalize symbols for each exchange
        binance_symbol = f"{symbol}USDT"
        bybit_symbol = f"{symbol}USDT"
        okx_symbol = f"{symbol}-USDT-SWAP"

        # Fetch funding data
        binance_funding = self.get_binance_funding_history(binance_symbol, limit=lookback_periods)
        bybit_funding = self.get_bybit_funding_history(bybit_symbol, limit=lookback_periods)
        okx_funding = self.get_okx_funding_history(okx_symbol, limit=min(lookback_periods, 100))

        # Fetch current OI for weighting
        binance_oi = self.get_binance_open_interest(binance_symbol) or 0
        bybit_oi_df = self.get_bybit_open_interest(bybit_symbol, interval_time="1h", limit=1)
        bybit_oi = float(bybit_oi_df.iloc[-1]["open_interest"]) if not bybit_oi_df.empty else 0

        total_oi = binance_oi + bybit_oi
        binance_weight = binance_oi / total_oi if total_oi > 0 else 0.5
        bybit_weight = bybit_oi / total_oi if total_oi > 0 else 0.5

        return {
            "binance": {"funding": binance_funding, "oi": binance_oi, "weight": binance_weight},
            "bybit": {"funding": bybit_funding, "oi": bybit_oi, "weight": bybit_weight},
            "okx": {"funding": okx_funding, "oi": 0, "weight": 0},  # OKX OI API not included
            "total_oi": total_oi
        }

    def get_oi_change_24h(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Calculate 24h open interest change across exchanges.

        Args:
            symbol: Base symbol (BTC, ETH)

        Returns:
            Dict with OI changes and current values
        """
        binance_symbol = f"{symbol}USDT"
        bybit_symbol = f"{symbol}USDT"

        # Fetch 24h OI history
        binance_oi_hist = self.get_binance_oi_history(binance_symbol, period="1h", limit=25)
        bybit_oi_hist = self.get_bybit_open_interest(bybit_symbol, interval_time="1h", limit=25)

        result = {}

        # Binance calculation
        if len(binance_oi_hist) >= 24:
            oi_now = binance_oi_hist.iloc[-1]["open_interest"]
            oi_24h = binance_oi_hist.iloc[-24]["open_interest"]
            change_pct = ((oi_now - oi_24h) / oi_24h * 100) if oi_24h > 0 else 0
            result["binance"] = {"oi_now": oi_now, "oi_24h_ago": oi_24h, "change_pct": change_pct}
        else:
            result["binance"] = {"oi_now": None, "oi_24h_ago": None, "change_pct": None}

        # Bybit calculation
        if len(bybit_oi_hist) >= 24:
            oi_now = bybit_oi_hist.iloc[-1]["open_interest"]
            oi_24h = bybit_oi_hist.iloc[-24]["open_interest"]
            change_pct = ((oi_now - oi_24h) / oi_24h * 100) if oi_24h > 0 else 0
            result["bybit"] = {"oi_now": oi_now, "oi_24h_ago": oi_24h, "change_pct": change_pct}
        else:
            result["bybit"] = {"oi_now": None, "oi_24h_ago": None, "change_pct": None}

        # Aggregate
        binance_change = result["binance"].get("change_pct")
        bybit_change = result["bybit"].get("change_pct")

        if binance_change is not None and bybit_change is not None:
            result["aggregate_change_pct"] = (binance_change + bybit_change) / 2
        elif binance_change is not None:
            result["aggregate_change_pct"] = binance_change
        elif bybit_change is not None:
            result["aggregate_change_pct"] = bybit_change
        else:
            result["aggregate_change_pct"] = None

        return result

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
