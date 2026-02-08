"""
Deribit API client for crypto options data.
Calculates 25-delta risk reversals and option skew metrics.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import time


class DeribitAPI:
    """Deribit options data aggregator."""

    BASE_URL = "https://www.deribit.com/api/v2/public"

    def __init__(self):
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes

    def get_instruments(
        self,
        currency: str = "BTC",
        kind: str = "option",
        expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all options instruments for a currency.

        Args:
            currency: Currency (BTC or ETH)
            kind: Instrument type (option, future)
            expired: Include expired options

        Returns:
            List of instrument details
        """
        cache_key = f"deribit_instruments_{currency}_{kind}_{expired}"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            url = f"{self.BASE_URL}/get_instruments"
            params = {
                "currency": currency,
                "kind": kind,
                "expired": str(expired).lower()
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            if data.get("result"):
                instruments = data["result"]
                # Cache
                self._cache[cache_key] = (instruments, time.time())
                return instruments

            return []

        except Exception as e:
            print(f"[warn] Deribit instruments API failed for {currency}: {e}")
            return []

    def get_ticker(self, instrument_name: str) -> Dict[str, Any]:
        """
        Get ticker data including IV and greeks for an option.

        Args:
            instrument_name: Full instrument name (e.g., "BTC-31MAR23-25000-C")

        Returns:
            Dict with price, IV, delta, and other metrics
        """
        cache_key = f"deribit_ticker_{instrument_name}"

        # Check cache (short TTL for real-time data)
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < 60:  # 1 min cache
                return cached_data.copy()

        try:
            url = f"{self.BASE_URL}/ticker"
            params = {"instrument_name": instrument_name}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get("result"):
                ticker = data["result"]
                # Cache
                self._cache[cache_key] = (ticker, time.time())
                return ticker

            return {}

        except Exception as e:
            print(f"[warn] Deribit ticker API failed for {instrument_name}: {e}")
            return {}

    def find_nearest_expiry(
        self,
        currency: str = "BTC",
        target_days: int = 30,
        tolerance_days: int = 7
    ) -> Optional[str]:
        """
        Find the option expiry nearest to target days.

        Args:
            currency: Currency (BTC or ETH)
            target_days: Target days to expiry
            tolerance_days: Maximum deviation from target

        Returns:
            Expiry date string (DDMMMYY format) or None
        """
        instruments = self.get_instruments(currency=currency, kind="option", expired=False)

        if not instruments:
            return None

        # Extract unique expiries
        expiries = set()
        for inst in instruments:
            name = inst.get("instrument_name", "")
            # Format: BTC-31MAR23-25000-C
            parts = name.split("-")
            if len(parts) >= 2:
                expiries.add(parts[1])

        # Parse and find nearest
        target_date = datetime.now() + timedelta(days=target_days)
        best_expiry = None
        min_diff = float("inf")

        for expiry_str in expiries:
            try:
                # Parse DDMMMYY format
                expiry_date = datetime.strptime(expiry_str, "%d%b%y")
                diff_days = abs((expiry_date - target_date).days)

                if diff_days < min_diff and diff_days <= tolerance_days:
                    min_diff = diff_days
                    best_expiry = expiry_str

            except ValueError:
                continue

        return best_expiry

    def find_option_by_delta(
        self,
        currency: str = "BTC",
        expiry: str = None,
        target_delta: float = 0.25,
        option_type: str = "call",
        tolerance: float = 0.05
    ) -> Optional[Dict[str, Any]]:
        """
        Find option with delta closest to target.

        Args:
            currency: Currency (BTC or ETH)
            expiry: Expiry date string (DDMMMYY)
            target_delta: Target delta value
            option_type: "call" or "put"
            tolerance: Maximum delta deviation

        Returns:
            Dict with instrument name, delta, IV, and other data
        """
        if expiry is None:
            return None

        instruments = self.get_instruments(currency=currency, kind="option", expired=False)

        # Filter by expiry and type
        option_suffix = "C" if option_type.lower() == "call" else "P"
        candidates = [
            inst for inst in instruments
            if expiry in inst.get("instrument_name", "") and
            inst.get("instrument_name", "").endswith(option_suffix)
        ]

        if not candidates:
            return None

        # Find closest delta
        best_option = None
        min_delta_diff = float("inf")

        for inst in candidates:
            instrument_name = inst.get("instrument_name")
            ticker = self.get_ticker(instrument_name)

            if not ticker:
                continue

            greeks = ticker.get("greeks", {})
            delta = greeks.get("delta")

            if delta is None:
                continue

            delta_diff = abs(delta - target_delta)

            if delta_diff < min_delta_diff and delta_diff <= tolerance:
                min_delta_diff = delta_diff
                mark_iv = ticker.get("mark_iv")

                best_option = {
                    "instrument_name": instrument_name,
                    "delta": delta,
                    "mark_iv": mark_iv,
                    "bid_iv": ticker.get("bid_iv"),
                    "ask_iv": ticker.get("ask_iv"),
                    "underlying_price": ticker.get("underlying_price"),
                    "mark_price": ticker.get("mark_price")
                }

        return best_option

    def calculate_25d_risk_reversal(
        self,
        currency: str = "BTC",
        target_days: int = 30,
        tolerance_days: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate 25-delta risk reversal (call IV - put IV).

        Args:
            currency: Currency (BTC or ETH)
            target_days: Target days to expiry
            tolerance_days: Maximum deviation from target expiry

        Returns:
            Dict with RR25, IVs, deltas, and metadata
        """
        # Find nearest expiry
        expiry = self.find_nearest_expiry(currency, target_days, tolerance_days)

        if expiry is None:
            return {
                "rr25": None,
                "call_iv": None,
                "put_iv": None,
                "call_delta": None,
                "put_delta": None,
                "expiry": None,
                "days_to_expiry": None,
                "error": "No suitable expiry found"
            }

        # Find 25-delta call
        call_option = self.find_option_by_delta(
            currency=currency,
            expiry=expiry,
            target_delta=0.25,
            option_type="call",
            tolerance=0.05
        )

        # Find 25-delta put
        put_option = self.find_option_by_delta(
            currency=currency,
            expiry=expiry,
            target_delta=-0.25,
            option_type="put",
            tolerance=0.05
        )

        if not call_option or not put_option:
            return {
                "rr25": None,
                "call_iv": None,
                "put_iv": None,
                "call_delta": None,
                "put_delta": None,
                "expiry": expiry,
                "days_to_expiry": None,
                "error": "Could not find suitable 25-delta options"
            }

        # Calculate RR25
        call_iv = call_option.get("mark_iv")
        put_iv = put_option.get("mark_iv")

        if call_iv is None or put_iv is None:
            return {
                "rr25": None,
                "call_iv": call_iv,
                "put_iv": put_iv,
                "call_delta": call_option.get("delta"),
                "put_delta": put_option.get("delta"),
                "expiry": expiry,
                "days_to_expiry": None,
                "error": "Missing IV data"
            }

        rr25 = call_iv - put_iv

        # Calculate days to expiry
        try:
            expiry_date = datetime.strptime(expiry, "%d%b%y")
            days_to_expiry = (expiry_date - datetime.now()).days
        except ValueError:
            days_to_expiry = None

        return {
            "rr25": rr25,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "call_delta": call_option.get("delta"),
            "put_delta": put_option.get("delta"),
            "call_instrument": call_option.get("instrument_name"),
            "put_instrument": put_option.get("instrument_name"),
            "underlying_price": call_option.get("underlying_price"),
            "expiry": expiry,
            "days_to_expiry": days_to_expiry
        }

    def get_historical_rr25(
        self,
        currency: str = "BTC",
        target_days: int = 30,
        lookback_days: int = 365
    ) -> pd.DataFrame:
        """
        Calculate historical 25-delta risk reversal for z-score analysis.

        Note: This requires historical option data which may not be available
        via public API. This is a placeholder for future implementation.

        Args:
            currency: Currency (BTC or ETH)
            target_days: Target days to expiry
            lookback_days: Days of history

        Returns:
            DataFrame with columns: [date, rr25]
        """
        # Public API doesn't provide historical options data
        # Would need premium API access or alternative data source
        print("[warn] Historical RR25 requires premium Deribit API access")
        return pd.DataFrame(columns=["date", "rr25"])

    def calculate_rr25_zscore(
        self,
        currency: str = "BTC",
        target_days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate z-score for current RR25 (requires historical data).

        Args:
            currency: Currency (BTC or ETH)
            target_days: Target days to expiry

        Returns:
            Dict with RR25 z-score and statistics
        """
        # Get current RR25
        current_rr = self.calculate_25d_risk_reversal(currency, target_days)

        if current_rr.get("rr25") is None:
            return {
                "z_score": None,
                "current_rr25": None,
                "mean_rr25": None,
                "std_rr25": None,
                "error": "Cannot calculate RR25"
            }

        # Historical data not available via free API
        # Return current value only
        return {
            "z_score": None,
            "current_rr25": current_rr["rr25"],
            "mean_rr25": None,
            "std_rr25": None,
            "error": "Historical data not available for z-score",
            "current_data": current_rr
        }

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
