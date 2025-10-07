"""
Beaconchain (Ethereum consensus layer) data API.
Provides validator queue and staking metrics from beaconcha.in.
"""

import requests
import time
from typing import Optional, Dict, Any
from datetime import datetime


class BeaconchainAPI:
    """Ethereum beacon chain data aggregator using beaconcha.in API."""

    def __init__(self):
        self.base_url = "https://beaconcha.in/api/v1"
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes (API updates every ~15 min)

    def get_validator_queue(self) -> Optional[Dict[str, Any]]:
        """
        Get current validator entry and exit queue statistics.

        Returns:
            Dictionary containing:
            - beaconchain_entering: Number of validators entering
            - beaconchain_exiting: Number of validators exiting
            - beaconchain_entering_balance: ETH in entering queue (Gwei)
            - beaconchain_exiting_balance: ETH in exiting queue (Gwei)
            - validatorscount: Total active validators
            - entry_eth: Entering balance in ETH
            - exit_eth: Exiting balance in ETH
            - entry_exit_ratio: Ratio of entering to exiting validators
            - entry_exit_eth_ratio: Ratio of entering to exiting ETH
        """
        cache_key = "validator_queue"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data

        try:
            url = f"{self.base_url}/validators/queue"
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            result = response.json()

            if result.get("status") != "OK":
                print(f"[warn] Beaconchain API returned non-OK status: {result.get('status')}")
                return None

            data = result.get("data", {})

            # Extract raw values
            entering_validators = data.get("beaconchain_entering", 0)
            exiting_validators = data.get("beaconchain_exiting", 0)
            entering_balance_gwei = data.get("beaconchain_entering_balance", 0)
            exiting_balance_gwei = data.get("beaconchain_exiting_balance", 0)
            total_validators = data.get("validatorscount", 0)

            # Convert Gwei to ETH (1 ETH = 1e9 Gwei)
            entry_eth = entering_balance_gwei / 1e9 if entering_balance_gwei else 0
            exit_eth = exiting_balance_gwei / 1e9 if exiting_balance_gwei else 0

            # Calculate ratios
            entry_exit_ratio = (
                entering_validators / exiting_validators
                if exiting_validators > 0
                else None
            )

            entry_exit_eth_ratio = (
                entry_eth / exit_eth
                if exit_eth > 0
                else None
            )

            # Estimate wait times based on churn rate
            # Post-Pectra: churn limit = min(MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT, total_validators // CHURN_LIMIT_QUOTIENT)
            # Simplified: assume ~256 validators per epoch (current rate), 225 epochs/day
            churn_per_epoch = 256
            epochs_per_day = 225

            entry_wait_days = (
                entering_validators / (churn_per_epoch * epochs_per_day)
                if entering_validators > 0
                else 0
            )

            exit_wait_days = (
                exiting_validators / (churn_per_epoch * epochs_per_day)
                if exiting_validators > 0
                else 0
            )

            wait_time_ratio = (
                entry_wait_days / exit_wait_days
                if exit_wait_days > 0
                else None
            )

            enriched_data = {
                # Raw values
                "beaconchain_entering": entering_validators,
                "beaconchain_exiting": exiting_validators,
                "beaconchain_entering_balance": entering_balance_gwei,
                "beaconchain_exiting_balance": exiting_balance_gwei,
                "validatorscount": total_validators,
                # Converted values
                "entry_eth": entry_eth,
                "exit_eth": exit_eth,
                # Ratios
                "entry_exit_ratio": entry_exit_ratio,
                "entry_exit_eth_ratio": entry_exit_eth_ratio,
                # Wait times (estimated)
                "entry_wait_days": entry_wait_days,
                "exit_wait_days": exit_wait_days,
                "wait_time_ratio": wait_time_ratio,
                # Metadata
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            self._cache[cache_key] = (enriched_data, time.time())
            return enriched_data

        except requests.exceptions.RequestException as e:
            print(f"[warn] Beaconchain API request failed: {e}")
            return None
        except Exception as e:
            print(f"[warn] Beaconchain API error: {e}")
            return None

    def get_validator_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get general validator statistics.
        Returns total validators, staked ETH, and APR if available.
        """
        # This could be extended to fetch additional metrics
        # For now, queue data includes validatorscount
        queue_data = self.get_validator_queue()
        if not queue_data:
            return None

        return {
            "total_validators": queue_data.get("validatorscount"),
            "timestamp": queue_data.get("timestamp"),
        }

    def clear_cache(self):
        """Clear the API cache."""
        self._cache.clear()


# Module-level instance (can be imported directly)
beacon_api = BeaconchainAPI()
