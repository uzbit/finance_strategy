"""
Crypto data APIs for real-time cryptocurrency market data.
Supports Binance, CoinGecko, and other crypto data sources.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import time


class CryptoAPI:
    """Cryptocurrency data aggregator."""

    def __init__(self, coingecko_api_key: str = ""):
        self.coingecko_api_key = coingecko_api_key
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes

    def get_binance_price(self, symbol: str) -> Optional[float]:
        """Get current price from Binance API."""
        try:
            # Convert symbol format (BTC-USD -> BTCUSDT)
            binance_symbol = symbol.replace('-USD', 'USDT').replace('-USDT', 'USDT')

            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": binance_symbol}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            return float(data["price"])

        except Exception as e:
            print(f"[warn] Binance API failed for {symbol}: {e}")
            return None

    def get_coingecko_data(self, coin_id: str) -> Dict[str, Any]:
        """Get comprehensive data from CoinGecko API."""
        cache_key = f"coingecko_{coin_id}"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data

        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false"
            }

            headers = {}
            if self.coingecko_api_key:
                headers["x-cg-demo-api-key"] = self.coingecko_api_key

            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()

            data = response.json()
            market_data = data.get("market_data", {})

            result = {
                "price_usd": market_data.get("current_price", {}).get("usd"),
                "price_change_24h": market_data.get("price_change_percentage_24h"),
                "volume_24h": market_data.get("total_volume", {}).get("usd"),
                "market_cap": market_data.get("market_cap", {}).get("usd"),
                "fear_greed_index": None  # Would need separate API
            }

            # Cache result
            self._cache[cache_key] = (result, time.time())
            return result

        except Exception as e:
            print(f"[warn] CoinGecko API failed for {coin_id}: {e}")
            return {}

    def get_stablecoin_prices(self) -> Dict[str, float]:
        """Get current stablecoin prices to detect depegging."""
        stablecoins = {
            "tether": "USDT",
            "usd-coin": "USDC",
            "dai": "DAI",
            "frax": "FRAX"
        }

        prices = {}

        for coin_id, symbol in stablecoins.items():
            try:
                data = self.get_coingecko_data(coin_id)
                price = data.get("price_usd")
                if price:
                    prices[symbol] = price
            except Exception:
                continue

        return prices

    def check_stablecoin_stress(self, depeg_threshold: float = 0.995,
                              duration_minutes: int = 15) -> Dict[str, Any]:
        """Check for stablecoin depegging events."""
        prices = self.get_stablecoin_prices()

        stress_signals = {}
        depegged_coins = []

        for coin, price in prices.items():
            is_depegged = price < depeg_threshold
            stress_signals[coin] = {
                "price": price,
                "depegged": is_depegged,
                "deviation": abs(1.0 - price)
            }

            if is_depegged:
                depegged_coins.append(coin)

        return {
            "depegged_coins": depegged_coins,
            "stress_level": len(depegged_coins),
            "prices": stress_signals,
            "timestamp": datetime.now().isoformat()
        }

    def get_crypto_correlation(self, symbols: List[str],
                             timeframe: str = "1d") -> Dict[str, float]:
        """Get crypto price movements for correlation analysis."""
        performance = {}

        symbol_mapping = {
            "BTC-USD": "bitcoin",
            "ETH-USD": "ethereum",
            "XMR-USD": "monero",
            "XRP-USD": "ripple",
            "ADA-USD": "cardano",
            "DOT-USD": "polkadot"
        }

        for symbol in symbols:
            coin_id = symbol_mapping.get(symbol)
            if not coin_id:
                continue

            try:
                data = self.get_coingecko_data(coin_id)
                price_change = data.get("price_change_24h")
                if price_change is not None:
                    performance[symbol] = price_change
            except Exception:
                continue

        return performance

    def get_crypto_fear_greed_index(self) -> Optional[Dict[str, Any]]:
        """Get crypto fear and greed index (if available)."""
        try:
            # Alternative.me crypto fear and greed index
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if "data" in data and data["data"]:
                fng_data = data["data"][0]
                return {
                    "value": int(fng_data["value"]),
                    "classification": fng_data["value_classification"],
                    "timestamp": fng_data["timestamp"]
                }

        except Exception as e:
            print(f"[warn] Fear & Greed Index API failed: {e}")

        return None

    def get_binance_funding_rates(self, symbols: List[str]) -> Dict[str, float]:
        """Get perpetual futures funding rates from Binance."""
        funding_rates = {}

        for symbol in symbols:
            try:
                # Convert to Binance perp format
                binance_symbol = symbol.replace('-USD', 'USDT') + 'PERP'
                if not binance_symbol.endswith('USDTPERP'):
                    binance_symbol = symbol.replace('-USD', '') + 'USDTPERP'

                url = "https://fapi.binance.com/fapi/v1/premiumIndex"
                params = {"symbol": binance_symbol}

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()
                funding_rate = float(data["lastFundingRate"])
                funding_rates[symbol] = funding_rate * 100  # Convert to percentage

            except Exception:
                continue

        return funding_rates

    def clear_cache(self):
        """Clear the crypto data cache."""
        self._cache.clear()

    @classmethod
    def get_supported_symbols(cls) -> Dict[str, str]:
        """Get supported crypto symbols and their descriptions."""
        return {
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",
            "XMR-USD": "Monero",
            "XRP-USD": "Ripple",
            "ADA-USD": "Cardano",
            "DOT-USD": "Polkadot",
            "USDT": "Tether (Stablecoin)",
            "USDC": "USD Coin (Stablecoin)",
            "DAI": "Dai (Stablecoin)"
        }


# Note: This module provides the CryptoAPI class.
# Instantiation should be done by the caller with proper API key.