"""
Real-time data aggregator for panic indicators.
Combines data from multiple sources for comprehensive market monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
from dataclasses import dataclass

from data.fred import FredAPI
from data.yahoo import yahoo_api
from data.crypto import CryptoAPI
from core.config import config


@dataclass
class MarketSnapshot:
    """Container for real-time market data snapshot."""
    timestamp: datetime
    equity_data: Dict[str, Any]
    credit_data: Dict[str, Any]
    rates_data: Dict[str, Any]
    crypto_data: Dict[str, Any]
    commodity_data: Dict[str, Any]


class RealTimeDataAggregator:
    """Aggregates real-time data from multiple sources for panic monitoring."""

    def __init__(self, crypto_api: CryptoAPI, fred_api: FredAPI):
        self.last_snapshot = None
        self.crypto_api = crypto_api
        self.fred_api = fred_api
        self.data_sources = {
            "fred": self.fred_api,
            "yahoo": yahoo_api,
            "crypto": self.crypto_api
        }

    def get_market_snapshot(self) -> MarketSnapshot:
        """Get comprehensive real-time market snapshot."""
        timestamp = datetime.now()

        # Collect data in parallel for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                "equity": executor.submit(self._get_equity_data),
                "credit": executor.submit(self._get_credit_data),
                "rates": executor.submit(self._get_rates_data),
                "crypto": executor.submit(self._get_crypto_data),
                "commodity": executor.submit(self._get_commodity_data)
            }

            # Gather results
            results = {}
            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=30)
                except Exception as e:
                    print(f"[warn] Failed to get {key} data: {e}")
                    results[key] = {}

        snapshot = MarketSnapshot(
            timestamp=timestamp,
            equity_data=results.get("equity", {}),
            credit_data=results.get("credit", {}),
            rates_data=results.get("rates", {}),
            crypto_data=results.get("crypto", {}),
            commodity_data=results.get("commodity", {})
        )

        self.last_snapshot = snapshot
        return snapshot

    def _get_equity_data(self) -> Dict[str, Any]:
        """Get equity market stress indicators."""
        equity_data = {}

        try:
            # SPY performance and volume
            spy_perf = yahoo_api.get_daily_performance(["SPY"])
            spy_volume = yahoo_api.get_volume_data("SPY")

            equity_data["spy_performance"] = spy_perf.get("SPY")
            equity_data["spy_volume_ratio"] = spy_volume.get("volume_ratio")

            # VIX data
            vix_price = yahoo_api.get_current_price("^VIX")
            equity_data["vix_level"] = vix_price

            # QQQ performance for tech exposure
            qqq_perf = yahoo_api.get_daily_performance(["QQQ"])
            equity_data["qqq_performance"] = qqq_perf.get("QQQ")

            # Get intraday data for crash detection (if market hours)
            market_hours = yahoo_api.check_market_hours()
            if market_hours.get("us_market_open"):
                spy_intraday = yahoo_api.get_intraday_data("SPY", period="1d", interval="5m")
                if not spy_intraday.empty:
                    # Calculate intraday move from market open
                    open_price = spy_intraday["Open"].iloc[0]
                    current_price = spy_intraday["Close"].iloc[-1]
                    intraday_change = (current_price - open_price) / open_price * 100
                    equity_data["spy_intraday_change"] = float(intraday_change)

        except Exception as e:
            print(f"[warn] Equity data collection failed: {e}")

        return equity_data

    def _get_credit_data(self) -> Dict[str, Any]:
        """Get credit market stress indicators."""
        credit_data = {}

        try:
            # High-yield credit ETF (HYG) performance and volume
            hyg_perf = yahoo_api.get_daily_performance(["HYG"])
            hyg_volume = yahoo_api.get_volume_data("HYG")

            credit_data["hyg_performance"] = hyg_perf.get("HYG")
            credit_data["hyg_volume_ratio"] = hyg_volume.get("volume_ratio")

            # Investment grade credit (LQD)
            lqd_perf = yahoo_api.get_daily_performance(["LQD"])
            credit_data["lqd_performance"] = lqd_perf.get("LQD")

            # Treasury ETFs for flight-to-quality
            tlt_perf = yahoo_api.get_daily_performance(["TLT"])
            credit_data["tlt_performance"] = tlt_perf.get("TLT")

        except Exception as e:
            print(f"[warn] Credit data collection failed: {e}")

        return credit_data

    def _get_rates_data(self) -> Dict[str, Any]:
        """Get interest rates and FX stress indicators."""
        rates_data = {}

        try:
            # Get latest Treasury yields from FRED (daily data)
            ten_year = self.fred_api.get_latest_value("DGS10", "2023-01-01")
            two_year = self.fred_api.get_latest_value("DGS2", "2023-01-01")

            if ten_year and two_year:
                rates_data["ten_year_yield"] = ten_year
                rates_data["two_year_yield"] = two_year
                rates_data["yield_curve_2_10"] = ten_year - two_year

            # Dollar strength (DXY proxy via UUP ETF)
            uup_perf = yahoo_api.get_daily_performance(["UUP"])
            rates_data["dollar_performance"] = uup_perf.get("UUP")

        except Exception as e:
            print(f"[warn] Rates data collection failed: {e}")

        return rates_data

    def _get_crypto_data(self) -> Dict[str, Any]:
        """Get cryptocurrency stress indicators."""
        crypto_data = {}

        try:
            # Major crypto performance
            crypto_symbols = ["BTC-USD", "ETH-USD"]
            crypto_perf = yahoo_api.get_daily_performance(crypto_symbols)

            crypto_data["btc_performance"] = crypto_perf.get("BTC-USD")
            crypto_data["eth_performance"] = crypto_perf.get("ETH-USD")

            # Stablecoin stress
            stablecoin_stress = self.crypto_api.check_stablecoin_stress()
            crypto_data["stablecoin_stress"] = stablecoin_stress

            # Fear & Greed Index
            fear_greed = self.crypto_api.get_crypto_fear_greed_index()
            if fear_greed:
                crypto_data["fear_greed_index"] = fear_greed["value"]

        except Exception as e:
            print(f"[warn] Crypto data collection failed: {e}")

        return crypto_data

    def _get_commodity_data(self) -> Dict[str, Any]:
        """Get commodity stress indicators."""
        commodity_data = {}

        try:
            # Oil performance (multiple proxies)
            oil_tickers = ["USO", "CL=F"]  # USO ETF and WTI futures
            oil_perf = yahoo_api.get_daily_performance(oil_tickers)

            # Use first available oil price
            for ticker in oil_tickers:
                if oil_perf.get(ticker) is not None:
                    commodity_data["oil_performance"] = oil_perf[ticker]
                    break

            # Copper via mining ETF (COPX) as growth proxy
            copper_perf = yahoo_api.get_daily_performance(["COPX"])
            commodity_data["copper_performance"] = copper_perf.get("COPX")

            # Gold as safe haven
            gold_perf = yahoo_api.get_daily_performance(["GLD"])
            commodity_data["gold_performance"] = gold_perf.get("GLD")

        except Exception as e:
            print(f"[warn] Commodity data collection failed: {e}")

        return commodity_data

    def get_cross_asset_correlation(self) -> Dict[str, Any]:
        """Analyze cross-asset correlations for stress detection."""
        if not self.last_snapshot:
            return {}

        snapshot = self.last_snapshot
        correlation_signals = {}

        try:
            # Extract key performance metrics
            equity_perf = snapshot.equity_data.get("spy_performance", 0)
            credit_perf = snapshot.credit_data.get("hyg_performance", 0)
            crypto_perf = snapshot.crypto_data.get("btc_performance", 0)

            # Check for "everything down together" scenario
            all_negative = all([
                perf is not None and perf < -3
                for perf in [equity_perf, credit_perf, crypto_perf]
            ])

            correlation_signals["everything_down"] = all_negative
            correlation_signals["equity_credit_crypto_down"] = all_negative

            # Dollar strength with risk-off
            dollar_perf = snapshot.rates_data.get("dollar_performance", 0)
            dollar_up_risk_down = (
                dollar_perf is not None and dollar_perf > 1.5 and
                equity_perf is not None and equity_perf < -2
            )

            correlation_signals["dollar_strength_risk_off"] = dollar_up_risk_down

        except Exception as e:
            print(f"[warn] Correlation analysis failed: {e}")

        return correlation_signals

    def check_data_freshness(self) -> Dict[str, bool]:
        """Check if data sources are providing fresh data."""
        freshness = {}

        try:
            # Check market hours
            market_hours = yahoo_api.check_market_hours()
            market_open = market_hours.get("us_market_open", False)

            # During market hours, data should be very fresh
            # Outside market hours, we can be more lenient
            max_age_minutes = 5 if market_open else 60

            if self.last_snapshot:
                age_minutes = (datetime.now() - self.last_snapshot.timestamp).total_seconds() / 60
                freshness["snapshot_fresh"] = age_minutes <= max_age_minutes
            else:
                freshness["snapshot_fresh"] = False

            freshness["market_open"] = market_open
            freshness["last_update"] = self.last_snapshot.timestamp.isoformat() if self.last_snapshot else None

        except Exception as e:
            print(f"[warn] Freshness check failed: {e}")
            freshness = {"snapshot_fresh": False, "market_open": None}

        return freshness


# Note: This module provides the RealTimeDataAggregator class.
# Instantiation should be done by the caller with proper API keys.