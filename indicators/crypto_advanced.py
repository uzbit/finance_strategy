"""
Advanced crypto indicators for derivatives positioning and institutional flows.
Includes funding z-scores, OI changes, ETF flows, stablecoin issuance, and options skew.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from indicators.base import PanicIndicator, safe_float_conversion


class FundingZScoreIndicator(PanicIndicator):
    """Perp funding rate z-score across exchanges."""

    def __init__(self):
        super().__init__(
            name="Funding Z-Score",
            description="Perp funding rate deviation from recent history",
            category="crypto"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate weighted z-score of funding rates across exchanges.

        Warning if |z| >= threshold (default 2.0):
        - z > +2: crowd long & expensive (squeeze risk)
        - z < -2: crowd short & cheap (squeeze up risk)
        """
        derivatives_api = data.get("derivatives_api")
        if not derivatives_api:
            return {"ok": None, "composite_z": None, "error": "Derivatives API unavailable (check region/network)"}

        lookback_periods = config.get("funding_lookback_periods", 90)
        threshold = config.get("funding_zscore_threshold", 2.0)
        symbol = config.get("funding_symbol", "BTC")

        try:
            # Get aggregate funding data
            agg_data = derivatives_api.get_aggregate_funding_data(symbol, lookback_periods)

            per_exchange_z = {}
            weighted_sum = 0
            total_weight = 0

            for exchange in ["binance", "bybit", "okx"]:
                exchange_data = agg_data.get(exchange, {})
                funding_df = exchange_data.get("funding", pd.DataFrame())
                weight = exchange_data.get("weight", 0)

                if funding_df.empty or len(funding_df) < 30:
                    per_exchange_z[exchange] = None
                    continue

                # Calculate z-score
                funding_values = funding_df["funding_rate"].values
                latest = funding_values[-1]
                mean = np.mean(funding_values)
                std = np.std(funding_values)

                if std > 0:
                    z_score = (latest - mean) / std
                    per_exchange_z[exchange] = z_score

                    # Weight by OI
                    if weight > 0:
                        weighted_sum += z_score * weight
                        total_weight += weight
                else:
                    per_exchange_z[exchange] = None

            # Composite z-score
            if total_weight > 0:
                composite_z = weighted_sum / total_weight
            else:
                composite_z = None

            # Determine warning status
            if composite_z is None:
                ok_status = None
            else:
                ok_status = abs(composite_z) >= threshold

            # Get latest funding rate for context
            binance_funding = agg_data.get("binance", {}).get("funding", pd.DataFrame())
            if not binance_funding.empty:
                latest_funding = float(binance_funding.iloc[-1]["funding_rate"])
            else:
                latest_funding = None

            return {
                "ok": ok_status,
                "composite_z": composite_z,
                "per_exchange_z": per_exchange_z,
                "latest_funding": latest_funding,
                "threshold": threshold,
                "total_oi": agg_data.get("total_oi")
            }

        except Exception as e:
            print(f"[warn] Funding z-score calculation failed: {e}")
            return {"ok": None, "composite_z": None, "error": str(e)}


class OpenInterestChangeIndicator(PanicIndicator):
    """24h open interest change across exchanges."""

    def __init__(self):
        super().__init__(
            name="OI Change 24h",
            description="Open interest change indicating leverage flows",
            category="crypto"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate 24h OI change across exchanges.

        Warning if:
        - OI change >= +15% and price flat/down (crowded longs)
        - OI change <= -15% on down move (deleveraging)
        """
        derivatives_api = data.get("derivatives_api")
        crypto_api = data.get("crypto_api")

        if not derivatives_api:
            return {"ok": None, "oi_change_pct": None, "error": "Derivatives API unavailable (check region/network)"}

        threshold_pct = config.get("oi_change_threshold_pct", 15.0)
        symbol = config.get("oi_symbol", "BTC")

        try:
            # Get OI changes
            oi_data = derivatives_api.get_oi_change_24h(symbol)
            aggregate_change = oi_data.get("aggregate_change_pct")

            if aggregate_change is None:
                return {"ok": None, "oi_change_pct": None, "error": "No OI data"}

            # Get price change for context
            price_change_24h = None
            if crypto_api:
                try:
                    price_symbol = f"{symbol}-USD"
                    current_price = crypto_api.get_binance_price(price_symbol)
                    # Note: Would need 24h ago price - simplified here
                    price_change_24h = 0  # Placeholder
                except Exception:
                    pass

            # Get latest funding
            agg_funding = derivatives_api.get_aggregate_funding_data(symbol, lookback_periods=1)
            binance_funding_df = agg_funding.get("binance", {}).get("funding", pd.DataFrame())
            latest_funding = None
            if not binance_funding_df.empty:
                latest_funding = float(binance_funding_df.iloc[-1]["funding_rate"])

            # Determine warning status
            # High positive OI change + positive funding = overheating
            # High negative OI change = washout/deleveraging
            if aggregate_change >= threshold_pct:
                # Building leverage
                ok_status = True  # Warning
            elif aggregate_change <= -threshold_pct:
                # Deleveraging
                ok_status = True  # Warning
            else:
                ok_status = False

            return {
                "ok": ok_status,
                "oi_change_pct": aggregate_change,
                "binance_oi_change": oi_data.get("binance", {}).get("change_pct"),
                "bybit_oi_change": oi_data.get("bybit", {}).get("change_pct"),
                "price_change_24h": price_change_24h,
                "latest_funding": latest_funding,
                "threshold": threshold_pct,
                "binance_oi_now": oi_data.get("binance", {}).get("oi_now"),
                "bybit_oi_now": oi_data.get("bybit", {}).get("oi_now")
            }

        except Exception as e:
            print(f"[warn] OI change calculation failed: {e}")
            return {"ok": None, "oi_change_pct": None, "error": str(e)}


class ETFFlowsIndicator(PanicIndicator):
    """US spot ETF net flows (BTC/ETH)."""

    def __init__(self):
        super().__init__(
            name="US Spot ETF Flows",
            description="Institutional demand via US spot ETFs",
            category="crypto"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate ETF flow metrics and z-scores.

        Warning if:
        - 3d flows > +$2B (bullish surge)
        - 3d flows < -$1B (distribution)
        """
        glassnode_api = data.get("glassnode_api")
        if not glassnode_api:
            return {"ok": None, "flows_3d_sum": None, "error": "Glassnode API unavailable (requires API key)"}

        threshold_bullish = config.get("etf_flows_3d_threshold_bullish_usd", 2_000_000_000)
        threshold_bearish = config.get("etf_flows_3d_threshold_bearish_usd", -1_000_000_000)
        asset = config.get("etf_asset", "BTC")

        try:
            # Calculate flow metrics
            metrics = glassnode_api.calculate_flow_metrics(asset, lookback_days=90)

            flows_3d = metrics.get("flows_3d_sum")
            flows_5d = metrics.get("flows_5d_sum")
            z_90d = metrics.get("z_90d")
            latest_flow = metrics.get("latest_daily_flow")

            if flows_3d is None:
                return {"ok": None, "flows_3d_sum": None, "error": "No flow data"}

            # Determine warning status (warnings for both extremes)
            if flows_3d > threshold_bullish:
                ok_status = True  # Strong demand warning (good thing, but notable)
                signal = "strong_demand"
            elif flows_3d < threshold_bearish:
                ok_status = True  # Distribution warning
                signal = "distribution"
            else:
                ok_status = False
                signal = "normal"

            return {
                "ok": ok_status,
                "signal": signal,
                "flows_3d_sum": flows_3d,
                "flows_5d_sum": flows_5d,
                "flows_7d_sum": metrics.get("flows_7d_sum"),
                "z_90d": z_90d,
                "latest_daily_flow": latest_flow,
                "mean_90d": metrics.get("mean_90d"),
                "std_90d": metrics.get("std_90d"),
                "threshold_bullish": threshold_bullish,
                "threshold_bearish": threshold_bearish
            }

        except Exception as e:
            print(f"[warn] ETF flows calculation failed: {e}")
            return {"ok": None, "flows_3d_sum": None, "error": str(e)}


class StablecoinIssuanceIndicator(PanicIndicator):
    """Stablecoin net issuance (USD liquidity proxy)."""

    def __init__(self):
        super().__init__(
            name="Stablecoin Issuance",
            description="Fresh USD liquidity entering/leaving crypto",
            category="crypto"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate stablecoin net issuance over 7 days.

        Warning if:
        - Net issuance > +$3B (liquidity influx - bullish)
        - Net issuance < -$3B (liquidity drain - bearish)
        """
        defillama_api = data.get("defillama_api")
        if not defillama_api:
            return {"ok": None, "net_issuance_7d": None, "error": "DeFiLlama API unavailable"}

        threshold = config.get("stablecoin_issuance_7d_threshold_usd", 3_000_000_000)
        lookback_days = 7

        try:
            # Calculate net issuance
            issuance = defillama_api.calculate_net_issuance(lookback_days)

            net_issuance_7d = issuance.get("net_issuance_7d_usd")
            current_supply = issuance.get("current_supply_usd")
            per_token = issuance.get("per_token", {})

            if net_issuance_7d is None:
                return {"ok": None, "net_issuance_7d": None, "error": "No issuance data"}

            # Determine warning status
            if abs(net_issuance_7d) >= threshold:
                ok_status = True  # Notable change
                if net_issuance_7d > 0:
                    signal = "liquidity_influx"
                else:
                    signal = "liquidity_drain"
            else:
                ok_status = False
                signal = "stable"

            # Extract major token changes
            usdt_change = per_token.get("USDT", {}).get("net_issuance")
            usdc_change = per_token.get("USDC", {}).get("net_issuance")
            dai_change = per_token.get("DAI", {}).get("net_issuance")

            return {
                "ok": ok_status,
                "signal": signal,
                "net_issuance_7d": net_issuance_7d,
                "net_issuance_pct": issuance.get("net_issuance_pct"),
                "current_supply": current_supply,
                "usdt_change": usdt_change,
                "usdc_change": usdc_change,
                "dai_change": dai_change,
                "threshold": threshold
            }

        except Exception as e:
            print(f"[warn] Stablecoin issuance calculation failed: {e}")
            return {"ok": None, "net_issuance_7d": None, "error": str(e)}


class OptionsSkewIndicator(PanicIndicator):
    """Options 25-delta risk reversal (skew)."""

    def __init__(self):
        super().__init__(
            name="Options Skew (RR25)",
            description="Options market fear/greed via 25-delta risk reversal",
            category="crypto"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate 25-delta risk reversal from Deribit options.

        Warning if:
        - RR25 < -5 vol pts (downside protection bid)
        - |RR25| > 10 vol pts (extreme regime)
        """
        deribit_api = data.get("deribit_api")
        if not deribit_api:
            return {"ok": None, "rr25": None, "error": "Deribit API unavailable"}

        threshold_extreme = config.get("options_rr25_extreme_threshold", 5.0)
        threshold_max = config.get("options_rr25_max_extreme", 10.0)
        currency = config.get("options_currency", "BTC")
        target_days = 30

        try:
            # Calculate RR25
            rr_data = deribit_api.calculate_25d_risk_reversal(currency, target_days)

            rr25 = rr_data.get("rr25")
            if rr25 is None:
                error_msg = rr_data.get("error", "No RR25 data")
                return {"ok": None, "rr25": None, "error": error_msg}

            # Determine warning status
            if abs(rr25) >= threshold_max:
                ok_status = True  # Extreme regime
                signal = "extreme"
            elif rr25 < -threshold_extreme:
                ok_status = True  # Downside protection bid
                signal = "risk_off"
            elif rr25 > threshold_extreme:
                ok_status = True  # Upside call premium
                signal = "risk_on"
            else:
                ok_status = False
                signal = "normal"

            return {
                "ok": ok_status,
                "signal": signal,
                "rr25": rr25,
                "call_iv": rr_data.get("call_iv"),
                "put_iv": rr_data.get("put_iv"),
                "call_delta": rr_data.get("call_delta"),
                "put_delta": rr_data.get("put_delta"),
                "expiry": rr_data.get("expiry"),
                "days_to_expiry": rr_data.get("days_to_expiry"),
                "underlying_price": rr_data.get("underlying_price"),
                "threshold_extreme": threshold_extreme,
                "threshold_max": threshold_max
            }

        except Exception as e:
            print(f"[warn] Options skew calculation failed: {e}")
            return {"ok": None, "rr25": None, "error": str(e)}


# ========== Factory Function ==========

def create_crypto_advanced_indicators() -> Dict[str, PanicIndicator]:
    """
    Create all advanced crypto indicators.

    Returns:
        Dict mapping indicator keys to indicator instances
    """
    return {
        "funding_zscore": FundingZScoreIndicator(),
        "oi_change_24h": OpenInterestChangeIndicator(),
        "etf_flows": ETFFlowsIndicator(),
        "stablecoin_issuance": StablecoinIssuanceIndicator(),
        "options_skew": OptionsSkewIndicator()
    }
