"""
Real-time panic indicators for immediate market stress detection.
Designed to catch crashes and extreme volatility in real-time.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from indicators.base import PanicIndicator, safe_float_conversion, calculate_percentage_change
from data.realtime import realtime_data


class VIXSpikeIndicator(PanicIndicator):
    """VIX volatility spike panic indicator."""

    def __init__(self):
        super().__init__(
            name="VIX Spike",
            description="VIX >40 indicates extreme fear",
            category="equity"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if VIX > panic threshold."""
        snapshot = realtime_data.get_market_snapshot()
        vix_level = snapshot.equity_data.get("vix_level")

        threshold = config.get("panic_thresholds", {}).get("vix_spike", 40.0)

        if vix_level is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        vix_float = safe_float_conversion(vix_level)
        if vix_float is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        return {
            "ok": vix_float > threshold,
            "latest": vix_float,
            "threshold": threshold
        }


class EquityStressIndicator(PanicIndicator):
    """SPY intraday crash detection."""

    def __init__(self):
        super().__init__(
            name="Equity Stress",
            description="SPY intraday drop >4% indicates panic selling",
            category="equity"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if SPY intraday drop > threshold."""
        snapshot = realtime_data.get_market_snapshot()

        # Check daily performance first
        spy_daily = snapshot.equity_data.get("spy_performance")
        spy_intraday = snapshot.equity_data.get("spy_intraday_change")

        threshold = config.get("panic_thresholds", {}).get("spy_intraday_drop", -4.0)

        # Use intraday if available (market hours), otherwise daily
        current_move = spy_intraday if spy_intraday is not None else spy_daily

        if current_move is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        move_float = safe_float_conversion(current_move)
        if move_float is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        return {
            "ok": move_float <= threshold,  # Negative threshold
            "latest": move_float,
            "threshold": threshold,
            "intraday": spy_intraday is not None
        }


class VolumeStressIndicator(PanicIndicator):
    """High volume stress across equity markets."""

    def __init__(self):
        super().__init__(
            name="Volume Stress",
            description="SPY volume >3x average indicates panic",
            category="equity"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if volume ratio > threshold."""
        snapshot = realtime_data.get_market_snapshot()
        volume_ratio = snapshot.equity_data.get("spy_volume_ratio")

        threshold = config.get("panic_thresholds", {}).get("volume_spike", 3.0)

        if volume_ratio is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        ratio_float = safe_float_conversion(volume_ratio)
        if ratio_float is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        return {
            "ok": ratio_float > threshold,
            "latest": ratio_float,
            "threshold": threshold
        }


class CreditStressIndicator(PanicIndicator):
    """High-yield credit market stress."""

    def __init__(self):
        super().__init__(
            name="Credit Stress",
            description="HYG >5% drop indicates credit panic",
            category="credit"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if HYG performance < threshold."""
        snapshot = realtime_data.get_market_snapshot()
        hyg_perf = snapshot.credit_data.get("hyg_performance")

        threshold = config.get("panic_thresholds", {}).get("hyg_drop", -5.0)

        if hyg_perf is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        perf_float = safe_float_conversion(hyg_perf)
        if perf_float is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        return {
            "ok": perf_float <= threshold,  # Negative threshold
            "latest": perf_float,
            "threshold": threshold
        }


class FlightToQualityIndicator(PanicIndicator):
    """Flight to quality in Treasury markets."""

    def __init__(self):
        super().__init__(
            name="Flight to Quality",
            description="TLT >3% gain indicates flight to safety",
            category="rates"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if TLT performance > threshold."""
        snapshot = realtime_data.get_market_snapshot()
        tlt_perf = snapshot.credit_data.get("tlt_performance")

        threshold = config.get("panic_thresholds", {}).get("tlt_rally", 3.0)

        if tlt_perf is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        perf_float = safe_float_conversion(tlt_perf)
        if perf_float is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        return {
            "ok": perf_float >= threshold,
            "latest": perf_float,
            "threshold": threshold
        }


class DollarStrengthIndicator(PanicIndicator):
    """Dollar strength during risk-off."""

    def __init__(self):
        super().__init__(
            name="Dollar Strength",
            description="UUP >2% gain during risk-off",
            category="rates"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if dollar up during equity stress."""
        snapshot = realtime_data.get_market_snapshot()

        dollar_perf = snapshot.rates_data.get("dollar_performance")
        spy_perf = snapshot.equity_data.get("spy_performance")

        dollar_threshold = config.get("panic_thresholds", {}).get("dollar_rally", 2.0)

        if dollar_perf is None or spy_perf is None:
            return {"ok": None, "latest": dollar_perf, "threshold": dollar_threshold}

        dollar_float = safe_float_conversion(dollar_perf)
        spy_float = safe_float_conversion(spy_perf)

        if dollar_float is None or spy_float is None:
            return {"ok": None, "latest": dollar_float, "threshold": dollar_threshold}

        # Dollar strength AND equity weakness
        dollar_strong = dollar_float >= dollar_threshold
        equity_weak = spy_float <= -2.0  # SPY down >2%

        return {
            "ok": dollar_strong and equity_weak,
            "latest": dollar_float,
            "threshold": dollar_threshold,
            "equity_performance": spy_float
        }


class CryptoStressIndicator(PanicIndicator):
    """Cryptocurrency market stress."""

    def __init__(self):
        super().__init__(
            name="Crypto Stress",
            description="BTC >10% drop indicates crypto panic",
            category="crypto"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if BTC performance < threshold."""
        snapshot = realtime_data.get_market_snapshot()
        btc_perf = snapshot.crypto_data.get("btc_performance")

        threshold = config.get("panic_thresholds", {}).get("btc_drop", -10.0)

        if btc_perf is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        perf_float = safe_float_conversion(btc_perf)
        if perf_float is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        return {
            "ok": perf_float <= threshold,  # Negative threshold
            "latest": perf_float,
            "threshold": threshold
        }


class StablecoinStressIndicator(PanicIndicator):
    """Stablecoin depegging stress."""

    def __init__(self):
        super().__init__(
            name="Stablecoin Stress",
            description="Stablecoin depegging indicates crypto panic",
            category="crypto"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if any major stablecoin depegged."""
        snapshot = realtime_data.get_market_snapshot()
        stablecoin_stress = snapshot.crypto_data.get("stablecoin_stress", {})

        depegged_coins = stablecoin_stress.get("depegged_coins", [])
        stress_level = stablecoin_stress.get("stress_level", 0)

        return {
            "ok": len(depegged_coins) > 0,
            "latest": stress_level,
            "depegged_coins": depegged_coins,
            "threshold": 1  # Any depegged coin triggers
        }


class CommodityStressIndicator(PanicIndicator):
    """Oil and commodity stress indicator."""

    def __init__(self):
        super().__init__(
            name="Commodity Stress",
            description="Oil >8% move indicates commodity stress",
            category="commodity"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if oil volatility > threshold."""
        snapshot = realtime_data.get_market_snapshot()
        oil_perf = snapshot.commodity_data.get("oil_performance")

        threshold = config.get("panic_thresholds", {}).get("oil_volatility", 8.0)

        if oil_perf is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        perf_float = safe_float_conversion(oil_perf)
        if perf_float is None:
            return {"ok": None, "latest": None, "threshold": threshold}

        # Check absolute move (up or down)
        abs_move = abs(perf_float)

        return {
            "ok": abs_move >= threshold,
            "latest": perf_float,
            "threshold": threshold,
            "absolute_move": abs_move
        }


class CrossAssetCorrelationIndicator(PanicIndicator):
    """Cross-asset correlation breakdown."""

    def __init__(self):
        super().__init__(
            name="Cross-Asset Correlation",
            description="Everything down together indicates systemic stress",
            category="equity"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if cross-asset correlations indicate systemic stress."""
        correlation_signals = realtime_data.get_cross_asset_correlation()

        everything_down = correlation_signals.get("everything_down", False)
        dollar_risk_off = correlation_signals.get("dollar_strength_risk_off", False)

        # Either scenario indicates stress
        stress_detected = everything_down or dollar_risk_off

        return {
            "ok": stress_detected,
            "latest": stress_detected,
            "everything_down": everything_down,
            "dollar_risk_off": dollar_risk_off,
            "threshold": True
        }


class MarketStructureIndicator(PanicIndicator):
    """Market structure breakdown indicator."""

    def __init__(self):
        super().__init__(
            name="Market Structure",
            description="Market microstructure stress indicators",
            category="equity"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market structure stress via multiple signals."""
        snapshot = realtime_data.get_market_snapshot()

        # Multiple stress signals
        stress_signals = []

        # VIX term structure (if available)
        vix_level = snapshot.equity_data.get("vix_level")
        if vix_level and safe_float_conversion(vix_level) > 35:
            stress_signals.append("vix_elevated")

        # Volume stress
        spy_volume = snapshot.equity_data.get("spy_volume_ratio")
        if spy_volume and safe_float_conversion(spy_volume) > 2.5:
            stress_signals.append("volume_stress")

        # Credit-equity divergence
        spy_perf = snapshot.equity_data.get("spy_performance")
        hyg_perf = snapshot.credit_data.get("hyg_performance")

        if spy_perf and hyg_perf:
            spy_float = safe_float_conversion(spy_perf)
            hyg_float = safe_float_conversion(hyg_perf)

            if spy_float and hyg_float and (spy_float - hyg_float) > 3.0:
                stress_signals.append("credit_equity_divergence")

        stress_count = len(stress_signals)
        threshold = config.get("panic_thresholds", {}).get("structure_stress", 2)

        return {
            "ok": stress_count >= threshold,
            "latest": stress_count,
            "threshold": threshold,
            "stress_signals": stress_signals
        }


# Factory function to create all panic indicators
def create_panic_indicators() -> Dict[str, PanicIndicator]:
    """Create all panic indicators."""
    return {
        "vix_spike": VIXSpikeIndicator(),
        "equity_stress": EquityStressIndicator(),
        "volume_stress": VolumeStressIndicator(),
        "credit_stress": CreditStressIndicator(),
        "flight_quality": FlightToQualityIndicator(),
        "dollar_strength": DollarStrengthIndicator(),
        "crypto_stress": CryptoStressIndicator(),
        "stablecoin_stress": StablecoinStressIndicator(),
        "commodity_stress": CommodityStressIndicator(),
        "cross_asset": CrossAssetCorrelationIndicator(),
        "market_structure": MarketStructureIndicator()
    }


def calculate_panic_score(indicators: Dict[str, PanicIndicator],
                         config: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall panic score based on active indicators."""
    active_indicators = []
    total_score = 0
    category_scores = {"equity": 0, "credit": 0, "rates": 0, "crypto": 0, "commodity": 0}

    for name, indicator in indicators.items():
        if indicator.is_warning():
            active_indicators.append(name)
            total_score += 1
            category_scores[indicator.category] += 1

    # Get panic levels from config
    panic_levels = config.get("panic_levels", {
        "low": 2, "medium": 4, "high": 6, "extreme": 8
    })

    # Determine panic level
    panic_level = "normal"
    if total_score >= panic_levels.get("extreme", 8):
        panic_level = "extreme"
    elif total_score >= panic_levels.get("high", 6):
        panic_level = "high"
    elif total_score >= panic_levels.get("medium", 4):
        panic_level = "medium"
    elif total_score >= panic_levels.get("low", 2):
        panic_level = "low"

    return {
        "total_score": total_score,
        "panic_level": panic_level,
        "active_indicators": active_indicators,
        "category_scores": category_scores,
        "max_possible": len(indicators),
        "timestamp": datetime.now().isoformat()
    }