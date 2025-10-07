"""
Macro economic indicators for long-term risk assessment.
Extracted from the original risk_dashboard.py with modular structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from indicators.base import MacroIndicator, AssetTrendIndicator, safe_float_conversion, calculate_percentage_change
# FRED API is now injected via data parameter in calculate() methods
from data.yahoo import yahoo_api


class YieldCurveIndicator(MacroIndicator):
    """Detects yield curve inversion (T10Y3M < 0)."""

    def __init__(self):
        super().__init__(
            name="Yield Curve Inverted",
            description="10Y minus 3M Treasury spread",
            series_id="T10Y3M"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if latest value < 0."""
        df = data.get("T10Y3M")
        if df is None or df.empty:
            return {"ok": None, "latest": None}

        latest = safe_float_conversion(df["value"].iloc[-1])
        if latest is None:
            return {"ok": None, "latest": None}

        return {"ok": latest < 0.0, "latest": latest}


class SahmRuleIndicator(MacroIndicator):
    """Sahm Rule: 3m avg unemployment vs minimum of prior 12 months."""

    def __init__(self):
        super().__init__(
            name="Sahm Rule Triggered",
            description="Unemployment rate recession indicator",
            series_id="UNRATE"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Sahm Rule: 3m avg unemployment >= 0.5 ppt above 12m low."""
        df = data.get("UNRATE")
        if df is None or df.empty or len(df) < 15:
            return {"ok": None, "gap": None, "latest_3m": None, "min_prior_12m": None}

        u = df["value"].copy()
        u3 = u.rolling(3).mean()

        # Calculate gaps for each period
        gaps = []
        for i in range(len(u3)):
            if i < 14:  # Need at least 15 months
                gaps.append(np.nan)
                continue

            # Prior 12 months of 3m avg (i-12 to i-1 inclusive)
            prior_window = u3.iloc[i-12:i]
            min_prior = prior_window.min()
            gap = u3.iloc[i] - min_prior
            gaps.append(gap)

        sahm = pd.Series(gaps, index=u3.index, name="gap")
        latest_gap = safe_float_conversion(sahm.dropna().iloc[-1])
        latest_u3 = safe_float_conversion(u3.dropna().iloc[-1])
        min_prior = safe_float_conversion((u3.iloc[-12-1:-1]).min()) if len(u3.dropna()) >= 13 else None

        return {
            "ok": latest_gap >= 0.5 if latest_gap is not None else None,
            "gap": latest_gap,
            "latest_3m": latest_u3,
            "min_prior_12m": min_prior
        }


class HighYieldSpreadIndicator(MacroIndicator):
    """High-yield credit spread jump indicator."""

    def __init__(self):
        super().__init__(
            name="High-Yield Spread Jump",
            description="Credit stress via high-yield bond spreads",
            series_id="BAMLH0A0HYM2"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if spread >= threshold above 6-month low."""
        df = data.get("BAMLH0A0HYM2")
        if df is None or df.empty:
            return {"ok": None, "delta": None, "recent_min": None, "latest": None}

        lookback_months = config.get("high_yield_lookback_months", 6)
        threshold_pts = config.get("high_yield_threshold_pts", 1.0)

        hy_m = df["value"].asfreq("MS").interpolate()
        recent = hy_m.tail(lookback_months * 30)  # Approximate monthly data

        if recent.empty:
            return {"ok": None, "delta": None, "recent_min": None, "latest": None}

        recent_min = safe_float_conversion(recent.min())
        latest = safe_float_conversion(hy_m.iloc[-1])

        if recent_min is None or latest is None:
            return {"ok": None, "delta": None, "recent_min": recent_min, "latest": latest}

        delta = latest - recent_min

        return {
            "ok": delta >= threshold_pts,
            "delta": delta,
            "recent_min": recent_min,
            "latest": latest
        }


class FinancialConditionsIndicator(MacroIndicator):
    """Chicago Fed National Financial Conditions Index."""

    def __init__(self):
        super().__init__(
            name="Financial Conditions Tight",
            description="NFCI > 0 indicates tight conditions",
            series_id="NFCI"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if NFCI > 0."""
        df = data.get("NFCI")
        if df is None or df.empty:
            return {"ok": None, "latest": None}

        latest = safe_float_conversion(df["value"].iloc[-1])
        if latest is None:
            return {"ok": None, "latest": None}

        return {"ok": latest > 0.0, "latest": latest}


class BuildingPermitsIndicator(MacroIndicator):
    """Housing permits decline indicator."""

    def __init__(self):
        super().__init__(
            name="Building Permits Falling",
            description="Housing market weakness via permits",
            series_id="PERMIT"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if 6-month change <= 0%."""
        df = data.get("PERMIT")
        if df is None or df.empty:
            return {"ok": None, "pct_change": None}

        months = config.get("permits_lookback_months", 6)
        if len(df) < months + 1:
            return {"ok": None, "pct_change": None}

        p = df["value"].asfreq("MS").interpolate()
        past = safe_float_conversion(p.iloc[-months-1])
        latest = safe_float_conversion(p.iloc[-1])

        if past is None or latest is None:
            return {"ok": None, "pct_change": None}

        pct = calculate_percentage_change(latest, past)

        return {"ok": pct <= 0.0, "pct_change": pct}


class VIXElevatedIndicator(MacroIndicator):
    """VIX volatility spike indicator."""

    def __init__(self):
        super().__init__(
            name="VIX Elevated",
            description="Market fear gauge via VIX",
            series_id="VIXCLS"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if VIX > threshold."""
        df = data.get("VIXCLS")
        if df is None or df.empty:
            return {"ok": None, "latest": None}

        threshold = config.get("vix_high_threshold", 25.0)
        latest = safe_float_conversion(df["value"].iloc[-1])

        if latest is None:
            return {"ok": None, "latest": None}

        return {"ok": latest > threshold, "latest": latest}


class ConsumerSentimentIndicator(MacroIndicator):
    """Consumer sentiment weakness indicator."""

    def __init__(self):
        super().__init__(
            name="Consumer Sentiment Weak",
            description="University of Michigan sentiment index",
            series_id="UMCSENT"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if sentiment < threshold or declining significantly."""
        df = data.get("UMCSENT")
        if df is None or df.empty or len(df) < 2:
            return {"ok": None, "latest": None, "recent_avg": None}

        threshold = config.get("consumer_sentiment_low_threshold", 80.0)
        lookback_months = config.get("consumer_sentiment_lookback_months", 6)

        s = df["value"].asfreq("MS").interpolate()
        if len(s) < lookback_months + 1:
            return {"ok": None, "latest": None, "recent_avg": None}

        latest = safe_float_conversion(s.iloc[-1])
        recent_avg = safe_float_conversion(s.iloc[-lookback_months:].mean())
        long_term_avg = safe_float_conversion(s.mean())

        if latest is None or recent_avg is None or long_term_avg is None:
            return {"ok": None, "latest": latest, "recent_avg": recent_avg}

        # Trigger if below threshold OR recent average significantly below long-term
        decline_significant = recent_avg < long_term_avg * 0.9
        trigger = latest < threshold or decline_significant

        return {"ok": trigger, "latest": latest, "recent_avg": recent_avg}


class RealRatesIndicator(MacroIndicator):
    """Real interest rates (TIPS) indicator."""

    def __init__(self):
        super().__init__(
            name="Real Rates Negative",
            description="10-Year TIPS real yield",
            series_id="DFII10"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if real rates < threshold."""
        df = data.get("DFII10")
        if df is None or df.empty:
            return {"ok": None, "latest": None}

        threshold = config.get("real_rates_low_threshold", 0.0)
        latest = safe_float_conversion(df["value"].iloc[-1])

        if latest is None:
            return {"ok": None, "latest": None}

        return {"ok": latest < threshold, "latest": latest}


class InflationExpectationsIndicator(MacroIndicator):
    """Inflation expectations indicator."""

    def __init__(self):
        super().__init__(
            name="Inflation Expectations High",
            description="10-Year breakeven inflation rate",
            series_id="T10YIE"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if inflation expectations > threshold."""
        df = data.get("T10YIE")
        if df is None or df.empty:
            return {"ok": None, "latest": None}

        threshold = config.get("inflation_expectations_high_threshold", 3.0)
        latest = safe_float_conversion(df["value"].iloc[-1])

        if latest is None:
            return {"ok": None, "latest": None}

        return {"ok": latest > threshold, "latest": latest}


class OilVolatilityIndicator(MacroIndicator):
    """Oil price volatility indicator."""

    def __init__(self):
        super().__init__(
            name="Oil Volatility High",
            description="WTI crude oil price volatility",
            series_id="DCOILWTICO"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if 3-month volatility > threshold."""
        df = data.get("DCOILWTICO")
        if df is None or df.empty:
            return {"ok": None, "volatility": None, "latest": None}

        lookback_months = config.get("oil_lookback_months", 3)
        threshold_pct = config.get("oil_volatility_threshold_pct", 20.0)

        if len(df) < lookback_months * 30:
            return {"ok": None, "volatility": None, "latest": None}

        # Calculate monthly percentage changes
        monthly = df["value"].resample("MS").last().dropna()
        if len(monthly) < lookback_months + 1:
            return {"ok": None, "volatility": None, "latest": None}

        pct_changes = monthly.pct_change().dropna() * 100
        recent_vol = safe_float_conversion(pct_changes.iloc[-lookback_months:].std())
        latest = safe_float_conversion(df["value"].iloc[-1])

        if recent_vol is None:
            return {"ok": None, "volatility": recent_vol, "latest": latest}

        return {"ok": recent_vol > threshold_pct, "volatility": recent_vol, "latest": latest}


class LeadingIndicatorsIndicator(MacroIndicator):
    """OECD Leading Economic Indicators."""

    def __init__(self):
        super().__init__(
            name="Leading Indicators Declining",
            description="OECD composite leading indicators",
            series_id="USALOLITOAASTSAM"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if 6-month decline > threshold."""
        df = data.get("USALOLITOAASTSAM")
        if df is None or df.empty:
            return {"ok": None, "pct_change": None, "latest": None}

        lookback_months = config.get("leading_indicators_lookback_months", 6)
        threshold_pct = config.get("leading_indicators_decline_threshold_pct", -2.0)

        if len(df) < lookback_months + 1:
            return {"ok": None, "pct_change": None, "latest": None}

        l = df["value"].asfreq("MS").interpolate()
        if len(l) < lookback_months + 1:
            return {"ok": None, "pct_change": None, "latest": None}

        past = safe_float_conversion(l.iloc[-lookback_months-1])
        latest = safe_float_conversion(l.iloc[-1])

        if past is None or latest is None:
            return {"ok": None, "pct_change": None, "latest": latest}

        pct_change = calculate_percentage_change(latest, past)

        return {"ok": pct_change <= threshold_pct, "pct_change": pct_change, "latest": latest}


class ValidatorQueueIndicator(MacroIndicator):
    """ETH validator queue health indicator."""

    def __init__(self):
        super().__init__(
            name="ETH Validator Exit Pressure",
            description="Percentage of validator queue that is exits",
            series_id=None  # Uses BeaconchainAPI, not FRED
        )

    def get_data_requirements(self) -> Dict[str, Any]:
        """Get data requirements for this indicator."""
        return {
            "source": "beacon",
            "frequency": "realtime"
        }

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate entry and exit percentages.

        Warning conditions:
        - Bull warning: entry % > entry_threshold (validators rushing in - overheated)
        - Bear warning: exit % > exit_threshold (validators leaving - exodus)
        """
        # Get beacon API data from the data dict
        beacon_data = data.get("beacon_queue")
        if not beacon_data:
            return {
                "ok": None,
                "exit_pct": None,
                "entry_pct": None,
                "entry_validators": None,
                "exit_validators": None,
                "total_queue": None,
                "signal": None
            }

        # Extract metrics
        entry_validators = beacon_data.get("beaconchain_entering", 0)
        exit_validators = beacon_data.get("beaconchain_exiting", 0)
        total_queue = entry_validators + exit_validators

        if total_queue == 0:
            return {
                "ok": None,
                "exit_pct": None,
                "entry_pct": None,
                "entry_validators": entry_validators,
                "exit_validators": exit_validators,
                "total_queue": total_queue,
                "signal": None
            }

        # Calculate percentages
        exit_pct = (exit_validators / total_queue) * 100
        entry_pct = (entry_validators / total_queue) * 100

        # Get thresholds from config
        entry_threshold = config.get("validator_entry_pct_threshold", 80.0)  # Default: 80% entry is too hot
        exit_threshold = config.get("validator_exit_pct_threshold", 80.0)    # Default: 80% exit is exodus

        # Determine warning condition
        signal = None
        trigger = False

        if entry_pct > entry_threshold:
            trigger = True
            signal = "bull_warning"  # Too many validators entering
        elif exit_pct > exit_threshold:
            trigger = True
            signal = "bear_warning"  # Too many validators exiting

        return {
            "ok": trigger,
            "exit_pct": exit_pct,
            "entry_pct": entry_pct,
            "entry_validators": entry_validators,
            "exit_validators": exit_validators,
            "total_queue": total_queue,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "signal": signal,
            "latest": exit_pct  # For standard formatting (show exit %)
        }


class TrendBreakIndicator(AssetTrendIndicator):
    """Asset price trend break indicator (200-day SMA)."""

    def __init__(self, ticker: str):
        super().__init__(
            name=f"{ticker} Trend Break (200d)",
            description=f"200-day SMA trend break for {ticker}",
            ticker=ticker
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if Close < SMA200 * (1 - filter)."""
        try:
            start_date = config.get("data_start_date", "2015-01-01")
            pct_filter = config.get("trend_filter_pct", 0.02)

            df = yahoo_api.get_price_data(self.ticker, period="2y")
            if df.empty or "Close" not in df:
                return {"ok": None, "latest": None, "sma200": None, "sma30": None, "error": "no data"}

            close = df["Close"]
            sma200 = close.rolling(200).mean()
            sma30 = close.rolling(30).mean()

            if len(sma200.dropna()) == 0:
                return {"ok": None, "latest": None, "sma200": None, "sma30": None, "error": "insufficient history for SMA200"}

            latest = safe_float_conversion(close.iloc[-1])
            s200 = safe_float_conversion(sma200.iloc[-1])
            s30 = safe_float_conversion(sma30.iloc[-1])

            if latest is None or s200 is None:
                return {"ok": None, "latest": latest, "sma200": s200, "sma30": s30}

            # Trigger if below 200-day SMA with filter
            trigger = latest < s200 * (1.0 - pct_filter)

            return {"ok": trigger, "latest": latest, "sma200": s200, "sma30": s30}

        except Exception as e:
            return {"ok": None, "latest": None, "sma200": None, "sma30": None, "error": str(e)}


class ShortTrendIndicator(AssetTrendIndicator):
    """Asset price short-term trend indicator (30-day SMA)."""

    def __init__(self, ticker: str):
        super().__init__(
            name=f"{ticker} Short Trend (30d)",
            description=f"30-day SMA trend break for {ticker}",
            ticker=ticker
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if Close < SMA30 * (1 - filter)."""
        try:
            pct_filter = config.get("short_trend_filter_pct", 0.015)  # 1.5% filter for 30-day

            df = yahoo_api.get_price_data(self.ticker, period="6mo")  # 6 months for 30-day SMA
            if df.empty or "Close" not in df:
                return {"ok": None, "latest": None, "sma30": None, "error": "no data"}

            close = df["Close"]
            sma30 = close.rolling(30).mean()

            if len(sma30.dropna()) == 0:
                return {"ok": None, "latest": None, "sma30": None, "error": "insufficient history for SMA30"}

            latest = safe_float_conversion(close.iloc[-1])
            s30 = safe_float_conversion(sma30.iloc[-1])

            if latest is None or s30 is None:
                return {"ok": None, "latest": latest, "sma30": s30}

            # Trigger if below 30-day SMA with filter
            trigger = latest < s30 * (1.0 - pct_filter)

            return {"ok": trigger, "latest": latest, "sma30": s30}

        except Exception as e:
            return {"ok": None, "latest": None, "sma30": None, "error": str(e)}


# Factory function to create all macro indicators
def create_macro_indicators() -> Dict[str, MacroIndicator]:
    """Create all macro indicators."""
    return {
        "yc": YieldCurveIndicator(),
        "sahm": SahmRuleIndicator(),
        "hy": HighYieldSpreadIndicator(),
        "nfci": FinancialConditionsIndicator(),
        "permits": BuildingPermitsIndicator(),
        "vix": VIXElevatedIndicator(),
        "sentiment": ConsumerSentimentIndicator(),
        "real_rates": RealRatesIndicator(),
        "inflation": InflationExpectationsIndicator(),
        "oil_vol": OilVolatilityIndicator(),
        "lei": LeadingIndicatorsIndicator(),
        "validator_queue": ValidatorQueueIndicator()
    }


def create_trend_indicators(tickers: list) -> Dict[str, AssetTrendIndicator]:
    """Create both 200-day and 30-day trend indicators for given tickers."""
    indicators = {}

    # Add 200-day trend indicators
    for ticker in tickers:
        indicators[f"{ticker}_200d"] = TrendBreakIndicator(ticker)
        indicators[f"{ticker}_30d"] = ShortTrendIndicator(ticker)

    return indicators