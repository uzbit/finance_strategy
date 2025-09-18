"""
Advanced technical analysis indicators for trend, momentum, and market structure.
Provides deeper insight into market dynamics beyond simple trend breaks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

from indicators.base import AssetTrendIndicator, safe_float_conversion, calculate_percentage_change
from data.yahoo import yahoo_api


class TrendReclaimIndicator(AssetTrendIndicator):
    """Trend reclaim/loss: Daily close back above/below 200-day SMA."""

    def __init__(self, ticker: str):
        super().__init__(
            name=f"{ticker} Trend Reclaim",
            description=f"Price reclaim/loss of 200-day SMA for {ticker}",
            ticker=ticker
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if price recently crossed 200-day SMA (up or down)."""
        try:
            buffer_pct = config.get("trend_reclaim_buffer_pct", 0.02)
            lookback_days = config.get("trend_reclaim_lookback_days", 5)

            df = yahoo_api.get_price_data(self.ticker, period="1y")
            if df.empty or "Close" not in df:
                return {"ok": None, "signal": None, "latest": None, "sma200": None}

            close = df["Close"]
            sma200 = close.rolling(200).mean()

            if len(sma200.dropna()) < lookback_days:
                return {"ok": None, "signal": None, "latest": None, "sma200": None}

            # Get recent data
            recent_close = close.tail(lookback_days)
            recent_sma = sma200.tail(lookback_days)

            latest_close = safe_float_conversion(close.iloc[-1])
            latest_sma = safe_float_conversion(sma200.iloc[-1])

            if latest_close is None or latest_sma is None:
                return {"ok": None, "signal": None, "latest": latest_close, "sma200": latest_sma}

            # Check for recent crossover with buffer
            upper_band = recent_sma * (1 + buffer_pct)
            lower_band = recent_sma * (1 - buffer_pct)

            # Look for crossovers in recent days
            signal = None
            crossed_recently = False

            for i in range(1, len(recent_close)):
                prev_close = recent_close.iloc[i-1]
                curr_close = recent_close.iloc[i]
                curr_upper = upper_band.iloc[i]
                curr_lower = lower_band.iloc[i]

                # Bullish reclaim: was below, now above
                if prev_close < curr_lower and curr_close > curr_upper:
                    signal = "reclaim"
                    crossed_recently = True
                    break
                # Bearish loss: was above, now below
                elif prev_close > curr_upper and curr_close < curr_lower:
                    signal = "loss"
                    crossed_recently = True
                    break

            return {
                "ok": crossed_recently,
                "signal": signal,
                "latest": latest_close,
                "sma200": latest_sma,
                "buffer_pct": buffer_pct * 100
            }

        except Exception as e:
            return {"ok": None, "signal": None, "latest": None, "sma200": None, "error": str(e)}


class MomentumFlipIndicator(AssetTrendIndicator):
    """Momentum flip: 50-day vs 200-day SMA crossover."""

    def __init__(self, ticker: str):
        super().__init__(
            name=f"{ticker} Momentum Flip",
            description=f"50/200 SMA crossover for {ticker}",
            ticker=ticker
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if 50-day SMA recently crossed 200-day SMA."""
        try:
            lookback_days = config.get("momentum_flip_lookback_days", 10)

            df = yahoo_api.get_price_data(self.ticker, period="1y")
            if df.empty or "Close" not in df:
                return {"ok": None, "signal": None, "sma50": None, "sma200": None}

            close = df["Close"]
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()

            if len(sma50.dropna()) < lookback_days or len(sma200.dropna()) < lookback_days:
                return {"ok": None, "signal": None, "sma50": None, "sma200": None}

            # Get recent data
            recent_50 = sma50.tail(lookback_days)
            recent_200 = sma200.tail(lookback_days)

            latest_50 = safe_float_conversion(sma50.iloc[-1])
            latest_200 = safe_float_conversion(sma200.iloc[-1])

            if latest_50 is None or latest_200 is None:
                return {"ok": None, "signal": None, "sma50": latest_50, "sma200": latest_200}

            # Look for crossover in recent days
            signal = None
            crossed_recently = False

            for i in range(1, len(recent_50)):
                prev_50 = recent_50.iloc[i-1]
                prev_200 = recent_200.iloc[i-1]
                curr_50 = recent_50.iloc[i]
                curr_200 = recent_200.iloc[i]

                # Golden cross: 50 crosses above 200
                if prev_50 <= prev_200 and curr_50 > curr_200:
                    signal = "golden_cross"
                    crossed_recently = True
                    break
                # Death cross: 50 crosses below 200
                elif prev_50 >= prev_200 and curr_50 < curr_200:
                    signal = "death_cross"
                    crossed_recently = True
                    break

            return {
                "ok": crossed_recently,
                "signal": signal,
                "sma50": latest_50,
                "sma200": latest_200,
                "spread": ((latest_50 - latest_200) / latest_200 * 100) if latest_200 else None
            }

        except Exception as e:
            return {"ok": None, "signal": None, "sma50": None, "sma200": None, "error": str(e)}


class VolatilitySqueezeIndicator(AssetTrendIndicator):
    """Squeeze → expansion: Low volatility followed by big move."""

    def __init__(self, ticker: str):
        super().__init__(
            name=f"{ticker} Volatility Squeeze",
            description=f"Bollinger Band squeeze expansion for {ticker}",
            ticker=ticker
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if volatility expanded after squeeze."""
        try:
            bb_period = config.get("bollinger_period", 20)
            bb_std = config.get("bollinger_std", 2.0)
            squeeze_threshold = config.get("squeeze_threshold_pct", 5.0)
            expansion_threshold = config.get("expansion_threshold_pct", 3.0)

            df = yahoo_api.get_price_data(self.ticker, period="3mo")
            if df.empty or "Close" not in df:
                return {"ok": None, "signal": None, "bb_width": None, "volume_spike": None}

            close = df["Close"]
            volume = df.get("Volume", pd.Series())

            if len(close) < bb_period + 10:
                return {"ok": None, "signal": None, "bb_width": None, "volume_spike": None}

            # Calculate Bollinger Bands
            sma = close.rolling(bb_period).mean()
            std = close.rolling(bb_period).std()
            upper_bb = sma + (std * bb_std)
            lower_bb = sma - (std * bb_std)
            bb_width = ((upper_bb - lower_bb) / sma) * 100

            # Calculate recent averages
            recent_width = bb_width.tail(5).mean()
            historical_width = bb_width.tail(50).mean()

            # Check for squeeze (narrow bands)
            is_squeezed = recent_width < squeeze_threshold

            # Check for expansion (recent move)
            if len(close) >= 3:
                recent_move = abs(calculate_percentage_change(close.iloc[-1], close.iloc[-3]))
                expanded = recent_move > expansion_threshold
            else:
                expanded = False

            # Check volume spike
            volume_spike = False
            if not volume.empty and len(volume) >= 20:
                recent_vol = volume.tail(3).mean()
                avg_vol = volume.tail(20).mean()
                volume_spike = recent_vol > avg_vol * 1.5

            # Signal: squeeze followed by expansion
            signal = None
            if is_squeezed and expanded:
                signal = "squeeze_expansion"
            elif is_squeezed:
                signal = "squeeze"
            elif expanded:
                signal = "expansion"

            return {
                "ok": signal == "squeeze_expansion",
                "signal": signal,
                "bb_width": recent_width,
                "expansion_move": recent_move if 'recent_move' in locals() else None,
                "volume_spike": volume_spike,
                "latest": recent_width
            }

        except Exception as e:
            return {"ok": None, "signal": None, "bb_width": None, "volume_spike": None, "error": str(e)}


class MarketBreadthIndicator(AssetTrendIndicator):
    """Breadth: % of assets making new 20-day highs."""

    def __init__(self, tickers: List[str]):
        super().__init__(
            name="Market Breadth",
            description="% of assets making new 20-day highs",
            ticker="BREADTH"  # Special ticker for multi-asset
        )
        self.tickers = tickers

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate breadth across multiple assets."""
        try:
            lookback_days = config.get("breadth_lookback_days", 20)
            healthy_threshold = config.get("breadth_healthy_threshold", 60.0)

            new_highs = 0
            total_assets = 0

            breadth_details = {}

            for ticker in self.tickers:
                try:
                    df = yahoo_api.get_price_data(ticker, period="3mo")
                    if df.empty or "Close" not in df:
                        continue

                    close = df["Close"]
                    if len(close) < lookback_days + 1:
                        continue

                    # Check if making new 20-day high
                    latest_price = close.iloc[-1]
                    period_high = close.tail(lookback_days).max()

                    is_new_high = latest_price >= period_high
                    if is_new_high:
                        new_highs += 1

                    total_assets += 1
                    breadth_details[ticker] = is_new_high

                except Exception:
                    continue

            if total_assets == 0:
                return {"ok": None, "breadth_pct": None, "new_highs": 0, "total": 0}

            breadth_pct = (new_highs / total_assets) * 100

            return {
                "ok": breadth_pct >= healthy_threshold,
                "breadth_pct": breadth_pct,
                "new_highs": new_highs,
                "total": total_assets,
                "details": breadth_details,
                "latest": breadth_pct
            }

        except Exception as e:
            return {"ok": None, "breadth_pct": None, "new_highs": 0, "total": 0, "error": str(e)}


class BTCDominanceIndicator(AssetTrendIndicator):
    """BTC vs Alts: BTC dominance for risk-on/risk-off."""

    def __init__(self):
        super().__init__(
            name="BTC Dominance Signal",
            description="BTC dominance trend for crypto risk sentiment",
            ticker="BTC-USD"
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze BTC vs ETH performance for dominance trend."""
        try:
            # Use BTC vs ETH as proxy for dominance
            btc_df = yahoo_api.get_price_data("BTC-USD", period="1mo")
            eth_df = yahoo_api.get_price_data("ETH-USD", period="1mo")

            if btc_df.empty or eth_df.empty:
                return {"ok": None, "signal": None, "btc_performance": None, "eth_performance": None}

            # Calculate recent performance
            if len(btc_df) < 7 or len(eth_df) < 7:
                return {"ok": None, "signal": None, "btc_performance": None, "eth_performance": None}

            btc_week_change = calculate_percentage_change(btc_df["Close"].iloc[-1], btc_df["Close"].iloc[-7])
            eth_week_change = calculate_percentage_change(eth_df["Close"].iloc[-1], eth_df["Close"].iloc[-7])

            # BTC outperforming = dominance rising
            dominance_rising = btc_week_change > eth_week_change

            # Determine signal based on price direction and dominance
            btc_up = btc_week_change > 0
            eth_up = eth_week_change > 0

            signal = None
            risk_mode = None

            if btc_up and eth_up:
                if dominance_rising:
                    signal = "btc_leading"
                    risk_mode = "risk_on_selective"
                else:
                    signal = "alt_season"
                    risk_mode = "risk_on_broad"
            elif not btc_up and not eth_up:
                if dominance_rising:
                    signal = "flight_to_btc"
                    risk_mode = "risk_off_mild"
                else:
                    signal = "crypto_dump"
                    risk_mode = "risk_off_severe"
            else:
                signal = "mixed"
                risk_mode = "uncertain"

            # Warning if risk-off detected
            warning = risk_mode in ["risk_off_mild", "risk_off_severe"]

            return {
                "ok": warning,
                "signal": signal,
                "risk_mode": risk_mode,
                "btc_performance": btc_week_change,
                "eth_performance": eth_week_change,
                "dominance_rising": dominance_rising,
                "latest": btc_week_change - eth_week_change  # Dominance spread
            }

        except Exception as e:
            return {"ok": None, "signal": None, "btc_performance": None, "eth_performance": None, "error": str(e)}


# Factory function to create technical indicators
def create_technical_indicators(tickers: List[str]) -> Dict[str, AssetTrendIndicator]:
    """Create advanced technical indicators for given tickers."""
    indicators = {}

    # Per-asset technical indicators
    for ticker in tickers:
        indicators[f"{ticker}_reclaim"] = TrendReclaimIndicator(ticker)
        indicators[f"{ticker}_momentum"] = MomentumFlipIndicator(ticker)
        indicators[f"{ticker}_squeeze"] = VolatilitySqueezeIndicator(ticker)

    # Market-wide indicators
    indicators["market_breadth"] = MarketBreadthIndicator(tickers)
    indicators["btc_dominance"] = BTCDominanceIndicator()

    return indicators