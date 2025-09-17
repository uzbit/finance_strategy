"""
Yahoo Finance API wrapper for real-time and historical market data.
Handles equity, ETF, and crypto price data with caching.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import time


class YahooFinanceAPI:
    """Yahoo Finance client with caching and error handling."""

    # Cache for recently fetched data
    _cache = {}
    _cache_timeout = 300  # 5 minute cache for real-time data

    def __init__(self):
        pass

    def get_price_data(self, ticker: str, period: str = "1y",
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical price data for a ticker.

        Args:
            ticker: Stock/ETF/crypto symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{ticker}_{period}"

        # Check cache
        if use_cache and cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data.copy()

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, auto_adjust=True)

            if df.empty:
                print(f"[warn] No data found for {ticker}")
                return pd.DataFrame()

            # Cache the result
            if use_cache:
                self._cache[cache_key] = (df.copy(), time.time())

            return df

        except Exception as e:
            print(f"[error] Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current/latest price for a ticker."""
        try:
            df = self.get_price_data(ticker, period="1d")
            if df.empty or "Close" not in df:
                return None
            return float(df["Close"].iloc[-1])
        except Exception:
            return None

    def get_intraday_data(self, ticker: str, period: str = "1d",
                         interval: str = "1m") -> pd.DataFrame:
        """
        Get intraday data with specified interval.

        Args:
            ticker: Symbol to fetch
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval,
                             auto_adjust=True)
            return df
        except Exception as e:
            print(f"[error] Failed to fetch intraday data for {ticker}: {e}")
            return pd.DataFrame()

    def get_daily_performance(self, tickers: List[str]) -> Dict[str, float]:
        """Get daily performance for multiple tickers."""
        performance = {}

        for ticker in tickers:
            try:
                df = self.get_price_data(ticker, period="5d")
                if df.empty or len(df) < 2:
                    performance[ticker] = None
                    continue

                # Calculate daily return
                latest_close = df["Close"].iloc[-1]
                prev_close = df["Close"].iloc[-2]
                daily_return = (latest_close - prev_close) / prev_close * 100

                performance[ticker] = float(daily_return)

            except Exception as e:
                print(f"[warn] Failed to get performance for {ticker}: {e}")
                performance[ticker] = None

        return performance

    def calculate_sma(self, ticker: str, period: int = 200,
                     timeframe: str = "1y") -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Simple Moving Average and current price.

        Returns:
            Tuple of (current_price, sma_value)
        """
        try:
            # Fetch enough data for the SMA calculation
            df = self.get_price_data(ticker, period="2y")  # Get extra data for SMA
            if df.empty or "Close" not in df:
                return None, None

            close_prices = df["Close"]
            sma = close_prices.rolling(window=period).mean()

            if len(sma.dropna()) == 0:
                return None, None

            current_price = float(close_prices.iloc[-1])
            sma_value = float(sma.iloc[-1])

            return current_price, sma_value

        except Exception as e:
            print(f"[error] Failed to calculate SMA for {ticker}: {e}")
            return None, None

    def get_volume_data(self, ticker: str, days: int = 20) -> Dict[str, float]:
        """Get volume statistics for a ticker."""
        try:
            df = self.get_price_data(ticker, period="2mo")
            if df.empty or "Volume" not in df:
                return {}

            volumes = df["Volume"].tail(days + 1)  # +1 for current day
            if len(volumes) < 2:
                return {}

            current_volume = float(volumes.iloc[-1])
            avg_volume = float(volumes.iloc[:-1].mean())  # Exclude current day for average

            return {
                "current_volume": current_volume,
                "average_volume": avg_volume,
                "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 0
            }

        except Exception as e:
            print(f"[error] Failed to get volume data for {ticker}: {e}")
            return {}

    def get_etf_info(self, ticker: str) -> Dict[str, Any]:
        """Get ETF information and stats."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "name": info.get("longName", ticker),
                "nav": info.get("navPrice"),
                "total_assets": info.get("totalAssets"),
                "yield": info.get("yield"),
                "expense_ratio": info.get("annualReportExpenseRatio")
            }

        except Exception as e:
            print(f"[error] Failed to get ETF info for {ticker}: {e}")
            return {}

    def check_market_hours(self) -> Dict[str, bool]:
        """Check if major markets are currently open."""
        try:
            # Use SPY as proxy for US market hours
            spy = yf.Ticker("SPY")
            info = spy.info

            # Get current time and market timezone info
            now = datetime.now()
            market_state = info.get("marketState", "UNKNOWN")

            return {
                "us_market_open": market_state in ["REGULAR", "PRE", "POST"],
                "market_state": market_state,
                "timestamp": now.isoformat()
            }

        except Exception as e:
            print(f"[error] Failed to check market hours: {e}")
            return {"us_market_open": None, "market_state": "UNKNOWN", "timestamp": None}

    def clear_cache(self):
        """Clear the price data cache."""
        self._cache.clear()

    @classmethod
    def get_common_tickers(cls) -> Dict[str, str]:
        """Get dictionary of commonly used tickers and descriptions."""
        return {
            # Equity Indices
            "SPY": "SPDR S&P 500 ETF",
            "QQQ": "Invesco QQQ Trust",
            "IWM": "iShares Russell 2000 ETF",
            "VTI": "Vanguard Total Stock Market ETF",

            # Credit ETFs
            "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
            "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
            "TLT": "iShares 20+ Year Treasury Bond ETF",
            "IEF": "iShares 7-10 Year Treasury Bond ETF",

            # Currency
            "UUP": "Invesco DB US Dollar Index Bullish Fund",
            "FXE": "Invesco CurrencyShares Euro Trust",

            # Commodities
            "GLD": "SPDR Gold Shares",
            "SLV": "iShares Silver Trust",
            "USO": "United States Oil Fund",
            "COPX": "Global X Copper Miners ETF",

            # Crypto
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",

            # Individual Stocks (from config)
            "NVDA": "NVIDIA Corporation",
            "SMCI": "Super Micro Computer",
            "TSM": "Taiwan Semiconductor"
        }


# Global instance for easy access
yahoo_api = YahooFinanceAPI()