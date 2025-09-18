"""
Dashboard orchestrator for coordinating indicators and output formatting.
Handles both macro and panic indicators with unified table output.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from core.config import config
from indicators.macro import create_macro_indicators, create_trend_indicators
from indicators.panic import create_panic_indicators, calculate_panic_score
from indicators.commodity import create_commodity_indicators, get_required_series
from indicators.technical import create_technical_indicators
from data.fred import FredAPI
from data.crypto import CryptoAPI
from data.realtime import RealTimeDataAggregator


@dataclass
class IndicatorRow:
    """Container for a single indicator's display data."""
    name: str
    category: str
    threshold: str
    current_value: str
    status: str
    flag: str
    details: str = ""


class RiskDashboard:
    """Main dashboard orchestrator for risk indicators."""

    def __init__(self):
        # Initialize APIs with config keys
        fred_api_key = config.get("fred_api_key", "")
        self.fred_api = FredAPI(fred_api_key)

        coingecko_api_key = config.get("coingecko_api_key", "")
        self.crypto_api = CryptoAPI(coingecko_api_key)

        # Initialize realtime data aggregator with proper APIs
        self.realtime_data = RealTimeDataAggregator(self.crypto_api, self.fred_api)

        self.macro_indicators = create_macro_indicators()
        self.panic_indicators = create_panic_indicators()
        self.commodity_indicators = create_commodity_indicators()
        # Always initialize trend indicators with config tickers
        default_tickers = config.get("tickers", ["SPY", "QQQ", "IWM"])
        self.trend_indicators = create_trend_indicators(default_tickers)
        self.technical_indicators = create_technical_indicators(default_tickers)
        self.last_update = None
        self.last_results = {}

    def initialize_trend_indicators(self, tickers: List[str] = None):
        """Initialize trend indicators for specified tickers."""
        if tickers is None:
            tickers = config.get("trend_tickers", ["SPY", "QQQ", "IWM"])

        self.trend_indicators = create_trend_indicators(tickers)
        self.technical_indicators = create_technical_indicators(tickers)

    def update_all_indicators(self) -> Dict[str, Any]:
        """Update all indicators with fresh data."""
        results = {
            "macro": {},
            "panic": {},
            "commodity": {},
            "trend": {},
            "technical": {},
            "metadata": {
                "update_time": datetime.now().isoformat(),
                "data_freshness": self.realtime_data.check_data_freshness()
            }
        }

        # Update macro indicators
        macro_data = self._fetch_macro_data()
        for name, indicator in self.macro_indicators.items():
            try:
                result = indicator.update(macro_data, config.data)
                results["macro"][name] = result
            except Exception as e:
                print(f"[warn] Failed to update macro indicator {name}: {e}")
                results["macro"][name] = {"ok": None, "error": str(e)}

        # Update commodity indicators
        for name, indicator in self.commodity_indicators.items():
            try:
                result = indicator.update(macro_data, config.data)  # Use same data as macro
                results["commodity"][name] = result
            except Exception as e:
                print(f"[warn] Failed to update commodity indicator {name}: {e}")
                results["commodity"][name] = {"ok": None, "error": str(e)}

        # Update panic indicators
        for name, indicator in self.panic_indicators.items():
            try:
                # Pass realtime data to panic indicators
                panic_data = {"realtime_data": self.realtime_data}
                result = indicator.update(panic_data, config.data)
                results["panic"][name] = result
            except Exception as e:
                print(f"[warn] Failed to update panic indicator {name}: {e}")
                results["panic"][name] = {"ok": None, "error": str(e)}

        # Update trend indicators
        for name, indicator in self.trend_indicators.items():
            try:
                result = indicator.update({}, config.data)
                results["trend"][name] = result
            except Exception as e:
                print(f"[warn] Failed to update trend indicator {name}: {e}")
                results["trend"][name] = {"ok": None, "error": str(e)}

        # Update technical indicators
        for name, indicator in self.technical_indicators.items():
            try:
                result = indicator.update({}, config.data)
                results["technical"][name] = result
            except Exception as e:
                print(f"[warn] Failed to update technical indicator {name}: {e}")
                results["technical"][name] = {"ok": None, "error": str(e)}

        # Calculate panic score
        results["panic_score"] = calculate_panic_score(self.panic_indicators, config.data)

        self.last_results = results
        self.last_update = datetime.now()
        return results

    def _fetch_macro_data(self) -> Dict[str, Any]:
        """Fetch data for all macro and commodity indicators."""
        macro_data = {}
        series_ids = set()

        # Collect all required series IDs from macro indicators
        for indicator in self.macro_indicators.values():
            req = indicator.get_data_requirements()
            if req.get("source") == "fred":
                series_ids.add(req["series_id"])

        # Add commodity series IDs
        commodity_series = get_required_series()
        series_ids.update(commodity_series)

        # Fetch data for all series
        start_date = config.get("data_start_date", "2015-01-01")
        for series_id in series_ids:
            try:
                df = self.fred_api.fetch_series(series_id, start_date)
                if not df.empty:
                    macro_data[series_id] = df
            except Exception as e:
                print(f"[warn] Failed to fetch {series_id}: {e}")

        return macro_data

    def format_table_output(self, show_details: bool = False) -> str:
        """Format dashboard output with separate tables for indicators and assets."""
        if not self.last_results:
            return "No data available. Run update_all_indicators() first."

        # Create indicators table (macro, commodity, panic)
        indicator_rows = []
        indicator_rows.extend(self._format_macro_indicators())
        indicator_rows.extend(self._format_commodity_indicators())
        indicator_rows.extend(self._format_panic_indicators())

        indicators_table = self._create_formatted_table(indicator_rows, show_details, "MARKET INDICATORS")

        # Create assets table (trend, technical)
        asset_rows = []
        if self.trend_indicators:
            asset_rows.extend(self._format_trend_indicators())
        if self.technical_indicators:
            asset_rows.extend(self._format_technical_indicators())

        assets_table = self._create_formatted_table(asset_rows, show_details, "ASSET ANALYSIS")

        # Add summary
        summary = self._create_summary()

        return f"{indicators_table}\n\n{assets_table}\n\n{summary}"

    def _format_macro_indicators(self) -> List[IndicatorRow]:
        """Format macro indicators for table display."""
        rows = []
        macro_results = self.last_results.get("macro", {})

        for name, indicator in self.macro_indicators.items():
            result = macro_results.get(name, {})
            rows.append(self._format_indicator_row(
                indicator, result, "Macro", name
            ))

        return rows

    def _format_commodity_indicators(self) -> List[IndicatorRow]:
        """Format commodity indicators for table display."""
        rows = []
        commodity_results = self.last_results.get("commodity", {})

        for name, indicator in self.commodity_indicators.items():
            result = commodity_results.get(name, {})
            rows.append(self._format_indicator_row(
                indicator, result, "Commodity", name
            ))

        return rows

    def _format_panic_indicators(self) -> List[IndicatorRow]:
        """Format panic indicators for table display."""
        rows = []
        panic_results = self.last_results.get("panic", {})

        for name, indicator in self.panic_indicators.items():
            result = panic_results.get(name, {})
            rows.append(self._format_indicator_row(
                indicator, result, "Panic", name
            ))

        return rows

    def _format_trend_indicators(self) -> List[IndicatorRow]:
        """Format trend indicators for table display."""
        rows = []
        trend_results = self.last_results.get("trend", {})

        for name, indicator in self.trend_indicators.items():
            result = trend_results.get(name, {})
            rows.append(self._format_indicator_row(
                indicator, result, "Trend", name
            ))

        return rows

    def _format_technical_indicators(self) -> List[IndicatorRow]:
        """Format technical indicators for table display."""
        rows = []
        technical_results = self.last_results.get("technical", {})

        for name, indicator in self.technical_indicators.items():
            result = technical_results.get(name, {})
            rows.append(self._format_indicator_row(
                indicator, result, "Technical", name
            ))

        return rows

    def _format_indicator_row(self, indicator, result: Dict[str, Any],
                            category: str, key: str) -> IndicatorRow:
        """Format a single indicator into a table row."""
        # Determine status
        ok = result.get("ok")
        if ok is None:
            status = "❓"
            flag = "NO DATA"
        elif ok:
            status = "⚠️"
            flag = "WARNING"
        else:
            status = "✅"
            flag = "OK"

        # Format threshold
        threshold = self._format_threshold(indicator, result, key)

        # Format current value
        current_value = self._format_current_value(result)

        # Format details
        details = self._format_details(result)

        return IndicatorRow(
            name=indicator.name,
            category=category,
            threshold=threshold,
            current_value=current_value,
            status=status,
            flag=flag,
            details=details
        )

    def _format_threshold(self, indicator, result: Dict[str, Any], key: str) -> str:
        """Format threshold value for display."""
        # Check if threshold is in result
        threshold = result.get("threshold")
        if threshold is not None:
            return str(threshold)

        # Look up threshold in config for panic indicators
        if hasattr(indicator, 'category') and indicator.category:
            panic_thresholds = config.get("panic_thresholds", {})
            threshold_key = self._get_threshold_key(key)
            threshold = panic_thresholds.get(threshold_key)
            if threshold is not None:
                return str(threshold)

        # For commodity indicators, check specific config values
        commodity_thresholds = {
            "energy_inflation": "energy_inflation_threshold_pct",
            "natgas_inflation": "natgas_inflation_threshold_pct",
            "food_inflation": "food_inflation_threshold_pct",
            "metals_inflation": "metals_inflation_threshold_pct",
            "import_inflation": "import_inflation_threshold_pct",
            "lumber_inflation": "lumber_inflation_threshold_pct",
            "composite_inflation": "composite_inflation_threshold_pct"
        }

        if key in commodity_thresholds:
            threshold = config.get(commodity_thresholds[key])
            if threshold is not None:
                return f"{threshold}%"

        # Default descriptions for macro indicators
        threshold_descriptions = {
            "yc": "< 0",
            "sahm": ">= 0.5",
            "hy": "Recent min + 1.0",
            "nfci": "> 0",
            "permits": "<= 0% (6m)",
            "vix": "> 25",
            "sentiment": "< 80",
            "real_rates": "< 0",
            "inflation": "> 3.0",
            "oil_vol": "> 20%",
            "lei": "<= -2% (6m)"
        }

        # Handle technical indicators specifically first
        if "reclaim" in key.lower():
            buffer_pct = result.get("buffer_pct", 2.0)
            return f"SMA200 cross ±{buffer_pct}%"
        elif "momentum" in key.lower():
            return "50/200 SMA cross"
        elif "squeeze" in key.lower():
            squeeze_thresh = config.get("squeeze_threshold_pct", 5.0)
            expansion_thresh = config.get("expansion_threshold_pct", 3.0)
            return f"BB <{squeeze_thresh}% → >{expansion_thresh}%"
        elif "breadth" in key.lower():
            threshold = config.get("breadth_healthy_threshold", 60.0)
            return f">= {threshold}% new highs"
        elif "dominance" in key.lower():
            return "BTC vs ETH trend"

        # For trend indicators, show the actual SMA threshold value
        if hasattr(indicator, 'ticker') and ("30d" in indicator.name or "200d" in indicator.name):
            # Check if it's a 30-day or 200-day indicator
            if "30d" in indicator.name:
                sma_value = result.get("sma30")
                filter_pct = config.get("short_trend_filter_pct", 0.015)
                if sma_value is not None:
                    threshold_value = sma_value * (1.0 - filter_pct)
                    return f"< {threshold_value:.1f}"
                else:
                    return "< SMA30"
            else:
                sma_value = result.get("sma200")
                filter_pct = config.get("trend_filter_pct", 0.02)
                if sma_value is not None:
                    threshold_value = sma_value * (1.0 - filter_pct)
                    return f"< {threshold_value:.1f}"
                else:
                    return "< SMA200"

        return threshold_descriptions.get(key, "N/A")

    def _get_threshold_key(self, indicator_key: str) -> str:
        """Map indicator key to config threshold key."""
        mapping = {
            "vix_spike": "vix_spike",
            "equity_stress": "spy_intraday_drop",
            "volume_stress": "volume_spike",
            "credit_stress": "hyg_drop",
            "flight_quality": "tlt_rally",
            "dollar_strength": "dollar_rally",
            "crypto_stress": "btc_drop",
            "commodity_stress": "oil_volatility"
        }
        return mapping.get(indicator_key, indicator_key)

    def _format_current_value(self, result: Dict[str, Any]) -> str:
        """Format current value for display."""
        latest = result.get("latest")

        # Handle technical indicators specifically
        signal = result.get("signal")
        if signal:
            # For reclaim/momentum indicators, show the signal
            if signal in ["reclaim", "loss", "golden_cross", "death_cross"]:
                return signal.replace("_", " ").title()
            elif signal in ["squeeze_expansion", "squeeze", "expansion"]:
                return signal.replace("_", " ").title()
            # For BTC dominance
            elif signal in ["btc_leading", "alt_season", "flight_to_btc", "crypto_dump", "mixed"]:
                return signal.replace("_", " ").title()

        # For breadth indicator, show the percentage
        breadth_pct = result.get("breadth_pct")
        if breadth_pct is not None:
            return f"{breadth_pct:.1f}%"

        # For momentum indicators, show the spread
        spread = result.get("spread")
        if spread is not None:
            return f"{spread:.2f}%"

        # For squeeze indicators, show BB width
        bb_width = result.get("bb_width")
        if bb_width is not None:
            return f"{bb_width:.1f}%"

        # For BTC dominance, show the performance spread
        btc_performance = result.get("btc_performance")
        eth_performance = result.get("eth_performance")
        if btc_performance is not None and eth_performance is not None:
            spread = btc_performance - eth_performance
            return f"{spread:.2f}%"

        # For commodity indicators, check for specific inflation values
        if latest is None:
            # Check for max inflation values in commodity indicators
            max_food = result.get("max_food_inflation")
            max_metals = result.get("max_metals_inflation")
            max_import = result.get("max_import_inflation")  # Legacy fallback
            composite_pressure = result.get("composite_pressure")  # New import indicator
            composite_score = result.get("composite_score")

            if max_food is not None:
                return f"{max_food:.1f}%"
            elif max_metals is not None:
                return f"{max_metals:.1f}%"
            elif composite_pressure is not None:
                return f"{composite_pressure:.1f}%"
            elif max_import is not None:
                return f"{max_import:.1f}%"
            elif composite_score is not None:
                return f"{composite_score:.1f}%"
            else:
                return "N/A"

        if isinstance(latest, bool):
            return "Yes" if latest else "No"

        if isinstance(latest, (int, float)):
            if abs(latest) >= 100:
                return f"{latest:.0f}"
            elif abs(latest) >= 10:
                return f"{latest:.1f}"
            else:
                return f"{latest:.2f}"

        return str(latest)

    def _format_details(self, result: Dict[str, Any]) -> str:
        """Format additional details for display."""
        details = []

        # Add specific details based on available data
        if "gap" in result and result["gap"] is not None:
            details.append(f"Gap: {result['gap']:.2f}")

        if "delta" in result and result["delta"] is not None:
            details.append(f"Delta: {result['delta']:.2f}")

        if "pct_change" in result and result["pct_change"] is not None:
            details.append(f"Change: {result['pct_change']:.1f}%")

        if "volatility" in result and result["volatility"] is not None:
            details.append(f"Vol: {result['volatility']:.1f}%")

        if "depegged_coins" in result:
            coins = result["depegged_coins"]
            if coins:
                details.append(f"Depegged: {', '.join(coins)}")

        # Add commodity-specific details
        if "corn_yoy" in result and result["corn_yoy"] is not None:
            details.append(f"Corn: {result['corn_yoy']:.1f}%")
        if "wheat_yoy" in result and result["wheat_yoy"] is not None:
            details.append(f"Wheat: {result['wheat_yoy']:.1f}%")
        if "steel_yoy" in result and result["steel_yoy"] is not None:
            details.append(f"Steel: {result['steel_yoy']:.1f}%")

        return " | ".join(details) if details else ""

    def _create_formatted_table(self, rows: List[IndicatorRow], show_details: bool, title: str = "") -> str:
        """Create formatted table from indicator rows."""
        if not rows:
            return f"=== {title} ===\nNo indicators to display."

        # Calculate column widths
        max_name = max(len(row.name) for row in rows)
        max_cat = max(len(row.category) for row in rows)
        max_thresh = max(len(row.threshold) for row in rows)
        max_value = max(len(row.current_value) for row in rows)
        max_flag = max(len(row.flag) for row in rows)

        # Ensure minimum widths
        name_width = max(max_name, 25)
        cat_width = max(max_cat, 8)
        thresh_width = max(max_thresh, 12)
        value_width = max(max_value, 10)
        flag_width = max(max_flag, 8)

        # Create header
        header = (
            f"{'Indicator':<{name_width}} | "
            f"{'Category':<{cat_width}} | "
            f"{'Threshold':<{thresh_width}} | "
            f"{'Current':<{value_width}} | "
            f"{'St':<3} | "
            f"{'Flag':<{flag_width}}"
        )

        if show_details:
            header += " | Details"

        separator = "-" * len(header.split(" | Details")[0])
        if show_details:
            separator += "-" * 20

        # Create table with title
        table_rows = []
        if title:
            table_rows.append(f"=== {title} ===")
        table_rows.extend([header, separator])

        for row in rows:
            table_row = (
                f"{row.name:<{name_width}} | "
                f"{row.category:<{cat_width}} | "
                f"{row.threshold:<{thresh_width}} | "
                f"{row.current_value:<{value_width}} | "
                f"{row.status:<3} | "
                f"{row.flag:<{flag_width}}"
            )

            if show_details and row.details:
                table_row += f" | {row.details}"

            table_rows.append(table_row)

        return "\n".join(table_rows)

    def _create_summary(self) -> str:
        """Create summary section."""
        if not self.last_results:
            return ""

        panic_score = self.last_results.get("panic_score", {})
        total_score = panic_score.get("total_score", 0)
        panic_level = panic_score.get("panic_level", "normal")
        active_indicators = panic_score.get("active_indicators", [])

        # Count warning indicators by category
        macro_warnings = sum(
            1 for result in self.last_results.get("macro", {}).values()
            if result.get("ok") is True
        )

        commodity_warnings = sum(
            1 for result in self.last_results.get("commodity", {}).values()
            if result.get("ok") is True
        )

        panic_warnings = sum(
            1 for result in self.last_results.get("panic", {}).values()
            if result.get("ok") is True
        )

        trend_warnings = sum(
            1 for result in self.last_results.get("trend", {}).values()
            if result.get("ok") is True
        )

        # Data freshness
        freshness = self.last_results.get("metadata", {}).get("data_freshness", {})
        market_open = freshness.get("market_open", False)
        snapshot_fresh = freshness.get("snapshot_fresh", False)

        summary_lines = [
            "=== SUMMARY ===",
            f"Macro Warnings: {macro_warnings}/{len(self.macro_indicators)}",
            f"Commodity Warnings: {commodity_warnings}/{len(self.commodity_indicators)}",
            f"Panic Score: {total_score}/{len(self.panic_indicators)} ({panic_level.upper()})",
            f"Trend Warnings: {trend_warnings}/{len(self.trend_indicators)}",
            f"Market Open: {'Yes' if market_open else 'No'}",
            f"Data Fresh: {'Yes' if snapshot_fresh else 'No'}",
            f"Last Update: {self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never'}"
        ]

        if active_indicators:
            summary_lines.append(f"Active Panic: {', '.join(active_indicators)}")

        return "\n".join(summary_lines)

    def get_warning_count(self) -> Dict[str, int]:
        """Get count of warnings by category."""
        if not self.last_results:
            return {}

        return {
            "macro": sum(
                1 for result in self.last_results.get("macro", {}).values()
                if result.get("ok") is True
            ),
            "panic": sum(
                1 for result in self.last_results.get("panic", {}).values()
                if result.get("ok") is True
            ),
            "trend": sum(
                1 for result in self.last_results.get("trend", {}).values()
                if result.get("ok") is True
            )
        }

    def get_panic_level(self) -> str:
        """Get current panic level."""
        if not self.last_results:
            return "unknown"

        panic_score = self.last_results.get("panic_score", {})
        return panic_score.get("panic_level", "normal")

    def export_results(self) -> Dict[str, Any]:
        """Export results for external use."""
        return {
            "results": self.last_results,
            "warning_counts": self.get_warning_count(),
            "panic_level": self.get_panic_level(),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }


# Global dashboard instance
dashboard = RiskDashboard()