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
from data.beacon import BeaconchainAPI
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


@dataclass
class AssetRow:
    """Container for asset-centric display data."""
    asset_name: str
    price: str  # Current price
    trend_200d: str  # Status symbol only
    trend_30d: str
    sma_cross: str
    momentum: str  # Status + metric value
    volatility: str


class RiskDashboard:
    """Main dashboard orchestrator for risk indicators."""

    def __init__(self):
        # Initialize APIs with config keys
        fred_api_key = config.get("fred_api_key", "")
        self.fred_api = FredAPI(fred_api_key)

        coingecko_api_key = config.get("coingecko_api_key", "")
        self.crypto_api = CryptoAPI(coingecko_api_key)

        # Initialize Beaconchain API (no key required)
        self.beacon_api = BeaconchainAPI()

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
            if req.get("source") == "fred" and req.get("series_id"):
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

        # Fetch beacon chain data for validator queue indicator
        try:
            beacon_queue = self.beacon_api.get_validator_queue()
            if beacon_queue:
                macro_data["beacon_queue"] = beacon_queue
        except Exception as e:
            print(f"[warn] Failed to fetch beacon queue data: {e}")

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

        # Create asset-centric table
        assets_table = self._format_asset_centric_table(show_details)

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

    def _group_results_by_asset(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Group trend and technical indicator results by asset ticker."""
        asset_results = {}

        # Process trend indicators
        trend_results = self.last_results.get("trend", {})
        for name, result in trend_results.items():
            # Extract ticker from names like "NVDA_200d", "SMCI_30d"
            if "_200d" in name:
                ticker = name.replace("_200d", "")
                if ticker not in asset_results:
                    asset_results[ticker] = {}
                asset_results[ticker]["trend_200d"] = result
            elif "_30d" in name:
                ticker = name.replace("_30d", "")
                if ticker not in asset_results:
                    asset_results[ticker] = {}
                asset_results[ticker]["trend_30d"] = result

        # Process technical indicators
        technical_results = self.last_results.get("technical", {})
        for name, result in technical_results.items():
            # Extract ticker from names like "NVDA_reclaim", "SMCI_momentum", "BTC-USD_squeeze"
            if "_reclaim" in name:
                ticker = name.replace("_reclaim", "")
                if ticker not in asset_results:
                    asset_results[ticker] = {}
                asset_results[ticker]["sma_cross"] = result
            elif "_momentum" in name:
                ticker = name.replace("_momentum", "")
                if ticker not in asset_results:
                    asset_results[ticker] = {}
                asset_results[ticker]["momentum"] = result
            elif "_squeeze" in name:
                ticker = name.replace("_squeeze", "")
                if ticker not in asset_results:
                    asset_results[ticker] = {}
                asset_results[ticker]["volatility"] = result

        return asset_results

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
            "lei": "<= -2% (6m)",
            "validator_queue": "Entry >80% | Exit >80%"
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

        # For Sahm Rule, show the gap value
        gap = result.get("gap")
        if gap is not None:
            return f"{gap:.2f}"

        # For indicators with pct_change (Building Permits, LEI)
        pct_change = result.get("pct_change")
        if pct_change is not None:
            return f"{pct_change:.1f}%"

        # For validator queue indicator, show both entry and exit percentages
        entry_pct = result.get("entry_pct")
        exit_pct = result.get("exit_pct")
        if entry_pct is not None and exit_pct is not None:
            return f"E:{entry_pct:.1f}% X:{exit_pct:.1f}%"

        # For commodity indicators, check for specific inflation values
        latest = result.get("latest")
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

    def _format_asset_centric_table(self, show_details: bool = False) -> str:
        """Format asset-centric table with indicators as columns."""
        if not self.last_results:
            return "No asset data available."

        # Group results by asset
        asset_results = self._group_results_by_asset()

        if not asset_results:
            return "No asset indicators found."

        # Create asset rows
        asset_rows = []
        for ticker in sorted(asset_results.keys()):
            indicators = asset_results[ticker]

            # Extract price from trend indicator (prefer 200d, fallback to 30d)
            price = self._extract_price_from_result(indicators.get("trend_200d"))
            if price == "—":
                price = self._extract_price_from_result(indicators.get("trend_30d"))

            # Convert each indicator to status + threshold value for trend indicators
            trend_200d = self._result_to_status_with_value(indicators.get("trend_200d"), "trend_200d")
            trend_30d = self._result_to_status_with_value(indicators.get("trend_30d"), "trend_30d")
            sma_cross = self._result_to_status_with_value(indicators.get("sma_cross"), "reclaim")

            # For momentum and volatility, show status + metric value
            momentum = self._result_to_status_with_value(indicators.get("momentum"), "momentum")
            volatility = self._result_to_status_with_value(indicators.get("volatility"), "squeeze")

            asset_rows.append(AssetRow(
                asset_name=ticker,
                price=price,
                trend_200d=trend_200d,
                trend_30d=trend_30d,
                sma_cross=sma_cross,
                momentum=momentum,
                volatility=volatility
            ))

        # Format table with Price column added
        header = "=== ASSET ANALYSIS ==="
        column_headers = ["Asset", "Price", "200d", "30d", "Cross", "Momentum", "Volatility"]

        # Calculate dynamic column widths (wider for threshold values)
        col_widths = [8, 10, 11, 11, 11, 15, 12]
        separator = "-" * (sum(col_widths) + (len(col_widths) - 1) * 3)  # 3 chars for " | "

        table_lines = [
            header,
            " | ".join(f"{h:<{w}}" for h, w in zip(column_headers, col_widths)),
            separator
        ]

        for row in asset_rows:
            line = f"{row.asset_name:<{col_widths[0]}} | {row.price:<{col_widths[1]}} | {row.trend_200d:<{col_widths[2]}} | {row.trend_30d:<{col_widths[3]}} | {row.sma_cross:<{col_widths[4]}} | {row.momentum:<{col_widths[5]}} | {row.volatility:<{col_widths[6]}}"
            table_lines.append(line)

        # Add market-wide indicators
        technical_results = self.last_results.get("technical", {})
        market_wide_lines = [
            "",
            "Market-Wide Indicators:"
        ]

        # Market breadth
        breadth_result = technical_results.get("market_breadth", {})
        breadth_status = self._result_to_status(breadth_result)
        breadth_pct = breadth_result.get("breadth_pct", "N/A")
        market_wide_lines.append(f"• Market Breadth: {breadth_status} ({breadth_pct}% making new 20-day highs)")

        # BTC dominance
        dominance_result = technical_results.get("btc_dominance", {})
        dominance_status = self._result_to_status(dominance_result)
        signal = dominance_result.get("signal", "N/A")
        risk_mode = dominance_result.get("risk_mode", "")
        if signal and risk_mode:
            signal_desc = signal.replace("_", " ").title()
            market_wide_lines.append(f"• BTC Dominance: {dominance_status} ({signal_desc} - {risk_mode.replace('_', ' ')} sentiment)")
        else:
            market_wide_lines.append(f"• BTC Dominance: {dominance_status}")

        # Add column definitions
        definition_lines = [
            "",
            "Column Definitions:",
            f"• 200d: Long-term trend | ⚠️ if Price < (SMA200 × (1 - {config.get('trend_filter_pct', 0.02):.1%}))",
            f"• 30d: Short-term trend | ⚠️ if Price < (SMA30 × (1 - {config.get('short_trend_filter_pct', 0.015):.1%}))",
            f"• Cross: SMA200 crossover | ⚠️ if recent cross detected within ±{config.get('trend_reclaim_buffer_pct', 0.02):.1%}",
            f"• Momentum: 50/200 SMA relationship | ⚠️ if golden/death cross",
            f"• Volatility: Bollinger Band dynamics | ⚠️ if squeeze or expansion"
        ]

        return "\n".join(table_lines + market_wide_lines + definition_lines)

    def _result_to_status_with_value(self, result: Dict[str, Any], indicator_type: str = "") -> str:
        """Convert indicator result to status symbol with current value."""
        if not result:
            return "—"

        # Get status symbol
        ok = result.get("ok")
        if ok is None:
            status = "❓"
        elif ok:
            status = "⚠️"
        else:
            status = "✅"

        # Get current value and format appropriately
        current_value = self._format_indicator_current_value(result, indicator_type)

        if current_value and current_value != "N/A":
            return f"{status} {current_value}"
        else:
            return status

    def _extract_price_from_result(self, result: Dict[str, Any]) -> str:
        """Extract formatted price from indicator result."""
        if not result:
            return "—"

        latest = result.get("latest")
        if latest is not None and isinstance(latest, (int, float)):
            if abs(latest) >= 100:
                return f"${latest:.0f}"
            elif abs(latest) >= 10:
                return f"${latest:.1f}"
            else:
                return f"${latest:.2f}"

        return "—"

    def _result_to_status_only(self, result: Dict[str, Any]) -> str:
        """Convert indicator result to status symbol only (no value)."""
        if not result:
            return "—"

        ok = result.get("ok")
        if ok is None:
            return "❓"
        elif ok:
            return "⚠️"
        else:
            return "✅"

    def _format_indicator_current_value(self, result: Dict[str, Any], indicator_type: str) -> str:
        """Format current value for specific indicator types."""
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

        # For trend indicators, show the threshold value
        if indicator_type == "trend_200d":
            sma200 = result.get("sma200")
            if sma200 is not None:
                threshold = sma200 * (1.0 - config.get("trend_filter_pct", 0.02))
                if abs(threshold) >= 100:
                    return f"${threshold:.0f}"
                elif abs(threshold) >= 10:
                    return f"${threshold:.1f}"
                else:
                    return f"${threshold:.2f}"

        if indicator_type == "trend_30d":
            sma30 = result.get("sma30")
            if sma30 is not None:
                threshold = sma30 * (1.0 - config.get("short_trend_filter_pct", 0.015))
                if abs(threshold) >= 100:
                    return f"${threshold:.0f}"
                elif abs(threshold) >= 10:
                    return f"${threshold:.1f}"
                else:
                    return f"${threshold:.2f}"

        # For reclaim indicator, show the SMA200 value
        if indicator_type == "reclaim":
            sma200 = result.get("sma200")
            if sma200 is not None:
                if abs(sma200) >= 100:
                    return f"${sma200:.0f}"
                elif abs(sma200) >= 10:
                    return f"${sma200:.1f}"
                else:
                    return f"${sma200:.2f}"

        # For momentum indicators, show the spread
        spread = result.get("spread")
        if spread is not None:
            return f"{spread:.2f}%"

        # For squeeze indicators, show BB width
        bb_width = result.get("bb_width")
        if bb_width is not None:
            return f"{bb_width:.1f}%"

        # For trend indicators, show the current price (fallback)
        latest = result.get("latest")
        if latest is not None:
            if isinstance(latest, (int, float)):
                if abs(latest) >= 100:
                    return f"${latest:.0f}"
                elif abs(latest) >= 10:
                    return f"${latest:.1f}"
                else:
                    return f"${latest:.2f}"
            else:
                return str(latest)

        return "N/A"

    def _result_to_status(self, result: Dict[str, Any]) -> str:
        """Convert indicator result to status symbol only (for backward compatibility)."""
        if not result:
            return "—"

        ok = result.get("ok")
        if ok is None:
            return "❓"
        elif ok:
            return "⚠️"
        else:
            return "✅"

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

    def format_html_output(self, output_file: str = "dashboard_output.html") -> str:
        """Generate pretty HTML output and save to file."""
        if not self.last_results:
            return "No data available. Run update_all_indicators() first."

        # Get summary data
        panic_score = self.last_results.get("panic_score", {})
        total_score = panic_score.get("total_score", 0)
        panic_level = panic_score.get("panic_level", "normal")

        macro_warnings = sum(1 for r in self.last_results.get("macro", {}).values() if r.get("ok") is True)
        commodity_warnings = sum(1 for r in self.last_results.get("commodity", {}).values() if r.get("ok") is True)
        trend_warnings = sum(1 for r in self.last_results.get("trend", {}).values() if r.get("ok") is True)

        freshness = self.last_results.get("metadata", {}).get("data_freshness", {})
        market_open = freshness.get("market_open", False)

        # Generate indicators table HTML
        indicator_rows = []
        indicator_rows.extend(self._format_macro_indicators())
        indicator_rows.extend(self._format_commodity_indicators())
        indicator_rows.extend(self._format_panic_indicators())

        indicators_html = self._generate_indicators_table_html(indicator_rows)

        # Generate asset table HTML
        asset_results = self._group_results_by_asset()
        assets_html = self._generate_assets_table_html(asset_results)

        # Build complete HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Risk Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            color: #333;
        }}

        .container {{ max-width: 1600px; margin: 0 auto; }}

        header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        h1 {{ color: #1e3c72; font-size: 2.2em; font-weight: 700; }}

        .timestamp {{ text-align: right; color: #666; }}
        .timestamp .time {{ font-size: 1.1em; font-weight: 600; color: #1e3c72; }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid;
        }}

        .summary-card.normal {{ border-left-color: #48bb78; }}
        .summary-card.warning {{ border-left-color: #f6ad55; }}
        .summary-card.danger {{ border-left-color: #f56565; }}

        .summary-card h3 {{ font-size: 0.9em; text-transform: uppercase; color: #666; margin-bottom: 10px; font-weight: 600; }}
        .summary-card .value {{ font-size: 2em; font-weight: 700; color: #1e3c72; }}
        .summary-card .label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}

        .section {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .section h2 {{ color: #1e3c72; font-size: 1.6em; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }}

        table {{ width: 100%; border-collapse: collapse; font-size: 0.95em; }}
        thead {{ background: #f7fafc; }}
        th {{ padding: 12px; text-align: left; font-weight: 600; color: #4a5568; border-bottom: 2px solid #e2e8f0; }}
        td {{ padding: 12px; border-bottom: 1px solid #f0f0f0; }}
        tr:hover {{ background: #f7fafc; }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-ok {{ background: #c6f6d5; color: #22543d; }}
        .badge-warning {{ background: #feebc8; color: #744210; }}
        .badge-danger {{ background: #fed7d7; color: #742a2a; }}

        .category-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 8px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .category-macro {{ background: #bee3f8; color: #2c5282; }}
        .category-commodity {{ background: #fbd38d; color: #744210; }}
        .category-panic {{ background: #fbb6ce; color: #702459; }}

        .asset-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}

        .asset-card {{
            background: #f7fafc;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }}

        .asset-card h3 {{ color: #1e3c72; margin-bottom: 15px; font-size: 1.2em; }}
        .asset-card .price {{ font-size: 1.5em; font-weight: 700; color: #2d3748; margin-bottom: 10px; }}

        .asset-detail {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.9em;
        }}

        .asset-detail:last-child {{ border-bottom: none; }}
        .asset-detail .label {{ color: #666; font-weight: 600; }}

        .market-wide {{
            background: #edf2f7;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}

        .market-wide h3 {{ color: #1e3c72; margin-bottom: 15px; }}
        .market-wide ul {{ list-style: none; }}
        .market-wide li {{ padding: 8px 0; padding-left: 25px; position: relative; }}
        .market-wide li:before {{ content: "▸"; position: absolute; left: 5px; color: #667eea; font-weight: bold; }}

        .legend {{
            background: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 0.85em;
        }}

        .legend h4 {{ color: #1e3c72; margin-bottom: 10px; }}
        .legend ul {{ list-style: none; }}
        .legend li {{ padding: 5px 0; color: #666; }}

        @media (max-width: 768px) {{
            .summary-cards {{ grid-template-columns: 1fr; }}
            .asset-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>📊 Market Risk Dashboard</h1>
                <p style="color: #666; margin-top: 5px;">Comprehensive Financial Risk Assessment</p>
            </div>
            <div class="timestamp">
                <div style="font-size: 0.9em;">Last Update</div>
                <div class="time">{self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never'}</div>
                <div style="color: {'#48bb78' if market_open else '#666'}; margin-top: 5px;">● Market {'Open' if market_open else 'Closed'}</div>
            </div>
        </header>

        <div class="summary-cards">
            <div class="summary-card {panic_level}">
                <h3>Panic Score</h3>
                <div class="value">{total_score}/{len(self.panic_indicators)}</div>
                <div class="label">{panic_level.upper()}</div>
            </div>
            <div class="summary-card {'warning' if macro_warnings > 0 else 'normal'}">
                <h3>Macro Warnings</h3>
                <div class="value">{macro_warnings}/{len(self.macro_indicators)}</div>
                <div class="label">{int(macro_warnings/len(self.macro_indicators)*100) if len(self.macro_indicators) > 0 else 0}% Indicators Active</div>
            </div>
            <div class="summary-card {'warning' if commodity_warnings > 0 else 'normal'}">
                <h3>Commodity Warnings</h3>
                <div class="value">{commodity_warnings}/{len(self.commodity_indicators)}</div>
                <div class="label">{'All Clear' if commodity_warnings == 0 else 'Some Pressure'}</div>
            </div>
            <div class="summary-card {'warning' if trend_warnings > 0 else 'normal'}">
                <h3>Trend Warnings</h3>
                <div class="value">{trend_warnings}/{len(self.trend_indicators)}</div>
                <div class="label">{'Some Weakness' if trend_warnings > 0 else 'All Strong'}</div>
            </div>
        </div>

        {indicators_html}
        {assets_html}
    </div>
</body>
</html>"""

        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)

        return output_file

    def _generate_indicators_table_html(self, rows: List[IndicatorRow]) -> str:
        """Generate HTML for indicators table."""
        if not rows:
            return ""

        table_rows_html = ""
        for row in rows:
            badge_class = "badge-ok" if row.flag == "OK" else "badge-warning" if row.flag == "WARNING" else "badge-danger"
            category_class = f"category-{row.category.lower()}"

            table_rows_html += f"""
                <tr>
                    <td>{row.name}</td>
                    <td><span class="category-badge {category_class}">{row.category}</span></td>
                    <td>{row.threshold}</td>
                    <td>{row.current_value}</td>
                    <td><span class="badge {badge_class}">{row.flag}</span></td>
                </tr>"""

        return f"""
        <div class="section">
            <h2>🎯 Market Indicators</h2>
            <table>
                <thead>
                    <tr>
                        <th>Indicator</th>
                        <th>Category</th>
                        <th>Threshold</th>
                        <th>Current</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>"""

    def _generate_assets_table_html(self, asset_results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
        """Generate HTML for asset cards."""
        if not asset_results:
            return ""

        asset_cards_html = ""
        for ticker in sorted(asset_results.keys()):
            indicators = asset_results[ticker]

            # Extract price
            price = self._extract_price_from_result(indicators.get("trend_200d"))
            if price == "—":
                price = self._extract_price_from_result(indicators.get("trend_30d"))

            # Get status values
            trend_200d = self._result_to_status_with_value(indicators.get("trend_200d"), "trend_200d")
            trend_30d = self._result_to_status_with_value(indicators.get("trend_30d"), "trend_30d")
            sma_cross = self._result_to_status_with_value(indicators.get("sma_cross"), "reclaim")
            momentum = self._result_to_status_with_value(indicators.get("momentum"), "momentum")
            volatility = self._result_to_status_with_value(indicators.get("volatility"), "squeeze")

            asset_cards_html += f"""
                <div class="asset-card">
                    <h3>{ticker}</h3>
                    <div class="price">{price}</div>
                    <div class="asset-detail">
                        <span class="label">200d:</span>
                        <span>{trend_200d}</span>
                    </div>
                    <div class="asset-detail">
                        <span class="label">30d:</span>
                        <span>{trend_30d}</span>
                    </div>
                    <div class="asset-detail">
                        <span class="label">Cross:</span>
                        <span>{sma_cross}</span>
                    </div>
                    <div class="asset-detail">
                        <span class="label">Momentum:</span>
                        <span>{momentum}</span>
                    </div>
                    <div class="asset-detail">
                        <span class="label">Volatility:</span>
                        <span>{volatility}</span>
                    </div>
                </div>"""

        # Get market-wide indicators
        technical_results = self.last_results.get("technical", {})
        breadth_result = technical_results.get("market_breadth", {})
        breadth_status = self._result_to_status(breadth_result)
        breadth_pct = breadth_result.get("breadth_pct", "N/A")

        dominance_result = technical_results.get("btc_dominance", {})
        dominance_status = self._result_to_status(dominance_result)
        signal = dominance_result.get("signal", "N/A")
        risk_mode = dominance_result.get("risk_mode", "")

        signal_desc = signal.replace("_", " ").title() if signal else "N/A"
        dominance_text = f"{signal_desc} - {risk_mode.replace('_', ' ')} sentiment" if signal and risk_mode else signal_desc

        return f"""
        <div class="section">
            <h2>💰 Asset Analysis</h2>
            <div class="asset-grid">
                {asset_cards_html}
            </div>

            <div class="market-wide">
                <h3>Market-Wide Indicators</h3>
                <ul>
                    <li>Market Breadth: {breadth_status} ({breadth_pct}% making new 20-day highs)</li>
                    <li>BTC Dominance: {dominance_status} ({dominance_text})</li>
                </ul>
            </div>

            <div class="legend">
                <h4>Column Definitions</h4>
                <ul>
                    <li><strong>200d:</strong> Long-term trend | ⚠️ if Price &lt; (SMA200 × (1 - {config.get('trend_filter_pct', 0.02):.1%}))</li>
                    <li><strong>30d:</strong> Short-term trend | ⚠️ if Price &lt; (SMA30 × (1 - {config.get('short_trend_filter_pct', 0.015):.1%}))</li>
                    <li><strong>Cross:</strong> SMA200 crossover | ⚠️ if recent cross detected within ±{config.get('trend_reclaim_buffer_pct', 0.02):.1%}</li>
                    <li><strong>Momentum:</strong> 50/200 SMA relationship | ⚠️ if golden/death cross</li>
                    <li><strong>Volatility:</strong> Bollinger Band dynamics | ⚠️ if squeeze or expansion</li>
                </ul>
            </div>
        </div>"""


# Global dashboard instance
dashboard = RiskDashboard()