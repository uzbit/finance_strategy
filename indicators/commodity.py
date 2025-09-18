"""
Commodity-based inflation indicators using US government APIs.
Tracks energy, food, metals, and import price pressures for inflation nowcasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from indicators.base import MacroIndicator, safe_float_conversion, calculate_percentage_change
# FRED API is now injected via data parameter in calculate() methods


class EnergyInflationIndicator(MacroIndicator):
    """Energy inflation pressure via gasoline and diesel prices."""

    def __init__(self):
        super().__init__(
            name="Energy Inflation Pressure",
            description="YoY retail gasoline & diesel price changes",
            series_id="GASREGCOVW"  # Weekly retail gasoline prices
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if energy prices up >threshold% YoY."""
        gas_df = data.get("GASREGCOVW")  # Weekly retail gasoline
        diesel_df = data.get("GASDESW")  # Weekly retail diesel

        threshold_pct = config.get("energy_inflation_threshold_pct", 15.0)

        if gas_df is None or gas_df.empty:
            return {"ok": None, "gas_yoy": None, "latest": None}

        # Calculate YoY change in gasoline prices
        gas_prices = gas_df["value"].dropna()
        if len(gas_prices) < 52:  # Need at least 1 year of weekly data
            return {"ok": None, "gas_yoy": None, "latest": None}

        latest_gas = safe_float_conversion(gas_prices.iloc[-1])
        year_ago_gas = safe_float_conversion(gas_prices.iloc[-52])

        if latest_gas is None or year_ago_gas is None:
            return {"ok": None, "gas_yoy": None, "latest": latest_gas}

        gas_yoy = calculate_percentage_change(latest_gas, year_ago_gas)

        # Also check diesel if available
        diesel_yoy = None
        if diesel_df is not None and not diesel_df.empty and len(diesel_df) >= 52:
            diesel_prices = diesel_df["value"].dropna()
            latest_diesel = safe_float_conversion(diesel_prices.iloc[-1])
            year_ago_diesel = safe_float_conversion(diesel_prices.iloc[-52])
            if latest_diesel is not None and year_ago_diesel is not None:
                diesel_yoy = calculate_percentage_change(latest_diesel, year_ago_diesel)

        # Use higher of gas/diesel inflation
        max_energy_inflation = gas_yoy
        if diesel_yoy is not None:
            max_energy_inflation = max(gas_yoy, diesel_yoy)

        return {
            "ok": max_energy_inflation >= threshold_pct,
            "gas_yoy": gas_yoy,
            "diesel_yoy": diesel_yoy,
            "max_inflation": max_energy_inflation,
            "latest": latest_gas,
            "threshold": threshold_pct
        }


class NaturalGasInflationIndicator(MacroIndicator):
    """Natural gas price momentum indicator."""

    def __init__(self):
        super().__init__(
            name="Natural Gas Inflation",
            description="Henry Hub natural gas price momentum",
            series_id="DHHNGSP"  # Daily Henry Hub natural gas spot price
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if nat gas prices up >threshold% (3-month annualized)."""
        df = data.get("DHHNGSP")
        if df is None or df.empty:
            return {"ok": None, "annualized_change": None, "latest": None}

        threshold_pct = config.get("natgas_inflation_threshold_pct", 25.0)
        lookback_days = config.get("natgas_lookback_days", 90)  # 3 months

        prices = df["value"].dropna()
        if len(prices) < lookback_days:
            return {"ok": None, "annualized_change": None, "latest": None}

        latest = safe_float_conversion(prices.iloc[-1])
        past = safe_float_conversion(prices.iloc[-lookback_days])

        if latest is None or past is None:
            return {"ok": None, "annualized_change": None, "latest": latest}

        # Calculate 3-month change and annualize
        period_change = calculate_percentage_change(latest, past)
        annualized_change = (1 + period_change/100) ** (365/lookback_days) - 1
        annualized_change *= 100

        return {
            "ok": annualized_change >= threshold_pct,
            "annualized_change": annualized_change,
            "period_change": period_change,
            "latest": latest,
            "threshold": threshold_pct
        }


class FoodInflationIndicator(MacroIndicator):
    """Food price inflation via key agricultural commodities."""

    def __init__(self):
        super().__init__(
            name="Food Inflation Pressure",
            description="Agricultural commodity price inflation",
            series_id="PMAIZMTUSDM"  # Global corn prices
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if food commodity prices up >threshold% YoY."""
        # Check multiple food commodity series
        corn_df = data.get("PMAIZMTUSDM")   # Corn (IMF Global Price)
        wheat_df = data.get("PWHEAMTUSDM")  # Wheat (IMF Global Price)
        soy_df = data.get("PSOYBUSDM")      # Soybeans (IMF Global Price)

        threshold_pct = config.get("food_inflation_threshold_pct", 20.0)

        food_changes = []
        commodity_data = {}

        # Calculate YoY changes for available commodities (monthly data)
        for name, df in [("corn", corn_df), ("wheat", wheat_df), ("soy", soy_df)]:
            if df is not None and not df.empty and len(df) >= 12:  # ~1 year monthly
                prices = df["value"].dropna()
                latest = safe_float_conversion(prices.iloc[-1])
                year_ago = safe_float_conversion(prices.iloc[-12] if len(prices) >= 12 else prices.iloc[0])

                if latest is not None and year_ago is not None:
                    yoy_change = calculate_percentage_change(latest, year_ago)
                    food_changes.append(yoy_change)
                    commodity_data[f"{name}_yoy"] = yoy_change
                    commodity_data[f"{name}_latest"] = latest

        if not food_changes:
            return {"ok": None, "max_food_inflation": None, "avg_food_inflation": None}

        max_food_inflation = max(food_changes)
        avg_food_inflation = sum(food_changes) / len(food_changes)

        result = {
            "ok": max_food_inflation >= threshold_pct,
            "max_food_inflation": max_food_inflation,
            "avg_food_inflation": avg_food_inflation,
            "threshold": threshold_pct
        }
        result.update(commodity_data)

        return result


class MetalsInflationIndicator(MacroIndicator):
    """Industrial metals inflation via PPI data."""

    def __init__(self):
        super().__init__(
            name="Metals Inflation Pressure",
            description="PPI for steel and industrial metals",
            series_id="WPU101"  # PPI for steel mill products
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if metals PPI up >threshold% YoY."""
        steel_df = data.get("WPU101")  # Steel mill products PPI
        chemicals_df = data.get("WPU06")  # Industrial chemicals PPI

        threshold_pct = config.get("metals_inflation_threshold_pct", 12.0)

        metals_changes = []
        metals_data = {}

        # Calculate YoY changes for available series
        for name, df in [("steel", steel_df), ("chemicals", chemicals_df)]:
            if df is not None and not df.empty and len(df) >= 12:  # 12 months
                values = df["value"].dropna()
                latest = safe_float_conversion(values.iloc[-1])
                year_ago = safe_float_conversion(values.iloc[-12] if len(values) >= 12 else values.iloc[0])

                if latest is not None and year_ago is not None:
                    yoy_change = calculate_percentage_change(latest, year_ago)
                    metals_changes.append(yoy_change)
                    metals_data[f"{name}_yoy"] = yoy_change
                    metals_data[f"{name}_latest"] = latest

        if not metals_changes:
            return {"ok": None, "max_metals_inflation": None}

        max_metals_inflation = max(metals_changes)

        result = {
            "ok": max_metals_inflation >= threshold_pct,
            "max_metals_inflation": max_metals_inflation,
            "threshold": threshold_pct
        }
        result.update(metals_data)

        return result


class ImportInflationIndicator(MacroIndicator):
    """Import price inflation pressure using BLS Import Price Indexes."""

    def __init__(self):
        super().__init__(
            name="Import Inflation Pressure",
            description="Core + Energy weighted import price inflation",
            series_id="IR"  # All imports (headline)
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate import inflation pressure using Core + Energy methodology."""
        # Get the data series
        headline_df = data.get("IR")           # All imports (headline)
        core_df = data.get("IREXFDFLS")        # Ex food & fuels (preferred)
        alt_core_df = data.get("IREXFUELS")    # Ex fuels (fallback)
        energy_df = data.get("IR10")           # Fuels & lubricants
        usd_df = data.get("DTWEXBGS")          # Broad USD index (optional)

        # Configuration
        threshold_pct = config.get("import_inflation_threshold_pct", 10.0)
        amber_threshold = config.get("import_inflation_amber_threshold", 2.0)
        red_threshold = config.get("import_inflation_red_threshold", 4.0)
        fast_threshold = config.get("import_inflation_3m_threshold", 6.0)

        # Weights for Core + Energy mix
        core_weight = config.get("import_inflation_core_weight", 0.75)
        energy_weight = config.get("import_inflation_energy_weight", 0.25)

        # FX adjustment (if enabled)
        fx_adjustment = config.get("import_inflation_fx_adjustment", False)
        fx_beta = config.get("import_inflation_fx_beta", 0.4)

        import_data = {}

        # Calculate headline YoY for simple fallback
        headline_yoy = None
        if headline_df is not None and not headline_df.empty and len(headline_df) >= 12:
            values = headline_df["value"].dropna()
            if len(values) >= 12:
                latest = safe_float_conversion(values.iloc[-1])
                year_ago = safe_float_conversion(values.iloc[-12])
                if latest is not None and year_ago is not None:
                    headline_yoy = calculate_percentage_change(latest, year_ago)
                    import_data["headline_yoy"] = headline_yoy

        # Calculate core inflation (prefer ex-food&fuels, fallback to ex-fuels)
        core_yoy = None
        core_source = None
        for source_name, df in [("ex_food_fuels", core_df), ("ex_fuels", alt_core_df)]:
            if df is not None and not df.empty and len(df) >= 12:
                values = df["value"].dropna()
                if len(values) >= 12:
                    latest = safe_float_conversion(values.iloc[-1])
                    year_ago = safe_float_conversion(values.iloc[-12])
                    if latest is not None and year_ago is not None:
                        core_yoy = calculate_percentage_change(latest, year_ago)
                        core_source = source_name
                        import_data[f"core_yoy_{source_name}"] = core_yoy
                        break

        # Calculate energy inflation
        energy_yoy = None
        if energy_df is not None and not energy_df.empty and len(energy_df) >= 12:
            values = energy_df["value"].dropna()
            if len(values) >= 12:
                latest = safe_float_conversion(values.iloc[-1])
                year_ago = safe_float_conversion(values.iloc[-12])
                if latest is not None and year_ago is not None:
                    energy_yoy = calculate_percentage_change(latest, year_ago)
                    import_data["energy_yoy"] = energy_yoy

        # Calculate 3-month annualized for headline (turning indicator)
        headline_3m_ann = None
        if headline_df is not None and not headline_df.empty and len(headline_df) >= 3:
            values = headline_df["value"].dropna()
            if len(values) >= 3:
                latest = safe_float_conversion(values.iloc[-1])
                three_months_ago = safe_float_conversion(values.iloc[-3])
                if latest is not None and three_months_ago is not None:
                    quarterly_change = latest / three_months_ago
                    headline_3m_ann = (quarterly_change ** 4 - 1) * 100
                    import_data["headline_3m_ann"] = headline_3m_ann

        # Calculate USD adjustment if enabled
        usd_yoy = None
        if fx_adjustment and usd_df is not None and not usd_df.empty and len(usd_df) >= 12:
            # Convert daily USD to monthly average, then calculate YoY
            usd_values = usd_df["value"].dropna()
            if len(usd_values) >= 250:  # ~12 months of daily data
                # Simple approach: use the values as if they're already monthly
                # (In practice, you'd want to resample daily to monthly first)
                latest_usd = safe_float_conversion(usd_values.iloc[-1])
                year_ago_usd = safe_float_conversion(usd_values.iloc[-250])  # ~250 trading days
                if latest_usd is not None and year_ago_usd is not None:
                    usd_yoy = calculate_percentage_change(latest_usd, year_ago_usd)
                    import_data["usd_yoy"] = usd_yoy

        # Calculate composite pressure based on available data
        composite_pressure = None
        method_used = None

        if core_yoy is not None and energy_yoy is not None:
            # Method B: Core + Energy mix
            composite_pressure = (core_weight * core_yoy) + (energy_weight * energy_yoy)
            method_used = f"core_energy_mix_{core_source}"

            # Apply FX adjustment if enabled
            if fx_adjustment and usd_yoy is not None:
                fx_adjustment_value = fx_beta * usd_yoy
                composite_pressure -= fx_adjustment_value
                method_used += "_fx_adjusted"
                import_data["fx_adjustment"] = fx_adjustment_value

        elif headline_yoy is not None:
            # Method A: Headline-only fallback
            composite_pressure = headline_yoy
            method_used = "headline_only"

        # Determine alert level
        alert_level = "normal"
        if composite_pressure is not None:
            # Check for red alert
            if composite_pressure > red_threshold or (headline_3m_ann and headline_3m_ann > fast_threshold):
                alert_level = "red"
            # Check for amber alert
            elif composite_pressure > amber_threshold and headline_3m_ann and headline_3m_ann > 0:
                alert_level = "amber"

        # Final result
        if composite_pressure is None:
            return {
                "ok": None,
                "composite_pressure": None,
                "alert_level": "no_data",
                "method_used": None
            }

        result = {
            "ok": composite_pressure >= threshold_pct,
            "composite_pressure": composite_pressure,
            "alert_level": alert_level,
            "method_used": method_used,
            "threshold": threshold_pct,
            "latest": composite_pressure  # For display formatting
        }
        result.update(import_data)

        return result


class LumberInflationIndicator(MacroIndicator):
    """Lumber price inflation indicator."""

    def __init__(self):
        super().__init__(
            name="Lumber Inflation Pressure",
            description="PPI for lumber and wood products",
            series_id="WPU08"  # PPI for lumber and wood products
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if lumber PPI up >threshold% YoY."""
        df = data.get("WPU08")
        if df is None or df.empty:
            return {"ok": None, "yoy_change": None, "latest": None}

        threshold_pct = config.get("lumber_inflation_threshold_pct", 25.0)

        values = df["value"].dropna()
        if len(values) < 12:  # Need at least 12 months
            return {"ok": None, "yoy_change": None, "latest": None}

        latest = safe_float_conversion(values.iloc[-1])
        year_ago = safe_float_conversion(values.iloc[-12])

        if latest is None or year_ago is None:
            return {"ok": None, "yoy_change": None, "latest": latest}

        yoy_change = calculate_percentage_change(latest, year_ago)

        return {
            "ok": yoy_change >= threshold_pct,
            "yoy_change": yoy_change,
            "latest": latest,
            "threshold": threshold_pct
        }


class CompositeInflationIndicator(MacroIndicator):
    """Composite commodity inflation pressure gauge."""

    def __init__(self):
        super().__init__(
            name="Composite Commodity Inflation",
            description="Weighted average of commodity inflation pressures",
            series_id="COMPOSITE"  # Not a real series, calculated from others
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Composite inflation score from multiple commodity categories."""
        # Create instances of other indicators to get their values
        energy_indicator = EnergyInflationIndicator()
        food_indicator = FoodInflationIndicator()
        metals_indicator = MetalsInflationIndicator()
        lumber_indicator = LumberInflationIndicator()

        # Calculate individual indicator results
        energy_result = energy_indicator.calculate(data, config)
        food_result = food_indicator.calculate(data, config)
        metals_result = metals_indicator.calculate(data, config)
        lumber_result = lumber_indicator.calculate(data, config)

        # Extract inflation rates
        inflation_rates = []
        weights = []
        categories = []

        # Energy (weight: 30% - high impact on everything)
        energy_inflation = energy_result.get("max_inflation")
        if energy_inflation is not None:
            inflation_rates.append(energy_inflation)
            weights.append(0.30)
            categories.append("energy")

        # Food (weight: 25% - essential commodity)
        food_inflation = food_result.get("max_food_inflation")
        if food_inflation is not None:
            inflation_rates.append(food_inflation)
            weights.append(0.25)
            categories.append("food")

        # Metals (weight: 25% - industrial input)
        metals_inflation = metals_result.get("max_metals_inflation")
        if metals_inflation is not None:
            inflation_rates.append(metals_inflation)
            weights.append(0.25)
            categories.append("metals")

        # Lumber (weight: 20% - housing/construction)
        lumber_inflation = lumber_result.get("yoy_change")
        if lumber_inflation is not None:
            inflation_rates.append(lumber_inflation)
            weights.append(0.20)
            categories.append("lumber")

        if not inflation_rates:
            return {"ok": None, "composite_score": None, "categories": []}

        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            # Normalize weights
            normalized_weights = [w / total_weight for w in weights]
            composite_score = sum(rate * weight for rate, weight in zip(inflation_rates, normalized_weights))
        else:
            composite_score = sum(inflation_rates) / len(inflation_rates)

        # Threshold for composite inflation
        threshold = config.get("composite_inflation_threshold_pct", 12.0)

        return {
            "ok": composite_score >= threshold,
            "composite_score": composite_score,
            "threshold": threshold,
            "categories": categories,
            "component_rates": dict(zip(categories, inflation_rates)),
            "latest": composite_score  # For display formatting
        }


# Factory function to create all commodity indicators
def create_commodity_indicators() -> Dict[str, MacroIndicator]:
    """Create all commodity inflation indicators."""
    return {
        "energy_inflation": EnergyInflationIndicator(),
        "natgas_inflation": NaturalGasInflationIndicator(),
        "food_inflation": FoodInflationIndicator(),
        "metals_inflation": MetalsInflationIndicator(),
        "import_inflation": ImportInflationIndicator(),
        "lumber_inflation": LumberInflationIndicator(),
        "composite_inflation": CompositeInflationIndicator()
    }


def get_required_series() -> List[str]:
    """Get list of all FRED series IDs needed for commodity indicators."""
    return [
        # Energy
        "GASREGCOVW",   # Weekly U.S. Regular Conventional Gasoline Prices
        "GASDESW",      # Weekly U.S. Diesel Sales Price
        "DHHNGSP",      # Daily Henry Hub natural gas spot price

        # Food/Agriculture (using actual FRED agricultural series)
        "PWHEAMTUSDM",  # Global price of Wheat (IMF)
        "PMAIZMTUSDM",  # Global price of Maize (Corn) (IMF)
        "PSOYBUSDM",    # Global price of Soybeans (IMF)

        # Metals/Industrial
        "WPU101",       # PPI: Steel mill products
        "WPU06",        # PPI: Industrial chemicals
        "WPU08",        # PPI: Lumber and wood products

        # Import prices (BLS Import Price Indexes - proper methodology)
        "IR",           # All imports (headline) - Index 2000=100
        "IREXFDFLS",    # Ex food & fuels - Index Dec 2010=100 (preferred)
        "IREXFUELS",    # Ex fuels - Index Dec 2001=100 (fallback)
        "IR10",         # Fuels & lubricants (energy shock)
        "DTWEXBGS",     # Broad U.S. Dollar Index (daily, for FX adjustment)
    ]