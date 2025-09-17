"""
Commodity-based inflation indicators using US government APIs.
Tracks energy, food, metals, and import price pressures for inflation nowcasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from indicators.base import MacroIndicator, safe_float_conversion, calculate_percentage_change
from data.fred import fred_api


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
    """Import price inflation pressure."""

    def __init__(self):
        super().__init__(
            name="Import Inflation Pressure",
            description="Import price index for fuels and industrial supplies",
            series_id="IMP3000"  # Import price index - fuels and lubricants
        )

    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """True if import prices up >threshold% YoY."""
        fuels_df = data.get("IMP3000")  # Import fuels & lubricants
        industrial_df = data.get("IMP1000")  # Import all commodities

        threshold_pct = config.get("import_inflation_threshold_pct", 10.0)

        import_changes = []
        import_data = {}

        # Calculate YoY changes for available series
        for name, df in [("fuels", fuels_df), ("commodities", industrial_df)]:
            if df is not None and not df.empty and len(df) >= 12:  # 12 months
                values = df["value"].dropna()
                latest = safe_float_conversion(values.iloc[-1])
                year_ago = safe_float_conversion(values.iloc[-12] if len(values) >= 12 else values.iloc[0])

                if latest is not None and year_ago is not None:
                    yoy_change = calculate_percentage_change(latest, year_ago)
                    import_changes.append(yoy_change)
                    import_data[f"{name}_yoy"] = yoy_change
                    import_data[f"{name}_latest"] = latest

        if not import_changes:
            return {"ok": None, "max_import_inflation": None}

        max_import_inflation = max(import_changes)

        result = {
            "ok": max_import_inflation >= threshold_pct,
            "max_import_inflation": max_import_inflation,
            "threshold": threshold_pct
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
        # This will be calculated by aggregating results from other indicators
        # For now, return a placeholder
        return {"ok": None, "composite_score": None}


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
        "GASREGCOVW",   # Weekly U.S. Regular Conventional Gasoline Prices (correct)
        "GASDESW",      # Weekly U.S. Diesel Sales Price (correct)
        "DHHNGSP",      # Daily Henry Hub natural gas spot price

        # Food/Agriculture (using actual FRED agricultural series)
        "PWHEAMTUSDM",  # Global price of Wheat (IMF)
        "PMAIZMTUSDM",  # Global price of Maize (Corn) (IMF)
        "PSOYBUSDM",    # Global price of Soybeans (IMF)

        # Metals/Industrial
        "WPU101",       # PPI: Steel mill products
        "WPU06",        # PPI: Industrial chemicals
        "WPU08",        # PPI: Lumber and wood products

        # Import prices (correct BLS series)
        "IMP3000",      # Import price index: Fuels and lubricants
        "IMP1000",      # Import price index: All commodities
    ]