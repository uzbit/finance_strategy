"""
Base classes for financial indicators.
Provides common interface and functionality for all indicator types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd


class BaseIndicator(ABC):
    """Abstract base class for all financial indicators."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.last_update = None
        self.last_result = None

    @abstractmethod
    def calculate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the indicator value.

        Args:
            data: Input data for calculation
            config: Configuration parameters

        Returns:
            Dictionary with calculation results including:
            - ok: Boolean indicating if threshold is breached
            - latest: Current value
            - Additional indicator-specific metrics
        """
        pass

    def get_status(self) -> str:
        """Get human-readable status of the indicator."""
        if self.last_result is None:
            return "No data"
        return "WARNING" if self.last_result.get("ok") else "OK"

    def is_warning(self) -> bool:
        """Check if indicator is currently in warning state."""
        return self.last_result is not None and self.last_result.get("ok", False)

    def update(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Update indicator with new data and return result."""
        self.last_result = self.calculate(data, config)
        self.last_update = datetime.now()
        return self.last_result


class MacroIndicator(BaseIndicator):
    """Base class for macro economic indicators."""

    def __init__(self, name: str, description: str, series_id: str):
        super().__init__(name, description)
        self.series_id = series_id

    def get_data_requirements(self) -> Dict[str, Any]:
        """Get data requirements for this indicator."""
        return {
            "source": "fred",
            "series_id": self.series_id,
            "frequency": "daily"
        }


class PanicIndicator(BaseIndicator):
    """Base class for real-time panic indicators."""

    def __init__(self, name: str, description: str, category: str):
        super().__init__(name, description)
        self.category = category  # equity, credit, rates, crypto, commodity

    def get_data_requirements(self) -> Dict[str, Any]:
        """Get data requirements for this indicator."""
        return {
            "source": "realtime",
            "category": self.category,
            "frequency": "realtime"
        }


class AssetTrendIndicator(BaseIndicator):
    """Base class for asset trend indicators."""

    def __init__(self, name: str, description: str, ticker: str):
        super().__init__(name, description)
        self.ticker = ticker

    def get_data_requirements(self) -> Dict[str, Any]:
        """Get data requirements for this indicator."""
        return {
            "source": "yahoo",
            "ticker": self.ticker,
            "frequency": "daily"
        }


class IndicatorResult:
    """Container for indicator calculation results."""

    def __init__(self, ok: bool, latest: Optional[float] = None,
                 threshold: Optional[float] = None, **kwargs):
        self.ok = ok
        self.latest = latest
        self.threshold = threshold
        self.metadata = kwargs
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result = {
            "ok": self.ok,
            "latest": self.latest,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat()
        }
        result.update(self.metadata)
        return result

    def __repr__(self) -> str:
        status = "WARN" if self.ok else "OK"
        return f"IndicatorResult(status={status}, value={self.latest})"


def safe_float_conversion(value: Any) -> Optional[float]:
    """Safely convert value to float, handling pandas Series and other types."""
    if value is None:
        return None

    try:
        if hasattr(value, 'item'):  # pandas scalar
            return float(value.item())
        elif hasattr(value, 'iloc'):  # pandas Series
            return float(value.iloc[-1])
        else:
            return float(value)
    except (ValueError, TypeError, IndexError):
        return None


def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values."""
    if previous == 0:
        return 0.0
    return (current - previous) / previous * 100.0


def validate_data_frame(df: pd.DataFrame, required_columns: list = None) -> bool:
    """Validate that DataFrame has required structure."""
    if df.empty:
        return False

    if required_columns:
        return all(col in df.columns for col in required_columns)

    return True