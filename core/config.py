"""
Configuration management for the finance strategy dashboard.
Handles loading and validation of settings for both macro and panic indicators.
"""

import os
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path

DEFAULT_CONFIG_FILE = "config.json"

class Config:
    """Configuration manager for the dashboard."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or DEFAULT_CONFIG_FILE
        self._config = self._load_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            # General settings
            "mode": "both",  # "macro", "panic", "both"
            "refresh_interval_minutes": 15,
            "tickers": ["SPY", "BTC-USD"],
            "fred_api_key": "",
            "data_start_date": "2015-01-01",

            # Macro indicator thresholds
            "trend_filter_pct": 0.02,
            "high_yield_lookback_months": 6,
            "high_yield_threshold_pts": 1.0,
            "permits_lookback_months": 6,
            "vix_high_threshold": 25.0,
            "consumer_sentiment_low_threshold": 80.0,
            "consumer_sentiment_lookback_months": 6,
            "real_rates_low_threshold": 0.0,
            "inflation_expectations_high_threshold": 3.0,
            "oil_volatility_threshold_pct": 20.0,
            "oil_lookback_months": 3,
            "leading_indicators_decline_threshold_pct": -2.0,
            "leading_indicators_lookback_months": 6,

            # Panic indicator thresholds
            "panic_thresholds": {
                "vix_spike": 40,
                "vix_term_flip": True,  # 9-day > VIX > 3-month
                "spy_intraday_drop": -4.0,
                "breadth_washout": 15,  # % stocks above 20-day MA
                "everything_down": 95,  # % stocks red
                "hyg_stress_pct": -2.5,
                "hyg_volume_multiple": 2.0,
                "etf_nav_dislocation": 1.0,  # % below NAV
                "treasury_shock_bp": 20,
                "dxy_spike": 1.5,
                "oil_crash": -8.0,
                "copper_crash": -3.0,
                "crypto_correlation": -3.0,
                "stablecoin_depeg": 0.995,  # below $0.995
                "stablecoin_depeg_minutes": 15
            },

            # Panic scoring levels
            "panic_scoring": {
                "normal": [0, 1],
                "elevated": [2, 3],
                "high": [4, 5],
                "extreme": [6, 99]
            }
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with fallback to defaults."""
        default_config = self._get_default_config()

        if not os.path.exists(self.config_path):
            print(f"[warn] Config file {self.config_path} not found. Using defaults.", file=sys.stderr)
            return default_config

        try:
            with open(self.config_path, 'r') as f:
                file_config = json.load(f)

            # Deep merge with defaults
            merged_config = self._deep_merge(default_config, file_config)
            return merged_config

        except (json.JSONDecodeError, IOError) as e:
            print(f"[warn] Config file error: {e}. Using defaults.", file=sys.stderr)
            return default_config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override values taking precedence."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_fred_api_key(self) -> str:
        """Get FRED API key from config or environment."""
        return self.get("fred_api_key") or os.environ.get("FRED_API_KEY", "")

    def get_panic_threshold(self, indicator: str) -> float:
        """Get panic threshold for specific indicator."""
        return self.get(f"panic_thresholds.{indicator}")

    def get_panic_level(self, flag_count: int) -> str:
        """Determine panic level based on flag count."""
        scoring = self.get("panic_scoring")

        for level, range_vals in scoring.items():
            if range_vals[0] <= flag_count <= range_vals[1]:
                return level

        return "extreme"  # fallback

    def is_mode_enabled(self, mode: str) -> bool:
        """Check if specific mode is enabled."""
        current_mode = self.get("mode", "both")
        return current_mode == mode or current_mode == "both"

    def load_config(self, config_path: str) -> None:
        """Load configuration from specified file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")

        self.config_path = config_path
        self._config = self._load_config()

    def save_updated_config(self, updates: Dict[str, Any]) -> None:
        """Save updated configuration to file."""
        updated_config = self._deep_merge(self._config, updates)

        try:
            with open(self.config_path, 'w') as f:
                json.dump(updated_config, f, indent=2)
            self._config = updated_config
        except IOError as e:
            print(f"[error] Failed to save config: {e}", file=sys.stderr)

    def validate_config(self) -> bool:
        """Validate configuration for required fields and reasonable values."""
        # Check required fields
        required_fields = ["tickers", "data_start_date"]
        for field in required_fields:
            if not self.get(field):
                print(f"[error] Missing required config field: {field}", file=sys.stderr)
                return False

        # Validate data types and ranges
        if not isinstance(self.get("tickers"), list):
            print("[error] 'tickers' must be a list", file=sys.stderr)
            return False

        # Validate panic thresholds
        panic_thresholds = self.get("panic_thresholds", {})
        if not isinstance(panic_thresholds, dict):
            print("[error] 'panic_thresholds' must be a dictionary", file=sys.stderr)
            return False

        return True

    @property
    def data(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()


# Global config instance
config = Config()