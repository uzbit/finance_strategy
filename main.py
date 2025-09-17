#!/usr/bin/env python3
"""
Unified Financial Risk Dashboard
Comprehensive market risk assessment with macro and panic indicators.

This is the main entry point for the modular risk dashboard system.
Provides both macro economic indicators and real-time panic detection.
"""

import sys
import argparse
from datetime import datetime
from typing import Optional

# Import the modular components
from core.dashboard import dashboard
from core.config import config


def run_dashboard(show_details: bool = False,
                 trend_tickers: Optional[list] = None,
                 panic_only: bool = False,
                 macro_only: bool = False) -> str:
    """
    Run the complete risk dashboard.

    Args:
        show_details: Show additional details in output
        trend_tickers: List of tickers for trend analysis
        panic_only: Show only panic indicators
        macro_only: Show only macro indicators

    Returns:
        Formatted table output
    """
    try:
        # Initialize trend indicators if specified (otherwise use default from config)
        if trend_tickers:
            dashboard.initialize_trend_indicators(trend_tickers)
        # Note: Dashboard already initializes with config.tickers by default

        # Update all indicators
        print("Fetching data and updating indicators...")
        results = dashboard.update_all_indicators()

        # Filter results if requested
        if panic_only:
            # Clear macro and trend results for display
            results["macro"] = {}
            dashboard.trend_indicators = {}
        elif macro_only:
            # Clear panic results for display
            results["panic"] = {}
            dashboard.panic_indicators = {}

        # Format and return output
        return dashboard.format_table_output(show_details=show_details)

    except Exception as e:
        return f"Error running dashboard: {e}"


def run_quick_check() -> str:
    """Run a quick check of just the most critical indicators."""
    try:
        # Update only panic indicators for speed
        results = dashboard.update_all_indicators()

        panic_score = results.get("panic_score", {})
        total_score = panic_score.get("total_score", 0)
        panic_level = panic_score.get("panic_level", "normal")
        active_indicators = panic_score.get("active_indicators", [])

        # Get market hours info
        freshness = results.get("metadata", {}).get("data_freshness", {})
        market_open = freshness.get("market_open", False)

        output = [
            "=== QUICK MARKET CHECK ===",
            f"Panic Level: {panic_level.upper()} ({total_score}/11)",
            f"Market Open: {'Yes' if market_open else 'No'}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]

        if active_indicators:
            output.append(f"Active Warnings: {', '.join(active_indicators)}")
        else:
            output.append("No active panic warnings")

        return "\n".join(output)

    except Exception as e:
        return f"Error in quick check: {e}"


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Financial Risk Dashboard - Macro and Panic Indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Full dashboard
  python main.py --details          # Full dashboard with details
  python main.py --quick            # Quick panic check only
  python main.py --panic-only       # Panic indicators only
  python main.py --macro-only       # Macro indicators only
  python main.py --tickers SPY QQQ  # Custom trend tickers
        """
    )

    parser.add_argument(
        "--details",
        action="store_true",
        help="Show additional details in output"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check - panic indicators only"
    )

    parser.add_argument(
        "--panic-only",
        action="store_true",
        help="Show only panic indicators"
    )

    parser.add_argument(
        "--macro-only",
        action="store_true",
        help="Show only macro indicators"
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Specify tickers for trend analysis (e.g., SPY QQQ IWM)"
    )

    parser.add_argument(
        "--config",
        help="Path to custom config file"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information"
    )

    args = parser.parse_args()

    # Handle version
    if args.version:
        print("Financial Risk Dashboard v2.0.0")
        print("Modular architecture with macro and panic indicators")
        return

    # Load custom config if specified
    if args.config:
        try:
            config.load_config(args.config)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            print(f"Error loading config: {e}")
            return

    # Handle quick check
    if args.quick:
        output = run_quick_check()
        print(output)
        return

    # Handle full dashboard
    try:
        output = run_dashboard(
            show_details=args.details,
            trend_tickers=args.tickers,
            panic_only=args.panic_only,
            macro_only=args.macro_only
        )
        print(output)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# Convenience functions for backwards compatibility
def summarize(show_details: bool = False) -> str:
    """Backwards compatible function matching original interface."""
    return run_dashboard(show_details=show_details)


def get_panic_score() -> dict:
    """Get current panic score."""
    try:
        dashboard.update_all_indicators()
        return dashboard.last_results.get("panic_score", {})
    except Exception as e:
        return {"error": str(e)}


def get_warning_count() -> dict:
    """Get warning count by category."""
    try:
        dashboard.update_all_indicators()
        return dashboard.get_warning_count()
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    main()