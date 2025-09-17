#!/usr/bin/env python3
"""
risk_dashboard.py
A simple, no-jargon rule checker to help you decide when to trim or exit positions.
It pulls a few public indicators and prints a "score" with plain-English suggestions.

Data sources:
- FRED API for macro time series (free key: https://fred.stlouisfed.org/docs/api/api_key.html)
  Series used:
    - T10Y3M  : 10-Year minus 3-Month Treasury spread (yield curve)
    - UNRATE  : Unemployment rate (for Sahm Rule)
    - ICSA    : Initial jobless claims (weekly) [optional, not used in scoring by default]
    - NFCI    : Chicago Fed National Financial Conditions Index
    - BAMLH0A0HYM2 : High-yield corporate bond spread (ICE BofA; on FRED)
    - PERMIT  : Building permits (new housing units, seasonally adjusted annual rate)
    - VIXCLS  : VIX Volatility Index (market fear gauge)
    - UMCSENT : University of Michigan Consumer Sentiment
    - DFII10  : 10-Year Treasury Inflation-Indexed (real interest rates)
    - T10YIE  : 10-Year Breakeven Inflation Rate (inflation expectations)
    - USALOLITOAASTSAM : OECD Leading Economic Indicators for US
    - DCOILWTICO : WTI Crude Oil Prices

- Yahoo Finance for prices:
    - e.g., SPY (stocks), BTC-USD (crypto). You can pass any list of tickers.

How the "signals" work (defaults you can edit in config.json):
- Yield curve inverted: latest T10Y3M < 0 → 1 red flag
- Sahm Rule tripped: 3-month avg unemployment is ≥ 0.50 ppt above its 12‑month low → 1 red flag
- High-yield spread jump: HY OAS is ≥ 1.0 ppt above its 6‑month low → 1 red flag
- Financial conditions tight: NFCI > 0 → 1 red flag
- Building permits falling: 6‑month change ≤ 0% → 1 red flag
- VIX elevated: VIX > 25 (market fear high) → 1 red flag
- Consumer sentiment weak: < 80 or declining significantly → 1 red flag
- Real rates negative: TIPS yield < 0% → 1 red flag
- Inflation expectations high: > 3% → 1 red flag
- Oil volatility high: 3-month volatility > 20% → 1 red flag
- Leading indicators declining: 6-month decline > 2% → 1 red flag
- Trend break per asset: Close < 200‑day SMA (with 2% filter) → 1 red flag per ticker

Score → suggestion (updated for expanded indicators):
- 0–2 flags: Hold steady.
- 3–4 flags: Take some profit on winners / tighten stops.
- 5–6 flags: Trim 10–20% of riskier holdings.
- 7–8 flags: Significant defensive positioning.
- 9+ flags : CAUTION: Multiple critical warnings.

Usage:
  python risk_dashboard.py --start 2015-01-01 --tickers SPY BTC-USD ETH-USD

Tip:
  Set your FRED API key in an env var FRED_API_KEY, or pass --fred-key YOURKEY.
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

import requests
import pandas as pd
import numpy as np
import yfinance as yf


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_START = "2015-01-01"
DEFAULT_CONFIG_FILE = "config.json"

SERIES = {
    "T10Y3M": "10Y minus 3M Treasury spread",
    "UNRATE": "Unemployment rate",
    "ICSA": "Initial jobless claims (weekly)",
    "NFCI": "Chicago Fed financial conditions",
    "BAMLH0A0HYM2": "High-yield corporate spread",
    "PERMIT": "Housing permits (SAAR)",
    "VIXCLS": "VIX Volatility Index",
    "UMCSENT": "University of Michigan Consumer Sentiment",
    "DFII10": "10-Year Treasury Inflation-Indexed (Real Rate)",
    "T10YIE": "10-Year Breakeven Inflation Rate",
    "USALOLITOAASTSAM": "OECD Leading Economic Indicators",
    "DCOILWTICO": "WTI Crude Oil Prices",
}

def load_config(config_path: str = DEFAULT_CONFIG_FILE) -> Dict[str, Any]:
    """Load configuration from JSON file. Falls back to defaults if file doesn't exist."""
    default_config = {
        "tickers": ["SPY", "BTC-USD"],
        "fred_api_key": "",
        "data_start_date": DEFAULT_START,
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
        "leading_indicators_lookback_months": 6
    }

    if not os.path.exists(config_path):
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Merge with defaults to ensure all keys exist
        return {**default_config, **config}
    except (json.JSONDecodeError, IOError) as e:
        print(f"[warn] Config file error: {e}. Using defaults.", file=sys.stderr)
        return default_config


def fred_get(series_id: str, start: str, api_key: str = "") -> pd.DataFrame:
    """Download a FRED series. Returns a DataFrame with datetime index and float 'value' column.
    Missing/non-numeric values are dropped.
    """
    params = {
        "series_id": series_id,
        "observation_start": start,
        "file_type": "json",
    }
    if api_key:
        params["api_key"] = api_key
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("observations", [])
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["value"], index=pd.to_datetime([]))
    df["date"] = pd.to_datetime(df["date"])
    # Coerce values to float, drop '.' or missing
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date")[["value"]].sort_index()
    return df


def yield_curve_inverted(t10y3m: pd.DataFrame) -> Dict[str, Any]:
    """True if latest value < 0."""
    if t10y3m.empty:
        return {"ok": None, "latest": None}
    latest = float(t10y3m["value"].iloc[-1])
    return {"ok": latest < 0.0, "latest": latest}


def sahm_rule(unrate: pd.DataFrame) -> Dict[str, Any]:
    """Sahm Rule: 3m avg unemployment vs the minimum 3m avg of the prior 12 months.
    Trigger if gap >= 0.5 percentage points.
    """
    if unrate.empty or len(unrate) < 15:
        return {"ok": None, "gap": None, "latest_3m": None, "min_prior_12m": None}

    u = unrate["value"].copy()
    u3 = u.rolling(3).mean()
    # For each date, compute the prior 12 months' min of the 3m avg (excluding current month)
    # Align indexes safely
    gaps = []
    for i in range(len(u3)):
        if i < 14:  # need at least 15 months to have a prior 12 months window after 3m avg
            gaps.append(np.nan)
            continue
        # prior 12 months of 3m avg (i-12 to i-1 inclusive)
        prior_window = u3.iloc[i-12:i]
        min_prior = prior_window.min()
        gap = u3.iloc[i] - min_prior
        gaps.append(gap)
    sahm = pd.Series(gaps, index=u3.index, name="gap")
    latest_gap = float(sahm.dropna().iloc[-1])
    latest_u3 = float(u3.dropna().iloc[-1])
    min_prior = float((u3.iloc[-12-1:-1]).min()) if len(u3.dropna()) >= 13 else np.nan
    return {"ok": latest_gap >= 0.5, "gap": latest_gap, "latest_3m": latest_u3, "min_prior_12m": min_prior}


def hy_spread_jump(hy: pd.DataFrame, lookback_months: int = 6, threshold_pts: float = 1.0) -> Dict[str, Any]:
    """True if the latest HY spread is >= threshold_pts above its min in the past lookback_months.
    Units are percentage points (so 1.0 = 100 bps).
    """
    if hy.empty:
        return {"ok": None, "delta": None, "recent_min": None, "latest": None}
    hy_m = hy["value"].asfreq("MS").interpolate()  # force to monthly freq for robustness
    recent = hy_m.tail(lookback_months * 30)  # Approximate daily data for months
    if recent.empty:
        return {"ok": None, "delta": None, "recent_min": None, "latest": None}
    recent_min = float(recent.min())
    latest = float(hy_m.iloc[-1])
    delta = latest - recent_min
    return {"ok": delta >= threshold_pts, "delta": delta, "recent_min": recent_min, "latest": latest}


def nfci_tight(nfci: pd.DataFrame) -> Dict[str, Any]:
    """True if NFCI > 0 (tighter-than-average financial conditions)."""
    if nfci.empty:
        return {"ok": None, "latest": None}
    latest = float(nfci["value"].iloc[-1])
    return {"ok": latest > 0.0, "latest": latest}


def permits_downtrend(permits: pd.DataFrame, months: int = 6) -> Dict[str, Any]:
    """True if 6‑month change is <= 0% (i.e., falling or flat)."""
    if permits.empty or len(permits) < months + 1:
        return {"ok": None, "pct_change": None}
    p = permits["value"].asfreq("MS").interpolate()
    past = float(p.iloc[-months-1])
    latest = float(p.iloc[-1])
    pct = (latest - past) / past * 100.0 if past != 0 else np.nan
    return {"ok": pct <= 0.0, "pct_change": pct}


def trend_break(ticker: str, start: str, pct_filter: float = 0.02) -> Dict[str, Any]:
    """True if Close < SMA200*(1 - pct_filter)."""
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    except Exception as e:
        return {"ok": None, "latest": None, "sma200": None, "error": str(e)}
    if df.empty or "Close" not in df:
        return {"ok": None, "latest": None, "sma200": None, "error": "no data"}
    close = df["Close"]
    sma200 = close.rolling(200).mean()
    if len(sma200.dropna()) == 0:
        return {"ok": None, "latest": None, "sma200": None, "error": "insufficient history for SMA200"}
    latest = close.iloc[-1].item()
    s = sma200.iloc[-1].item()
    trigger = latest < s * (1.0 - pct_filter)
    return {"ok": trigger, "latest": latest, "sma200": s}


def vix_elevated(vix: pd.DataFrame, threshold: float = 25.0) -> Dict[str, Any]:
    """True if VIX > threshold (indicating elevated fear/volatility)."""
    if vix.empty:
        return {"ok": None, "latest": None}
    latest = float(vix["value"].iloc[-1])
    return {"ok": latest > threshold, "latest": latest}


def consumer_sentiment_weak(sentiment: pd.DataFrame, threshold: float = 80.0, lookback_months: int = 6) -> Dict[str, Any]:
    """True if consumer sentiment is below threshold or declining significantly."""
    if sentiment.empty or len(sentiment) < 2:
        return {"ok": None, "latest": None, "recent_avg": None}

    s = sentiment["value"].asfreq("MS").interpolate()
    if len(s) < lookback_months + 1:
        return {"ok": None, "latest": None, "recent_avg": None}

    latest = float(s.iloc[-1])
    recent_avg = float(s.iloc[-lookback_months:].mean())

    # Trigger if latest is below threshold OR recent average is significantly below long-term average
    long_term_avg = float(s.mean())
    decline_significant = recent_avg < long_term_avg * 0.9  # 10% below long-term average

    trigger = latest < threshold or decline_significant
    return {"ok": trigger, "latest": latest, "recent_avg": recent_avg}


def real_rates_negative(real_rates: pd.DataFrame, threshold: float = 0.0) -> Dict[str, Any]:
    """True if real rates (TIPS) < threshold (typically 0, indicating negative real rates)."""
    if real_rates.empty:
        return {"ok": None, "latest": None}
    latest = float(real_rates["value"].iloc[-1])
    return {"ok": latest < threshold, "latest": latest}


def inflation_expectations_high(inflation_exp: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
    """True if inflation expectations > threshold."""
    if inflation_exp.empty:
        return {"ok": None, "latest": None}
    latest = float(inflation_exp["value"].iloc[-1])
    return {"ok": latest > threshold, "latest": latest}


def oil_volatility_high(oil: pd.DataFrame, lookback_months: int = 3, threshold_pct: float = 20.0) -> Dict[str, Any]:
    """True if oil price volatility (std dev of monthly changes) > threshold."""
    if oil.empty or len(oil) < lookback_months * 30:  # Rough daily data estimate
        return {"ok": None, "volatility": None, "latest": None}

    # Calculate monthly percentage changes
    monthly = oil["value"].resample("MS").last().dropna()
    if len(monthly) < lookback_months + 1:
        return {"ok": None, "volatility": None, "latest": None}

    pct_changes = monthly.pct_change().dropna() * 100
    recent_vol = float(pct_changes.iloc[-lookback_months:].std())
    latest = float(oil["value"].iloc[-1])

    return {"ok": recent_vol > threshold_pct, "volatility": recent_vol, "latest": latest}


def leading_indicators_declining(lei: pd.DataFrame, lookback_months: int = 6, threshold_pct: float = -2.0) -> Dict[str, Any]:
    """True if leading economic indicators declined by more than threshold over lookback period."""
    if lei.empty or len(lei) < lookback_months + 1:
        return {"ok": None, "pct_change": None, "latest": None}

    l = lei["value"].asfreq("MS").interpolate()
    if len(l) < lookback_months + 1:
        return {"ok": None, "pct_change": None, "latest": None}

    past = float(l.iloc[-lookback_months-1])
    latest = float(l.iloc[-1])
    pct_change = (latest - past) / past * 100.0 if past != 0 else 0.0

    return {"ok": pct_change <= threshold_pct, "pct_change": pct_change, "latest": latest}


def summarize(flags: Dict[str, Any], asset_flags: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Create a table summary showing thresholds, values, and status."""
    lines = []
    total = 0

    def status_symbol(b):
        return "🔴" if b else "🟢"

    def status_text(b):
        return "YES" if b else "NO"

    # Table header
    lines.append("=" * 95)
    lines.append("MARKET RISK DASHBOARD")
    lines.append("=" * 95)
    lines.append(f"{'INDICATOR':<35} {'THRESHOLD':<15} {'CURRENT':<15} {'STATUS':<6} {'FLAG'}")
    lines.append("-" * 95)

    # Macro indicators
    if flags["yc"]["ok"] is not None:
        yc_val = f'{flags["yc"]["latest"]:.2f}%'
        lines.append(f"{'Yield Curve Inverted':<35} {'< 0%':<15} {yc_val:<15} {status_symbol(flags['yc']['ok']):<6} {status_text(flags['yc']['ok'])}")
        total += int(bool(flags["yc"]["ok"]))
    else:
        lines.append(f"{'Yield Curve Inverted':<35} {'< 0%':<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["sahm"]["ok"] is not None:
        sahm_val = f'{flags["sahm"]["gap"]:.2f} pts'
        lines.append(f"{'Sahm Rule Triggered':<35} {'≥ 0.5 pts':<15} {sahm_val:<15} {status_symbol(flags['sahm']['ok']):<6} {status_text(flags['sahm']['ok'])}")
        total += int(bool(flags["sahm"]["ok"]))
    else:
        lines.append(f"{'Sahm Rule Triggered':<35} {'≥ 0.5 pts':<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["hy"]["ok"] is not None:
        hy_thresh = f'≥ {config["high_yield_threshold_pts"]:.1f} pts'
        hy_val = f'{flags["hy"]["delta"]:.2f} pts'
        lines.append(f"{'High-Yield Spread Jump':<35} {hy_thresh:<15} {hy_val:<15} {status_symbol(flags['hy']['ok']):<6} {status_text(flags['hy']['ok'])}")
        total += int(bool(flags["hy"]["ok"]))
    else:
        hy_thresh = f'≥ {config["high_yield_threshold_pts"]:.1f} pts'
        lines.append(f"{'High-Yield Spread Jump':<35} {hy_thresh:<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["nfci"]["ok"] is not None:
        nfci_val = f'{flags["nfci"]["latest"]:.2f}'
        lines.append(f"{'Financial Conditions Tight':<35} {'> 0':<15} {nfci_val:<15} {status_symbol(flags['nfci']['ok']):<6} {status_text(flags['nfci']['ok'])}")
        total += int(bool(flags["nfci"]["ok"]))
    else:
        lines.append(f"{'Financial Conditions Tight':<35} {'> 0':<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["permits"]["ok"] is not None:
        permits_val = f'{flags["permits"]["pct_change"]:.1f}%'
        lines.append(f"{'Building Permits Falling':<35} {'≤ 0%':<15} {permits_val:<15} {status_symbol(flags['permits']['ok']):<6} {status_text(flags['permits']['ok'])}")
        total += int(bool(flags["permits"]["ok"]))
    else:
        lines.append(f"{'Building Permits Falling':<35} {'≤ 0%':<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["vix"]["ok"] is not None:
        vix_thresh = f'> {config["vix_high_threshold"]:.0f}'
        vix_val = f'{flags["vix"]["latest"]:.1f}'
        lines.append(f"{'VIX Elevated':<35} {vix_thresh:<15} {vix_val:<15} {status_symbol(flags['vix']['ok']):<6} {status_text(flags['vix']['ok'])}")
        total += int(bool(flags["vix"]["ok"]))
    else:
        vix_thresh = f'> {config["vix_high_threshold"]:.0f}'
        lines.append(f"{'VIX Elevated':<35} {vix_thresh:<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["sentiment"]["ok"] is not None:
        sentiment_thresh = f'< {config["consumer_sentiment_low_threshold"]:.0f}'
        sentiment_val = f'{flags["sentiment"]["latest"]:.1f}'
        lines.append(f"{'Consumer Sentiment Weak':<35} {sentiment_thresh:<15} {sentiment_val:<15} {status_symbol(flags['sentiment']['ok']):<6} {status_text(flags['sentiment']['ok'])}")
        total += int(bool(flags["sentiment"]["ok"]))
    else:
        sentiment_thresh = f'< {config["consumer_sentiment_low_threshold"]:.0f}'
        lines.append(f"{'Consumer Sentiment Weak':<35} {sentiment_thresh:<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["real_rates"]["ok"] is not None:
        real_thresh = f'< {config["real_rates_low_threshold"]:.1f}%'
        real_val = f'{flags["real_rates"]["latest"]:.2f}%'
        lines.append(f"{'Real Rates Negative':<35} {real_thresh:<15} {real_val:<15} {status_symbol(flags['real_rates']['ok']):<6} {status_text(flags['real_rates']['ok'])}")
        total += int(bool(flags["real_rates"]["ok"]))
    else:
        real_thresh = f'< {config["real_rates_low_threshold"]:.1f}%'
        lines.append(f"{'Real Rates Negative':<35} {real_thresh:<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["inflation"]["ok"] is not None:
        inflation_thresh = f'> {config["inflation_expectations_high_threshold"]:.1f}%'
        inflation_val = f'{flags["inflation"]["latest"]:.2f}%'
        lines.append(f"{'Inflation Expectations High':<35} {inflation_thresh:<15} {inflation_val:<15} {status_symbol(flags['inflation']['ok']):<6} {status_text(flags['inflation']['ok'])}")
        total += int(bool(flags["inflation"]["ok"]))
    else:
        inflation_thresh = f'> {config["inflation_expectations_high_threshold"]:.1f}%'
        lines.append(f"{'Inflation Expectations High':<35} {inflation_thresh:<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["oil_vol"]["ok"] is not None:
        oil_thresh = f'> {config["oil_volatility_threshold_pct"]:.0f}%'
        oil_val = f'{flags["oil_vol"]["volatility"]:.1f}%'
        lines.append(f"{'Oil Volatility High':<35} {oil_thresh:<15} {oil_val:<15} {status_symbol(flags['oil_vol']['ok']):<6} {status_text(flags['oil_vol']['ok'])}")
        total += int(bool(flags["oil_vol"]["ok"]))
    else:
        oil_thresh = f'> {config["oil_volatility_threshold_pct"]:.0f}%'
        lines.append(f"{'Oil Volatility High':<35} {oil_thresh:<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    if flags["lei"]["ok"] is not None:
        lei_thresh = f'≤ {config["leading_indicators_decline_threshold_pct"]:.1f}%'
        lei_val = f'{flags["lei"]["pct_change"]:.2f}%'
        lines.append(f"{'Leading Indicators Declining':<35} {lei_thresh:<15} {lei_val:<15} {status_symbol(flags['lei']['ok']):<6} {status_text(flags['lei']['ok'])}")
        total += int(bool(flags["lei"]["ok"]))
    else:
        lei_thresh = f'≤ {config["leading_indicators_decline_threshold_pct"]:.1f}%'
        lines.append(f"{'Leading Indicators Declining':<35} {lei_thresh:<15} {'NO DATA':<15} {'❓':<6} {'--'}")

    # Asset trend section
    if asset_flags:
        lines.append("-" * 95)
        lines.append("ASSET TREND ANALYSIS")
        lines.append("-" * 95)
        for ticker, res in asset_flags.items():
            if res["ok"] is None:
                trend_name = f'{ticker} Trend Break'
                lines.append(f"{trend_name:<35} {'< 200-day SMA':<15} {'NO DATA':<15} {'❓':<6} {'--'}")
            else:
                current_vs_sma = f"{res['latest']:.2f} vs {res['sma200']:.2f}"
                trend_name = f'{ticker} Trend Break'
                lines.append(f"{trend_name:<35} {'< 200-day SMA':<15} {current_vs_sma:<15} {status_symbol(res['ok']):<6} {status_text(res['ok'])}")
                total += int(bool(res["ok"]))

    # Summary
    lines.append("=" * 95)
    lines.append(f"TOTAL RED FLAGS: {total}")

    # Suggestion (updated for expanded indicator set)
    if total <= 2:
        suggestion = "🟢 HOLD STEADY - No broad warning from this checklist."
    elif total <= 4:
        suggestion = "🟡 CAUTION - Consider taking some profit and tightening risk on volatile positions."
    elif total <= 6:
        suggestion = "🟠 WARNING - Consider trimming 10–20% of your riskier holdings."
    elif total <= 8:
        suggestion = "🔴 ALERT - Consider significant defensive positioning. Market stress is building."
    else:
        suggestion = "🚨 DANGER - Multiple critical warnings. Consider defensive stance until signals improve."

    lines.append(f"RECOMMENDATION: {suggestion}")
    lines.append("=" * 95)

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Simple market/risk checklist in plain English.")
    p.add_argument("--config", default=DEFAULT_CONFIG_FILE, help="Path to config file")
    p.add_argument("--start", help="Data start date (YYYY-MM-DD, overrides config)")
    p.add_argument("--fred-key", help="FRED API key (overrides config and env)")
    p.add_argument("--tickers", nargs="*", help="Tickers to trend-check (overrides config)")
    p.add_argument("--filter", type=float, help="Trend filter (overrides config)")
    args = p.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args if provided
    start_date = args.start or config["data_start_date"]
    fred_key = args.fred_key or config["fred_api_key"] or os.environ.get("FRED_API_KEY", "")
    tickers = args.tickers or config["tickers"]
    trend_filter = args.filter if args.filter is not None else config["trend_filter_pct"]

    # --- Fetch macro series ---
    try:
        t10y3m = fred_get("T10Y3M", start_date, fred_key)
    except Exception as e:
        t10y3m = pd.DataFrame()
        print(f"[warn] T10Y3M fetch failed: {e}", file=sys.stderr)

    try:
        unrate = fred_get("UNRATE", start_date, fred_key)
    except Exception as e:
        unrate = pd.DataFrame()
        print(f"[warn] UNRATE fetch failed: {e}", file=sys.stderr)

    try:
        nfci = fred_get("NFCI", start_date, fred_key)
    except Exception as e:
        nfci = pd.DataFrame()
        print(f"[warn] NFCI fetch failed: {e}", file=sys.stderr)

    try:
        hy = fred_get("BAMLH0A0HYM2", start_date, fred_key)
    except Exception as e:
        hy = pd.DataFrame()
        print(f"[warn] HY spread fetch failed: {e}", file=sys.stderr)

    try:
        permits = fred_get("PERMIT", start_date, fred_key)
    except Exception as e:
        permits = pd.DataFrame()
        print(f"[warn] PERMIT fetch failed: {e}", file=sys.stderr)

    try:
        vix = fred_get("VIXCLS", start_date, fred_key)
    except Exception as e:
        vix = pd.DataFrame()
        print(f"[warn] VIX fetch failed: {e}", file=sys.stderr)

    try:
        consumer_sentiment = fred_get("UMCSENT", start_date, fred_key)
    except Exception as e:
        consumer_sentiment = pd.DataFrame()
        print(f"[warn] Consumer sentiment fetch failed: {e}", file=sys.stderr)

    try:
        real_rates = fred_get("DFII10", start_date, fred_key)
    except Exception as e:
        real_rates = pd.DataFrame()
        print(f"[warn] Real rates fetch failed: {e}", file=sys.stderr)

    try:
        inflation_exp = fred_get("T10YIE", start_date, fred_key)
    except Exception as e:
        inflation_exp = pd.DataFrame()
        print(f"[warn] Inflation expectations fetch failed: {e}", file=sys.stderr)

    try:
        leading_indicators = fred_get("USALOLITOAASTSAM", start_date, fred_key)
    except Exception as e:
        leading_indicators = pd.DataFrame()
        print(f"[warn] Leading indicators fetch failed: {e}", file=sys.stderr)

    try:
        oil = fred_get("DCOILWTICO", start_date, fred_key)
    except Exception as e:
        oil = pd.DataFrame()
        print(f"[warn] Oil prices fetch failed: {e}", file=sys.stderr)

    # --- Compute macro flags ---
    flags = {
        "yc": yield_curve_inverted(t10y3m),
        "sahm": sahm_rule(unrate),
        "hy": hy_spread_jump(hy, config["high_yield_lookback_months"], config["high_yield_threshold_pts"]),
        "nfci": nfci_tight(nfci),
        "permits": permits_downtrend(permits, config["permits_lookback_months"]),
        "vix": vix_elevated(vix, config["vix_high_threshold"]),
        "sentiment": consumer_sentiment_weak(consumer_sentiment, config["consumer_sentiment_low_threshold"], config["consumer_sentiment_lookback_months"]),
        "real_rates": real_rates_negative(real_rates, config["real_rates_low_threshold"]),
        "inflation": inflation_expectations_high(inflation_exp, config["inflation_expectations_high_threshold"]),
        "oil_vol": oil_volatility_high(oil, config["oil_lookback_months"], config["oil_volatility_threshold_pct"]),
        "lei": leading_indicators_declining(leading_indicators, config["leading_indicators_lookback_months"], config["leading_indicators_decline_threshold_pct"]),
    }

    # --- Asset trend flags ---
    asset_flags = {}
    for t in tickers:
        asset_flags[t] = trend_break(t, start_date, pct_filter=trend_filter)

    print(summarize(flags, asset_flags, config))


if __name__ == "__main__":
    main()
