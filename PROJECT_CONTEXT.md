# Finance Strategy - Project Context

## Overview
This is a **Financial Risk Dashboard** system that provides comprehensive market risk assessment through multiple indicators. It monitors market conditions and warns of potential risks by tracking macro-economic indicators and real-time panic signals.

## Architecture

### Core Modules (`core/`)
- Dashboard orchestration and configuration management
- `dashboard.py` - Main dashboard coordination
- `config.py` - Configuration loader and manager

### Data Sources (`data/`)
- `fred.py` - FRED economic data (Federal Reserve Economic Data)
- `yahoo.py` - Yahoo Finance market data
- `crypto.py` - Cryptocurrency data via CoinGecko API
- `realtime.py` - Real-time market data fetching
- `beacon.py` - (New/untracked file)

### Indicators (`indicators/`)
- `macro.py` - Macro-economic health indicators
- `panic.py` - Real-time market stress/panic indicators
- `technical.py` - Technical trend analysis
- `commodity.py` - Commodity price indicators
- `base.py` - Base indicator classes

## Key Features

### Panic Indicators (11 total)
- VIX spike (threshold: 40)
- VIX term structure flip
- SPY intraday drop (-4%)
- Breadth washout (15% threshold)
- "Everything down" correlation (95%)
- HYG stress (-2.5% with 2x volume)
- ETF NAV dislocation (1%)
- Treasury shock (20 basis points)
- DXY spike (1.5%)
- Oil crash (-8%)
- Copper crash (-3%)
- Crypto correlation breakdown

### Macro Indicators
- Consumer sentiment
- Inflation expectations
- Real rates
- Building permits
- Leading economic indicators
- High yield spreads
- Validator activity (crypto)
- Commodity inflation (energy, food, metals, lumber)

### Panic Scoring System
- **Normal**: 0-1 indicators active
- **Elevated**: 2-3 indicators active
- **High**: 4-5 indicators active
- **Extreme**: 6+ indicators active

## Configuration
- Default config: `config.json`
- Conservative mode: `config_conservative.json`
- API keys: FRED, CoinGecko
- Configurable thresholds for all indicators

## Usage Modes

### CLI Commands
```bash
python main.py                    # Full dashboard (default config)
python main.py --conservative     # Conservative thresholds
python main.py --details          # Full dashboard with details
python main.py --quick            # Quick panic check only
python main.py --panic-only       # Panic indicators only
python main.py --macro-only       # Macro indicators only
python main.py --tickers SPY QQQ  # Custom trend tickers
```

## Current Development Status
- Version: 2.0.0 (modular architecture)
- Recent commits focused on adding more indicators
- Modified files: `core/dashboard.py`, `indicators/macro.py`
- New untracked file: `data/beacon.py`

## Technical Details
- Python 3.x
- Dependencies: yfinance, FRED API, CoinGecko API
- Refresh interval: 15 minutes (configurable)
- Historical data start: 2019-01-01
