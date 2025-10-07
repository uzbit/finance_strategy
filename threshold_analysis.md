# Threshold Analysis: Regular vs Conservative

## Summary
Analysis of all thresholds to ensure conservative values are tighter (more sensitive to warnings).

## Legend
- ✅ = Conservative is correctly tighter
- ❌ = Conservative is WRONG (looser or missing)
- ⚠️ = Needs verification

---

## Trend Filters (warning when price < SMA * (1 - filter))
| Parameter | Regular | Conservative | Direction | Status |
|-----------|---------|--------------|-----------|--------|
| trend_filter_pct | 0.02 (2%) | 0.01 (1%) | Lower = tighter | ✅ |
| short_trend_filter_pct | 0.015 (1.5%) | 0.008 (0.8%) | Lower = tighter | ✅ |

---

## Macro Indicators
| Parameter | Regular | Conservative | Condition | Status |
|-----------|---------|--------------|-----------|--------|
| high_yield_threshold_pts | 1.0 | 0.75 | Spread increase > threshold | ✅ Lower triggers earlier |
| vix_high_threshold | 25.0 | 20.0 | VIX > threshold | ✅ Lower triggers earlier |
| consumer_sentiment_low_threshold | 80.0 | 85.0 | Sentiment < threshold | ✅ Higher triggers earlier |
| real_rates_low_threshold | 0.0 | 0.5 | Real rates < threshold | ✅ Higher triggers earlier |
| inflation_expectations_high_threshold | 3.0 | 2.5 | Inflation > threshold | ✅ Lower triggers earlier |
| oil_volatility_threshold_pct | 20.0 | 15.0 | Volatility > threshold | ✅ Lower triggers earlier |
| leading_indicators_decline_threshold_pct | -2.0 | -1.5 | Decline <= threshold | ✅ -1.5 is less negative (earlier) |
| validator_entry_pct_threshold | 80.0 | **MISSING** | Entry % > threshold | ❌ Should be ~70.0 |
| validator_exit_pct_threshold | 80.0 | **MISSING** | Exit % > threshold | ❌ Should be ~70.0 |

---

## Commodity Inflation (all are: inflation > threshold)
| Parameter | Regular | Conservative | Status |
|-----------|---------|--------------|--------|
| energy_inflation_threshold_pct | 15.0 | 10.0 | ✅ |
| natgas_inflation_threshold_pct | 25.0 | 20.0 | ✅ |
| food_inflation_threshold_pct | 20.0 | 15.0 | ✅ |
| metals_inflation_threshold_pct | 12.0 | 8.0 | ✅ |
| import_inflation_threshold_pct | 10.0 | 6.0 | ✅ |
| import_inflation_amber_threshold | 2.0 | 1.5 | ✅ |
| import_inflation_red_threshold | 4.0 | 3.0 | ✅ |
| import_inflation_3m_threshold | 6.0 | 4.5 | ✅ |
| lumber_inflation_threshold_pct | 25.0 | 18.0 | ✅ |
| composite_inflation_threshold_pct | 12.0 | 9.0 | ✅ |

---

## Technical Indicators
| Parameter | Regular | Conservative | Meaning | Status |
|-----------|---------|--------------|---------|--------|
| trend_reclaim_buffer_pct | 0.02 | 0.015 | Crossover detection buffer | ✅ Tighter |
| trend_reclaim_lookback_days | 5 | 7 | Days to look for cross | ✅ Longer catches more |
| momentum_flip_lookback_days | 10 | 15 | Days to look for flip | ✅ Longer catches more |
| bollinger_std | 2.0 | 1.8 | Standard deviations | ✅ Tighter bands |
| squeeze_threshold_pct | 5.0 | 4.0 | BB width < threshold | ✅ Lower triggers earlier |
| expansion_threshold_pct | 3.0 | 2.5 | BB width > threshold | ✅ Lower triggers earlier |
| breadth_healthy_threshold | 60.0 | 65.0 | % stocks > 20d MA | ✅ Higher is stricter |

---

## Panic Thresholds
| Parameter | Regular | Conservative | Condition | Status |
|-----------|---------|--------------|-----------|--------|
| vix_spike | 40 | 30 | VIX > threshold | ✅ |
| spy_intraday_drop | -4.0% | -3.0% | Drop < threshold | ✅ -3% is less severe (earlier) |
| breadth_washout | 15% | 20% | Breadth < threshold | ✅ 20% catches earlier |
| everything_down | 95% | 90% | Correlation > threshold | ✅ |
| hyg_stress_pct | -2.5% | -2.0% | Drop < threshold | ✅ -2% is less severe (earlier) |
| hyg_volume_multiple | 2.0x | 1.75x | Volume > threshold | ✅ |
| etf_nav_dislocation | 1.0% | 0.75% | Dislocation > threshold | ✅ |
| treasury_shock_bp | 20bp | 15bp | Shock > threshold | ✅ |
| dxy_spike | 1.5% | 1.2% | Spike > threshold | ✅ |
| oil_crash | -8.0% | -6.0% | Drop < threshold | ✅ -6% is less severe (earlier) |
| copper_crash | -3.0% | -2.5% | Drop < threshold | ✅ -2.5% is less severe (earlier) |
| crypto_correlation | -3.0% | -2.5% | Drop < threshold | ✅ -2.5% is less severe (earlier) |
| stablecoin_depeg | 0.995 | 0.996 | Price < threshold | ✅ Higher catches earlier |
| stablecoin_depeg_minutes | 15 | 10 | Duration >= threshold | ✅ Shorter duration (earlier) |

---

## Panic Scoring Levels
| Level | Regular Range | Conservative Range | Status |
|-------|---------------|-------------------|--------|
| Normal | 0-1 | 0 only | ✅ Stricter |
| Elevated | 2-3 | 1-2 | ✅ Escalates faster |
| High | 4-5 | 3-4 | ✅ Escalates faster |
| Extreme | 6+ | 5+ | ✅ Escalates faster |

---

## Issues Found

### ❌ Missing Parameters in Conservative Config
1. **validator_entry_pct_threshold**: Missing (should be ~70.0, down from 80.0)
2. **validator_exit_pct_threshold**: Missing (should be ~70.0, down from 80.0)

### ✅ All Other Thresholds
All other conservative thresholds are correctly configured to be more sensitive/stricter than regular thresholds.

---

## Recommended Fixes

Add to `config_conservative.json`:
```json
"validator_entry_pct_threshold": 70.0,
"validator_exit_pct_threshold": 70.0,
```
