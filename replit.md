# Blueprint Trader AI

## Overview

Blueprint Trader AI is an automated trading signal bot designed to identify high-probability trading opportunities across multiple markets (forex, metals, indices, energies, crypto). The bot integrates with Discord for signal delivery and utilizes OANDA's practice API for market data. Its core purpose is to provide automated, risk-managed trading signals, adhering to 5%ers High Stakes challenge rules.

## 5%ERS HIGH STAKES 10K CHALLENGE (2-STEP)

This is specifically the **High Stakes 10K Challenge** - a 2-step evaluation to pass.

### Challenge Rules
| Step | Profit Target | On $10K Account |
|------|---------------|-----------------|
| **Step 1** | 8% | $800 profit needed |
| **Step 2** | 5% | $500+ profit (from new balance) |

**Risk Limits:**
- **Maximum Drawdown**: 10% ($1,000 - cannot drop below $9,000)
- **Daily Drawdown**: 5% ($500 max loss per day)
- **Minimum Profitable Days**: 3 days required per step
- **Maximum Trades Per Day**: 12

**Dynamic Risk Management (SAFE MODE - Updated Nov 2025):**
- **Base Risk**: 2.0% ($200/trade) when account is healthy
- **Reduced Risk**: 1.0% ($100/trade) at 2%+ drawdown
- **Minimum Risk**: 0.5% ($50/trade) at 4%+ drawdown
- **Max Total Exposure**: 4% (CRITICAL: ensures ALL concurrent SLs can't breach 5% daily DD)
- **Daily Loss Limit**: 5% (matches 5%ers rule)
- **Commissions**: 0.30% round-trip (0.15% entry + 0.15% exit) deducted from P/L
- When trades hit TP1, SL moves to BE reducing effective risk to ~30%
- **Safety Guarantee**: Even if ALL open trades hit SL simultaneously, max loss = 4% (under 5% daily limit)

## CURRENT LIVE STRATEGY: V3 Pro (November 2025)

### V3 Pro - Fibonacci-Based TradingView Approach (LIVE)
**Strategy aligned with user's TradingView setups**: NO RSI, NO MACD, NO SMC

| Component | Implementation |
|-----------|----------------|
| Timeframe | Daily only (2-8 day trade duration) |
| Entry Zones | Supply/Demand zones + 0.5-0.66 Fibonacci retracement (Optimal Entry Zone) |
| Weekly S/R | Weekly support/resistance confluence scoring |
| Monthly S/R | Optional - extracted from daily candle aggregation (if improves performance) |
| BoS Confirmation | Break of Structure validation before entry |
| Stop Loss | Swing high/low (1.0 Fib level) + ATR buffer |
| Take Profit | Fibonacci extensions: -0.25 (TP1), -0.68 (TP2), -1.0 (TP3) |
| Partial TP | 50% closed at 1.5R, SL moves to breakeven |
| Break-Even | Moves SL to entry at +1R |

**2024 Backtest Results (All Forex + Crypto 13-Asset Portfolio):**
- Total Trades: **929** (all assets 50+ trades/year)
- Total P/L: **+$62,600** from backtesting
- Challenge P/L: **+$86,295** with dynamic risk sizing
- Monthly Pass Rate: **90%** (9/10 months) with ZERO breaches
- Average Win Rate: **33.3%**
- Pass months: April, May, June, July, August, September, October, November, December
- Fail months: March (0 profitable days but passed Step 1)

### All-Forex + Crypto Portfolio
| Asset | Trades | Win% | Total R | P/L | Type |
|-------|--------|------|---------|-----|------|
| EUR_USD | 100 | 38.0% | 35.9R | $8,976 | Forex |
| EUR_NZD | 100 | 33.0% | 24.4R | $6,097 | Forex |
| AUD_USD | 75 | 29.3% | 18.0R | $4,492 | Forex |
| GBP_USD | 72 | 25.0% | 17.6R | $4,407 | Forex |
| USD_CAD | 63 | 33.3% | 16.8R | $4,192 | Forex |
| USD_JPY | 57 | 35.1% | 14.9R | $3,724 | Forex |
| USD_CHF | 71 | 25.4% | 12.2R | $3,056 | Forex |
| EUR_JPY | 59 | 27.1% | 6.6R | $1,651 | Forex |
| GBP_CAD | 52 | 32.7% | 10.1R | $2,513 | Forex |
| ETH_USD | 70 | 38.6% | 29.6R | $7,411 | Crypto |
| BTC_USD | 59 | 45.8% | 26.0R | $6,504 | Crypto |
| LTC_USD | 88 | 35.2% | 24.4R | $6,100 | Crypto |
| BCH_USD | 63 | 33.3% | 13.9R | $3,478 | Crypto |

## LEGACY STRATEGIES

### Strategy V3 - HTF S/R + BOS
**Status**: Superseded by V3 Pro

| Component | Implementation |
|-----------|----------------|
| Entry | Price at HTF S/R zone (Daily/Weekly) + BOS confirmation on H4 |
| BOS Detection | Two-stage: Soft BOS (wick touch) queues 3 bars, Confirmed BOS triggers entry |

### Strategy V4 - Archer Academy Style
**Status**: Superseded by V3 Pro

Based on Supply/Demand zones with Base identification (RBD/DBR/RBR/DBD patterns).

## MONTHLY PASS RATE REALITY

**Optimization Results (November 2025 - SAFE MODE):**
- Uses **4% max total exposure** to ensure account NEVER breaches
- Uses **concurrent exposure tracking** to prevent multiple SL breach
- V3 Pro optimized: Dynamic risk (2.0/1.0/0.5%) + 4% max exposure + partial TP at 1.0R
- **Commissions included**: 0.30% round-trip deducted from all trades
- **Safety First**: Even worst-case (all trades hit SL) = 4% loss (under 5% daily limit)
- Portfolio approach with multiple assets recommended for meeting "3 profitable days" requirement

## KEY FILES

| File | Purpose |
|------|---------|
| `strategy_v3_pro.py` | **CURRENT** - Daily S/D + Golden Pocket + Wyckoff |
| `strategy.py` | Wrapper for Discord bot scanning (uses V3 Pro) |
| `challenge_5ers_v3_pro.py` | V3 Pro challenge backtest runner |
| `challenge_risk_manager.py` | Dynamic position sizing with DD-adaptive risk |
| `strategy_v3.py` | Legacy HTF S/R + BOS signal generation |
| `strategy_v4_archer.py` | Legacy Archer Academy Supply/Demand zones |
| `challenge_simulator.py` | 5%ers rules simulation |
| `main.py` | Discord bot with slash commands and autoscan |

## User Preferences

- Preferred communication style: Simple, everyday language
- Strategy requirement: NO RSI, NO MACD, NO SMC
- Focus: Fibonacci-based TradingView approach with 0.5-0.66 entry zones and Fib extension TPs

## System Architecture

### Core Components
- **HTF S/R Zones**: Identified from Daily and Weekly candles
- **Supply/Demand Zones**: Base identification before impulse moves (Archer style)
- **Break of Structure (BOS)**: Two-stage confirmation with soft/confirmed detection
- **Structural Take Profits**: Pure price structure swing levels

### Data Layer
- OANDA v20 API for real-time and historical market data
- TTL-based intelligent caching system
- Live price integration for trade monitoring

### Discord Interface
- Professional embeds for trade setups
- 4-hour autoscan loop
- Slash commands for backtesting and analysis

## External Dependencies

### Services
- **Discord API:** Bot communication and signal delivery
- **OANDA v20 API:** Market data (practice endpoint)

### Environment Variables
- `DISCORD_BOT_TOKEN`: Discord bot authentication
- `OANDA_API_KEY`: OANDA API key
- `OANDA_ACCOUNT_ID`: OANDA account identifier

### Python Dependencies
- `discord-py`: Discord bot framework
- `pandas`: Data manipulation
- `requests`: HTTP client for OANDA API

### Dependency Management
- **`uv`:** Used with `pyproject.toml` and `uv.lock`
