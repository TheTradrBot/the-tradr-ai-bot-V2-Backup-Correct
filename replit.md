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

**Dynamic Risk Management (Optimized with Concurrent Exposure Tracking):**
- **Base Risk**: 3.0% ($300/trade) when account is healthy
- **Reduced Risk**: 1.5% ($150/trade) at 2.5%+ drawdown
- **Minimum Risk**: 0.5% ($50/trade) at 5%+ drawdown
- **Max Total Exposure**: 7% (ensures multiple SLs can't breach 10% DD)
- **Daily Loss Limit**: 4% internal cap (safety buffer under 5% rule)
- **Fees**: 0.1% per trade deducted from P/L
- When trades hit TP1, SL moves to BE reducing effective risk to ~30%

## CURRENT LIVE STRATEGY: V3 Pro (November 2025)

### V3 Pro - Fibonacci-Based TradingView Approach (LIVE)
**Strategy aligned with user's TradingView setups**: NO RSI, NO MACD, NO SMC

| Component | Implementation |
|-----------|----------------|
| Timeframe | Daily only (2-8 day trade duration) |
| Entry Zones | Supply/Demand zones + 0.5-0.66 Fibonacci retracement (Optimal Entry Zone) |
| Weekly S/R | Weekly support/resistance confluence scoring |
| BoS Confirmation | Break of Structure validation before entry |
| Stop Loss | Swing high/low (1.0 Fib level) + ATR buffer |
| Take Profit | Fibonacci extensions: -0.25 (TP1), -0.68 (TP2), -1.0 (TP3) |
| Partial TP | 50% closed at 1.5R, SL moves to breakeven |
| Break-Even | Moves SL to entry at +1R |

**2024 Backtest Results (MEGA Portfolio - 35 Assets):**
- Total P/L: **+$69,398 (694% yearly return)**
- Monthly Pass Rate: **70%** (7/10 months) with ZERO breaches
- Trades: 754 total across 35 instruments
- Dynamic risk sizing: 3.0% base → 1.5% mid-DD → 0.5% deep-DD
- Concurrent exposure cap: 7% max (prevents multiple SL breach)
- Partial TP: 50% closed at 1R to protect capital + SL moved to entry+0.1R
- Pass months: May ($10K), June ($2K), July ($20K), August ($5K), October ($21K), November ($9K), December ($4K)
- Fail months: March, April (near-miss), September (sparse setups)

### Asset-Specific Configurations
| Asset | Confluence | Min RR | Notes |
|-------|------------|--------|-------|
| EUR_USD | 2 | 2.0 | Balanced |
| BCO_USD | 2 | 1.5 | Best performer (74.1%) |
| WTICO_USD | 3 | 3.0 | High RR swings |
| NZD_USD | 3 | 3.0 | Strong trends |
| EUR_JPY | 2 | 1.5 | Good volatility |
| ETH_USD | 3 | 2.5 | Crypto swing trades |

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

**Optimization Results (November 2025):**
- Achieved **70% monthly pass rate** (7/10 months) with mega portfolio approach
- Uses **concurrent exposure tracking** to prevent multiple SL breach
- V3 Pro optimized: Dynamic risk (3.0/1.5/0.5%) + 7% max exposure + partial TP at 1R
- Failing months: March (few setups), April (near-miss), September (counter-trend losses)
- **70%+ per-asset target remains unrealistic** - portfolio approach is required
- **Portfolio-level approach (694% yearly) achieves 70% pass rate with ZERO breaches**

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
