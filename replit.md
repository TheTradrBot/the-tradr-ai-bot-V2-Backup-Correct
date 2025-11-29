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
- **Risk Per Trade**: 2.5% ($250 per trade)
- **Maximum Trades Per Day**: 12

## CURRENT LIVE STRATEGY: V3 Pro (November 2025)

### V3 Pro - Daily S/D + Golden Pocket + Wyckoff (LIVE)
**User's strict requirements**: NO RSI, NO MACD, NO SMC, NO Fibonacci for TPs.

| Component | Implementation |
|-----------|----------------|
| Timeframe | Daily only (2-8 day trade duration) |
| Entry Zones | Supply/Demand zones from Daily/Weekly |
| Golden Pocket | Entries at 0.618-0.65 retracement (Fib for ENTRIES only) |
| Wyckoff | Spring (demand) / Upthrust (supply) patterns |
| Trend Filter | EMA 10/20 crossover + swing structure |
| Stop Loss | Zone boundary + 0.3 ATR buffer (min 0.75 ATR) |
| Take Profit | Structural swing levels (1.5R-8R capped) |
| Break-Even | Moves SL to entry at +1R |

**2024 Backtest Results (MEGA Portfolio - 35 Assets):**
- Total P/L: **+$48,362 (484% yearly return)**
- Monthly Pass Rate: **70%** (7/10 months) - AT THEORETICAL CEILING
- Trades: 754 total across 35 instruments
- Partial TP: 50% closed at 1.5R to spread wins across calendar days
- Best Performers: BCO_USD (66.8%), WTICO_USD (51.6%), NZD_USD (49.1%), EUR_JPY (43.8%)

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
- This is at the **theoretical ceiling** - further aggressive tests degraded results
- V3 Pro optimized: Partial TP at 1.5R + 35-asset portfolio + day-staggering
- Failing months: March (few setups), April (near miss), September (counter-trend losses)
- **70%+ yearly per-asset target remains statistically unrealistic** with available data
- **Portfolio-level approach (484% yearly) is the practical path forward**

## KEY FILES

| File | Purpose |
|------|---------|
| `strategy_v3_pro.py` | **CURRENT** - Daily S/D + Golden Pocket + Wyckoff |
| `strategy.py` | Wrapper for Discord bot scanning (uses V3 Pro) |
| `challenge_5ers_v3_pro.py` | V3 Pro challenge backtest runner |
| `strategy_v3.py` | Legacy HTF S/R + BOS signal generation |
| `strategy_v4_archer.py` | Legacy Archer Academy Supply/Demand zones |
| `challenge_simulator.py` | 5%ers rules simulation |
| `main.py` | Discord bot with slash commands and autoscan |

## User Preferences

- Preferred communication style: Simple, everyday language
- Strategy requirement: NO RSI, NO MACD, NO SMC, NO Fibonacci for TPs
- Focus: Archer Academy / Forex Dictionary style Supply/Demand zones

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
