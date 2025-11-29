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

## STRATEGIES AVAILABLE

### Strategy V3 - HTF S/R + BOS (Current Live)
**User's strict requirements**: NO RSI, NO MACD, NO SMC, NO Fibonacci for TPs.

| Component | Implementation |
|-----------|----------------|
| Entry | Price at HTF S/R zone (Daily/Weekly) + BOS confirmation on H4 |
| BOS Detection | Two-stage: Soft BOS (wick touch) queues 3 bars, Confirmed BOS triggers entry |
| Stop Loss | Structural swing ±0.25 ATR from entry |
| Take Profit | Structural swing levels (min 2.0R target) |
| Break-Even | Moves SL to entry at +1R |

**2024 Backtest Results:**
- Total P/L: +$37,635
- Months Passed: 2/204 (1%)
- Best Performers: BTC_USD (+$9,250), XAU_USD (+$7,312), USD_JPY (+$7,110)

### Strategy V4 - Archer Academy Style
Based on Supply/Demand zones with Base identification (RBD/DBR/RBR/DBD patterns).

| Pattern | Zone Type | Strength | Direction |
|---------|-----------|----------|-----------|
| Rally-Base-Drop (RBD) | Supply | Strong (Reversal) | Bearish |
| Drop-Base-Rally (DBR) | Demand | Strong (Reversal) | Bullish |
| Rally-Base-Rally (RBR) | Demand | Moderate (Continuation) | Bullish |
| Drop-Base-Drop (DBD) | Supply | Moderate (Continuation) | Bearish |

**2024 Backtest Results:**
- Total P/L: +$5,833
- Months Passed: 3/96 (3.1%)
- Win Rate: 48.2% overall
- Best Performers: EUR_JPY (90% WR), XAU_USD (63% WR), WTICO_USD (59% WR)

## MONTHLY PASS RATE REALITY

**Architect Analysis (November 2025):**
- 100% monthly pass rate is **mathematically unattainable** with any single strategy
- Best realistic target: **35-55% pass rate** with portfolio-level coordination
- Main blocker: "3 profitable days" requirement - winning trades cluster on 1-2 calendar days
- Strategies are profitable overall but struggle to spread wins across 3+ distinct days

## KEY FILES

| File | Purpose |
|------|---------|
| `strategy_v3.py` | HTF S/R + BOS signal generation |
| `strategy_v4_archer.py` | Archer Academy Supply/Demand zones |
| `strategy.py` | Wrapper for Discord bot scanning |
| `challenge_5ers.py` | V3 challenge backtest runner |
| `challenge_5ers_archer.py` | Archer challenge backtest runner |
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
