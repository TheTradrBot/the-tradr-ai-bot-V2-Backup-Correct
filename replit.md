# Blueprint Trader AI

## Overview

Blueprint Trader AI is an automated trading signal bot designed to identify high-probability trading opportunities across multiple markets (forex, metals, indices, energies, crypto). The bot integrates with Discord for signal delivery and utilizes OANDA's practice API for market data. Its core purpose is to provide automated, risk-managed trading signals, adhering to 5%ers High Stakes challenge rules.

## 5%ERS HIGH STAKES 10K CHALLENGE (2-STEP)

This is specifically the **High Stakes 10K Challenge** - a 2-step evaluation to pass.

### Challenge Rules (Implemented in `challenge_5ers.py` and `challenge_simulator.py`)
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

## CURRENT Strategy - V3 HTF S/R + BOS (NO RSI, NO SMC, NO Fibonacci TPs)

**User's strict requirements**: Strategy uses ONLY HTF S/R zones + Break of Structure (BOS) confirmation with structural take profits. NO RSI, NO SMC, NO Fibonacci for TPs.

### V3 Strategy Components (strategy_v3.py)
- **Entry**: Price at HTF S/R zone (Daily/Weekly) + BOS confirmation on H4
- **BOS Detection**: Two-stage confirmation - Soft BOS (wick touch) queues signal for 3 bars, Confirmed BOS (close break or engulfing) triggers entry
- **Stop Loss**: Structural swing ±0.25 ATR from entry
- **Take Profit**: Structural swing levels (min 2.0R target)
- **Confluence Scoring**: Zone (+1) + HTF Bias aligned (+1) + BOS (+2 confirmed/+1 soft) + Momentum (+1)
- **Break-Even**: Moves SL to entry at +1R

### Current Parameters
| Setting | Value |
|---------|-------|
| Min R:R | 2.0 |
| Min Confluence | 2 |
| Cooldown | 4 bars |
| Max Daily Trades | 2 per asset |
| Zone Tolerance | 0.5% daily, 0.8% weekly |

### Latest Backtest Results (2024, 17 Assets)
| Metric | Value |
|--------|-------|
| **Total P/L** | **+$37,635** |
| **Avg Monthly** | +$184 |
| **Months Passed** | 2/204 (1%) |
| **Best Performers** | BTC_USD (+$9,250), XAU_USD (+$7,312), USD_JPY (+$7,110) |

### Monthly Pass Rate Reality
**Architect Analysis (November 2025):**
- 100% monthly pass rate (204/204) is **mathematically unattainable** with any single strategy
- Best realistic target: **35-55% pass rate** with portfolio-level coordination
- Main blocker: "3 profitable days" requirement - winning trades cluster on 1-2 calendar days
- Strategy is profitable overall but struggles to spread wins across 3+ distinct days

### Recommended Next Steps
1. **Portfolio Scheduler**: Coordinate entries across multiple assets to engineer 3+ profitable days
2. **Session-Aware Spacing**: Stagger executions by trading session (Asia/London/NY)
3. **Partial Profit Banking**: Scale out 50% at first target to bank intra-day wins

### Key Files
| File | Purpose |
|------|---------|
| `strategy_v3.py` | **ACTIVE** - HTF S/R + BOS signal generation |
| `challenge_5ers.py` | Challenge backtest runner |
| `challenge_simulator.py` | 5%ers rules simulation with position sizing |
| `backtest_v3.py` | V3 strategy backtesting engine |
| `main.py` | Discord bot with slash commands and autoscan |

## User Preferences

Preferred communication style: Simple, everyday language.
Strategy requirement: NO RSI, NO SMC, NO Fibonacci for TPs - only HTF S/R + BOS.

## System Architecture

### Core Components and Design

The bot's architecture is built around a modular strategy pipeline using the V3 approach:
- **HTF S/R Zones**: Identified from Daily and Weekly candles with configurable tolerance
- **Break of Structure (BOS)**: Two-stage confirmation with soft/confirmed detection
- **Structural Take Profits**: Pure price structure swing levels (no Fibonacci)
- **TradeSignal Dataclass**: Encapsulates signal information including entry, SL, TP, confluence score, status

A robust **Data Layer** integrates with the OANDA v20 API, supported by an **Intelligent Caching System** with TTL-based, thread-safe operations.
Professional Discord embed formatting is handled by `discord_output.py` and `formatting.py`.
The main **Bot** (`main.py`) manages Discord slash commands, a 4-hour autoscan loop, and trade tracking.

### UI/UX Decisions

Discord is the primary interface for user interaction, providing professional embeds for trade setups, updates, and backtest results. Command-line interactions are available for backtesting and optimization.

### Technical Implementations & Feature Specifications

- **Signal Statuses**: ACTIVE (confirmed BOS, enter now), WATCHING (soft BOS, waiting for confirmation)
- **Risk Management**: $250 per trade (2.5% of $10K), max 10% drawdown, max 5% daily drawdown
- **Position Sizing**: Dynamically calculates lot sizes based on account risk, stop loss distance, and pip value
- **Live Price Integration**: Trade entries and TP/SL monitoring use live OANDA prices

### System Design Choices

- **Conservative Exit Logic**: Checks SL before TP on same bar, implements break-even at +1R
- **No Look-Ahead Bias**: Implemented in backtesting engine for accurate performance evaluation
- **Pending Signal Queue**: Soft BOS signals are queued for 3 bars awaiting confirmed BOS

## External Dependencies

### Services

- **Discord API:** Used for bot communication and delivering trade signals and updates.
- **OANDA v20 API:** Provides real-time and historical market data (utilizes the practice endpoint).

### Environment Variables

- `DISCORD_BOT_TOKEN`: Required for Discord bot authentication.
- `OANDA_API_KEY`: OANDA API key (enables autoscan functionality).
- `OANDA_ACCOUNT_ID`: OANDA account identifier.

### Python Dependencies

- `discord-py`: Asynchronous framework for Discord bot development.
- `pandas`: Library for data manipulation and analysis.
- `requests`: HTTP client for interacting with the OANDA API.

### Dependency Management

- **`uv`:** Used with `pyproject.toml` and `uv.lock` to ensure consistent dependency versions across all environments.
