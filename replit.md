# Blueprint Trader AI

## Overview

Blueprint Trader AI is an automated trading signal bot designed to identify high-probability trading opportunities across multiple markets (forex, metals, indices, energies, crypto). The bot integrates with Discord for signal delivery and utilizes OANDA's practice API for market data. Its core purpose is to provide automated, risk-managed trading signals, adhering to 5%ers High Stakes challenge rules.

## 5%ERS 10K CHALLENGE IMPLEMENTATION

### Challenge Rules (Implemented in `challenge_5ers.py`)
- **Step 1**: 8% profit target ($800 on $10,000 account)
- **Step 2**: 5% profit target (based on new starting balance after Step 1)
- **Maximum Drawdown**: 10% ($1,000 - cannot drop below $9,000)
- **Daily Drawdown**: 5% ($500 max loss per day)
- **Minimum Profitable Days**: 3 days required per step
- **Risk Per Trade**: 2.5% ($250 per trade)
- **Maximum Trades Per Day**: 12

### CURRENT OPTIMIZED Strategy (Active) - Regime-Aware Multi-Strategy
The strategy uses market regime detection to apply different trading approaches:

**Regime Detection:**
- **TRENDING**: Price range >3% with EMA divergence >0.5%
- **RANGING**: Price range <1.5%
- **VOLATILE**: All other conditions

**Strategy per Regime:**
| Regime | Entry Logic | R:R | ATR Mult | Notes |
|--------|-------------|-----|----------|-------|
| TRENDING | EMA crossover + momentum | 3.0:1 | 0.5x | Ride trends |
| RANGING | RSI extremes (<30/>70) | 1.5:1 | 0.8x | Mean reversion |
| VOLATILE | Breakout candles (>60% body) | 2.5:1 | 0.6x | Momentum plays |
| FALLBACK | Extreme RSI/EMA bounce | 2.0:1 | 0.7x | Catch opportunities |

**SMC Confluence Filters:**
- Order Blocks (OB), Fair Value Gaps (FVG), Liquidity Sweeps

### Latest Backtest Results (Sep 2024 - Oct 2025)
| Period | Total Net | Avg Monthly | Pass Rate | Best Month | $3k+ Months |
|--------|-----------|-------------|-----------|------------|-------------|
| 14 Months | **+$13,798** | **+$986** | 21% (3/14) | +$8,067 (Oct 2025) | 3/14 (21%) |

### Monthly Breakdown
| Month | Trades | Win Rate | Net P/L | Result |
|-------|--------|----------|---------|--------|
| Oct 2025 | 35 | 54.3% | +$8,067 | **PASSED** |
| Nov 2024 | 66 | 39.4% | +$6,168 | **PASSED** |
| Jan 2025 | 29 | 48.3% | +$4,605 | **PASSED** |
| May 2025 | 37 | 37.8% | +$1,847 | Step 1 only |
| Sep 2025 | 38 | 34.2% | +$112 | Step 1 only |

### Key Findings
- **Trade Volume Matters**: Months with 29+ trades and 33%+ win rate tend to be profitable
- **Low Trade Months**: Months with <10 trades consistently lose (~-$800)
- **Best Configuration**: 2.5% risk, 12 trades/day max, multi-strategy approach
- **Mathematical Reality**: $3k+ profit EVERY month is not achievable due to market variance

### Trading Fees (Per Asset)
| Asset Type | Spread | Commission | Avg Fee/Trade |
|------------|--------|------------|---------------|
| Forex Majors | 1-1.5 pips | $7/lot | ~$12-17 |
| Cross Pairs | 1.5-2 pips | $7/lot | ~$15-20 |
| XAU/USD | 25 pips | $6/lot | ~$3.50 |
| Crypto | 0.20% | - | ~$20-25 |
| Indices | 0.5-1 pt | $0 | ~$0.50 |

### Assets Traded (10 total)
Forex: EUR/USD, GBP/USD, USD/CHF, AUD/USD, USD/CAD, EUR/GBP, GBP/JPY
Metals: XAU/USD (Gold)
Crypto: ETH/USD
Indices: NAS100

### Lot Size Formula
```
lot_size = risk_usd / (sl_pips × pip_value)
```
Example: $250 risk / (15 pips × $10/pip) = 1.67 lots

### Discord Commands
- `/pass <month> <year>`: Check if challenge would be passed for given month
- Exports CSV with trade details including lot sizes

## OPTIMIZED SMC STRATEGY RESULTS (2024-2025 Backtest)

**12/16 ASSETS HIT +50% TARGET (Smart Money Concepts):**

| Asset | Annual Return | Trades | Win Rate | R:R | Strategy |
|-------|---------------|--------|----------|-----|----------|
| EUR_USD | +70% | 294 | 9.5% | 12:1 | SMC + High R:R |
| GBP_USD | +80% | 284 | 9.9% | 12:1 | SMC + High R:R |
| USD_JPY | +87% | 265 | 12.1% | 10:1 | SMC + High R:R |
| NZD_USD | +90% | 313 | 9.9% | 12:1 | SMC + High R:R |
| USD_CAD | +127% | 287 | 16.0% | 8:1 | SMC |
| EUR_GBP | +83% | 288 | 18.4% | 6:1 | SMC |
| EUR_JPY | +59% | 277 | 20.2% | 5:1 | SMC |
| XAU_USD | +97% | 274 | 19.3% | 6:1 | SMC |
| BTC_USD | +55% | 286 | 10.8% | 10:1 | SMC + High R:R |
| ETH_USD | +114% | 276 | 10.9% | 12:1 | SMC + High R:R |
| SPX500_USD | +94% | 275 | 14.9% | 8:1 | SMC |
| NAS100_USD | +102% | 304 | 19.1% | 6:1 | SMC |

**Total Portfolio Return: +$1,075,000 (+1075%)**

### Strategy Files
- `strategy_smc.py` - SMC (Smart Money Concepts) optimized strategy with Order Blocks, FVG, Liquidity Sweeps
- `strategy_final.py` - RSI mean reversion multi-strategy implementation
- `strategy_optimized.py` - RSI(2) mean reversion base strategy
- `strategy_v3.py` - Original V3 S/R + Fib strategy

### SMC Components
- **Order Blocks (OB)**: Last bullish/bearish candle before impulse move
- **Fair Value Gaps (FVG)**: 3-candle price imbalance zones
- **Liquidity Sweeps**: Price breaks swing high/low then reverses
- **Trend Filter**: EMA-based trend confirmation

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Components and Design

The bot's architecture is built around a modular strategy pipeline, allowing for independent trading modules like `Reversal` (HTF S/R mean-reversion) and `Trend` (EMA pullback continuation). It uses a 7-pillar confluence system for signal generation, evaluated across multiple timeframes (Monthly, Weekly, Daily, H4).
A `ScanResult` dataclass encapsulates setup information.
The `Backtest Engine` ensures walk-forward simulation without look-ahead bias and conservative exit logic.
A robust `Data Layer` integrates with the OANDA v20 API, supported by an `Intelligent Caching System` with TTL-based, thread-safe operations for efficient API response management.
Professional Discord embed formatting is handled by `discord_output.py` and `formatting.py`.
The main `Bot` (`main.py`) manages Discord slash commands, a 4-hour autoscan loop, and trade tracking.
The system incorporates a 5%ers 100K High Stakes risk model for position sizing, including calculations for account size, risk percentage, USD risk, and lot size. Trade exits are laddered (TP1 @ 50%, TP2 @ 30%, Runner @ 20% with trailing SL).
The bot also includes features for forex holiday filtering, price validation against reference data, and an optimization framework for strategy parameter tuning.

### UI/UX Decisions

Discord is the primary interface for user interaction, providing professional embeds for trade setups, updates, and backtest results. Command-line interactions are available for backtesting and optimization.

### Technical Implementations & Feature Specifications

- **7 Pillars of Confluence:** HTF Bias, Location (S/R, supply/demand), Fibonacci (50%-79.6%), Liquidity, Structure, Confirmation (4H BOS, momentum), and R:R (min 1.5R).
- **Trade Status Levels:** ACTIVE (entry triggered), WATCHING (waiting for confirmation), SCAN (low confluence).
- **Risk Management:** Configurable `ACCOUNT_SIZE`, `RISK_PER_TRADE_PCT`, `MAX_DAILY_LOSS`, `MAX_TOTAL_DRAWDOWN`, `MAX_OPEN_RISK`.
- **Position Sizing:** Dynamically calculates lot sizes based on account risk, stop loss distance, and pip value.
- **Strategy Optimization:** Includes `data_loader.py` for CSV historical data, `strategy_core.py` for parameterized signal generation and simulation, `backtest_engine.py` for comprehensive performance metrics, `optimizer.py` for parameter search, and `report.py` for analysis.
- **Live Price Integration:** Trade entries and TP/SL monitoring use live OANDA prices, preventing stale data issues.
- **Caching:** TTL-based caching for OANDA API responses (Monthly: 1hr, Weekly: 30min, Daily: 10min, H4: 5min).
- **Discord Commands:** Extensive commands for scanning, trading, analysis (backtest, export), and system management.

### System Design Choices

- **Modular Strategy:** Allows for independent development and deployment of trading modules.
- **Single Source of Truth:** Backtesting functions leverage `strategy_core.py` to ensure consistency between backtesting and live operations.
- **Conservative Exit Logic:** Prioritizes realistic backtest results by checking SL before TP on the same bar and implementing trailing stops.
- **No Look-Ahead Bias:** Implemented in the backtesting engine for accurate performance evaluation.

## External Dependencies

### Services

- **Discord API:** Used for bot communication and delivering trade signals and updates.
- **OANDA v20 API:** Provides real-time and historical market data (utilizes the practice endpoint).

### Environment Variables

- `DISCORD_BOT_TOKEN`: Required for Discord bot authentication.
- `OANDA_API_KEY`: OANDA API key (enables autoscan functionality).
- `OANDA_ACCOUNT_ID`: OANDA account identifier.
- `USE_OPTIMIZED_STRATEGY`: Toggle for using optimized strategy parameters.

### Python Dependencies

- `discord-py`: Asynchronous framework for Discord bot development.
- `pandas`: Library for data manipulation and analysis.
- `requests`: HTTP client for interacting with the OANDA API.

### Dependency Management

- **`uv`:** Used with `pyproject.toml` and `uv.lock` to ensure consistent dependency versions across all environments.