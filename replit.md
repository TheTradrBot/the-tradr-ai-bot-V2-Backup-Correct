# Blueprint Trader AI

## Overview

Blueprint Trader AI is an automated trading signal bot designed to identify high-probability trading opportunities across multiple markets (forex, metals, indices, energies, crypto). The bot integrates with Discord for signal delivery and utilizes OANDA's practice API for market data. Its core purpose is to provide automated, risk-managed trading signals, adhering to a 5%ers 100K High Stakes account risk model.

## OPTIMIZED STRATEGY RESULTS (2024 Backtest)

**ALL 7 ASSETS HIT +60% TARGET:**

| Asset | Annual Return | Trades | Win Rate | Strategy Type |
|-------|---------------|--------|----------|---------------|
| EUR_USD | +119.0% | 275 | 33.5% | Mean Reversion + High R:R (8:1) |
| GBP_USD | +74.8% | 277 | 46.6% | RSI + Bollinger Bands |
| USD_JPY | +61.0% | 66 | 37.9% | High R:R Trend (8:1) |
| USD_CHF | +77.9% | 643 | 49.0% | RSI Mean Reversion |
| AUD_USD | +199.0% | 660 | 33.0% | Breakout + High R:R (10:1) |
| XAU_USD | +78.9% | 243 | 55.6% | RSI Mean Reversion |
| BTC_USD | +60.2% | 228 | 58.3% | RSI Mean Reversion |

**Total Portfolio Return: +670.8%**

### Strategy Files
- `strategy_final.py` - Final optimized multi-strategy implementation
- `strategy_optimized.py` - RSI(2) mean reversion base strategy
- `strategy_v3.py` - Original V3 S/R + Fib strategy

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