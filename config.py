# config.py
"""
Configuration for Blueprint Trader AI.

You keep:
- secrets (tokens, API keys) in Replit Secrets (env vars)
- public config (channel IDs, instruments, intervals) here
"""

import os

# How strict the confluence engine is.
# "standard"  = stricter, fewer trades (current behaviour).
# "aggressive" = more trades, looser filters.
SIGNAL_MODE = "standard"  # later you can change this to "aggressive"


# ==== Discord ====

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Replace these with your actual channel IDs (integers)
SCAN_CHANNEL_ID = 1442194985142190230        # <- scan channel ID
TRADES_CHANNEL_ID = 1442195008525565962      # <- trades channel ID
TRADE_UPDATES_CHANNEL_ID = 1438452127767859254  # <- trade updates channel ID

# Autoscan interval (hours)
SCAN_INTERVAL_HOURS = 4  # every 4H as per your spec


# ==== Data source: OANDA (practice) ====

OANDA_API_KEY = os.getenv("OANDA_API_KEY")          # set in Replit secrets
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")    # set in Replit secrets
OANDA_API_URL = "https://api-fxpractice.oanda.com"  # practice endpoint

# Granularity mapping for OANDA
GRANULARITY_MAP = {
    "M": "M",      # Monthly
    "W": "W",      # Weekly
    "D": "D",      # Daily
    "H4": "H4",    # 4-hour
}


# ==== Instruments & groups ====
# These are example sets. You can expand them later.

# OANDA FX pairs
FOREX_PAIRS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "USD_CHF",
    "USD_CAD",
    "NZD_USD",
    # Add more as needed or later fetch all from OANDA
]

# Metals (subset of commodities)
METALS = [
    "XAU_USD",  # Gold
    "XAG_USD",  # Silver
]

# Indices
INDICES = [
    "NAS100_USD",  # Nasdaq
    "SPX500_USD",  # S&P 500
]

# Energies
ENERGIES = [
    "WTICO_USD",   # WTI Crude
    "BCO_USD",     # Brent
]

# Crypto
CRYPTO_ASSETS = [
    "BTC_USD",
    "ETH_USD",
]

# Convenience groups

def all_market_instruments() -> list[str]:
    """All instruments Blueprint can scan."""
    return sorted(set(
        FOREX_PAIRS + METALS + INDICES + ENERGIES + CRYPTO_ASSETS
    ))
