# data.py
"""
Data access layer for Blueprint Trader AI.

Currently uses OANDA v20 REST API for OHLCV candles.
"""

import datetime as dt
from typing import List, Dict, Any, Optional

import requests

from config import OANDA_API_KEY, OANDA_API_URL, GRANULARITY_MAP


def _oanda_headers() -> Dict[str, str]:
    if not OANDA_API_KEY:
        raise ValueError("OANDA_API_KEY not set in environment (Replit secrets).")
    return {"Authorization": f"Bearer {OANDA_API_KEY}"}


def get_ohlcv(
    instrument: str,
    timeframe: str = "D",
    count: int = 200,
) -> List[Dict[str, Any]]:
    """
    Fetch OHLCV candles from OANDA for a given instrument and timeframe.

    timeframe: "D", "H4", "W", "M"
    Returns a list of dicts:
    {
      "time": datetime,
      "open": float,
      "high": float,
      "low": float,
      "close": float,
      "volume": float,
    }
    """

    granularity = GRANULARITY_MAP.get(timeframe, timeframe)
    url = f"{OANDA_API_URL}/v3/instruments/{instrument}/candles"

    params = {
        "granularity": granularity,
        "count": count,
        "price": "M",   # mid prices
    }

    headers = _oanda_headers()
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        print(f"[data.get_ohlcv] Error {resp.status_code} for {instrument}, {timeframe}: {resp.text}")
        return []

    data = resp.json()
    candles = []

    for c in data.get("candles", []):
        if not c.get("complete", True):
            continue
        time_str = c["time"]
        # trim trailing Z and nanos
        t = time_str.split(".")[0].replace("Z", "")
        time_dt = dt.datetime.fromisoformat(t)

        mid = c["mid"]
        candles.append({
            "time": time_dt,
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": float(c.get("volume", 0)),
        })

    return candles
