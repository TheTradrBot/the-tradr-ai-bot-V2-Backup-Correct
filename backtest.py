# backtest.py
"""
Very lightweight placeholder backtest for Blueprint Trader AI.

Important:
- This does NOT simulate the full Weekly→Daily→4H Blueprint logic candle by candle.
- It only looks at recent Daily data for the asset and returns coarse, illustrative stats.
- Use it as a rough feel / demo, not as real performance analytics.
"""

from __future__ import annotations

from typing import Dict, List
from data import get_ohlcv


def run_backtest(asset: str, period: str) -> Dict:
    """
    Run a very simplified backtest over recent Daily data.

    Returns a dict with keys:
      - asset
      - period
      - total_trades
      - win_rate
      - net_return_pct
      - trades: list of { "exit_reason": "TP1"|"TP2"|"TP3"|"SL" }
      - notes
    """
    # Pull up to 260 Daily candles (~1 year of data depending on asset)
    candles = get_ohlcv(asset, timeframe="D", count=260)

    if not candles or len(candles) < 50:
        # Not enough data for anything meaningful
        return {
            "asset": asset,
            "period": period if period else "Recent Daily data",
            "total_trades": 0,
            "win_rate": 0.0,
            "net_return_pct": 0.0,
            "trades": [],
            "notes": (
                "Not enough Daily candles to build even a rough Blueprint-style swing backtest. "
                "This is a placeholder; full Weekly→Daily→H4 simulation is not implemented yet."
            ),
        }

    # --- Super crude heuristic for trade count ---
    # Idea: one potential swing trade per ~40 Daily candles.
    total_trades = max(1, len(candles) // 40)

    # --- Fake but reasonable stats for now ---
    # Assume a moderate edge, e.g. ~55% winrate on 1R-ish average.
    win_rate = 55.0
    net_return_per_trade = 0.3  # 0.3% equity per trade at 1% risk (very approximate)
    net_return_pct = total_trades * net_return_per_trade

    # --- Build a dummy trade list for TP/SL breakdown ---
    trades: List[Dict] = []

    # Distribute exits:
    #  40% TP1, 30% TP2, 10% TP3, 20% SL
    tp1 = int(total_trades * 0.4)
    tp2 = int(total_trades * 0.3)
    tp3 = int(total_trades * 0.1)
    sl = max(0, total_trades - tp1 - tp2 - tp3)

    for _ in range(tp1):
        trades.append({"exit_reason": "TP1"})
    for _ in range(tp2):
        trades.append({"exit_reason": "TP2"})
    for _ in range(tp3):
        trades.append({"exit_reason": "TP3"})
    for _ in range(sl):
        trades.append({"exit_reason": "SL"})

    notes = (
        "Placeholder backtest: estimated number of Blueprint-style swing trades based on the "
        "length of the recent Daily sequence. Stats are coarse and do NOT represent a true "
        "candle-by-candle Weekly→Daily→H4 Blueprint simulation. "
        "Treat this as a rough illustration only."
    )

    return {
        "asset": asset,
        "period": period if period else "Recent Daily data",
        "total_trades": total_trades,
        "win_rate": win_rate,
        "net_return_pct": net_return_pct,
        "trades": trades,
        "notes": notes,
    }
