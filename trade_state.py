# trade_state.py
"""
In-memory trade registry for Blueprint Trader AI.

We store active trade ideas (6/7+ confluence) so that:
- /trade can show current active trades
- autoscan can detect TP/SL hits and send updates

For now, state is in-memory only (reset when the bot restarts).
"""

from typing import Dict, List, Tuple

from strategy import ScanResult
from data import get_ohlcv


_active_trades: Dict[str, ScanResult] = {}


def _key(res: ScanResult) -> str:
    return f"{res.symbol}:{res.direction}"


def register_trade(res: ScanResult) -> None:
    """
    Register or update an active trade idea.
    Newest signal for a (symbol, direction) pair overwrites older one.
    """
    _active_trades[_key(res)] = res


def list_trades() -> List[ScanResult]:
    """
    Return a list of currently tracked trades.
    """
    return list(_active_trades.values())


def evaluate_trades_for_updates() -> List[Tuple[ScanResult, str, float, float]]:
    """
    Check each active trade for TP/SL hits using the latest H4 candle.

    Returns a list of events:
      (ScanResult, event_type, event_price, event_rr)

    event_type in {"TP1", "TP2", "TP3", "SL"}
    event_rr is the R-multiple at that level (based on entry & SL), or NaN.
    """
    events: List[Tuple[ScanResult, str, float, float]] = []

    for trade in _active_trades.values():
        if trade.is_closed:
            continue

        candles = get_ohlcv(trade.symbol, timeframe="H4", count=1)
        if not candles:
            continue

        c = candles[-1]
        high = c["high"]
        low = c["low"]

        entry = trade.entry
        sl = trade.stop_loss

        if entry is None or sl is None:
            continue

        # Compute risk (for RR calculation)
        if trade.direction == "bullish":
            risk = entry - sl
        else:
            risk = sl - entry

        if risk <= 0:
            risk = None  # invalid for RR, but we can still send update
        # ---- Bearish or Bullish logic ----

        if trade.direction == "bullish":
            # 1) SL first (conservative)
            if not trade.sl_hit and low <= sl:
                trade.sl_hit = True
                trade.is_closed = True
                trade.status = "closed - SL hit"
                rr = -1.0 if risk else float("nan")
                events.append((trade, "SL", sl, rr))
                continue  # no TP events once SL hit

            # 2) TP1 / TP2 / TP3
            if trade.tp1 is not None and (not trade.tp1_hit) and high >= trade.tp1:
                trade.tp1_hit = True
                rr = ((trade.tp1 - entry) / risk) if risk else float("nan")
                events.append((trade, "TP1", trade.tp1, rr))

            if trade.tp2 is not None and (not trade.tp2_hit) and high >= trade.tp2:
                trade.tp2_hit = True
                rr = ((trade.tp2 - entry) / risk) if risk else float("nan")
                events.append((trade, "TP2", trade.tp2, rr))

            if trade.tp3 is not None and (not trade.tp3_hit) and high >= trade.tp3:
                trade.tp3_hit = True
                trade.is_closed = True
                trade.status = "closed - TP3 hit"
                rr = ((trade.tp3 - entry) / risk) if risk else float("nan")
                events.append((trade, "TP3", trade.tp3, rr))

        else:  # bearish
            # 1) SL first
            if not trade.sl_hit and high >= sl:
                trade.sl_hit = True
                trade.is_closed = True
                trade.status = "closed - SL hit"
                rr = -1.0 if risk else float("nan")
                events.append((trade, "SL", sl, rr))
                continue

            # 2) TP1 / TP2 / TP3
            if trade.tp1 is not None and (not trade.tp1_hit) and low <= trade.tp1:
                trade.tp1_hit = True
                rr = ((entry - trade.tp1) / risk) if risk else float("nan")
                events.append((trade, "TP1", trade.tp1, rr))

            if trade.tp2 is not None and (not trade.tp2_hit) and low <= trade.tp2:
                trade.tp2_hit = True
                rr = ((entry - trade.tp2) / risk) if risk else float("nan")
                events.append((trade, "TP2", trade.tp2, rr))

            if trade.tp3 is not None and (not trade.tp3_hit) and low <= trade.tp3:
                trade.tp3_hit = True
                trade.is_closed = True
                trade.status = "closed - TP3 hit"
                rr = ((entry - trade.tp3) / risk) if risk else float("nan")
                events.append((trade, "TP3", trade.tp3, rr))

    return events
