"""
Enhanced Backtest Engine for Blueprint Trader AI.

Features:
- Walk-forward simulation with no look-ahead bias
- Proper trade execution simulation using candle H/L
- Partial profit taking support
- Detailed trade logging
- Multiple exit scenarios
- Uses unified strategy_core for signal generation
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Tuple, Optional

from data import get_ohlcv
from config import SIGNAL_MODE
from strategy_core import (
    generate_signals,
    get_default_params,
    Signal,
    StrategyParams,
    _atr,
    _find_pivots,
    _find_structure_sl,
)
from forex_holidays import should_skip_trading
from price_validation import validate_entry_price


def _parse_partial_date(s: str, for_start: bool) -> Optional[date]:
    """Parse date strings like 'Jan 2024', '2024-01-01', 'Now'."""
    s = s.strip()
    if not s:
        return None

    lower = s.lower()
    if lower in ("now", "today"):
        return date.today()

    fmts = ["%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%Y/%m/%d"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass

    month_fmts = ["%b %Y", "%B %Y"]
    for fmt in month_fmts:
        try:
            dt = datetime.strptime(s, fmt).date()
            if for_start:
                return date(dt.year, dt.month, 1)
            else:
                if dt.month == 12:
                    return date(dt.year, 12, 31)
                else:
                    next_month = date(dt.year, dt.month + 1, 1)
                    return next_month - timedelta(days=1)
        except Exception:
            pass

    return None


def _parse_period(period_str: str) -> Tuple[Optional[date], Optional[date]]:
    """Parse 'Jan 2024 - Sep 2024' into (start_date, end_date)."""
    s = period_str.strip()
    if "-" in s:
        left, right = s.split("-", 1)
    else:
        left, right = s, "now"

    start = _parse_partial_date(left.strip(), for_start=True)
    end = _parse_partial_date(right.strip(), for_start=False)

    if start and end and start > end:
        start, end = end, start

    return start, end


def _candle_to_datetime(candle: Dict) -> Optional[datetime]:
    """Get datetime from a candle dict, normalized to UTC."""
    t = candle.get("time") or candle.get("timestamp") or candle.get("date")
    if t is None:
        return None

    dt = None
    if isinstance(t, datetime):
        dt = t
    elif isinstance(t, date):
        dt = datetime(t.year, t.month, t.day, tzinfo=timezone.utc)
    elif isinstance(t, (int, float)):
        try:
            dt = datetime.utcfromtimestamp(t).replace(tzinfo=timezone.utc)
        except Exception:
            return None
    elif isinstance(t, str):
        s = t.strip()
        try:
            s2 = s.replace("Z", "+00:00")
            if "." in s2:
                head, tail = s2.split(".", 1)
                decimals = "".join(ch for ch in tail if ch.isdigit())[:6]
                rest = tail[len(decimals):]
                s2 = f"{head}.{decimals}{rest}"
            dt = datetime.fromisoformat(s2)
        except Exception:
            pass

        if dt is None:
            fmts = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]
            for fmt in fmts:
                try:
                    dt = datetime.strptime(s[:len(fmt)], fmt)
                    break
                except Exception:
                    continue
    
    if dt is not None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
    
    return dt


def _candle_to_date(candle: Dict) -> Optional[date]:
    dt = _candle_to_datetime(candle)
    return dt.date() if dt else None


def _build_date_list(candles: List[Dict]) -> List[Optional[date]]:
    return [_candle_to_date(c) for c in candles]


def _maybe_exit_trade(
    trade: Dict,
    high: float,
    low: float,
    exit_date: date,
) -> Optional[Dict]:
    """
    Check if trade hits TP or SL on a candle.
    Conservative approach: if SL and any TP are both hit on same bar, assume SL hit first.
    Trailing stop moves to breakeven after TP1 hit.
    """
    direction = trade["direction"]
    entry = trade["entry"]
    sl = trade.get("trailing_sl", trade["sl"])
    tp1 = trade["tp1"]
    tp2 = trade["tp2"]
    tp3 = trade["tp3"]
    risk = trade["risk"]
    tp1_hit = trade.get("tp1_hit", False)

    if direction == "bullish":
        hit_tp3 = tp3 is not None and high >= tp3
        hit_tp2 = tp2 is not None and high >= tp2
        hit_tp1 = tp1 is not None and high >= tp1
        hit_sl = low <= sl

        if hit_sl:
            if tp1_hit:
                rr = (sl - entry) / risk
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": max(rr, 0.0),
                    "exit_reason": "TP1+Trail",
                }
            else:
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": -1.0,
                    "exit_reason": "SL",
                }
        
        if tp1_hit:
            if hit_tp3:
                rr = (tp3 - entry) / risk
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": rr,
                    "exit_reason": "TP3",
                }
            elif hit_tp2:
                rr = (tp2 - entry) / risk
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": rr,
                    "exit_reason": "TP2",
                }
        elif hit_tp1 and not tp1_hit:
            trade["tp1_hit"] = True
            new_sl = entry
            trade["trailing_sl"] = new_sl
            if low <= new_sl:
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": 0.0,
                    "exit_reason": "TP1+Trail",
                }
            return None

    else:
        hit_tp3 = tp3 is not None and low <= tp3
        hit_tp2 = tp2 is not None and low <= tp2
        hit_tp1 = tp1 is not None and low <= tp1
        hit_sl = high >= sl

        if hit_sl:
            if tp1_hit:
                rr = (entry - sl) / risk
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": max(rr, 0.0),
                    "exit_reason": "TP1+Trail",
                }
            else:
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": -1.0,
                    "exit_reason": "SL",
                }
        
        if tp1_hit:
            if hit_tp3:
                rr = (entry - tp3) / risk
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": rr,
                    "exit_reason": "TP3",
                }
            elif hit_tp2:
                rr = (entry - tp2) / risk
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": rr,
                    "exit_reason": "TP2",
                }
        elif hit_tp1 and not tp1_hit:
            trade["tp1_hit"] = True
            new_sl = entry
            trade["trailing_sl"] = new_sl
            if high >= new_sl:
                return {
                    "entry_date": trade["entry_date"].isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "direction": direction,
                    "rr": 0.0,
                    "exit_reason": "TP1+Trail",
                }
            return None

    return None


def _signal_to_bar_date(signal: Signal, candles: List[Dict]) -> Optional[date]:
    """Get the date for a signal based on its bar_index."""
    if signal.bar_index < len(candles):
        return _candle_to_date(candles[signal.bar_index])
    return None


def _compute_trade_levels_from_candle(
    candles: List[Dict],
    bar_index: int,
    direction: str,
    params: StrategyParams,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute entry, SL, and TP levels from candle data.
    
    This is used as a fallback when the signal doesn't have valid trade levels,
    or to ensure we always have proper execution prices based on actual candle data.
    
    Args:
        candles: Full list of candles
        bar_index: Index of the current bar
        direction: Trade direction ('bullish' or 'bearish')
        params: Strategy parameters
    
    Returns:
        Tuple of (entry, sl, tp1, tp2, tp3)
    """
    if bar_index < 20 or bar_index >= len(candles):
        return None, None, None, None, None
    
    candle_slice = candles[:bar_index + 1]
    current = candles[bar_index]
    
    atr = _atr(candle_slice, 14)
    if atr <= 0:
        return None, None, None, None, None
    
    entry = current["close"]
    
    structure_sl = _find_structure_sl(candle_slice, direction, lookback=params.structure_sl_lookback)
    
    if direction == "bullish":
        if structure_sl is not None:
            sl = min(entry - atr * params.atr_sl_multiplier, structure_sl - atr * 0.4)
        else:
            sl = entry - atr * params.atr_sl_multiplier
        
        risk = entry - sl
        if risk <= 0:
            return None, None, None, None, None
        
        tp1 = entry + risk * params.atr_tp1_multiplier
        tp2 = entry + risk * params.atr_tp2_multiplier
        tp3 = entry + risk * params.atr_tp3_multiplier
    else:
        if structure_sl is not None:
            sl = max(entry + atr * params.atr_sl_multiplier, structure_sl + atr * 0.4)
        else:
            sl = entry + atr * params.atr_sl_multiplier
        
        risk = sl - entry
        if risk <= 0:
            return None, None, None, None, None
        
        tp1 = entry - risk * params.atr_tp1_multiplier
        tp2 = entry - risk * params.atr_tp2_multiplier
        tp3 = entry - risk * params.atr_tp3_multiplier
    
    return entry, sl, tp1, tp2, tp3


def run_backtest(asset: str, period: str) -> Dict:
    """
    Walk-forward backtest of the Blueprint strategy.
    
    Uses unified generate_signals() from strategy_core for signal generation,
    ensuring consistency with live scanning.
    
    Key improvements:
    - No look-ahead bias: uses only data available at each point
    - Proper trade execution simulation
    - Holiday filtering using forex_holidays
    - Price validation for entries
    - Detailed trade logging
    - Conservative exit assumptions
    """
    daily = get_ohlcv(asset, timeframe="D", count=2000, use_cache=False)
    if not daily:
        return {
            "asset": asset,
            "period": period,
            "total_trades": 0,
            "win_rate": 0.0,
            "net_return_pct": 0.0,
            "trades": [],
            "notes": "No Daily data available.",
        }

    weekly = get_ohlcv(asset, timeframe="W", count=500, use_cache=False) or []
    monthly = get_ohlcv(asset, timeframe="M", count=240, use_cache=False) or []
    h4 = get_ohlcv(asset, timeframe="H4", count=2000, use_cache=False) or []

    daily_dates = _build_date_list(daily)

    start_req, end_req = _parse_period(period)

    period_indices: List[int] = []

    if start_req or end_req:
        last_d = next((d for d in reversed(daily_dates) if d is not None), None)
        first_d = next((d for d in daily_dates if d is not None), None)

        end_date = end_req or last_d
        start_date = start_req or first_d

        if start_date is None or end_date is None:
            start_idx = max(0, len(daily) - 260)
            period_indices = list(range(start_idx, len(daily)))
            period_label = "Last 260 Daily candles"
        else:
            for i, d in enumerate(daily_dates):
                if d is None:
                    continue
                if start_date <= d <= end_date:
                    period_indices.append(i)

            if not period_indices:
                start_idx = max(0, len(daily) - 260)
                period_indices = list(range(start_idx, len(daily)))
                period_label = "Last 260 Daily candles"
            else:
                sd = daily_dates[period_indices[0]]
                ed = daily_dates[period_indices[-1]]
                period_label = f"{sd.isoformat()} - {ed.isoformat()}" if sd and ed else period
    else:
        start_idx = max(0, len(daily) - 260)
        period_indices = list(range(start_idx, len(daily)))
        period_label = "Last 260 Daily candles"

    if not period_indices:
        return {
            "asset": asset,
            "period": period,
            "total_trades": 0,
            "win_rate": 0.0,
            "net_return_pct": 0.0,
            "trades": [],
            "notes": "No candles found in requested period.",
        }

    params = get_default_params(asset)
    
    signals = generate_signals(
        candles=daily,
        symbol=asset,
        params=params,
        monthly_candles=monthly if monthly else None,
        weekly_candles=weekly if weekly else None,
        h4_candles=h4 if h4 else None,
    )

    signal_by_bar = {s.bar_index: s for s in signals if s.is_active or s.is_watching}

    trades: List[Dict] = []
    open_trade: Optional[Dict] = None
    
    cooldown_bars = params.cooldown_bars
    last_trade_idx = -cooldown_bars - 1
    holidays_skipped = 0
    price_validation_failed = 0

    for idx in period_indices:
        c = daily[idx]
        d_i = daily_dates[idx]
        if d_i is None:
            continue

        high = c["high"]
        low = c["low"]

        if open_trade is not None and idx > open_trade["entry_index"]:
            closed = _maybe_exit_trade(open_trade, high, low, d_i)
            if closed is not None:
                trades.append(closed)
                open_trade = None
                last_trade_idx = idx
                continue

        if open_trade is not None:
            continue

        if idx - last_trade_idx < cooldown_bars:
            continue

        if should_skip_trading(d_i):
            holidays_skipped += 1
            continue

        signal = signal_by_bar.get(idx)
        if signal is None:
            continue
        
        direction = signal.direction
        
        entry = signal.entry
        sl = signal.stop_loss
        tp1 = signal.tp1
        tp2 = signal.tp2
        tp3 = signal.tp3
        
        if entry is None or sl is None or tp1 is None:
            fallback_entry, fallback_sl, fallback_tp1, fallback_tp2, fallback_tp3 = \
                _compute_trade_levels_from_candle(daily, idx, direction, params)
            
            if fallback_entry is None or fallback_sl is None or fallback_tp1 is None:
                continue
            
            entry = entry if entry is not None else fallback_entry
            sl = sl if sl is not None else fallback_sl
            tp1 = tp1 if tp1 is not None else fallback_tp1
            tp2 = tp2 if tp2 is not None else fallback_tp2
            tp3 = tp3 if tp3 is not None else fallback_tp3

        is_valid_entry, entry_msg = validate_entry_price(
            entry_price=entry,
            candle_high=high,
            candle_low=low,
            direction=direction,
        )
        
        if not is_valid_entry:
            price_validation_failed += 1
            entry = c["close"]

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        open_trade = {
            "asset": asset,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "tp4": None,
            "tp5": None,
            "risk": risk,
            "entry_date": d_i,
            "entry_index": idx,
            "confluence": signal.confluence_score,
        }

    from config import ACCOUNT_SIZE, RISK_PER_TRADE_PCT
    
    total_trades = len(trades)
    if total_trades > 0:
        wins = sum(1 for t in trades if t["rr"] > 0)
        win_rate = wins / total_trades * 100.0
        total_rr = sum(t["rr"] for t in trades)
        net_return_pct = total_rr * RISK_PER_TRADE_PCT * 100
        avg_rr = total_rr / total_trades
    else:
        win_rate = 0.0
        net_return_pct = 0.0
        total_rr = 0.0
        avg_rr = 0.0

    risk_per_trade_usd = ACCOUNT_SIZE * RISK_PER_TRADE_PCT
    total_profit_usd = total_rr * risk_per_trade_usd
    
    running_pnl = 0.0
    max_drawdown = 0.0
    peak = 0.0
    
    for t in trades:
        running_pnl += t["rr"] * risk_per_trade_usd
        if running_pnl > peak:
            peak = running_pnl
        drawdown = peak - running_pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    max_drawdown_pct = (max_drawdown / ACCOUNT_SIZE) * 100 if ACCOUNT_SIZE > 0 else 0.0

    tp1_trail_hits = sum(1 for t in trades if t.get("exit_reason") == "TP1+Trail")
    tp2_hits = sum(1 for t in trades if t.get("exit_reason") == "TP2")
    tp3_hits = sum(1 for t in trades if t.get("exit_reason") == "TP3")
    sl_hits = sum(1 for t in trades if t.get("exit_reason") == "SL")
    
    wins = tp1_trail_hits + tp2_hits + tp3_hits

    notes_text = (
        f"Backtest Summary - {asset} ({period_label}, 100K 5%ers model)\n"
        f"Trades: {total_trades}\n"
        f"Win rate: {win_rate:.1f}%\n"
        f"Total profit: +${total_profit_usd:,.0f} (+{net_return_pct:.1f}%)\n"
        f"Max drawdown: -{max_drawdown_pct:.1f}%\n"
        f"Expectancy: {avg_rr:+.2f}R / trade\n"
        f"TP1+Trail ({tp1_trail_hits}), TP2 ({tp2_hits}), TP3 ({tp3_hits}), SL ({sl_hits})"
    )
    
    if holidays_skipped > 0:
        notes_text += f"\nHolidays skipped: {holidays_skipped}"
    if price_validation_failed > 0:
        notes_text += f"\nPrice validations adjusted: {price_validation_failed}"

    return {
        "asset": asset,
        "period": period_label,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "net_return_pct": net_return_pct,
        "total_profit_usd": total_profit_usd,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_rr": avg_rr,
        "tp1_trail_hits": tp1_trail_hits,
        "tp2_hits": tp2_hits,
        "tp3_hits": tp3_hits,
        "sl_hits": sl_hits,
        "trades": trades,
        "notes": notes_text,
        "account_size": ACCOUNT_SIZE,
        "risk_per_trade_pct": RISK_PER_TRADE_PCT,
        "holidays_skipped": holidays_skipped,
        "price_validation_adjusted": price_validation_failed,
    }
