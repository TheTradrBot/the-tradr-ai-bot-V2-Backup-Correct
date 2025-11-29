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

import csv
import os
from data import get_ohlcv
from config import SIGNAL_MODE
from strategy_core import (
    generate_signals,
    generate_signals_mtf,
    generate_htf_confluence_signals,
    get_default_params,
    Signal,
    StrategyParams,
    HTFConfluenceParams,
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


def _maybe_exit_trade_laddered(
    trade: Dict,
    high: float,
    low: float,
    exit_date: date,
) -> Optional[Dict]:
    """
    Check if trade hits TP or SL on a candle with laddered exit system.
    
    Exit strategy (position sizing):
    - TP1: Take 50% profit at 1.5R
    - TP2: Take 30% profit at 2.5R, move SL to TP1 level
    - TP3: Take 20% (runner) at 4R, or trail with SL at TP2
    
    Conservative: if SL and TP are both hit on same bar, assume SL hit first
    """
    direction = trade["direction"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp1 = trade["tp1"]
    tp2 = trade.get("tp2")
    tp3 = trade.get("tp3")
    risk = trade["risk"]
    
    tp1_hit = trade.get("tp1_hit", False)
    tp2_hit = trade.get("tp2_hit", False)
    
    if risk <= 0:
        return None

    if direction == "bullish":
        current_sl = sl
        if tp1_hit:
            current_sl = tp1
        if tp2_hit and tp2:
            current_sl = tp2
        
        hit_sl = low <= current_sl
        hit_tp1 = not tp1_hit and tp1 is not None and high >= tp1
        hit_tp2 = tp1_hit and not tp2_hit and tp2 is not None and high >= tp2
        hit_tp3 = tp2_hit and tp3 is not None and high >= tp3
        
        if hit_sl and (hit_tp1 or hit_tp2 or hit_tp3):
            pnl = 0.0
            if tp1_hit:
                pnl += 0.5 * ((tp1 - entry) / risk)
            if tp2_hit:
                pnl += 0.3 * ((tp2 - entry) / risk)
            remaining = 1.0 - (0.5 if tp1_hit else 0) - (0.3 if tp2_hit else 0)
            pnl += remaining * ((current_sl - entry) / risk)
            
            return {
                "entry_date": trade["entry_date"].isoformat(),
                "exit_date": exit_date.isoformat(),
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "exit_price": current_sl,
                "rr": pnl,
                "exit_reason": "SL_RUNNER" if tp2_hit else ("SL_PARTIAL" if tp1_hit else "SL"),
                "tp1_hit": tp1_hit,
                "tp2_hit": tp2_hit,
            }
        
        if hit_sl:
            pnl = 0.0
            remaining = 1.0
            if tp1_hit:
                pnl += 0.5 * ((tp1 - entry) / risk)
                remaining -= 0.5
            if tp2_hit:
                pnl += 0.3 * ((tp2 - entry) / risk)
                remaining -= 0.3
            pnl += remaining * (-1.0)
            
            return {
                "entry_date": trade["entry_date"].isoformat(),
                "exit_date": exit_date.isoformat(),
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "exit_price": current_sl,
                "rr": pnl,
                "exit_reason": "SL_RUNNER" if tp2_hit else ("SL_PARTIAL" if tp1_hit else "SL"),
                "tp1_hit": tp1_hit,
                "tp2_hit": tp2_hit,
            }
        
        if hit_tp3:
            pnl = 0.5 * ((tp1 - entry) / risk) + 0.3 * ((tp2 - entry) / risk) + 0.2 * ((tp3 - entry) / risk)
            return {
                "entry_date": trade["entry_date"].isoformat(),
                "exit_date": exit_date.isoformat(),
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "exit_price": tp3,
                "rr": pnl,
                "exit_reason": "TP3_FULL",
                "tp1_hit": True,
                "tp2_hit": True,
            }
        
        if hit_tp2:
            trade["tp2_hit"] = True
            return None
        
        if hit_tp1:
            trade["tp1_hit"] = True
            return None

    else:
        current_sl = sl
        if tp1_hit:
            current_sl = tp1
        if tp2_hit and tp2:
            current_sl = tp2
        
        hit_sl = high >= current_sl
        hit_tp1 = not tp1_hit and tp1 is not None and low <= tp1
        hit_tp2 = tp1_hit and not tp2_hit and tp2 is not None and low <= tp2
        hit_tp3 = tp2_hit and tp3 is not None and low <= tp3
        
        if hit_sl and (hit_tp1 or hit_tp2 or hit_tp3):
            pnl = 0.0
            if tp1_hit:
                pnl += 0.5 * ((entry - tp1) / risk)
            if tp2_hit:
                pnl += 0.3 * ((entry - tp2) / risk)
            remaining = 1.0 - (0.5 if tp1_hit else 0) - (0.3 if tp2_hit else 0)
            pnl += remaining * ((entry - current_sl) / risk)
            
            return {
                "entry_date": trade["entry_date"].isoformat(),
                "exit_date": exit_date.isoformat(),
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "exit_price": current_sl,
                "rr": pnl,
                "exit_reason": "SL_RUNNER" if tp2_hit else ("SL_PARTIAL" if tp1_hit else "SL"),
                "tp1_hit": tp1_hit,
                "tp2_hit": tp2_hit,
            }
        
        if hit_sl:
            pnl = 0.0
            remaining = 1.0
            if tp1_hit:
                pnl += 0.5 * ((entry - tp1) / risk)
                remaining -= 0.5
            if tp2_hit:
                pnl += 0.3 * ((entry - tp2) / risk)
                remaining -= 0.3
            pnl += remaining * (-1.0)
            
            return {
                "entry_date": trade["entry_date"].isoformat(),
                "exit_date": exit_date.isoformat(),
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "exit_price": current_sl,
                "rr": pnl,
                "exit_reason": "SL_RUNNER" if tp2_hit else ("SL_PARTIAL" if tp1_hit else "SL"),
                "tp1_hit": tp1_hit,
                "tp2_hit": tp2_hit,
            }
        
        if hit_tp3:
            pnl = 0.5 * ((entry - tp1) / risk) + 0.3 * ((entry - tp2) / risk) + 0.2 * ((entry - tp3) / risk)
            return {
                "entry_date": trade["entry_date"].isoformat(),
                "exit_date": exit_date.isoformat(),
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "exit_price": tp3,
                "rr": pnl,
                "exit_reason": "TP3_FULL",
                "tp1_hit": True,
                "tp2_hit": True,
            }
        
        if hit_tp2:
            trade["tp2_hit"] = True
            return None
        
        if hit_tp1:
            trade["tp1_hit"] = True
            return None

    return None


def _maybe_exit_trade(
    trade: Dict,
    high: float,
    low: float,
    exit_date: date,
) -> Optional[Dict]:
    """Simple exit - backward compatible wrapper."""
    return _maybe_exit_trade_laddered(trade, high, low, exit_date)


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
    htf_params = HTFConfluenceParams()
    
    use_htf_confluence = True
    
    if use_htf_confluence and h4 and weekly and monthly:
        signals = generate_htf_confluence_signals(
            daily_candles=daily,
            h4_candles=h4,
            weekly_candles=weekly,
            monthly_candles=monthly,
            symbol=asset,
            params=htf_params,
        )
    elif h4 and weekly and monthly:
        signals = generate_signals_mtf(
            daily_candles=daily,
            h4_candles=h4,
            weekly_candles=weekly,
            monthly_candles=monthly,
            symbol=asset,
            params=params,
        )
    else:
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
        
        if d_i.weekday() == 4:
            continue

        signal = signal_by_bar.get(idx)
        if signal is None:
            for lookback in range(1, 4):
                prev_idx = idx - lookback
                if prev_idx >= 0 and prev_idx in signal_by_bar:
                    prev_signal = signal_by_bar[prev_idx]
                    if prev_signal.is_active or prev_signal.is_watching:
                        signal = prev_signal
                        break
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
            if risk > 0:
                if direction == "bullish":
                    tp1 = entry + risk * params.atr_tp1_multiplier
                    tp2 = entry + risk * params.atr_tp2_multiplier
                    tp3 = entry + risk * params.atr_tp3_multiplier
                else:
                    tp1 = entry - risk * params.atr_tp1_multiplier
                    tp2 = entry - risk * params.atr_tp2_multiplier
                    tp3 = entry - risk * params.atr_tp3_multiplier

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

    tp1_hits = sum(1 for t in trades if t.get("exit_reason") in ("TP1", "TP1+Trail"))
    tp2_hits = sum(1 for t in trades if t.get("exit_reason") == "TP2")
    tp3_hits = sum(1 for t in trades if t.get("exit_reason") == "TP3")
    sl_hits = sum(1 for t in trades if t.get("exit_reason") == "SL")
    
    wins = tp1_hits + tp2_hits + tp3_hits

    notes_text = (
        f"Backtest Summary - {asset} ({period_label}, 100K 5%ers model)\n"
        f"Trades: {total_trades}\n"
        f"Win rate: {win_rate:.1f}%\n"
        f"Total profit: +${total_profit_usd:,.0f} (+{net_return_pct:.1f}%)\n"
        f"Max drawdown: -{max_drawdown_pct:.1f}%\n"
        f"Expectancy: {avg_rr:+.2f}R / trade\n"
        f"TP1 ({tp1_hits}), TP2 ({tp2_hits}), TP3 ({tp3_hits}), SL ({sl_hits})"
    )
    
    if holidays_skipped > 0:
        notes_text += f"\nHolidays skipped: {holidays_skipped}"
    if price_validation_failed > 0:
        notes_text += f"\nPrice validations adjusted: {price_validation_failed}"

    result = {
        "asset": asset,
        "period": period_label,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "net_return_pct": net_return_pct,
        "total_profit_usd": total_profit_usd,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_rr": avg_rr,
        "tp1_trail_hits": tp1_hits,
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
    
    return result


def export_trades_to_csv(result: Dict, output_dir: str = "backtest_results") -> str:
    """
    Export backtest trades to CSV file with exact trade details.
    
    Args:
        result: Backtest result dictionary from run_backtest()
        output_dir: Directory to save CSV files
        
    Returns:
        Path to the created CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    asset = result.get("asset", "UNKNOWN")
    period = result.get("period", "").replace(" ", "_").replace("-", "_")
    filename = f"{asset}_{period}_trades.csv"
    filepath = os.path.join(output_dir, filename)
    
    trades = result.get("trades", [])
    
    if not trades:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["No trades to export"])
        return filepath
    
    fieldnames = [
        "entry_date", "exit_date", "direction", "entry", "sl", "tp1", "tp2", "tp3",
        "exit_price", "rr", "exit_reason", "tp1_hit", "tp2_hit"
    ]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for trade in trades:
            row = {k: trade.get(k, "") for k in fieldnames}
            for price_field in ["entry", "sl", "tp1", "tp2", "tp3", "exit_price"]:
                if row[price_field] and isinstance(row[price_field], float):
                    row[price_field] = f"{row[price_field]:.5f}"
            if row["rr"] and isinstance(row["rr"], float):
                row["rr"] = f"{row['rr']:.2f}"
            writer.writerow(row)
    
    with open(filepath, 'a') as f:
        f.write(f"\n# Summary:\n")
        f.write(f"# Asset: {asset}\n")
        f.write(f"# Period: {result.get('period', '')}\n")
        f.write(f"# Total Trades: {result.get('total_trades', 0)}\n")
        f.write(f"# Win Rate: {result.get('win_rate', 0):.1f}%\n")
        f.write(f"# Net Return: {result.get('net_return_pct', 0):.1f}%\n")
        f.write(f"# Avg R:R: {result.get('avg_rr', 0):.2f}\n")
    
    return filepath


def run_yearly_backtest(asset: str, year: int) -> Dict:
    """
    Run backtest for a specific asset and year.
    
    Args:
        asset: Trading asset symbol
        year: Year to backtest
        
    Returns:
        Backtest result dictionary
    """
    period = f"Jan {year} - Dec {year}"
    return run_backtest(asset, period)


def run_all_yearly_backtests(assets: List[str], years: List[int], export_csv: bool = True) -> Dict:
    """
    Run backtests for all assets across all years and optionally export to CSV.
    
    Args:
        assets: List of trading asset symbols
        years: List of years to backtest
        export_csv: Whether to export trades to CSV
        
    Returns:
        Dictionary with all results organized by year and asset
    """
    all_results = {}
    
    for year in years:
        all_results[year] = {}
        
        for asset in assets:
            print(f"Running backtest: {asset} {year}...")
            result = run_yearly_backtest(asset, year)
            all_results[year][asset] = result
            
            if export_csv and result.get("total_trades", 0) > 0:
                csv_path = export_trades_to_csv(result)
                result["csv_path"] = csv_path
                print(f"  -> Exported to {csv_path}")
            
            trades = result.get("total_trades", 0)
            wr = result.get("win_rate", 0)
            ret = result.get("net_return_pct", 0)
            
            status = "PASS" if trades >= 50 and wr >= 70 and ret >= 50 else "FAIL"
            print(f"  -> {trades} trades, {wr:.1f}% WR, {ret:.1f}% return [{status}]")
    
    return all_results
