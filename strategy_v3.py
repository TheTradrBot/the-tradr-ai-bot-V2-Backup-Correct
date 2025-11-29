"""
Strategy V3: HTF Support/Resistance + Break of Structure

Based on user specifications:
- HTF S/R zones from Weekly/Daily
- Break of Structure (BOS) confirmation on H4
- Structural Take Profits (swing highs/lows) - NO Fibonacci TPs
- NO RSI, NO SMC
- Minimum 4:1 R:R for high expectancy
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd


@dataclass
class TradeSignal:
    """Represents a trade signal from the V3 strategy."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry: float
    stop_loss: float
    take_profit: float
    r_multiple: float
    status: str  # 'active', 'watching', 'scan'
    htf_zone: Tuple[float, float]  # (zone_low, zone_high)
    bos_level: float
    confluence_score: int
    reasoning: str
    timestamp: datetime


def calculate_atr(candles: List[Dict], period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(candles) < period + 1:
        return 0.0
    
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i-1]['close']
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)
    
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0.0
    
    return sum(trs[-period:]) / period


def find_swing_highs(candles: List[Dict], lookback: int = 5) -> List[float]:
    """Find swing high levels (local maxima)."""
    swing_highs = []
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i]['high']
        is_swing = True
        
        for j in range(i - lookback, i + lookback + 1):
            if j != i and candles[j]['high'] >= high:
                is_swing = False
                break
        
        if is_swing:
            swing_highs.append(high)
    
    return sorted(set(swing_highs), reverse=True)


def find_swing_lows(candles: List[Dict], lookback: int = 5) -> List[float]:
    """Find swing low levels (local minima)."""
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        low = candles[i]['low']
        is_swing = True
        
        for j in range(i - lookback, i + lookback + 1):
            if j != i and candles[j]['low'] <= low:
                is_swing = False
                break
        
        if is_swing:
            swing_lows.append(low)
    
    return sorted(set(swing_lows))


def identify_sr_zones(candles: List[Dict], tolerance_pct: float = 0.005, min_touches: int = 1) -> List[Tuple[float, float]]:
    """
    Identify Support/Resistance zones from price action.
    Groups nearby swing levels into zones.
    
    Args:
        candles: Price candles
        tolerance_pct: % tolerance for grouping levels (default 0.5%)
        min_touches: Minimum touches to form a zone (default 1)
    """
    swing_highs = find_swing_highs(candles, lookback=2)
    swing_lows = find_swing_lows(candles, lookback=2)
    
    all_levels = sorted(swing_highs + swing_lows)
    
    if not all_levels:
        return []
    
    zones = []
    current_zone = [all_levels[0]]
    
    for level in all_levels[1:]:
        zone_avg = sum(current_zone) / len(current_zone)
        if abs(level - zone_avg) / zone_avg <= tolerance_pct:
            current_zone.append(level)
        else:
            zone_low = min(current_zone)
            zone_high = max(current_zone)
            zones.append((zone_low, zone_high))
            current_zone = [level]
    
    if current_zone:
        zone_low = min(current_zone)
        zone_high = max(current_zone)
        zones.append((zone_low, zone_high))
    
    zones = [(z[0], z[1]) for z in zones if len([l for l in all_levels if z[0] <= l <= z[1]]) >= min_touches]
    
    return zones


def detect_bos(candles: List[Dict], direction: str, lookback: int = 20) -> Optional[float]:
    """
    Detect Break of Structure (BOS) - relaxed detection.
    
    For LONG: Price breaks above recent swing high (wick or close)
    For SHORT: Price breaks below recent swing low (wick or close)
    """
    if len(candles) < lookback + 2:
        return None
    
    recent_candles = candles[-lookback:]
    current_bar = candles[-1]
    prev_bar = candles[-2] if len(candles) > 1 else current_bar
    current_close = current_bar['close']
    current_high = current_bar['high']
    current_low = current_bar['low']
    
    if direction == 'long':
        swing_highs = find_swing_highs(recent_candles, lookback=2)
        if swing_highs:
            sorted_swings = sorted(swing_highs, reverse=True)
            for swing in sorted_swings[:5]:
                if current_close > swing:
                    return swing
                elif current_high > swing:
                    return swing
                elif prev_bar['close'] > swing:
                    return swing
    else:
        swing_lows = find_swing_lows(recent_candles, lookback=2)
        if swing_lows:
            sorted_swings = sorted(swing_lows)
            for swing in sorted_swings[:5]:
                if current_close < swing:
                    return swing
                elif current_low < swing:
                    return swing
                elif prev_bar['close'] < swing:
                    return swing
    
    return None


def detect_soft_bos(candles: List[Dict], direction: str, lookback: int = 20) -> Optional[float]:
    """
    Detect SOFT Break of Structure - wick touch only.
    Used for 'watching' status signals.
    """
    if len(candles) < lookback + 5:
        return None
    
    recent_candles = candles[-lookback:]
    current_bar = candles[-1]
    
    if direction == 'long':
        swing_highs = find_swing_highs(recent_candles, lookback=3)
        if swing_highs:
            for swing in sorted(swing_highs, reverse=True)[:3]:
                if current_bar['high'] >= swing * 0.9995:
                    return swing
    else:
        swing_lows = find_swing_lows(recent_candles, lookback=3)
        if swing_lows:
            for swing in sorted(swing_lows)[:3]:
                if current_bar['low'] <= swing * 1.0005:
                    return swing
    
    return None


def get_htf_bias(weekly_candles: List[Dict], daily_candles: List[Dict]) -> str:
    """
    Determine Higher Timeframe bias based on Weekly and Daily structure.
    """
    if len(weekly_candles) < 5 or len(daily_candles) < 20:
        return 'neutral'
    
    weekly_closes = [c['close'] for c in weekly_candles[-5:]]
    daily_closes = [c['close'] for c in daily_candles[-20:]]
    
    weekly_higher_highs = weekly_candles[-1]['high'] > weekly_candles[-2]['high']
    weekly_higher_lows = weekly_candles[-1]['low'] > weekly_candles[-2]['low']
    
    daily_sma = sum(daily_closes) / len(daily_closes)
    current_price = daily_candles[-1]['close']
    
    if weekly_higher_highs and weekly_higher_lows and current_price > daily_sma:
        return 'bullish'
    elif not weekly_higher_highs and not weekly_higher_lows and current_price < daily_sma:
        return 'bearish'
    
    return 'neutral'


def find_structural_tp(
    candles: List[Dict],
    entry: float,
    stop_loss: float,
    direction: str,
    min_rr: float = 4.0
) -> Optional[float]:
    """
    Find structural take profit level (swing high/low).
    NO Fibonacci - uses pure price structure.
    Returns TP that gives at least min_rr R:R ratio.
    """
    risk = abs(entry - stop_loss)
    min_reward = risk * min_rr
    
    if direction == 'long':
        swing_highs = find_swing_highs(candles, lookback=5)
        valid_tps = [h for h in swing_highs if h > entry + min_reward]
        if valid_tps:
            return min(valid_tps)
        else:
            return entry + min_reward
    else:
        swing_lows = find_swing_lows(candles, lookback=5)
        valid_tps = [l for l in swing_lows if l < entry - min_reward]
        if valid_tps:
            return max(valid_tps)
        else:
            return entry - min_reward


def is_price_at_zone(price: float, zone: Tuple[float, float], atr: float) -> bool:
    """Check if price is at or near an S/R zone."""
    zone_low, zone_high = zone
    buffer = atr * 0.5
    return (zone_low - buffer) <= price <= (zone_high + buffer)


def generate_signal(
    symbol: str,
    h4_candles: List[Dict],
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_rr: float = 3.0,
    min_confluence: int = 2
) -> Optional[TradeSignal]:
    """
    Generate a V3 trading signal with improved confluence scoring.
    
    Entry conditions:
    1. Price at or near HTF S/R zone (+1)
    2. HTF bias aligned (+1) OR price structure direction (+1)
    3. BOS confirmation on H4 (+2 confirmed, +1 soft)
    4. H4 momentum aligned (+1)
    5. Zone reaction pattern (+1)
    6. Structural TP at swing level
    7. Minimum R:R of 3:1
    
    Min confluence: 2 for watching, 3 for active
    """
    if len(h4_candles) < 50 or len(daily_candles) < 50 or len(weekly_candles) < 10:
        return None
    
    current_bar = h4_candles[-1]
    current_price = current_bar['close']
    current_time = current_bar['time']
    
    htf_bias = get_htf_bias(weekly_candles, daily_candles)
    
    daily_zones = identify_sr_zones(daily_candles, tolerance_pct=0.004)
    weekly_zones = identify_sr_zones(weekly_candles, tolerance_pct=0.006)
    
    all_zones = daily_zones + weekly_zones
    
    if not all_zones:
        return None
    
    daily_atr = calculate_atr(daily_candles, period=14)
    h4_atr = calculate_atr(h4_candles, period=14)
    
    if daily_atr == 0 or h4_atr == 0:
        return None
    
    active_zone = None
    zone_buffer = daily_atr * 0.75
    for zone in all_zones:
        zone_low, zone_high = zone
        if (zone_low - zone_buffer) <= current_price <= (zone_high + zone_buffer):
            active_zone = zone
            break
    
    if not active_zone:
        return None
    
    zone_low, zone_high = active_zone
    zone_mid = (zone_low + zone_high) / 2
    
    recent_lows = [c['low'] for c in h4_candles[-10:]]
    recent_highs = [c['high'] for c in h4_candles[-10:]]
    
    if current_price > zone_mid:
        if htf_bias == 'bullish' or min(recent_lows) >= zone_low - h4_atr * 0.3:
            direction = 'long'
        else:
            direction = 'long'
    else:
        if htf_bias == 'bearish' or max(recent_highs) <= zone_high + h4_atr * 0.3:
            direction = 'short'
        else:
            direction = 'short'
    
    if direction is None:
        return None
    
    bos_level = detect_bos(h4_candles, direction, lookback=20)
    soft_bos = detect_soft_bos(h4_candles, direction, lookback=20) if not bos_level else None
    
    confluence = 0
    reasoning_parts = []
    
    confluence += 1
    reasoning_parts.append(f"At HTF zone {zone_low:.5f}-{zone_high:.5f}")
    
    if htf_bias != 'neutral':
        if (direction == 'long' and htf_bias == 'bullish') or (direction == 'short' and htf_bias == 'bearish'):
            confluence += 1
            reasoning_parts.append(f"HTF bias aligned: {htf_bias}")
    
    if bos_level:
        confluence += 2
        reasoning_parts.append(f"BOS confirmed at {bos_level:.5f}")
    elif soft_bos:
        confluence += 1
        reasoning_parts.append(f"Soft BOS at {soft_bos:.5f}")
    
    h4_trend = 'up' if h4_candles[-1]['close'] > h4_candles[-5]['close'] else 'down'
    if (direction == 'long' and h4_trend == 'up') or (direction == 'short' and h4_trend == 'down'):
        confluence += 1
        reasoning_parts.append("H4 momentum aligned")
    
    if direction == 'long':
        zone_reaction = any(c['low'] <= zone_high and c['close'] > c['open'] for c in h4_candles[-5:])
    else:
        zone_reaction = any(c['high'] >= zone_low and c['close'] < c['open'] for c in h4_candles[-5:])
    
    if zone_reaction:
        confluence += 1
        reasoning_parts.append("Zone reaction pattern")
    
    if confluence < min_confluence:
        return None
    
    entry = current_price
    
    if direction == 'long':
        stop_loss = zone_low - (h4_atr * 0.5)
    else:
        stop_loss = zone_high + (h4_atr * 0.5)
    
    take_profit = find_structural_tp(
        daily_candles,
        entry,
        stop_loss,
        direction,
        min_rr=min_rr
    )
    
    if take_profit is None:
        return None
    
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    
    if risk == 0:
        return None
    
    r_multiple = reward / risk
    
    if r_multiple < min_rr:
        return None
    
    if bos_level:
        status = 'active'
    elif confluence >= min_confluence:
        status = 'watching'
    else:
        status = 'scan'
    
    reasoning = " | ".join(reasoning_parts)
    
    return TradeSignal(
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        r_multiple=round(r_multiple, 2),
        status=status,
        htf_zone=active_zone,
        bos_level=bos_level if bos_level else 0.0,
        confluence_score=confluence,
        reasoning=reasoning,
        timestamp=current_time
    )


def backtest_v3(
    symbol: str,
    h4_candles: List[Dict],
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    start_date: datetime,
    end_date: datetime,
    risk_per_trade: float = 250.0,
    min_rr: float = 2.0,
    min_confluence: int = 2,
    max_trades_per_day: int = 8,
    cooldown_bars: int = 2
) -> List[Dict]:
    """
    Backtest the V3 strategy over a date range.
    
    Handles both 'active' and 'watching' signals:
    - Active signals: Enter immediately
    - Watching signals: Queue for entry on next bar with BOS confirmation
    
    Returns list of completed trades with P/L.
    """
    trades = []
    
    h4_df = pd.DataFrame(h4_candles)
    h4_df['time'] = pd.to_datetime(h4_df['time'])
    
    start_idx = h4_df[h4_df['time'] >= start_date].index.min()
    end_idx = h4_df[h4_df['time'] <= end_date].index.max()
    
    if pd.isna(start_idx) or pd.isna(end_idx):
        return []
    
    lookback_start = max(0, start_idx - 100)
    
    h4_df = h4_df.iloc[lookback_start:end_idx + 1].reset_index(drop=True)
    start_bar = min(100, start_idx - lookback_start)
    
    if len(h4_df) < start_bar + 10:
        return []
    
    daily_df = pd.DataFrame(daily_candles)
    daily_df['time'] = pd.to_datetime(daily_df['time'])
    
    weekly_df = pd.DataFrame(weekly_candles)
    weekly_df['time'] = pd.to_datetime(weekly_df['time'])
    
    in_trade = False
    trade_entry = None
    last_signal_bar = -cooldown_bars
    daily_trades = {}
    pending_signal = None
    pending_bars = 0
    max_pending_bars = 3
    
    for i in range(start_bar, len(h4_df)):
        current_bar = h4_df.iloc[i]
        current_time = current_bar['time']
        current_date = current_time.date()
        
        if current_date not in daily_trades:
            daily_trades[current_date] = 0
        
        if in_trade:
            bar_high = current_bar['high']
            bar_low = current_bar['low']
            entry_price = trade_entry['entry_price']
            stop_loss = trade_entry['stop_loss']
            take_profit = trade_entry['take_profit']
            risk = abs(entry_price - trade_entry.get('original_sl', stop_loss))
            
            if trade_entry['direction'] == 'long':
                current_r = (bar_high - entry_price) / risk if risk > 0 else 0
                if current_r > trade_entry.get('highest_r', 0):
                    trade_entry['highest_r'] = current_r
                
                if current_r >= 1.0 and not trade_entry.get('be_triggered'):
                    trade_entry['original_sl'] = stop_loss
                    trade_entry['stop_loss'] = entry_price
                    trade_entry['be_triggered'] = True
                
                if bar_low <= stop_loss:
                    if trade_entry.get('be_triggered'):
                        trade_entry['exit_price'] = stop_loss
                        trade_entry['exit_type'] = 'BE'
                        trade_entry['exit_time'] = current_time
                        trade_entry['pnl'] = 0
                        trade_entry['r_result'] = 0.0
                    else:
                        trade_entry['exit_price'] = stop_loss
                        trade_entry['exit_type'] = 'SL'
                        trade_entry['exit_time'] = current_time
                        trade_entry['pnl'] = -risk_per_trade
                        trade_entry['r_result'] = -1.0
                    trades.append(trade_entry)
                    in_trade = False
                    trade_entry = None
                elif bar_high >= take_profit:
                    trade_entry['exit_price'] = take_profit
                    trade_entry['exit_type'] = 'TP'
                    trade_entry['exit_time'] = current_time
                    trade_entry['pnl'] = risk_per_trade * trade_entry['r_multiple']
                    trade_entry['r_result'] = trade_entry['r_multiple']
                    trades.append(trade_entry)
                    in_trade = False
                    trade_entry = None
            else:
                current_r = (entry_price - bar_low) / risk if risk > 0 else 0
                if current_r > trade_entry.get('highest_r', 0):
                    trade_entry['highest_r'] = current_r
                
                if current_r >= 1.0 and not trade_entry.get('be_triggered'):
                    trade_entry['original_sl'] = stop_loss
                    trade_entry['stop_loss'] = entry_price
                    trade_entry['be_triggered'] = True
                
                if bar_high >= stop_loss:
                    if trade_entry.get('be_triggered'):
                        trade_entry['exit_price'] = stop_loss
                        trade_entry['exit_type'] = 'BE'
                        trade_entry['exit_time'] = current_time
                        trade_entry['pnl'] = 0
                        trade_entry['r_result'] = 0.0
                    else:
                        trade_entry['exit_price'] = stop_loss
                        trade_entry['exit_type'] = 'SL'
                        trade_entry['exit_time'] = current_time
                        trade_entry['pnl'] = -risk_per_trade
                        trade_entry['r_result'] = -1.0
                    trades.append(trade_entry)
                    in_trade = False
                    trade_entry = None
                elif bar_low <= take_profit:
                    trade_entry['exit_price'] = take_profit
                    trade_entry['exit_type'] = 'TP'
                    trade_entry['exit_time'] = current_time
                    trade_entry['pnl'] = risk_per_trade * trade_entry['r_multiple']
                    trade_entry['r_result'] = trade_entry['r_multiple']
                    trades.append(trade_entry)
                    in_trade = False
                    trade_entry = None
            continue
        
        if pending_signal and not in_trade:
            pending_bars += 1
            if pending_bars > max_pending_bars:
                pending_signal = None
                pending_bars = 0
            else:
                h4_history = h4_df.iloc[max(0, i-100):i+1].to_dict('records')
                bos_now = detect_bos(h4_history, pending_signal.direction, lookback=20)
                if bos_now and daily_trades[current_date] < max_trades_per_day:
                    trade_entry = {
                        'symbol': symbol,
                        'direction': pending_signal.direction,
                        'entry_price': current_bar['close'],
                        'stop_loss': pending_signal.stop_loss,
                        'take_profit': pending_signal.take_profit,
                        'r_multiple': pending_signal.r_multiple,
                        'entry_time': current_time,
                        'confluence': pending_signal.confluence_score,
                        'reasoning': pending_signal.reasoning + " | Pending->Active",
                        'exit_price': None,
                        'exit_type': None,
                        'exit_time': None,
                        'pnl': None,
                        'r_result': None
                    }
                    in_trade = True
                    last_signal_bar = i
                    daily_trades[current_date] += 1
                    pending_signal = None
                    pending_bars = 0
                    continue
        
        if i - last_signal_bar < cooldown_bars:
            continue
        
        if daily_trades[current_date] >= max_trades_per_day:
            continue
        
        h4_history = h4_df.iloc[max(0, i-100):i+1].to_dict('records')
        daily_history = daily_df[daily_df['time'] <= current_time].tail(100).to_dict('records')
        weekly_history = weekly_df[weekly_df['time'] <= current_time].tail(20).to_dict('records')
        
        signal = generate_signal(
            symbol=symbol,
            h4_candles=h4_history,
            daily_candles=daily_history,
            weekly_candles=weekly_history,
            min_rr=min_rr,
            min_confluence=2
        )
        
        if signal:
            if signal.status == 'active' and signal.confluence_score >= min_confluence:
                trade_entry = {
                    'symbol': symbol,
                    'direction': signal.direction,
                    'entry_price': signal.entry,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'r_multiple': signal.r_multiple,
                    'entry_time': current_time,
                    'confluence': signal.confluence_score,
                    'reasoning': signal.reasoning,
                    'exit_price': None,
                    'exit_type': None,
                    'exit_time': None,
                    'pnl': None,
                    'r_result': None,
                    'be_triggered': False,
                    'highest_r': 0.0
                }
                in_trade = True
                last_signal_bar = i
                daily_trades[current_date] += 1
            elif signal.status == 'watching' and not pending_signal:
                pending_signal = signal
                pending_bars = 0
    
    return trades


def calculate_backtest_stats(trades: List[Dict]) -> Dict[str, Any]:
    """Calculate statistics from backtest trades."""
    if not trades:
        return {
            'total_trades': 0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_winner': 0.0,
            'avg_loser': 0.0,
            'profit_factor': 0.0,
            'avg_r': 0.0,
            'max_drawdown': 0.0
        }
    
    total_trades = len(trades)
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0.0
    
    total_pnl = sum(t['pnl'] for t in trades)
    
    avg_winner = sum(t['pnl'] for t in winners) / len(winners) if winners else 0.0
    avg_loser = sum(t['pnl'] for t in losers) / len(losers) if losers else 0.0
    
    gross_profit = sum(t['pnl'] for t in winners)
    gross_loss = abs(sum(t['pnl'] for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_r = sum(t['r_result'] for t in trades) / total_trades if total_trades > 0 else 0.0
    
    equity_curve = []
    running_pnl = 0
    for t in trades:
        running_pnl += t['pnl']
        equity_curve.append(running_pnl)
    
    max_drawdown = 0
    peak = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        drawdown = peak - eq
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return {
        'total_trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_winner': round(avg_winner, 2),
        'avg_loser': round(avg_loser, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_r': round(avg_r, 2),
        'max_drawdown': round(max_drawdown, 2)
    }
