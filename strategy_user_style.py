"""
Strategy User Style: BoS Confirmation + Golden Pocket Entry + Fib Extension Targets

User's Trading Style:
1. Identify HTF Supply/Demand zones (Daily/Weekly)
2. Wait for price to reach zone and get rejected
3. Wait for BoS (Break of Structure) confirmation
4. Enter at Golden Pocket (0.618-0.66 retracement) AFTER BoS
5. Target at Fib extensions (-0.25, -0.68, -1.0, -1.42, -1.618)

Key Differences from V3 Pro:
- BoS confirmation required before entry
- Entry at Golden Pocket AFTER rejection (not at zone)
- Fib extension targets instead of structural swing levels
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd


@dataclass
class SwingZone:
    """Supply/Demand zone with retracement levels."""
    zone_type: str
    high: float
    low: float
    swing_high: float
    swing_low: float
    golden_pocket_high: float
    golden_pocket_low: float
    strength: int
    created_bar: int
    is_fresh: bool = True


@dataclass
class TradeSignal:
    """Trade signal with full context."""
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    r_multiple: float
    status: str
    zone: SwingZone
    entry_type: str
    confluence: int
    reasoning: str
    timestamp: str


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


def find_swing_points(candles: List[Dict], lookback: int = 3) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Find swing highs and lows with their bar indices."""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i]['high']
        low = candles[i]['low']
        
        is_swing_high = True
        is_swing_low = True
        
        for j in range(i - lookback, i + lookback + 1):
            if j != i:
                if candles[j]['high'] >= high:
                    is_swing_high = False
                if candles[j]['low'] <= low:
                    is_swing_low = False
        
        if is_swing_high:
            swing_highs.append((i, high))
        if is_swing_low:
            swing_lows.append((i, low))
    
    return swing_highs, swing_lows


def identify_supply_demand_zones(candles: List[Dict], atr: float) -> List[SwingZone]:
    """Identify Supply and Demand zones based on price structure."""
    if len(candles) < 20:
        return []
    
    zones = []
    swing_highs, swing_lows = find_swing_points(candles, lookback=3)
    
    for idx, high in swing_highs[-10:]:
        if idx >= len(candles) - 1:
            continue
        
        zone_high = high
        zone_low = high - atr * 0.5
        
        prev_low = min(c['low'] for c in candles[max(0, idx-10):idx])
        gp_low, gp_high = calculate_golden_pocket(high, prev_low, 'short')
        
        impulse_down = high - min(c['low'] for c in candles[idx:min(len(candles), idx+5)])
        strength = 1
        if impulse_down > atr * 1.5:
            strength = 2
        if impulse_down > atr * 2.5:
            strength = 3
        
        zones.append(SwingZone(
            zone_type='supply',
            high=zone_high,
            low=zone_low,
            swing_high=high,
            swing_low=prev_low,
            golden_pocket_high=gp_high,
            golden_pocket_low=gp_low,
            strength=strength,
            created_bar=idx,
            is_fresh=True
        ))
    
    for idx, low in swing_lows[-10:]:
        if idx >= len(candles) - 1:
            continue
        
        zone_low = low
        zone_high = low + atr * 0.5
        
        prev_high = max(c['high'] for c in candles[max(0, idx-10):idx])
        gp_low, gp_high = calculate_golden_pocket(prev_high, low, 'long')
        
        impulse_up = max(c['high'] for c in candles[idx:min(len(candles), idx+5)]) - low
        strength = 1
        if impulse_up > atr * 1.5:
            strength = 2
        if impulse_up > atr * 2.5:
            strength = 3
        
        zones.append(SwingZone(
            zone_type='demand',
            high=zone_high,
            low=zone_low,
            swing_high=prev_high,
            swing_low=low,
            golden_pocket_high=gp_high,
            golden_pocket_low=gp_low,
            strength=strength,
            created_bar=idx,
            is_fresh=True
        ))
    
    return zones


def calculate_golden_pocket(swing_high: float, swing_low: float, direction: str) -> Tuple[float, float]:
    """Calculate Golden Pocket zone (61.8% - 66% retracement)."""
    range_size = swing_high - swing_low
    
    if direction == 'long':
        gp_high = swing_high - (range_size * 0.618)
        gp_low = swing_high - (range_size * 0.66)
    else:
        gp_low = swing_low + (range_size * 0.618)
        gp_high = swing_low + (range_size * 0.66)
    
    return min(gp_high, gp_low), max(gp_high, gp_low)


def calculate_fib_extensions(swing_high: float, swing_low: float, direction: str) -> Dict[str, float]:
    """
    Calculate Fibonacci extension levels for targets.
    
    For LONG (from demand): Extensions go above the swing high
    For SHORT (from supply): Extensions go below the swing low
    """
    range_size = swing_high - swing_low
    
    if direction == 'long':
        return {
            '-0.25': swing_high + (range_size * 0.25),
            '-0.5': swing_high + (range_size * 0.5),
            '-0.68': swing_high + (range_size * 0.68),
            '-1.0': swing_high + range_size,
            '-1.42': swing_high + (range_size * 1.42),
            '-1.618': swing_high + (range_size * 1.618),
        }
    else:
        return {
            '-0.25': swing_low - (range_size * 0.25),
            '-0.5': swing_low - (range_size * 0.5),
            '-0.68': swing_low - (range_size * 0.68),
            '-1.0': swing_low - range_size,
            '-1.42': swing_low - (range_size * 1.42),
            '-1.618': swing_low - (range_size * 1.618),
        }


def detect_bos(candles: List[Dict], direction: str, lookback: int = 10) -> Tuple[bool, Optional[float]]:
    """
    Detect Break of Structure (BoS).
    
    For LONG: Price must break above a recent swing high (confirms demand zone)
    For SHORT: Price must break below a recent swing low (confirms supply zone)
    
    Returns: (bos_confirmed, bos_level)
    """
    if len(candles) < lookback + 5:
        return False, None
    
    recent = candles[-lookback:]
    current_close = candles[-1]['close']
    
    swing_highs, swing_lows = find_swing_points(candles[:-3], lookback=2)
    
    if direction == 'long':
        recent_swing_highs = [(idx, price) for idx, price in swing_highs if idx >= len(candles) - lookback - 5]
        if recent_swing_highs:
            bos_level = max(price for _, price in recent_swing_highs[-3:]) if recent_swing_highs else None
            if bos_level and current_close > bos_level:
                return True, bos_level
    else:
        recent_swing_lows = [(idx, price) for idx, price in swing_lows if idx >= len(candles) - lookback - 5]
        if recent_swing_lows:
            bos_level = min(price for _, price in recent_swing_lows[-3:]) if recent_swing_lows else None
            if bos_level and current_close < bos_level:
                return True, bos_level
    
    return False, None


def price_in_golden_pocket(price: float, zone: SwingZone, direction: str) -> bool:
    """Check if price is in the Golden Pocket zone."""
    return zone.golden_pocket_low <= price <= zone.golden_pocket_high


def detect_daily_trend(candles: List[Dict], lookback: int = 20) -> str:
    """Detect daily trend using EMA crossover and swing structure."""
    if len(candles) < lookback + 10:
        return 'neutral'
    
    closes = [c['close'] for c in candles]
    
    ema10 = []
    ema20 = []
    
    mult10 = 2 / (10 + 1)
    mult20 = 2 / (20 + 1)
    
    ema10_val = sum(closes[:10]) / 10
    ema20_val = sum(closes[:20]) / 20
    
    for i in range(20, len(closes)):
        ema10_val = (closes[i] * mult10) + (ema10_val * (1 - mult10))
        ema20_val = (closes[i] * mult20) + (ema20_val * (1 - mult20))
        ema10.append(ema10_val)
        ema20.append(ema20_val)
    
    if len(ema10) < 3:
        return 'neutral'
    
    ema_bullish = ema10[-1] > ema20[-1]
    ema_bearish = ema10[-1] < ema20[-1]
    
    recent = candles[-lookback:]
    highs = [c['high'] for c in recent]
    lows = [c['low'] for c in recent]
    
    higher_highs = highs[-1] > max(highs[:-5]) if len(highs) > 5 else False
    higher_lows = min(lows[-5:]) > min(lows[:-5]) if len(lows) > 5 else False
    lower_lows = lows[-1] < min(lows[:-5]) if len(lows) > 5 else False
    lower_highs = max(highs[-5:]) < max(highs[:-5]) if len(highs) > 5 else False
    
    if ema_bullish and (higher_highs or higher_lows):
        return 'bullish'
    elif ema_bearish and (lower_lows or lower_highs):
        return 'bearish'
    
    return 'neutral'


def generate_user_style_signal(
    symbol: str,
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_rr: float = 1.5,
    min_confluence: int = 3
) -> Optional[TradeSignal]:
    """
    Generate User Style trading signal.
    
    Entry Requirements:
    1. Price touched Supply/Demand zone (within last 10 bars)
    2. BoS confirmed (break of structure in trade direction)
    3. Current price in Golden Pocket (0.618-0.66 retracement)
    4. Trend aligned
    
    Targets: Fibonacci extensions (-0.25 for 1R, then -0.68/-1.0 for full TP)
    """
    if len(daily_candles) < 60 or len(weekly_candles) < 12:
        return None
    
    current_bar = daily_candles[-1]
    current_price = current_bar['close']
    current_time = current_bar['time']
    
    daily_atr = calculate_atr(daily_candles, 14)
    weekly_atr = calculate_atr(weekly_candles, 14)
    
    if daily_atr == 0:
        return None
    
    daily_trend = detect_daily_trend(daily_candles, 20)
    
    daily_zones = identify_supply_demand_zones(daily_candles, daily_atr)
    weekly_zones = identify_supply_demand_zones(weekly_candles, weekly_atr)
    
    all_zones = []
    for z in weekly_zones:
        z.strength += 2
        all_zones.append(('weekly', z))
    for z in daily_zones:
        if z.strength >= 2:
            all_zones.append(('daily', z))
    
    for timeframe, zone in all_zones:
        if zone.zone_type == 'demand':
            direction = 'long'
            if daily_trend == 'bearish':
                continue
        else:
            direction = 'short'
            if daily_trend == 'bullish':
                continue
        
        zone_touched = False
        for i in range(-10, 0):
            if i >= -len(daily_candles):
                bar = daily_candles[i]
                if zone.zone_type == 'demand':
                    if bar['low'] <= zone.high:
                        zone_touched = True
                        break
                else:
                    if bar['high'] >= zone.low:
                        zone_touched = True
                        break
        
        if not zone_touched:
            continue
        
        bos_confirmed, bos_level = detect_bos(daily_candles, direction, lookback=10)
        if not bos_confirmed:
            continue
        
        in_gp = price_in_golden_pocket(current_price, zone, direction)
        if not in_gp:
            continue
        
        confluence = 0
        reasoning_parts = []
        
        confluence += 1
        reasoning_parts.append("Zone touched")
        
        confluence += 1
        reasoning_parts.append(f"BoS confirmed at {bos_level:.5f}")
        
        confluence += 1
        reasoning_parts.append("Price in Golden Pocket")
        
        if timeframe == 'weekly':
            confluence += 1
            reasoning_parts.append("Weekly timeframe zone")
        
        if daily_trend == direction.replace('long', 'bullish').replace('short', 'bearish'):
            confluence += 1
            reasoning_parts.append(f"Trend aligned: {daily_trend}")
        
        if zone.strength >= 3:
            confluence += 1
            reasoning_parts.append("Strong zone (large impulse)")
        
        if confluence < min_confluence:
            continue
        
        entry = current_price
        
        if direction == 'long':
            stop_loss = zone.swing_low - (daily_atr * 0.3)
        else:
            stop_loss = zone.swing_high + (daily_atr * 0.3)
        
        risk = abs(entry - stop_loss)
        min_sl = daily_atr * 0.75
        if risk < min_sl:
            if direction == 'long':
                stop_loss = entry - min_sl
            else:
                stop_loss = entry + min_sl
            risk = min_sl
        
        fib_extensions = calculate_fib_extensions(zone.swing_high, zone.swing_low, direction)
        
        if direction == 'long':
            tp_options = [
                (fib_extensions['-0.25'], 0.25),
                (fib_extensions['-0.5'], 0.5),
                (fib_extensions['-0.68'], 0.68),
                (fib_extensions['-1.0'], 1.0),
            ]
            for tp, mult in tp_options:
                rr = (tp - entry) / risk
                if rr >= min_rr:
                    take_profit = tp
                    break
            else:
                take_profit = fib_extensions['-0.68']
        else:
            tp_options = [
                (fib_extensions['-0.25'], 0.25),
                (fib_extensions['-0.5'], 0.5),
                (fib_extensions['-0.68'], 0.68),
                (fib_extensions['-1.0'], 1.0),
            ]
            for tp, mult in tp_options:
                rr = (entry - tp) / risk
                if rr >= min_rr:
                    take_profit = tp
                    break
            else:
                take_profit = fib_extensions['-0.68']
        
        if direction == 'long':
            r_multiple = (take_profit - entry) / risk
        else:
            r_multiple = (entry - take_profit) / risk
        
        if r_multiple < min_rr:
            continue
        
        r_multiple = min(r_multiple, 8.0)
        
        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            r_multiple=r_multiple,
            status='pending',
            zone=zone,
            entry_type='bos_golden_pocket',
            confluence=confluence,
            reasoning=' | '.join(reasoning_parts),
            timestamp=str(current_time)
        )
    
    return None


def backtest_user_style(
    symbol: str,
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_rr: float = 1.5,
    min_confluence: int = 3,
    partial_tp: bool = True,
    partial_tp_r: float = 1.0
) -> List[Dict]:
    """
    Backtest the User Style strategy.
    
    Returns list of trade dictionaries with entry, exit, P&L info.
    """
    if len(daily_candles) < 100 or len(weekly_candles) < 20:
        return []
    
    trades = []
    position = None
    
    for i in range(100, len(daily_candles)):
        current_candles = daily_candles[:i+1]
        current_bar = current_candles[-1]
        current_time = current_bar['time']
        
        weekly_cutoff = current_time
        filtered_weekly = [w for w in weekly_candles if w['time'] <= weekly_cutoff]
        
        if len(filtered_weekly) < 20:
            continue
        
        if position is not None:
            high = current_bar['high']
            low = current_bar['low']
            close = current_bar['close']
            
            if position['direction'] == 'long':
                if partial_tp and not position.get('partial_taken') and high >= position.get('partial_tp', position['tp']):
                    position['partial_taken'] = True
                    position['partial_exit_time'] = current_time
                    position['sl'] = position['entry'] + (position['risk'] * 0.1)
                
                if low <= position['sl']:
                    exit_price = position['sl']
                    pnl_r = (exit_price - position['entry']) / position['risk']
                    
                    if position.get('partial_taken'):
                        pnl_r = partial_tp_r * 0.5 + pnl_r * 0.5
                        result = 'PARTIAL_WIN' if pnl_r > 0 else 'BE'
                    else:
                        result = 'LOSS'
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry': position['entry'],
                        'exit': exit_price,
                        'stop_loss': position['original_sl'],
                        'take_profit': position['tp'],
                        'pnl_usd': pnl_r * 100,
                        'r_multiple': round(pnl_r, 2),
                        'result': result,
                        'entry_type': position['entry_type'],
                        'zone_type': position['zone_type'],
                        'confluence': position['confluence'],
                        'reasoning': position['reasoning'],
                        'duration_days': (current_time - position['entry_time']).days if hasattr(current_time, 'days') else 0,
                        'highest_r': position.get('highest_r', 0),
                        'partial_taken': position.get('partial_taken', False)
                    })
                    position = None
                    continue
                
                elif high >= position['tp']:
                    exit_price = position['tp']
                    pnl_r = (exit_price - position['entry']) / position['risk']
                    
                    if position.get('partial_taken'):
                        pnl_r = partial_tp_r * 0.5 + pnl_r * 0.5
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry': position['entry'],
                        'exit': exit_price,
                        'stop_loss': position['original_sl'],
                        'take_profit': position['tp'],
                        'pnl_usd': pnl_r * 100,
                        'r_multiple': round(pnl_r, 2),
                        'result': 'WIN',
                        'entry_type': position['entry_type'],
                        'zone_type': position['zone_type'],
                        'confluence': position['confluence'],
                        'reasoning': position['reasoning'],
                        'duration_days': (current_time - position['entry_time']).days if hasattr(current_time, 'days') else 0,
                        'highest_r': position.get('highest_r', 0),
                        'partial_taken': position.get('partial_taken', False)
                    })
                    position = None
                    continue
                
                current_r = (high - position['entry']) / position['risk']
                position['highest_r'] = max(position.get('highest_r', 0), current_r)
                
            else:
                if partial_tp and not position.get('partial_taken') and low <= position.get('partial_tp', position['tp']):
                    position['partial_taken'] = True
                    position['partial_exit_time'] = current_time
                    position['sl'] = position['entry'] - (position['risk'] * 0.1)
                
                if high >= position['sl']:
                    exit_price = position['sl']
                    pnl_r = (position['entry'] - exit_price) / position['risk']
                    
                    if position.get('partial_taken'):
                        pnl_r = partial_tp_r * 0.5 + pnl_r * 0.5
                        result = 'PARTIAL_WIN' if pnl_r > 0 else 'BE'
                    else:
                        result = 'LOSS'
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry': position['entry'],
                        'exit': exit_price,
                        'stop_loss': position['original_sl'],
                        'take_profit': position['tp'],
                        'pnl_usd': pnl_r * 100,
                        'r_multiple': round(pnl_r, 2),
                        'result': result,
                        'entry_type': position['entry_type'],
                        'zone_type': position['zone_type'],
                        'confluence': position['confluence'],
                        'reasoning': position['reasoning'],
                        'duration_days': (current_time - position['entry_time']).days if hasattr(current_time, 'days') else 0,
                        'highest_r': position.get('highest_r', 0),
                        'partial_taken': position.get('partial_taken', False)
                    })
                    position = None
                    continue
                
                elif low <= position['tp']:
                    exit_price = position['tp']
                    pnl_r = (position['entry'] - exit_price) / position['risk']
                    
                    if position.get('partial_taken'):
                        pnl_r = partial_tp_r * 0.5 + pnl_r * 0.5
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry': position['entry'],
                        'exit': exit_price,
                        'stop_loss': position['original_sl'],
                        'take_profit': position['tp'],
                        'pnl_usd': pnl_r * 100,
                        'r_multiple': round(pnl_r, 2),
                        'result': 'WIN',
                        'entry_type': position['entry_type'],
                        'zone_type': position['zone_type'],
                        'confluence': position['confluence'],
                        'reasoning': position['reasoning'],
                        'duration_days': (current_time - position['entry_time']).days if hasattr(current_time, 'days') else 0,
                        'highest_r': position.get('highest_r', 0),
                        'partial_taken': position.get('partial_taken', False)
                    })
                    position = None
                    continue
                
                current_r = (position['entry'] - low) / position['risk']
                position['highest_r'] = max(position.get('highest_r', 0), current_r)
        
        if position is None:
            signal = generate_user_style_signal(
                symbol, 
                current_candles, 
                filtered_weekly,
                min_rr=min_rr,
                min_confluence=min_confluence
            )
            
            if signal:
                risk = abs(signal.entry - signal.stop_loss)
                
                if signal.direction == 'long':
                    partial_tp_price = signal.entry + (risk * partial_tp_r)
                else:
                    partial_tp_price = signal.entry - (risk * partial_tp_r)
                
                position = {
                    'entry_time': current_time,
                    'entry': signal.entry,
                    'sl': signal.stop_loss,
                    'original_sl': signal.stop_loss,
                    'tp': signal.take_profit,
                    'partial_tp': partial_tp_price,
                    'direction': signal.direction,
                    'risk': risk,
                    'entry_type': signal.entry_type,
                    'zone_type': signal.zone.zone_type,
                    'confluence': signal.confluence,
                    'reasoning': signal.reasoning,
                    'highest_r': 0,
                    'partial_taken': False
                }
    
    return trades


def run_user_style_backtest_for_asset(
    asset: str,
    year: int,
    min_rr: float = 1.5,
    min_confluence: int = 3,
    partial_tp: bool = True,
    partial_tp_r: float = 1.0,
    start_month: int = 1,
    end_month: int = 12
) -> Dict[str, Any]:
    """Run User Style backtest for a specific asset and year with month range."""
    from data import get_ohlcv
    from datetime import datetime
    
    daily_candles = get_ohlcv(asset, 'D', count=750)
    weekly_candles = get_ohlcv(asset, 'W', count=200)
    
    if not daily_candles or not weekly_candles:
        return {'error': f'No data for {asset}'}
    
    start_date = datetime(year, start_month, 1)
    if end_month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, end_month + 1, 1)
    
    daily_filtered = [c for c in daily_candles if start_date <= c['time'] < end_date]
    
    all_trades = backtest_user_style(
        asset,
        daily_candles,
        weekly_candles,
        min_rr=min_rr,
        min_confluence=min_confluence,
        partial_tp=partial_tp,
        partial_tp_r=partial_tp_r
    )
    
    trades = [t for t in all_trades if start_date <= t['entry_time'] < end_date]
    
    if not trades:
        return {
            'asset': asset,
            'year': year,
            'start_month': start_month,
            'end_month': end_month,
            'trades': [],
            'total_trades': 0,
            'total_r': 0,
            'win_rate': 0,
            'wins': 0,
            'losses': 0,
            'be': 0
        }
    
    total_r = sum(t['r_multiple'] for t in trades)
    wins = sum(1 for t in trades if t['result'] in ['WIN', 'PARTIAL_WIN'])
    losses = sum(1 for t in trades if t['result'] == 'LOSS')
    be = sum(1 for t in trades if t['result'] == 'BE')
    
    return {
        'asset': asset,
        'year': year,
        'start_month': start_month,
        'end_month': end_month,
        'trades': trades,
        'total_trades': len(trades),
        'total_r': round(total_r, 2),
        'win_rate': round(100 * wins / len(trades), 1) if trades else 0,
        'wins': wins,
        'losses': losses,
        'be': be
    }
