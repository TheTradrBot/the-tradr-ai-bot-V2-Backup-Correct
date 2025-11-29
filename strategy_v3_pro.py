"""
Strategy V3 Pro: Daily Swing Trading with Fibonacci + Weekly S/R
Enhanced with optional Monthly S/R from 4H data

Key Methods:
- Optimal entry zone at 0.5-0.66 Fibonacci retracement (expanded from golden pocket)
- Wyckoff Spring/Upthrust for reversal detection  
- Break of Structure (BoS) confirmation
- Weekly S/R levels as key confluence
- OPTIONAL: Monthly S/R from 4H candle aggregation (if improves performance)
- Trade duration: 2-8 days
- Fibonacci Extension Take Profits at -0.25, -0.68, -1.0 levels

Constraints: NO RSI, NO MACD, NO SMC
Target: 70%+ yearly return, pass 5%ers every month
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


def calculate_optimal_entry_zone(swing_high: float, swing_low: float, direction: str) -> Tuple[float, float]:
    """
    Calculate Optimal Entry Zone (50% - 66% Fibonacci retracement).
    This wider zone matches the TradingView approach for better entries.
    For bullish: measure retracement from swing high back toward swing low
    For bearish: measure retracement from swing low back toward swing high
    """
    range_size = swing_high - swing_low
    
    if direction == 'long':
        zone_high = swing_high - (range_size * 0.5)
        zone_low = swing_high - (range_size * 0.66)
    else:
        zone_low = swing_low + (range_size * 0.5)
        zone_high = swing_low + (range_size * 0.66)
    
    return min(zone_high, zone_low), max(zone_high, zone_low)


def calculate_fib_extension_tps(swing_high: float, swing_low: float, direction: str) -> Tuple[float, float, float]:
    """
    Calculate Fibonacci Extension Take Profit levels at -0.25, -0.68, -1.0.
    These are extension levels beyond the swing range.
    """
    range_size = swing_high - swing_low
    
    if direction == 'long':
        tp1 = swing_high + (range_size * 0.25)
        tp2 = swing_high + (range_size * 0.68)
        tp3 = swing_high + (range_size * 1.0)
    else:
        tp1 = swing_low - (range_size * 0.25)
        tp2 = swing_low - (range_size * 0.68)
        tp3 = swing_low - (range_size * 1.0)
    
    return tp1, tp2, tp3


def detect_break_of_structure(candles: List[Dict], direction: str, lookback: int = 20) -> Tuple[bool, float]:
    """
    Detect Break of Structure (BoS) - when price breaks a significant swing point.
    Returns (bos_detected, bos_level).
    """
    if len(candles) < lookback + 5:
        return False, 0.0
    
    recent = candles[-lookback:]
    swing_highs, swing_lows = find_swing_points(recent, lookback=2)
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, 0.0
    
    current_close = candles[-1]['close']
    current_high = candles[-1]['high']
    current_low = candles[-1]['low']
    
    if direction == 'long':
        last_significant_high = swing_highs[-2][1] if len(swing_highs) >= 2 else swing_highs[-1][1]
        if current_close > last_significant_high or current_high > last_significant_high:
            return True, last_significant_high
    else:
        last_significant_low = swing_lows[-2][1] if len(swing_lows) >= 2 else swing_lows[-1][1]
        if current_close < last_significant_low or current_low < last_significant_low:
            return True, last_significant_low
    
    return False, 0.0


def find_weekly_sr_levels(weekly_candles: List[Dict], lookback: int = 12) -> List[Dict]:
    """
    Find Weekly Support/Resistance levels from weekly candle data.
    Returns list of S/R levels with type and price.
    """
    if len(weekly_candles) < lookback:
        return []
    
    recent = weekly_candles[-lookback:]
    swing_highs, swing_lows = find_swing_points(recent, lookback=2)
    
    sr_levels = []
    
    for idx, price in swing_highs:
        sr_levels.append({
            'type': 'resistance',
            'price': price,
            'bar_idx': idx
        })
    
    for idx, price in swing_lows:
        sr_levels.append({
            'type': 'support', 
            'price': price,
            'bar_idx': idx
        })
    
    sr_levels.sort(key=lambda x: x['price'], reverse=True)
    
    return sr_levels


def aggregate_daily_to_4h(daily_candles: List[Dict]) -> List[Dict]:
    """Convert daily candles to 4H buckets by averaging."""
    if len(daily_candles) < 2:
        return daily_candles
    
    candles_4h = {}
    
    for candle in daily_candles:
        time_val = candle['time']
        if isinstance(time_val, str):
            try:
                dt = datetime.fromisoformat(time_val.replace('Z', '+00:00'))
            except:
                continue
        else:
            dt = time_val
        
        hour_bucket = (dt.hour // 4) * 4
        key = f"{dt.year}-{dt.month:02d}-{dt.day:02d}-{hour_bucket:02d}"
        
        if key not in candles_4h:
            candles_4h[key] = {
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close']
            }
        else:
            candles_4h[key]['high'] = max(candles_4h[key]['high'], candle['high'])
            candles_4h[key]['low'] = min(candles_4h[key]['low'], candle['low'])
            candles_4h[key]['close'] = candle['close']
    
    return [{'time': k, **v} for k, v in sorted(candles_4h.items())]


def extract_monthly_sr_from_daily(daily_candles: List[Dict]) -> List[Dict]:
    """Extract Monthly S/R levels by aggregating daily candles into monthly bars."""
    if len(daily_candles) < 30:
        return []
    
    monthly_bars = {}
    
    for candle in daily_candles:
        time_val = candle['time']
        if isinstance(time_val, str):
            try:
                dt = datetime.fromisoformat(time_val.replace('Z', '+00:00'))
            except:
                continue
        else:
            dt = time_val
        
        month_key = f"{dt.year}-{dt.month:02d}"
        
        if month_key not in monthly_bars:
            monthly_bars[month_key] = {
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close']
            }
        else:
            monthly_bars[month_key]['high'] = max(monthly_bars[month_key]['high'], candle['high'])
            monthly_bars[month_key]['low'] = min(monthly_bars[month_key]['low'], candle['low'])
            monthly_bars[month_key]['close'] = candle['close']
    
    monthly_candles = [{'time': k, **v} for k, v in sorted(monthly_bars.items())]
    return find_weekly_sr_levels(monthly_candles, lookback=min(12, len(monthly_candles)))


def price_near_monthly_sr(price: float, monthly_sr: List[Dict], daily_atr: float) -> bool:
    """Check if price is near a monthly S/R level."""
    if not monthly_sr:
        return False
    buffer = daily_atr * 1.5
    for sr in monthly_sr:
        if abs(price - sr['price']) <= buffer:
            return True
    return False


def price_near_weekly_sr(price: float, sr_levels: List[Dict], atr: float, buffer_mult: float = 0.5) -> Tuple[bool, str]:
    """
    Check if price is near a weekly S/R level.
    Returns (is_near, sr_type).
    """
    buffer = atr * buffer_mult
    
    for sr in sr_levels:
        if abs(price - sr['price']) <= buffer:
            return True, sr['type']
    
    return False, ''


def identify_supply_demand_zones(candles: List[Dict], atr: float) -> List[SwingZone]:
    """
    Identify Supply and Demand zones based on price structure.
    Uses significant swing points and calculates Optimal Entry Zone (0.5-0.66 Fib).
    """
    if len(candles) < 30:
        return []
    
    swing_highs, swing_lows = find_swing_points(candles, lookback=3)
    
    zones = []
    
    for i in range(len(swing_lows) - 1):
        low_idx, low_price = swing_lows[i]
        
        for j in range(len(swing_highs)):
            high_idx, high_price = swing_highs[j]
            
            if high_idx > low_idx:
                range_size = high_price - low_price
                if range_size > atr * 2:
                    oez_low, oez_high = calculate_optimal_entry_zone(high_price, low_price, 'long')
                    
                    zone_low = low_price
                    zone_high = low_price + (atr * 0.5)
                    
                    strength = 2 if range_size > atr * 4 else 1
                    
                    zones.append(SwingZone(
                        zone_type='demand',
                        high=zone_high,
                        low=zone_low,
                        swing_high=high_price,
                        swing_low=low_price,
                        golden_pocket_high=oez_high,
                        golden_pocket_low=oez_low,
                        strength=strength,
                        created_bar=low_idx,
                        is_fresh=True
                    ))
                break
    
    for i in range(len(swing_highs) - 1):
        high_idx, high_price = swing_highs[i]
        
        for j in range(len(swing_lows)):
            low_idx, low_price = swing_lows[j]
            
            if low_idx > high_idx:
                range_size = high_price - low_price
                if range_size > atr * 2:
                    oez_high, oez_low = calculate_optimal_entry_zone(high_price, low_price, 'short')
                    
                    zone_high = high_price
                    zone_low = high_price - (atr * 0.5)
                    
                    strength = 2 if range_size > atr * 4 else 1
                    
                    zones.append(SwingZone(
                        zone_type='supply',
                        high=zone_high,
                        low=zone_low,
                        swing_high=high_price,
                        swing_low=low_price,
                        golden_pocket_high=oez_high,
                        golden_pocket_low=oez_low,
                        strength=strength,
                        created_bar=high_idx,
                        is_fresh=True
                    ))
                break
    
    return zones


def detect_wyckoff_spring(candles: List[Dict], zone: SwingZone, atr: float, lookback: int = 10) -> bool:
    """
    Detect Wyckoff Spring pattern at demand zone.
    Spring = Price breaks below support briefly, then reverses sharply back.
    
    Strict criteria:
    - Break below support by less than 1 ATR (not a breakdown)
    - Close back above support
    - Next bar must be bullish with close above spring high
    - Pattern should trap bears
    """
    if zone.zone_type != 'demand' or len(candles) < lookback:
        return False
    
    recent = candles[-lookback:]
    support = zone.low
    
    spring_bar = None
    for i, bar in enumerate(recent[:-1]):
        break_below = support - bar['low']
        
        if break_below > 0 and break_below < atr:
            if bar['close'] > support:
                spring_bar = i
                break
    
    if spring_bar is not None and spring_bar + 2 < len(recent):
        spring = recent[spring_bar]
        next_bar = recent[spring_bar + 1]
        confirm_bar = recent[spring_bar + 2]
        
        if next_bar['close'] > next_bar['open'] and next_bar['close'] > spring['high']:
            if confirm_bar['close'] > confirm_bar['open'] or confirm_bar['close'] > next_bar['close']:
                return True
    
    return False


def detect_wyckoff_upthrust(candles: List[Dict], zone: SwingZone, atr: float, lookback: int = 10) -> bool:
    """
    Detect Wyckoff Upthrust pattern at supply zone.
    Upthrust = Price breaks above resistance briefly (but not too far), then reverses sharply back.
    
    Strict criteria:
    - Break above resistance by less than 1 ATR (not a breakout)
    - Close back below resistance
    - Next bar must be bearish with close below upthrust low
    - Pattern should trap bulls
    """
    if zone.zone_type != 'supply' or len(candles) < lookback:
        return False
    
    recent = candles[-lookback:]
    resistance = zone.high
    
    upthrust_bar = None
    for i, bar in enumerate(recent[:-1]):
        break_above = bar['high'] - resistance
        
        if break_above > 0 and break_above < atr:
            if bar['close'] < resistance:
                upthrust_bar = i
                break
    
    if upthrust_bar is not None and upthrust_bar + 2 < len(recent):
        upthrust = recent[upthrust_bar]
        next_bar = recent[upthrust_bar + 1]
        confirm_bar = recent[upthrust_bar + 2]
        
        if next_bar['close'] < next_bar['open'] and next_bar['close'] < upthrust['low']:
            if confirm_bar['close'] < confirm_bar['open'] or confirm_bar['close'] < next_bar['close']:
                return True
    
    return False


def price_in_optimal_zone(price: float, zone: SwingZone, direction: str) -> bool:
    """Check if price is within the Optimal Entry Zone (0.5-0.66 Fib)."""
    return zone.golden_pocket_low <= price <= zone.golden_pocket_high


def price_at_zone(price: float, zone: SwingZone, atr: float) -> bool:
    """Check if price is at or near the zone."""
    buffer = atr * 0.3
    return (zone.low - buffer) <= price <= (zone.high + buffer)


def find_fib_extension_tp(zone: SwingZone, entry: float, stop_loss: float, direction: str, min_rr: float = 2.5) -> Optional[Tuple[float, float, float]]:
    """
    Find Fibonacci Extension Take Profit levels.
    Uses -0.25, -0.68, -1.0 extension levels from the swing range.
    Returns (tp1, tp2, tp3) or None if min_rr not met.
    """
    tp1, tp2, tp3 = calculate_fib_extension_tps(zone.swing_high, zone.swing_low, direction)
    
    risk = abs(entry - stop_loss)
    if risk == 0:
        return None
    
    if direction == 'long':
        tp1_rr = (tp1 - entry) / risk
        tp2_rr = (tp2 - entry) / risk
        tp3_rr = (tp3 - entry) / risk
    else:
        tp1_rr = (entry - tp1) / risk
        tp2_rr = (entry - tp2) / risk
        tp3_rr = (entry - tp3) / risk
    
    if tp1_rr >= min_rr:
        return tp1, tp2, tp3
    elif tp2_rr >= min_rr:
        return tp2, tp3, tp3
    elif tp3_rr >= min_rr:
        return tp3, tp3, tp3
    
    return None


def detect_daily_trend(candles: List[Dict], lookback: int = 20) -> str:
    """Detect trend on daily timeframe using EMA crossover + structure."""
    if len(candles) < lookback + 5:
        return 'neutral'
    
    closes = [c['close'] for c in candles]
    
    ema_10 = sum(closes[-10:]) / 10
    ema_20 = sum(closes[-20:]) / 20
    
    recent = candles[-lookback:]
    swing_highs, swing_lows = find_swing_points(recent, lookback=2)
    
    has_hh_hl = False
    has_lh_ll = False
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        higher_highs = swing_highs[-1][1] > swing_highs[-2][1]
        higher_lows = swing_lows[-1][1] > swing_lows[-2][1]
        lower_highs = swing_highs[-1][1] < swing_highs[-2][1]
        lower_lows = swing_lows[-1][1] < swing_lows[-2][1]
        
        has_hh_hl = higher_highs or higher_lows
        has_lh_ll = lower_highs or lower_lows
    
    current_close = candles[-1]['close']
    
    bullish_ema = ema_10 > ema_20 and current_close > ema_10
    bearish_ema = ema_10 < ema_20 and current_close < ema_10
    
    if bullish_ema and has_hh_hl:
        return 'bullish'
    elif bearish_ema and has_lh_ll:
        return 'bearish'
    
    if bullish_ema:
        return 'bullish'
    elif bearish_ema:
        return 'bearish'
    
    return 'neutral'


def generate_v3_pro_signal(
    symbol: str,
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_rr: float = 2.5,
    min_confluence: int = 3
) -> Optional[TradeSignal]:
    """
    Generate V3 Pro trading signal - FIBONACCI + WEEKLY S/R VERSION.
    
    Entry Types:
    1. Optimal Zone: Entry when price retraces to 0.5-0.66 Fib at fresh zone
    2. Wyckoff Spring: Entry after spring pattern at demand zone
    3. Wyckoff Upthrust: Entry after upthrust pattern at supply zone
    4. Zone Rejection: Entry on rejection candle at zone
    
    Key Features:
    - Optimal Entry Zone: 0.5-0.66 Fib retracement (wider than golden pocket)
    - Weekly S/R levels as confluence
    - Break of Structure (BoS) confirmation
    - Stop Loss at 1.0 Fib level (swing high/low with buffer)
    - Fibonacci Extension TPs at -0.25, -0.68, -1.0
    - Confluence score >= 3
    - Minimum R:R of 2.5:1
    - Target hold: 2-8 days
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
    
    weekly_sr_levels = find_weekly_sr_levels(weekly_candles, lookback=12)
    
    daily_zones = identify_supply_demand_zones(daily_candles, daily_atr)
    weekly_zones = identify_supply_demand_zones(weekly_candles, weekly_atr)
    
    all_zones = []
    for z in weekly_zones:
        z.strength += 2
        all_zones.append(('weekly', z))
    for z in daily_zones:
        if z.strength >= 1:
            all_zones.append(('daily', z))
    
    for timeframe, zone in all_zones:
        if not zone.is_fresh:
            continue
        
        if zone.zone_type == 'demand':
            direction = 'long'
        else:
            direction = 'short'
        
        confluence = 0
        reasoning_parts = []
        entry_type = None
        
        at_zone = price_at_zone(current_price, zone, daily_atr)
        in_oez = price_in_optimal_zone(current_price, zone, direction)
        
        if not at_zone and not in_oez:
            continue
        
        if in_oez:
            confluence += 2
            reasoning_parts.append("Price in Optimal Entry Zone (0.5-0.66 Fib)")
            entry_type = 'optimal_zone'
        
        if at_zone:
            confluence += 1
            reasoning_parts.append(f"Price at {zone.zone_type} zone")
        
        if timeframe == 'weekly':
            confluence += 2
            reasoning_parts.append("Weekly zone confluence")
        
        near_weekly_sr, sr_type = price_near_weekly_sr(current_price, weekly_sr_levels, daily_atr, 0.5)
        if near_weekly_sr:
            confluence += 1
            reasoning_parts.append(f"Near weekly {sr_type}")
        
        if daily_trend == 'bullish' and direction == 'long':
            confluence += 1
            reasoning_parts.append("Bullish daily trend")
        elif daily_trend == 'bearish' and direction == 'short':
            confluence += 1
            reasoning_parts.append("Bearish daily trend")
        
        bos_detected, bos_level = detect_break_of_structure(daily_candles, direction, 20)
        if bos_detected:
            confluence += 1
            reasoning_parts.append("Break of Structure confirmed")
        
        has_confirmation = False
        if detect_wyckoff_spring(daily_candles, zone, daily_atr, 10):
            confluence += 1
            has_confirmation = True
            reasoning_parts.append("Wyckoff spring pattern")
        
        if detect_wyckoff_upthrust(daily_candles, zone, daily_atr, 10):
            confluence += 1
            has_confirmation = True
            reasoning_parts.append("Wyckoff upthrust pattern")
        
        if not has_confirmation and entry_type == 'optimal_zone':
            continue
        
        if entry_type is None:
            entry_type = 'zone_entry'
        
        entry = current_price
        
        sl_buffer = daily_atr * 0.2
        
        if direction == 'long':
            stop_loss = zone.swing_low - sl_buffer
        else:
            stop_loss = zone.swing_high + sl_buffer
        
        fib_tps = find_fib_extension_tp(zone, entry, stop_loss, direction, min_rr)
        
        if fib_tps is None:
            continue
        
        take_profit = fib_tps[0]
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk < daily_atr * 0.5:
            continue
        
        r_multiple = reward / risk if risk > 0 else 0
        
        if r_multiple < min_rr:
            continue
        
        if r_multiple > 8.0:
            r_multiple = 8.0
            if direction == 'long':
                take_profit = entry + (risk * 8.0)
            else:
                take_profit = entry - (risk * 8.0)
        
        if confluence >= 4:
            status = 'ACTIVE'
        else:
            status = 'WATCHING'
        
        reasoning = " | ".join(reasoning_parts)
        
        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            r_multiple=round(r_multiple, 2),
            status=status,
            zone=zone,
            entry_type=entry_type,
            confluence=confluence,
            reasoning=reasoning,
            timestamp=current_time
        )
    
    return None


def format_time(time_val) -> str:
    """Convert time value to string format YYYY-MM-DD."""
    if isinstance(time_val, str):
        return time_val[:10]
    elif hasattr(time_val, 'strftime'):
        return time_val.strftime('%Y-%m-%d')
    return str(time_val)[:10]


def backtest_v3_pro(
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_rr: float = 2.5,
    min_confluence: int = 3,
    risk_per_trade: float = 250.0,
    cooldown_days: int = 2,
    max_daily_trades: int = 1,
    day_stagger: bool = True,
    partial_tp: bool = True,
    partial_tp_r: float = 1.5
) -> List[Dict]:
    """
    Backtest V3 Pro strategy on daily timeframe.
    
    FEATURES:
    - partial_tp: Close 50% at partial_tp_r (1.5R) to lock in profitable day
    - day_stagger: Limit 1 trade per day to spread wins across calendar days
    
    Trades held for 2-8 days typically.
    """
    trades = []
    last_trade_bar = -cooldown_days
    daily_trade_count = {}
    zone_used = {}
    days_with_open_trades = set()
    
    for i in range(60, len(daily_candles)):
        daily_slice = daily_candles[:i+1]
        
        weekly_end = 0
        for j, w in enumerate(weekly_candles):
            if w['time'] <= daily_candles[i]['time']:
                weekly_end = j + 1
        weekly_slice = weekly_candles[:weekly_end]
        
        if len(weekly_slice) < 12:
            continue
        
        current_bar = daily_candles[i]
        current_date = format_time(current_bar['time'])
        
        if current_date not in daily_trade_count:
            daily_trade_count[current_date] = 0
        
        if i - last_trade_bar < cooldown_days:
            continue
        
        if daily_trade_count[current_date] >= max_daily_trades:
            continue
        
        signal = generate_v3_pro_signal(
            symbol="BACKTEST",
            daily_candles=daily_slice,
            weekly_candles=weekly_slice,
            min_rr=min_rr,
            min_confluence=min_confluence
        )
        
        if signal is None:
            continue
        
        zone_key = f"{signal.zone.zone_type}_{signal.zone.high:.5f}_{signal.zone.low:.5f}"
        if zone_key in zone_used:
            continue
        
        if signal.status != 'ACTIVE':
            continue
        
        zone_used[zone_key] = True
        
        entry = signal.entry
        sl = signal.stop_loss
        tp = signal.take_profit
        
        hit_tp = False
        hit_sl = False
        exit_price = None
        exit_bar = None
        be_activated = False
        highest_r = 0.0
        partial_taken = False
        partial_exit_bar = None
        partial_exit_price = None
        
        risk = abs(entry - sl)
        
        if signal.direction == 'long':
            partial_level = entry + (risk * partial_tp_r) if partial_tp else None
        else:
            partial_level = entry - (risk * partial_tp_r) if partial_tp else None
        
        for j in range(i + 1, min(i + 40, len(daily_candles))):
            bar = daily_candles[j]
            
            if signal.direction == 'long':
                current_r = (bar['high'] - entry) / risk if risk > 0 else 0
                highest_r = max(highest_r, current_r)
                
                be_level = entry + risk
                
                if bar['low'] <= sl:
                    hit_sl = True
                    exit_price = sl if not be_activated else entry
                    exit_bar = j
                    break
                
                if partial_tp and not partial_taken and partial_level and bar['high'] >= partial_level:
                    partial_taken = True
                    partial_exit_bar = j
                    partial_exit_price = partial_level
                    be_activated = True
                    sl = entry
                
                if bar['high'] >= be_level and not be_activated:
                    be_activated = True
                    sl = entry
                
                if bar['high'] >= tp:
                    hit_tp = True
                    exit_price = tp
                    exit_bar = j
                    break
            else:
                current_r = (entry - bar['low']) / risk if risk > 0 else 0
                highest_r = max(highest_r, current_r)
                
                be_level = entry - risk
                
                if bar['high'] >= sl:
                    hit_sl = True
                    exit_price = sl if not be_activated else entry
                    exit_bar = j
                    break
                
                if partial_tp and not partial_taken and partial_level and bar['low'] <= partial_level:
                    partial_taken = True
                    partial_exit_bar = j
                    partial_exit_price = partial_level
                    be_activated = True
                    sl = entry
                
                if bar['low'] <= be_level and not be_activated:
                    be_activated = True
                    sl = entry
                
                if bar['low'] <= tp:
                    hit_tp = True
                    exit_price = tp
                    exit_bar = j
                    break
        
        if exit_price is None and exit_bar is None:
            exit_bar = min(i + 39, len(daily_candles) - 1)
            exit_price = daily_candles[exit_bar]['close']
        
        if exit_price is not None:
            risk_amount = abs(entry - signal.stop_loss)
            
            if partial_taken and partial_exit_price:
                if signal.direction == 'long':
                    partial_pnl = (partial_exit_price - entry) / risk_amount if risk_amount > 0 else 0
                    remaining_pnl = (exit_price - entry) / risk_amount if risk_amount > 0 else 0
                else:
                    partial_pnl = (entry - partial_exit_price) / risk_amount if risk_amount > 0 else 0
                    remaining_pnl = (entry - exit_price) / risk_amount if risk_amount > 0 else 0
                
                r_multiple = (partial_pnl * 0.5) + (remaining_pnl * 0.5)
                pnl_usd = r_multiple * risk_per_trade
                
                partial_date = format_time(daily_candles[partial_exit_bar]['time']) if partial_exit_bar else None
                
                if partial_pnl >= partial_tp_r * 0.9:
                    trades.append({
                        'entry_time': signal.timestamp,
                        'exit_time': daily_candles[partial_exit_bar]['time'] if partial_exit_bar else None,
                        'direction': signal.direction,
                        'entry': entry,
                        'exit': partial_exit_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': partial_level,
                        'pnl_usd': partial_pnl * 0.5 * risk_per_trade,
                        'r_multiple': partial_pnl * 0.5,
                        'result': 'PARTIAL_WIN',
                        'entry_type': signal.entry_type,
                        'zone_type': signal.zone.zone_type,
                        'confluence': signal.confluence,
                        'reasoning': f"Partial TP at {partial_tp_r}R",
                        'duration_days': partial_exit_bar - i if partial_exit_bar else 0,
                        'highest_r': partial_pnl
                    })
            else:
                if signal.direction == 'long':
                    pnl_pips = exit_price - entry
                else:
                    pnl_pips = entry - exit_price
                
                if risk_amount > 0:
                    r_multiple = pnl_pips / risk_amount
                    pnl_usd = r_multiple * risk_per_trade
                else:
                    r_multiple = 0
                    pnl_usd = 0
            
            if hit_tp:
                result = 'WIN'
            elif be_activated and abs(exit_price - entry) < risk_amount * 0.1:
                result = 'BE'
                pnl_usd = 0
                r_multiple = 0
            else:
                result = 'LOSS'
            
            trade_duration = exit_bar - i if exit_bar else 0
            
            trades.append({
                'entry_time': signal.timestamp,
                'exit_time': daily_candles[exit_bar]['time'] if exit_bar else None,
                'direction': signal.direction,
                'entry': entry,
                'exit': exit_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'pnl_usd': pnl_usd,
                'r_multiple': r_multiple,
                'result': result,
                'entry_type': signal.entry_type,
                'zone_type': signal.zone.zone_type,
                'confluence': signal.confluence,
                'reasoning': signal.reasoning,
                'duration_days': trade_duration,
                'highest_r': highest_r,
                'partial_taken': partial_taken
            })
            
            last_trade_bar = i
            daily_trade_count[current_date] += 1
    
    return trades


def calculate_backtest_stats(trades: List[Dict]) -> Dict:
    """Calculate comprehensive backtest statistics."""
    if not trades:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakevens': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'avg_r': 0.0,
            'avg_duration': 0.0
        }
    
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    bes = [t for t in trades if t['result'] == 'BE']
    
    total_pnl = sum(t['pnl_usd'] for t in trades)
    
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    avg_win = sum(t['pnl_usd'] for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t['pnl_usd'] for t in losses) / len(losses)) if losses else 0
    
    gross_profit = sum(t['pnl_usd'] for t in wins)
    gross_loss = abs(sum(t['pnl_usd'] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    equity = 10000
    peak = equity
    max_dd = 0
    for t in trades:
        equity += t['pnl_usd']
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)
    
    avg_r = sum(t['r_multiple'] for t in trades) / len(trades)
    avg_duration = sum(t['duration_days'] for t in trades) / len(trades)
    
    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'breakevens': len(bes),
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown': round(max_dd, 1),
        'avg_r': round(avg_r, 2),
        'avg_duration': round(avg_duration, 1)
    }
