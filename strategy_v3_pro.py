"""
Strategy V3 Pro: Daily Swing Trading with Golden Pocket + Wyckoff

Key Methods:
- Golden Pocket entries (0.618-0.65 Fibonacci retracement) at Supply/Demand zones
- Wyckoff Spring/Upthrust for reversal detection
- Daily timeframe structure (no H4 BOS)
- Trade duration: 2-8 days
- Structural Take Profits (swing highs/lows) - NO Fibonacci TPs

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


def calculate_golden_pocket(swing_high: float, swing_low: float, direction: str) -> Tuple[float, float]:
    """
    Calculate Golden Pocket zone (61.8% - 65% retracement).
    For bullish: measure from swing low to swing high
    For bearish: measure from swing high to swing low
    """
    range_size = swing_high - swing_low
    
    if direction == 'long':
        gp_high = swing_high - (range_size * 0.618)
        gp_low = swing_high - (range_size * 0.65)
    else:
        gp_low = swing_low + (range_size * 0.618)
        gp_high = swing_low + (range_size * 0.65)
    
    return min(gp_high, gp_low), max(gp_high, gp_low)


def identify_supply_demand_zones(candles: List[Dict], atr: float) -> List[SwingZone]:
    """
    Identify Supply and Demand zones based on price structure.
    Uses significant swing points and calculates Golden Pocket levels.
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
                    gp_low, gp_high = calculate_golden_pocket(high_price, low_price, 'long')
                    
                    zone_low = low_price
                    zone_high = low_price + (atr * 0.5)
                    
                    strength = 2 if range_size > atr * 4 else 1
                    
                    zones.append(SwingZone(
                        zone_type='demand',
                        high=zone_high,
                        low=zone_low,
                        swing_high=high_price,
                        swing_low=low_price,
                        golden_pocket_high=gp_high,
                        golden_pocket_low=gp_low,
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
                    gp_low, gp_high = calculate_golden_pocket(high_price, low_price, 'short')
                    
                    zone_low = high_price - (atr * 0.5)
                    zone_high = high_price
                    
                    strength = 2 if range_size > atr * 4 else 1
                    
                    zones.append(SwingZone(
                        zone_type='supply',
                        high=zone_high,
                        low=zone_low,
                        swing_high=high_price,
                        swing_low=low_price,
                        golden_pocket_high=gp_high,
                        golden_pocket_low=gp_low,
                        strength=strength,
                        created_bar=high_idx,
                        is_fresh=True
                    ))
                break
    
    return zones


def detect_wyckoff_spring(candles: List[Dict], zone: SwingZone, atr: float, lookback: int = 10) -> bool:
    """
    Detect Wyckoff Spring pattern at demand zone.
    Spring = Price breaks below support briefly (but not too deep), then reverses sharply back.
    
    Strict criteria:
    - Break below support by less than 1 ATR (not a breakdown)
    - Close back above support
    - Next bar must be bullish with close above spring high
    - Volume should ideally increase on reversal
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


def price_in_golden_pocket(price: float, zone: SwingZone, direction: str) -> bool:
    """Check if price is within the Golden Pocket zone."""
    return zone.golden_pocket_low <= price <= zone.golden_pocket_high


def price_at_zone(price: float, zone: SwingZone, atr: float) -> bool:
    """Check if price is at or near the zone."""
    buffer = atr * 0.3
    return (zone.low - buffer) <= price <= (zone.high + buffer)


def find_structural_tp(candles: List[Dict], entry: float, stop_loss: float, direction: str, min_rr: float = 2.5) -> Optional[float]:
    """
    Find structural take profit at swing high/low.
    NO Fibonacci for TPs - pure price structure.
    """
    risk = abs(entry - stop_loss)
    min_reward = risk * min_rr
    
    swing_highs, swing_lows = find_swing_points(candles, lookback=5)
    
    if direction == 'long':
        valid_tps = [h for _, h in swing_highs if h > entry + min_reward]
        if valid_tps:
            return min(valid_tps)
        return entry + min_reward
    else:
        valid_tps = [l for _, l in swing_lows if l < entry - min_reward]
        if valid_tps:
            return max(valid_tps)
        return entry - min_reward


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


def check_engulfing_candle(candles: List[Dict], direction: str) -> bool:
    """Check for engulfing candle pattern."""
    if len(candles) < 2:
        return False
    
    current = candles[-1]
    prev = candles[-2]
    
    if direction == 'long':
        return (current['close'] > current['open'] and
                prev['close'] < prev['open'] and
                current['open'] < prev['close'] and
                current['close'] > prev['open'])
    else:
        return (current['close'] < current['open'] and
                prev['close'] > prev['open'] and
                current['open'] > prev['close'] and
                current['close'] < prev['open'])


def check_rejection_candle(candles: List[Dict], zone: SwingZone, direction: str) -> bool:
    """Check for rejection candle (pin bar) at zone."""
    if len(candles) < 1:
        return False
    
    bar = candles[-1]
    body = abs(bar['close'] - bar['open'])
    total_range = bar['high'] - bar['low']
    
    if total_range == 0:
        return False
    
    body_ratio = body / total_range
    
    if body_ratio > 0.4:
        return False
    
    if direction == 'long':
        lower_wick = min(bar['open'], bar['close']) - bar['low']
        return lower_wick > total_range * 0.5 and bar['low'] <= zone.high
    else:
        upper_wick = bar['high'] - max(bar['open'], bar['close'])
        return upper_wick > total_range * 0.5 and bar['high'] >= zone.low


def check_momentum_aligned(candles: List[Dict], direction: str, lookback: int = 5) -> bool:
    """Check if recent momentum aligns with trade direction."""
    if len(candles) < lookback + 1:
        return False
    
    recent = candles[-(lookback+1):]
    
    bullish_bars = sum(1 for c in recent if c['close'] > c['open'])
    bearish_bars = sum(1 for c in recent if c['close'] < c['open'])
    
    if direction == 'long':
        return bullish_bars >= bearish_bars and recent[-1]['close'] > recent[-1]['open']
    else:
        return bearish_bars >= bullish_bars and recent[-1]['close'] < recent[-1]['open']


def check_strong_impulse(candles: List[Dict], zone: SwingZone, direction: str, atr: float) -> bool:
    """Check if there was a strong impulse move away from the zone (validates zone quality)."""
    if len(candles) < 10:
        return False
    
    if direction == 'long':
        impulse_size = zone.swing_high - zone.swing_low
        return impulse_size > atr * 3
    else:
        impulse_size = zone.swing_high - zone.swing_low
        return impulse_size > atr * 3


def generate_v3_pro_signal(
    symbol: str,
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_rr: float = 2.5,
    min_confluence: int = 4
) -> Optional[TradeSignal]:
    """
    Generate V3 Pro trading signal - STRICT CRITERIA VERSION.
    
    Entry Types:
    1. Golden Pocket: Entry when price retraces to 0.618-0.65 at fresh zone WITH confirmation
    2. Wyckoff Spring: Entry after spring pattern at demand zone
    3. Wyckoff Upthrust: Entry after upthrust pattern at supply zone
    4. Zone Rejection: Entry on rejection candle at zone
    
    STRICT Requirements:
    - Weekly zones preferred (daily only with high confluence)
    - Daily trend MUST be aligned
    - Momentum aligned (recent bars confirm direction)
    - Fresh zone (first retest)
    - Strong impulse away from zone originally
    - Confluence score >= 4
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
    
    if daily_trend == 'neutral':
        return None
    
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
        if not zone.is_fresh:
            continue
        
        if zone.zone_type == 'demand':
            direction = 'long'
            if daily_trend == 'bearish':
                continue
        else:
            direction = 'short'
            if daily_trend == 'bullish':
                continue
        
        confluence = 0
        reasoning_parts = []
        entry_type = None
        
        at_zone = price_at_zone(current_price, zone, daily_atr)
        in_gp = price_in_golden_pocket(current_price, zone, direction)
        
        if not at_zone and not in_gp:
            continue
        
        if in_gp:
            confluence += 2
            reasoning_parts.append("Price in Golden Pocket (0.618-0.65)")
            entry_type = 'golden_pocket'
        elif at_zone:
            confluence += 1
            reasoning_parts.append(f"Price at {zone.zone_type} zone")
        
        if direction == 'long' and detect_wyckoff_spring(daily_candles, zone, daily_atr):
            confluence += 2
            reasoning_parts.append("Wyckoff Spring detected")
            entry_type = 'wyckoff_spring'
        elif direction == 'short' and detect_wyckoff_upthrust(daily_candles, zone, daily_atr):
            confluence += 2
            reasoning_parts.append("Wyckoff Upthrust detected")
            entry_type = 'wyckoff_upthrust'
        
        if check_rejection_candle(daily_candles, zone, direction):
            confluence += 1
            reasoning_parts.append("Rejection candle at zone")
            if entry_type is None:
                entry_type = 'rejection'
        
        if check_engulfing_candle(daily_candles, direction):
            confluence += 1
            reasoning_parts.append("Engulfing pattern")
            if entry_type is None:
                entry_type = 'engulfing'
        
        if (direction == 'long' and daily_trend == 'bullish') or \
           (direction == 'short' and daily_trend == 'bearish'):
            confluence += 1
            reasoning_parts.append(f"Daily trend aligned: {daily_trend}")
        
        if timeframe == 'weekly':
            confluence += 1
            reasoning_parts.append("Weekly timeframe zone")
        
        if zone.strength >= 2:
            confluence += 1
            reasoning_parts.append("Strong zone (large impulse)")
        
        if check_momentum_aligned(daily_candles, direction, 3):
            confluence += 1
            reasoning_parts.append("Momentum aligned")
        
        if check_strong_impulse(daily_candles, zone, direction, daily_atr):
            confluence += 1
            reasoning_parts.append("Strong impulse from zone")
        
        if confluence < min_confluence:
            continue
        
        has_confirmation = (
            entry_type in ['wyckoff_spring', 'wyckoff_upthrust'] or
            check_rejection_candle(daily_candles, zone, direction) or
            check_engulfing_candle(daily_candles, direction)
        )
        
        if not has_confirmation and entry_type == 'golden_pocket':
            continue
        
        if entry_type is None:
            entry_type = 'zone_entry'
        
        entry = current_price
        
        min_sl_distance = daily_atr * 0.75
        
        if direction == 'long':
            raw_sl = zone.low - (daily_atr * 0.3)
            stop_loss = min(raw_sl, entry - min_sl_distance)
        else:
            raw_sl = zone.high + (daily_atr * 0.3)
            stop_loss = max(raw_sl, entry + min_sl_distance)
        
        take_profit = find_structural_tp(daily_candles, entry, stop_loss, direction, min_rr)
        
        if take_profit is None:
            continue
        
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


def backtest_v3_pro(
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_rr: float = 2.5,
    min_confluence: int = 3,
    risk_per_trade: float = 250.0,
    cooldown_days: int = 2,
    max_daily_trades: int = 1,
    day_stagger: bool = True
) -> List[Dict]:
    """
    Backtest V3 Pro strategy on daily timeframe.
    Trades held for 2-8 days typically.
    
    day_stagger: If True, limits to 1 trade per day to spread wins across calendar days
                 (helps pass 5%ers "3 profitable days" requirement)
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
        current_date = current_bar['time'][:10]
        
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
        
        for j in range(i + 1, min(i + 40, len(daily_candles))):
            bar = daily_candles[j]
            
            risk = abs(entry - sl)
            
            if signal.direction == 'long':
                current_r = (bar['high'] - entry) / risk if risk > 0 else 0
                highest_r = max(highest_r, current_r)
                
                be_level = entry + risk
                
                if bar['low'] <= sl:
                    hit_sl = True
                    exit_price = sl if not be_activated else entry
                    exit_bar = j
                    break
                
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
            if signal.direction == 'long':
                pnl_pips = exit_price - entry
            else:
                pnl_pips = entry - exit_price
            
            risk_amount = abs(entry - signal.stop_loss)
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
                'highest_r': highest_r
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
