"""
Strategy V4 - Archer Academy Style (Forex Dictionary)

Based on Supply/Demand zones with Base identification.
NO RSI, NO MACD, NO SMC, NO Fibonacci for TPs.

Key Patterns:
- Rally-Base-Drop (RBD) = Supply Zone (bearish reversal) - STRONG
- Drop-Base-Rally (DBR) = Demand Zone (bullish reversal) - STRONG
- Rally-Base-Rally (RBR) = Demand Zone (bullish continuation) - MODERATE
- Drop-Base-Drop (DBD) = Supply Zone (bearish continuation) - MODERATE

Entry: Fresh zone retest + BOS confirmation
Exit: Opposite zone or structural swing level
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import math


@dataclass
class SupplyDemandZone:
    """Represents a Supply or Demand zone."""
    zone_type: str  # 'supply' or 'demand'
    pattern: str    # 'RBD', 'DBR', 'RBR', 'DBD'
    high: float
    low: float
    strength: str   # 'strong' (reversal) or 'moderate' (continuation)
    creation_time: str
    creation_bar: int
    is_fresh: bool  # Has not been retested
    impulse_strength: float  # How strong was the departure move


@dataclass
class TradeSignal:
    """Represents a trading signal."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry: float
    stop_loss: float
    take_profit: float
    zone: SupplyDemandZone
    status: str  # 'ACTIVE', 'WATCHING'
    confluence: int
    reasoning: str
    rr_ratio: float
    timestamp: str


def calculate_atr(candles: List[Dict], period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(candles) < period + 1:
        return 0.0
    
    tr_values = []
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i-1]['close']
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return sum(tr_values) / len(tr_values) if tr_values else 0.0
    
    return sum(tr_values[-period:]) / period


def is_base_candle(candle: Dict, avg_range: float) -> bool:
    """Check if a candle is a consolidation/base candle (small body, tight range)."""
    body = abs(candle['close'] - candle['open'])
    full_range = candle['high'] - candle['low']
    
    if avg_range == 0:
        return False
    
    is_small_body = body < avg_range * 0.7
    is_tight_range = full_range < avg_range * 1.0
    
    return is_small_body and is_tight_range


def is_impulse_candle(candle: Dict, avg_range: float, direction: str) -> bool:
    """Check if a candle is an impulse candle (large body in direction)."""
    body = abs(candle['close'] - candle['open'])
    
    if avg_range == 0:
        return False
    
    is_large = body > avg_range * 0.5
    
    if direction == 'up':
        is_bullish = candle['close'] > candle['open']
        return is_large and is_bullish
    else:
        is_bearish = candle['close'] < candle['open']
        return is_large and is_bearish


def find_base(candles: List[Dict], start_idx: int, direction: str, avg_range: float) -> Optional[Tuple[int, int]]:
    """
    Find a consolidation base (2-6 candles) before an impulse move.
    Returns (base_start_idx, base_end_idx) or None.
    """
    base_candles = []
    
    for i in range(start_idx, max(start_idx - 8, 0), -1):
        if is_base_candle(candles[i], avg_range):
            base_candles.insert(0, i)
        else:
            break
    
    if 2 <= len(base_candles) <= 6:
        return (base_candles[0], base_candles[-1])
    
    return None


def calculate_impulse_strength(candles: List[Dict], start_idx: int, end_idx: int) -> float:
    """Calculate the strength of an impulse move (number of strong candles * avg size)."""
    if start_idx >= end_idx or end_idx >= len(candles):
        return 0.0
    
    impulse_candles = candles[start_idx:end_idx + 1]
    if not impulse_candles:
        return 0.0
    
    total_move = abs(impulse_candles[-1]['close'] - impulse_candles[0]['open'])
    num_candles = len(impulse_candles)
    
    if num_candles == 0:
        return 0.0
    
    return total_move / num_candles


def identify_supply_demand_zones(
    candles: List[Dict],
    lookback: int = 100,
    min_impulse_candles: int = 3
) -> List[SupplyDemandZone]:
    """
    Identify Supply and Demand zones using Archer Academy patterns.
    
    Patterns:
    - RBD (Rally-Base-Drop): Supply zone, strong reversal
    - DBR (Drop-Base-Rally): Demand zone, strong reversal
    - RBR (Rally-Base-Rally): Demand zone, moderate continuation
    - DBD (Drop-Base-Drop): Supply zone, moderate continuation
    """
    zones = []
    
    if len(candles) < lookback:
        return zones
    
    avg_range = calculate_atr(candles, 14)
    if avg_range == 0:
        return zones
    
    start_idx = max(0, len(candles) - lookback)
    
    for i in range(start_idx + min_impulse_candles, len(candles) - min_impulse_candles):
        rally_before = 0
        drop_before = 0
        for j in range(max(0, i - 5), i):
            if is_impulse_candle(candles[j], avg_range, 'up'):
                rally_before += 1
            elif is_impulse_candle(candles[j], avg_range, 'down'):
                drop_before += 1
        
        base = find_base(candles, i, 'neutral', avg_range)
        
        if base is None:
            continue
        
        base_start, base_end = base
        
        rally_after = 0
        drop_after = 0
        impulse_end = min(base_end + min_impulse_candles + 2, len(candles) - 1)
        
        for j in range(base_end + 1, impulse_end + 1):
            if j < len(candles):
                if is_impulse_candle(candles[j], avg_range, 'up'):
                    rally_after += 1
                elif is_impulse_candle(candles[j], avg_range, 'down'):
                    drop_after += 1
        
        base_candles = candles[base_start:base_end + 1]
        zone_high = max(c['high'] for c in base_candles)
        zone_low = min(c['low'] for c in base_candles)
        
        impulse_strength = calculate_impulse_strength(candles, base_end + 1, impulse_end)
        
        pattern: Optional[str] = None
        zone_type: str = ''
        strength: str = ''
        
        if rally_before >= 2 and drop_after >= min_impulse_candles:
            pattern = 'RBD'
            zone_type = 'supply'
            strength = 'strong'
        elif drop_before >= 2 and rally_after >= min_impulse_candles:
            pattern = 'DBR'
            zone_type = 'demand'
            strength = 'strong'
        elif rally_before >= 2 and rally_after >= min_impulse_candles:
            pattern = 'RBR'
            zone_type = 'demand'
            strength = 'moderate'
        elif drop_before >= 2 and drop_after >= min_impulse_candles:
            pattern = 'DBD'
            zone_type = 'supply'
            strength = 'moderate'
        
        if pattern and zone_type and strength:
            zone = SupplyDemandZone(
                zone_type=zone_type,
                pattern=pattern,
                high=zone_high,
                low=zone_low,
                strength=strength,
                creation_time=candles[base_start]['time'],
                creation_bar=base_start,
                is_fresh=True,
                impulse_strength=impulse_strength
            )
            zones.append(zone)
    
    return zones


def check_zone_retest(
    current_price: float,
    zone: SupplyDemandZone,
    atr: float,
    buffer_mult: float = 0.3
) -> bool:
    """Check if price is retesting a zone."""
    buffer = atr * buffer_mult
    
    if zone.zone_type == 'demand':
        return (zone.low - buffer) <= current_price <= (zone.high + buffer)
    else:
        return (zone.low - buffer) <= current_price <= (zone.high + buffer)


def detect_bos_at_zone(
    candles: List[Dict],
    zone: SupplyDemandZone,
    lookback: int = 10
) -> Optional[str]:
    """
    Detect Break of Structure at a zone retest.
    Returns 'confirmed', 'soft', or None.
    """
    if len(candles) < lookback + 2:
        return None
    
    recent = candles[-lookback:]
    current = candles[-1]
    prev = candles[-2]
    
    if zone.zone_type == 'demand':
        swing_highs = []
        for i in range(2, len(recent) - 2):
            if recent[i]['high'] > recent[i-1]['high'] and recent[i]['high'] > recent[i-2]['high'] and \
               recent[i]['high'] > recent[i+1]['high'] and recent[i]['high'] > recent[i+2]['high']:
                swing_highs.append(recent[i]['high'])
        
        if swing_highs:
            last_swing = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
            
            if current['close'] > last_swing:
                return 'confirmed'
            elif current['high'] > last_swing:
                return 'soft'
    else:
        swing_lows = []
        for i in range(2, len(recent) - 2):
            if recent[i]['low'] < recent[i-1]['low'] and recent[i]['low'] < recent[i-2]['low'] and \
               recent[i]['low'] < recent[i+1]['low'] and recent[i]['low'] < recent[i+2]['low']:
                swing_lows.append(recent[i]['low'])
        
        if swing_lows:
            last_swing = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
            
            if current['close'] < last_swing:
                return 'confirmed'
            elif current['low'] < last_swing:
                return 'soft'
    
    return None


def find_engulfing_confirmation(candles: List[Dict], direction: str) -> bool:
    """Check for engulfing candle pattern as zone confirmation."""
    if len(candles) < 2:
        return False
    
    current = candles[-1]
    prev = candles[-2]
    
    curr_body = abs(current['close'] - current['open'])
    prev_body = abs(prev['close'] - prev['open'])
    
    if direction == 'long':
        is_bullish = current['close'] > current['open']
        engulfs = current['open'] <= prev['close'] and current['close'] >= prev['open']
        is_significant = curr_body > prev_body * 1.2
        return is_bullish and engulfs and is_significant
    else:
        is_bearish = current['close'] < current['open']
        engulfs = current['open'] >= prev['close'] and current['close'] <= prev['open']
        is_significant = curr_body > prev_body * 1.2
        return is_bearish and engulfs and is_significant


def find_rejection_candle(candles: List[Dict], zone: SupplyDemandZone, direction: str) -> bool:
    """Check for rejection candle (pin bar / hammer) at zone."""
    if len(candles) < 1:
        return False
    
    current = candles[-1]
    body = abs(current['close'] - current['open'])
    full_range = current['high'] - current['low']
    
    if full_range == 0:
        return False
    
    if direction == 'long':
        lower_wick = min(current['open'], current['close']) - current['low']
        wick_ratio = lower_wick / full_range
        body_ratio = body / full_range
        
        touches_zone = current['low'] <= zone.high
        
        return wick_ratio > 0.5 and body_ratio < 0.35 and touches_zone
    else:
        upper_wick = current['high'] - max(current['open'], current['close'])
        wick_ratio = upper_wick / full_range
        body_ratio = body / full_range
        
        touches_zone = current['high'] >= zone.low
        
        return wick_ratio > 0.5 and body_ratio < 0.35 and touches_zone


def find_structural_target(
    candles: List[Dict],
    entry: float,
    stop_loss: float,
    direction: str,
    min_rr: float = 2.0,
    lookback: int = 100
) -> Optional[float]:
    """Find structural take profit (swing high/low)."""
    risk = abs(entry - stop_loss)
    min_reward = risk * min_rr
    
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    
    if direction == 'long':
        swing_highs = []
        for i in range(2, len(recent) - 2):
            if recent[i]['high'] > recent[i-1]['high'] and recent[i]['high'] > recent[i+1]['high']:
                if recent[i]['high'] > recent[i-2]['high'] or recent[i]['high'] > recent[i+2]['high']:
                    swing_highs.append(recent[i]['high'])
        
        all_highs = sorted(set(swing_highs), reverse=True)[:10]
        valid_targets = [h for h in all_highs if h > entry + min_reward]
        if valid_targets:
            return min(valid_targets)
        return entry + min_reward
    else:
        swing_lows = []
        for i in range(2, len(recent) - 2):
            if recent[i]['low'] < recent[i-1]['low'] and recent[i]['low'] < recent[i+1]['low']:
                if recent[i]['low'] < recent[i-2]['low'] or recent[i]['low'] < recent[i+2]['low']:
                    swing_lows.append(recent[i]['low'])
        
        all_lows = sorted(set(swing_lows))[:10]
        valid_targets = [l for l in all_lows if l < entry - min_reward]
        if valid_targets:
            return max(valid_targets)
        return entry - min_reward


def generate_archer_signal(
    symbol: str,
    h4_candles: List[Dict],
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_rr: float = 2.0,
    min_confluence: int = 2,
    prefer_fresh: bool = True,
    prefer_reversal: bool = True
) -> Optional[TradeSignal]:
    """
    Generate a trading signal using Archer Academy methodology.
    
    Entry conditions:
    1. Price at fresh Supply/Demand zone (+2 if fresh, +1 if tested once)
    2. Zone is reversal pattern (RBD/DBR) (+2) or continuation (RBR/DBD) (+1)
    3. BOS confirmation (+2 confirmed, +1 soft)
    4. Engulfing or rejection candle (+1)
    5. Higher timeframe zone alignment (+1)
    
    Min confluence for ACTIVE: 4
    Min confluence for WATCHING: 2
    """
    if len(h4_candles) < 100 or len(daily_candles) < 50:
        return None
    
    current_bar = h4_candles[-1]
    current_price = current_bar['close']
    current_time = current_bar['time']
    
    h4_atr = calculate_atr(h4_candles, 14)
    daily_atr = calculate_atr(daily_candles, 14)
    
    if h4_atr == 0 or daily_atr == 0:
        return None
    
    h4_zones = identify_supply_demand_zones(h4_candles, lookback=100)
    daily_zones = identify_supply_demand_zones(daily_candles, lookback=50)
    
    all_zones = []
    
    for zone in daily_zones:
        zone.is_fresh = True
        all_zones.append(('daily', zone))
    
    for zone in h4_zones:
        all_zones.append(('h4', zone))
    
    if prefer_reversal:
        all_zones.sort(key=lambda x: (
            0 if x[1].strength == 'strong' else 1,
            0 if x[1].is_fresh else 1,
            -x[1].impulse_strength
        ))
    
    for timeframe, zone in all_zones:
        if not check_zone_retest(current_price, zone, h4_atr):
            continue
        
        direction = 'long' if zone.zone_type == 'demand' else 'short'
        
        confluence = 0
        reasoning_parts = []
        
        if zone.is_fresh:
            confluence += 2
            reasoning_parts.append(f"Fresh {zone.pattern} zone")
        else:
            confluence += 1
            reasoning_parts.append(f"Tested {zone.pattern} zone")
        
        if zone.strength == 'strong':
            confluence += 2
            reasoning_parts.append(f"Strong reversal pattern")
        else:
            confluence += 1
            reasoning_parts.append(f"Continuation pattern")
        
        bos = detect_bos_at_zone(h4_candles, zone)
        if bos == 'confirmed':
            confluence += 2
            reasoning_parts.append("Confirmed BOS")
        elif bos == 'soft':
            confluence += 1
            reasoning_parts.append("Soft BOS")
        
        has_engulfing = find_engulfing_confirmation(h4_candles, direction)
        has_rejection = find_rejection_candle(h4_candles, zone, direction)
        
        if has_engulfing:
            confluence += 1
            reasoning_parts.append("Engulfing candle")
        elif has_rejection:
            confluence += 1
            reasoning_parts.append("Rejection candle")
        
        if timeframe == 'daily':
            confluence += 1
            reasoning_parts.append("Daily TF zone")
        
        htf_aligned = False
        for _, d_zone in [(tf, z) for tf, z in all_zones if tf == 'daily']:
            if d_zone.zone_type == zone.zone_type:
                if (d_zone.low - daily_atr) <= zone.low <= (d_zone.high + daily_atr):
                    htf_aligned = True
                    break
        
        if htf_aligned and timeframe == 'h4':
            confluence += 1
            reasoning_parts.append("Aligned with Daily zone")
        
        if confluence < min_confluence:
            continue
        
        entry = current_price
        
        if direction == 'long':
            stop_loss = zone.low - (h4_atr * 0.25)
        else:
            stop_loss = zone.high + (h4_atr * 0.25)
        
        take_profit = find_structural_target(
            h4_candles, entry, stop_loss, direction, min_rr
        )
        
        if take_profit is None:
            continue
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < min_rr:
            continue
        
        if confluence >= 4 and bos == 'confirmed':
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
            zone=zone,
            status=status,
            confluence=confluence,
            reasoning=reasoning,
            rr_ratio=rr_ratio,
            timestamp=current_time
        )
    
    return None


def backtest_archer_strategy(
    h4_candles: List[Dict],
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_rr: float = 2.0,
    min_confluence: int = 2,
    risk_per_trade: float = 250.0,
    cooldown_bars: int = 4,
    max_daily_trades: int = 2
) -> List[Dict]:
    """
    Backtest the Archer strategy with proper trade management.
    """
    trades = []
    last_trade_bar = -cooldown_bars
    daily_trade_count = {}
    zone_retests = {}
    pending_signals = []
    
    for i in range(100, len(h4_candles)):
        h4_slice = h4_candles[:i+1]
        
        daily_end = 0
        for j, d in enumerate(daily_candles):
            if d['time'] <= h4_candles[i]['time']:
                daily_end = j + 1
        daily_slice = daily_candles[:daily_end]
        
        if len(daily_slice) < 30:
            continue
        
        current_bar = h4_candles[i]
        current_date = current_bar['time'][:10]
        
        if current_date not in daily_trade_count:
            daily_trade_count[current_date] = 0
        
        if i - last_trade_bar < cooldown_bars:
            continue
        
        if daily_trade_count[current_date] >= max_daily_trades:
            continue
        
        new_pending = []
        for sig, bars_waiting in pending_signals:
            if bars_waiting >= 3:
                continue
            
            zone_key = f"{sig.zone.zone_type}_{sig.zone.high}_{sig.zone.low}"
            if zone_key in zone_retests:
                continue
            
            bos = detect_bos_at_zone(h4_slice, sig.zone)
            if bos == 'confirmed':
                entry = current_bar['close']
                h4_atr = calculate_atr(h4_slice, 14)
                
                if sig.direction == 'long':
                    sl = sig.zone.low - (h4_atr * 0.25)
                else:
                    sl = sig.zone.high + (h4_atr * 0.25)
                
                tp = find_structural_target(h4_slice, entry, sl, sig.direction, min_rr)
                
                if tp is None:
                    continue
                
                risk = abs(entry - sl)
                reward = abs(tp - entry)
                rr_ratio = reward / risk if risk > 0 else 0
                
                if rr_ratio < min_rr:
                    continue
                
                zone_retests[zone_key] = True
                
                signal = TradeSignal(
                    symbol=sig.symbol,
                    direction=sig.direction,
                    entry=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    zone=sig.zone,
                    status='ACTIVE',
                    confluence=sig.confluence + 1,
                    reasoning=sig.reasoning + " | BOS confirmed",
                    rr_ratio=rr_ratio,
                    timestamp=current_bar['time']
                )
                
                hit_tp = False
                hit_sl = False
                exit_price = None
                exit_bar = None
                be_activated = False
                
                for j in range(i + 1, len(h4_candles)):
                    bar = h4_candles[j]
                    
                    if signal.direction == 'long':
                        risk = entry - sl
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
                        risk = sl - entry
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
                    
                    trades.append({
                        'entry_time': signal.timestamp,
                        'exit_time': h4_candles[exit_bar]['time'] if exit_bar else None,
                        'direction': signal.direction,
                        'entry': entry,
                        'exit': exit_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'pnl_usd': pnl_usd,
                        'r_multiple': r_multiple,
                        'result': 'WIN' if hit_tp else ('BE' if be_activated and exit_price == entry else 'LOSS'),
                        'zone_pattern': signal.zone.pattern,
                        'zone_strength': signal.zone.strength,
                        'confluence': signal.confluence,
                        'reasoning': signal.reasoning
                    })
                    
                    last_trade_bar = i
                    daily_trade_count[current_date] += 1
                
                continue
            
            new_pending.append((sig, bars_waiting + 1))
        
        pending_signals = new_pending
        
        signal = generate_archer_signal(
            symbol="BACKTEST",
            h4_candles=h4_slice,
            daily_candles=daily_slice,
            weekly_candles=weekly_candles,
            min_rr=min_rr,
            min_confluence=min_confluence
        )
        
        if signal is None:
            continue
        
        zone_key = f"{signal.zone.zone_type}_{signal.zone.high}_{signal.zone.low}"
        if zone_key in zone_retests:
            continue
        
        if signal.status == 'WATCHING':
            pending_signals.append((signal, 0))
            continue
        
        zone_retests[zone_key] = True
        entry = signal.entry
        sl = signal.stop_loss
        tp = signal.take_profit
        
        hit_tp = False
        hit_sl = False
        exit_price = None
        exit_bar = None
        be_activated = False
        
        for j in range(i + 1, len(h4_candles)):
            bar = h4_candles[j]
            
            if signal.direction == 'long':
                risk = entry - sl
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
                risk = sl - entry
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
            
            trades.append({
                'entry_time': signal.timestamp,
                'exit_time': h4_candles[exit_bar]['time'] if exit_bar else None,
                'direction': signal.direction,
                'entry': entry,
                'exit': exit_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'pnl_usd': pnl_usd,
                'r_multiple': r_multiple,
                'result': 'WIN' if hit_tp else ('BE' if be_activated and exit_price == entry else 'LOSS'),
                'zone_pattern': signal.zone.pattern,
                'zone_strength': signal.zone.strength,
                'confluence': signal.confluence,
                'reasoning': signal.reasoning
            })
            
            last_trade_bar = i
            daily_trade_count[current_date] += 1
    
    return trades


def calculate_backtest_stats(trades: List[Dict]) -> Dict:
    """Calculate backtest statistics."""
    if not trades:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_winner': 0,
            'avg_loser': 0,
            'profit_factor': 0,
            'max_drawdown': 0
        }
    
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    breakeven = [t for t in trades if t['result'] == 'BE']
    
    total_pnl = sum(t['pnl_usd'] for t in trades)
    
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    avg_winner = sum(t['pnl_usd'] for t in wins) / len(wins) if wins else 0
    avg_loser = sum(t['pnl_usd'] for t in losses) / len(losses) if losses else 0
    
    gross_profit = sum(t['pnl_usd'] for t in trades if t['pnl_usd'] > 0)
    gross_loss = abs(sum(t['pnl_usd'] for t in trades if t['pnl_usd'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    equity = 10000
    peak = equity
    max_dd = 0
    
    for t in trades:
        equity += t['pnl_usd']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'breakeven': len(breakeven),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd
    }
