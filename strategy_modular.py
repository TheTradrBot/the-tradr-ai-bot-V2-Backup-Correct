"""
Modular Trading Strategy Pipeline

Two independent modules:
1. HTF Reversal (Mean-Reversion) - Trade reversals at Monthly/Weekly S/R levels
2. Daily Trend (Continuation) - Trade trend continuation using daily patterns

Each module can be tuned and backtested independently, then ensembled.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date


@dataclass
class ModularSignal:
    """Signal from modular strategy."""
    symbol: str
    direction: str  # "bullish" or "bearish"
    bar_index: int
    timestamp: Optional[str] = None
    entry: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    confluence_score: int = 0
    module: str = ""  # "reversal" or "trend"
    notes: Dict = field(default_factory=dict)


@dataclass 
class ReversalParams:
    """Parameters for HTF Reversal module."""
    sr_tolerance_pct: float = 0.5  # % tolerance around S/R levels
    min_touches: int = 2  # Minimum S/R level touches
    lookback_bars: int = 100  # Bars to look back for S/R
    confirmation_bars: int = 1  # Bars needed for confirmation
    sl_buffer_atr: float = 0.5  # SL buffer as ATR multiple
    tp1_rr: float = 1.0
    tp2_rr: float = 1.5
    tp3_rr: float = 2.0
    cooldown_bars: int = 1


@dataclass
class TrendParams:
    """Parameters for Daily Trend module."""
    ema_fast: int = 8
    ema_slow: int = 21
    atr_period: int = 14
    trend_lookback: int = 20
    sl_buffer_atr: float = 1.0
    tp1_rr: float = 1.0
    tp2_rr: float = 1.5
    tp3_rr: float = 2.5
    cooldown_bars: int = 1


def _calculate_atr(candles: List[Dict], period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(candles) < period + 1:
        return 0.0
    
    trs = []
    for i in range(1, min(len(candles), period + 1)):
        high = candles[-i].get("high", 0)
        low = candles[-i].get("low", 0)
        prev_close = candles[-i-1].get("close", 0) if i < len(candles) else 0
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)
    
    return sum(trs) / len(trs) if trs else 0.0


def _calculate_ema(values: List[float], period: int) -> List[float]:
    """Calculate EMA series."""
    if len(values) < period:
        return []
    
    k = 2 / (period + 1)
    ema = [sum(values[:period]) / period]
    
    for price in values[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    
    return ema


def _find_swing_levels(candles: List[Dict], lookback: int = 3) -> Tuple[List[float], List[float]]:
    """Find swing high/low levels for S/R detection."""
    if len(candles) < lookback * 2 + 1:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i].get("high", 0)
        low = candles[i].get("low", 0)
        
        is_swing_high = all(
            high >= candles[i-j].get("high", 0) and high >= candles[i+j].get("high", 0)
            for j in range(1, lookback + 1)
        )
        
        is_swing_low = all(
            low <= candles[i-j].get("low", 0) and low <= candles[i+j].get("low", 0)
            for j in range(1, lookback + 1)
        )
        
        if is_swing_high:
            swing_highs.append(high)
        if is_swing_low:
            swing_lows.append(low)
    
    return swing_highs, swing_lows


def _cluster_levels(levels: List[float], tolerance_pct: float = 1.0) -> List[Tuple[float, int]]:
    """Cluster nearby price levels and count touches."""
    if not levels:
        return []
    
    sorted_levels = sorted(levels)
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        if current_cluster and abs(level - current_cluster[-1]) / current_cluster[-1] * 100 < tolerance_pct:
            current_cluster.append(level)
        else:
            if current_cluster:
                avg_level = sum(current_cluster) / len(current_cluster)
                clusters.append((avg_level, len(current_cluster)))
            current_cluster = [level]
    
    if current_cluster:
        avg_level = sum(current_cluster) / len(current_cluster)
        clusters.append((avg_level, len(current_cluster)))
    
    return sorted(clusters, key=lambda x: x[1], reverse=True)


def _determine_trend(candles: List[Dict], lookback: int = 20) -> str:
    """Determine trend direction based on recent price action."""
    if len(candles) < lookback:
        return "neutral"
    
    recent = candles[-lookback:]
    closes = [c.get("close", 0) for c in recent]
    
    if not closes:
        return "neutral"
    
    highs = [c.get("high", 0) for c in recent]
    lows = [c.get("low", 0) for c in recent]
    
    higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
    lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
    lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
    
    bullish_score = higher_highs + higher_lows
    bearish_score = lower_highs + lower_lows
    
    if bullish_score > bearish_score * 1.3:
        return "bullish"
    elif bearish_score > bullish_score * 1.3:
        return "bearish"
    
    return "neutral"


def generate_reversal_signals(
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    monthly_candles: List[Dict],
    symbol: str,
    params: ReversalParams = None,
) -> List[ModularSignal]:
    """
    Generate reversal signals at HTF S/R levels.
    
    Strategy:
    1. Find Monthly/Weekly swing S/R levels
    2. Wait for price to approach these levels
    3. Look for reversal confirmation (rejection candle)
    4. Enter with SL beyond level, TPs at R:R targets
    """
    if params is None:
        params = ReversalParams()
    
    signals = []
    
    if not daily_candles or len(daily_candles) < 50:
        return signals
    
    monthly_swing_highs, monthly_swing_lows = _find_swing_levels(monthly_candles or [], lookback=2)
    weekly_swing_highs, weekly_swing_lows = _find_swing_levels(weekly_candles or [], lookback=2)
    
    all_resistance = monthly_swing_highs + weekly_swing_highs
    all_support = monthly_swing_lows + weekly_swing_lows
    
    resistance_clusters = _cluster_levels(all_resistance, params.sr_tolerance_pct)
    support_clusters = _cluster_levels(all_support, params.sr_tolerance_pct)
    
    resistance_levels = [level for level, touches in resistance_clusters if touches >= params.min_touches][:10]
    support_levels = [level for level, touches in support_clusters if touches >= params.min_touches][:10]
    
    cooldown_until = 0
    
    for i in range(params.lookback_bars, len(daily_candles)):
        if i <= cooldown_until:
            continue
        
        current = daily_candles[i]
        current_high = current.get("high", 0)
        current_low = current.get("low", 0)
        current_close = current.get("close", 0)
        current_open = current.get("open", 0)
        
        recent_slice = daily_candles[max(0, i-20):i+1]
        atr = _calculate_atr(recent_slice, 14)
        
        if atr <= 0:
            continue
        
        tolerance = current_close * params.sr_tolerance_pct / 100
        
        for res_level in resistance_levels:
            if current_high >= res_level - tolerance and current_high <= res_level + tolerance:
                if current_close < current_open:
                    body = abs(current_close - current_open)
                    upper_wick = current_high - max(current_open, current_close)
                    
                    if upper_wick > body * 0.3:
                        entry = current_close
                        sl = res_level + atr * params.sl_buffer_atr
                        risk = sl - entry
                        
                        if risk > 0:
                            tp1 = entry - risk * params.tp1_rr
                            tp2 = entry - risk * params.tp2_rr
                            tp3 = entry - risk * params.tp3_rr
                            
                            timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                            
                            signal = ModularSignal(
                                symbol=symbol,
                                direction="bearish",
                                bar_index=i,
                                timestamp=str(timestamp) if timestamp else None,
                                entry=entry,
                                stop_loss=sl,
                                tp1=tp1,
                                tp2=tp2,
                                tp3=tp3,
                                confluence_score=5,
                                module="reversal",
                                notes={
                                    "level": res_level,
                                    "level_type": "resistance",
                                    "rejection": "upper_wick",
                                }
                            )
                            signals.append(signal)
                            cooldown_until = i + params.cooldown_bars
                            break
        
        if i <= cooldown_until:
            continue
        
        for sup_level in support_levels:
            if current_low >= sup_level - tolerance and current_low <= sup_level + tolerance:
                if current_close > current_open:
                    body = abs(current_close - current_open)
                    lower_wick = min(current_open, current_close) - current_low
                    
                    if lower_wick > body * 0.3:
                        entry = current_close
                        sl = sup_level - atr * params.sl_buffer_atr
                        risk = entry - sl
                        
                        if risk > 0:
                            tp1 = entry + risk * params.tp1_rr
                            tp2 = entry + risk * params.tp2_rr
                            tp3 = entry + risk * params.tp3_rr
                            
                            timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                            
                            signal = ModularSignal(
                                symbol=symbol,
                                direction="bullish",
                                bar_index=i,
                                timestamp=str(timestamp) if timestamp else None,
                                entry=entry,
                                stop_loss=sl,
                                tp1=tp1,
                                tp2=tp2,
                                tp3=tp3,
                                confluence_score=5,
                                module="reversal",
                                notes={
                                    "level": sup_level,
                                    "level_type": "support",
                                    "rejection": "lower_wick",
                                }
                            )
                            signals.append(signal)
                            cooldown_until = i + params.cooldown_bars
                            break
    
    return signals


def generate_trend_signals(
    daily_candles: List[Dict],
    symbol: str,
    params: TrendParams = None,
) -> List[ModularSignal]:
    """
    Generate trend continuation signals.
    
    Strategy:
    1. Determine trend using EMA and swing structure
    2. Wait for pullback to EMA/key level
    3. Enter on resumption candle
    4. SL below recent swing, TPs at R:R targets
    """
    if params is None:
        params = TrendParams()
    
    signals = []
    
    if not daily_candles or len(daily_candles) < params.ema_slow + 20:
        return signals
    
    closes = [c.get("close", 0) for c in daily_candles]
    ema_fast = _calculate_ema(closes, params.ema_fast)
    ema_slow = _calculate_ema(closes, params.ema_slow)
    
    cooldown_until = 0
    
    for i in range(params.ema_slow + 10, len(daily_candles)):
        if i <= cooldown_until:
            continue
        
        ema_idx = i - params.ema_slow
        if ema_idx < 0 or ema_idx >= len(ema_fast) or ema_idx >= len(ema_slow):
            continue
        
        ema_f = ema_fast[ema_idx]
        ema_s = ema_slow[ema_idx]
        
        current = daily_candles[i]
        prev = daily_candles[i-1]
        current_close = current.get("close", 0)
        current_open = current.get("open", 0)
        current_low = current.get("low", 0)
        current_high = current.get("high", 0)
        prev_low = prev.get("low", 0)
        prev_high = prev.get("high", 0)
        
        recent_slice = daily_candles[max(0, i-20):i+1]
        atr = _calculate_atr(recent_slice, 14)
        
        if atr <= 0:
            continue
        
        trend = _determine_trend(daily_candles[:i], params.trend_lookback)
        
        if trend == "bullish" and ema_f > ema_s:
            if current_low <= ema_f * 1.01:
                if current_close > current_open:
                    recent_lows = [daily_candles[j].get("low", float("inf")) for j in range(max(0, i-5), i)]
                    swing_low = min(recent_lows) if recent_lows else current_low
                    
                    entry = current_close
                    sl = swing_low - atr * params.sl_buffer_atr
                    risk = entry - sl
                    
                    if risk > 0 and risk < atr * 3:
                        tp1 = entry + risk * params.tp1_rr
                        tp2 = entry + risk * params.tp2_rr
                        tp3 = entry + risk * params.tp3_rr
                        
                        timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                        
                        signal = ModularSignal(
                            symbol=symbol,
                            direction="bullish",
                            bar_index=i,
                            timestamp=str(timestamp) if timestamp else None,
                            entry=entry,
                            stop_loss=sl,
                            tp1=tp1,
                            tp2=tp2,
                            tp3=tp3,
                            confluence_score=4,
                            module="trend",
                            notes={
                                "trend": trend,
                                "ema_pullback": True,
                                "entry_type": "bullish_resumption",
                            }
                        )
                        signals.append(signal)
                        cooldown_until = i + params.cooldown_bars
        
        elif trend == "bearish" and ema_f < ema_s:
            if current_high >= ema_f * 0.99:
                if current_close < current_open:
                    recent_highs = [daily_candles[j].get("high", 0) for j in range(max(0, i-5), i)]
                    swing_high = max(recent_highs) if recent_highs else current_high
                    
                    entry = current_close
                    sl = swing_high + atr * params.sl_buffer_atr
                    risk = sl - entry
                    
                    if risk > 0 and risk < atr * 3:
                        tp1 = entry - risk * params.tp1_rr
                        tp2 = entry - risk * params.tp2_rr
                        tp3 = entry - risk * params.tp3_rr
                        
                        timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                        
                        signal = ModularSignal(
                            symbol=symbol,
                            direction="bearish",
                            bar_index=i,
                            timestamp=str(timestamp) if timestamp else None,
                            entry=entry,
                            stop_loss=sl,
                            tp1=tp1,
                            tp2=tp2,
                            tp3=tp3,
                            confluence_score=4,
                            module="trend",
                            notes={
                                "trend": trend,
                                "ema_pullback": True,
                                "entry_type": "bearish_resumption",
                            }
                        )
                        signals.append(signal)
                        cooldown_until = i + params.cooldown_bars
    
    return signals


def generate_ensemble_signals(
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    monthly_candles: List[Dict],
    symbol: str,
    reversal_params: ReversalParams = None,
    trend_params: TrendParams = None,
) -> List[ModularSignal]:
    """
    Generate signals from both modules and combine them.
    
    Signals are sorted by bar_index and deduplicated
    (no signals on same bar from different modules).
    """
    reversal_signals = generate_reversal_signals(
        daily_candles, weekly_candles, monthly_candles, symbol, reversal_params
    )
    
    trend_signals = generate_trend_signals(daily_candles, symbol, trend_params)
    
    used_bars = set()
    combined = []
    
    all_signals = sorted(reversal_signals + trend_signals, key=lambda s: s.bar_index)
    
    for signal in all_signals:
        if signal.bar_index not in used_bars:
            combined.append(signal)
            used_bars.add(signal.bar_index)
    
    return combined
