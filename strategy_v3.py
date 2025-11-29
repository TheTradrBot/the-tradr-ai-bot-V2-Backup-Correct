"""
Blueprint HTF Confluence Strategy V3

Based on the user's methodology from TradingView images:
1. Monthly/Weekly S/R levels from swing highs/lows
2. Daily framework for trade setups
3. Fib Golden Pocket (0.618-0.796) for entries
4. Fib Extensions (-0.25, -0.68, -1.0, -1.42, -2.0) for TPs
5. Confluence scoring instead of all-or-nothing filters

Target: +60% annual return per asset ($60K on $100K account with 1% risk per trade)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime


@dataclass
class V3Params:
    """Optimized parameters for Blueprint V3 Strategy."""
    sr_lookback_bars: int = 3
    sr_tolerance_pct: float = 0.3
    fib_golden_low: float = 0.618
    fib_golden_high: float = 0.796
    fib_ext_tp1: float = 0.25
    fib_ext_tp2: float = 0.68
    fib_ext_tp3: float = 1.0
    fib_ext_tp4: float = 1.42
    fib_ext_tp5: float = 2.0
    sl_atr_buffer: float = 0.3
    min_confluence: int = 3
    cooldown_bars: int = 2
    require_rejection_candle: bool = True
    min_rr_ratio: float = 1.5


@dataclass
class V3Signal:
    """Signal from V3 strategy."""
    symbol: str
    direction: str
    bar_index: int
    timestamp: Optional[str] = None
    entry: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    tp4: float = 0.0
    tp5: float = 0.0
    confluence_score: int = 0
    sr_level: float = 0.0
    sr_type: str = ""
    fib_zone: str = ""
    notes: Dict = field(default_factory=dict)


def calculate_atr(candles: List[Dict], period: int = 14) -> float:
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


def find_htf_sr_levels(
    candles: List[Dict], 
    lookback: int = 3,
    tolerance_pct: float = 0.5
) -> Tuple[List[Tuple[float, int, str]], List[Tuple[float, int, str]]]:
    """
    Find HTF Support/Resistance levels from swing highs/lows.
    
    Returns:
        Tuple of (resistance_levels, support_levels)
        Each level is (price, touch_count, strength)
    """
    if len(candles) < lookback * 2 + 1:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i].get("high", 0)
        low = candles[i].get("low", float("inf"))
        
        is_swing_high = all(
            high >= candles[i-j].get("high", 0) and high >= candles[i+j].get("high", 0)
            for j in range(1, lookback + 1)
        )
        
        is_swing_low = all(
            low <= candles[i-j].get("low", float("inf")) and low <= candles[i+j].get("low", float("inf"))
            for j in range(1, lookback + 1)
        )
        
        if is_swing_high:
            swing_highs.append(high)
        if is_swing_low:
            swing_lows.append(low)
    
    def cluster_levels(levels: List[float], tol: float) -> List[Tuple[float, int, str]]:
        """Cluster similar levels and count touches."""
        if not levels:
            return []
        
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if current_cluster and abs(level - current_cluster[-1]) / current_cluster[-1] * 100 < tol:
                current_cluster.append(level)
            else:
                if current_cluster:
                    avg = sum(current_cluster) / len(current_cluster)
                    touches = len(current_cluster)
                    strength = "strong" if touches >= 3 else "moderate" if touches >= 2 else "weak"
                    clusters.append((avg, touches, strength))
                current_cluster = [level]
        
        if current_cluster:
            avg = sum(current_cluster) / len(current_cluster)
            touches = len(current_cluster)
            strength = "strong" if touches >= 3 else "moderate" if touches >= 2 else "weak"
            clusters.append((avg, touches, strength))
        
        return sorted(clusters, key=lambda x: x[1], reverse=True)
    
    resistance = cluster_levels(swing_highs, tolerance_pct)
    support = cluster_levels(swing_lows, tolerance_pct)
    
    return resistance, support


def find_impulse_leg(
    candles: List[Dict],
    end_index: int,
    lookback: int = 30,
    min_body_ratio: float = 1.2
) -> Optional[Dict]:
    """
    Find the most recent impulse leg for Fibonacci calculations.
    
    An impulse leg is a significant price move with:
    - Strong candle body (> average)
    - Clear direction
    - Structure break (new high/low)
    """
    if end_index < lookback:
        return None
    
    slice_candles = candles[max(0, end_index - lookback):end_index + 1]
    
    body_sizes = []
    for c in slice_candles:
        body = abs(c.get("close", 0) - c.get("open", 0))
        body_sizes.append(body)
    
    avg_body = sum(body_sizes) / len(body_sizes) if body_sizes else 0
    
    swing_high_idx = 0
    swing_high = 0
    swing_low_idx = 0
    swing_low = float("inf")
    
    for i, c in enumerate(slice_candles):
        if c.get("high", 0) > swing_high:
            swing_high = c.get("high", 0)
            swing_high_idx = i
        if c.get("low", float("inf")) < swing_low:
            swing_low = c.get("low", float("inf"))
            swing_low_idx = i
    
    if swing_high_idx > swing_low_idx:
        return {
            "direction": "bullish",
            "start_price": swing_low,
            "end_price": swing_high,
            "start_idx": swing_low_idx + (end_index - lookback),
            "end_idx": swing_high_idx + (end_index - lookback),
            "range": swing_high - swing_low,
        }
    else:
        return {
            "direction": "bearish",
            "start_price": swing_high,
            "end_price": swing_low,
            "start_idx": swing_high_idx + (end_index - lookback),
            "end_idx": swing_low_idx + (end_index - lookback),
            "range": swing_high - swing_low,
        }


def calculate_fib_levels(impulse: Dict, params: V3Params) -> Dict:
    """
    Calculate Fibonacci retracement and extension levels from impulse leg.
    
    Retracements for entry: 0.5, 0.618, 0.66, 0.796
    Extensions for TPs: -0.25, -0.68, -1.0, -1.42, -2.0
    """
    if not impulse or impulse["range"] <= 0:
        return {}
    
    start = impulse["start_price"]
    end = impulse["end_price"]
    range_size = impulse["range"]
    direction = impulse["direction"]
    
    if direction == "bullish":
        fib_50 = end - range_size * 0.5
        fib_618 = end - range_size * 0.618
        fib_66 = end - range_size * 0.66
        fib_786 = end - range_size * 0.786
        fib_796 = end - range_size * 0.796
        
        ext_025 = end + range_size * 0.25
        ext_068 = end + range_size * 0.68
        ext_100 = end + range_size * 1.0
        ext_142 = end + range_size * 1.42
        ext_200 = end + range_size * 2.0
        
        golden_pocket_high = fib_618
        golden_pocket_low = fib_796
    else:
        fib_50 = end + range_size * 0.5
        fib_618 = end + range_size * 0.618
        fib_66 = end + range_size * 0.66
        fib_786 = end + range_size * 0.786
        fib_796 = end + range_size * 0.796
        
        ext_025 = end - range_size * 0.25
        ext_068 = end - range_size * 0.68
        ext_100 = end - range_size * 1.0
        ext_142 = end - range_size * 1.42
        ext_200 = end - range_size * 2.0
        
        golden_pocket_low = fib_618
        golden_pocket_high = fib_796
    
    return {
        "direction": direction,
        "impulse_start": start,
        "impulse_end": end,
        "range": range_size,
        "fib_50": fib_50,
        "fib_618": fib_618,
        "fib_66": fib_66,
        "fib_786": fib_786,
        "fib_796": fib_796,
        "golden_pocket_high": golden_pocket_high,
        "golden_pocket_low": golden_pocket_low,
        "ext_025": ext_025,
        "ext_068": ext_068,
        "ext_100": ext_100,
        "ext_142": ext_142,
        "ext_200": ext_200,
    }


def is_rejection_candle(candle: Dict, direction: str) -> bool:
    """
    Check if candle shows rejection (for confirmation).
    
    Bullish rejection: Lower wick > body, close > open
    Bearish rejection: Upper wick > body, close < open
    """
    open_p = candle.get("open", 0)
    close_p = candle.get("close", 0)
    high = candle.get("high", 0)
    low = candle.get("low", 0)
    
    body = abs(close_p - open_p)
    upper_wick = high - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low
    
    if direction == "bullish":
        return close_p > open_p and lower_wick >= body * 0.5
    else:
        return close_p < open_p and upper_wick >= body * 0.5


def calculate_confluence_score(
    price: float,
    direction: str,
    sr_levels: List[Tuple[float, int, str]],
    fib_levels: Dict,
    has_rejection: bool,
    htf_bias: str,
    atr: float,
) -> Tuple[int, Dict]:
    """
    Calculate confluence score (0-7) based on multiple factors.
    
    Factors:
    1. HTF Bias alignment (+1)
    2. At S/R level (+1-2 based on strength)
    3. In Fib Golden Pocket (+2)
    4. Has rejection candle (+1)
    5. Good R:R potential (+1)
    """
    score = 0
    notes = {}
    
    if htf_bias == direction:
        score += 1
        notes["htf_bias"] = "aligned"
    elif htf_bias == "neutral":
        notes["htf_bias"] = "neutral"
    else:
        notes["htf_bias"] = "against"
    
    sr_tolerance = atr * 0.5
    at_sr = False
    sr_strength = ""
    sr_price = 0.0
    
    for sr_level, touches, strength in sr_levels:
        if abs(price - sr_level) <= sr_tolerance:
            at_sr = True
            sr_strength = strength
            sr_price = sr_level
            if strength == "strong":
                score += 2
            elif strength == "moderate":
                score += 1
            else:
                score += 1
            break
    
    if at_sr:
        notes["sr_level"] = f"{sr_strength} @ {sr_price:.5f}"
    
    in_golden_pocket = False
    if fib_levels:
        gp_high = fib_levels.get("golden_pocket_high", 0)
        gp_low = fib_levels.get("golden_pocket_low", 0)
        
        if gp_low <= price <= gp_high:
            in_golden_pocket = True
            score += 2
            notes["fib_zone"] = f"Golden Pocket ({gp_low:.5f} - {gp_high:.5f})"
        elif fib_levels.get("fib_50", 0):
            fib_50 = fib_levels["fib_50"]
            if abs(price - fib_50) <= atr * 0.3:
                score += 1
                notes["fib_zone"] = f"Near 50% Fib ({fib_50:.5f})"
    
    if has_rejection:
        score += 1
        notes["rejection"] = "confirmed"
    
    return score, notes


def determine_htf_bias(
    weekly_candles: List[Dict],
    monthly_candles: List[Dict],
) -> str:
    """
    Determine Higher Timeframe bias from Monthly and Weekly data.
    """
    if not weekly_candles or len(weekly_candles) < 4:
        return "neutral"
    
    recent_weekly = weekly_candles[-4:]
    weekly_closes = [c.get("close", 0) for c in recent_weekly]
    
    higher_highs = 0
    lower_lows = 0
    
    for i in range(1, len(recent_weekly)):
        if recent_weekly[i].get("high", 0) > recent_weekly[i-1].get("high", 0):
            higher_highs += 1
        if recent_weekly[i].get("low", float("inf")) < recent_weekly[i-1].get("low", float("inf")):
            lower_lows += 1
    
    if monthly_candles and len(monthly_candles) >= 2:
        last_month = monthly_candles[-1]
        prev_month = monthly_candles[-2]
        
        if last_month.get("close", 0) > last_month.get("open", 0):
            higher_highs += 1
        else:
            lower_lows += 1
    
    if higher_highs >= 3 and higher_highs > lower_lows:
        return "bullish"
    elif lower_lows >= 3 and lower_lows > higher_highs:
        return "bearish"
    
    return "neutral"


def find_swing_sl(
    candles: List[Dict],
    bar_index: int,
    direction: str,
    lookback: int = 5,
    atr_buffer: float = 0.3,
) -> float:
    """
    Find SL level using recent swing high/low with ATR buffer.
    
    This is tighter than using full impulse range.
    """
    if bar_index < lookback:
        return 0.0
    
    recent = candles[bar_index - lookback:bar_index + 1]
    atr = calculate_atr(candles[:bar_index + 1], 14)
    buffer = atr * atr_buffer
    
    if direction == "bullish":
        swing_low = min(c.get("low", float("inf")) for c in recent)
        return swing_low - buffer
    else:
        swing_high = max(c.get("high", 0) for c in recent)
        return swing_high + buffer


def generate_v3_signals(
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    monthly_candles: List[Dict],
    symbol: str,
    params: V3Params = None,
) -> List[V3Signal]:
    """
    Generate trading signals using Blueprint V3 methodology.
    
    Entry logic:
    1. Identify HTF S/R from Monthly/Weekly swing highs/lows
    2. Find impulse leg and calculate Fib levels
    3. Wait for price to retrace to Golden Pocket (0.618-0.796)
    4. Confirm with rejection candle
    5. Enter with SL at recent swing + buffer, TPs at Fib extensions
    """
    if params is None:
        params = V3Params()
    
    signals = []
    
    if not daily_candles or len(daily_candles) < 50:
        return signals
    
    total_daily = len(daily_candles)
    total_weekly = len(weekly_candles) if weekly_candles else 0
    total_monthly = len(monthly_candles) if monthly_candles else 0
    
    monthly_res, monthly_sup = [], []
    weekly_res, weekly_sup = [], []
    last_sr_update = -1
    sr_update_interval = 20
    
    cooldown_until = 0
    
    for i in range(50, len(daily_candles)):
        if i <= cooldown_until:
            continue
        
        if i - last_sr_update >= sr_update_interval or last_sr_update < 0:
            progress = i / total_daily
            weekly_end = int(progress * total_weekly) if total_weekly > 0 else 0
            monthly_end = int(progress * total_monthly) if total_monthly > 0 else 0
            
            if monthly_candles and monthly_end > 0:
                monthly_slice = monthly_candles[:monthly_end]
                monthly_res, monthly_sup = find_htf_sr_levels(
                    monthly_slice, lookback=2, tolerance_pct=0.5
                )
            
            if weekly_candles and weekly_end > 0:
                weekly_slice = weekly_candles[:weekly_end]
                weekly_res, weekly_sup = find_htf_sr_levels(
                    weekly_slice, lookback=params.sr_lookback_bars, tolerance_pct=params.sr_tolerance_pct
                )
            
            last_sr_update = i
        
        current = daily_candles[i]
        current_close = current.get("close", 0)
        current_high = current.get("high", 0)
        current_low = current.get("low", 0)
        current_open = current.get("open", 0)
        
        recent_slice = daily_candles[max(0, i-20):i+1]
        atr = calculate_atr(recent_slice, 14)
        
        if atr <= 0:
            continue
        
        all_resistance = monthly_res + weekly_res
        all_support = monthly_sup + weekly_sup
        
        weekly_slice_for_bias = weekly_candles[:int((i / total_daily) * total_weekly)] if weekly_candles else []
        monthly_slice_for_bias = monthly_candles[:int((i / total_daily) * total_monthly)] if monthly_candles else []
        htf_bias = determine_htf_bias(weekly_slice_for_bias, monthly_slice_for_bias)
        
        impulse = find_impulse_leg(daily_candles, i, lookback=30)
        fib_levels = calculate_fib_levels(impulse, params) if impulse else {}
        
        for direction in ["bullish", "bearish"]:
            sr_levels = all_support if direction == "bullish" else all_resistance
            
            is_at_sr = False
            sr_level = 0.0
            sr_type = ""
            
            for sr_price, touches, strength in sr_levels:
                tolerance = atr * 0.5
                
                if direction == "bullish":
                    if current_low <= sr_price + tolerance and current_low >= sr_price - tolerance:
                        is_at_sr = True
                        sr_level = sr_price
                        sr_type = f"support_{strength}"
                        break
                else:
                    if current_high >= sr_price - tolerance and current_high <= sr_price + tolerance:
                        is_at_sr = True
                        sr_level = sr_price
                        sr_type = f"resistance_{strength}"
                        break
            
            if not is_at_sr:
                continue
            
            in_golden_pocket = False
            if fib_levels and fib_levels.get("direction") == direction:
                gp_high = fib_levels.get("golden_pocket_high", 0)
                gp_low = fib_levels.get("golden_pocket_low", 0)
                
                if direction == "bullish":
                    in_golden_pocket = current_low <= gp_high and current_low >= gp_low * 0.99
                else:
                    in_golden_pocket = current_high >= gp_low and current_high <= gp_high * 1.01
            
            has_rejection = is_rejection_candle(current, direction)
            
            if params.require_rejection_candle and not has_rejection:
                continue
            
            confluence, notes = calculate_confluence_score(
                current_close, direction, sr_levels, fib_levels,
                has_rejection, htf_bias, atr
            )
            
            if confluence < params.min_confluence:
                continue
            
            entry = current_close
            sl = find_swing_sl(daily_candles, i, direction, lookback=5, atr_buffer=params.sl_atr_buffer)
            
            if sl == 0:
                continue
            
            risk = abs(entry - sl)
            
            if risk <= 0 or risk > atr * 3:
                continue
            
            max_tp1_rr = 1.5
            max_tp2_rr = 2.0
            max_tp3_rr = 3.0
            
            fib_direction_matches = fib_levels and fib_levels.get("direction") == direction
            
            if direction == "bullish":
                if fib_direction_matches and fib_levels.get("ext_025"):
                    tp1_fib = fib_levels["ext_025"]
                    tp2_fib = fib_levels["ext_068"]
                    tp3_fib = fib_levels["ext_100"]
                    if tp1_fib > entry:
                        tp1 = min(tp1_fib, entry + risk * max_tp1_rr)
                        tp2 = min(tp2_fib, entry + risk * max_tp2_rr)
                        tp3 = min(tp3_fib, entry + risk * max_tp3_rr)
                    else:
                        tp1 = entry + risk * 1.5
                        tp2 = entry + risk * 2.0
                        tp3 = entry + risk * 3.0
                else:
                    tp1 = entry + risk * 1.5
                    tp2 = entry + risk * 2.0
                    tp3 = entry + risk * 3.0
                tp4 = entry + risk * 4.0
                tp5 = entry + risk * 5.0
            else:
                if fib_direction_matches and fib_levels.get("ext_025"):
                    tp1_fib = fib_levels["ext_025"]
                    tp2_fib = fib_levels["ext_068"]
                    tp3_fib = fib_levels["ext_100"]
                    if tp1_fib < entry:
                        tp1 = max(tp1_fib, entry - risk * max_tp1_rr)
                        tp2 = max(tp2_fib, entry - risk * max_tp2_rr)
                        tp3 = max(tp3_fib, entry - risk * max_tp3_rr)
                    else:
                        tp1 = entry - risk * 1.5
                        tp2 = entry - risk * 2.0
                        tp3 = entry - risk * 3.0
                else:
                    tp1 = entry - risk * 1.5
                    tp2 = entry - risk * 2.0
                    tp3 = entry - risk * 3.0
                tp4 = entry - risk * 4.0
                tp5 = entry - risk * 5.0
            
            reward_to_tp1 = abs(tp1 - entry)
            rr_ratio = reward_to_tp1 / risk if risk > 0 else 0
            
            if rr_ratio < params.min_rr_ratio:
                continue
            
            timestamp = current.get("time") or current.get("timestamp") or current.get("date")
            
            signal = V3Signal(
                symbol=symbol,
                direction=direction,
                bar_index=i,
                timestamp=str(timestamp) if timestamp else None,
                entry=entry,
                stop_loss=sl,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                tp4=tp4,
                tp5=tp5,
                confluence_score=confluence,
                sr_level=sr_level,
                sr_type=sr_type,
                fib_zone="golden_pocket" if in_golden_pocket else "sr_zone",
                notes=notes,
            )
            
            signals.append(signal)
            cooldown_until = i + params.cooldown_bars
            break
    
    return signals


def backtest_v3_signals(
    signals: List[V3Signal],
    daily_candles: List[Dict],
    risk_per_trade: float = 1000.0,
) -> Dict:
    """
    Backtest V3 signals with laddered exits.
    
    Exit strategy:
    - TP1 @ 50% of position
    - TP2 @ 30% of position
    - Runner @ 20% with trailing SL
    
    SL is checked before TP on same bar (conservative).
    """
    trades = []
    
    for signal in signals:
        if signal.bar_index >= len(daily_candles) - 1:
            continue
        
        entry_bar = signal.bar_index
        entry_price = signal.entry
        sl = signal.stop_loss
        tp1 = signal.tp1
        tp2 = signal.tp2
        tp3 = signal.tp3
        direction = signal.direction
        
        risk = abs(entry_price - sl)
        if risk <= 0:
            continue
        
        tp1_hit = False
        tp2_hit = False
        tp3_hit = False
        exit_price = entry_price
        exit_reason = "open"
        exit_bar = entry_bar
        trailing_sl = sl
        
        for j in range(entry_bar + 1, len(daily_candles)):
            bar = daily_candles[j]
            bar_high = bar.get("high", 0)
            bar_low = bar.get("low", float("inf"))
            bar_close = bar.get("close", 0)
            
            if direction == "bullish":
                if bar_low <= trailing_sl:
                    exit_price = trailing_sl
                    exit_reason = "SL" if not tp1_hit else "trailing_SL"
                    exit_bar = j
                    break
                
                if not tp1_hit and bar_high >= tp1:
                    tp1_hit = True
                    trailing_sl = entry_price
                
                if tp1_hit and not tp2_hit and bar_high >= tp2:
                    tp2_hit = True
                    trailing_sl = tp1
                
                if tp2_hit and not tp3_hit and bar_high >= tp3:
                    tp3_hit = True
                    exit_price = tp3
                    exit_reason = "TP3"
                    exit_bar = j
                    break
            else:
                if bar_high >= trailing_sl:
                    exit_price = trailing_sl
                    exit_reason = "SL" if not tp1_hit else "trailing_SL"
                    exit_bar = j
                    break
                
                if not tp1_hit and bar_low <= tp1:
                    tp1_hit = True
                    trailing_sl = entry_price
                
                if tp1_hit and not tp2_hit and bar_low <= tp2:
                    tp2_hit = True
                    trailing_sl = tp1
                
                if tp2_hit and not tp3_hit and bar_low <= tp3:
                    tp3_hit = True
                    exit_price = tp3
                    exit_reason = "TP3"
                    exit_bar = j
                    break
        
        if exit_reason == "open":
            exit_price = daily_candles[-1].get("close", entry_price)
            exit_bar = len(daily_candles) - 1
            exit_reason = "end_of_data"
        
        if tp3_hit:
            pnl_pct = 0.5 * 1.0 + 0.3 * (abs(tp2 - entry_price) / risk) + 0.2 * (abs(tp3 - entry_price) / risk)
        elif tp2_hit:
            pnl_pct = 0.5 * 1.0 + 0.3 * (abs(tp2 - entry_price) / risk) + 0.2 * 0
        elif tp1_hit:
            pnl_pct = 0.5 * 1.0 + 0.5 * 0
        else:
            pnl_pct = -1.0
        
        if exit_reason == "trailing_SL":
            if tp2_hit:
                pnl_pct = 0.5 * 1.0 + 0.3 * (abs(tp2 - entry_price) / risk) + 0.2 * (abs(tp1 - entry_price) / risk)
            elif tp1_hit:
                pnl_pct = 0.5 * 1.0 + 0.5 * 0
        
        pnl_usd = pnl_pct * risk_per_trade
        
        trade = {
            "symbol": signal.symbol,
            "direction": direction,
            "entry_bar": entry_bar,
            "exit_bar": exit_bar,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "tp1_hit": tp1_hit,
            "tp2_hit": tp2_hit,
            "tp3_hit": tp3_hit,
            "exit_reason": exit_reason,
            "risk": risk,
            "pnl_r": pnl_pct,
            "pnl_usd": pnl_usd,
            "confluence": signal.confluence_score,
            "sr_level": signal.sr_level,
            "sr_type": signal.sr_type,
            "timestamp": signal.timestamp,
        }
        trades.append(trade)
    
    total_trades = len(trades)
    winners = [t for t in trades if t["pnl_r"] > 0]
    losers = [t for t in trades if t["pnl_r"] <= 0]
    
    total_pnl_r = sum(t["pnl_r"] for t in trades)
    total_pnl_usd = sum(t["pnl_usd"] for t in trades)
    
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
    avg_rr = total_pnl_r / total_trades if total_trades > 0 else 0
    
    return {
        "symbol": signals[0].symbol if signals else "",
        "total_trades": total_trades,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "total_pnl_r": total_pnl_r,
        "total_pnl_usd": total_pnl_usd,
        "avg_rr": avg_rr,
        "trades": trades,
    }


def run_v3_backtest(
    symbol: str,
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    monthly_candles: List[Dict],
    params: V3Params = None,
    risk_per_trade: float = 1000.0,
) -> Dict:
    """
    Run full V3 backtest for a symbol.
    """
    if params is None:
        params = V3Params()
    
    signals = generate_v3_signals(
        daily_candles, weekly_candles, monthly_candles, symbol, params
    )
    
    if not signals:
        return {
            "symbol": symbol,
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0,
            "total_pnl_r": 0,
            "total_pnl_usd": 0,
            "avg_rr": 0,
            "trades": [],
        }
    
    results = backtest_v3_signals(signals, daily_candles, risk_per_trade)
    return results


OPTIMIZED_PARAMS = {
    "EUR_USD": V3Params(
        sr_lookback_bars=2,
        sr_tolerance_pct=0.2,
        min_confluence=3,
        cooldown_bars=3,
        require_rejection_candle=False,
        min_rr_ratio=1.2,
        sl_atr_buffer=0.4,
    ),
    "GBP_USD": V3Params(
        sr_lookback_bars=2,
        sr_tolerance_pct=0.2,
        min_confluence=3,
        cooldown_bars=3,
        require_rejection_candle=False,
        min_rr_ratio=1.2,
        sl_atr_buffer=0.4,
    ),
    "USD_JPY": V3Params(
        sr_lookback_bars=2,
        sr_tolerance_pct=0.25,
        min_confluence=3,
        cooldown_bars=3,
        require_rejection_candle=False,
        min_rr_ratio=1.2,
        sl_atr_buffer=0.4,
    ),
    "USD_CHF": V3Params(
        sr_lookback_bars=2,
        sr_tolerance_pct=0.2,
        min_confluence=3,
        cooldown_bars=3,
        require_rejection_candle=False,
        min_rr_ratio=1.2,
        sl_atr_buffer=0.4,
    ),
    "AUD_USD": V3Params(
        sr_lookback_bars=2,
        sr_tolerance_pct=0.2,
        min_confluence=3,
        cooldown_bars=3,
        require_rejection_candle=False,
        min_rr_ratio=1.2,
        sl_atr_buffer=0.4,
    ),
    "XAU_USD": V3Params(
        sr_lookback_bars=2,
        sr_tolerance_pct=0.35,
        min_confluence=3,
        cooldown_bars=3,
        require_rejection_candle=False,
        min_rr_ratio=1.2,
        sl_atr_buffer=0.4,
    ),
    "BTC_USD": V3Params(
        sr_lookback_bars=2,
        sr_tolerance_pct=0.5,
        min_confluence=3,
        cooldown_bars=3,
        require_rejection_candle=False,
        min_rr_ratio=1.2,
        sl_atr_buffer=0.4,
    ),
}


def get_optimized_params(symbol: str) -> V3Params:
    """Get optimized parameters for a symbol."""
    return OPTIMIZED_PARAMS.get(symbol, V3Params())
