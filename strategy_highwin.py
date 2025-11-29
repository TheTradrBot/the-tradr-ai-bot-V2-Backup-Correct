"""
High Win-Rate Strategy for 5%ers Challenge
Target: 65%+ win rate, 50%+ annual return per asset

Strategy: Bollinger Band + RSI(2) Mean Reversion
1. Range Detection: ADX < 22, ATR compression
2. Entry: Price spikes outside 2.2σ BB, RSI(2) <= 8 / >= 92, closes back inside
3. Exit: TP1 at mid-band (0.85R, 70%), TP2 at opposite inner band (1.4R, 20%), runner 10%
4. Risk: 0.5 ATR stop, 1.25% risk per trade, break-even after TP1
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime
import math

ASSET_PARAMS = {
    'EUR_USD': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'GBP_USD': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'USD_JPY': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'USD_CHF': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'AUD_USD': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'USD_CAD': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'NZD_USD': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'EUR_GBP': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'EUR_JPY': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'GBP_JPY': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 85, 'rsi_os': 15, 'adx_max': 35},
    'XAU_USD': {'bb_mult': 2.2, 'atr_sl': 0.4, 'rsi_ob': 82, 'rsi_os': 18, 'adx_max': 40},
    'XAG_USD': {'bb_mult': 2.2, 'atr_sl': 0.4, 'rsi_ob': 82, 'rsi_os': 18, 'adx_max': 40},
    'SPX500_USD': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 82, 'rsi_os': 18, 'adx_max': 40},
    'NAS100_USD': {'bb_mult': 2.0, 'atr_sl': 0.35, 'rsi_ob': 82, 'rsi_os': 18, 'adx_max': 40},
    'BTC_USD': {'bb_mult': 2.2, 'atr_sl': 0.45, 'rsi_ob': 80, 'rsi_os': 20, 'adx_max': 45},
    'ETH_USD': {'bb_mult': 2.2, 'atr_sl': 0.45, 'rsi_ob': 80, 'rsi_os': 20, 'adx_max': 45},
}

STRATEGY_CONFIG = {
    'risk_per_trade_pct': 2.5,
    'max_daily_loss_pct': 4.5,
    'soft_daily_loss_pct': 3.0,
    'max_total_drawdown_pct': 9.0,
    'tp1_ratio': 1.5,
    'tp1_size': 0.50,
    'tp2_ratio': 2.5,
    'tp2_size': 0.30,
    'tp3_ratio': 4.0,
    'tp3_size': 0.20,
    'cooldown_after_loss': 1,
    'max_trades_per_day': 12,
    'bb_period': 20,
    'atr_period': 14,
}


def calculate_rsi(closes: List[float], period: int = 2) -> List[float]:
    """Calculate RSI with specified period (default RSI(2) for mean reversion)."""
    if len(closes) < period + 1:
        return []
    
    rsi_values = []
    gains = []
    losses = []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    
    for i in range(period - 1, len(gains)):
        avg_gain = sum(gains[i-period+1:i+1]) / period
        avg_loss = sum(losses[i-period+1:i+1]) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    return rsi_values


def calculate_bollinger_bands(closes: List[float], period: int = 20, mult: float = 2.0) -> Dict:
    """Calculate Bollinger Bands."""
    if len(closes) < period:
        return {'upper': [], 'middle': [], 'lower': []}
    
    upper = []
    middle = []
    lower = []
    
    for i in range(period - 1, len(closes)):
        window = closes[i-period+1:i+1]
        sma = sum(window) / period
        std = (sum((x - sma) ** 2 for x in window) / period) ** 0.5
        
        middle.append(sma)
        upper.append(sma + mult * std)
        lower.append(sma - mult * std)
    
    return {'upper': upper, 'middle': middle, 'lower': lower}


def calculate_adx(candles: List[Dict], period: int = 14) -> List[float]:
    """Calculate ADX for trend strength detection."""
    if len(candles) < period + 1:
        return []
    
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []
    
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        close_prev = candles[i-1]['close']
        high_prev = candles[i-1]['high']
        low_prev = candles[i-1]['low']
        
        tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
        tr_list.append(tr)
        
        plus_dm = max(0, high - high_prev) if high - high_prev > low_prev - low else 0
        minus_dm = max(0, low_prev - low) if low_prev - low > high - high_prev else 0
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
    
    adx_values = []
    
    for i in range(period - 1, len(tr_list)):
        atr = sum(tr_list[i-period+1:i+1]) / period
        if atr == 0:
            atr = 0.0001
        
        plus_di = 100 * sum(plus_dm_list[i-period+1:i+1]) / period / atr
        minus_di = 100 * sum(minus_dm_list[i-period+1:i+1]) / period / atr
        
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        
        adx_values.append(dx)
    
    smoothed_adx = []
    if len(adx_values) >= period:
        for i in range(period - 1, len(adx_values)):
            smoothed_adx.append(sum(adx_values[i-period+1:i+1]) / period)
    
    return smoothed_adx if smoothed_adx else adx_values


def calculate_atr(candles: List[Dict], period: int = 14) -> float:
    """Calculate ATR for volatility measurement."""
    if len(candles) < period + 1:
        return 0.0001
    
    tr_list = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i]['high'] - candles[i]['low'],
            abs(candles[i]['high'] - candles[i-1]['close']),
            abs(candles[i]['low'] - candles[i-1]['close'])
        )
        tr_list.append(tr)
    
    return sum(tr_list[-period:]) / period if tr_list else 0.0001


def calculate_ema(values: List[float], period: int) -> List[float]:
    """Calculate EMA for trend detection."""
    if len(values) < period:
        return []
    
    multiplier = 2 / (period + 1)
    ema = [sum(values[:period]) / period]
    
    for i in range(period, len(values)):
        ema.append((values[i] - ema[-1]) * multiplier + ema[-1])
    
    return ema


def detect_regime(candles: List[Dict], adx_threshold: float = 25) -> str:
    """Detect market regime: 'trend', 'range', or 'rotation'."""
    if len(candles) < 30:
        return 'range'
    
    adx = calculate_adx(candles[-50:], 14)
    if not adx:
        return 'range'
    
    current_adx = adx[-1] if adx else 0
    
    closes = [c['close'] for c in candles[-20:]]
    ema_fast = calculate_ema(closes, 8)
    ema_slow = calculate_ema(closes, 21)
    
    if current_adx >= adx_threshold:
        return 'trend'
    elif current_adx < 20:
        return 'range'
    else:
        return 'rotation'


def find_swing_points(candles: List[Dict], lookback: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Find swing highs and lows for order block detection."""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        is_swing_high = True
        is_swing_low = True
        
        for j in range(1, lookback + 1):
            if candles[i]['high'] <= candles[i-j]['high'] or candles[i]['high'] <= candles[i+j]['high']:
                is_swing_high = False
            if candles[i]['low'] >= candles[i-j]['low'] or candles[i]['low'] >= candles[i+j]['low']:
                is_swing_low = False
        
        if is_swing_high:
            swing_highs.append({'idx': i, 'price': candles[i]['high'], 'time': candles[i]['time']})
        if is_swing_low:
            swing_lows.append({'idx': i, 'price': candles[i]['low'], 'time': candles[i]['time']})
    
    return swing_highs, swing_lows


def find_order_blocks(candles: List[Dict], lookback: int = 20) -> Tuple[List[Dict], List[Dict]]:
    """Find bullish and bearish order blocks."""
    bullish_obs = []
    bearish_obs = []
    
    for i in range(lookback, len(candles) - 3):
        if candles[i]['close'] < candles[i]['open']:
            if candles[i+1]['close'] > candles[i]['high'] and candles[i+2]['close'] > candles[i+1]['close']:
                bullish_obs.append({
                    'idx': i,
                    'high': candles[i]['high'],
                    'low': candles[i]['low'],
                    'time': candles[i]['time']
                })
        
        if candles[i]['close'] > candles[i]['open']:
            if candles[i+1]['close'] < candles[i]['low'] and candles[i+2]['close'] < candles[i+1]['close']:
                bearish_obs.append({
                    'idx': i,
                    'high': candles[i]['high'],
                    'low': candles[i]['low'],
                    'time': candles[i]['time']
                })
    
    return bullish_obs[-5:], bearish_obs[-5:]


def check_liquidity_sweep(candles: List[Dict], swing_points: List[Dict], direction: str) -> bool:
    """Check if recent price action swept liquidity at swing points within last 3 candles."""
    lookback = STRATEGY_CONFIG.get('liquidity_sweep_lookback', 3)
    if len(candles) < lookback or not swing_points:
        return False
    
    recent_candles = candles[-lookback:]
    
    for swing in swing_points[-3:]:
        if direction == 'long':
            for c in recent_candles:
                if c['low'] < swing['price'] and c['close'] > swing['price']:
                    return True
        else:
            for c in recent_candles:
                if c['high'] > swing['price'] and c['close'] < swing['price']:
                    return True
    
    return False


def generate_bb_signal(candles: List[Dict], symbol: str) -> Optional[Dict]:
    """Generate Bollinger Band + RSI(2) mean-reversion signal."""
    if len(candles) < 30:
        return None
    
    params = ASSET_PARAMS.get(symbol, ASSET_PARAMS['EUR_USD'])
    closes = [c['close'] for c in candles]
    
    bb = calculate_bollinger_bands(closes, STRATEGY_CONFIG['bb_period'], params['bb_mult'])
    rsi = calculate_rsi(closes, 2)
    adx = calculate_adx(candles, 14)
    atr = calculate_atr(candles)
    
    if not bb['upper'] or not rsi or len(rsi) < 2:
        return None
    
    current_price = candles[-1]['close']
    prev_price = candles[-2]['close']
    current_rsi = rsi[-1]
    prev_rsi = rsi[-2] if len(rsi) > 1 else 50
    current_adx = adx[-1] if adx else 15
    
    upper = bb['upper'][-1]
    lower = bb['lower'][-1]
    middle = bb['middle'][-1]
    prev_upper = bb['upper'][-2] if len(bb['upper']) > 1 else upper
    prev_lower = bb['lower'][-2] if len(bb['lower']) > 1 else lower
    
    if current_adx > params['adx_max']:
        return None
    
    sl_distance = atr * params['atr_sl']
    
    long_condition = (
        (current_price <= lower * 1.002 or prev_price < prev_lower) and
        current_rsi <= params['rsi_os'] + 5
    )
    
    short_condition = (
        (current_price >= upper * 0.998 or prev_price > prev_upper) and
        current_rsi >= params['rsi_ob'] - 5
    )
    
    if long_condition:
        entry = current_price
        sl = entry - sl_distance
        tp1 = entry + sl_distance * 1.5
        tp2 = entry + sl_distance * 2.5
        tp3 = entry + sl_distance * 4.0
        
        return {
            'direction': 'long',
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'signal_type': 'bb_reversal',
            'rsi': current_rsi,
            'adx': current_adx,
            'time': candles[-1]['time']
        }
    
    elif short_condition:
        entry = current_price
        sl = entry + sl_distance
        tp1 = entry - sl_distance * 1.5
        tp2 = entry - sl_distance * 2.5
        tp3 = entry - sl_distance * 4.0
        
        return {
            'direction': 'short',
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'signal_type': 'bb_reversal',
            'rsi': current_rsi,
            'adx': current_adx,
            'time': candles[-1]['time']
        }
    
    return None


def generate_momentum_signal(candles: List[Dict], symbol: str) -> Optional[Dict]:
    """Generate momentum breakout signal for trend markets."""
    if len(candles) < 100:
        return None
    
    params = ASSET_PARAMS.get(symbol, ASSET_PARAMS['EUR_USD'])
    closes = [c['close'] for c in candles]
    
    ema_34 = calculate_ema(closes, 34)
    ema_89 = calculate_ema(closes, 89)
    
    if len(ema_34) < 3 or len(ema_89) < 3:
        return None
    
    current_price = closes[-1]
    atr = calculate_atr(candles)
    
    swing_highs, swing_lows = find_swing_points(candles[-50:])
    
    ema_34_current = ema_34[-1]
    ema_89_current = ema_89[-1]
    ema_34_prev = ema_34[-2]
    ema_89_prev = ema_89[-2]
    
    bullish_cross = ema_34_prev < ema_89_prev and ema_34_current > ema_89_current
    bearish_cross = ema_34_prev > ema_89_prev and ema_34_current < ema_89_current
    
    strong_uptrend = ema_34_current > ema_89_current and current_price > ema_34_current
    strong_downtrend = ema_34_current < ema_89_current and current_price < ema_34_current
    
    if (bullish_cross or strong_uptrend) and check_liquidity_sweep(candles, swing_lows, 'long'):
        sl = current_price - (atr * params['atr_mult'] * 1.2)
        tp1 = current_price + (atr * params['atr_mult'] * 1.0)
        tp2 = current_price + (atr * params['atr_mult'] * 2.0)
        tp3 = current_price + (atr * params['atr_mult'] * 3.0)
        
        return {
            'direction': 'long',
            'entry': current_price,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'signal_type': 'momentum',
            'ema_cross': bullish_cross,
            'time': candles[-1]['time']
        }
    
    elif (bearish_cross or strong_downtrend) and check_liquidity_sweep(candles, swing_highs, 'short'):
        sl = current_price + (atr * params['atr_mult'] * 1.2)
        tp1 = current_price - (atr * params['atr_mult'] * 1.0)
        tp2 = current_price - (atr * params['atr_mult'] * 2.0)
        tp3 = current_price - (atr * params['atr_mult'] * 3.0)
        
        return {
            'direction': 'short',
            'entry': current_price,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'signal_type': 'momentum',
            'ema_cross': bearish_cross,
            'time': candles[-1]['time']
        }
    
    return None


def generate_signal(candles: List[Dict], symbol: str) -> Optional[Dict]:
    """Main signal generator - BB + RSI mean reversion focused."""
    if len(candles) < 50:
        return None
    
    signal = generate_bb_signal(candles, symbol)
    if signal:
        signal['regime'] = 'bb_mean_reversion'
        return signal
    
    return None


def simulate_trade(candles: List[Dict], signal: Dict, start_idx: int, single_tp: bool = True) -> Dict:
    """Simulate trade with single TP or staggered exits."""
    entry = signal['entry']
    sl = signal['sl']
    tp1 = signal['tp1']
    tp2 = signal['tp2']
    tp3 = signal['tp3']
    direction = signal['direction']
    
    target_tp = tp3 if single_tp else tp1
    
    result = {
        'entry': entry,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'direction': direction,
        'signal_type': signal.get('signal_type', 'unknown'),
        'regime': signal.get('regime', 'unknown'),
        'entry_time': signal['time'],
        'exit_time': None,
        'exit_price': None,
        'pnl_pct': 0,
        'hit_tp1': False,
        'hit_tp2': False,
        'hit_tp3': False,
        'hit_sl': False,
        'partial_pnl': 0,
    }
    
    if single_tp:
        for i in range(start_idx + 1, min(start_idx + 100, len(candles))):
            candle = candles[i]
            
            if direction == 'long':
                if candle['low'] <= sl:
                    result['hit_sl'] = True
                    result['exit_price'] = sl
                    result['exit_time'] = candle['time']
                    result['pnl_pct'] = (sl - entry) / entry
                    break
                
                if candle['high'] >= target_tp:
                    result['hit_tp2'] = True
                    result['exit_price'] = target_tp
                    result['exit_time'] = candle['time']
                    result['pnl_pct'] = (target_tp - entry) / entry
                    break
            else:
                if candle['high'] >= sl:
                    result['hit_sl'] = True
                    result['exit_price'] = sl
                    result['exit_time'] = candle['time']
                    result['pnl_pct'] = (entry - sl) / entry
                    break
                
                if candle['low'] <= target_tp:
                    result['hit_tp2'] = True
                    result['exit_price'] = target_tp
                    result['exit_time'] = candle['time']
                    result['pnl_pct'] = (entry - target_tp) / entry
                    break
    else:
        remaining_size = 1.0
        partial_pnl = 0
        
        for i in range(start_idx + 1, min(start_idx + 100, len(candles))):
            candle = candles[i]
            
            if direction == 'long':
                if candle['low'] <= sl:
                    result['hit_sl'] = True
                    result['exit_price'] = sl
                    result['exit_time'] = candle['time']
                    sl_loss = (sl - entry) / entry * remaining_size
                    result['pnl_pct'] = partial_pnl + sl_loss
                    break
                
                if not result['hit_tp1'] and candle['high'] >= tp1:
                    result['hit_tp1'] = True
                    pnl = (tp1 - entry) / entry * STRATEGY_CONFIG['tp1_size']
                    partial_pnl += pnl
                    remaining_size -= STRATEGY_CONFIG['tp1_size']
                    sl = entry
                
                if not result['hit_tp2'] and candle['high'] >= tp2:
                    result['hit_tp2'] = True
                    pnl = (tp2 - entry) / entry * STRATEGY_CONFIG['tp2_size']
                    partial_pnl += pnl
                    remaining_size -= STRATEGY_CONFIG['tp2_size']
                    sl = tp1
                
                if not result['hit_tp3'] and candle['high'] >= tp3:
                    result['hit_tp3'] = True
                    pnl = (tp3 - entry) / entry * STRATEGY_CONFIG['tp3_size']
                    partial_pnl += pnl
                    remaining_size = 0
                    result['exit_price'] = tp3
                    result['exit_time'] = candle['time']
                    result['pnl_pct'] = partial_pnl
                    break
            
            else:
                if candle['high'] >= sl:
                    result['hit_sl'] = True
                    result['exit_price'] = sl
                    result['exit_time'] = candle['time']
                    sl_loss = (entry - sl) / entry * remaining_size
                    result['pnl_pct'] = partial_pnl + sl_loss
                    break
                
                if not result['hit_tp1'] and candle['low'] <= tp1:
                    result['hit_tp1'] = True
                    pnl = (entry - tp1) / entry * STRATEGY_CONFIG['tp1_size']
                    partial_pnl += pnl
                    remaining_size -= STRATEGY_CONFIG['tp1_size']
                    sl = entry
                
                if not result['hit_tp2'] and candle['low'] <= tp2:
                    result['hit_tp2'] = True
                    pnl = (entry - tp2) / entry * STRATEGY_CONFIG['tp2_size']
                    partial_pnl += pnl
                    remaining_size -= STRATEGY_CONFIG['tp2_size']
                    sl = tp1
                
                if not result['hit_tp3'] and candle['low'] <= tp3:
                    result['hit_tp3'] = True
                    pnl = (entry - tp3) / entry * STRATEGY_CONFIG['tp3_size']
                    partial_pnl += pnl
                    remaining_size = 0
                    result['exit_price'] = tp3
                    result['exit_time'] = candle['time']
                    result['pnl_pct'] = partial_pnl
                    break
    
    if result['exit_time'] is None:
        result['exit_time'] = candles[min(start_idx + 100, len(candles) - 1)]['time']
        last_price = candles[min(start_idx + 100, len(candles) - 1)]['close']
        result['exit_price'] = last_price
        if single_tp:
            if direction == 'long':
                result['pnl_pct'] = (last_price - entry) / entry
            else:
                result['pnl_pct'] = (entry - last_price) / entry
        else:
            if direction == 'long':
                result['pnl_pct'] = partial_pnl + (last_price - entry) / entry * remaining_size
            else:
                result['pnl_pct'] = partial_pnl + (entry - last_price) / entry * remaining_size
            result['partial_pnl'] = partial_pnl
    
    result['won'] = result['pnl_pct'] > 0
    
    return result


def backtest_asset(candles: List[Dict], symbol: str, account_size: float = 10000) -> Dict:
    """Backtest strategy on single asset."""
    if len(candles) < 200:
        return {'trades': [], 'total_pnl': 0, 'win_rate': 0, 'total_return': 0}
    
    trades = []
    equity = account_size
    peak_equity = account_size
    max_drawdown = 0
    daily_pnl = {}
    cooldown = 0
    
    i = 100
    while i < len(candles) - 10:
        if cooldown > 0:
            cooldown -= 1
            i += 1
            continue
        
        signal = generate_signal(candles[:i+1], symbol)
        
        if signal:
            trade_result = simulate_trade(candles, signal, i)
            
            risk_amount = equity * (STRATEGY_CONFIG['risk_per_trade_pct'] / 100)
            pnl_usd = risk_amount * trade_result['pnl_pct'] * 10
            
            trade_result['pnl_usd'] = pnl_usd
            trade_result['equity_after'] = equity + pnl_usd
            
            trades.append(trade_result)
            equity += pnl_usd
            
            if equity > peak_equity:
                peak_equity = equity
            
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_drawdown:
                max_drawdown = dd
            
            if trade_result['pnl_pct'] < 0:
                cooldown = STRATEGY_CONFIG['cooldown_after_loss']
            
            i += 5
        else:
            i += 1
    
    if not trades:
        return {'trades': [], 'total_pnl': 0, 'win_rate': 0, 'total_return': 0, 'max_drawdown': 0}
    
    wins = sum(1 for t in trades if t['won'])
    win_rate = (wins / len(trades)) * 100 if trades else 0
    total_pnl = sum(t['pnl_usd'] for t in trades)
    total_return = ((equity - account_size) / account_size) * 100
    
    return {
        'trades': trades,
        'total_trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'final_equity': equity,
        'max_drawdown': max_drawdown,
    }


def run_yearly_backtest(symbol: str, year: int) -> Dict:
    """Run backtest for a specific year."""
    try:
        from data_loader import load_ohlcv_from_csv, resample_to_timeframe, df_to_candle_list
        import numpy as np
        
        df = load_ohlcv_from_csv(symbol)
        
        diffs = np.diff(df.index[:100].astype(np.int64)) / 1e9 / 3600
        median_diff = np.median(diffs[diffs > 0]) if len(diffs[diffs > 0]) > 0 else 24
        if median_diff <= 1.5:
            df = resample_to_timeframe(df, 'H4')
        
        df_year = df[df.index.year == year]
        
        if len(df_year) < 100:
            return {'error': f'Insufficient data for {symbol} in {year}'}
        
        candles = df_to_candle_list(df_year)
        for c in candles:
            if not isinstance(c['time'], str):
                c['time'] = c['time'].strftime('%Y-%m-%dT%H:%M:%S')
        
        result = backtest_asset(candles, symbol)
        result['symbol'] = symbol
        result['year'] = year
        
        return result
        
    except Exception as e:
        return {'error': str(e), 'symbol': symbol, 'year': year}
