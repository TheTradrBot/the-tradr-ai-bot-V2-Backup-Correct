"""Debug why signals aren't being generated even when at zones."""

from datetime import datetime
import pandas as pd
from data import get_ohlcv
from strategy_v3 import (
    calculate_atr, is_price_at_zone, identify_sr_zones,
    get_htf_bias, detect_bos, find_structural_tp
)

symbol = 'GBP_JPY'
year = 2024
month = 1
start_date = datetime(year, month, 1)
end_date = datetime(year, month, 31)

print(f'Debugging signal generation for {symbol} {year}-{month:02d}')
print('=' * 60)

h4_candles = get_ohlcv(symbol, timeframe="H4", count=5000)
daily_candles = get_ohlcv(symbol, timeframe="D", count=1000)
weekly_candles = get_ohlcv(symbol, timeframe="W", count=200)

h4_df = pd.DataFrame(h4_candles)
h4_df['time'] = pd.to_datetime(h4_df['time'])

daily_df = pd.DataFrame(daily_candles)
daily_df['time'] = pd.to_datetime(daily_df['time'])

weekly_df = pd.DataFrame(weekly_candles)
weekly_df['time'] = pd.to_datetime(weekly_df['time'])

date_range_h4 = h4_df[(h4_df['time'] >= start_date) & (h4_df['time'] <= end_date)]

checked = 0
for i in range(100, min(len(date_range_h4), 120)):
    current_bar = date_range_h4.iloc[i]
    current_time = current_bar['time']
    current_price = current_bar['close']
    
    h4_history = date_range_h4.iloc[max(0, i-100):i+1].to_dict('records')
    daily_history = daily_df[daily_df['time'] <= current_time].tail(100).to_dict('records')
    weekly_history = weekly_df[weekly_df['time'] <= current_time].tail(20).to_dict('records')
    
    if len(h4_history) < 50 or len(daily_history) < 50 or len(weekly_history) < 10:
        continue
    
    daily_zones = identify_sr_zones(daily_history, tolerance_pct=0.003)
    weekly_zones = identify_sr_zones(weekly_history, tolerance_pct=0.005)
    all_zones = daily_zones + weekly_zones
    
    daily_atr = calculate_atr(daily_history, period=14)
    h4_atr = calculate_atr(h4_history, period=14)
    
    active_zone = None
    for zone in all_zones:
        if is_price_at_zone(current_price, zone, daily_atr):
            active_zone = zone
            break
    
    if not active_zone:
        continue
    
    checked += 1
    if checked > 3:
        break
    
    print(f'\nBar at {current_time} | Price: {current_price:.5f}')
    print(f'  Active zone: {active_zone[0]:.5f} - {active_zone[1]:.5f}')
    print(f'  Daily ATR: {daily_atr:.5f} | H4 ATR: {h4_atr:.5f}')
    
    htf_bias = get_htf_bias(weekly_history, daily_history)
    print(f'  HTF Bias: {htf_bias}')
    
    zone_low, zone_high = active_zone
    zone_mid = (zone_low + zone_high) / 2
    
    if htf_bias == 'bullish' or (htf_bias == 'neutral' and current_price > zone_mid):
        direction = 'long'
    elif htf_bias == 'bearish' or (htf_bias == 'neutral' and current_price < zone_mid):
        direction = 'short'
    else:
        direction = None
    
    print(f'  Direction decided: {direction}')
    
    if direction:
        bos_level = detect_bos(h4_history, direction, lookback=20)
        print(f'  BOS Level: {bos_level}')
        
        confluence = 1
        reasoning_parts = [f"At zone {zone_low:.5f}-{zone_high:.5f}"]
        
        if htf_bias != 'neutral':
            confluence += 1
            reasoning_parts.append(f"HTF bias: {htf_bias}")
        
        if bos_level:
            confluence += 2
            reasoning_parts.append(f"BOS at {bos_level:.5f}")
        
        h4_trend = 'up' if h4_history[-1]['close'] > h4_history[-5]['close'] else 'down'
        if (direction == 'long' and h4_trend == 'up') or (direction == 'short' and h4_trend == 'down'):
            confluence += 1
            reasoning_parts.append("H4 momentum aligned")
        
        print(f'  Confluence: {confluence} | Reasoning: {", ".join(reasoning_parts)}')
        print(f'  Min confluence needed: 3')
        
        if confluence >= 3:
            entry = current_price
            if direction == 'long':
                stop_loss = zone_low - (h4_atr * 0.5)
            else:
                stop_loss = zone_high + (h4_atr * 0.5)
            
            take_profit = find_structural_tp(daily_history, entry, stop_loss, direction, min_rr=4.0)
            print(f'  Entry: {entry:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}')
            
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry) if take_profit else 0
            r_multiple = reward / risk if risk > 0 else 0
            print(f'  Risk: {risk:.5f} | Reward: {reward:.5f} | R:R: {r_multiple:.2f}')
            print(f'  Min R:R needed: 4.0')
            
            if bos_level:
                print(f'  Would be ACTIVE trade!')
            else:
                print(f'  Would be WATCHING only (no BOS)')
        else:
            print(f'  REJECTED: Confluence {confluence} < 3')

print('\n' + '=' * 60)
print('KEY INSIGHT: Signals need:')
print('  1. Price at HTF S/R zone')
print('  2. HTF bias aligned OR neutral (confluence +1 if not neutral)')
print('  3. BOS confirmation (confluence +2)')
print('  4. H4 momentum aligned (confluence +1)')
print('  5. Total confluence >= 3')
print('  6. R:R >= 4.0')
print('  7. Only "active" (with BOS) signals are traded')
print()
print('The BOS requirement is very strict - it requires price to')
print('break above recent swing high (long) or below swing low (short).')
