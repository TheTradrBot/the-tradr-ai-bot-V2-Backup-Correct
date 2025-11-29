"""Debug backtest process for V3 strategy."""

from datetime import datetime
import pandas as pd
from data import get_ohlcv
from strategy_v3 import generate_signal, calculate_atr, is_price_at_zone

symbol = 'GBP_JPY'
year = 2024
month = 1
start_date = datetime(year, month, 1)
end_date = datetime(year, month, 31)

print(f'Debugging backtest for {symbol} {year}-{month:02d}')
print('=' * 60)

h4_candles = get_ohlcv(symbol, timeframe="H4", count=5000)
daily_candles = get_ohlcv(symbol, timeframe="D", count=1000)
weekly_candles = get_ohlcv(symbol, timeframe="W", count=200)

print(f'Total H4 candles fetched: {len(h4_candles)}')
print(f'Total Daily candles: {len(daily_candles)}')
print(f'Total Weekly candles: {len(weekly_candles)}')

h4_df = pd.DataFrame(h4_candles)
h4_df['time'] = pd.to_datetime(h4_df['time'])

date_range_h4 = h4_df[(h4_df['time'] >= start_date) & (h4_df['time'] <= end_date)]
print(f'H4 candles in date range: {len(date_range_h4)}')

if len(date_range_h4) == 0:
    print('ERROR: No H4 data in date range!')
    print(f'  First H4 candle: {h4_df["time"].min()}')
    print(f'  Last H4 candle: {h4_df["time"].max()}')
    exit()

daily_df = pd.DataFrame(daily_candles)
daily_df['time'] = pd.to_datetime(daily_df['time'])

weekly_df = pd.DataFrame(weekly_candles)
weekly_df['time'] = pd.to_datetime(weekly_df['time'])

print(f'Daily date range: {daily_df["time"].min()} to {daily_df["time"].max()}')
print(f'Weekly date range: {weekly_df["time"].min()} to {weekly_df["time"].max()}')

print()
print('Scanning for signals (checking every bar)...')

signal_count = 0
signal_active_count = 0
signal_watching_count = 0
at_zone_count = 0

for i in range(100, len(date_range_h4)):
    current_bar = date_range_h4.iloc[i]
    current_time = current_bar['time']
    current_price = current_bar['close']
    
    h4_history = date_range_h4.iloc[max(0, i-100):i+1].to_dict('records')
    daily_history = daily_df[daily_df['time'] <= current_time].tail(100).to_dict('records')
    weekly_history = weekly_df[weekly_df['time'] <= current_time].tail(20).to_dict('records')
    
    if len(h4_history) < 50 or len(daily_history) < 50 or len(weekly_history) < 10:
        continue
    
    from strategy_v3 import identify_sr_zones
    daily_zones = identify_sr_zones(daily_history, tolerance_pct=0.003)
    weekly_zones = identify_sr_zones(weekly_history, tolerance_pct=0.005)
    all_zones = daily_zones + weekly_zones
    
    daily_atr = calculate_atr(daily_history, period=14)
    
    for zone in all_zones:
        if is_price_at_zone(current_price, zone, daily_atr):
            at_zone_count += 1
            break
    
    signal = generate_signal(
        symbol=symbol,
        h4_candles=h4_history,
        daily_candles=daily_history,
        weekly_candles=weekly_history,
        min_rr=4.0,
        min_confluence=3
    )
    
    if signal:
        signal_count += 1
        if signal.status == 'active':
            signal_active_count += 1
            print(f'  ACTIVE: {current_time} | {signal.direction} | Entry: {signal.entry:.5f} | R:R: {signal.r_multiple}')
        elif signal.status == 'watching':
            signal_watching_count += 1
        
        if signal_count <= 5:
            print(f'  Signal at {current_time}: {signal.direction} {signal.status} | R:R: {signal.r_multiple} | Confluence: {signal.confluence_score}')

print()
print('Summary:')
print(f'  Total bars checked: {len(date_range_h4) - 100}')
print(f'  Bars at zone: {at_zone_count}')
print(f'  Total signals: {signal_count}')
print(f'  Active signals: {signal_active_count}')
print(f'  Watching signals: {signal_watching_count}')
print()
print('Only ACTIVE signals would be taken as trades in backtest.')
