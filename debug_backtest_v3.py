"""Debug the actual backtest_v3 function."""

from datetime import datetime
import pandas as pd
from data import get_ohlcv
from strategy_v3 import (
    generate_signal, detect_bos, calculate_atr
)

symbol = 'GBP_JPY'
year = 2024
month = 1
start_date = datetime(year, month, 1)
end_date = datetime(year, month, 31)
risk_per_trade = 250.0
min_rr = 3.0
max_trades_per_day = 5
cooldown_bars = 4

print(f'Debugging backtest_v3 for {symbol} {year}-{month:02d}')
print('=' * 60)

h4_candles = get_ohlcv(symbol, timeframe="H4", count=5000)
daily_candles = get_ohlcv(symbol, timeframe="D", count=1000)
weekly_candles = get_ohlcv(symbol, timeframe="W", count=200)

h4_df = pd.DataFrame(h4_candles)
h4_df['time'] = pd.to_datetime(h4_df['time'])

print(f'Full H4 data: {len(h4_df)} candles from {h4_df["time"].min()} to {h4_df["time"].max()}')

start_idx = h4_df[h4_df['time'] >= start_date].index.min()
end_idx = h4_df[h4_df['time'] <= end_date].index.max()

print(f'Start index (first bar >= Jan 1): {start_idx}')
print(f'End index (last bar <= Jan 31): {end_idx}')

if pd.isna(start_idx) or pd.isna(end_idx):
    print('ERROR: Could not find date range')
    exit()

lookback_start = max(0, start_idx - 100)
print(f'Lookback start (for 100 bars history): {lookback_start}')

h4_df = h4_df.iloc[lookback_start:end_idx + 1].reset_index(drop=True)
start_bar = min(100, start_idx - lookback_start)

print(f'After slicing: {len(h4_df)} bars, start_bar = {start_bar}')
print(f'First bar in slice: {h4_df.iloc[0]["time"]}')
print(f'Bar at start_bar: {h4_df.iloc[start_bar]["time"]}')
print(f'Last bar: {h4_df.iloc[-1]["time"]}')

daily_df = pd.DataFrame(daily_candles)
daily_df['time'] = pd.to_datetime(daily_df['time'])

weekly_df = pd.DataFrame(weekly_candles)
weekly_df['time'] = pd.to_datetime(weekly_df['time'])

print()
print('Simulating backtest loop:')
print('-' * 40)

trades = []
in_trade = False
trade_entry = None
last_signal_bar = -cooldown_bars
daily_trades = {}
signals_generated = 0
entries = 0

for i in range(start_bar, len(h4_df)):
    current_bar = h4_df.iloc[i]
    current_time = current_bar['time']
    current_date = current_time.date()
    
    if current_date not in daily_trades:
        daily_trades[current_date] = 0
    
    if i - last_signal_bar < cooldown_bars:
        continue
    
    if daily_trades[current_date] >= max_trades_per_day:
        continue
    
    if in_trade:
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
        signals_generated += 1
        print(f'Bar {i} | {current_time} | Signal: {signal.direction} {signal.status} | Confluence: {signal.confluence_score} | R:R: {signal.r_multiple}')
        
        if signal.status in ('active', 'watching'):
            trade_entry = {
                'symbol': symbol,
                'direction': signal.direction,
                'entry_price': signal.entry,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'r_multiple': signal.r_multiple,
                'entry_time': current_time,
            }
            in_trade = True
            last_signal_bar = i
            daily_trades[current_date] += 1
            entries += 1
            print(f'  -> ENTERED TRADE! Entry: {signal.entry:.5f}, SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}')

print()
print(f'Total signals: {signals_generated}')
print(f'Total entries: {entries}')
print(f'In trade at end: {in_trade}')
