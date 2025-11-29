"""Full debug of backtest_v3 with trade management."""

from datetime import datetime
import pandas as pd
from data import get_ohlcv
from strategy_v3 import generate_signal, detect_bos

symbol = 'GBP_JPY'
year = 2024
month = 1
start_date = datetime(year, month, 1)
end_date = datetime(year, month, 31)
risk_per_trade = 250.0
min_rr = 3.0
max_trades_per_day = 5
cooldown_bars = 4

print(f'Full backtest simulation for {symbol} {year}-{month:02d}')
print('=' * 60)

h4_candles = get_ohlcv(symbol, timeframe="H4", count=5000)
daily_candles = get_ohlcv(symbol, timeframe="D", count=1000)
weekly_candles = get_ohlcv(symbol, timeframe="W", count=200)

h4_df = pd.DataFrame(h4_candles)
h4_df['time'] = pd.to_datetime(h4_df['time'])

start_idx = h4_df[h4_df['time'] >= start_date].index.min()
end_idx = h4_df[h4_df['time'] <= end_date].index.max()
lookback_start = max(0, start_idx - 100)

h4_df = h4_df.iloc[lookback_start:end_idx + 1].reset_index(drop=True)
start_bar = min(100, start_idx - lookback_start)

daily_df = pd.DataFrame(daily_candles)
daily_df['time'] = pd.to_datetime(daily_df['time'])

weekly_df = pd.DataFrame(weekly_candles)
weekly_df['time'] = pd.to_datetime(weekly_df['time'])

print(f'Total bars: {len(h4_df)}, start_bar: {start_bar}')
print()

trades = []
in_trade = False
trade_entry = None
last_signal_bar = -cooldown_bars
daily_trades = {}

for i in range(start_bar, len(h4_df)):
    current_bar = h4_df.iloc[i]
    current_time = current_bar['time']
    current_date = current_time.date()
    
    if current_date not in daily_trades:
        daily_trades[current_date] = 0
    
    if in_trade:
        bar_high = current_bar['high']
        bar_low = current_bar['low']
        
        sl_hit = False
        tp_hit = False
        
        if trade_entry['direction'] == 'long':
            if bar_low <= trade_entry['stop_loss']:
                sl_hit = True
            if bar_high >= trade_entry['take_profit']:
                tp_hit = True
        else:
            if bar_high >= trade_entry['stop_loss']:
                sl_hit = True
            if bar_low <= trade_entry['take_profit']:
                tp_hit = True
        
        if sl_hit:
            trade_entry['exit_price'] = trade_entry['stop_loss']
            trade_entry['exit_type'] = 'SL'
            trade_entry['exit_time'] = current_time
            trade_entry['pnl'] = -risk_per_trade
            trade_entry['r_result'] = -1.0
            trades.append(trade_entry)
            print(f'  EXITED SL at bar {i} | {current_time} | P/L: -${risk_per_trade}')
            in_trade = False
            trade_entry = None
        elif tp_hit:
            trade_entry['exit_price'] = trade_entry['take_profit']
            trade_entry['exit_type'] = 'TP'
            trade_entry['exit_time'] = current_time
            trade_entry['pnl'] = risk_per_trade * trade_entry['r_multiple']
            trade_entry['r_result'] = trade_entry['r_multiple']
            trades.append(trade_entry)
            print(f'  EXITED TP at bar {i} | {current_time} | P/L: +${trade_entry["pnl"]:.0f}')
            in_trade = False
            trade_entry = None
        continue
    
    if i - last_signal_bar < cooldown_bars:
        continue
    
    if daily_trades[current_date] >= max_trades_per_day:
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
    
    if signal and signal.status in ('active', 'watching'):
        print(f'Bar {i} | {current_time} | ENTER {signal.direction.upper()} | Entry: {signal.entry:.5f} | SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f} | R:R: {signal.r_multiple}')
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

print()
print('=' * 60)
print(f'Completed trades: {len(trades)}')
print(f'Still in trade: {in_trade}')

if trade_entry and in_trade:
    print(f'Open trade: {trade_entry["direction"]} from {trade_entry["entry_time"]}')
    print(f'  Entry: {trade_entry["entry_price"]:.5f}')
    print(f'  SL: {trade_entry["stop_loss"]:.5f}')
    print(f'  TP: {trade_entry["take_profit"]:.5f}')
    last_bar = h4_df.iloc[-1]
    print(f'  Last bar price: {last_bar["close"]:.5f} (range: {last_bar["low"]:.5f} - {last_bar["high"]:.5f})')

if trades:
    total_pnl = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    print(f'\nTotal P/L: ${total_pnl:.0f}')
    print(f'Win rate: {len(wins)/len(trades)*100:.1f}%')
