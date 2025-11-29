"""Debug signal generation for V3 strategy."""

from datetime import datetime
from data import get_ohlcv
from strategy_v3 import generate_signal, identify_sr_zones, detect_bos, get_htf_bias

symbol = 'GBP_JPY'
print(f'Debugging signal generation for {symbol}')
print('=' * 60)

h4_candles = get_ohlcv(symbol, timeframe="H4", count=500)
daily_candles = get_ohlcv(symbol, timeframe="D", count=200)
weekly_candles = get_ohlcv(symbol, timeframe="W", count=52)

print(f'H4 candles: {len(h4_candles)}')
print(f'Daily candles: {len(daily_candles)}')
print(f'Weekly candles: {len(weekly_candles)}')

if len(h4_candles) < 50:
    print('ERROR: Not enough H4 candles')
    exit()

print()
print('Current price analysis:')
current_price = h4_candles[-1]['close']
current_time = h4_candles[-1]['time']
print(f'  Current price: {current_price:.5f}')
print(f'  Current time: {current_time}')

print()
print('HTF Bias:')
htf_bias = get_htf_bias(weekly_candles, daily_candles)
print(f'  Bias: {htf_bias}')

print()
print('S/R Zones (Daily):')
daily_zones = identify_sr_zones(daily_candles, tolerance_pct=0.003)
print(f'  Found {len(daily_zones)} zones')
for i, zone in enumerate(daily_zones[:5]):
    distance = abs(current_price - (zone[0] + zone[1])/2) / current_price * 100
    print(f'    Zone {i+1}: {zone[0]:.5f} - {zone[1]:.5f} (distance: {distance:.2f}%)')

print()
print('S/R Zones (Weekly):')
weekly_zones = identify_sr_zones(weekly_candles, tolerance_pct=0.005)
print(f'  Found {len(weekly_zones)} zones')
for i, zone in enumerate(weekly_zones[:5]):
    distance = abs(current_price - (zone[0] + zone[1])/2) / current_price * 100
    print(f'    Zone {i+1}: {zone[0]:.5f} - {zone[1]:.5f} (distance: {distance:.2f}%)')

print()
print('BOS Detection:')
bos_long = detect_bos(h4_candles, 'long', lookback=20)
bos_short = detect_bos(h4_candles, 'short', lookback=20)
print(f'  BOS Long: {bos_long if bos_long else "None"}')
print(f'  BOS Short: {bos_short if bos_short else "None"}')

print()
print('Signal Generation (min_confluence=3):')
signal = generate_signal(symbol, h4_candles, daily_candles, weekly_candles, min_rr=4.0, min_confluence=3)
if signal:
    print(f'  Signal: {signal.direction} {signal.status}')
    print(f'  Entry: {signal.entry:.5f}')
    print(f'  SL: {signal.stop_loss:.5f}')
    print(f'  TP: {signal.take_profit:.5f}')
    print(f'  R:R: {signal.r_multiple}')
    print(f'  Confluence: {signal.confluence_score}')
    print(f'  Reasoning: {signal.reasoning}')
else:
    print('  No signal generated')

print()
print('Signal Generation (min_confluence=2):')
signal2 = generate_signal(symbol, h4_candles, daily_candles, weekly_candles, min_rr=4.0, min_confluence=2)
if signal2:
    print(f'  Signal: {signal2.direction} {signal2.status}')
    print(f'  Entry: {signal2.entry:.5f}')
    print(f'  SL: {signal2.stop_loss:.5f}')
    print(f'  TP: {signal2.take_profit:.5f}')
    print(f'  R:R: {signal2.r_multiple}')
    print(f'  Confluence: {signal2.confluence_score}')
    print(f'  Reasoning: {signal2.reasoning}')
else:
    print('  No signal generated')

print()
print('Signal Generation (min_rr=2.0, min_confluence=2):')
signal3 = generate_signal(symbol, h4_candles, daily_candles, weekly_candles, min_rr=2.0, min_confluence=2)
if signal3:
    print(f'  Signal: {signal3.direction} {signal3.status}')
    print(f'  Entry: {signal3.entry:.5f}')
    print(f'  SL: {signal3.stop_loss:.5f}')
    print(f'  TP: {signal3.take_profit:.5f}')
    print(f'  R:R: {signal3.r_multiple}')
    print(f'  Confluence: {signal3.confluence_score}')
    print(f'  Reasoning: {signal3.reasoning}')
else:
    print('  No signal generated')
