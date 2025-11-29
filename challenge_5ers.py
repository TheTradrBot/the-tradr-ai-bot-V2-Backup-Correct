"""
5%ers 10K High Stakes Challenge Strategy
=========================================
Account: $10,000
Target: Pass Step 1 (8%) + Step 2 (5%) within 2 weeks

CHALLENGE RULES:
- Step 1 Profit Target: 8% ($800)
- Step 2 Profit Target: 5% ($500)
- Max Drawdown: 10% absolute (can't go below $9,000)
- Daily Drawdown: 5% of previous day's closing equity/balance (higher)
- Minimum 3 profitable trading days required
- Stop loss required within 3 minutes
- Leverage: Up to 1:100

RISK MANAGEMENT:
- Risk per trade: 1-2% ($100-$200)
- Max trades per day: 3 (to control daily DD)
- Prefer 6:1 R:R for faster target achievement
"""

import os
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import discord

OANDA_API_KEY = os.environ.get('OANDA_API_KEY', '')
BASE_URL = 'https://api-fxpractice.oanda.com'
HEADERS = {'Authorization': f'Bearer {OANDA_API_KEY}', 'Content-Type': 'application/json'}

CHALLENGE_CONFIG = {
    'account_size': 10000,
    'step1_target_pct': 8.0,
    'step2_target_pct': 5.0,
    'max_drawdown_pct': 10.0,
    'daily_drawdown_pct': 5.0,
    'risk_per_trade_pct': 1.5,
    'max_trades_per_day': 3,
    'min_profitable_days': 3,
    'leverage': 100,
}

ASSET_CONFIGS = {
    'EUR_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 10.0, 'pip_size': 0.0001},
    'GBP_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 10.0, 'pip_size': 0.0001},
    'USD_JPY': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 9.0, 'pip_size': 0.01},
    'USD_CHF': {'tp_rr': 4.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 11.0, 'pip_size': 0.0001},
    'AUD_USD': {'tp_rr': 4.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 10.0, 'pip_size': 0.0001},
    'NZD_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 10.0, 'pip_size': 0.0001},
    'USD_CAD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 7.5, 'pip_size': 0.0001},
    'EUR_GBP': {'tp_rr': 5.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 12.5, 'pip_size': 0.0001},
    'EUR_JPY': {'tp_rr': 5.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 9.0, 'pip_size': 0.01},
    'GBP_JPY': {'tp_rr': 4.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 9.0, 'pip_size': 0.01},
    'XAU_USD': {'tp_rr': 5.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 1.0, 'pip_size': 0.01},
    'BTC_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 1.0, 'pip_size': 1.0},
    'ETH_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 1.0, 'pip_size': 0.01},
    'SPX500_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 1.0, 'pip_size': 0.1},
    'NAS100_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 1.0, 'pip_size': 0.1},
}


@dataclass
class ChallengeState:
    starting_balance: float = 10000.0
    current_balance: float = 10000.0
    current_equity: float = 10000.0
    daily_starting_equity: float = 10000.0
    step: int = 1
    profitable_days: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    trades_today: int = 0
    daily_pnl: float = 0.0
    current_date: str = ""
    passed_step1: bool = False
    passed_step2: bool = False
    failed: bool = False
    fail_reason: str = ""
    step1_days: int = 0
    step2_days: int = 0


@dataclass
class TradeSignal:
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    lot_size: float
    risk_usd: float
    potential_profit: float
    risk_reward: float
    confluence: List[str]
    timestamp: str


def fetch_candles(symbol: str, count: int = 2000, granularity: str = 'H4', 
                  from_time: str = None, to_time: str = None) -> List[Dict]:
    url = f'{BASE_URL}/v3/instruments/{symbol}/candles'
    params = {'granularity': granularity, 'price': 'MBA'}
    
    if from_time and to_time:
        params['from'] = from_time
        params['to'] = to_time
    else:
        params['count'] = count
    
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            return [{'time': c.get('time'), 
                     'open': float(c.get('mid', {}).get('o', 0)),
                     'high': float(c.get('mid', {}).get('h', 0)),
                     'low': float(c.get('mid', {}).get('l', 0)),
                     'close': float(c.get('mid', {}).get('c', 0))}
                    for c in data.get('candles', []) if c.get('complete')]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
    return []


def calculate_ema(closes: List[float], period: int) -> List[float]:
    if len(closes) < period:
        return []
    ema = [sum(closes[:period]) / period]
    mult = 2 / (period + 1)
    for i in range(period, len(closes)):
        ema.append(closes[i] * mult + ema[-1] * (1 - mult))
    return ema


def calculate_atr(candles: List[Dict], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0001
    trs = []
    for i in range(1, len(candles)):
        tr = max(candles[i]['high'] - candles[i]['low'],
                 abs(candles[i]['high'] - candles[i-1]['close']),
                 abs(candles[i]['low'] - candles[i-1]['close']))
        trs.append(tr)
    return sum(trs[-period:]) / period


def find_order_blocks(candles: List[Dict]) -> List[Dict]:
    obs = []
    for i in range(60, len(candles) - 1):
        curr, next_bar = candles[i], candles[i + 1]
        curr_body = abs(curr['close'] - curr['open'])
        next_body = abs(next_bar['close'] - next_bar['open'])
        if next_body > curr_body * 1.5:
            if next_bar['close'] < next_bar['open'] and curr['close'] > curr['open']:
                obs.append({'type': 'bearish', 'high': curr['high'], 'low': curr['low'], 'idx': i})
            elif next_bar['close'] > next_bar['open'] and curr['close'] < curr['open']:
                obs.append({'type': 'bullish', 'high': curr['high'], 'low': curr['low'], 'idx': i})
    return obs


def find_fair_value_gaps(candles: List[Dict]) -> List[Dict]:
    fvgs = []
    for i in range(2, len(candles)):
        c1, c3 = candles[i - 2], candles[i]
        if c3['low'] > c1['high']:
            fvgs.append({'type': 'bullish', 'high': c3['low'], 'low': c1['high'], 'idx': i})
        elif c3['high'] < c1['low']:
            fvgs.append({'type': 'bearish', 'high': c1['low'], 'low': c3['high'], 'idx': i})
    return fvgs


def find_liquidity_sweeps(candles: List[Dict], lookback: int = 30) -> List[Dict]:
    sweeps = []
    for i in range(lookback + 1, len(candles)):
        curr = candles[i]
        highs = [candles[j]['high'] for j in range(i - lookback, i)]
        lows = [candles[j]['low'] for j in range(i - lookback, i)]
        if curr['high'] > max(highs) and curr['close'] < max(highs):
            sweeps.append({'idx': i, 'dir': 'bearish'})
        elif curr['low'] < min(lows) and curr['close'] > min(lows):
            sweeps.append({'idx': i, 'dir': 'bullish'})
    return sweeps


def calculate_lot_size(risk_usd: float, sl_pips: float, pip_value: float) -> float:
    """Calculate lot size based on risk and stop loss distance."""
    if sl_pips <= 0 or pip_value <= 0:
        return 0.01
    lot_size = risk_usd / (sl_pips * pip_value)
    lot_size = max(0.01, min(lot_size, 10.0))
    return round(lot_size, 2)


def calculate_sl_pips(entry: float, sl: float, pip_size: float) -> float:
    """Calculate stop loss distance in pips."""
    return abs(entry - sl) / pip_size


def check_daily_drawdown(state: ChallengeState, potential_loss: float) -> bool:
    """Check if trade would violate daily drawdown limit."""
    max_daily_loss = state.daily_starting_equity * (CHALLENGE_CONFIG['daily_drawdown_pct'] / 100)
    return (state.daily_pnl - potential_loss) >= -max_daily_loss


def check_max_drawdown(state: ChallengeState, potential_loss: float) -> bool:
    """Check if trade would violate max drawdown limit."""
    min_equity = state.starting_balance * (1 - CHALLENGE_CONFIG['max_drawdown_pct'] / 100)
    return (state.current_balance - potential_loss) >= min_equity


def run_challenge_backtest(month: int, year: int) -> Dict:
    """Run backtest for a specific month simulating the 5%ers challenge."""
    
    target_month = f"{year}-{month:02d}"
    
    all_signals = []
    
    for symbol, config in ASSET_CONFIGS.items():
        candles = fetch_candles(symbol, count=2000, granularity='H4')
        if len(candles) < 200:
            continue
        
        tp_rr = config['tp_rr']
        atr_mult = config['atr_mult']
        ema_period = config['ema']
        pip_value = config['pip_value']
        pip_size = config['pip_size']
        
        closes = [c['close'] for c in candles]
        ema = calculate_ema(closes, ema_period)
        obs = find_order_blocks(candles)
        fvgs = find_fair_value_gaps(candles)
        sweeps = find_liquidity_sweeps(candles)
        
        used_obs, used_fvgs = set(), set()
        
        for i in range(150, len(candles) - 1):
            candle_month = candles[i]['time'][:7]
            if candle_month != target_month:
                continue
            
            price = candles[i]['close']
            ema_idx = i - ema_period
            if ema_idx < 0 or ema_idx >= len(ema):
                continue
            
            trend = 'LONG' if price > ema[ema_idx] else 'SHORT'
            trend_type = 'bullish' if trend == 'LONG' else 'bearish'
            
            active_obs = [ob for ob in obs if ob['idx'] < i and ob['idx'] not in used_obs and 
                          i - ob['idx'] < 100 and ob['type'] == trend_type]
            active_fvgs = [fvg for fvg in fvgs if fvg['idx'] < i and fvg['idx'] not in used_fvgs and 
                           i - fvg['idx'] < 60 and fvg['type'] == trend_type]
            curr_sweeps = [s for s in sweeps if s['idx'] == i and s['dir'] == trend_type]
            
            in_ob = any(ob['low'] <= price <= ob['high'] for ob in active_obs)
            in_fvg = any(fvg['low'] <= price <= fvg['high'] for fvg in active_fvgs)
            has_sweep = bool(curr_sweeps)
            
            if not (in_ob or in_fvg or has_sweep):
                continue
            
            confluence = []
            if in_ob: confluence.append('OB')
            if in_fvg: confluence.append('FVG')
            if has_sweep: confluence.append('SWEEP')
            
            for ob in active_obs:
                if ob['low'] <= price <= ob['high']:
                    used_obs.add(ob['idx'])
                    break
            for fvg in active_fvgs:
                if fvg['low'] <= price <= fvg['high']:
                    used_fvgs.add(fvg['idx'])
                    break
            
            entry = price
            atr = calculate_atr(candles[max(0, i-30):i+1])
            
            if trend == 'LONG':
                sl = entry - atr * atr_mult
                tp = entry + (entry - sl) * tp_rr
            else:
                sl = entry + atr * atr_mult
                tp = entry - (sl - entry) * tp_rr
            
            exit_reason = 'OPEN'
            for j in range(i + 1, min(i + 80, len(candles))):
                bar = candles[j]
                if trend == 'LONG':
                    if bar['low'] <= sl:
                        exit_reason = 'SL'
                        break
                    if bar['high'] >= tp:
                        exit_reason = 'TP'
                        break
                else:
                    if bar['high'] >= sl:
                        exit_reason = 'SL'
                        break
                    if bar['low'] <= tp:
                        exit_reason = 'TP'
                        break
            
            if exit_reason == 'OPEN':
                continue
            
            sl_pips = calculate_sl_pips(entry, sl, pip_size)
            
            all_signals.append({
                'time': candles[i]['time'],
                'date': candles[i]['time'][:10],
                'symbol': symbol,
                'direction': trend,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'tp_rr': tp_rr,
                'sl_pips': sl_pips,
                'pip_value': pip_value,
                'result': exit_reason,
                'confluence': confluence
            })
    
    all_signals.sort(key=lambda x: x['time'])
    
    state = ChallengeState()
    all_trades = []
    daily_results = {}
    processed_dates = set()
    step1_pass_date = None
    step2_pass_date = None
    
    for sig in all_signals:
        if state.failed:
            break
        if state.passed_step2:
            break
        
        candle_date = sig['date']
        
        if candle_date not in processed_dates:
            if state.current_date and state.daily_pnl > 0:
                state.profitable_days += 1
            
            if state.current_date:
                daily_results[state.current_date] = {
                    'pnl': state.daily_pnl,
                    'trades': state.trades_today,
                    'balance': state.current_balance
                }
            
            processed_dates.add(candle_date)
            state.current_date = candle_date
            state.daily_starting_equity = max(state.current_balance, state.current_equity)
            state.daily_pnl = 0.0
            state.trades_today = 0
        
        if state.trades_today >= CHALLENGE_CONFIG['max_trades_per_day']:
            continue
        
        risk_usd = state.current_balance * (CHALLENGE_CONFIG['risk_per_trade_pct'] / 100)
        lot_size = calculate_lot_size(risk_usd, sig['sl_pips'], sig['pip_value'])
        potential_profit = risk_usd * sig['tp_rr']
        
        if not check_daily_drawdown(state, risk_usd):
            continue
        if not check_max_drawdown(state, risk_usd):
            continue
        
        if sig['result'] == 'TP':
            pnl = potential_profit
        else:
            pnl = -risk_usd
        
        state.current_balance += pnl
        state.current_equity = state.current_balance
        state.daily_pnl += pnl
        state.trades_today += 1
        state.total_trades += 1
        if pnl > 0:
            state.winning_trades += 1
        
        trade = {
            'trade_num': state.total_trades,
            'symbol': sig['symbol'],
            'date': candle_date,
            'time': sig['time'],
            'direction': sig['direction'],
            'entry': round(sig['entry'], 5),
            'stop_loss': round(sig['sl'], 5),
            'take_profit': round(sig['tp'], 5),
            'lot_size': lot_size,
            'risk_usd': round(risk_usd, 2),
            'result': sig['result'],
            'pnl': round(pnl, 2),
            'balance': round(state.current_balance, 2),
            'confluence': '+'.join(sig['confluence']),
            'step': state.step
        }
        all_trades.append(trade)
        
        min_equity = 10000 * (1 - CHALLENGE_CONFIG['max_drawdown_pct'] / 100)
        if state.current_balance < min_equity:
            state.failed = True
            state.fail_reason = f"Max drawdown hit (balance ${state.current_balance:.2f} < ${min_equity:.2f})"
            break
        
        max_daily_loss = state.daily_starting_equity * (CHALLENGE_CONFIG['daily_drawdown_pct'] / 100)
        if state.daily_pnl < -max_daily_loss:
            state.failed = True
            state.fail_reason = f"Daily drawdown hit (${state.daily_pnl:.2f} loss > ${max_daily_loss:.2f} limit)"
            break
        
        if state.step == 1 and not state.passed_step1:
            target = 10000 * (1 + CHALLENGE_CONFIG['step1_target_pct'] / 100)
            if state.current_balance >= target and state.profitable_days >= CHALLENGE_CONFIG['min_profitable_days']:
                state.passed_step1 = True
                state.step = 2
                step1_pass_date = candle_date
                state.starting_balance = state.current_balance
        elif state.step == 2 and not state.passed_step2:
            target = state.starting_balance * (1 + CHALLENGE_CONFIG['step2_target_pct'] / 100)
            if state.current_balance >= target:
                state.passed_step2 = True
                step2_pass_date = candle_date
    
    if state.current_date and state.daily_pnl > 0:
        state.profitable_days += 1
    if state.current_date:
        daily_results[state.current_date] = {
            'pnl': state.daily_pnl,
            'trades': state.trades_today,
            'balance': state.current_balance
        }
    
    trading_days = len(processed_dates)
    if step1_pass_date:
        step1_days = len([d for d in processed_dates if d <= step1_pass_date])
    else:
        step1_days = trading_days
    
    if step2_pass_date and step1_pass_date:
        step2_days = len([d for d in processed_dates if step1_pass_date < d <= step2_pass_date])
    else:
        step2_days = 0
    
    total_pnl = state.current_balance - 10000
    win_rate = (state.winning_trades / state.total_trades * 100) if state.total_trades > 0 else 0
    
    return {
        'month': month,
        'year': year,
        'passed_step1': state.passed_step1,
        'passed_step2': state.passed_step2,
        'failed': state.failed,
        'fail_reason': state.fail_reason,
        'step1_days': step1_days,
        'step2_days': step2_days,
        'total_days': trading_days,
        'total_trades': state.total_trades,
        'winning_trades': state.winning_trades,
        'win_rate': win_rate,
        'profitable_days': state.profitable_days,
        'final_balance': state.current_balance,
        'total_pnl': total_pnl,
        'return_pct': (total_pnl / 10000) * 100,
        'trades': all_trades,
        'daily_results': daily_results
    }


def format_challenge_result(result: Dict) -> discord.Embed:
    """Format challenge backtest result as Discord embed."""
    
    month_name = datetime(result['year'], result['month'], 1).strftime('%B %Y')
    
    if result['failed']:
        color = discord.Color.red()
        status = "FAILED"
        title = f"5%ers Challenge - {month_name}"
    elif result['passed_step2']:
        color = discord.Color.green()
        status = "PASSED BOTH STEPS"
        title = f"5%ers Challenge PASSED - {month_name}"
    elif result['passed_step1']:
        color = discord.Color.gold()
        status = "PASSED STEP 1 ONLY"
        title = f"5%ers Challenge - {month_name}"
    else:
        color = discord.Color.orange()
        status = "IN PROGRESS"
        title = f"5%ers Challenge - {month_name}"
    
    embed = discord.Embed(title=title, color=color)
    
    embed.add_field(name="Status", value=status, inline=True)
    embed.add_field(name="Account", value="$10,000", inline=True)
    embed.add_field(name="Final Balance", value=f"${result['final_balance']:,.2f}", inline=True)
    
    if result['passed_step1']:
        embed.add_field(name="Step 1 (8%)", value=f"PASSED in {result['step1_days']} days", inline=True)
    else:
        embed.add_field(name="Step 1 (8%)", value=f"Not passed ({result['step1_days']} days)", inline=True)
    
    if result['passed_step2']:
        embed.add_field(name="Step 2 (5%)", value=f"PASSED in {result['step2_days']} days", inline=True)
    else:
        embed.add_field(name="Step 2 (5%)", value=f"Not passed", inline=True)
    
    total_days = result['step1_days'] + result['step2_days']
    embed.add_field(name="Total Days", value=f"{total_days}", inline=True)
    
    embed.add_field(name="Total Trades", value=str(result['total_trades']), inline=True)
    embed.add_field(name="Win Rate", value=f"{result['win_rate']:.1f}%", inline=True)
    embed.add_field(name="Profitable Days", value=str(result['profitable_days']), inline=True)
    
    pnl_str = f"+${result['total_pnl']:,.2f}" if result['total_pnl'] >= 0 else f"-${abs(result['total_pnl']):,.2f}"
    embed.add_field(name="Total P/L", value=pnl_str, inline=True)
    embed.add_field(name="Return", value=f"{result['return_pct']:+.1f}%", inline=True)
    
    if result['failed']:
        embed.add_field(name="Fail Reason", value=result['fail_reason'], inline=False)
    
    if result['trades']:
        recent = result['trades'][-5:]
        trades_str = ""
        for t in recent:
            pnl_str = f"+${t['pnl']:.0f}" if t['pnl'] > 0 else f"-${abs(t['pnl']):.0f}"
            trades_str += f"{t['symbol']} {t['direction']} {t['lot_size']}L | {t['result']} {pnl_str}\n"
        embed.add_field(name="Recent Trades", value=f"```{trades_str}```", inline=False)
    
    return embed


def export_challenge_trades_csv(result: Dict, filename: str = None) -> str:
    """Export challenge trades to CSV file."""
    import csv
    
    if filename is None:
        filename = f"challenge_{result['month']:02d}_{result['year']}_trades.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Trade#', 'Date', 'Symbol', 'Direction', 'Entry', 'Stop Loss', 'Take Profit', 
                        'Lot Size', 'Risk ($)', 'Result', 'P/L ($)', 'Balance', 'Confluence', 'Step'])
        
        for t in result['trades']:
            writer.writerow([
                t['trade_num'], t['date'], t['symbol'], t['direction'], t['entry'],
                t['stop_loss'], t['take_profit'], t['lot_size'], t['risk_usd'],
                t['result'], t['pnl'], t['balance'], t['confluence'], t['step']
            ])
    
    return filename


if __name__ == "__main__":
    print("Testing 5%ers Challenge Backtest...")
    result = run_challenge_backtest(11, 2024)
    
    print(f"\n{'='*60}")
    print(f"5%ERS 10K CHALLENGE - November 2024")
    print(f"{'='*60}")
    print(f"Passed Step 1: {'YES' if result['passed_step1'] else 'NO'} ({result['step1_days']} days)")
    print(f"Passed Step 2: {'YES' if result['passed_step2'] else 'NO'} ({result['step2_days']} days)")
    print(f"Failed: {'YES - ' + result['fail_reason'] if result['failed'] else 'NO'}")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Profitable Days: {result['profitable_days']}")
    print(f"Final Balance: ${result['final_balance']:,.2f}")
    print(f"Total P/L: ${result['total_pnl']:+,.2f} ({result['return_pct']:+.1f}%)")
    
    filename = export_challenge_trades_csv(result)
    print(f"\nTrades exported to: {filename}")
