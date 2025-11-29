"""
5%ers 10K High Stakes Challenge - V3 Strategy Backtest
=======================================================

Uses strategy_v3.py (HTF S/R + BOS + Structural TPs)
NO RSI, NO SMC, NO Fibonacci TPs

CHALLENGE RULES:
- Account: $10,000
- Step 1: 8% profit target ($800)
- Step 2: 5% profit target ($500)
- Max Drawdown: 10% (cannot go below $9,000)
- Daily Drawdown: 5% of previous day's balance
- Minimum 3 profitable days
- Risk per trade: $250 (2.5%)
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from data import get_ohlcv
from strategy_v3 import backtest_v3, calculate_backtest_stats
from challenge_simulator import simulate_challenge, CHALLENGE_RULES

BACKTEST_ASSETS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'NZD_USD', 'USD_CAD',
    'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
    'XAU_USD', 'XAG_USD',
    'NAS100_USD', 'SPX500_USD',
    'WTICO_USD',
    'BTC_USD', 'ETH_USD',
]

ASSET_PARAMS = {
    'EUR_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'GBP_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'USD_JPY': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'USD_CHF': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'AUD_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'NZD_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'USD_CAD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'EUR_GBP': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'EUR_JPY': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'GBP_JPY': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'XAU_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'XAG_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'NAS100_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'SPX500_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'WTICO_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'BTC_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
    'ETH_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2},
}


def fetch_data_for_backtest(symbol: str, count: int = 5000) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Fetch H4, Daily, and Weekly data for backtesting."""
    h4_candles = get_ohlcv(symbol, timeframe="H4", count=count)
    daily_candles = get_ohlcv(symbol, timeframe="D", count=min(count // 6, 1000))
    weekly_candles = get_ohlcv(symbol, timeframe="W", count=min(count // 30, 200))
    
    return h4_candles, daily_candles, weekly_candles


def run_backtest(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    risk_per_trade: float = 250.0
) -> Dict:
    """
    Run V3 strategy backtest for a single asset.
    
    Returns dict with trades and stats.
    """
    params = ASSET_PARAMS.get(symbol, {
        'min_rr': 4.0,
        'min_confluence': 3,
        'cooldown': 6,
        'max_daily': 3
    })
    
    h4_candles, daily_candles, weekly_candles = fetch_data_for_backtest(symbol)
    
    if len(h4_candles) < 100:
        return {'error': f'Insufficient H4 data for {symbol}', 'trades': [], 'stats': {}}
    
    trades = backtest_v3(
        symbol=symbol,
        h4_candles=h4_candles,
        daily_candles=daily_candles,
        weekly_candles=weekly_candles,
        start_date=start_date,
        end_date=end_date,
        risk_per_trade=risk_per_trade,
        min_rr=params['min_rr'],
        min_confluence=params['min_confluence'],
        max_trades_per_day=params['max_daily'],
        cooldown_bars=params['cooldown']
    )
    
    stats = calculate_backtest_stats(trades)
    
    return {
        'symbol': symbol,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'trades': trades,
        'stats': stats,
    }


def run_challenge_backtest(
    symbol: str,
    year: int,
    month: Optional[int] = None
) -> Dict:
    """
    Run backtest and simulate 5%ers challenge.
    
    If month is None, runs for full year.
    Returns challenge simulation results.
    """
    if month:
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    else:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
    
    backtest_result = run_backtest(symbol, start_date, end_date)
    
    if 'error' in backtest_result:
        return backtest_result
    
    challenge_trades = []
    for trade in backtest_result['trades']:
        entry_time = trade.get('entry_time')
        if hasattr(entry_time, 'isoformat'):
            entry_time_str = entry_time.isoformat()
        else:
            entry_time_str = str(entry_time)
        
        challenge_trades.append({
            'entry_time': entry_time_str,
            'pnl_r': trade.get('r_result', 0),
            'direction': trade.get('direction', 'long'),
        })
    
    challenge_result = simulate_challenge(challenge_trades)
    
    return {
        'symbol': symbol,
        'year': year,
        'month': month,
        'backtest': backtest_result['stats'],
        'trades': backtest_result['trades'],
        'challenge': challenge_result,
    }


def run_monthly_challenges(symbol: str, year: int) -> Dict:
    """Run challenge simulation for each month of the year."""
    monthly_results = {}
    
    for month in range(1, 13):
        result = run_challenge_backtest(symbol, year, month)
        month_key = f"{year}-{month:02d}"
        monthly_results[month_key] = result
    
    passed_months = sum(
        1 for r in monthly_results.values() 
        if r.get('challenge', {}).get('passed', False)
    )
    
    step1_months = sum(
        1 for r in monthly_results.values()
        if r.get('challenge', {}).get('step1_passed', False)
    )
    
    total_pnl = sum(
        r.get('challenge', {}).get('total_pnl', 0) 
        for r in monthly_results.values()
    )
    
    return {
        'symbol': symbol,
        'year': year,
        'monthly_results': monthly_results,
        'summary': {
            'total_months': 12,
            'passed_months': passed_months,
            'step1_months': step1_months,
            'total_pnl': total_pnl,
            'avg_monthly_pnl': total_pnl / 12,
        }
    }


def run_all_assets_backtest(year: int) -> Dict:
    """Run backtests for all assets for a year."""
    results = {}
    
    for symbol in BACKTEST_ASSETS:
        print(f"[backtest] Running {symbol} {year}...")
        try:
            result = run_monthly_challenges(symbol, year)
            results[symbol] = result
            
            summary = result.get('summary', {})
            print(f"  {symbol}: {summary.get('passed_months', 0)}/12 passed, "
                  f"${summary.get('total_pnl', 0):,.0f} total P&L")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")
            results[symbol] = {'error': str(e)}
    
    return results


def generate_backtest_summary(results: Dict) -> str:
    """Generate text summary of backtest results."""
    lines = []
    lines.append("=" * 60)
    lines.append("5%ERS CHALLENGE BACKTEST SUMMARY - V3 STRATEGY")
    lines.append("=" * 60)
    lines.append("")
    
    total_passed = 0
    total_months = 0
    total_pnl = 0
    
    for symbol, result in results.items():
        if 'error' in result:
            lines.append(f"{symbol}: ERROR - {result['error']}")
            continue
        
        summary = result.get('summary', {})
        passed = summary.get('passed_months', 0)
        pnl = summary.get('total_pnl', 0)
        
        total_passed += passed
        total_months += 12
        total_pnl += pnl
        
        status = "PASS ALL" if passed == 12 else f"{passed}/12"
        lines.append(f"{symbol:12} | {status:10} | ${pnl:>10,.0f}")
    
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"TOTAL: {total_passed}/{total_months} months passed")
    lines.append(f"TOTAL P&L: ${total_pnl:,.0f}")
    lines.append(f"AVG MONTHLY: ${total_pnl / max(total_months, 1):,.0f}")
    
    return "\n".join(lines)


def format_challenge_result_for_discord(result: Dict) -> str:
    """Format challenge result for Discord display."""
    if 'error' in result:
        return f"Error: {result['error']}"
    
    symbol = result.get('symbol', 'Unknown')
    year = result.get('year', '')
    month = result.get('month', '')
    
    challenge = result.get('challenge', {})
    backtest = result.get('backtest', {})
    
    passed = challenge.get('passed', False)
    step1 = challenge.get('step1_passed', False)
    step2 = challenge.get('step2_passed', False)
    blown = challenge.get('blown', False)
    
    final_balance = challenge.get('final_balance', 10000)
    total_pnl = challenge.get('total_pnl', 0)
    total_trades = challenge.get('total_trades', 0)
    win_rate = challenge.get('win_rate', 0)
    profitable_days = challenge.get('profitable_days', 0)
    
    period = f"{year}" if not month else f"{year}-{month:02d}"
    
    status = "PASSED" if passed else "BLOWN" if blown else "INCOMPLETE"
    
    lines = [
        f"**{symbol} - {period}**",
        f"Status: **{status}**",
        f"",
        f"Step 1 (8%): {'Passed' if step1 else 'Not Passed'}",
        f"Step 2 (5%): {'Passed' if step2 else 'Not Passed'}",
        f"",
        f"Final Balance: ${final_balance:,.0f}",
        f"Total P&L: ${total_pnl:,.0f}",
        f"Total Trades: {total_trades}",
        f"Win Rate: {win_rate:.1f}%",
        f"Profitable Days: {profitable_days}",
    ]
    
    if blown:
        lines.append(f"Blown Reason: {challenge.get('blown_reason', 'Unknown')}")
    
    return "\n".join(lines)


def export_trades_to_csv(trades: List[Dict], filepath: str) -> str:
    """Export trades to CSV file."""
    import csv
    
    fieldnames = [
        'symbol', 'direction', 'entry_time', 'exit_time',
        'entry_price', 'exit_price', 'stop_loss', 'take_profit',
        'r_multiple', 'r_result', 'pnl', 'exit_type', 'reasoning'
    ]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for trade in trades:
            row = {
                'symbol': trade.get('symbol', ''),
                'direction': trade.get('direction', ''),
                'entry_time': trade.get('entry_time', ''),
                'exit_time': trade.get('exit_time', ''),
                'entry_price': trade.get('entry_price', ''),
                'exit_price': trade.get('exit_price', ''),
                'stop_loss': trade.get('stop_loss', ''),
                'take_profit': trade.get('take_profit', ''),
                'r_multiple': trade.get('r_multiple', ''),
                'r_result': trade.get('r_result', ''),
                'pnl': trade.get('pnl', ''),
                'exit_type': trade.get('exit_type', ''),
                'reasoning': trade.get('reasoning', ''),
            }
            writer.writerow(row)
    
    return filepath


if __name__ == '__main__':
    print("Running V3 Strategy Backtest...")
    print("")
    
    results = run_all_assets_backtest(2024)
    
    print("")
    print(generate_backtest_summary(results))
