"""
5%ers Challenge Backtest Runner - Archer Academy Strategy

Tests the V4 Archer strategy across all assets for the 5%ers challenge.
"""

import os
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from data import get_ohlcv
from strategy_v4_archer import (
    generate_archer_signal,
    backtest_archer_strategy,
    calculate_backtest_stats
)
from challenge_simulator import simulate_challenge, run_multi_month_backtest


def convert_candles_to_dict(candles: List) -> List[Dict]:
    """Convert candle datetime objects to string format."""
    result = []
    for c in candles:
        result.append({
            'time': c['time'].strftime('%Y-%m-%dT%H:%M:%S') if hasattr(c['time'], 'strftime') else str(c['time']),
            'open': c['open'],
            'high': c['high'],
            'low': c['low'],
            'close': c['close'],
            'volume': c.get('volume', 0)
        })
    return result

ASSETS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'NZD_USD', 'USD_CAD',
    'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
    'XAU_USD', 'XAG_USD',
    'NAS100_USD', 'SPX500_USD',
    'WTICO_USD',
    'BTC_USD', 'ETH_USD',
]

ASSET_PARAMS = {
    'EUR_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'GBP_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'USD_JPY': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'USD_CHF': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'AUD_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'NZD_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'USD_CAD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'EUR_GBP': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'EUR_JPY': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'GBP_JPY': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'XAU_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'XAG_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'NAS100_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'SPX500_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'WTICO_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'BTC_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
    'ETH_USD': {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 3, 'max_daily': 1},
}


def fetch_data_for_backtest(symbol: str, count: int = 5000) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Fetch H4, Daily, and Weekly data for backtesting."""
    h4_raw = get_ohlcv(symbol, timeframe='H4', count=count)
    daily_raw = get_ohlcv(symbol, timeframe='D', count=count // 6)
    weekly_raw = get_ohlcv(symbol, timeframe='W', count=count // 30)
    
    h4 = convert_candles_to_dict(h4_raw)
    daily = convert_candles_to_dict(daily_raw)
    weekly = convert_candles_to_dict(weekly_raw)
    
    return h4, daily, weekly


def filter_candles_by_year(candles: List[Dict], year: int) -> List[Dict]:
    """Filter candles to only include those from a specific year."""
    return [c for c in candles if c['time'].startswith(str(year))]


def run_archer_backtest_for_asset(
    symbol: str,
    year: int,
    params: Dict = None
) -> Dict:
    """Run Archer backtest for a single asset."""
    if params is None:
        params = ASSET_PARAMS.get(symbol, {'min_rr': 2.0, 'min_confluence': 2, 'cooldown': 4, 'max_daily': 2})
    
    h4, daily, weekly = fetch_data_for_backtest(symbol)
    
    if not h4 or not daily:
        return {'symbol': symbol, 'error': 'No data available'}
    
    h4_year = [c for c in h4 if c['time'].startswith(str(year))]
    daily_year = [c for c in daily if c['time'].startswith(str(year))]
    
    if len(h4_year) < 100:
        return {'symbol': symbol, 'error': f'Insufficient data for {year}'}
    
    trades = backtest_archer_strategy(
        h4_candles=h4_year,
        daily_candles=daily_year,
        weekly_candles=weekly,
        min_rr=params['min_rr'],
        min_confluence=params['min_confluence'],
        cooldown_bars=params['cooldown'],
        max_daily_trades=params['max_daily']
    )
    
    stats = calculate_backtest_stats(trades)
    
    challenge_trades = []
    for t in trades:
        challenge_trades.append({
            'entry_time': t['entry_time'],
            'exit_time': t['exit_time'],
            'pnl': t['pnl_usd'],
            'result': t['result']
        })
    
    monthly_results = []
    months_with_trades = {}
    for t in challenge_trades:
        if t['entry_time']:
            month_key = t['entry_time'][:7]
            if month_key not in months_with_trades:
                months_with_trades[month_key] = []
            months_with_trades[month_key].append(t)
    
    for month_key, month_trades in sorted(months_with_trades.items()):
        month_pnl = sum(t['pnl'] for t in month_trades)
        profitable_days = set()
        for t in month_trades:
            if t['pnl'] > 0 and t['exit_time']:
                profitable_days.add(t['exit_time'][:10])
        
        step1_passed = month_pnl >= 800 and len(profitable_days) >= 3
        passed = step1_passed
        
        monthly_results.append({
            'month': month_key,
            'pnl': month_pnl,
            'trades': len(month_trades),
            'profitable_days': len(profitable_days),
            'passed': passed
        })
    
    return {
        'symbol': symbol,
        'year': year,
        'trades': trades,
        'stats': stats,
        'challenge': {'monthly_results': monthly_results}
    }


def run_all_assets_backtest(year: int) -> List[Dict]:
    """Run backtest for all assets."""
    results = []
    
    for symbol in ASSETS:
        print(f"[backtest] Running {symbol} {year}...")
        params = ASSET_PARAMS.get(symbol)
        result = run_archer_backtest_for_asset(symbol, year, params)
        
        if 'error' in result:
            print(f"  {symbol}: {result['error']}")
        else:
            challenge = result['challenge']
            total_pnl = result['stats']['total_pnl']
            months_passed = sum(1 for m in challenge.get('monthly_results', []) if m.get('passed'))
            total_months = len(challenge.get('monthly_results', []))
            print(f"  {symbol}: {months_passed}/{total_months} passed, ${total_pnl:,.0f} total P&L, {result['stats']['win_rate']:.1f}% win rate")
        
        results.append(result)
    
    return results


def generate_backtest_summary(results: List[Dict]) -> str:
    """Generate a summary of backtest results."""
    lines = []
    lines.append("=" * 60)
    lines.append("5%ERS CHALLENGE BACKTEST SUMMARY - ARCHER STRATEGY")
    lines.append("=" * 60)
    lines.append("")
    
    total_passed = 0
    total_months = 0
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    
    for result in results:
        if 'error' in result:
            continue
        
        symbol = result['symbol']
        stats = result['stats']
        challenge = result['challenge']
        
        months_passed = sum(1 for m in challenge.get('monthly_results', []) if m.get('passed'))
        months_total = len(challenge.get('monthly_results', []))
        
        total_passed += months_passed
        total_months += months_total
        total_pnl += stats['total_pnl']
        total_trades += stats['total_trades']
        total_wins += stats['wins']
        
        lines.append(f"{symbol:12} | {months_passed}/{months_total:2}    | ${stats['total_pnl']:>9,.0f} | {stats['win_rate']:.1f}%")
    
    lines.append("")
    lines.append("-" * 60)
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    lines.append(f"TOTAL: {total_passed}/{total_months} months passed")
    lines.append(f"TOTAL P&L: ${total_pnl:,.0f}")
    lines.append(f"TOTAL TRADES: {total_trades}")
    lines.append(f"OVERALL WIN RATE: {overall_wr:.1f}%")
    lines.append(f"AVG MONTHLY: ${total_pnl / max(total_months, 1):,.0f}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    
    print(f"Running Archer Strategy Backtest for {year}...")
    print("=" * 60)
    
    results = run_all_assets_backtest(year)
    summary = generate_backtest_summary(results)
    print(summary)
