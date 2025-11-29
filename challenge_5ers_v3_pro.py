"""
5%ers Challenge Backtest Runner for V3 Pro Strategy

Tests the V3 Pro (Golden Pocket + Wyckoff) strategy against
5%ers High Stakes 10K Challenge rules.

Target: 70%+ yearly return per asset, pass both steps every month
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import calendar
from data import get_ohlcv
from strategy_v3_pro import backtest_v3_pro, calculate_backtest_stats
from challenge_simulator import simulate_challenge


ASSETS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'NZD_USD', 'USD_CAD',
    'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY',
    'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_NZD',
    'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_NZD',
    'AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'NZD_CAD', 'NZD_CHF',
    'XAU_USD', 'XAG_USD',
    'NAS100_USD', 'SPX500_USD', 'US30_USD', 'DE30_EUR', 'UK100_GBP',
    'WTICO_USD', 'BCO_USD', 'NATGAS_USD',
    'BTC_USD', 'ETH_USD'
]

AGGRESSIVE_ASSET_CONFIGS = {
    'EUR_USD': {'conf': 2, 'rr': 1.5},
    'GBP_USD': {'conf': 2, 'rr': 1.5},
    'USD_JPY': {'conf': 4, 'rr': 2.5},
    'USD_CHF': {'conf': 2, 'rr': 1.5},
    'USD_CAD': {'conf': 2, 'rr': 1.5},
    'AUD_USD': {'conf': 2, 'rr': 1.5},
    'NZD_USD': {'conf': 2, 'rr': 1.5},
    'EUR_GBP': {'conf': 2, 'rr': 1.5},
    'EUR_JPY': {'conf': 2, 'rr': 1.5},
    'GBP_JPY': {'conf': 2, 'rr': 1.5},
    'XAU_USD': {'conf': 2, 'rr': 1.5},
    'XAG_USD': {'conf': 2, 'rr': 1.5},
    'WTICO_USD': {'conf': 2, 'rr': 1.5},
    'BCO_USD': {'conf': 2, 'rr': 1.5},
    'NAS100_USD': {'conf': 2, 'rr': 1.5},
    'SPX500_USD': {'conf': 2, 'rr': 1.5},
    'BTC_USD': {'conf': 2, 'rr': 1.5},
    'ETH_USD': {'conf': 2, 'rr': 1.5},
}


def fetch_data_for_v3_pro(symbol: str) -> tuple:
    """Fetch Daily and Weekly data for V3 Pro backtesting."""
    daily_candles = get_ohlcv(symbol, timeframe="D", count=500)
    weekly_candles = get_ohlcv(symbol, timeframe="W", count=200)
    
    daily_list = []
    for c in daily_candles:
        daily_list.append({
            'time': c['time'].strftime("%Y-%m-%dT%H:%M:%S") if hasattr(c['time'], 'strftime') else str(c['time']),
            'open': c['open'],
            'high': c['high'],
            'low': c['low'],
            'close': c['close'],
            'volume': c.get('volume', 0)
        })
    
    weekly_list = []
    for c in weekly_candles:
        weekly_list.append({
            'time': c['time'].strftime("%Y-%m-%dT%H:%M:%S") if hasattr(c['time'], 'strftime') else str(c['time']),
            'open': c['open'],
            'high': c['high'],
            'low': c['low'],
            'close': c['close'],
            'volume': c.get('volume', 0)
        })
    
    return daily_list, weekly_list


def run_v3_pro_backtest_for_asset(
    symbol: str,
    year: int,
    min_rr: float = None,
    min_confluence: int = None,
    risk_per_trade: float = 250.0,
    partial_tp: bool = True,
    partial_tp_r: float = 1.5,
    start_month: int = 1,
    end_month: int = 12
) -> Dict:
    """Run V3 Pro backtest for a single asset for a year with AGGRESSIVE settings. Optionally filter by month range."""
    
    try:
        # Use asset-specific config if available, otherwise use defaults
        if symbol in AGGRESSIVE_ASSET_CONFIGS:
            config = AGGRESSIVE_ASSET_CONFIGS[symbol]
            if min_rr is None:
                min_rr = config['rr']
            if min_confluence is None:
                min_confluence = config['conf']
        else:
            if min_rr is None:
                min_rr = 1.5
            if min_confluence is None:
                min_confluence = 2
        
        daily_candles, weekly_candles = fetch_data_for_v3_pro(symbol)
        
        if len(daily_candles) < 100:
            return {'error': f'Insufficient data: {len(daily_candles)} daily candles'}
        
        year_str = str(year)
        daily_candles = [c for c in daily_candles if c['time'].startswith(year_str)]
        weekly_candles = [c for c in weekly_candles if c['time'].startswith(year_str) or c['time'].startswith(str(year-1))]
        
        if len(daily_candles) < 60:
            return {'error': f'Insufficient data for {year}: {len(daily_candles)} daily candles'}
        
        trades = backtest_v3_pro(
            daily_candles=daily_candles,
            weekly_candles=weekly_candles,
            min_rr=min_rr,
            min_confluence=min_confluence,
            risk_per_trade=risk_per_trade,
            partial_tp=partial_tp,
            partial_tp_r=partial_tp_r
        )
        
        if start_month > 1 or end_month < 12:
            filtered_trades = []
            for t in trades:
                entry_time_str = t['entry_time']
                try:
                    month = int(entry_time_str.split('-')[1])
                    if start_month <= month <= end_month:
                        filtered_trades.append(t)
                except (ValueError, IndexError):
                    pass
            trades = filtered_trades
        
        stats = calculate_backtest_stats(trades)
        
        return {
            'symbol': symbol,
            'year': year,
            'trades': trades,
            'stats': stats
        }
    
    except Exception as e:
        import traceback
        return {'error': f'{str(e)}\n{traceback.format_exc()}'}


def run_monthly_challenge_simulation(
    symbol: str,
    year: int,
    month: int,
    min_rr: float = 2.5,
    min_confluence: int = 3
) -> Dict:
    """
    Run 5%ers challenge simulation for a single month.
    Returns pass/fail for Step 1 and Step 2.
    """
    
    try:
        daily_candles, weekly_candles = fetch_data_for_v3_pro(symbol)
        
        if len(daily_candles) < 60:
            return {'error': f'Insufficient data for {symbol} {year}-{month:02d}'}
        
        month_start = f"{year}-{month:02d}"
        
        trades = backtest_v3_pro(
            daily_candles=daily_candles,
            weekly_candles=weekly_candles,
            min_rr=min_rr,
            min_confluence=min_confluence,
            risk_per_trade=250.0
        )
        
        month_trades = [t for t in trades if t['entry_time'].startswith(month_start)]
        
        if not month_trades:
            return {
                'symbol': symbol,
                'year': year,
                'month': month,
                'trades': [],
                'step1_passed': False,
                'step2_passed': False,
                'total_pnl': 0,
                'profitable_days': 0,
                'reason': 'No trades in month'
            }
        
        result = simulate_challenge(month_trades)
        
        return {
            'symbol': symbol,
            'year': year,
            'month': month,
            'trades': month_trades,
            'step1_passed': result.get('step1_passed', False),
            'step2_passed': result.get('step2_passed', False),
            'total_pnl': result.get('final_balance', 10000) - 10000,
            'profitable_days': result.get('profitable_days', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'daily_drawdown_hit': result.get('daily_dd_breach', False),
            'result': result
        }
    
    except Exception as e:
        return {'error': str(e)}


def run_full_year_challenge(
    symbol: str,
    year: int,
    min_rr: float = 2.5,
    min_confluence: int = 3
) -> Dict:
    """Run challenge simulation for all 12 months of a year."""
    
    results = {
        'symbol': symbol,
        'year': year,
        'months': [],
        'total_step1_passed': 0,
        'total_step2_passed': 0,
        'total_both_passed': 0,
        'total_pnl': 0,
        'monthly_results': []
    }
    
    for month in range(1, 13):
        month_result = run_monthly_challenge_simulation(
            symbol, year, month, min_rr, min_confluence
        )
        
        if 'error' in month_result:
            results['months'].append({
                'month': month,
                'error': month_result['error']
            })
            continue
        
        results['months'].append(month_result)
        
        if month_result.get('step1_passed'):
            results['total_step1_passed'] += 1
        if month_result.get('step2_passed'):
            results['total_step2_passed'] += 1
        if month_result.get('step1_passed') and month_result.get('step2_passed'):
            results['total_both_passed'] += 1
        
        results['total_pnl'] += month_result.get('total_pnl', 0)
    
    return results


def run_all_assets_v3_pro(year: int, min_rr: float = 2.5, min_confluence: int = 3) -> Dict:
    """Run V3 Pro backtest on all assets for a year."""
    
    results = {}
    
    for symbol in ASSETS:
        print(f"[V3 Pro] Running {symbol} {year}...")
        
        year_result = run_full_year_challenge(symbol, year, min_rr, min_confluence)
        
        if 'error' not in year_result:
            results[symbol] = year_result
            print(f"  {symbol}: {year_result['total_both_passed']}/12 months passed, ${year_result['total_pnl']:,.0f} P/L")
        else:
            print(f"  {symbol}: Error - {year_result.get('error', 'Unknown')}")
    
    return results


def generate_v3_pro_summary(results: Dict) -> str:
    """Generate summary report for V3 Pro backtest."""
    
    lines = [
        "=" * 70,
        "V3 PRO STRATEGY - 5%ERS CHALLENGE BACKTEST SUMMARY",
        "=" * 70,
        "",
        f"{'Asset':<12} | {'Both Pass':<10} | {'Total P/L':<12} | {'Win Rate':<8}",
        "-" * 60
    ]
    
    total_both = 0
    total_months = 0
    total_pnl = 0
    
    for symbol, data in results.items():
        both_passed = data.get('total_both_passed', 0)
        pnl = data.get('total_pnl', 0)
        
        all_trades = []
        for m in data.get('months', []):
            if 'trades' in m:
                all_trades.extend(m['trades'])
        
        wins = len([t for t in all_trades if t.get('result') == 'WIN'])
        win_rate = (wins / len(all_trades) * 100) if all_trades else 0
        
        lines.append(f"{symbol:<12} | {both_passed:>2}/12     | ${pnl:>10,.0f} | {win_rate:>5.1f}%")
        
        total_both += both_passed
        total_months += 12
        total_pnl += pnl
    
    lines.append("-" * 60)
    lines.append(f"TOTAL: {total_both}/{total_months} months passed both steps")
    lines.append(f"TOTAL P/L: ${total_pnl:,.0f}")
    lines.append(f"PASS RATE: {total_both/total_months*100:.1f}%")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running V3 Pro Strategy Backtest 2024...")
    print("=" * 60)
    
    results = run_all_assets_v3_pro(2024)
    summary = generate_v3_pro_summary(results)
    print(summary)
