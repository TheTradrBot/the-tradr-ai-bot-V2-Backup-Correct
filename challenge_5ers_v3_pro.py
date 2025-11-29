"""
5%ers Challenge Backtest Runner for V3 Pro Strategy

Tests the V3 Pro (Golden Pocket + Wyckoff) strategy against
5%ers High Stakes 10K Challenge rules.

Target: 70%+ yearly return per asset, pass both steps every month
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import calendar
from data import get_ohlcv, get_ohlcv_range
from strategy_v3_pro import backtest_v3_pro, calculate_backtest_stats
from challenge_simulator import simulate_challenge
from challenge_risk_manager import simulate_with_concurrent_tracking, RiskConfig


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


def fetch_data_for_date_range(
    symbol: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int
) -> tuple:
    """
    Fetch Daily and Weekly data for a specific date range.
    
    This function fetches historical data including a warmup period
    (60 days before start date for indicator calculations).
    """
    warmup_start_year = start_year
    warmup_start_month = start_month - 3
    if warmup_start_month < 1:
        warmup_start_month += 12
        warmup_start_year -= 1
    
    weekly_start_year = start_year - 1
    weekly_start_month = start_month
    
    daily_candles = get_ohlcv_range(
        instrument=symbol,
        timeframe="D",
        start_year=warmup_start_year,
        start_month=warmup_start_month,
        end_year=end_year,
        end_month=end_month
    )
    
    weekly_candles = get_ohlcv_range(
        instrument=symbol,
        timeframe="W",
        start_year=weekly_start_year,
        start_month=weekly_start_month,
        end_year=end_year,
        end_month=end_month
    )
    
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


def run_backtest_date_range(
    symbol: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    min_rr: float = None,
    min_confluence: int = None,
    risk_per_trade: float = 250.0,
    partial_tp: bool = True,
    partial_tp_r: float = 1.5
) -> Dict:
    """
    Run V3 Pro backtest for a specific date range (e.g., Jan 2023 - Jul 2024).
    
    This fetches historical data from OANDA for the specified period.
    """
    try:
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
        
        print(f"[backtest] Running {symbol} from {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        
        daily_candles, weekly_candles = fetch_data_for_date_range(
            symbol, start_year, start_month, end_year, end_month
        )
        
        if len(daily_candles) < 100:
            return {'error': f'Insufficient data: only {len(daily_candles)} daily candles fetched. Need at least 100.'}
        
        if len(weekly_candles) < 20:
            return {'error': f'Insufficient weekly data: only {len(weekly_candles)} weekly candles fetched.'}
        
        print(f"[backtest] Data loaded: {len(daily_candles)} daily, {len(weekly_candles)} weekly candles")
        
        trades = backtest_v3_pro(
            daily_candles=daily_candles,
            weekly_candles=weekly_candles,
            min_rr=min_rr,
            min_confluence=min_confluence,
            risk_per_trade=risk_per_trade,
            partial_tp=partial_tp,
            partial_tp_r=partial_tp_r
        )
        
        range_start = f"{start_year}-{start_month:02d}"
        range_end = f"{end_year}-{end_month:02d}"
        filtered_trades = []
        for t in trades:
            entry_time = t.get('entry_time', '')[:7]
            if range_start <= entry_time <= range_end:
                filtered_trades.append(t)
        
        stats = calculate_backtest_stats(filtered_trades)
        
        print(f"[backtest] Completed: {len(filtered_trades)} trades in range")
        
        return {
            'symbol': symbol,
            'start': f"{start_year}-{start_month:02d}",
            'end': f"{end_year}-{end_month:02d}",
            'trades': filtered_trades,
            'stats': stats
        }
    
    except Exception as e:
        import traceback
        return {'error': f'{str(e)}\n{traceback.format_exc()}'}


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
    """Run V3 Pro backtest for a single asset for a year with AGGRESSIVE settings. Optionally filter by month range.
    
    IMPORTANT: Includes warmup period (3 months prior) to ensure zones calculate correctly.
    """
    
    try:
        from datetime import datetime, timedelta as td
        
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
        
        def parse_time(time_str):
            try:
                if hasattr(time_str, 'strftime'):
                    return time_str
                return datetime.fromisoformat(time_str.replace('Z', '+00:00').split('+')[0])
            except:
                return datetime.min
        
        from calendar import monthrange
        
        warmup_start = datetime(year, start_month, 1) - td(days=120)
        weekly_warmup_start = datetime(year, start_month, 1) - td(days=400)
        last_day = monthrange(year, end_month)[1]
        range_end = datetime(year, end_month, last_day, 23, 59, 59)
        
        filtered_daily = []
        for c in daily_candles:
            c_time = parse_time(c['time'])
            if c_time >= warmup_start and c_time <= range_end:
                filtered_daily.append(c)
        
        filtered_weekly = []
        for c in weekly_candles:
            c_time = parse_time(c['time'])
            if c_time >= weekly_warmup_start and c_time <= range_end:
                filtered_weekly.append(c)
        
        if len(filtered_daily) < 60:
            return {'error': f'Insufficient data for {year}: {len(filtered_daily)} daily candles'}
        
        trades = backtest_v3_pro(
            daily_candles=filtered_daily,
            weekly_candles=filtered_weekly,
            min_rr=min_rr,
            min_confluence=min_confluence,
            risk_per_trade=risk_per_trade,
            partial_tp=partial_tp,
            partial_tp_r=partial_tp_r
        )
        
        if start_month > 1 or end_month < 12:
            filtered_trades = []
            target_start = datetime(year, start_month, 1)
            target_last_day = monthrange(year, end_month)[1]
            target_end = datetime(year, end_month, target_last_day, 23, 59, 59)
            
            for t in trades:
                entry_time_str = t.get('entry_time', '')
                try:
                    entry_dt = parse_time(entry_time_str)
                    if target_start <= entry_dt <= target_end:
                        filtered_trades.append(t)
                except (ValueError, IndexError, TypeError):
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


PORTFOLIO_ASSETS = [
    'EUR_USD', 'GBP_USD', 'AUD_USD', 'USD_CAD', 'USD_JPY', 'USD_CHF',
    'EUR_JPY', 'GBP_CAD',
    'ETH_USD', 'BTC_USD', 'LTC_USD', 'BCH_USD',
]


def run_portfolio_month_challenge(
    year: int,
    month: int,
    assets: Optional[List[str]] = None,
    min_rr: float = 2.0,
    min_confluence: int = 2
) -> Dict:
    """
    Run 5%ers challenge simulation for a month using ALL assets.
    Combines trades from all assets and simulates as a single challenge account.
    
    This matches real trading where you'd trade multiple pairs in a month.
    """
    if assets is None:
        assets = PORTFOLIO_ASSETS
    
    all_trades = []
    asset_stats = {}
    
    month_prefix = f"{year}-{month:02d}"
    
    for symbol in assets:
        try:
            daily_candles, weekly_candles = fetch_data_for_v3_pro(symbol)
            
            if len(daily_candles) < 60:
                continue
            
            trades = backtest_v3_pro(
                daily_candles=daily_candles,
                weekly_candles=weekly_candles,
                min_rr=min_rr,
                min_confluence=min_confluence,
                risk_per_trade=250.0
            )
            
            month_trades = [t for t in trades if t['entry_time'].startswith(month_prefix)]
            
            for t in month_trades:
                trade_copy = t.copy()
                trade_copy['symbol'] = symbol
                all_trades.append(trade_copy)
            
            asset_stats[symbol] = len(month_trades)
            
        except Exception as e:
            print(f"[portfolio_month] Error processing {symbol}: {e}")
            continue
    
    if not all_trades:
        return {
            'year': year,
            'month': month,
            'trades': [],
            'step1_passed': False,
            'step2_passed': False,
            'total_pnl': 0,
            'profitable_days': 0,
            'asset_breakdown': asset_stats,
            'reason': 'No trades found across all assets'
        }
    
    all_trades.sort(key=lambda t: t['entry_time'])
    
    config = RiskConfig(
        base_risk_pct=2.0,
        reduced_risk_pct=1.0,
        min_risk_pct=0.5,
        max_total_exposure_pct=4.0,
        partial_tp_r=1.0,
        partial_close_pct=50.0,
        round_trip_fee_pct=0.30,
    )
    
    result = simulate_with_concurrent_tracking(all_trades, config)
    
    wins = len([t for t in all_trades if t.get('result') in ['WIN', 'PARTIAL_WIN']])
    win_rate = (wins / len(all_trades) * 100) if all_trades else 0
    total_r = sum(t.get('r_result', t.get('r_multiple', 0)) for t in all_trades)
    
    return {
        'year': year,
        'month': month,
        'trades': all_trades,
        'total_trades': len(all_trades),
        'step1_passed': result.get('step1_passed', False),
        'step2_passed': result.get('step2_passed', False),
        'total_pnl': result.get('total_pnl', 0),
        'final_balance': result.get('final_balance', 10000),
        'profitable_days': result.get('profitable_days', 0),
        'max_drawdown': result.get('max_drawdown', 0),
        'daily_drawdown_hit': result.get('daily_dd_breach', False),
        'blown': result.get('blown', False),
        'blown_reason': result.get('blown_reason', ''),
        'win_rate': result.get('win_rate', win_rate),
        'total_r': total_r,
        'asset_breakdown': asset_stats,
        'result': result,
        'skipped_trades': result.get('skipped_trades', 0),
    }


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
