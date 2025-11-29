"""
2024 Backtest and 5%ers Challenge Validation Script

Comprehensive validation of V3 Pro strategy across 10 diverse assets:
- Forex: EUR_USD, GBP_USD, USD_JPY, AUD_USD, EUR_JPY
- Metals: XAU_USD
- Indices: NAS100_USD
- Energies: BCO_USD, WTICO_USD
- Crypto: BTC_USD

Features:
1. Baseline backtests with trade count, win rate, R-multiples, P/L
2. Parameter tuning if trade count < 50
3. 5%ers monthly challenge simulation
4. Comprehensive reporting
"""

from datetime import datetime
from typing import List, Dict, Any, Tuple
from data import get_ohlcv
from strategy_v3_pro import backtest_v3_pro, calculate_backtest_stats
from challenge_risk_manager import simulate_with_concurrent_tracking, run_monthly_challenge_simulation, RiskConfig


VALIDATION_ASSETS = [
    'EUR_USD',    # Forex major - top performer
    'GBP_USD',    # Forex major
    'USD_JPY',    # Forex major
    'AUD_USD',    # Forex commodity
    'EUR_JPY',    # Forex cross
    'USD_CHF',    # Forex - replaced XAU_USD
    'EUR_NZD',    # Forex cross - replaced NAS100_USD
    'XAG_USD',    # Metal Silver - replaced BCO_USD
    'ETH_USD',    # Crypto Ethereum - replaced WTICO_USD
    'BTC_USD',    # Crypto Bitcoin
]

BASELINE_PARAMS = {
    'min_confluence': 2,
    'min_rr': 2.0,
}

TUNED_PARAMS = {
    'min_confluence': 2,
    'min_rr': 1.5,
}

MIN_TRADE_COUNT = 50


def fetch_data_for_backtest(symbol: str, year: int = 2024) -> Tuple[List[Dict], List[Dict]]:
    """Fetch daily and weekly data for backtesting."""
    daily_candles = get_ohlcv(symbol, timeframe="D", count=500)
    weekly_candles = get_ohlcv(symbol, timeframe="W", count=200)
    
    if not daily_candles or not weekly_candles:
        return [], []
    
    daily_list = []
    for c in daily_candles:
        time_str = c['time'].strftime("%Y-%m-%dT%H:%M:%S") if hasattr(c['time'], 'strftime') else str(c['time'])
        daily_list.append({
            'time': time_str,
            'open': c['open'],
            'high': c['high'],
            'low': c['low'],
            'close': c['close'],
            'volume': c.get('volume', 0)
        })
    
    weekly_list = []
    for c in weekly_candles:
        time_str = c['time'].strftime("%Y-%m-%dT%H:%M:%S") if hasattr(c['time'], 'strftime') else str(c['time'])
        weekly_list.append({
            'time': time_str,
            'open': c['open'],
            'high': c['high'],
            'low': c['low'],
            'close': c['close'],
            'volume': c.get('volume', 0)
        })
    
    year_str = str(year)
    daily_list = [c for c in daily_list if c['time'].startswith(year_str)]
    weekly_list = [c for c in weekly_list if c['time'].startswith(year_str) or c['time'].startswith(str(year-1))]
    
    return daily_list, weekly_list


def run_backtest_with_params(
    symbol: str,
    daily_candles: List[Dict],
    weekly_candles: List[Dict],
    min_confluence: int,
    min_rr: float,
    risk_per_trade: float = 250.0
) -> Dict:
    """Run backtest with specific parameters."""
    if len(daily_candles) < 60 or len(weekly_candles) < 12:
        return {'error': f'Insufficient data: {len(daily_candles)} daily, {len(weekly_candles)} weekly'}
    
    try:
        trades = backtest_v3_pro(
            daily_candles=daily_candles,
            weekly_candles=weekly_candles,
            min_rr=min_rr,
            min_confluence=min_confluence,
            risk_per_trade=risk_per_trade,
            partial_tp=True,
            partial_tp_r=1.5
        )
        
        for t in trades:
            t['symbol'] = symbol
        
        stats = calculate_backtest_stats(trades)
        
        long_trades = [t for t in trades if t['direction'] == 'long']
        short_trades = [t for t in trades if t['direction'] == 'short']
        
        long_wins = len([t for t in long_trades if t['result'] in ['WIN', 'PARTIAL_WIN']])
        short_wins = len([t for t in short_trades if t['result'] in ['WIN', 'PARTIAL_WIN']])
        
        total_r = sum(t.get('r_multiple', 0) for t in trades)
        
        return {
            'symbol': symbol,
            'trades': trades,
            'stats': stats,
            'trade_count': len(trades),
            'total_r': round(total_r, 2),
            'total_pnl': stats['total_pnl'],
            'win_rate': stats['win_rate'],
            'long_count': len(long_trades),
            'short_count': len(short_trades),
            'long_wins': long_wins,
            'short_wins': short_wins,
            'long_pnl': sum(t.get('pnl_usd', 0) for t in long_trades),
            'short_pnl': sum(t.get('pnl_usd', 0) for t in short_trades),
        }
    except Exception as e:
        return {'error': str(e)}


def validate_asset(symbol: str, year: int = 2024) -> Dict:
    """
    Validate a single asset with parameter tuning logic.
    
    CRITICAL: Only accept tuned parameters if they improve BOTH trade count AND dollar returns.
    """
    print(f"  Processing {symbol}...")
    
    daily_candles, weekly_candles = fetch_data_for_backtest(symbol, year)
    
    if not daily_candles:
        return {'symbol': symbol, 'error': 'No data available'}
    
    baseline_result = run_backtest_with_params(
        symbol=symbol,
        daily_candles=daily_candles,
        weekly_candles=weekly_candles,
        min_confluence=BASELINE_PARAMS['min_confluence'],
        min_rr=BASELINE_PARAMS['min_rr']
    )
    
    if 'error' in baseline_result:
        return {'symbol': symbol, 'error': baseline_result['error']}
    
    result = {
        'symbol': symbol,
        'baseline': baseline_result,
        'tuned': None,
        'params_adjusted': False,
        'final': baseline_result,
        'tuning_reason': None
    }
    
    if baseline_result['trade_count'] < MIN_TRADE_COUNT:
        tuned_result = run_backtest_with_params(
            symbol=symbol,
            daily_candles=daily_candles,
            weekly_candles=weekly_candles,
            min_confluence=TUNED_PARAMS['min_confluence'],
            min_rr=TUNED_PARAMS['min_rr']
        )
        
        if 'error' not in tuned_result:
            result['tuned'] = tuned_result
            
            trade_count_improved = tuned_result['trade_count'] > baseline_result['trade_count']
            pnl_improved = tuned_result['total_pnl'] > baseline_result['total_pnl']
            
            if trade_count_improved and pnl_improved:
                result['params_adjusted'] = True
                result['final'] = tuned_result
                result['tuning_reason'] = f"Both improved: trades {baseline_result['trade_count']}→{tuned_result['trade_count']}, P/L ${baseline_result['total_pnl']:,.0f}→${tuned_result['total_pnl']:,.0f}"
            else:
                result['tuning_reason'] = f"Tuning rejected: trades {'✓' if trade_count_improved else '✗'}, P/L {'✓' if pnl_improved else '✗'}"
    
    return result


def convert_trades_for_challenge(trades: List[Dict]) -> List[Dict]:
    """Convert backtest trades to challenge simulation format."""
    challenge_trades = []
    for t in trades:
        entry_time = t.get('entry_time', '')
        if isinstance(entry_time, str):
            pass
        elif hasattr(entry_time, 'strftime'):
            entry_time = entry_time.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            entry_time = str(entry_time)
        
        exit_time = t.get('exit_time', '')
        if isinstance(exit_time, str):
            pass
        elif hasattr(exit_time, 'strftime'):
            exit_time = exit_time.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            exit_time = str(exit_time) if exit_time else entry_time
        
        challenge_trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'r_multiple': t.get('r_multiple', 0),
            'pnl_r': t.get('r_multiple', 0),
            'result': t.get('result', 'UNKNOWN'),
            'highest_r': t.get('highest_r', 0),
            'symbol': t.get('symbol', 'UNKNOWN'),
            'direction': t.get('direction', 'unknown'),
        })
    
    return challenge_trades


def run_monthly_challenge_for_asset(trades: List[Dict], year: int = 2024) -> Dict:
    """Run 5%ers monthly challenge simulation for an asset's trades."""
    challenge_trades = convert_trades_for_challenge(trades)
    
    config = RiskConfig(
        base_risk_pct=3.0,
        reduced_risk_pct=1.5,
        min_risk_pct=0.5,
        max_dd_pct=10.0,
        daily_dd_pct=5.0,
        max_total_exposure_pct=7.0,
        partial_tp_r=1.0,
        partial_close_pct=50.0,
    )
    
    monthly_results = run_monthly_challenge_simulation(challenge_trades, config, year)
    
    return monthly_results


def print_separator(char: str = "=", width: int = 100):
    """Print a separator line."""
    print(char * width)


def print_header(title: str, width: int = 100):
    """Print a centered header."""
    print_separator("=", width)
    print(f"{title:^{width}}")
    print_separator("=", width)


def run_full_validation(year: int = 2024):
    """Run complete 2024 validation for all assets."""
    print_header(f"2024 BACKTEST & 5%ERS CHALLENGE VALIDATION")
    print(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Year: {year}")
    print(f"Assets: {len(VALIDATION_ASSETS)}")
    print()
    
    print("Phase 1: Running Backtests...")
    print_separator("-", 100)
    
    all_results = []
    all_trades_combined = []
    
    for symbol in VALIDATION_ASSETS:
        result = validate_asset(symbol, year)
        all_results.append(result)
        
        if 'error' not in result and result['final']:
            all_trades_combined.extend(result['final']['trades'])
    
    print()
    print_header("PHASE 1: BACKTEST RESULTS")
    
    print()
    print(f"{'Asset':<12} | {'Trades':>7} | {'Win%':>6} | {'Total R':>8} | {'P/L ($)':>12} | {'Longs':>6} | {'Shorts':>6} | {'Adjusted':>10}")
    print("-" * 100)
    
    total_trades = 0
    total_pnl = 0
    total_r = 0
    adjusted_count = 0
    
    for r in all_results:
        if 'error' in r:
            print(f"{r['symbol']:<12} | ERROR: {r['error']}")
            continue
        
        final = r['final']
        adjusted = "Yes" if r['params_adjusted'] else "No"
        
        print(f"{r['symbol']:<12} | {final['trade_count']:>7} | {final['win_rate']:>5.1f}% | {final['total_r']:>8.1f} | ${final['total_pnl']:>10,.0f} | {final['long_count']:>6} | {final['short_count']:>6} | {adjusted:>10}")
        
        total_trades += final['trade_count']
        total_pnl += final['total_pnl']
        total_r += final['total_r']
        if r['params_adjusted']:
            adjusted_count += 1
    
    print("-" * 100)
    avg_win_rate = sum(r['final']['win_rate'] for r in all_results if 'error' not in r) / len([r for r in all_results if 'error' not in r]) if all_results else 0
    print(f"{'TOTAL':<12} | {total_trades:>7} | {avg_win_rate:>5.1f}% | {total_r:>8.1f} | ${total_pnl:>10,.0f} |        |        | {adjusted_count:>10}")
    
    print()
    print("Parameter Tuning Summary:")
    print("-" * 60)
    for r in all_results:
        if 'error' not in r and r['tuning_reason']:
            print(f"  {r['symbol']}: {r['tuning_reason']}")
    
    print()
    print_header("PHASE 2: DIRECTION BREAKDOWN")
    
    print()
    print(f"{'Asset':<12} | {'Long Trades':>12} | {'Long Wins':>10} | {'Long P/L':>12} | {'Short Trades':>13} | {'Short Wins':>11} | {'Short P/L':>12}")
    print("-" * 100)
    
    for r in all_results:
        if 'error' in r:
            continue
        
        final = r['final']
        long_wr = (final['long_wins'] / final['long_count'] * 100) if final['long_count'] > 0 else 0
        short_wr = (final['short_wins'] / final['short_count'] * 100) if final['short_count'] > 0 else 0
        
        print(f"{r['symbol']:<12} | {final['long_count']:>12} | {final['long_wins']:>10} | ${final['long_pnl']:>10,.0f} | {final['short_count']:>13} | {final['short_wins']:>11} | ${final['short_pnl']:>10,.0f}")
    
    print()
    print_header("PHASE 3: 5%ERS MONTHLY CHALLENGE SIMULATION")
    
    print()
    print("Running challenge simulation with combined portfolio trades...")
    print()
    
    config = RiskConfig(
        base_risk_pct=3.0,
        reduced_risk_pct=1.5,
        min_risk_pct=0.5,
        max_dd_pct=10.0,
        daily_dd_pct=5.0,
        max_total_exposure_pct=7.0,
        partial_tp_r=1.0,
        partial_close_pct=50.0,
    )
    
    challenge_trades = convert_trades_for_challenge(all_trades_combined)
    monthly_results = run_monthly_challenge_simulation(challenge_trades, config, year)
    
    print(f"{'Month':<10} | {'Trades':>7} | {'Taken':>6} | {'P/L ($)':>12} | {'Return%':>8} | {'Prof.Days':>10} | {'Step1':>6} | {'Step2':>6} | {'Blown':>6} | {'Status':>10}")
    print("-" * 110)
    
    months_passed = 0
    months_blown = 0
    total_monthly_pnl = 0
    
    for month_str, data in sorted(monthly_results.get('monthly_results', {}).items()):
        step1 = "Yes" if data.get('step1') else "No"
        step2 = "Yes" if data.get('step2') else "No"
        blown = "Yes" if data.get('blown') else "No"
        
        if data.get('passed'):
            status = "PASS"
            months_passed += 1
        elif data.get('blown'):
            status = "BLOWN"
            months_blown += 1
        else:
            status = "FAIL"
        
        print(f"{month_str:<10} | {data.get('trades_available', 0):>7} | {data.get('trades_taken', 0):>6} | ${data.get('pnl', 0):>10,.0f} | {data.get('return_pct', 0):>7.1f}% | {data.get('profitable_days', 0):>10} | {step1:>6} | {step2:>6} | {blown:>6} | {status:>10}")
        
        total_monthly_pnl += data.get('pnl', 0)
    
    print("-" * 110)
    total_months = monthly_results.get('total_months', 12)
    pass_rate = (months_passed / total_months * 100) if total_months > 0 else 0
    print(f"{'SUMMARY':<10} | {len(all_trades_combined):>7} | {'-':>6} | ${total_monthly_pnl:>10,.0f} | {'-':>8} | {'-':>10} | {'-':>6} | {'-':>6} | {months_blown:>6} | {months_passed}/{total_months} ({pass_rate:.0f}%)")
    
    print()
    print_header("PHASE 4: CHALLENGE REQUIREMENTS CHECK")
    
    print()
    print("5%ers Challenge Requirements:")
    print("-" * 60)
    print(f"  - Step 1 Target: 8% profit                    (${10000 * 0.08:,.0f} on $10,000)")
    print(f"  - Step 2 Target: 5% additional profit         (${10000 * 0.05:,.0f} on $10,000)")
    print(f"  - Max Drawdown: 10%                           (Cannot go below $9,000)")
    print(f"  - Daily Drawdown: 5% of previous day balance")
    print(f"  - Minimum 3 profitable trading days")
    print()
    
    print("Monthly Requirement Breakdown:")
    print("-" * 100)
    print(f"{'Month':<10} | {'8% Target':>12} | {'DD < 10%':>10} | {'Daily DD < 5%':>14} | {'3+ Prof Days':>12} | {'All Met':>10}")
    print("-" * 100)
    
    for month_str, data in sorted(monthly_results.get('monthly_results', {}).items()):
        target_met = "Yes" if data.get('step1') else "No"
        max_dd_ok = "Yes" if not data.get('blown') else "No"
        daily_dd_ok = "Yes" if not data.get('blown') else "No"
        prof_days_ok = "Yes" if data.get('profitable_days', 0) >= 3 else "No"
        all_met = "Yes" if data.get('passed') else "No"
        
        print(f"{month_str:<10} | {target_met:>12} | {max_dd_ok:>10} | {daily_dd_ok:>14} | {prof_days_ok:>12} | {all_met:>10}")
    
    print()
    print_header("FINAL SUMMARY & RECOMMENDATIONS")
    
    print()
    print("=== OVERALL STATISTICS ===")
    print()
    print(f"  Total Trades Across All Assets:    {total_trades}")
    print(f"  Total P/L (Backtest):              ${total_pnl:,.0f}")
    print(f"  Total R-Multiple:                  {total_r:.1f}R")
    print(f"  Average Win Rate:                  {avg_win_rate:.1f}%")
    print(f"  Assets with Parameter Tuning:      {adjusted_count}/{len(VALIDATION_ASSETS)}")
    print()
    print(f"  Challenge Months Passed:           {months_passed}/{total_months}")
    print(f"  Challenge Pass Rate:               {pass_rate:.1f}%")
    print(f"  Challenge Months Blown:            {months_blown}")
    print(f"  Total Challenge P/L:               ${total_monthly_pnl:,.0f}")
    
    print()
    print("=== RECOMMENDATIONS ===")
    print()
    
    assets_below_50 = [r for r in all_results if 'error' not in r and r['final']['trade_count'] < 50]
    if assets_below_50:
        print("  1. LOW TRADE COUNT ASSETS (< 50 trades/year):")
        for r in assets_below_50:
            print(f"     - {r['symbol']}: {r['final']['trade_count']} trades")
        print("     Recommendation: Consider further parameter loosening or adding more timeframes")
        print()
    
    losing_assets = [r for r in all_results if 'error' not in r and r['final']['total_pnl'] < 0]
    if losing_assets:
        print("  2. NEGATIVE P/L ASSETS:")
        for r in losing_assets:
            print(f"     - {r['symbol']}: ${r['final']['total_pnl']:,.0f}")
        print("     Recommendation: Review entry criteria or consider removing from portfolio")
        print()
    
    if pass_rate < 50:
        print("  3. LOW CHALLENGE PASS RATE:")
        print(f"     Current pass rate: {pass_rate:.1f}%")
        print("     Target: At least 50% monthly pass rate")
        print("     Recommendations:")
        print("       - Increase base risk percentage for faster profit accumulation")
        print("       - Focus on high-probability setups with higher confluence")
        print("       - Ensure trades are spread across trading days for 3+ profitable days")
        print()
    
    if months_blown > 0:
        print("  4. RISK MANAGEMENT:")
        print(f"     Months blown: {months_blown}")
        print("     Recommendations:")
        print("       - Reduce max concurrent exposure")
        print("       - Implement stricter drawdown monitoring")
        print("       - Consider reducing risk per trade when approaching DD limits")
        print()
    
    top_performers = sorted([r for r in all_results if 'error' not in r], 
                           key=lambda x: x['final']['total_pnl'], reverse=True)[:3]
    if top_performers:
        print("  5. TOP PERFORMING ASSETS:")
        for i, r in enumerate(top_performers, 1):
            print(f"     {i}. {r['symbol']}: ${r['final']['total_pnl']:,.0f} ({r['final']['win_rate']:.1f}% win rate)")
        print("     Recommendation: Consider allocating higher risk to these assets")
        print()
    
    print_separator("=", 100)
    print(f"{'VALIDATION COMPLETE':^100}")
    print_separator("=", 100)
    
    return {
        'all_results': all_results,
        'monthly_results': monthly_results,
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'total_r': total_r,
        'avg_win_rate': avg_win_rate,
        'pass_rate': pass_rate,
        'months_passed': months_passed,
        'months_blown': months_blown,
    }


if __name__ == "__main__":
    results = run_full_validation(2024)
