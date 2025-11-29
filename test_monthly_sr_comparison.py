"""
Compare performance: Daily+Weekly S/R vs Daily+Weekly+Monthly S/R
Only integrates monthly S/R if it produces better outcomes.
"""

from challenge_5ers_v3_pro import run_backtest_date_range
from strategy_v3_pro import backtest_v3_pro, calculate_backtest_stats

def compare_strategy_versions():
    """Test both strategy versions on recent data."""
    
    print("=" * 80)
    print("MONTHLY S/R PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    
    assets = ['EUR_USD', 'USD_JPY', 'GBP_USD']
    
    results = {}
    
    for symbol in assets:
        print(f"\n{'='*80}")
        print(f"Testing {symbol} (Oct 2024 - Nov 2024)")
        print(f"{'='*80}")
        
        result = run_backtest_date_range(
            symbol=symbol,
            start_year=2024,
            start_month=10,
            end_year=2024,
            end_month=11,
            min_rr=1.5,
            min_confluence=2
        )
        
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
            continue
        
        daily_candles = result.get('daily_candles', [])
        weekly_candles = result.get('weekly_candles', [])
        
        if not daily_candles or len(daily_candles) < 100:
            print(f"  SKIP: Not enough data")
            continue
        
        print(f"\n  Data: {len(daily_candles)} daily, {len(weekly_candles)} weekly candles")
        
        print(f"\n  [WITHOUT Monthly S/R]")
        trades_without = backtest_v3_pro(
            daily_candles=daily_candles,
            weekly_candles=weekly_candles,
            min_rr=1.5,
            min_confluence=2,
            use_monthly_sr=False
        )
        stats_without = calculate_backtest_stats(trades_without)
        
        print(f"    Trades: {stats_without['total_trades']}")
        print(f"    Win Rate: {stats_without['win_rate']}%")
        print(f"    P/L: ${stats_without['total_pnl']:.2f}")
        print(f"    Avg R: {stats_without['avg_r']:.2f}")
        print(f"    Profit Factor: {stats_without['profit_factor']}")
        
        print(f"\n  [WITH Monthly S/R]")
        trades_with = backtest_v3_pro(
            daily_candles=daily_candles,
            weekly_candles=weekly_candles,
            min_rr=1.5,
            min_confluence=2,
            use_monthly_sr=True
        )
        stats_with = calculate_backtest_stats(trades_with)
        
        print(f"    Trades: {stats_with['total_trades']}")
        print(f"    Win Rate: {stats_with['win_rate']}%")
        print(f"    P/L: ${stats_with['total_pnl']:.2f}")
        print(f"    Avg R: {stats_with['avg_r']:.2f}")
        print(f"    Profit Factor: {stats_with['profit_factor']}")
        
        improvement_pct = 0
        if stats_without['total_pnl'] > 0:
            improvement_pct = ((stats_with['total_pnl'] - stats_without['total_pnl']) / abs(stats_without['total_pnl'])) * 100
        else:
            improvement_pct = ((stats_with['total_pnl'] - stats_without['total_pnl']))
        
        if stats_with['total_pnl'] > stats_without['total_pnl']:
            print(f"\n  ✓ IMPROVEMENT: +{improvement_pct:.1f}% P&L with Monthly S/R")
            print(f"    Monthly S/R SHOULD BE ENABLED for {symbol}")
            results[symbol] = 'BETTER'
        elif stats_with['total_pnl'] < stats_without['total_pnl']:
            print(f"\n  ✗ REGRESSION: {improvement_pct:.1f}% P&L with Monthly S/R")
            print(f"    Monthly S/R SHOULD NOT be enabled for {symbol}")
            results[symbol] = 'WORSE'
        else:
            print(f"\n  = NO DIFFERENCE")
            results[symbol] = 'SAME'
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    better_count = sum(1 for v in results.values() if v == 'BETTER')
    worse_count = sum(1 for v in results.values() if v == 'WORSE')
    
    for symbol, result in results.items():
        status = "✓ ENABLE" if result == 'BETTER' else ("✗ DISABLE" if result == 'WORSE' else "= SAME")
        print(f"  {symbol}: {status}")
    
    print()
    print(f"Monthly S/R improves performance on {better_count}/{len(results)} assets")
    
    if better_count > worse_count:
        print("\n✓ RECOMMENDATION: KEEP monthly S/R ENABLED (improves more assets than it hurts)")
        return True
    else:
        print("\n✗ RECOMMENDATION: DISABLE monthly S/R (helps fewer assets than it hurts)")
        return False

if __name__ == '__main__':
    should_use_monthly_sr = compare_strategy_versions()
    
    print("\n" + "="*80)
    if should_use_monthly_sr:
        print("STATUS: Monthly S/R integration ENABLED - already added to strategy")
    else:
        print("STATUS: Monthly S/R integration DISABLED - remove from production use")
    print("="*80)
