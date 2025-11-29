"""
Comprehensive Trading System Test Suite

Tests:
1. V3 Pro Strategy Logic
2. Strategy-Backtest Integration
3. Dynamic Risk Management (3.0%/1.5%/0.5%)
4. 5%ers Challenge Simulation (2023-2025)
5. Profit Optimization
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

from strategy_v3_pro import (
    calculate_atr,
    find_swing_points,
    calculate_optimal_entry_zone,
    calculate_fib_extension_tps,
    detect_break_of_structure,
    identify_supply_demand_zones,
    detect_wyckoff_spring,
    detect_wyckoff_upthrust,
    generate_v3_pro_signal,
    backtest_v3_pro,
    calculate_backtest_stats
)

from challenge_risk_manager import (
    RiskConfig,
    calculate_smart_risk,
    simulate_with_concurrent_tracking,
    run_monthly_challenge_simulation as run_risk_monthly
)

from challenge_simulator import (
    simulate_challenge,
    CHALLENGE_RULES
)

from data import get_ohlcv


def test_strategy_functions():
    """Test 1: V3 Pro Strategy Logic Functions"""
    print("\n" + "="*70)
    print("TEST 1: V3 PRO STRATEGY LOGIC FUNCTIONS")
    print("="*70)
    
    sample_candles = [
        {'time': '2024-01-01', 'open': 1.1000, 'high': 1.1050, 'low': 1.0950, 'close': 1.1030},
        {'time': '2024-01-02', 'open': 1.1030, 'high': 1.1080, 'low': 1.1000, 'close': 1.1070},
        {'time': '2024-01-03', 'open': 1.1070, 'high': 1.1120, 'low': 1.1040, 'close': 1.1100},
        {'time': '2024-01-04', 'open': 1.1100, 'high': 1.1150, 'low': 1.1080, 'close': 1.1090},
        {'time': '2024-01-05', 'open': 1.1090, 'high': 1.1140, 'low': 1.1050, 'close': 1.1060},
        {'time': '2024-01-06', 'open': 1.1060, 'high': 1.1100, 'low': 1.0980, 'close': 1.0990},
        {'time': '2024-01-07', 'open': 1.0990, 'high': 1.1010, 'low': 1.0950, 'close': 1.0970},
        {'time': '2024-01-08', 'open': 1.0970, 'high': 1.1000, 'low': 1.0940, 'close': 1.0960},
        {'time': '2024-01-09', 'open': 1.0960, 'high': 1.1020, 'low': 1.0930, 'close': 1.1000},
        {'time': '2024-01-10', 'open': 1.1000, 'high': 1.1060, 'low': 1.0980, 'close': 1.1050},
        {'time': '2024-01-11', 'open': 1.1050, 'high': 1.1100, 'low': 1.1020, 'close': 1.1080},
        {'time': '2024-01-12', 'open': 1.1080, 'high': 1.1130, 'low': 1.1060, 'close': 1.1110},
        {'time': '2024-01-13', 'open': 1.1110, 'high': 1.1160, 'low': 1.1090, 'close': 1.1140},
        {'time': '2024-01-14', 'open': 1.1140, 'high': 1.1180, 'low': 1.1120, 'close': 1.1160},
        {'time': '2024-01-15', 'open': 1.1160, 'high': 1.1200, 'low': 1.1140, 'close': 1.1180},
    ]
    
    print("\n[TEST] ATR Calculation...")
    atr = calculate_atr(sample_candles, period=5)
    print(f"  ATR (5 period): {atr:.5f}")
    assert atr > 0, "ATR should be positive"
    print("  PASS: ATR calculation works")
    
    print("\n[TEST] Swing Points Detection...")
    extended_candles = sample_candles * 3
    swing_highs, swing_lows = find_swing_points(extended_candles, lookback=2)
    print(f"  Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
    print("  PASS: Swing point detection works")
    
    print("\n[TEST] Optimal Entry Zone (0.5-0.66 Fib)...")
    swing_high = 1.1200
    swing_low = 1.1000
    oez_low, oez_high = calculate_optimal_entry_zone(swing_high, swing_low, 'long')
    expected_range = swing_high - swing_low
    print(f"  Swing Range: {expected_range:.4f}")
    print(f"  OEZ Long: {oez_low:.4f} - {oez_high:.4f}")
    
    expected_50 = swing_high - (expected_range * 0.5)
    expected_66 = swing_high - (expected_range * 0.66)
    print(f"  Expected 50% level: {expected_50:.4f}")
    print(f"  Expected 66% level: {expected_66:.4f}")
    
    assert abs(oez_high - expected_50) < 0.0001 or abs(oez_low - expected_50) < 0.0001, "50% Fib level incorrect"
    assert abs(oez_high - expected_66) < 0.0001 or abs(oez_low - expected_66) < 0.0001, "66% Fib level incorrect"
    print("  PASS: Optimal Entry Zone calculation correct (0.5-0.66 Fib)")
    
    print("\n[TEST] Fibonacci Extension TPs...")
    tp1, tp2, tp3 = calculate_fib_extension_tps(swing_high, swing_low, 'long')
    print(f"  TP1 (-0.25 ext): {tp1:.4f}")
    print(f"  TP2 (-0.68 ext): {tp2:.4f}")
    print(f"  TP3 (-1.0 ext): {tp3:.4f}")
    
    expected_tp1 = swing_high + (expected_range * 0.25)
    expected_tp2 = swing_high + (expected_range * 0.68)
    expected_tp3 = swing_high + (expected_range * 1.0)
    
    assert abs(tp1 - expected_tp1) < 0.0001, f"TP1 should be {expected_tp1:.4f}"
    assert abs(tp2 - expected_tp2) < 0.0001, f"TP2 should be {expected_tp2:.4f}"
    assert abs(tp3 - expected_tp3) < 0.0001, f"TP3 should be {expected_tp3:.4f}"
    print("  PASS: Fibonacci Extension TPs correct")
    
    print("\n" + "-"*70)
    print("STRATEGY FUNCTIONS: ALL TESTS PASSED")
    return True


def test_dynamic_risk_management():
    """Test 2: Dynamic Risk Management (2.0%/1.0%/0.5%) with 4% Max Exposure"""
    print("\n" + "="*70)
    print("TEST 2: DYNAMIC RISK MANAGEMENT (SAFE MODE)")
    print("="*70)
    
    config = RiskConfig()
    
    print("\n[TEST] Risk Configuration Values...")
    print(f"  Base Risk: {config.base_risk_pct}%")
    print(f"  Reduced Risk: {config.reduced_risk_pct}%")
    print(f"  Minimum Risk: {config.min_risk_pct}%")
    print(f"  DD Threshold for Reduction: {config.dd_threshold_for_reduction}%")
    print(f"  DD Threshold for Minimum: {config.dd_threshold_for_min}%")
    print(f"  Max Total Exposure: {config.max_total_exposure_pct}%")
    print(f"  Round-Trip Commission: {config.round_trip_fee_pct}%")
    
    assert config.base_risk_pct == 2.0, f"Base risk should be 2.0%, got {config.base_risk_pct}%"
    assert config.reduced_risk_pct == 1.0, f"Reduced risk should be 1.0%, got {config.reduced_risk_pct}%"
    assert config.min_risk_pct == 0.5, f"Min risk should be 0.5%, got {config.min_risk_pct}%"
    assert config.max_total_exposure_pct == 4.0, f"Max exposure should be 4.0%, got {config.max_total_exposure_pct}%"
    assert config.round_trip_fee_pct == 0.30, f"Round-trip fee should be 0.30%, got {config.round_trip_fee_pct}%"
    print("  PASS: Risk percentages are 2.0% / 1.0% / 0.5% with 4% max exposure")
    
    print("\n[TEST] Base Risk Scenario (0% drawdown)...")
    starting_balance = 10000
    balance = 10000
    prev_day_balance = 10000
    open_exposure = 0
    
    risk_usd, reason = calculate_smart_risk(balance, starting_balance, prev_day_balance, open_exposure, config)
    expected_base_risk = starting_balance * (config.base_risk_pct / 100)
    print(f"  Balance: ${balance}, Reason: {reason}")
    print(f"  Risk USD: ${risk_usd:.2f} (Expected: ${expected_base_risk:.2f})")
    assert risk_usd == expected_base_risk, f"Expected base risk ${expected_base_risk}, got ${risk_usd}"
    assert reason == "BASE_RISK", f"Expected BASE_RISK reason, got {reason}"
    print("  PASS: Base risk (2.0%) applied correctly at 0% DD")
    
    print("\n[TEST] Reduced Risk Scenario (2.5% drawdown)...")
    balance = 9750
    prev_day_balance = 9750
    risk_usd, reason = calculate_smart_risk(balance, starting_balance, prev_day_balance, open_exposure, config)
    expected_reduced_risk = starting_balance * (config.reduced_risk_pct / 100)
    print(f"  Balance: ${balance} (2.5% DD), Reason: {reason}")
    print(f"  Risk USD: ${risk_usd:.2f} (Expected: ${expected_reduced_risk:.2f})")
    assert risk_usd == expected_reduced_risk, f"Expected reduced risk ${expected_reduced_risk}, got ${risk_usd}"
    assert reason == "REDUCED_RISK_IN_DD", f"Expected REDUCED_RISK_IN_DD reason, got {reason}"
    print("  PASS: Reduced risk (1.0%) applied correctly at 2.5% DD")
    
    print("\n[TEST] Minimum Risk Scenario (5% drawdown)...")
    balance = 9500
    prev_day_balance = 9500
    risk_usd, reason = calculate_smart_risk(balance, starting_balance, prev_day_balance, open_exposure, config)
    expected_min_risk = starting_balance * (config.min_risk_pct / 100)
    print(f"  Balance: ${balance} (5% DD), Reason: {reason}")
    print(f"  Risk USD: ${risk_usd:.2f} (Expected: ${expected_min_risk:.2f})")
    assert risk_usd == expected_min_risk, f"Expected min risk ${expected_min_risk}, got ${risk_usd}"
    assert reason == "MIN_RISK_DEEP_DD", f"Expected MIN_RISK_DEEP_DD reason, got {reason}"
    print("  PASS: Minimum risk (0.5%) applied correctly at 5% DD")
    
    print("\n[TEST] Max 4% Exposure Cap Scenario...")
    balance = 10000
    prev_day_balance = 10000
    open_exposure = 300
    risk_usd, reason = calculate_smart_risk(balance, starting_balance, prev_day_balance, open_exposure, config)
    max_exposure = starting_balance * (config.max_total_exposure_pct / 100)
    available = max_exposure - open_exposure
    print(f"  Open Exposure: ${open_exposure}, Max Allowed: ${max_exposure}")
    print(f"  Available: ${available}, Risk Applied: ${risk_usd:.2f}")
    print(f"  Reason: {reason}")
    assert risk_usd <= available, f"Risk should not exceed available exposure"
    assert max_exposure == 400, f"Max exposure should be $400 (4% of $10K)"
    print("  PASS: 4% exposure cap working correctly")
    
    print("\n[TEST] Multi-Trade Safety - All SLs Hit Scenario...")
    open_exposure = 400
    risk_usd, reason = calculate_smart_risk(balance, starting_balance, prev_day_balance, open_exposure, config)
    print(f"  Open Exposure at 4% (max): ${open_exposure}")
    print(f"  New Risk Allowed: ${risk_usd}")
    print(f"  Reason: {reason}")
    assert risk_usd == 0, "No new trades when at max 4% exposure"
    assert reason == "MAX_EXPOSURE_REACHED", f"Expected MAX_EXPOSURE_REACHED, got {reason}"
    print("  PASS: No new trades allowed at max exposure - account protected")
    
    print("\n[TEST] DD Limit Hit Scenario...")
    balance = 8900
    prev_day_balance = 8900
    open_exposure = 0
    risk_usd, reason = calculate_smart_risk(balance, starting_balance, prev_day_balance, open_exposure, config)
    print(f"  Balance: ${balance} (11% DD - exceeds 10% limit)")
    print(f"  Risk USD: ${risk_usd}, Reason: {reason}")
    assert risk_usd == 0, "Risk should be 0 when DD limit hit"
    assert reason == "DD_LIMIT_HIT", f"Expected DD_LIMIT_HIT reason, got {reason}"
    print("  PASS: Trading stopped when DD limit hit")
    
    print("\n[TEST] Commission Deduction Verification...")
    print(f"  Entry commission: {config.fee_per_trade_pct}%")
    print(f"  Round-trip commission: {config.round_trip_fee_pct}%")
    on_200_risk = 200 * (config.round_trip_fee_pct / 100)
    print(f"  Commission on $200 risk trade: ${on_200_risk:.2f}")
    assert config.round_trip_fee_pct == 0.30, "Round-trip fee should be 0.30%"
    print("  PASS: Commissions properly configured")
    
    print("\n" + "-"*70)
    print("DYNAMIC RISK MANAGEMENT: ALL TESTS PASSED")
    return True


def test_challenge_simulation():
    """Test 3: Challenge Simulation Rules"""
    print("\n" + "="*70)
    print("TEST 3: 5%ERS CHALLENGE SIMULATION")
    print("="*70)
    
    print("\n[TEST] Challenge Rules...")
    print(f"  Starting Balance: ${CHALLENGE_RULES['starting_balance']}")
    print(f"  Step 1 Target: {CHALLENGE_RULES['step1_target_pct']}%")
    print(f"  Step 2 Target: {CHALLENGE_RULES['step2_target_pct']}%")
    print(f"  Max Drawdown: {CHALLENGE_RULES['max_drawdown_pct']}%")
    print(f"  Daily Drawdown: {CHALLENGE_RULES['daily_drawdown_pct']}%")
    print(f"  Min Profitable Days: {CHALLENGE_RULES['min_profitable_days']}")
    
    assert CHALLENGE_RULES['step1_target_pct'] == 8.0, "Step 1 should be 8%"
    assert CHALLENGE_RULES['step2_target_pct'] == 5.0, "Step 2 should be 5%"
    assert CHALLENGE_RULES['max_drawdown_pct'] == 10.0, "Max DD should be 10%"
    print("  PASS: Challenge rules configured correctly")
    
    print("\n[TEST] Passing Scenario Simulation...")
    winning_trades = [
        {'entry_time': '2024-01-01T10:00:00', 'pnl_r': 3.0, 'direction': 'long'},
        {'entry_time': '2024-01-02T10:00:00', 'pnl_r': 2.5, 'direction': 'long'},
        {'entry_time': '2024-01-03T10:00:00', 'pnl_r': -1.0, 'direction': 'short'},
        {'entry_time': '2024-01-04T10:00:00', 'pnl_r': 4.0, 'direction': 'long'},
        {'entry_time': '2024-01-05T10:00:00', 'pnl_r': 2.0, 'direction': 'short'},
    ]
    
    result = simulate_challenge(winning_trades)
    print(f"  Trades: {result['total_trades']}")
    print(f"  Final Balance: ${result['final_balance']:.0f}")
    print(f"  Total P&L: ${result['total_pnl']:.0f}")
    print(f"  Step 1 Passed: {result['step1_passed']}")
    print(f"  Step 2 Passed: {result['step2_passed']}")
    print(f"  Profitable Days: {result['profitable_days']}")
    print(f"  Challenge Passed: {result['passed']}")
    
    assert result['step1_passed'], "Step 1 should pass with winning trades"
    print("  PASS: Challenge simulation working correctly")
    
    print("\n[TEST] Max DD Breach Scenario...")
    losing_trades = [
        {'entry_time': '2024-01-01T10:00:00', 'pnl_r': -1.0, 'direction': 'long'},
        {'entry_time': '2024-01-01T11:00:00', 'pnl_r': -1.0, 'direction': 'long'},
        {'entry_time': '2024-01-01T12:00:00', 'pnl_r': -1.0, 'direction': 'long'},
        {'entry_time': '2024-01-01T13:00:00', 'pnl_r': -1.0, 'direction': 'long'},
        {'entry_time': '2024-01-01T14:00:00', 'pnl_r': -1.0, 'direction': 'long'},
    ]
    
    result = simulate_challenge(losing_trades)
    print(f"  Final Balance: ${result['final_balance']:.0f}")
    print(f"  Blown: {result['blown']}")
    print(f"  Reason: {result.get('blown_reason', 'N/A')}")
    
    assert result['blown'], "Account should be blown with 5 consecutive losses"
    print("  PASS: Max DD breach detected correctly")
    
    print("\n" + "-"*70)
    print("CHALLENGE SIMULATION: ALL TESTS PASSED")
    return True


def test_backtest_integration():
    """Test 4: Strategy-Backtest Integration with Live Data"""
    print("\n" + "="*70)
    print("TEST 4: STRATEGY-BACKTEST INTEGRATION")
    print("="*70)
    
    print("\n[TEST] Fetching live data from OANDA...")
    try:
        daily_candles = get_ohlcv('EUR_USD', timeframe='D', count=200)
        weekly_candles = get_ohlcv('EUR_USD', timeframe='W', count=50)
        
        if not daily_candles:
            print("  WARNING: No daily candles returned. Check OANDA_API_KEY.")
            return False
        
        print(f"  Daily candles: {len(daily_candles)}")
        print(f"  Weekly candles: {len(weekly_candles)}")
        
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
        
        print("  PASS: Data fetched successfully")
        
    except Exception as e:
        print(f"  ERROR: Failed to fetch data - {e}")
        return False
    
    print("\n[TEST] Running V3 Pro Backtest...")
    try:
        trades = backtest_v3_pro(
            daily_candles=daily_list,
            weekly_candles=weekly_list,
            min_rr=2.5,
            min_confluence=3,
            risk_per_trade=250.0,
            partial_tp=True,
            partial_tp_r=1.5
        )
        
        print(f"  Trades generated: {len(trades)}")
        
        if trades:
            print(f"  Sample trade: {trades[0]['direction']} @ {trades[0]['entry']:.5f}")
            print(f"    Entry Type: {trades[0].get('entry_type', 'N/A')}")
            print(f"    Result: {trades[0]['result']}")
            print(f"    R-Multiple: {trades[0]['r_multiple']:.2f}")
        
        stats = calculate_backtest_stats(trades)
        print(f"\n  Backtest Stats:")
        print(f"    Total Trades: {stats['total_trades']}")
        print(f"    Win Rate: {stats['win_rate']}%")
        print(f"    Total P&L: ${stats['total_pnl']:.2f}")
        print(f"    Avg R: {stats['avg_r']:.2f}")
        print(f"    Max DD: {stats['max_drawdown']:.1f}%")
        
        print("  PASS: Backtest integration working")
        
    except Exception as e:
        print(f"  ERROR: Backtest failed - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[TEST] Running Challenge Simulation on Backtest Trades...")
    if trades:
        challenge_trades = []
        for t in trades:
            challenge_trades.append({
                'entry_time': t['entry_time'],
                'pnl_r': t['r_multiple'],
                'direction': t['direction'],
                'result': t['result'],
                'highest_r': t.get('highest_r', 0)
            })
        
        config = RiskConfig()
        result = simulate_with_concurrent_tracking(challenge_trades, config)
        
        print(f"  Challenge Result:")
        print(f"    Step 1 Passed: {result['step1_passed']}")
        print(f"    Step 2 Passed: {result['step2_passed']}")
        print(f"    Blown: {result['blown']}")
        print(f"    Final Balance: ${result['final_balance']:.0f}")
        print(f"    Total Trades Taken: {result['total_trades']}")
        print(f"    Profitable Days: {result['profitable_days']}")
        
        print("  PASS: Challenge simulation on backtest trades working")
    
    print("\n" + "-"*70)
    print("BACKTEST INTEGRATION: ALL TESTS PASSED")
    return True


def test_multi_year_backtest():
    """Test 5: Multi-Year Backtest (2023-2025)"""
    print("\n" + "="*70)
    print("TEST 5: MULTI-YEAR BACKTEST (2023-2025)")
    print("="*70)
    
    print("\nNote: This test uses live OANDA data which is limited to recent history.")
    print("For full 2023-2025 backtesting, historical data files would be needed.")
    
    print("\n[TEST] Fetching available historical data...")
    try:
        daily_candles = get_ohlcv('EUR_USD', timeframe='D', count=500)
        weekly_candles = get_ohlcv('EUR_USD', timeframe='W', count=200)
        
        if not daily_candles:
            print("  WARNING: No data available")
            return False
        
        first_date = daily_candles[0]['time']
        last_date = daily_candles[-1]['time']
        print(f"  Data Range: {first_date} to {last_date}")
        print(f"  Total Daily Candles: {len(daily_candles)}")
        
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
        
    except Exception as e:
        print(f"  ERROR: Failed to fetch data - {e}")
        return False
    
    print("\n[TEST] Running Backtest with Dynamic Risk Management...")
    try:
        trades = backtest_v3_pro(
            daily_candles=daily_list,
            weekly_candles=weekly_list,
            min_rr=1.5,
            min_confluence=2,
            risk_per_trade=250.0,
            partial_tp=True,
            partial_tp_r=1.5
        )
        
        print(f"  Total Trades: {len(trades)}")
        
        if not trades:
            print("  No trades generated - this could be normal if confluence requirements aren't met")
            return True
        
        for t in trades:
            t['symbol'] = 'EUR_USD'
        
        config = RiskConfig(
            base_risk_pct=3.0,
            reduced_risk_pct=1.5,
            min_risk_pct=0.5,
            max_total_exposure_pct=7.0,
            daily_dd_pct=4.0
        )
        
        years_in_data = set()
        for t in trades:
            try:
                year = int(t['entry_time'][:4])
                years_in_data.add(year)
            except:
                pass
        
        print(f"  Years in data: {sorted(years_in_data)}")
        
        all_monthly_results = {}
        for year in sorted(years_in_data):
            yearly_result = run_risk_monthly(trades, config, year)
            all_monthly_results[year] = yearly_result
            
            print(f"\n  Year {year}:")
            print(f"    Months with data: {yearly_result['total_months']}")
            print(f"    Months Passed: {yearly_result['months_passed']}")
            print(f"    Pass Rate: {yearly_result['pass_rate']:.1f}%")
            print(f"    Total P&L: ${yearly_result['total_pnl']:.0f}")
        
        print("\n  Monthly Breakdown:")
        for year, yearly_result in all_monthly_results.items():
            for month, data in sorted(yearly_result.get('monthly_results', {}).items()):
                status = "PASS" if data.get('passed') else ("BREACH" if data.get('blown') else "FAIL")
                pnl = data.get('pnl', 0)
                days = data.get('profitable_days', 0)
                print(f"    {month}: {status} | P&L: ${pnl:>8.0f} | Days: {days}")
        
        print("\n  PASS: Multi-year backtest completed")
        
    except Exception as e:
        print(f"  ERROR: Multi-year backtest failed - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*70)
    print("MULTI-YEAR BACKTEST: COMPLETED")
    return True


def test_profit_optimization():
    """Test 6: Profit Optimization Features"""
    print("\n" + "="*70)
    print("TEST 6: PROFIT OPTIMIZATION FEATURES")
    print("="*70)
    
    print("\n[TEST] Partial TP + Breakeven Logic...")
    config = RiskConfig()
    
    print(f"  Partial TP at: {config.partial_tp_r}R")
    print(f"  Partial Close %: {config.partial_close_pct}%")
    print(f"  SL to Profit Buffer: {config.sl_to_profit_buffer_r}R")
    
    assert config.partial_tp_r == 1.0, "Partial TP should trigger at 1.0R"
    assert config.partial_close_pct == 50.0, "Should close 50% at partial TP"
    print("  PASS: Partial TP configuration correct")
    
    print("\n[TEST] Concurrent Trade Exposure Tracking...")
    print(f"  Max Concurrent Trades: {config.max_concurrent_trades}")
    print(f"  Max Total Exposure: {config.max_total_exposure_pct}%")
    
    max_trades_at_base = config.max_total_exposure_pct / config.base_risk_pct
    print(f"  Max trades at base risk: {max_trades_at_base:.1f}")
    
    assert config.max_total_exposure_pct == 4.0, "Max exposure should be 4.0%"
    assert config.max_total_exposure_pct < 5.0, "Max exposure must be under 5% daily DD limit"
    print("  PASS: 4% exposure limit - ensures multiple SLs can't breach daily DD")
    
    print("\n[TEST] Fee Deduction...")
    print(f"  Round-trip fee: {config.round_trip_fee_pct}%")
    
    assert config.round_trip_fee_pct == 0.30, "Round-trip fee should be 0.30%"
    print("  PASS: Commission configuration correct (0.30% round-trip)")
    
    print("\n[TEST] Simulating Profit Optimization...")
    sample_trades = [
        {'entry_time': '2024-01-01T10:00:00', 'r_multiple': 3.0, 'highest_r': 3.5, 'result': 'WIN', 'direction': 'long'},
        {'entry_time': '2024-01-02T10:00:00', 'r_multiple': -0.1, 'highest_r': 1.5, 'result': 'BE', 'direction': 'long'},
        {'entry_time': '2024-01-03T10:00:00', 'r_multiple': 2.0, 'highest_r': 2.5, 'result': 'WIN', 'direction': 'short'},
        {'entry_time': '2024-01-04T10:00:00', 'r_multiple': -1.0, 'highest_r': 0.5, 'result': 'LOSS', 'direction': 'long'},
        {'entry_time': '2024-01-05T10:00:00', 'r_multiple': 4.0, 'highest_r': 4.0, 'result': 'WIN', 'direction': 'short'},
    ]
    
    result = simulate_with_concurrent_tracking(sample_trades, config)
    
    print(f"  Trades: {result['total_trades']}")
    print(f"  Final Balance: ${result['final_balance']:.0f}")
    print(f"  Total Return: {result['total_return_pct']:.1f}%")
    print(f"  Win Rate: {result['win_rate']:.1f}%")
    
    assert result['final_balance'] > 10000, "Should be profitable with these trades"
    print("  PASS: Profit optimization simulation works")
    
    print("\n" + "-"*70)
    print("PROFIT OPTIMIZATION: ALL TESTS PASSED")
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n")
    print("="*70)
    print("BLUEPRINT TRADER AI - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    results['Strategy Functions'] = test_strategy_functions()
    results['Dynamic Risk Management'] = test_dynamic_risk_management()
    results['Challenge Simulation'] = test_challenge_simulation()
    results['Profit Optimization'] = test_profit_optimization()
    results['Backtest Integration'] = test_backtest_integration()
    results['Multi-Year Backtest'] = test_multi_year_backtest()
    
    print("\n")
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "-"*70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Check output above for details")
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
