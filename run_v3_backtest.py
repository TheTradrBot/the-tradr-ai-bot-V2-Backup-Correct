"""
Run V3 Strategy Backtest

Target: +60% annual return per asset ($60K on $100K account with 1% risk per trade)
"""

import os
import sys
from datetime import datetime, timedelta

from backtest_engine import load_backtest_data
from strategy_v3 import (
    generate_v3_signals,
    backtest_v3_signals,
    run_v3_backtest,
    V3Params,
    get_optimized_params,
    OPTIMIZED_PARAMS,
)

ASSETS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "AUD_USD",
    "XAU_USD",
    "BTC_USD",
]

ACCOUNT_SIZE = 100_000
RISK_PER_TRADE_USD = 1000  # 1% of 100K


def run_all_v3_backtests(use_optimized: bool = True):
    """Run V3 backtests for all assets."""
    
    print("=" * 100)
    print("BLUEPRINT V3 STRATEGY - BACKTEST RESULTS")
    print(f"Account: ${ACCOUNT_SIZE:,} | Risk per Trade: ${RISK_PER_TRADE_USD:,}")
    print(f"Period: Jan 2024 - Nov 2024")
    print(f"Parameters: {'Optimized' if use_optimized else 'Default'}")
    print("=" * 100)
    
    all_results = []
    total_pnl = 0
    total_trades = 0
    total_winners = 0
    
    for symbol in ASSETS:
        print(f"\nProcessing {symbol}...")
        
        try:
            data = load_backtest_data(symbol, year=2024, use_csv=True)
            daily = data.get("daily", [])
            weekly = data.get("weekly", [])
            monthly = data.get("monthly", [])
        except Exception as e:
            print(f"  Error loading data for {symbol}: {e}")
            continue
        
        if not daily or len(daily) < 50:
            print(f"  Insufficient data for {symbol}")
            continue
        
        params = get_optimized_params(symbol) if use_optimized else V3Params()
        
        result = run_v3_backtest(
            symbol=symbol,
            daily_candles=daily,
            weekly_candles=weekly,
            monthly_candles=monthly,
            params=params,
            risk_per_trade=RISK_PER_TRADE_USD,
        )
        
        all_results.append(result)
        total_pnl += result["total_pnl_usd"]
        total_trades += result["total_trades"]
        total_winners += result["winners"]
    
    print("\n" + "=" * 100)
    print(f"{'Asset':<12} {'Trades':<8} {'Winners':<10} {'Win Rate':<10} {'Total R':<12} {'P/L ($)':<15} {'Final Account':<15}")
    print("-" * 100)
    
    for r in all_results:
        pnl_str = f"+${r['total_pnl_usd']:,.0f}" if r['total_pnl_usd'] >= 0 else f"-${abs(r['total_pnl_usd']):,.0f}"
        final = ACCOUNT_SIZE + r['total_pnl_usd']
        
        print(f"{r['symbol']:<12} {r['total_trades']:<8} {r['winners']:<10} {r['win_rate']:>7.1f}%   {r['total_pnl_r']:>+9.2f}R  {pnl_str:<15} ${final:,.0f}")
    
    print("-" * 100)
    
    overall_wr = total_winners / total_trades * 100 if total_trades > 0 else 0
    total_pnl_str = f"+${total_pnl:,.0f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.0f}"
    final_total = ACCOUNT_SIZE + total_pnl
    
    print(f"{'TOTAL':<12} {total_trades:<8} {total_winners:<10} {overall_wr:>7.1f}%   {'':<12} {total_pnl_str:<15} ${final_total:,.0f}")
    
    annual_return_pct = (total_pnl / ACCOUNT_SIZE) * 100
    print(f"\nAnnual Return: {annual_return_pct:+.1f}%")
    print(f"Target: +60% (${ACCOUNT_SIZE * 0.6:,.0f})")
    print("=" * 100)
    
    return {
        "results": all_results,
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "total_winners": total_winners,
        "overall_wr": overall_wr,
        "annual_return_pct": annual_return_pct,
    }


def optimize_parameters():
    """Run parameter optimization to find best settings per asset."""
    
    print("=" * 100)
    print("PARAMETER OPTIMIZATION - Finding Best Settings per Asset")
    print("=" * 100)
    
    param_grid = {
        "sr_lookback_bars": [2, 3],
        "sr_tolerance_pct": [0.2, 0.3, 0.4, 0.5],
        "min_confluence": [2, 3],
        "cooldown_bars": [1, 2],
        "require_rejection_candle": [False, True],
        "min_rr_ratio": [1.0, 1.2],
        "sl_atr_buffer": [0.2, 0.3, 0.4],
    }
    
    best_params = {}
    
    for symbol in ASSETS:
        print(f"\nOptimizing {symbol}...")
        
        try:
            data = load_backtest_data(symbol, year=2024, use_csv=True)
            daily = data.get("daily", [])
            weekly = data.get("weekly", [])
            monthly = data.get("monthly", [])
        except Exception as e:
            print(f"  Error loading data: {e}")
            continue
        
        if not daily or len(daily) < 50:
            print(f"  Insufficient data")
            continue
        
        best_pnl = float("-inf")
        best_config = None
        
        for sr_lb in param_grid["sr_lookback_bars"]:
            for sr_tol in param_grid["sr_tolerance_pct"]:
                for min_conf in param_grid["min_confluence"]:
                    for cd in param_grid["cooldown_bars"]:
                        for rej in param_grid["require_rejection_candle"]:
                            for min_rr in param_grid["min_rr_ratio"]:
                                for sl_buf in param_grid["sl_atr_buffer"]:
                                    params = V3Params(
                                        sr_lookback_bars=sr_lb,
                                        sr_tolerance_pct=sr_tol,
                                        min_confluence=min_conf,
                                        cooldown_bars=cd,
                                        require_rejection_candle=rej,
                                        min_rr_ratio=min_rr,
                                        sl_atr_buffer=sl_buf,
                                    )
                                    
                                    result = run_v3_backtest(
                                        symbol=symbol,
                                        daily_candles=daily,
                                        weekly_candles=weekly,
                                        monthly_candles=monthly,
                                        params=params,
                                        risk_per_trade=RISK_PER_TRADE_USD,
                                    )
                                    
                                    if result["total_trades"] < 10:
                                        continue
                                    
                                    if result["total_pnl_usd"] > best_pnl:
                                        best_pnl = result["total_pnl_usd"]
                                        best_config = {
                                            "params": params,
                                            "result": result,
                                        }
        
        if best_config:
            best_params[symbol] = best_config
            p = best_config["params"]
            r = best_config["result"]
            print(f"  Best: {r['total_trades']} trades, {r['win_rate']:.1f}% WR, ${r['total_pnl_usd']:+,.0f}")
            print(f"  Params: sr_tol={p.sr_tolerance_pct}, min_conf={p.min_confluence}, sl_buf={p.sl_atr_buffer}")
        else:
            print(f"  No profitable configuration found")
    
    print("\n" + "=" * 100)
    print("OPTIMIZED PARAMETERS CODE:")
    print("=" * 100)
    
    print("\nOPTIMIZED_PARAMS = {")
    for symbol, config in best_params.items():
        p = config["params"]
        print(f'    "{symbol}": V3Params(')
        print(f'        sr_lookback_bars={p.sr_lookback_bars},')
        print(f'        sr_tolerance_pct={p.sr_tolerance_pct},')
        print(f'        min_confluence={p.min_confluence},')
        print(f'        cooldown_bars={p.cooldown_bars},')
        print(f'        require_rejection_candle={p.require_rejection_candle},')
        print(f'        min_rr_ratio={p.min_rr_ratio},')
        print(f'        sl_atr_buffer={p.sl_atr_buffer},')
        print(f'    ),')
    print("}")
    
    return best_params


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "optimize":
        optimize_parameters()
    else:
        run_all_v3_backtests(use_optimized=True)
