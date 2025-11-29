"""
Comprehensive Backtest Runner for all Forex Assets.

Runs backtests on all forex pairs across multiple years and 
identifies which assets need optimization to reach the 50% annual return target.

Target Metrics per Asset:
- Win Rate: 70-100%
- Annual Return: ≥50%
- Trades per Year: ≥50
- Max Drawdown: ≤30%
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from backtest import run_backtest
from config import FOREX_PAIRS
from strategy_core import StrategyParams, get_default_params


@dataclass
class AssetResult:
    asset: str
    period: str
    total_trades: int
    win_rate: float
    net_return_pct: float
    max_drawdown_pct: float
    avg_rr: float
    meets_target: bool
    

def run_asset_backtest(asset: str, period: str = "Jan 2024 - Nov 2024") -> Dict[str, Any]:
    """Run backtest for a single asset using asset-specific optimized parameters."""
    try:
        result = run_backtest(asset, period)
        params = get_default_params(asset)
        result["params"] = {
            "min_confluence": params.min_confluence,
            "min_quality_factors": params.min_quality_factors,
            "atr_sl_multiplier": params.atr_sl_multiplier,
            "atr_tp1_multiplier": params.atr_tp1_multiplier,
            "atr_tp2_multiplier": params.atr_tp2_multiplier,
            "atr_tp3_multiplier": params.atr_tp3_multiplier,
        }
        return result
    except Exception as e:
        print(f"Error running backtest for {asset}: {e}")
        return None


def run_all_forex_backtests(period: str = "Jan 2024 - Nov 2024") -> List[Dict]:
    """Run backtests for all forex pairs with optimized asset-specific parameters."""
    results = []
    
    print("\n" + "=" * 100)
    print(f"RUNNING OPTIMIZED BACKTESTS FOR ALL FOREX PAIRS - {period}")
    print("Target: ≥50% Return, 70-100% Win Rate, ≥50 Trades/Year, ≤30% Max DD")
    print("=" * 100)
    print(f"{'Asset':<12} {'Trades':<8} {'Win Rate':<10} {'Return %':<12} {'Max DD %':<10} {'Avg RR':<10} {'TP1/TP2/TP3':<15} {'Status':<8}")
    print("-" * 100)
    
    major_pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD"]
    
    for asset in major_pairs:
        result = run_asset_backtest(asset, period)
        if result:
            results.append(result)
            
            win_rate = result.get("win_rate", 0)
            net_return = result.get("net_return_pct", 0)
            max_dd = result.get("max_drawdown_pct", 0)
            total_trades = result.get("total_trades", 0)
            
            meets_return = net_return >= 50.0
            meets_win_rate = 70.0 <= win_rate <= 100.0
            meets_trades = total_trades >= 50
            meets_dd = max_dd <= 30.0
            
            all_pass = meets_return and meets_win_rate and meets_trades and meets_dd
            status = "✓ PASS" if all_pass else "✗ FAIL"
            
            params = result.get("params", {})
            tp_info = f"{params.get('atr_tp1_multiplier', 0):.1f}/{params.get('atr_tp2_multiplier', 0):.1f}/{params.get('atr_tp3_multiplier', 0):.1f}"
            
            print(f"{asset:<12} {total_trades:<8} "
                  f"{win_rate:>7.1f}%  "
                  f"{net_return:>10.1f}%  "
                  f"{max_dd:>8.1f}%  "
                  f"{result.get('avg_rr', 0):>8.2f}  "
                  f"{tp_info:<15} "
                  f"[{status}]")
    
    print("-" * 100)
    
    if results:
        total_trades = sum(r.get("total_trades", 0) for r in results)
        avg_return = sum(r.get("net_return_pct", 0) for r in results) / len(results)
        avg_win_rate = sum(r.get("win_rate", 0) for r in results) / len(results)
        avg_dd = sum(r.get("max_drawdown_pct", 0) for r in results) / len(results)
        passing = sum(1 for r in results if r.get("net_return_pct", 0) >= 50.0)
        
        print(f"\nSUMMARY:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Avg Return: {avg_return:.1f}%")
        print(f"  Avg Win Rate: {avg_win_rate:.1f}%")
        print(f"  Avg Max DD: {avg_dd:.1f}%")
        print(f"  Meeting 50% Target: {passing}/{len(results)} assets ({passing/len(results)*100:.1f}%)")
        
        print("\nEXIT BREAKDOWN:")
        tp1_total = sum(r.get("tp1_trail_hits", 0) for r in results)
        tp2_total = sum(r.get("tp2_hits", 0) for r in results)
        tp3_total = sum(r.get("tp3_hits", 0) for r in results)
        sl_total = sum(r.get("sl_hits", 0) for r in results)
        print(f"  TP1+Trail: {tp1_total}, TP2: {tp2_total}, TP3: {tp3_total}, SL: {sl_total}")
    
    print("=" * 100 + "\n")
    
    return results


def analyze_underperformers(results: List[Dict]) -> List[str]:
    """Identify assets that don't meet the 50% return target."""
    underperformers = []
    
    for result in results:
        if result.get("net_return_pct", 0) < 50.0:
            underperformers.append(result.get("asset"))
    
    return underperformers


def save_results_to_json(results: List[Dict], filename: str = "backtest_results.json"):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {
            "total_assets": len(results),
            "passing": sum(1 for r in results if r.get("net_return_pct", 0) >= 50.0),
            "avg_return": sum(r.get("net_return_pct", 0) for r in results) / len(results) if results else 0,
        }
    }
    
    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    results = run_all_forex_backtests("Jan 2024 - Nov 2024")
    
    underperformers = analyze_underperformers(results)
    
    if underperformers:
        print(f"\nAssets needing optimization: {', '.join(underperformers)}")
    else:
        print("\nAll assets meet the 50% return target!")
    
    save_results_to_json(results)
