"""
Comprehensive Backtest Runner for all Forex Assets.

Runs backtests on all forex pairs across multiple years and 
identifies which assets need optimization to reach the 50% annual return target.
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
    """Run backtest for a single asset."""
    try:
        result = run_backtest(asset, period)
        return result
    except Exception as e:
        print(f"Error running backtest for {asset}: {e}")
        return None


def run_all_forex_backtests(period: str = "Jan 2024 - Nov 2024") -> List[Dict]:
    """Run backtests for all forex pairs."""
    results = []
    
    print("\n" + "=" * 80)
    print(f"RUNNING BACKTESTS FOR ALL FOREX PAIRS - {period}")
    print("=" * 80)
    print(f"{'Asset':<12} {'Trades':<8} {'Win Rate':<10} {'Return %':<12} {'Max DD %':<10} {'Avg RR':<10} {'Target':<8}")
    print("-" * 80)
    
    major_pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD"]
    
    for asset in major_pairs:
        result = run_asset_backtest(asset, period)
        if result:
            results.append(result)
            
            meets_target = result.get("net_return_pct", 0) >= 50.0
            status = "PASS" if meets_target else "FAIL"
            
            print(f"{asset:<12} {result.get('total_trades', 0):<8} "
                  f"{result.get('win_rate', 0):>7.1f}%  "
                  f"{result.get('net_return_pct', 0):>10.1f}%  "
                  f"{result.get('max_drawdown_pct', 0):>8.1f}%  "
                  f"{result.get('avg_rr', 0):>8.2f}  "
                  f"[{status}]")
    
    print("-" * 80)
    
    if results:
        total_trades = sum(r.get("total_trades", 0) for r in results)
        avg_return = sum(r.get("net_return_pct", 0) for r in results) / len(results)
        passing = sum(1 for r in results if r.get("net_return_pct", 0) >= 50.0)
        
        print(f"\nSUMMARY: {total_trades} total trades, {avg_return:.1f}% avg return")
        print(f"Meeting 50% Target: {passing}/{len(results)} assets ({passing/len(results)*100:.1f}%)")
    
    print("=" * 80 + "\n")
    
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
