"""
Strategy Optimizer for Blueprint Trader AI.

This module provides a machine-learning style optimization loop
that tunes strategy parameters to meet performance targets:
- Total trades: >= 60 per year per asset
- Win rate: 70-100%
- Net return: +50% to +400%
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from backtest import run_backtest
from strategy_core import StrategyParams, get_default_params


@dataclass
class OptimizationResult:
    """Result of a single optimization run."""
    asset: str
    period: str
    params: Dict[str, Any]
    total_trades: int
    win_rate: float
    net_return_pct: float
    meets_targets: bool
    target_violations: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OptimizationTargets:
    """Performance targets for optimization."""
    min_trades: int = 60
    min_win_rate: float = 70.0
    max_win_rate: float = 100.0
    min_return_pct: float = 50.0
    max_return_pct: float = 400.0


def check_targets(
    result: Dict,
    targets: OptimizationTargets = None,
) -> Tuple[bool, List[str]]:
    """
    Check if backtest result meets optimization targets.
    
    Returns:
        Tuple of (meets_all_targets, list_of_violations)
    """
    if targets is None:
        targets = OptimizationTargets()
    
    violations = []
    
    trades = result.get("total_trades", 0)
    if trades < targets.min_trades:
        violations.append(f"Trades {trades} < {targets.min_trades}")
    
    win_rate = result.get("win_rate", 0)
    if win_rate < targets.min_win_rate:
        violations.append(f"Win rate {win_rate:.1f}% < {targets.min_win_rate}%")
    if win_rate > targets.max_win_rate:
        violations.append(f"Win rate {win_rate:.1f}% > {targets.max_win_rate}%")
    
    net_return = result.get("net_return_pct", 0)
    if net_return < targets.min_return_pct:
        violations.append(f"Return {net_return:.1f}% < {targets.min_return_pct}%")
    if net_return > targets.max_return_pct:
        violations.append(f"Return {net_return:.1f}% > {targets.max_return_pct}%")
    
    return len(violations) == 0, violations


def generate_param_grid() -> List[Dict[str, Any]]:
    """
    Generate a grid of parameter combinations to test.
    
    Returns:
        List of parameter dictionaries
    """
    param_ranges = {
        "min_confluence": [1, 2, 3],
        "min_quality_factors": [0, 1, 2],
        "atr_sl_multiplier": [1.0, 1.5, 2.0],
        "atr_tp1_multiplier": [0.4, 0.6, 0.8],
        "atr_tp2_multiplier": [0.8, 1.1, 1.5],
        "atr_tp3_multiplier": [1.5, 1.8, 2.5],
        "fib_low": [0.382, 0.5],
        "fib_high": [0.786, 0.886],
        "min_rr_ratio": [0.5, 1.0, 1.5],
    }
    
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    
    combinations = list(itertools.product(*values))
    
    grid = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        grid.append(params)
    
    return grid


def run_optimization(
    asset: str,
    period: str,
    max_iterations: int = 100,
    targets: OptimizationTargets = None,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Run optimization loop to find parameters that meet targets.
    
    Args:
        asset: Asset symbol to optimize
        period: Time period for backtest
        max_iterations: Maximum parameter combinations to try
        targets: Performance targets
        verbose: Print progress
        
    Returns:
        Best OptimizationResult found
    """
    if targets is None:
        targets = OptimizationTargets()
    
    param_grid = generate_param_grid()
    
    if len(param_grid) > max_iterations:
        import random
        random.shuffle(param_grid)
        param_grid = param_grid[:max_iterations]
    
    best_result = None
    best_score = float("-inf")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing {asset} for {period}")
        print(f"Testing {len(param_grid)} parameter combinations")
        print(f"Targets: trades>={targets.min_trades}, WR {targets.min_win_rate}-{targets.max_win_rate}%, return {targets.min_return_pct}-{targets.max_return_pct}%")
        print(f"{'='*60}\n")
    
    for i, params in enumerate(param_grid):
        try:
            result = run_backtest(asset, period)
            
            meets, violations = check_targets(result, targets)
            
            trades = result.get("total_trades", 0)
            win_rate = result.get("win_rate", 0)
            net_return = result.get("net_return_pct", 0)
            
            score = 0
            score += min(trades / targets.min_trades, 1.0) * 30
            score += min(win_rate / targets.min_win_rate, 1.0) * 40
            score += min(net_return / targets.min_return_pct, 1.0) * 30
            
            if meets:
                score += 100
            
            if score > best_score:
                best_score = score
                best_result = OptimizationResult(
                    asset=asset,
                    period=result.get("period", period),
                    params=params,
                    total_trades=trades,
                    win_rate=win_rate,
                    net_return_pct=net_return,
                    meets_targets=meets,
                    target_violations=violations,
                )
                
                if verbose:
                    status = "PASS" if meets else "BEST"
                    print(f"[{i+1}/{len(param_grid)}] {status}: trades={trades}, WR={win_rate:.1f}%, return={net_return:.1f}%")
                
                if meets:
                    if verbose:
                        print(f"\nTargets met! Stopping early.")
                    break
                    
        except Exception as e:
            if verbose:
                print(f"[{i+1}/{len(param_grid)}] Error: {e}")
            continue
    
    if best_result is None:
        default_params = get_default_params()
        best_result = OptimizationResult(
            asset=asset,
            period=period,
            params=default_params.to_dict(),
            total_trades=0,
            win_rate=0,
            net_return_pct=0,
            meets_targets=False,
            target_violations=["No valid backtests completed"],
        )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimization Complete for {asset}")
        print(f"Best: trades={best_result.total_trades}, WR={best_result.win_rate:.1f}%, return={best_result.net_return_pct:.1f}%")
        print(f"Meets targets: {best_result.meets_targets}")
        if best_result.target_violations:
            print(f"Violations: {', '.join(best_result.target_violations)}")
        print(f"{'='*60}\n")
    
    return best_result


def optimize_all_assets(
    assets: List[str],
    period: str = "Jan 2024 - Dec 2024",
    max_iterations_per_asset: int = 50,
    save_results: bool = True,
) -> Dict[str, OptimizationResult]:
    """
    Run optimization for multiple assets.
    
    Args:
        assets: List of asset symbols
        period: Time period for backtests
        max_iterations_per_asset: Max parameter combinations per asset
        save_results: Save results to JSON file
        
    Returns:
        Dictionary mapping assets to their optimization results
    """
    results = {}
    
    for asset in assets:
        try:
            result = run_optimization(
                asset=asset,
                period=period,
                max_iterations=max_iterations_per_asset,
            )
            results[asset] = result
        except Exception as e:
            print(f"Error optimizing {asset}: {e}")
            continue
    
    if save_results:
        output = {
            asset: result.to_dict()
            for asset, result in results.items()
        }
        
        filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    return results


def print_optimization_summary(results: Dict[str, OptimizationResult]) -> str:
    """
    Generate a summary of optimization results.
    
    Returns:
        Summary text
    """
    lines = []
    lines.append("=" * 70)
    lines.append("OPTIMIZATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"{'Asset':<12} {'Trades':>8} {'Win Rate':>10} {'Return':>10} {'Status':>10}")
    lines.append("-" * 70)
    
    passed = 0
    failed = 0
    
    for asset, result in results.items():
        status = "PASS" if result.meets_targets else "FAIL"
        if result.meets_targets:
            passed += 1
        else:
            failed += 1
        
        lines.append(
            f"{asset:<12} {result.total_trades:>8} "
            f"{result.win_rate:>9.1f}% {result.net_return_pct:>9.1f}% "
            f"{status:>10}"
        )
    
    lines.append("-" * 70)
    lines.append(f"Total: {len(results)} assets | Passed: {passed} | Failed: {failed}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    from config import FOREX_PAIRS
    
    test_assets = ["EUR_USD", "GBP_USD", "XAU_USD"]
    
    results = optimize_all_assets(
        assets=test_assets,
        period="Jan 2024 - Dec 2024",
        max_iterations_per_asset=20,
    )
    
    print(print_optimization_summary(results))
