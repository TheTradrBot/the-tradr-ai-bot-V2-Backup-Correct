"""
Modular Backtest Engine

Backtests the modular strategy pipeline with separate modules:
1. Reversal module - HTF S/R mean-reversion
2. Trend module - EMA pullback continuation
3. Ensemble - Combined signals from both modules
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from strategy_modular import (
    ModularSignal,
    generate_reversal_signals,
    generate_trend_signals,
    generate_ensemble_signals,
    ReversalParams,
    TrendParams,
)
from data import get_ohlcv


@dataclass
class TradeResult:
    """Result of a single trade."""
    symbol: str
    direction: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    exit_type: str  # "tp1", "tp2", "tp3", "sl", "runner_sl"
    pnl_r: float  # P/L in risk units
    module: str
    timestamp: Optional[str] = None


def _simulate_trade(
    signal: ModularSignal,
    candles: List[Dict],
    max_hold_bars: int = 50,
) -> Optional[TradeResult]:
    """
    Simulate a trade from signal to exit.
    
    Uses laddered exits:
    - TP1 hit: Close 50% position, move SL to breakeven
    - TP2 hit: Close 30% position, move SL to TP1
    - TP3 hit: Close remaining 20%
    - SL hit: Full loss
    
    Returns weighted R-multiple.
    """
    if signal.bar_index >= len(candles) - 1:
        return None
    
    entry = signal.entry
    sl = signal.stop_loss
    tp1 = signal.tp1
    tp2 = signal.tp2
    tp3 = signal.tp3
    direction = signal.direction
    
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    
    tp1_hit = False
    tp2_hit = False
    current_sl = sl
    
    weighted_pnl = 0.0
    remaining_position = 1.0
    
    for i in range(signal.bar_index + 1, min(signal.bar_index + max_hold_bars + 1, len(candles))):
        bar = candles[i]
        high = bar.get("high", 0)
        low = bar.get("low", 0)
        
        if direction == "bullish":
            if low <= current_sl:
                loss = (current_sl - entry) / risk * remaining_position
                weighted_pnl += loss
                
                exit_type = "sl" if not tp1_hit else ("runner_sl" if tp2_hit else "partial_sl")
                return TradeResult(
                    symbol=signal.symbol,
                    direction=direction,
                    entry_bar=signal.bar_index,
                    exit_bar=i,
                    entry_price=entry,
                    exit_price=current_sl,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    exit_type=exit_type,
                    pnl_r=weighted_pnl,
                    module=signal.module,
                    timestamp=signal.timestamp,
                )
            
            if not tp1_hit and high >= tp1:
                pnl_tp1 = (tp1 - entry) / risk * 0.5
                weighted_pnl += pnl_tp1
                remaining_position -= 0.5
                tp1_hit = True
                current_sl = entry
            
            if tp1_hit and not tp2_hit and high >= tp2:
                pnl_tp2 = (tp2 - entry) / risk * 0.3
                weighted_pnl += pnl_tp2
                remaining_position -= 0.3
                tp2_hit = True
                current_sl = tp1
            
            if tp2_hit and high >= tp3:
                pnl_tp3 = (tp3 - entry) / risk * 0.2
                weighted_pnl += pnl_tp3
                
                return TradeResult(
                    symbol=signal.symbol,
                    direction=direction,
                    entry_bar=signal.bar_index,
                    exit_bar=i,
                    entry_price=entry,
                    exit_price=tp3,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    exit_type="tp3_full",
                    pnl_r=weighted_pnl,
                    module=signal.module,
                    timestamp=signal.timestamp,
                )
        
        else:
            if high >= current_sl:
                loss = (entry - current_sl) / risk * remaining_position
                weighted_pnl += loss
                
                exit_type = "sl" if not tp1_hit else ("runner_sl" if tp2_hit else "partial_sl")
                return TradeResult(
                    symbol=signal.symbol,
                    direction=direction,
                    entry_bar=signal.bar_index,
                    exit_bar=i,
                    entry_price=entry,
                    exit_price=current_sl,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    exit_type=exit_type,
                    pnl_r=weighted_pnl,
                    module=signal.module,
                    timestamp=signal.timestamp,
                )
            
            if not tp1_hit and low <= tp1:
                pnl_tp1 = (entry - tp1) / risk * 0.5
                weighted_pnl += pnl_tp1
                remaining_position -= 0.5
                tp1_hit = True
                current_sl = entry
            
            if tp1_hit and not tp2_hit and low <= tp2:
                pnl_tp2 = (entry - tp2) / risk * 0.3
                weighted_pnl += pnl_tp2
                remaining_position -= 0.3
                tp2_hit = True
                current_sl = tp1
            
            if tp2_hit and low <= tp3:
                pnl_tp3 = (entry - tp3) / risk * 0.2
                weighted_pnl += pnl_tp3
                
                return TradeResult(
                    symbol=signal.symbol,
                    direction=direction,
                    entry_bar=signal.bar_index,
                    exit_bar=i,
                    entry_price=entry,
                    exit_price=tp3,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    exit_type="tp3_full",
                    pnl_r=weighted_pnl,
                    module=signal.module,
                    timestamp=signal.timestamp,
                )
    
    if remaining_position < 1.0:
        final_close = candles[-1].get("close", entry)
        if direction == "bullish":
            final_pnl = (final_close - entry) / risk * remaining_position
        else:
            final_pnl = (entry - final_close) / risk * remaining_position
        weighted_pnl += final_pnl
    
    return TradeResult(
        symbol=signal.symbol,
        direction=direction,
        entry_bar=signal.bar_index,
        exit_bar=min(signal.bar_index + max_hold_bars, len(candles) - 1),
        entry_price=entry,
        exit_price=candles[-1].get("close", entry),
        stop_loss=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        exit_type="timeout" if remaining_position == 1.0 else "partial_timeout",
        pnl_r=weighted_pnl,
        module=signal.module,
        timestamp=signal.timestamp,
    )


def run_modular_backtest(
    symbol: str,
    period: str = "Jan 2024 - Nov 2024",
    mode: str = "ensemble",  # "reversal", "trend", or "ensemble"
    reversal_params: ReversalParams = None,
    trend_params: TrendParams = None,
) -> Dict:
    """
    Run backtest for modular strategy.
    
    Args:
        symbol: Trading pair symbol
        period: Backtest period description
        mode: Which module(s) to use
        reversal_params: Parameters for reversal module
        trend_params: Parameters for trend module
    
    Returns:
        Dictionary with backtest results
    """
    daily = get_ohlcv(symbol, timeframe="D", count=2000, use_cache=False)
    weekly = get_ohlcv(symbol, timeframe="W", count=500, use_cache=False) or []
    monthly = get_ohlcv(symbol, timeframe="M", count=240, use_cache=False) or []
    
    if not daily or len(daily) < 100:
        return {
            "symbol": symbol,
            "period": period,
            "mode": mode,
            "total_trades": 0,
            "win_rate": 0,
            "net_return_pct": 0,
            "avg_rr": 0,
            "error": "Insufficient data",
        }
    
    approx_2024_start = len(daily) - 260
    daily_2024 = daily[approx_2024_start:]
    
    if mode == "reversal":
        signals = generate_reversal_signals(daily, weekly, monthly, symbol, reversal_params)
    elif mode == "trend":
        signals = generate_trend_signals(daily, symbol, trend_params)
    else:
        signals = generate_ensemble_signals(daily, weekly, monthly, symbol, reversal_params, trend_params)
    
    signals_2024 = [s for s in signals if s.bar_index >= approx_2024_start]
    
    trades = []
    for signal in signals_2024:
        result = _simulate_trade(signal, daily)
        if result:
            trades.append(result)
    
    if not trades:
        return {
            "symbol": symbol,
            "period": period,
            "mode": mode,
            "total_trades": 0,
            "win_rate": 0,
            "net_return_pct": 0,
            "avg_rr": 0,
            "trades": [],
        }
    
    wins = sum(1 for t in trades if t.pnl_r > 0)
    total_rr = sum(t.pnl_r for t in trades)
    
    risk_per_trade = 1.0
    net_return_pct = total_rr * risk_per_trade
    
    return {
        "symbol": symbol,
        "period": period,
        "mode": mode,
        "total_trades": len(trades),
        "win_rate": (wins / len(trades)) * 100 if trades else 0,
        "net_return_pct": net_return_pct,
        "avg_rr": total_rr / len(trades) if trades else 0,
        "total_wins": wins,
        "total_losses": len(trades) - wins,
        "trades": trades,
        "by_module": {
            "reversal": sum(1 for t in trades if t.module == "reversal"),
            "trend": sum(1 for t in trades if t.module == "trend"),
        },
    }


def run_all_modular_backtests(
    mode: str = "ensemble",
    reversal_params: ReversalParams = None,
    trend_params: TrendParams = None,
) -> Dict:
    """Run backtests for all assets and summarize."""
    assets = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "XAU_USD", "BTC_USD"]
    
    results = []
    for asset in assets:
        try:
            result = run_modular_backtest(asset, mode=mode, reversal_params=reversal_params, trend_params=trend_params)
            results.append(result)
        except Exception as e:
            results.append({
                "symbol": asset,
                "mode": mode,
                "total_trades": 0,
                "win_rate": 0,
                "net_return_pct": 0,
                "avg_rr": 0,
                "error": str(e),
            })
    
    total_trades = sum(r.get("total_trades", 0) for r in results)
    total_wins = sum(r.get("total_wins", 0) for r in results)
    total_return = sum(r.get("net_return_pct", 0) for r in results)
    
    return {
        "mode": mode,
        "results": results,
        "summary": {
            "total_trades": total_trades,
            "overall_win_rate": (total_wins / total_trades * 100) if total_trades > 0 else 0,
            "avg_return": total_return / len(assets) if assets else 0,
            "total_return": total_return,
        },
    }


if __name__ == "__main__":
    print("=" * 100)
    print("MODULAR STRATEGY BACKTEST")
    print("=" * 100)
    
    for mode in ["reversal", "trend", "ensemble"]:
        print(f"\n{'='*50}")
        print(f"MODE: {mode.upper()}")
        print("="*50)
        
        all_results = run_all_modular_backtests(mode=mode)
        
        print(f"{'Asset':<12} {'Trades':<8} {'Wins':<6} {'Win Rate':<12} {'Return %':<12} {'Avg RR':<10}")
        print("-" * 70)
        
        for r in all_results["results"]:
            print(f"{r['symbol']:<12} {r['total_trades']:<8} {r.get('total_wins', 0):<6} "
                  f"{r['win_rate']:>8.1f}%    {r['net_return_pct']:>10.1f}%  {r['avg_rr']:>8.2f}")
        
        print("-" * 70)
        s = all_results["summary"]
        print(f"SUMMARY: {s['total_trades']} trades, {s['overall_win_rate']:.1f}% WR, "
              f"{s['avg_return']:.1f}% avg return")
