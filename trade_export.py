"""
Trade Export Module for Blueprint Trader AI.

This module handles exporting backtest trades to CSV format
for analysis and verification.

Uses V3 Pro strategy (Fibonacci-based Daily S/D + Golden Pocket + Wyckoff)
"""

import csv
import io
from datetime import datetime
from typing import Dict, List, Optional

from strategy_v3_pro import backtest_v3_pro
from data import get_ohlcv


def export_trades_to_csv(
    asset: str,
    year: int,
    trades: Optional[List[Dict]] = None,
) -> str:
    """
    Export backtest trades to CSV format.
    
    Args:
        asset: The asset symbol (e.g., EUR_USD)
        year: The year to backtest
        trades: Optional pre-computed trades list. If None, runs backtest.
        
    Returns:
        CSV string content
    """
    if trades is None:
        result = get_backtest_with_trades(asset, year)
        trades = result.get("trades", [])
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    headers = [
        "Asset",
        "Direction",
        "Entry Date",
        "Entry Price",
        "Stop Loss",
        "Take Profit",
        "Exit Date",
        "Exit Price",
        "Exit Reason",
        "R Multiple",
        "R Result",
        "P/L ($)",
    ]
    writer.writerow(headers)
    
    for trade in trades:
        entry_price = trade.get("entry_price", 0)
        stop_loss = trade.get("stop_loss", 0)
        take_profit = trade.get("take_profit", 0)
        
        entry_time = trade.get("entry_time", "")
        if hasattr(entry_time, 'isoformat'):
            entry_time = entry_time.isoformat()
        
        exit_time = trade.get("exit_time", "")
        if hasattr(exit_time, 'isoformat'):
            exit_time = exit_time.isoformat()
        
        row = [
            trade.get("symbol", asset),
            trade.get("direction", ""),
            entry_time,
            f"{entry_price:.5f}" if entry_price else "",
            f"{stop_loss:.5f}" if stop_loss else "",
            f"{take_profit:.5f}" if take_profit else "",
            exit_time,
            f"{trade.get('exit_price', 0):.5f}" if trade.get("exit_price") else "",
            trade.get("exit_type", ""),
            f"{trade.get('r_multiple', 0):+.2f}",
            f"{trade.get('r_result', 0):+.2f}",
            f"${trade.get('pnl', 0):+.0f}",
        ]
        writer.writerow(row)
    
    return output.getvalue()


def get_backtest_with_trades(asset: str, year: int) -> Dict:
    """
    Run backtest and return results with full trade details.
    
    Args:
        asset: The asset symbol
        year: The year to backtest
        
    Returns:
        Dictionary with backtest results including detailed trades
    """
    try:
        # Fetch candle data
        daily = get_ohlcv(asset, timeframe='D', count=500)
        weekly = get_ohlcv(asset, timeframe='W', count=200)
        
        if not daily or len(daily) < 50:
            return {"trades": [], "total_trades": 0, "win_rate": 0, "net_pnl": 0}
        
        # Convert to dict format for backtest
        daily_list = []
        for c in daily:
            time_str = c['time'].strftime('%Y-%m-%dT%H:%M:%S') if hasattr(c['time'], 'strftime') else str(c['time'])
            daily_list.append({
                'time': time_str,
                'open': c['open'],
                'high': c['high'],
                'low': c['low'],
                'close': c['close'],
                'volume': c.get('volume', 0)
            })
        
        weekly_list = []
        for c in weekly:
            time_str = c['time'].strftime('%Y-%m-%dT%H:%M:%S') if hasattr(c['time'], 'strftime') else str(c['time'])
            weekly_list.append({
                'time': time_str,
                'open': c['open'],
                'high': c['high'],
                'low': c['low'],
                'close': c['close'],
                'volume': c.get('volume', 0)
            })
        
        # Filter by year
        year_str = str(year)
        daily_list = [c for c in daily_list if c['time'].startswith(year_str)]
        weekly_list = [c for c in weekly_list if c['time'].startswith(year_str) or c['time'].startswith(str(year-1))]
        
        if not daily_list:
            return {"trades": [], "total_trades": 0, "win_rate": 0, "net_pnl": 0}
        
        # Run backtest with V3 Pro strategy
        trades = backtest_v3_pro(
            daily_candles=daily_list,
            weekly_candles=weekly_list,
            min_rr=2.0,
            min_confluence=2,
            risk_per_trade=250.0,
            partial_tp=True,
            partial_tp_r=1.5
        )
        
        if not trades:
            return {"trades": [], "total_trades": 0, "win_rate": 0, "net_pnl": 0}
        
        # Calculate stats
        total_trades = len(trades)
        wins = len([t for t in trades if t.get('result') in ['WIN', 'PARTIAL_WIN']])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        net_pnl = sum(t.get('pnl_usd', 0) for t in trades)
        
        return {
            "trades": trades,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "net_pnl": net_pnl,
        }
    
    except Exception as e:
        print(f"[trade_export] Error in get_backtest_with_trades: {e}")
        import traceback
        traceback.print_exc()
        return {"trades": [], "total_trades": 0, "win_rate": 0, "net_pnl": 0}


def generate_trade_summary(trades: List[Dict]) -> str:
    """
    Generate a text summary of trades.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Summary text
    """
    if not trades:
        return "No trades in period."
    
    total = len(trades)
    wins = sum(1 for t in trades if t.get("r_result", t.get("r_multiple", 0)) > 0)
    losses = sum(1 for t in trades if t.get("r_result", t.get("r_multiple", 0)) < 0)
    breakeven = total - wins - losses
    
    total_rr = sum(t.get("r_result", t.get("r_multiple", 0)) for t in trades)
    avg_rr = total_rr / total if total > 0 else 0
    
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    
    win_rate = (wins / total * 100) if total > 0 else 0
    
    lines = [
        f"**Trade Summary**",
        f"Total Trades: {total}",
        f"Wins: {wins} | Losses: {losses} | B/E: {breakeven}",
        f"Win Rate: {win_rate:.1f}%",
        f"Total R: {total_rr:+.2f}R",
        f"Avg R per Trade: {avg_rr:+.2f}R",
        f"Total P/L: ${total_pnl:+,.0f}",
    ]
    
    return "\n".join(lines)
