"""
Trade Export Module for Blueprint Trader AI.

This module handles exporting backtest trades to CSV format
for analysis and verification.
"""

import csv
import io
from datetime import datetime
from typing import Dict, List, Optional
from strategy_highwin import run_yearly_backtest


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
        result = run_yearly_backtest(asset, year)
        trades = result.get("trades", [])
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    headers = [
        "Asset",
        "Direction",
        "Entry Date",
        "Entry Price",
        "Stop Loss",
        "TP (4R)",
        "Exit Date",
        "Exit Price",
        "Exit Reason",
        "Result (R)",
        "P/L ($)",
    ]
    writer.writerow(headers)
    
    for trade in trades:
        entry_price = trade.get("entry", 0)
        stop_loss = trade.get("sl", 0)
        tp = trade.get("tp3", 0)
        
        row = [
            asset,
            trade.get("direction", ""),
            trade.get("entry_time", ""),
            f"{entry_price:.5f}" if entry_price else "",
            f"{stop_loss:.5f}" if stop_loss else "",
            f"{tp:.5f}" if tp else "",
            trade.get("exit_time", ""),
            f"{trade.get('exit_price', 0):.5f}" if trade.get("exit_price") else "",
            trade.get("exit_reason", ""),
            f"{trade.get('r_multiple', 0):+.2f}",
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
    result = run_yearly_backtest(asset, year)
    return result


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
    wins = sum(1 for t in trades if t.get("r_multiple", 0) > 0)
    losses = sum(1 for t in trades if t.get("r_multiple", 0) < 0)
    breakeven = total - wins - losses
    
    total_rr = sum(t.get("r_multiple", 0) for t in trades)
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
