"""
Trade Export Module for Blueprint Trader AI.

This module handles exporting backtest trades to CSV format
for analysis and verification.
"""

import csv
import io
from datetime import datetime
from typing import Dict, List, Optional
from backtest import run_backtest


def export_trades_to_csv(
    asset: str,
    period: str,
    trades: Optional[List[Dict]] = None,
) -> str:
    """
    Export backtest trades to CSV format.
    
    Args:
        asset: The asset symbol (e.g., EUR_USD)
        period: The time period string (e.g., 'Jan 2024 - Dec 2024')
        trades: Optional pre-computed trades list. If None, runs backtest.
        
    Returns:
        CSV string content
    """
    if trades is None:
        result = run_backtest(asset, period)
        trades = result.get("trades", [])
        asset = result.get("asset", asset)
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    headers = [
        "Asset",
        "Direction",
        "Entry Date",
        "Entry Price",
        "Stop Loss",
        "TP1",
        "TP2", 
        "TP3",
        "Exit Date",
        "Exit Price",
        "Exit Reason",
        "Result (R)",
        "Result (%)",
    ]
    writer.writerow(headers)
    
    for trade in trades:
        entry_price = trade.get("entry_price", trade.get("entry", 0))
        stop_loss = trade.get("stop_loss", trade.get("sl", 0))
        
        exit_price = trade.get("exit_price", 0)
        if not exit_price:
            rr = trade.get("rr", 0)
            risk = trade.get("risk", abs(entry_price - stop_loss) if entry_price and stop_loss else 1)
            if trade.get("direction") == "bullish":
                exit_price = entry_price + (rr * risk)
            else:
                exit_price = entry_price - (rr * risk)
        
        rr = trade.get("rr", 0)
        result_pct = rr * 1.0
        
        row = [
            asset,
            trade.get("direction", ""),
            trade.get("entry_date", ""),
            f"{entry_price:.5f}" if entry_price else "",
            f"{stop_loss:.5f}" if stop_loss else "",
            f"{trade.get('tp1', 0):.5f}" if trade.get("tp1") else "",
            f"{trade.get('tp2', 0):.5f}" if trade.get("tp2") else "",
            f"{trade.get('tp3', 0):.5f}" if trade.get("tp3") else "",
            trade.get("exit_date", ""),
            f"{exit_price:.5f}" if exit_price else "",
            trade.get("exit_reason", ""),
            f"{rr:+.2f}",
            f"{result_pct:+.2f}%",
        ]
        writer.writerow(row)
    
    return output.getvalue()


def get_backtest_with_trades(asset: str, period: str) -> Dict:
    """
    Run backtest and return results with full trade details.
    
    Args:
        asset: The asset symbol
        period: The time period string
        
    Returns:
        Dictionary with backtest results including detailed trades
    """
    result = run_backtest(asset, period)
    
    trades = result.get("trades", [])
    enhanced_trades = []
    
    for trade in trades:
        enhanced_trade = dict(trade)
        
        if "entry_price" not in enhanced_trade:
            enhanced_trade["entry_price"] = enhanced_trade.get("entry", 0)
        if "stop_loss" not in enhanced_trade:
            enhanced_trade["stop_loss"] = enhanced_trade.get("sl", 0)
        
        enhanced_trades.append(enhanced_trade)
    
    result["trades"] = enhanced_trades
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
    wins = sum(1 for t in trades if t.get("rr", 0) > 0)
    losses = sum(1 for t in trades if t.get("rr", 0) < 0)
    breakeven = total - wins - losses
    
    total_rr = sum(t.get("rr", 0) for t in trades)
    avg_rr = total_rr / total if total > 0 else 0
    
    win_rate = (wins / total * 100) if total > 0 else 0
    
    exit_reasons = {}
    for t in trades:
        reason = t.get("exit_reason", "Unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    summary = f"""Trade Summary
=============
Total Trades: {total}
Wins: {wins} | Losses: {losses} | Breakeven: {breakeven}
Win Rate: {win_rate:.1f}%
Total R: {total_rr:+.2f}
Average R: {avg_rr:+.2f}

Exit Breakdown:
"""
    for reason, count in sorted(exit_reasons.items()):
        summary += f"  {reason}: {count}\n"
    
    return summary
