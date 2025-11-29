"""
Trade Export Module for Blueprint Trader AI.

This module handles exporting backtest trades to CSV format
for analysis and verification.

Uses V3 Pro strategy (Fibonacci-based Daily S/D + Golden Pocket + Wyckoff)

VERIFIED: This module uses the SAME V3 Pro strategy as live trading.
Backtest results from this module are directly applicable to 5%ers challenge trades.
"""

import csv
import io
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from calendar import monthrange

from strategy_v3_pro import backtest_v3_pro
from data import get_ohlcv


MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}


def parse_date_range(period: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse date range string like 'Jan 24 - Aug 24' into start and end dates.
    
    Args:
        period: Date range string (e.g., 'Jan 24 - Aug 24', 'Mar 23 - Jun 24')
        
    Returns:
        Tuple of (start_date, end_date) or (None, None) if invalid
    """
    try:
        pattern = r'([A-Za-z]{3})\s*(\d{2,4})\s*[-–]\s*([A-Za-z]{3})\s*(\d{2,4})'
        match = re.match(pattern, period.strip())
        
        if not match:
            return None, None
        
        start_month_str, start_year_str, end_month_str, end_year_str = match.groups()
        
        start_month = MONTH_MAP.get(start_month_str.lower())
        end_month = MONTH_MAP.get(end_month_str.lower())
        
        if start_month is None or end_month is None:
            return None, None
        
        start_year = int(start_year_str)
        end_year = int(end_year_str)
        
        if start_year < 100:
            start_year += 2000
        if end_year < 100:
            end_year += 2000
        
        start_date = datetime(start_year, start_month, 1)
        
        last_day = monthrange(end_year, end_month)[1]
        end_date = datetime(end_year, end_month, last_day, 23, 59, 59)
        
        if start_date > end_date:
            print(f"[parse_date_range] Invalid range: start ({start_date}) > end ({end_date})")
            return None, None
        
        return start_date, end_date
        
    except Exception as e:
        print(f"[parse_date_range] Error: {e}")
        return None, None


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


def export_trades_to_csv_range(
    asset: str,
    start_date: datetime,
    end_date: datetime,
    trades: Optional[List[Dict]] = None,
) -> str:
    """
    Export backtest trades to CSV format for a date range.
    
    Args:
        asset: The asset symbol (e.g., EUR_USD)
        start_date: Start date of the range
        end_date: End date of the range
        trades: Optional pre-computed trades list. If None, runs backtest.
        
    Returns:
        CSV string content
    """
    if trades is None:
        result = get_backtest_with_trades_range(asset, start_date, end_date)
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
            f"${trade.get('pnl_usd', trade.get('pnl', 0)):+.0f}",
        ]
        writer.writerow(row)
    
    return output.getvalue()


def get_backtest_with_trades_range(asset: str, start_date: datetime, end_date: datetime) -> Dict:
    """
    Run backtest and return results with full trade details for a date range.
    
    Uses V3 Pro strategy - SAME strategy used for live trading signals.
    Results are directly applicable to 5%ers challenge performance.
    
    IMPORTANT: Includes warmup period for zone calculations:
    - 3 months of daily data before start_date
    - 1 year of weekly data before start_date
    
    Args:
        asset: The asset symbol
        start_date: Start date of the range
        end_date: End date of the range
        
    Returns:
        Dictionary with backtest results including detailed trades
    """
    try:
        from data import get_ohlcv_range
        from datetime import timedelta as td
        
        warmup_start = start_date - td(days=120)
        weekly_warmup_start = start_date - td(days=400)
        
        daily = get_ohlcv(asset, timeframe='D', count=1000)
        weekly = get_ohlcv(asset, timeframe='W', count=400)
        
        if not daily or len(daily) < 50:
            return {"trades": [], "total_trades": 0, "win_rate": 0, "net_pnl": 0, "total_r": 0}
        
        def parse_time(time_str):
            try:
                if hasattr(time_str, 'strftime'):
                    return time_str
                return datetime.fromisoformat(time_str.replace('Z', '+00:00').split('+')[0])
            except:
                return datetime.min
        
        daily_list = []
        for c in daily:
            c_time = parse_time(c['time'])
            if c_time >= warmup_start and c_time <= end_date:
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
            c_time = parse_time(c['time'])
            if c_time >= weekly_warmup_start and c_time <= end_date:
                time_str = c['time'].strftime('%Y-%m-%dT%H:%M:%S') if hasattr(c['time'], 'strftime') else str(c['time'])
                weekly_list.append({
                    'time': time_str,
                    'open': c['open'],
                    'high': c['high'],
                    'low': c['low'],
                    'close': c['close'],
                    'volume': c.get('volume', 0)
                })
        
        if not daily_list:
            return {"trades": [], "total_trades": 0, "win_rate": 0, "net_pnl": 0, "total_r": 0}
        
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
            return {"trades": [], "total_trades": 0, "win_rate": 0, "net_pnl": 0, "total_r": 0}
        
        def trade_in_range(t):
            entry = t.get('entry_time', '')
            if hasattr(entry, 'strftime'):
                entry_dt = entry
            else:
                try:
                    entry_dt = datetime.fromisoformat(entry.replace('Z', '+00:00').split('+')[0])
                except:
                    return True
            return start_date <= entry_dt <= end_date
        
        trades = [t for t in trades if trade_in_range(t)]
        
        if not trades:
            return {"trades": [], "total_trades": 0, "win_rate": 0, "net_pnl": 0, "total_r": 0}
        
        total_trades = len(trades)
        wins = len([t for t in trades if t.get('result') in ['WIN', 'PARTIAL_WIN']])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        net_pnl = sum(t.get('pnl_usd', 0) for t in trades)
        total_r = sum(t.get('r_result', t.get('r_multiple', 0)) for t in trades)
        
        return {
            "trades": trades,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "total_r": total_r,
        }
    
    except Exception as e:
        print(f"[trade_export] Error in get_backtest_with_trades_range: {e}")
        import traceback
        traceback.print_exc()
        return {"trades": [], "total_trades": 0, "win_rate": 0, "net_pnl": 0, "total_r": 0}
