"""
Enhanced Discord formatting for Blueprint Trader AI.

Provides clean, comprehensive output for scan results.
V3 Strategy: HTF S/R + BOS + Structural TPs
"""

from __future__ import annotations

from typing import List, Optional
from strategy import ScanResult


def format_scan_summary(results: List[ScanResult]) -> str:
    """
    Format a list of scan results into a compact summary.
    Shows: Symbol | Direction | Confluence | Status
    """
    if not results:
        return "No setups found."

    results = sorted(results, key=lambda r: (-r.confluence_score, r.symbol))
    
    lines: List[str] = []
    
    for res in results:
        direction_emoji = "🟢" if res.direction == "long" else "🔴" if res.direction == "short" else "⚪"
        
        if res.status == "active":
            status_tag = "👀 ACTIVE"
        elif res.status == "watching":
            status_tag = "⏳ WATCHING"
        else:
            status_tag = "📊 SCAN"
        
        line = f"{direction_emoji} **{res.symbol}** | {res.direction.upper()} | {res.confluence_score}/5 | {status_tag}"
        lines.append(line)
    
    return "\n".join(lines)


def format_scan_group(group_name: str, results: List[ScanResult]) -> str:
    """
    Format a group of scan results with header.
    Shows compact view for multiple instruments.
    """
    if not results:
        return f"📊 **{group_name}**\n_No setups found._"

    results = sorted(results, key=lambda r: (-r.confluence_score, r.symbol))

    lines: List[str] = []
    lines.append(f"📊 **{group_name} Scan**")
    lines.append("")
    
    active_count = sum(1 for r in results if r.status == "active")
    
    if active_count > 0:
        lines.append(f"👀 {active_count} active setup(s)")
        lines.append("")
    
    for res in results:
        direction_emoji = "🟢" if res.direction == "long" else "🔴"
        
        if res.status == "active":
            status = "👀"
        elif res.status == "watching":
            status = "⏳"
        else:
            status = "📊"
        
        bos = "✓" if res.bos_level > 0 else "○"
        zone = "✓" if res.htf_zone_low > 0 else "○"
        
        line = (
            f"{status} {direction_emoji} **{res.symbol}** "
            f"| {res.confluence_score}/5 "
            f"| Zone:{zone} BOS:{bos} R:R={res.r_multiple:.1f}"
        )
        lines.append(line)
    
    return "\n".join(lines)


def format_detailed_scan(res: ScanResult) -> str:
    """
    Format a single scan result with full details.
    Used for /scan command response.
    """
    direction_emoji = "🟢" if res.direction == "long" else "🔴"
    
    if res.status == "active":
        status_line = "👀 **ACTIVE SETUP** - Entry triggered"
    elif res.status == "watching":
        status_line = "⏳ **WATCHING** - Waiting for BOS confirmation"
    else:
        status_line = "📊 **SCAN ONLY** - No actionable setup yet"
    
    lines: List[str] = []
    lines.append(f"{direction_emoji} **{res.symbol}** | {res.direction.upper()}")
    lines.append(f"Confluence: **{res.confluence_score}/5**")
    lines.append(status_line)
    lines.append("")
    
    lines.append("**V3 Strategy Analysis:**")
    
    zone_check = "✅" if res.htf_zone_low > 0 else "⚪"
    lines.append(f"{zone_check} HTF Zone: {res.htf_zone_low:.5f} - {res.htf_zone_high:.5f}")
    
    bos_check = "✅" if res.bos_level > 0 else "⚪"
    lines.append(f"{bos_check} BOS Level: {res.bos_level:.5f}")
    
    lines.append("")
    lines.append(f"**Trade Levels:**")
    lines.append(f"Entry: {res.entry:.5f}")
    lines.append(f"Stop Loss: {res.stop_loss:.5f}")
    lines.append(f"Take Profit: {res.take_profit:.5f}")
    lines.append(f"R:R = {res.r_multiple:.1f}")
    
    lines.append("")
    lines.append(f"**Reasoning:** {res.reasoning}")
    
    return "\n".join(lines)


def format_autoscan_output(markets: dict) -> List[str]:
    """
    Format autoscan results for Discord channels.
    Returns list of message strings.
    Shows only active setups.
    """
    messages: List[str] = []
    
    summary_lines = ["📊 **4H AUTOSCAN COMPLETE**", ""]
    
    total_signals = 0
    
    for group_name, scan_results in markets.items():
        if not scan_results:
            summary_lines.append(f"**{group_name.title()}**: No signals")
            continue
        
        signal_count = len(scan_results)
        total_signals += signal_count
        
        if signal_count > 0:
            summary_lines.append(f"**{group_name.title()}**: 🎯 {signal_count} signal(s)")
        else:
            summary_lines.append(f"**{group_name.title()}**: No signals")
    
    summary_lines.append("")
    summary_lines.append(f"**Total**: 🎯 {total_signals} signal(s)")
    summary_lines.append("")
    summary_lines.append("Strategy: V3 HTF S/R + BOS + Structural TPs")
    
    messages.append("\n".join(summary_lines))
    
    for group_name, scan_results in markets.items():
        if scan_results:
            group_lines = [f"", f"**{group_name.title()} Signals:**"]
            for res in scan_results:
                emoji = "🟢" if res.direction == "long" else "🔴"
                dir_text = "LONG" if res.direction == "long" else "SHORT"
                group_lines.append(
                    f"{emoji} **{res.symbol}** {dir_text} | R:R={res.r_multiple:.1f}"
                )
                if res.entry:
                    group_lines.append(f"   Entry: {res.entry:.5f} | SL: {res.stop_loss:.5f} | TP: {res.take_profit:.5f}")
                if res.reasoning:
                    group_lines.append(f"   {res.reasoning}")
                group_lines.append("")
            messages.append("\n".join(group_lines))
    
    return messages


def format_trade_update(symbol: str, direction: str, event_type: str, price: float, level: float) -> str:
    """Format a trade update message."""
    emoji = "✅" if event_type.startswith("TP") else "❌"
    
    lines = [
        f"🔔 **Trade Update - {symbol}**",
        f"Direction: {direction.upper()}",
        f"{emoji} {event_type} hit at {price:.5f}",
        f"Level: {level:.5f}"
    ]
    
    return "\n".join(lines)


def format_backtest_result(result: dict) -> str:
    """Format backtest results for Discord - V3 Strategy."""
    asset = result.get("asset", result.get("symbol", "Unknown"))
    period = result.get("period", "Unknown")
    stats = result.get("stats", result)
    
    total = stats.get("total_trades", 0)
    win_rate = stats.get("win_rate", 0.0)
    total_pnl = stats.get("total_pnl", 0.0)
    max_drawdown = stats.get("max_drawdown", 0.0)
    avg_r = stats.get("avg_r", 0.0)
    winners = stats.get("winners", 0)
    losers = stats.get("losers", 0)
    
    profit_emoji = "📈" if total_pnl > 0 else "📉" if total_pnl < 0 else "➖"
    wr_emoji = "🎯" if win_rate >= 50 else "📊" if win_rate >= 30 else "⚠️"
    
    sign = "+" if total_pnl >= 0 else ""
    
    lines = [
        f"📊 **Backtest Results - {asset}**",
        f"Period: {period}",
        f"Strategy: V3 HTF S/R + BOS + Structural TPs",
        "",
        f"**Performance:**",
        f"{profit_emoji} Total P/L: **{sign}${total_pnl:,.0f}**",
        f"{wr_emoji} Win Rate: **{win_rate:.1f}%** ({total} trades)",
        f"📊 Winners: {winners} | Losers: {losers}",
        f"📉 Max Drawdown: **${max_drawdown:,.0f}**",
        f"📈 Avg R/Trade: **{avg_r:+.2f}R**",
    ]
    
    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length."""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."
