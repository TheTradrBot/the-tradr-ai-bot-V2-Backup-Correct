# formatting.py
"""
Formatting of Blueprint Trader AI scan and trade messages for Discord.
"""

from typing import List, Tuple
from strategy import ScanResult


def format_scan_result(res: ScanResult) -> str:
    """
    Short, clean scan summary (no entries/SL), for scan channel or /scan.
    """
    dir_emoji = "🟢" if res.direction == "bullish" else "🔴"
    return (
        f"{dir_emoji} **{res.symbol}** | "
        f"{res.direction.upper()} | Confluence **{res.confluence_score}/7**\n"
        f"• Bias: {res.htf_bias}\n"
        f"• Location: {res.location_note}\n"
        f"• Fib: {res.fib_note}\n"
        f"• Liquidity: {res.liquidity_note}\n"
        f"• 4H: {res.confirmation_note}\n"
        f"• R/R: {res.rr_note}\n"
        f"• Summary: {res.summary_reason}"
    )


def format_scan_group(
    group_name: str,
    results: List[ScanResult],
) -> str:
    """
    Formatted block for one group (e.g. Forex, Metals) for autoscan.
    Only pass results with confluence >= 5.
    """
    if not results:
        return f"**{group_name}**\n_No high-confluence setups (≥5/7) right now._"

    header = f"**{group_name} – High-confluence setups (≥5/7)**"
    lines: List[str] = [header, ""]
    for res in results:
        lines.append(format_scan_result(res))
        lines.append("")  # blank line between assets

    return "\n".join(lines)


def format_trade_idea(res: ScanResult, is_new: bool = True) -> str:
    """
    Detailed trade idea message for TRADES channel when confluence >= 6.
    """
    dir_emoji = "📈" if res.direction == "bullish" else "📉"
    status_txt = res.status or "active"
    header = f"{dir_emoji} **{res.symbol} {res.direction.upper()} IDEA**"

    entry = res.entry or 0.0
    sl = res.stop_loss or 0.0
    tp1 = res.tp1 or 0.0
    tp2 = res.tp2 or 0.0
    tp3 = res.tp3 or 0.0

    rr1 = f"{res.rr1:.1f}R" if res.rr1 is not None else "n/a"
    rr2 = f"{res.rr2:.1f}R" if res.rr2 is not None else "n/a"
    rr3 = f"{res.rr3:.1f}R" if res.rr3 is not None else "n/a"

    body = (
        f"Status: **{status_txt}**\n\n"
        f"**Entry:** {entry:.5f}\n"
        f"**Stop:** {sl:.5f}\n"
        f"**TP1:** {tp1:.5f} ({rr1})\n"
        f"**TP2:** {tp2:.5f} ({rr2})\n"
        f"**TP3:** {tp3:.5f} ({rr3})\n\n"
        f"**Confluence:** {res.confluence_score}/7\n"
        f"- {res.htf_bias}\n"
        f"- {res.location_note}\n"
        f"- {res.fib_note}\n"
        f"- {res.liquidity_note}\n"
        f"- 4H: {res.confirmation_note}\n"
        f"- {res.rr_note}\n\n"
        f"**Blueprint reasoning:**\n{res.summary_reason}\n\n"
        f"⚠️ Educational only. Not financial advice."
    )

    return f"{header}\n\n{body}"


def format_trade_overview(
    trades_info: List[Tuple[ScanResult, float, float]]
) -> str:
    """
    Format a summary of all active trades for /trade.
    trades_info: list of (ScanResult, current_price, current_rr)
    """
    if not trades_info:
        return "No active Blueprint trades being tracked right now."

    lines: List[str] = []
    lines.append("**Current active Blueprint trades**")
    lines.append("")

    for idea, current_price, current_rr in trades_info:
        dir_emoji = "📈" if idea.direction == "bullish" else "📉"
        status = idea.status or ("closed" if idea.is_closed else "active")

        entry_str = f"{idea.entry:.5f}" if idea.entry is not None else "n/a"
        sl_str = f"{idea.stop_loss:.5f}" if idea.stop_loss is not None else "n/a"
        tp1_str = f"{idea.tp1:.5f}" if idea.tp1 is not None else "n/a"
        tp2_str = f"{idea.tp2:.5f}" if idea.tp2 is not None else "n/a"
        tp3_str = f"{idea.tp3:.5f}" if idea.tp3 is not None else "n/a"

        if current_price != current_price:  # NaN check
            price_str = "n/a"
        else:
            price_str = f"{current_price:.5f}"

        if current_rr != current_rr:
            rr_str = "n/a"
        else:
            rr_str = f"{current_rr:.2f}R"

        lines.append(
            f"{dir_emoji} **{idea.symbol} {idea.direction.upper()}** | Status: **{status}**\n"
            f"Entry: `{entry_str}` | SL: `{sl_str}`\n"
            f"TP1: `{tp1_str}` | TP2: `{tp2_str}` | TP3: `{tp3_str}`\n"
            f"Current: `{price_str}` | Approx progress: `{rr_str}`\n"
        )
        lines.append("")

    return "\n".join(lines)


def format_trade_update(
    idea: ScanResult,
    event_type: str,
    event_price: float,
    event_rr: float,
) -> str:
    """
    Format a TP/SL hit update for TRADE_UPDATES channel.
    event_type: "TP1", "TP2", "TP3", or "SL"
    """
    dir_emoji = "📈" if idea.direction == "bullish" else "📉"
    pair_label = f"{idea.symbol} {idea.direction.upper()}"

    # R multiple
    rr_str = "n/a" if event_rr != event_rr else f"{event_rr:.2f}R"

    entry_str = f"{idea.entry:.5f}" if idea.entry is not None else "n/a"
    price_str = f"{event_price:.5f}"

    if event_type == "SL":
        icon = "❌"
        sl_str = f"{idea.stop_loss:.5f}" if idea.stop_loss is not None else "n/a"
        status_txt = idea.status or "closed"

        return (
            f"{icon} {dir_emoji} **{pair_label} – Stoploss hit**\n\n"
            f"Entry: `{entry_str}` | SL: `{sl_str}`\n"
            f"Hit price: `{price_str}`\n"
            f"Result: `{rr_str}`\n"
            f"Status: **{status_txt}**\n"
            f"⚠️ Educational only. Not financial advice."
        )

    # TP events
    icon = "✅"
    status_txt = idea.status or ("closed" if idea.is_closed else "active")

    tp_val = None
    if event_type == "TP1":
        tp_val = idea.tp1
    elif event_type == "TP2":
        tp_val = idea.tp2
    elif event_type == "TP3":
        tp_val = idea.tp3

    tp_str = f"{tp_val:.5f}" if tp_val is not None else "n/a"

    return (
        f"{icon} {dir_emoji} **{pair_label} – {event_type} hit**\n\n"
        f"Entry: `{entry_str}` | {event_type}: `{tp_str}`\n"
        f"Hit price: `{price_str}`\n"
        f"Approx result: `{rr_str}`\n"
        f"Status: **{status_txt}**\n"
        f"⚠️ Educational only. Not financial advice."
    )
