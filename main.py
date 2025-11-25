import discord
from discord.ext import tasks

from config import (
    DISCORD_TOKEN,
    SCAN_CHANNEL_ID,
    TRADES_CHANNEL_ID,
    TRADE_UPDATES_CHANNEL_ID,
    SCAN_INTERVAL_HOURS,
    FOREX_PAIRS,
    METALS,
    INDICES,
    ENERGIES,
    CRYPTO_ASSETS,
    SIGNAL_MODE,
)

from strategy import (
    scan_single_asset,
    scan_forex,
    scan_crypto,
    scan_metals,
    scan_indices,
    scan_energies,
    scan_all_markets,
    ScanResult,
)

from formatting import (
    format_scan_group,
)

from backtest import run_backtest
from data import get_ohlcv


# ========= Global trade stores =========

# Trades that are ACTIVE (4H confirmed) and tracked for TP/SL updates.
# Key: f"{symbol}_{direction}" (e.g. "EUR_USD_bullish")
ACTIVE_TRADES: dict[str, ScanResult] = {}

# Per-trade progress: which TP(s) or SL have already been hit.
# TRADE_PROGRESS[trade_key] = {"tp1": bool, ..., "tp5": bool, "sl": bool}
TRADE_PROGRESS: dict[str, dict[str, bool]] = {}


# ========= Helpers =========

def split_message(text: str, limit: int = 1900) -> list[str]:
    """
    Split a long message into chunks under Discord's character limit.
    We split on line breaks to avoid cutting in the middle of words.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            if current:
                current += "\n" + line
            else:
                current = line

    if current:
        chunks.append(current)

    return chunks


def _ensure_trade_progress(trade_key: str) -> None:
    """
    Make sure TRADE_PROGRESS has an entry for this trade key.
    """
    if trade_key not in TRADE_PROGRESS:
        TRADE_PROGRESS[trade_key] = {
            "tp1": False,
            "tp2": False,
            "tp3": False,
            "tp4": False,
            "tp5": False,
            "sl": False,
        }


def _compute_trade_progress(idea: ScanResult) -> tuple[float, float]:
    """
    Compute (current_price, approx_RR) for a trade idea based on latest daily close.

    RR is measured from entry toward SL:
    - LONG: (current - entry) / (entry - SL)
    - SHORT: (entry - current) / (SL - entry)
    """
    candles = get_ohlcv(idea.symbol, timeframe="D", count=1)
    if not candles:
        return float("nan"), float("nan")

    current_price = candles[-1]["close"]

    if idea.entry is None or idea.stop_loss is None:
        return current_price, float("nan")

    entry = idea.entry
    sl = idea.stop_loss

    if idea.direction == "bullish":
        risk = entry - sl
        if risk <= 0:
            return current_price, float("nan")
        rr = (current_price - entry) / risk
    else:
        risk = sl - entry
        if risk <= 0:
            return current_price, float("nan")
        rr = (entry - current_price) / risk

    return current_price, rr


async def check_trade_updates(updates_channel: discord.abc.Messageable) -> None:
    """
    For every ACTIVE trade:
      - Check latest H4 (or Daily) price.
      - Detect TP1–TP5 hits or SL hit.
      - Send clean updates into TRADE_UPDATES channel.
      - Remove trade from ACTIVE_TRADES once SL or all TPs are done.
    """
    if not ACTIVE_TRADES:
        return

    # Copy keys because we may remove trades from ACTIVE_TRADES while iterating
    trade_keys = list(ACTIVE_TRADES.keys())

    for key in trade_keys:
        trade = ACTIVE_TRADES.get(key)
        if trade is None:
            continue

        # Get latest 4H price; fallback to Daily if H4 unavailable
        candles = get_ohlcv(trade.symbol, timeframe="H4", count=1)
        if not candles:
            candles = get_ohlcv(trade.symbol, timeframe="D", count=1)
        if not candles:
            continue

        price = candles[-1]["close"]
        _ensure_trade_progress(key)
        progress = TRADE_PROGRESS[key]

        entry = trade.entry
        sl = trade.stop_loss
        direction = trade.direction.lower()

        events: list[str] = []
        closed = False

        # --- SL check ---
        if sl is not None and not progress["sl"]:
            if direction == "bullish" and price <= sl:
                progress["sl"] = True
                closed = True
                events.append(f"❌ SL hit at {price:.5f} (SL {sl:.5f}).")
            elif direction == "bearish" and price >= sl:
                progress["sl"] = True
                closed = True
                events.append(f"❌ SL hit at {price:.5f} (SL {sl:.5f}).")

        # --- TP checks (TP1–TP5) ---
        tp_levels = [
            ("TP1", "tp1", trade.tp1),
            ("TP2", "tp2", trade.tp2),
            ("TP3", "tp3", trade.tp3),
            ("TP4", "tp4", trade.tp4),
            ("TP5", "tp5", trade.tp5),
        ]

        if not progress["sl"]:
            for label, flag, level in tp_levels:
                if level is None or progress[flag]:
                    continue

                if direction == "bullish" and price >= level:
                    progress[flag] = True
                    events.append(
                        f"✅ {label} hit at {price:.5f} (level {level:.5f})."
                    )
                elif direction == "bearish" and price <= level:
                    progress[flag] = True
                    events.append(
                        f"✅ {label} hit at {price:.5f} (level {level:.5f})."
                    )

        if not events:
            continue

        # Build update message
        lines: list[str] = []
        lines.append(f"🔔 **Update – {trade.symbol} {trade.direction.upper()}**")
        if entry is not None:
            lines.append(f"Entry: {entry:.5f}")
        lines.extend(events)

        # Decide whether the trade is fully 'done'
        all_tps_hit_or_unused = True
        for label, flag, level in tp_levels:
            if level is None:
                continue
            if not progress[flag]:
                all_tps_hit_or_unused = False
                break

        if progress["sl"] or all_tps_hit_or_unused:
            closed = True

        if closed:
            lines.append("Trade closed and removed from active tracking.")
            ACTIVE_TRADES.pop(key, None)
            TRADE_PROGRESS.pop(key, None)

        await updates_channel.send("\n".join(lines)[:1900])


# ========= Bot setup =========

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Bot(intents=intents)


@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user} (ID: {bot.user.id})")
    print("Blueprint Trader AI is online.")
    if not autoscan_loop.is_running():
        autoscan_loop.start()


# ========= Slash commands =========

@bot.slash_command(description="Show all available commands.")
async def help(ctx: discord.ApplicationContext):
    commands_text = """
**Blueprint Trader AI – Commands**

`/scan asset` – Manual scan of a single asset (OANDA instrument, e.g. `EUR_USD`)  
`/forex` – Scan all configured OANDA forex pairs  
`/crypto` – Scan configured crypto assets  
`/com` – Scan all commodities (metals + energies)  
`/indices` – Scan configured indices (e.g. NQ & SP500)  
`/market` – Scan all asset groups  
`/trade` – Show current active trades with a short status update  
`/live` – Show latest live price for all configured assets  
`/backtest asset period` – Backtest the Blueprint strategy over a period.  
   ➜ Example: `/backtest EUR_USD "Jan 2024 - Sep 2024"`
"""
    await ctx.respond(commands_text, ephemeral=True)


@bot.slash_command(description="Scan a single asset with the Blueprint engine.")
async def scan(ctx: discord.ApplicationContext, asset: str):
    await ctx.defer()
    result = scan_single_asset(asset)

    if not result:
        await ctx.respond(
            f"No high-confluence (≥{4 if SIGNAL_MODE == 'standard' else 3}/7) setup found on {asset} right now. "
            f"No trade idea for now."
        )
        return

    # Status line based on result.status
    if result.status == "active":
        status_line = "Status: ✅ **ACTIVE trade** (4H confirmation present)."
        trade_line = (
            f"• Trade: Entry {result.entry:.5f}, SL {result.stop_loss:.5f}, "
            f"TP1 {result.tp1:.5f}, TP2 {result.tp2:.5f}, TP3 {result.tp3:.5f}, "
            f"TP4 {result.tp4:.5f}, TP5 {result.tp5:.5f}"
        )
    elif result.status == "in_progress":
        status_line = "Status: 🕒 **In progress** (HTF zone valid, waiting for 4H confirmation)."
        trade_line = "• Trade: No active entry yet – 4H confirmation not triggered."
    else:
        status_line = "Status: 🔎 Scan only (not enough confluence for a trade idea)."
        trade_line = ""

    msg_lines = [
        f"{result.symbol} | {result.direction.upper()} | Confluence {result.confluence_score}/7",
        status_line,
        f"• Bias: {result.htf_bias}",
        f"• Location: {result.location_note}",
        f"• Fib: {result.fib_note}",
        f"• Liquidity: {result.liquidity_note}",
        f"• Structure: {result.structure_note}",
        f"• 4H: {result.confirmation_note}",
        f"• R/R: {result.rr_note}",
        f"• Summary: {result.summary_reason}",
    ]

    if trade_line:
        msg_lines.append(trade_line)

    await ctx.respond("\n".join(msg_lines)[:2000])


@bot.slash_command(description="Scan all configured OANDA forex pairs.")
async def forex(ctx: discord.ApplicationContext):
    await ctx.defer()
    scan_results, _ = scan_forex()

    if not scan_results:
        await ctx.respond("No high-confluence setups in Forex right now.")
        return

    msg = format_scan_group("Forex", scan_results)
    chunks = split_message(msg, limit=1900)

    first = True
    for chunk in chunks:
        if first:
            await ctx.respond(chunk)
            first = False
        else:
            await ctx.followup.send(chunk)


@bot.slash_command(description="Scan configured crypto assets.")
async def crypto(ctx: discord.ApplicationContext):
    await ctx.defer()
    scan_results, _ = scan_crypto()

    if not scan_results:
        await ctx.respond("No high-confluence setups in Crypto right now.")
        return

    msg = format_scan_group("Crypto", scan_results)
    chunks = split_message(msg, limit=1900)

    first = True
    for chunk in chunks:
        if first:
            await ctx.respond(chunk)
            first = False
        else:
            await ctx.followup.send(chunk)


@bot.slash_command(description="Scan all commodities (metals + energies).")
async def com(ctx: discord.ApplicationContext):
    await ctx.defer()
    scan_results_m, _ = scan_metals()
    scan_results_e, _ = scan_energies()
    combined = scan_results_m + scan_results_e

    if not combined:
        await ctx.respond("No high-confluence setups in Commodities right now.")
        return

    msg = format_scan_group("Commodities", combined)
    chunks = split_message(msg, limit=1900)

    first = True
    for chunk in chunks:
        if first:
            await ctx.respond(chunk)
            first = False
        else:
            await ctx.followup.send(chunk)


@bot.slash_command(description="Scan configured indices (e.g. NQ & SP500).")
async def indices(ctx: discord.ApplicationContext):
    await ctx.defer()
    scan_results, _ = scan_indices()

    if not scan_results:
        await ctx.respond("No high-confluence setups in Indices right now.")
        return

    msg = format_scan_group("Indices", scan_results)
    chunks = split_message(msg, limit=1900)

    first = True
    for chunk in chunks:
        if first:
            await ctx.respond(chunk)
            first = False
        else:
            await ctx.followup.send(chunk)


@bot.slash_command(description="Scan all markets (Forex, Metals, Indices, Energies, Crypto).")
async def market(ctx: discord.ApplicationContext):
    await ctx.defer()
    markets = scan_all_markets()

    first_message_sent = False

    for name in ["Forex", "Metals", "Indices", "Energies", "Crypto"]:
        scan_results, _ = markets.get(name, ([], []))
        if not scan_results:
            continue

        msg = format_scan_group(name, scan_results)
        chunks = split_message(msg, limit=1900)

        for chunk in chunks:
            if not first_message_sent:
                await ctx.respond(chunk)
                first_message_sent = True
            else:
                await ctx.followup.send(chunk)


@bot.slash_command(description="Show current active Blueprint trades.")
async def trade(ctx: discord.ApplicationContext):
    if not ACTIVE_TRADES:
        await ctx.respond("No active Blueprint trades being tracked right now.")
        return

    lines: list[str] = []
    lines.append("📈 **Active Blueprint trades**")

    for key, t in ACTIVE_TRADES.items():
        # Guard against None values
        entry = t.entry if t.entry is not None else 0.0
        sl = t.stop_loss if t.stop_loss is not None else 0.0
        tp1 = t.tp1 if t.tp1 is not None else 0.0
        tp2 = t.tp2 if t.tp2 is not None else 0.0
        tp3 = t.tp3 if t.tp3 is not None else 0.0
        tp4 = t.tp4 if t.tp4 is not None else 0.0
        tp5 = t.tp5 if t.tp5 is not None else 0.0

        current_price, rr = _compute_trade_progress(t)
        if rr == rr:  # not NaN
            rr_str = f"{rr:+.2f}R"
        else:
            rr_str = "N/A"

        lines.append(
            f"• **{t.symbol}** | {t.direction.upper()} | Confluence {t.confluence_score}/7"
        )
        lines.append(
            f"  Entry: {entry:.5f} | SL: {sl:.5f} | "
            f"TP1: {tp1:.5f} | TP2: {tp2:.5f} | TP3: {tp3:.5f} | "
            f"TP4: {tp4:.5f} | TP5: {tp5:.5f}"
        )
        lines.append(
            f"  Status: {t.status} | Last D close: {current_price:.5f} | Progress: {rr_str}"
        )
        lines.append(f"  Bias: {t.htf_bias}")
        lines.append("")

    msg = "\n".join(lines)
    await ctx.respond(msg[:2000])


@bot.slash_command(description="Show the latest live price for all configured assets.")
async def live(ctx: discord.ApplicationContext):
    await ctx.defer()
    groups = {
        "Forex": FOREX_PAIRS,
        "Metals": METALS,
        "Indices": INDICES,
        "Energies": ENERGIES,
        "Crypto": CRYPTO_ASSETS,
    }

    lines: list[str] = []
    lines.append("**Live prices (latest daily candle close)**")
    lines.append("")

    for name, symbols in groups.items():
        lines.append(f"__{name}__")
        if not symbols:
            lines.append("_No instruments configured._")
            lines.append("")
            continue

        for sym in symbols:
            candles = get_ohlcv(sym, timeframe="D", count=1)
            if not candles:
                lines.append(f"{sym}: N/A")
            else:
                price = candles[-1]["close"]
                lines.append(f"{sym}: `{price:.5f}`")
        lines.append("")

    msg = "\n".join(lines)
    chunks = split_message(msg, limit=1900)

    first = True
    for chunk in chunks:
        if first:
            await ctx.respond(chunk)
            first = False
        else:
            await ctx.followup.send(chunk)


@bot.slash_command(
    description='Backtest an asset over a date range. Usage: /backtest EUR_USD "Jan 2024 - Sep 2024".'
)
async def backtest(ctx: discord.ApplicationContext, asset: str, period: str):
    await ctx.defer()
    result = run_backtest(asset, period)

    total = result["total_trades"]
    win_rate = result["win_rate"]
    net_ret = result["net_return_pct"]
    trades = result.get("trades", [])

    tp1_count = sum(1 for t in trades if t.get("exit_reason") == "TP1")
    tp2_count = sum(1 for t in trades if t.get("exit_reason") == "TP2")
    tp3_count = sum(1 for t in trades if t.get("exit_reason") == "TP3")
    sl_count = sum(1 for t in trades if t.get("exit_reason") == "SL")

    msg_lines: list[str] = []
    msg_lines.append(f"📊 **Backtest – {result['asset']}**")
    msg_lines.append(f"Period: {result['period']}")
    msg_lines.append(f"Total trades: {total}")
    msg_lines.append(f"Winrate: {win_rate:.1f}%")
    msg_lines.append(f"Net return (1% risk/trade): {net_ret:.1f}%")
    msg_lines.append("")

    if total > 0:
        msg_lines.append("Exit breakdown:")
        msg_lines.append(f"- TP1 hits: {tp1_count}")
        msg_lines.append(f"- TP2 hits: {tp2_count}")
        msg_lines.append(f"- TP3 hits: {tp3_count}")
        msg_lines.append(f"- SL hits:  {sl_count}")
        msg_lines.append("")

    msg_lines.append(f"Note: {result['notes']}")

    await ctx.respond("\n".join(msg_lines))



# ========= Autoscan loop =========

@tasks.loop(hours=SCAN_INTERVAL_HOURS)
async def autoscan_loop():
    await bot.wait_until_ready()
    print("⏱️ Running 4H autoscan...")

    scan_channel = bot.get_channel(SCAN_CHANNEL_ID)
    trades_channel = bot.get_channel(TRADES_CHANNEL_ID)

    if scan_channel is None:
        print("❌ Scan channel not found.")
        return

    markets = scan_all_markets()  # {'Forex': (scan_results, trade_ideas), ...}

    # 1) Post scan summaries per market
    for group_name, (scan_results, trade_ideas) in markets.items():
        if not scan_results:
            continue

        lines: list[str] = []
        lines.append(f"📊 **{group_name} 4H autoscan**")

        for res in scan_results:
            status_tag = (
                "ACTIVE" if res.status == "active"
                else "INP" if res.status == "in_progress"
                else "SCAN"
            )
            lines.append(
                f"{res.symbol} | {res.direction.upper()} | {res.confluence_score}/7 ({status_tag}) – "
                f"loc={'Y' if 'price in upper' in res.location_note or 'price in lower' in res.location_note else 'N'}, "
                f"fib={'Y' if 'golden pocket' in res.fib_note else 'N'}, "
                f"liq={'Y' if 'liquidity' in res.liquidity_note else 'N'}, "
                f"struct={'Y' if 'Structure supports' in res.structure_note else 'N'}, "
                f"4H={'Y' if 'H4: structure aligned' in res.confirmation_note else 'N'}, "
                f"rr={'Y' if 'Approx first target' in res.rr_note else 'N'}"
            )

        for chunk in split_message("\n".join(lines), limit=1900):
            await scan_channel.send(chunk)

        # 2) Handle new ACTIVE trades
        if trades_channel is not None:
            for trade in trade_ideas:
                if trade.status != "active":
                    continue

                trade_key = f"{trade.symbol}_{trade.direction}"
                if trade_key in ACTIVE_TRADES:
                    continue

                ACTIVE_TRADES[trade_key] = trade
                _ensure_trade_progress(trade_key)

                t_lines: list[str] = []
                t_lines.append(f"🎯 **New {trade.direction.upper()} trade – {trade.symbol}**")
                t_lines.append(f"Confluence: {trade.confluence_score}/7")

                if trade.entry is not None and trade.stop_loss is not None:
                    t_lines.append(
                        f"Entry: {trade.entry:.5f} | SL: {trade.stop_loss:.5f}"
                    )
                if trade.tp1 is not None:
                    t_lines.append(
                        f"TP1: {trade.tp1:.5f} | TP2: {trade.tp2:.5f} | "
                        f"TP3: {trade.tp3:.5f} | TP4: {trade.tp4:.5f} | TP5: {trade.tp5:.5f}"
                    )

                t_lines.append(f"Bias: {trade.htf_bias}")
                t_lines.append(f"Location: {trade.location_note}")
                t_lines.append(f"Fib: {trade.fib_note}")
                t_lines.append(f"Liquidity: {trade.liquidity_note}")
                t_lines.append(f"Structure: {trade.structure_note}")
                t_lines.append(f"4H: {trade.confirmation_note}")

                await trades_channel.send("\n".join(t_lines)[:1900])

    # 3) Check TP/SL updates for ACTIVE trades and send to TRADE_UPDATES channel
    updates_channel = bot.get_channel(TRADE_UPDATES_CHANNEL_ID)
    if updates_channel is not None and ACTIVE_TRADES:
        await check_trade_updates(updates_channel)

    print("✅ Autoscan finished.")


# ========= Run bot =========

if not DISCORD_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN not found. Set it in Replit Secrets.")

bot.run(DISCORD_TOKEN)
