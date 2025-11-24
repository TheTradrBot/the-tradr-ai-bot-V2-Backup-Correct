# strategy.py
"""
Blueprint Trader AI – TOP-DOWN strategy logic.

Implements a confluence-based HTF swing framework inspired by strategy_spec.md,
with an explicit top-down flow:

1) Monthly & Weekly:
   - Build S/R from swing highs/lows (clustered levels).
   - Identify basic supply/demand zone *from the last weekly impulse origin*.
   - Infer Weekly trend (HH/HL vs LH/LL).

2) Weekly Fibonacci:
   - Identify last meaningful Weekly impulse leg in direction of trend.
   - Compute Weekly golden pocket (0.618–0.796) using body → wick.

3) Daily context:
   - Infer Daily structure (trend / range).
   - Compute Daily range & location.
   - Optionally find a Daily impulse & golden pocket (secondary fib).
   - Detect Daily equal highs/lows liquidity pools.
   - Detect structural frameworks (H&S, inverse H&S, Bullish N, Bearish V)
     in context with HTF S/R + OB + Fibonacci.

4) 4H:
   - Confirmation ONLY: simple BOS-style breakout on H4.

Confluence score (0–7):

  1. HTF bias (Monthly+Weekly+Daily direction alignment or valid HTF reversal)
  2. Location (Monthly/Weekly/Daily S/R edges + HTF supply/demand)
  3. Fibonacci (Weekly primary, Daily secondary)
  4. Liquidity (equal highs/lows, magnets)
  5. 4H confirmation (BOS)
  6. R/R (from entry & SL)
  7. Structure (Weekly & Daily structure + frameworks fit the trade direction)

Only when confluence >= threshold and 4H confirms do we create an ACTIVE trade.
Otherwise high-confluence setups stay as IN_PROGRESS (watchlist).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Dict

from data import get_ohlcv
from config import (
    FOREX_PAIRS,
    METALS,
    INDICES,
    ENERGIES,
    CRYPTO_ASSETS,
    SIGNAL_MODE,
)

Direction = Literal["bullish", "bearish"]


# ===== Result dataclass =====


@dataclass
class ScanResult:
    symbol: str
    direction: Direction
    confluence_score: int
    timeframe: str  # e.g. "Weekly→Daily with H4 confirmation"
    htf_bias: str
    location_note: str
    fib_note: str
    liquidity_note: str
    structure_note: str
    confirmation_note: str
    rr_note: str
    summary_reason: str
    # trade fields
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    rr1: Optional[float] = None
    rr2: Optional[float] = None
    rr3: Optional[float] = None
    status: Optional[str] = None  # "active" / "in_progress" / None

    # lifecycle flags for TP/SL
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    sl_hit: bool = False
    is_closed: bool = False


# ===== Basic candle helpers =====


def _body_price(c: Dict[str, float]) -> float:
    """
    Return the 'start' of the candle body for fib anchor:
    - If candle is bullish: use open (start of up-move).
    - If bearish or doji-ish: use close (start of down-move).
    """
    o = c["open"]
    cl = c["close"]
    return o if cl > o else cl


def _is_bullish(c: Dict[str, float]) -> bool:
    return c["close"] > c["open"]


def _is_bearish(c: Dict[str, float]) -> bool:
    return c["close"] < c["open"]


# ===== Swing detection =====


def _find_swings(
    candles: List[Dict[str, float]],
    lookback: int = 2,
) -> List[Tuple[int, str, float]]:
    """
    Very simple fractal-based swing detector on highs/lows.

    Returns list of (index, 'high'/'low', price) sorted by index.
    """
    n = len(candles)
    swings: List[Tuple[int, str, float]] = []

    if n < 2 * lookback + 1:
        return swings

    for i in range(lookback, n - lookback):
        high = candles[i]["high"]
        low = candles[i]["low"]

        # swing high
        if all(high >= candles[i - k]["high"] for k in range(1, lookback + 1)) and \
           all(high >= candles[i + k]["high"] for k in range(1, lookback + 1)):
            swings.append((i, "high", high))

        # swing low
        if all(low <= candles[i - k]["low"] for k in range(1, lookback + 1)) and \
           all(low <= candles[i + k]["low"] for k in range(1, lookback + 1)):
            swings.append((i, "low", low))

    swings.sort(key=lambda x: x[0])
    return swings


# ===== Trend inference (generic) =====


def _infer_trend(
    candles: List[Dict[str, float]],
    swings: List[Tuple[int, str, float]],
    label: str,
) -> Tuple[str, str]:
    """
    Infer trend from last few swing highs/lows.

    Returns:
      (trend, note)
      trend in {"bullish", "bearish", "range"}
    """
    if len(swings) < 4:
        return "range", f"{label} structure unclear (not enough swings)."

    last_swings = swings[-6:]
    highs = [s for s in last_swings if s[1] == "high"]
    lows = [s for s in last_swings if s[1] == "low"]

    if len(highs) < 2 or len(lows) < 2:
        return "range", f"{label} structure lacks enough swing highs/lows."

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]

    if h2[2] > h1[2] and l2[2] > l1[2]:
        note = (
            f"{label} HH/HL structure "
            f"(highs {h1[2]:.5f} → {h2[2]:.5f}, "
            f"lows {l1[2]:.5f} → {l2[2]:.5f})."
        )
        return "bullish", note
    elif h2[2] < h1[2] and l2[2] < l1[2]:
        note = (
            f"{label} LH/LL structure "
            f"(highs {h1[2]:.5f} → {h2[2]:.5f}, "
            f"lows {l1[2]:.5f} → {l2[2]:.5f})."
        )
        return "bearish", note
    else:
        return "range", f"{label} swings mixed (no clean HH/HL or LH/LL)."


# ===== Range & location (generic) =====


def _range_and_location(
    candles: List[Dict[str, float]],
    close_price: float,
    label: str,
) -> Tuple[float, float, float, str, bool]:
    """
    Compute basic range and where current price sits in it for a timeframe.

    Returns:
      (range_low, range_high, position, location_note, is_edge)
      position in [0,1]: 0 = range_low, 1 = range_high
      is_edge = True if near bottom/top (good location), False if mid-range.
    """
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    range_high = max(highs)
    range_low = min(lows)
    span = range_high - range_low
    if span <= 0:
        return range_low, range_high, 0.5, f"{label} range degenerated (no span).", False

    pos = (close_price - range_low) / span  # 0..1

    # Define edges based on signal mode
    if SIGNAL_MODE == "aggressive":
        lower_edge = 0.35
        upper_edge = 0.65
    else:  # "standard"
        lower_edge = 0.25
        upper_edge = 0.75

    if pos <= lower_edge:
        location_note = (
            f"{label}: price in lower edge of range "
            f"({range_low:.5f}–{range_high:.5f}), near support area."
        )
        is_edge = True
    elif pos >= upper_edge:
        location_note = (
            f"{label}: price in upper edge of range "
            f"({range_low:.5f}–{range_high:.5f}), near resistance area."
        )
        is_edge = True
    else:
        location_note = (
            f"{label}: price mid-range "
            f"({range_low:.5f}–{range_high:.5f}), no clear edge."
        )
        is_edge = False

    return range_low, range_high, pos, location_note, is_edge


# ===== S/R from swings (clustered, tiered) =====


def _sr_context_from_swings(
    candles: List[Dict[str, float]],
    swings: List[Tuple[int, str, float]],
    close_price: float,
    label: str,
) -> Tuple[bool, str]:
    """
    Build horizontal S/R from clustered swing highs/lows, with a notion of "tier":

      - Tier 1: Monthly, 4+ taps (strong multi-year level).
      - Tier 2: Monthly/Weekly, 2–3 taps (solid HTF level).
      - Tier 3: Daily, 2+ taps (local S/R refinement).

    We then check if current price is near any Tier 1–2 level on HTF (Monthly/Weekly),
    or Tier 2–3 on Daily.

    A level is also labelled by role:
      - Mostly highs → resistance bias.
      - Mostly lows → support bias.
      - Mixed → flip level.
    """
    if not swings:
        return False, f"{label}: no clear swings yet for S/R."

    # Cluster levels by price proximity
    raw = sorted(swings, key=lambda s: s[2])
    clusters: List[Dict] = []
    # tolerance ~0.15% of price
    for idx, kind, price in raw:
        if not clusters:
            clusters.append({
                "center": price,
                "taps": 1,
                "high_hits": 1 if kind == "high" else 0,
                "low_hits": 1 if kind == "low" else 0,
            })
            continue
        last = clusters[-1]
        center = last["center"]
        tol = center * 0.0015
        if abs(price - center) <= tol:
            # same cluster
            new_taps = last["taps"] + 1
            new_center = (center * last["taps"] + price) / new_taps
            last["center"] = new_center
            last["taps"] = new_taps
            if kind == "high":
                last["high_hits"] += 1
            else:
                last["low_hits"] += 1
        else:
            clusters.append({
                "center": price,
                "taps": 1,
                "high_hits": 1 if kind == "high" else 0,
                "low_hits": 1 if kind == "low" else 0,
            })

    if not clusters:
        return False, f"{label}: no raw S/R clusters."

    # Assign tier & role
    for c in clusters:
        taps = c["taps"]
        # tier
        if label == "Monthly" and taps >= 4:
            tier = 1
        elif label in ("Monthly", "Weekly") and taps >= 2:
            tier = 2
        else:
            # Daily or weaker
            tier = 3
        c["tier"] = tier

        # role
        if c["high_hits"] > c["low_hits"]:
            role = "resistance"
        elif c["low_hits"] > c["high_hits"]:
            role = "support"
        else:
            role = "flip"
        c["role"] = role

    # Find nearest cluster
    nearest = min(clusters, key=lambda c: abs(c["center"] - close_price))
    dist_frac = abs(close_price - nearest["center"]) / max(nearest["center"], 1e-8)

    # Distance threshold ~0.25% of price for HTF, slightly tighter for Daily
    if label in ("Monthly", "Weekly"):
        max_frac = 0.0025
        allowed_tiers = {1, 2}
    else:
        max_frac = 0.0018
        allowed_tiers = {2, 3}

    is_near = dist_frac <= max_frac and nearest["tier"] in allowed_tiers

    role_txt = nearest["role"]
    tier_txt = nearest["tier"]
    taps_txt = nearest["taps"]
    lvl = nearest["center"]

    if is_near:
        note = (
            f"{label}: price near Tier {tier_txt} {role_txt} S/R "
            f"around {lvl:.5f} (≈{taps_txt} taps, dist≈{dist_frac*100:.2f}%)."
        )
        return True, note
    else:
        note = (
            f"{label}: nearest Tier {tier_txt} {role_txt} S/R at "
            f"{lvl:.5f} (≈{taps_txt} taps, dist≈{dist_frac*100:.2f}%)."
        )
        return False, note


# ===== Fibonacci & impulse legs (generic) =====


def _find_impulse_leg(
    candles: List[Dict[str, float]],
    swings: List[Tuple[int, str, float]],
    trend: str,
    min_move_frac: float = 0.005,
) -> Optional[Tuple[int, int, float, float]]:
    """
    Identify last meaningful impulse leg in direction of trend.

    Returns:
      (start_index, end_index, start_price_body, end_price_wick)
      or None if no valid leg.
    """
    if trend not in ("bullish", "bearish"):
        return None
    if len(swings) < 4:
        return None

    sw = swings[-8:]

    if trend == "bullish":
        highs = [s for s in sw if s[1] == "high"]
        lows = [s for s in sw if s[1] == "low"]
        if not highs or not lows:
            return None
        last_high = highs[-1]
        low_candidates = [s for s in lows if s[0] < last_high[0]]
        if not low_candidates:
            return None
        last_low = low_candidates[-1]

        start_idx = last_low[0]
        end_idx = last_high[0]
        if end_idx <= start_idx:
            return None

        start_c = candles[start_idx]
        end_c = candles[end_idx]

        start_price = _body_price(start_c)
        end_price = end_c["high"]
        if end_price <= start_price:
            return None

        if (end_price - start_price) / max(start_price, 1e-8) < min_move_frac:
            return None

        return start_idx, end_idx, start_price, end_price

    # bearish
    highs = [s for s in sw if s[1] == "high"]
    lows = [s for s in sw if s[1] == "low"]
    if not highs or not lows:
        return None
    last_low = lows[-1]
    high_candidates = [s for s in highs if s[0] < last_low[0]]
    if not high_candidates:
        return None
    last_high = high_candidates[-1]

    start_idx = last_high[0]
    end_idx = last_low[0]
    if end_idx <= start_idx:
        return None

    start_c = candles[start_idx]
    end_c = candles[end_idx]

    start_price = _body_price(start_c)
    end_price = end_c["low"]
    if end_price >= start_price:
        return None

    if (start_price - end_price) / max(start_price, 1e-8) < min_move_frac:
        return None

    return start_idx, end_idx, start_price, end_price


def _fib_retrace_zone(
    start_price: float,
    end_price: float,
) -> Tuple[float, float]:
    """
    Return price band for 0.618..0.796 retracement.
    Direction-aware based on start/end prices.
    """
    diff = end_price - start_price
    gp1 = start_price + 0.618 * diff
    gp2 = start_price + 0.796 * diff
    return min(gp1, gp2), max(gp1, gp2)


def _check_fib_confluence(
    leg: Optional[Tuple[int, int, float, float]],
    current_price: float,
    label: str,
) -> Tuple[bool, str, Optional[float], Optional[float]]:
    """
    Check if current price sits in golden pocket of last impulse leg.

    Returns:
      (fib_ok, note, gp_low, gp_high)
    """
    if leg is None:
        return False, f"{label}: no clear impulse leg for fib.", None, None

    _, _, start_price, end_price = leg
    gp_low, gp_high = _fib_retrace_zone(start_price, end_price)

    if gp_low <= current_price <= gp_high:
        note = (
            f"{label}: price inside golden pocket "
            f"({gp_low:.5f}–{gp_high:.5f}) of last impulse leg."
        )
        return True, note, gp_low, gp_high
    else:
        note = (
            f"{label}: price outside golden pocket "
            f"({gp_low:.5f}–{gp_high:.5f}) of last impulse leg."
        )
        return False, note, gp_low, gp_high


# ===== Supply/Demand from impulse origin (Monthly/Weekly/Daily) =====


def _ob_context_from_impulse(
    candles: List[Dict[str, float]],
    direction: Direction,
    leg: Optional[Tuple[int, int, float, float]],
    current_price: float,
    label: str,
) -> Tuple[bool, str]:
    """
    Build a supply/demand zone from the origin of the last impulse, closer to the spec:

      - For bullish context: demand zone from the last cluster of bearish candles
        before the impulsive rally that forms the impulse leg.
      - For bearish context: supply zone from the last cluster of bullish candles
        before the impulsive selloff that forms the impulse leg.

    We then check whether current price is:
      - inside this zone (ideal),
      - or clearly above/below it (potential future retest area).

    We also try to detect if the zone has already been "used" (tapped multiple times)
    since the impulse, by counting deep re-entries into the zone.
    """
    if leg is None or not candles:
        return False, f"{label}: no impulse origin for supply/demand."

    start_idx, end_idx, _, _ = leg
    if start_idx <= 0 or start_idx >= len(candles):
        return False, f"{label}: invalid impulse origin for supply/demand."

    # 1) Find last 1–3 opposing candles before the impulse start
    base_indices: List[int] = []
    i = start_idx
    max_span = 3  # up to 3-candle cluster

    if direction == "bullish":
        # We want bearish candles before the rally
        j = i
        while j >= 0 and len(base_indices) < max_span:
            if _is_bearish(candles[j]):
                base_indices.append(j)
            else:
                if base_indices:
                    break
            j -= 1
        zone_type = "demand"
    else:
        # bearish direction → we want bullish base candles
        j = i
        while j >= 0 and len(base_indices) < max_span:
            if _is_bullish(candles[j]):
                base_indices.append(j)
            else:
                if base_indices:
                    break
            j -= 1
        zone_type = "supply"

    if not base_indices:
        # fallback: just use the impulse-start candle
        base_indices = [start_idx]

    # 2) Build zone bounds from this base cluster
    lows = [candles[k]["low"] for k in base_indices]
    highs = [candles[k]["high"] for k in base_indices]
    opens = [candles[k]["open"] for k in base_indices]
    closes = [candles[k]["close"] for k in base_indices]

    if direction == "bullish":
        zone_low = min(lows)
        zone_high = max(opens + closes)  # body top
    else:
        zone_low = min(opens + closes)   # body bottom
        zone_high = max(highs)

    # 3) Check "fresh vs used" – look after the impulse for taps into the zone
    taps_after = 0
    for idx in range(end_idx + 1, len(candles)):
        c = candles[idx]
        if c["low"] <= zone_high and c["high"] >= zone_low:
            taps_after += 1

    if taps_after == 0:
        freshness = "fresh"
    elif taps_after == 1:
        freshness = "first retest"
    else:
        freshness = "used"

    # 4) Check where current price is relative to the zone
    tol = current_price * 0.0005  # small buffer (~0.05%)
    inside = (zone_low - tol) <= current_price <= (zone_high + tol)

    if inside:
        note = (
            f"{label}: price inside {freshness} {zone_type} zone from last impulse origin "
            f"[{zone_low:.5f}–{zone_high:.5f}]."
        )
        return True, note
    else:
        if current_price < zone_low:
            relation = "below"
        elif current_price > zone_high:
            relation = "above"
        else:
            relation = "between"

        note = (
            f"{label}: current price {relation} {freshness} {zone_type} zone from last impulse origin "
            f"[{zone_low:.5f}–{zone_high:.5f}]."
        )
        return False, note


# ===== Liquidity: equal highs/lows (Daily) =====


def _find_equal_levels(
    candles: List[Dict[str, float]],
    tolerance_frac: float = 0.0005,
) -> Tuple[List[float], List[float]]:
    """
    Approximate equal highs/lows detection.

    tolerance_frac: fraction of price (e.g. 0.0005 = 0.05%).
    Returns:
      (equal_highs, equal_lows)
    """
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    equal_highs: List[float] = []
    equal_lows: List[float] = []

    if len(candles) < 5:
        return equal_highs, equal_lows

    # near-equal highs
    for i in range(1, len(candles) - 1):
        h = highs[i]
        for j in range(i + 1, min(i + 5, len(candles))):
            h2 = highs[j]
            mid = (h + h2) / 2.0
            tol = mid * tolerance_frac
            if abs(h - h2) <= tol:
                equal_highs.append(mid)
                break

    # near-equal lows
    for i in range(1, len(candles) - 1):
        l = lows[i]
        for j in range(i + 1, min(i + 5, len(candles))):
            l2 = lows[j]
            mid = (l + l2) / 2.0
            tol = mid * tolerance_frac
            if abs(l - l2) <= tol:
                equal_lows.append(mid)
                break

    # de-duplicate
    def _dedupe(levels: List[float]) -> List[float]:
        levels = sorted(levels)
        if not levels:
            return levels
        deduped = [levels[0]]
        for lv in levels[1:]:
            if abs(lv - deduped[-1]) > deduped[-1] * tolerance_frac * 2:
                deduped.append(lv)
        return deduped

    return _dedupe(equal_highs), _dedupe(equal_lows)


def _liquidity_context(
    candles: List[Dict[str, float]],
    current_price: float,
) -> Tuple[bool, str]:
    """
    Basic liquidity context via equal highs/lows:

    - Detect equal highs/lows (magnetic levels).
    - Check if price is near one of them.
    """
    equal_highs, equal_lows = _find_equal_levels(candles)
    if not equal_highs and not equal_lows:
        return False, "Daily: no clear equal highs/lows liquidity pools."

    all_levels = [(lv, "high") for lv in equal_highs] + [(lv, "low") for lv in equal_lows]
    if not all_levels:
        return False, "Daily: no clear equal highs/lows liquidity pools."

    nearest_level, kind = min(all_levels, key=lambda x: abs(x[0] - current_price))
    dist = abs(current_price - nearest_level) / max(nearest_level, 1e-8)

    if dist <= 0.002:  # within ~0.2%
        note = f"Daily: price sitting near {kind} liquidity pool around {nearest_level:.5f}."
        return True, note
    else:
        note = f"Daily: nearest {kind} liquidity pool at {nearest_level:.5f}, not yet engaged."
        return False, note


# ===== 4H confirmation =====


def _h4_confirmation(
    symbol: str,
    direction: Direction,
) -> Tuple[bool, str]:
    """
    Simple 4H BOS-type confirmation:

    - Bullish: last close > max high of previous N candles.
    - Bearish: last close < min low of previous N candles.
    """
    h4 = get_ohlcv(symbol, timeframe="H4", count=60)
    if not h4 or len(h4) < 10:
        return False, "H4: no sufficient data for confirmation."

    prev = h4[:-1]
    last = h4[-1]
    prev_high = max(c["high"] for c in prev)
    prev_low = min(c["low"] for c in prev)
    last_close = last["close"]

    if direction == "bullish":
        if last_close > prev_high:
            return True, "H4 bullish BOS (last close broke above recent H4 highs)."
        else:
            return False, "H4: no bullish BOS above recent H4 highs."
    else:
        if last_close < prev_low:
            return True, "H4 bearish BOS (last close broke below recent H4 lows)."
        else:
            return False, "H4: no bearish BOS below recent H4 lows."


# ===== R/R & trade structure =====


def _build_trade_from_context(
    symbol: str,  # unused, kept for compatibility
    direction: Direction,
    daily_candles: List[Dict[str, float]],
    daily_swings: List[Tuple[int, str, float]],
    current_price: float,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    str,
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    Build entry/SL/TP1/TP2/TP3 and R-multiples roughly according to Blueprint logic.

    Simplified rules:
      - Entry: current Daily close.
      - SL: beyond nearest swing low/high in direction of trade.
      - TP1/TP2/TP3: at 1.5R, 2.5R, 4R.
    """
    if not daily_candles:
        return (None, None, None, None, None, "No candles for R/R.", None, None, None)

    entry = current_price

    lows = [s for s in daily_swings if s[1] == "low"]
    highs = [s for s in daily_swings if s[1] == "high"]

    if direction == "bullish":
        if not lows:
            return (None, None, None, None, None, "No Daily swing lows for SL.", None, None, None)
        last_low = lows[-1][2]
        sl = last_low * (1.0 - 0.001)  # small buffer
        risk = entry - sl
        if risk <= 0:
            return (None, None, None, None, None, "Invalid risk (entry <= SL).", None, None, None)

        tp1 = entry + 1.5 * risk
        tp2 = entry + 2.5 * risk
        tp3 = entry + 4.0 * risk
    else:
        if not highs:
            return (None, None, None, None, None, "No Daily swing highs for SL.", None, None, None)
        last_high = highs[-1][2]
        sl = last_high * (1.0 + 0.001)
        risk = sl - entry
        if risk <= 0:
            return (None, None, None, None, None, "Invalid risk (entry >= SL).", None, None, None)

        tp1 = entry - 1.5 * risk
        tp2 = entry - 2.5 * risk
        tp3 = entry - 4.0 * risk

    rr1 = (tp1 - entry) / risk if direction == "bullish" else (entry - tp1) / risk
    rr2 = (tp2 - entry) / risk if direction == "bullish" else (entry - tp2) / risk
    rr3 = (tp3 - entry) / risk if direction == "bullish" else (entry - tp3) / risk

    rr_note = (
        f"Approx R/R: TP1 ~{rr1:.1f}R, TP2 ~{rr2:.1f}R, TP3 ~{rr3:.1f}R "
        f"from entry {entry:.5f} and SL {sl:.5f}."
    )

    return entry, sl, tp1, tp2, tp3, rr_note, rr1, rr2, rr3


# ===== Structural frameworks (H&S / inverse H&S / N / V) =====


def _detect_head_shoulders(
    swings: List[Tuple[int, str, float]],
    pattern_type: str,
    label: str,
) -> Tuple[bool, str]:
    """
    Detect a very simple H&S / inverse H&S shape from swing highs/lows.

    pattern_type:
      - "bearish" → Head & Shoulders (swing highs)
      - "bullish" → inverse Head & Shoulders (swing lows)
    """
    if pattern_type == "bearish":
        hl = [s for s in swings if s[1] == "high"]
        if len(hl) < 3:
            return False, f"{label}: no clear H&S (insufficient swing highs)."
        h1, h2, h3 = hl[-3], hl[-2], hl[-1]
        # ensure chronological
        if not (h1[0] < h2[0] < h3[0]):
            return False, f"{label}: H&S swings not in chronological order."
        # shoulders similar
        mid = (h1[2] + h3[2]) / 2.0
        if mid <= 0:
            return False, f"{label}: invalid prices for H&S."
        shoulders_close = abs(h1[2] - h3[2]) / mid <= 0.005  # ~0.5%
        # head higher
        head_above = h2[2] > h1[2] * (1.0 + 0.002) and h2[2] > h3[2] * (1.0 + 0.002)

        if shoulders_close and head_above:
            note = (
                f"{label}: Bearish Head & Shoulders structure in recent swings "
                f"(shoulders ~{h1[2]:.5f}/{h3[2]:.5f}, head {h2[2]:.5f})."
            )
            return True, note
        return False, f"{label}: no clean Bearish H&S in last swings."

    else:  # "bullish" → inverse H&S on lows
        ll = [s for s in swings if s[1] == "low"]
        if len(ll) < 3:
            return False, f"{label}: no clear inverse H&S (insufficient swing lows)."
        l1, l2, l3 = ll[-3], ll[-2], ll[-1]
        if not (l1[0] < l2[0] < l3[0]):
            return False, f"{label}: inverse H&S swings not in chronological order."
        mid = (l1[2] + l3[2]) / 2.0
        if mid <= 0:
            return False, f"{label}: invalid prices for inverse H&S."
        shoulders_close = abs(l1[2] - l3[2]) / mid <= 0.005
        head_below = l2[2] < l1[2] * (1.0 - 0.002) and l2[2] < l3[2] * (1.0 - 0.002)

        if shoulders_close and head_below:
            note = (
                f"{label}: Bullish inverse Head & Shoulders structure in recent swings "
                f"(shoulders ~{l1[2]:.5f}/{l3[2]:.5f}, head {l2[2]:.5f})."
            )
            return True, note
        return False, f"{label}: no clean inverse H&S in last swings."


def _framework_context(
    direction: Direction,
    daily_swings: List[Tuple[int, str, float]],
    daily_trend: str,
    weekly_trend: str,
    monthly_trend: str,
    monthly_edge: bool,
    weekly_edge: bool,
    daily_edge: bool,
    monthly_sr_ok: bool,
    weekly_sr_ok: bool,
    daily_sr_ok: bool,
    fib_ok: bool,
    weekly_ob_ok: bool,
    daily_ob_ok: bool,
) -> Tuple[bool, str]:
    """
    Detect structural frameworks in line with the Blueprint spec:

    - Reversal frameworks: H&S / inverse H&S at or near HTF S/R/edges.
    - Continuation frameworks: Bullish N / Bearish V when:
        Weekly trend with direction, fib_ok, and inside/near OB.

    Returns:
      (framework_ok, note)
    """
    frameworks: List[str] = []
    framework_ok = False

    # --- Reversal frameworks (H&S / inverse H&S on Daily) ---
    near_htf_level = (
        monthly_edge or weekly_edge or daily_edge or
        monthly_sr_ok or weekly_sr_ok or daily_sr_ok
    )

    if near_htf_level:
        if direction == "bearish":
            hs_found, hs_note = _detect_head_shoulders(daily_swings, "bearish", "Daily")
            if hs_found:
                frameworks.append(
                    hs_note + " At HTF resistance/edge, valid bearish reversal context."
                )
                framework_ok = True
        else:
            ihs_found, ihs_note = _detect_head_shoulders(daily_swings, "bullish", "Daily")
            if ihs_found:
                frameworks.append(
                    ihs_note + " At HTF support/edge, valid bullish reversal context."
                )
                framework_ok = True

    # --- Continuation frameworks (Bullish N / Bearish V) ---
    # Weekly trend + fib golden pocket + OB context = N/V-style continuation.
    if direction == "bullish":
        if weekly_trend == "bullish" and fib_ok and (weekly_ob_ok or daily_ob_ok):
            frameworks.append(
                "Daily/Weekly: Bullish N continuation candidate "
                "(impulse + golden pocket pullback into demand)."
            )
            framework_ok = True
    else:
        if weekly_trend == "bearish" and fib_ok and (weekly_ob_ok or daily_ob_ok):
            frameworks.append(
                "Daily/Weekly: Bearish V continuation candidate "
                "(impulse + golden pocket pullback into supply)."
            )
            framework_ok = True

    if not frameworks:
        return False, (
            "No clear H&S / inverse H&S / N/V framework detected at HTF level in recent swings."
        )

    return framework_ok, " ".join(frameworks)


# ===== Core scan logic for a single asset (TOP-DOWN) =====


def scan_single_asset(symbol: str) -> Optional[ScanResult]:
    """
    Scan a single asset for Blueprint-style swing opportunities with strict top-down:

    1) Monthly & Weekly S/R + supply/demand.
    2) Weekly structure & Weekly fib golden pocket.
    3) Daily context (structure, range, Daily fib, liquidity, frameworks).
    4) 4H confirmation BOS.
    """
    # --- Load data ---
    daily = get_ohlcv(symbol, timeframe="D", count=240)
    if not daily or len(daily) < 60:
        return None

    current_close = daily[-1]["close"]

    weekly = get_ohlcv(symbol, timeframe="W", count=260)
    # Monthly may not be available in every data source; try "M" then "MN".
    monthly = get_ohlcv(symbol, timeframe="M", count=240)
    if not monthly:
        monthly = get_ohlcv(symbol, timeframe="MN", count=240)

    # --- Swings ---
    daily_swings = _find_swings(daily, lookback=2)
    weekly_swings = _find_swings(weekly, lookback=1) if weekly else []
    monthly_swings = _find_swings(monthly, lookback=1) if monthly else []

    # --- Monthly context ---
    if monthly and monthly_swings:
        monthly_trend, monthly_trend_note = _infer_trend(monthly, monthly_swings, "Monthly")
        m_low, m_high, m_pos, monthly_loc_note, monthly_edge = _range_and_location(
            monthly, current_close, "Monthly"
        )
        monthly_sr_ok, monthly_sr_note = _sr_context_from_swings(
            monthly, monthly_swings, current_close, "Monthly"
        )
    else:
        monthly_trend = "unknown"
        monthly_trend_note = "Monthly data unavailable or insufficient."
        monthly_loc_note = "Monthly: no range info."
        monthly_edge = False
        monthly_sr_ok = False
        monthly_sr_note = "Monthly: no S/R context."

    # --- Weekly context ---
    if weekly and weekly_swings:
        weekly_trend, weekly_trend_note = _infer_trend(weekly, weekly_swings, "Weekly")
        w_low, w_high, w_pos, weekly_loc_note, weekly_edge = _range_and_location(
            weekly, current_close, "Weekly"
        )
        weekly_sr_ok, weekly_sr_note = _sr_context_from_swings(
            weekly, weekly_swings, current_close, "Weekly"
        )
        weekly_impulse = _find_impulse_leg(weekly, weekly_swings, weekly_trend, min_move_frac=0.005)
    else:
        weekly_trend = "unknown"
        weekly_trend_note = "Weekly data unavailable or insufficient."
        weekly_loc_note = "Weekly: no range info."
        weekly_edge = False
        weekly_sr_ok = False
        weekly_sr_note = "Weekly: no S/R context."
        weekly_impulse = None

    # --- Daily context ---
    daily_trend, daily_trend_note = _infer_trend(daily, daily_swings, "Daily")
    d_low, d_high, d_pos, daily_loc_note, daily_edge = _range_and_location(
        daily, current_close, "Daily"
    )
    daily_sr_ok, daily_sr_note = _sr_context_from_swings(
        daily, daily_swings, current_close, "Daily"
    )
    daily_impulse = _find_impulse_leg(daily, daily_swings, daily_trend, min_move_frac=0.003)

    # --- Decide direction from Weekly (and Monthly) first ---
    direction: Optional[Direction] = None

    # Preferred: Monthly+Weekly alignment
    if monthly_trend in ("bullish", "bearish") and weekly_trend == monthly_trend:
        if monthly_trend == "bullish" and w_pos <= 0.5:
            direction = "bullish"
        elif monthly_trend == "bearish" and w_pos >= 0.5:
            direction = "bearish"

    # Fallback: Weekly only
    if direction is None and weekly_trend in ("bullish", "bearish"):
        if weekly_trend == "bullish" and w_pos <= 0.5:
            direction = "bullish"
        elif weekly_trend == "bearish" and w_pos >= 0.5:
            direction = "bearish"

    # Fallback: Daily if higher TFs unclear
    if direction is None and daily_trend in ("bullish", "bearish"):
        if daily_trend == "bullish" and d_pos <= 0.5:
            direction = "bullish"
        elif daily_trend == "bearish" and d_pos >= 0.5:
            direction = "bearish"

    if direction is None:
        # No meaningful top-down direction
        return None

    # --- Weekly Fibonacci (primary fib) ---
    weekly_fib_ok, weekly_fib_note, weekly_gp_low, weekly_gp_high = _check_fib_confluence(
        weekly_impulse, current_close, "Weekly"
    )

    # --- Daily Fibonacci (secondary fib, mostly for entry refinement) ---
    daily_fib_ok, daily_fib_note, daily_gp_low, daily_gp_high = _check_fib_confluence(
        daily_impulse, current_close, "Daily"
    )

    # Combined fib confluence: Weekly priority, Daily secondary fallback
    fib_ok = weekly_fib_ok or daily_fib_ok
    if weekly_fib_ok and daily_fib_ok:
        fib_note = f"{weekly_fib_note} Daily also aligned: {daily_fib_note}"
    elif weekly_fib_ok:
        fib_note = weekly_fib_note
    elif daily_fib_ok:
        fib_note = daily_fib_note + " (no clear Weekly impulse)."
    else:
        fib_note = f"{weekly_fib_note} {daily_fib_note}"

    # --- Supply/Demand from Weekly and Daily impulse origins ---
    if direction in ("bullish", "bearish"):
        weekly_ob_ok, weekly_ob_note = _ob_context_from_impulse(
            weekly if weekly else daily,  # fallback
            direction,
            weekly_impulse,
            current_close,
            "Weekly",
        )
        daily_ob_ok, daily_ob_note = _ob_context_from_impulse(
            daily,
            direction,
            daily_impulse,
            current_price=current_close,
            label="Daily",
        )
    else:
        weekly_ob_ok, weekly_ob_note = False, "Weekly: no OB context."
        daily_ob_ok, daily_ob_note = False, "Daily: no OB context."

    # --- Liquidity (Daily equal highs/lows) ---
    liquidity_ok, liquidity_note = _liquidity_context(daily, current_close)

    # --- 4H confirmation (BOS) ---
    confirmation_ok, confirmation_note = _h4_confirmation(symbol, direction)

    # --- R/R & trade structure from Daily ---
    entry, sl, tp1, tp2, tp3, rr_note, rr1, rr2, rr3 = _build_trade_from_context(
        symbol, direction, daily, daily_swings, current_close
    )

    min_rr1 = 1.2 if SIGNAL_MODE == "aggressive" else 1.5
    rr_ok = entry is not None and sl is not None and rr1 is not None and rr1 >= min_rr1

    # --- Structural frameworks (H&S / inverse H&S / N/V) ---
    framework_ok, framework_note = _framework_context(
        direction=direction,
        daily_swings=daily_swings,
        daily_trend=daily_trend,
        weekly_trend=weekly_trend,
        monthly_trend=monthly_trend,
        monthly_edge=monthly_edge,
        weekly_edge=weekly_edge,
        daily_edge=daily_edge,
        monthly_sr_ok=monthly_sr_ok,
        weekly_sr_ok=weekly_sr_ok,
        daily_sr_ok=daily_sr_ok,
        fib_ok=fib_ok,
        weekly_ob_ok=weekly_ob_ok,
        daily_ob_ok=daily_ob_ok,
    )

    # --- HTF bias & structure flags ---

    # Weekly + Daily alignment relative to the chosen direction
    if direction == "bullish":
        weekly_daily_aligned = (weekly_trend == "bullish" and daily_trend != "bearish")
    else:
        weekly_daily_aligned = (weekly_trend == "bearish" and daily_trend != "bullish")

    # Monthly supports trend if same direction or neutral/range/unknown
    monthly_supports_trend = (
        monthly_trend in ("range", "unknown") or monthly_trend == direction
    )

    # HTF **reversal mode**:
    # Monthly trend is opposite the chosen direction,
    # but price is at Monthly edge or Monthly S/R → acceptable reversal context.
    monthly_opposite_reversal = False
    if monthly_trend not in ("range", "unknown") and monthly_trend != direction:
        if monthly_edge or monthly_sr_ok:
            monthly_opposite_reversal = True

    htf_bias_ok = weekly_daily_aligned and (monthly_supports_trend or monthly_opposite_reversal)

    if htf_bias_ok:
        if monthly_opposite_reversal:
            mode_txt = "HTF reversal bias"
        else:
            mode_txt = "HTF trend-continuation bias"
        htf_bias_note = (
            f"{mode_txt}: {direction} "
            f"(Monthly={monthly_trend}, Weekly={weekly_trend}, Daily={daily_trend})."
        )
    else:
        htf_bias_note = (
            "HTF bias mixed: Monthly/Weekly/Daily not cleanly aligned "
            f"(Monthly={monthly_trend}, Weekly={weekly_trend}, Daily={daily_trend})."
        )

    # Structure: Weekly+Daily trend OR valid framework at HTF level
    if direction == "bullish":
        structure_trend_ok = (weekly_trend == "bullish" and daily_trend in ("bullish", "range"))
    else:
        structure_trend_ok = (weekly_trend == "bearish" and daily_trend in ("bearish", "range"))

    structure_ok = structure_trend_ok or framework_ok

    # --- Location confluence ---
    location_ok = (
        monthly_edge
        or weekly_edge
        or daily_edge
        or monthly_sr_ok
        or weekly_sr_ok
        or daily_sr_ok
        or weekly_ob_ok
        or daily_ob_ok
    )

    location_note_parts = [
        monthly_loc_note,
        monthly_sr_note,
        weekly_loc_note,
        weekly_sr_note,
        weekly_ob_note,
        daily_loc_note,
        daily_sr_note,
        daily_ob_note,
    ]
    location_note = " ".join([p for p in location_note_parts if p])

    # --- Confluence scoring (7/7) ---
    if SIGNAL_MODE == "aggressive":
        min_scan = 3
        min_trade = 5
    else:
        min_scan = 4
        min_trade = 6

    score_components = {
        "htf_bias": htf_bias_ok,
        "location": location_ok,
        "fib": fib_ok,
        "liquidity": liquidity_ok,
        "confirmation": confirmation_ok,
        "rr": rr_ok,
        "structure": structure_ok,
    }

    confluence = sum(1 for v in score_components.values() if v)

    # Debug
    print(
        f"[scan_single_asset] {symbol}: confluence {confluence}/7 (mode={SIGNAL_MODE}) "
        f"-> {score_components}"
    )

    if confluence < min_scan:
        return None

    structure_note_full = (
        f"{monthly_trend_note} | {weekly_trend_note} | {daily_trend_note} | {framework_note}"
    )

    summary_reason = (
        f"{direction.upper()} candidate with {confluence}/7 confluence: "
        f"htf_bias={'yes' if htf_bias_ok else 'no'}, "
        f"location={'yes' if location_ok else 'no'}, "
        f"fib={'yes' if fib_ok else 'no'}, "
        f"liquidity={'yes' if liquidity_ok else 'no'}, "
        f"4H_confirm={'yes' if confirmation_ok else 'no'}, "
        f"rr={'yes' if rr_ok else 'no'}, "
        f"struct={'yes' if structure_ok else 'no'}."
    )

    result = ScanResult(
        symbol=symbol,
        direction=direction,
        confluence_score=confluence,
        timeframe="Weekly→Daily with H4 confirmation",
        htf_bias=htf_bias_note,
        location_note=location_note,
        fib_note=fib_note,
        liquidity_note=liquidity_note,
        structure_note=structure_note_full,
        confirmation_note=confirmation_note,
        rr_note=rr_note,
        summary_reason=summary_reason,
    )

    # Only attach a full trade when confluence is high enough
    if confluence >= min_trade and entry is not None and sl is not None:
        result.entry = entry
        result.stop_loss = sl
        result.tp1 = tp1
        result.tp2 = tp2
        result.tp3 = tp3
        result.rr1 = rr1
        result.rr2 = rr2
        result.rr3 = rr3
        # 4H decides if it's active or just a prepared zone
        result.status = "active" if confirmation_ok else "in_progress"

    return result


# ===== Group scans =====


def scan_group(symbols: List[str]) -> Tuple[List[ScanResult], List[ScanResult]]:
    """
    Scan a list of symbols.

    Returns:
      (scan_results, trade_ideas)
      where trade_ideas are those with high confluence and valid trade fields.
    """
    scan_results: List[ScanResult] = []
    trade_ideas: List[ScanResult] = []

    for sym in symbols:
        res = scan_single_asset(sym)
        if res is None:
            continue
        scan_results.append(res)
        if (
            res.entry is not None
            and res.stop_loss is not None
            and res.status == "active"
        ):
            trade_ideas.append(res)

    return scan_results, trade_ideas


def scan_forex() -> Tuple[List[ScanResult], List[ScanResult]]:
    return scan_group(FOREX_PAIRS)


def scan_crypto() -> Tuple[List[ScanResult], List[ScanResult]]:
    return scan_group(CRYPTO_ASSETS)


def scan_metals() -> Tuple[List[ScanResult], List[ScanResult]]:
    return scan_group(METALS)


def scan_indices() -> Tuple[List[ScanResult], List[ScanResult]]:
    return scan_group(INDICES)


def scan_energies() -> Tuple[List[ScanResult], List[ScanResult]]:
    return scan_group(ENERGIES)


def scan_all_markets() -> Dict[str, Tuple[List[ScanResult], List[ScanResult]]]:
    """
    Scan all configured markets and return a dict:
      {
        "Forex": (scan_results, trade_ideas),
        "Metals": ...,
        "Indices": ...,
        "Energies": ...,
        "Crypto": ...,
      }
    """
    results: Dict[str, Tuple[List[ScanResult], List[ScanResult]]] = {}
    results["Forex"] = scan_forex()
    results["Metals"] = scan_metals()
    results["Indices"] = scan_indices()
    results["Energies"] = scan_energies()
    results["Crypto"] = scan_crypto()
    return results
