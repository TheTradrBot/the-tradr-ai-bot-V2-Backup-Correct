"""
Strategy Core Module for Blueprint Trader AI.

This module provides the single source of truth for trading rules,
used by both backtests and live scanning/Discord outputs.

The strategy is parameterized to allow optimization while staying
faithful to the Blueprint confluence-based approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any


@dataclass
class StrategyParams:
    """
    Strategy parameters that can be optimized.
    
    These control confluence thresholds, SL/TP ratios, filters, etc.
    
    Optimized defaults for high win rate:
    - min_confluence=4: Only take high-quality setups
    - min_quality_factors=2: Require multiple confirmations
    - atr_sl_multiplier=1.0: Tight stops for precise entries
    - atr_tp1_multiplier=1.5: TP1 at 1.5R for quick profits
    - atr_tp2_multiplier=2.5: TP2 at 2.5R for runners
    - atr_tp3_multiplier=4.0: TP3 at 4R for big moves
    """
    min_confluence: int = 4
    min_quality_factors: int = 2
    
    atr_sl_multiplier: float = 1.0
    atr_tp1_multiplier: float = 1.5
    atr_tp2_multiplier: float = 2.5
    atr_tp3_multiplier: float = 4.0
    
    fib_low: float = 0.382
    fib_high: float = 0.886
    
    structure_sl_lookback: int = 35
    liquidity_sweep_lookback: int = 12
    
    use_htf_filter: bool = True
    use_structure_filter: bool = True
    use_liquidity_filter: bool = True
    use_fib_filter: bool = True
    use_confirmation_filter: bool = True
    
    require_htf_alignment: bool = False
    require_confirmation_for_active: bool = True
    require_rr_for_active: bool = True
    
    min_rr_ratio: float = 1.0
    risk_per_trade_pct: float = 1.0
    
    cooldown_bars: int = 0
    max_open_trades: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            "min_confluence": self.min_confluence,
            "min_quality_factors": self.min_quality_factors,
            "atr_sl_multiplier": self.atr_sl_multiplier,
            "atr_tp1_multiplier": self.atr_tp1_multiplier,
            "atr_tp2_multiplier": self.atr_tp2_multiplier,
            "atr_tp3_multiplier": self.atr_tp3_multiplier,
            "fib_low": self.fib_low,
            "fib_high": self.fib_high,
            "structure_sl_lookback": self.structure_sl_lookback,
            "liquidity_sweep_lookback": self.liquidity_sweep_lookback,
            "use_htf_filter": self.use_htf_filter,
            "use_structure_filter": self.use_structure_filter,
            "use_liquidity_filter": self.use_liquidity_filter,
            "use_fib_filter": self.use_fib_filter,
            "use_confirmation_filter": self.use_confirmation_filter,
            "require_htf_alignment": self.require_htf_alignment,
            "require_confirmation_for_active": self.require_confirmation_for_active,
            "require_rr_for_active": self.require_rr_for_active,
            "min_rr_ratio": self.min_rr_ratio,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "cooldown_bars": self.cooldown_bars,
            "max_open_trades": self.max_open_trades,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        """Create parameters from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


@dataclass
class Signal:
    """Represents a trading signal/setup."""
    symbol: str
    direction: str
    bar_index: int
    timestamp: Any
    
    confluence_score: int = 0
    quality_factors: int = 0
    
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    
    is_active: bool = False
    is_watching: bool = False
    
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Trade:
    """Represents a completed trade for backtest analysis."""
    symbol: str
    direction: str
    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    stop_loss: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    
    risk: float = 0.0
    reward: float = 0.0
    rr: float = 0.0
    
    is_winner: bool = False
    exit_reason: str = ""
    
    confluence_score: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "risk": self.risk,
            "reward": self.reward,
            "rr": self.rr,
            "is_winner": self.is_winner,
            "exit_reason": self.exit_reason,
            "confluence_score": self.confluence_score,
        }


@dataclass
class ImpulseCandle:
    """
    Represents an impulse candle for N-pattern detection.
    
    The impulse candle is the first leg of the N-pattern:
    - Strong body (> 1.5x average body size)
    - Breaks recent structure (BOS - Break of Structure)
    - Used as the anchor for Fibonacci retracement and extension calculations
    """
    index: int
    open_price: float
    high: float
    low: float
    close: float
    direction: str
    body_size: float
    avg_body: float
    timestamp: Any = None
    
    @property
    def fib_anchor_start(self) -> float:
        """Start point for Fib calculation (wick extreme: LOW for bullish, HIGH for bearish)."""
        if self.direction == "bullish":
            return self.low
        else:
            return self.high
    
    @property
    def fib_anchor_end(self) -> float:
        """End point for Fib calculation (HIGH for bullish, LOW for bearish)."""
        if self.direction == "bullish":
            return self.high
        else:
            return self.low
    
    @property
    def impulse_range(self) -> float:
        """The range of the impulse leg for Fib calculations."""
        return abs(self.fib_anchor_end - self.fib_anchor_start)


def _atr(candles: List[Dict], period: int = 14) -> float:
    """
    Calculate Average True Range (ATR).
    
    Args:
        candles: List of OHLCV candle dictionaries
        period: ATR period (default 14)
    
    Returns:
        ATR value or 0 if insufficient data
    """
    if len(candles) < period + 1:
        return 0.0
    
    tr_values = []
    for i in range(1, len(candles)):
        high = candles[i].get("high")
        low = candles[i].get("low")
        prev_close = candles[i - 1].get("close")
        
        if high is None or low is None or prev_close is None:
            continue
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return sum(tr_values) / len(tr_values) if tr_values else 0.0
    
    atr_val = sum(tr_values[:period]) / period
    for tr in tr_values[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    
    return atr_val


def detect_impulse_candle(
    candles: List[Dict],
    direction: str,
    lookback: int = 20,
    body_multiplier: float = 1.5,
    structure_lookback: int = 5,
) -> Optional[ImpulseCandle]:
    """
    Detect an impulse candle that forms the first leg of the N-pattern.
    
    An impulse candle is:
    1. A strong candle with body > 1.5x average body size
    2. Breaks recent structure (BOS) - makes new high/low beyond last N bars
    3. The starting point for Fibonacci retracement/extension calculations
    
    Args:
        candles: List of OHLCV candle dictionaries (oldest to newest)
        direction: Trade direction ("bullish" or "bearish")
        lookback: Period for calculating average body size
        body_multiplier: Minimum multiple of average body for impulse (default 1.5)
        structure_lookback: Bars to check for structure break (default 5)
    
    Returns:
        ImpulseCandle object if found, None otherwise
    """
    if not candles or len(candles) < lookback + structure_lookback:
        return None
    
    body_sizes = []
    for i in range(max(0, len(candles) - lookback - 1), len(candles) - 1):
        c = candles[i]
        body = abs(c.get("close", 0) - c.get("open", 0))
        body_sizes.append(body)
    
    if not body_sizes:
        return None
    
    avg_body = sum(body_sizes) / len(body_sizes)
    if avg_body <= 0:
        return None
    
    for i in range(len(candles) - 1, max(len(candles) - lookback - 1, structure_lookback), -1):
        c = candles[i]
        open_price = c.get("open", 0)
        close_price = c.get("close", 0)
        high = c.get("high", 0)
        low = c.get("low", 0)
        body = abs(close_price - open_price)
        
        if body < avg_body * body_multiplier:
            continue
        
        is_bullish_candle = close_price > open_price
        is_bearish_candle = close_price < open_price
        
        if direction == "bullish" and not is_bullish_candle:
            continue
        if direction == "bearish" and not is_bearish_candle:
            continue
        
        structure_start = max(0, i - structure_lookback)
        structure_end = i
        
        if direction == "bullish":
            prior_highs = [candles[j].get("high", 0) for j in range(structure_start, structure_end)]
            if not prior_highs:
                continue
            prior_high = max(prior_highs)
            if high <= prior_high:
                continue
        else:
            prior_lows = [candles[j].get("low", float("inf")) for j in range(structure_start, structure_end)]
            if not prior_lows:
                continue
            prior_low = min(prior_lows)
            if low >= prior_low:
                continue
        
        timestamp = c.get("time") or c.get("timestamp") or c.get("date")
        
        return ImpulseCandle(
            index=i,
            open_price=open_price,
            high=high,
            low=low,
            close=close_price,
            direction=direction,
            body_size=body,
            avg_body=avg_body,
            timestamp=timestamp,
        )
    
    return None


def find_all_impulse_candles(
    candles: List[Dict],
    direction: str,
    lookback: int = 20,
    body_multiplier: float = 1.5,
    structure_lookback: int = 5,
    max_age_bars: int = 50,
) -> List[ImpulseCandle]:
    """
    Find all valid impulse candles in recent history.
    
    Args:
        candles: List of OHLCV candle dictionaries
        direction: Trade direction ("bullish" or "bearish")
        lookback: Period for average body calculation
        body_multiplier: Minimum body multiplier for impulse
        structure_lookback: Bars to check for structure break
        max_age_bars: Maximum age of impulse candles to consider
    
    Returns:
        List of ImpulseCandle objects (most recent first)
    """
    if not candles or len(candles) < lookback + structure_lookback:
        return []
    
    body_sizes = []
    for i in range(max(0, len(candles) - lookback - 1), len(candles) - 1):
        c = candles[i]
        body = abs(c.get("close", 0) - c.get("open", 0))
        body_sizes.append(body)
    
    if not body_sizes:
        return []
    
    avg_body = sum(body_sizes) / len(body_sizes)
    if avg_body <= 0:
        return []
    
    impulses = []
    min_index = max(0, len(candles) - max_age_bars)
    
    for i in range(len(candles) - 1, max(min_index, structure_lookback), -1):
        c = candles[i]
        open_price = c.get("open", 0)
        close_price = c.get("close", 0)
        high = c.get("high", 0)
        low = c.get("low", 0)
        body = abs(close_price - open_price)
        
        if body < avg_body * body_multiplier:
            continue
        
        is_bullish_candle = close_price > open_price
        is_bearish_candle = close_price < open_price
        
        if direction == "bullish" and not is_bullish_candle:
            continue
        if direction == "bearish" and not is_bearish_candle:
            continue
        
        structure_start = max(0, i - structure_lookback)
        structure_end = i
        
        if direction == "bullish":
            prior_highs = [candles[j].get("high", 0) for j in range(structure_start, structure_end)]
            if not prior_highs:
                continue
            prior_high = max(prior_highs)
            if high <= prior_high:
                continue
        else:
            prior_lows = [candles[j].get("low", float("inf")) for j in range(structure_start, structure_end)]
            if not prior_lows:
                continue
            prior_low = min(prior_lows)
            if low >= prior_low:
                continue
        
        timestamp = c.get("time") or c.get("timestamp") or c.get("date")
        
        impulses.append(ImpulseCandle(
            index=i,
            open_price=open_price,
            high=high,
            low=low,
            close=close_price,
            direction=direction,
            body_size=body,
            avg_body=avg_body,
            timestamp=timestamp,
        ))
    
    return impulses


def check_golden_pocket_entry(
    candles: List[Dict],
    impulse: ImpulseCandle,
    current_bar_index: int,
) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], str]:
    """
    Check if price has retraced into the Golden Pocket zone (0.618-0.786).
    
    Blueprint N-Pattern Entry Logic:
    1. After an impulse candle, wait for price to retrace
    2. Entry triggers when price enters the 0.618-0.786 Fibonacci zone
    3. SL is placed below the retracement swing low (with ATR buffer)
    4. TPs are Fibonacci extensions from the impulse leg
    
    Args:
        candles: Full list of candles
        impulse: The ImpulseCandle to calculate from
        current_bar_index: Index of the current bar to check
    
    Returns:
        Tuple of (is_entry, entry, sl, tp1, tp2, tp3, tp4, tp5, note)
    """
    if not candles or current_bar_index <= impulse.index:
        return False, None, None, None, None, None, None, None, "Not enough data after impulse"
    
    if current_bar_index >= len(candles):
        return False, None, None, None, None, None, None, None, "Index out of range"
    
    impulse_range = impulse.impulse_range
    if impulse_range <= 0:
        return False, None, None, None, None, None, None, None, "Invalid impulse range"
    
    if impulse.direction == "bullish":
        fib_618 = impulse.high - impulse_range * 0.618
        fib_786 = impulse.high - impulse_range * 0.786
        golden_pocket_high = fib_618
        golden_pocket_low = fib_786
    else:
        fib_618 = impulse.low + impulse_range * 0.618
        fib_786 = impulse.low + impulse_range * 0.786
        golden_pocket_low = fib_618
        golden_pocket_high = fib_786
    
    current = candles[current_bar_index]
    current_close = current.get("close", 0)
    current_high = current.get("high", 0)
    current_low = current.get("low", 0)
    
    in_zone = golden_pocket_low <= current_close <= golden_pocket_high
    touches_zone = (
        (current_low <= golden_pocket_high and current_high >= golden_pocket_low)
    )
    
    if not (in_zone or touches_zone):
        return False, None, None, None, None, None, None, None, f"Price not in Golden Pocket ({golden_pocket_low:.5f} - {golden_pocket_high:.5f})"
    
    retracement_candles = candles[impulse.index + 1:current_bar_index + 1]
    if not retracement_candles:
        return False, None, None, None, None, None, None, None, "No retracement candles"
    
    atr = _atr(candles[:current_bar_index + 1], 14)
    atr_buffer = atr * 0.3 if atr > 0 else impulse_range * 0.1
    
    if impulse.direction == "bullish":
        retracement_low = min(c.get("low", float("inf")) for c in retracement_candles)
        sl = retracement_low - atr_buffer
        
        if current_close < impulse.low:
            return False, None, None, None, None, None, None, None, "Price below impulse low - invalid retracement"
        
        entry = current_close
        
        tp1 = impulse.high + impulse_range * 0.25
        tp2 = impulse.high + impulse_range * 0.68
        tp3 = impulse.high + impulse_range * 1.0
        tp4 = impulse.high + impulse_range * 1.5
        tp5 = impulse.high + impulse_range * 2.0
    else:
        retracement_high = max(c.get("high", 0) for c in retracement_candles)
        sl = retracement_high + atr_buffer
        
        if current_close > impulse.high:
            return False, None, None, None, None, None, None, None, "Price above impulse high - invalid retracement"
        
        entry = current_close
        
        tp1 = impulse.low - impulse_range * 0.25
        tp2 = impulse.low - impulse_range * 0.68
        tp3 = impulse.low - impulse_range * 1.0
        tp4 = impulse.low - impulse_range * 1.5
        tp5 = impulse.low - impulse_range * 2.0
    
    risk = abs(entry - sl)
    if risk <= 0:
        return False, None, None, None, None, None, None, None, "Invalid risk (SL at entry)"
    
    reward = abs(tp1 - entry)
    rr = reward / risk if risk > 0 else 0
    
    note = f"N-Pattern: Entry={entry:.5f}, SL={sl:.5f}, TP1={tp1:.5f}, R:R={rr:.2f}"
    
    return True, entry, sl, tp1, tp2, tp3, tp4, tp5, note


def compute_n_pattern_trade_levels(
    candles: List[Dict],
    direction: str,
    current_bar_index: int,
    params: Optional["StrategyParams"] = None,
) -> Tuple[str, bool, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[ImpulseCandle]]:
    """
    Compute trade levels using the N-pattern detection system.
    
    This is the main entry point for N-pattern based trade level calculation.
    It finds impulse candles and checks for Golden Pocket entries.
    
    Args:
        candles: Daily OHLCV candles
        direction: Trade direction
        current_bar_index: Index of current bar
        params: Strategy parameters
    
    Returns:
        Tuple of (note, is_valid, entry, sl, tp1, tp2, tp3, tp4, tp5, impulse)
    """
    if params is None:
        from strategy_core import StrategyParams
        params = StrategyParams()
    
    if not candles or current_bar_index < 25:
        return "N-Pattern: Insufficient data", False, None, None, None, None, None, None, None, None
    
    candle_slice = candles[:current_bar_index + 1]
    
    impulses = find_all_impulse_candles(
        candle_slice,
        direction,
        lookback=20,
        body_multiplier=1.2,
        structure_lookback=5,
        max_age_bars=50,
    )
    
    if not impulses:
        return "N-Pattern: No impulse candle found", False, None, None, None, None, None, None, None, None
    
    for impulse in impulses:
        if current_bar_index <= impulse.index:
            continue
        
        is_entry, entry, sl, tp1, tp2, tp3, tp4, tp5, note = check_golden_pocket_entry(
            candle_slice,
            impulse,
            current_bar_index,
        )
        
        if is_entry:
            return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5, impulse
    
    return "N-Pattern: Awaiting Golden Pocket retracement", False, None, None, None, None, None, None, None, None


def find_sr_zones(candles: List[Dict], lookback: int = 3, tolerance_pct: float = 0.005) -> Tuple[List[float], List[float]]:
    """
    Find significant Support and Resistance zones from candle data.
    
    Used for Monthly and Weekly S/R identification.
    
    Args:
        candles: List of OHLCV candle dictionaries (Monthly or Weekly)
        lookback: Number of bars to look back/forward for pivot identification
        tolerance_pct: Percentage tolerance for grouping similar levels
    
    Returns:
        Tuple of (resistance_zones, support_zones) as lists of price levels
    """
    if not candles or len(candles) < lookback * 2 + 1:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i].get("high", 0)
        low = candles[i].get("low", float("inf"))
        
        is_swing_high = True
        is_swing_low = True
        
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if candles[j].get("high", 0) > high:
                is_swing_high = False
            if candles[j].get("low", float("inf")) < low:
                is_swing_low = False
        
        if is_swing_high:
            swing_highs.append(high)
        if is_swing_low:
            swing_lows.append(low)
    
    def cluster_levels(levels: List[float], tolerance: float) -> List[float]:
        """Cluster nearby levels into zones."""
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    resistance_zones = cluster_levels(swing_highs, tolerance_pct)
    support_zones = cluster_levels(swing_lows, tolerance_pct)
    
    return resistance_zones, support_zones


def check_at_sr_zone(
    price: float,
    resistance_zones: List[float],
    support_zones: List[float],
    atr: float,
    zone_tolerance_atr: float = 1.0,
) -> Tuple[bool, str, str]:
    """
    Check if price is at a significant S/R zone.
    
    Args:
        price: Current price
        resistance_zones: List of resistance levels
        support_zones: List of support levels
        atr: Average True Range for tolerance calculation
        zone_tolerance_atr: Tolerance in ATR multiples
    
    Returns:
        Tuple of (is_at_sr, zone_type, suggested_direction)
        - zone_type: "resistance", "support", or "none"
        - suggested_direction: "bearish" (at resistance), "bullish" (at support), or "none"
    """
    tolerance = atr * zone_tolerance_atr
    
    for res in resistance_zones:
        if abs(price - res) < tolerance:
            return True, "resistance", "bearish"
    
    for sup in support_zones:
        if abs(price - sup) < tolerance:
            return True, "support", "bullish"
    
    return False, "none", "none"


def detect_daily_bos(candles: List[Dict], lookback: int = 10) -> Tuple[bool, str]:
    """
    Detect Break of Structure (BOS) on Daily timeframe.
    
    Args:
        candles: Daily candles
        lookback: Number of bars to check for structure break
    
    Returns:
        Tuple of (has_bos, direction)
        - direction: "bullish" (broke above), "bearish" (broke below)
    """
    if not candles or len(candles) < lookback + 2:
        return False, "none"
    
    current = candles[-1]
    recent = candles[-(lookback+1):-1]
    
    if not recent:
        return False, "none"
    
    prior_high = max(c.get("high", 0) for c in recent)
    prior_low = min(c.get("low", float("inf")) for c in recent)
    
    current_close = current.get("close", 0)
    current_high = current.get("high", 0)
    current_low = current.get("low", float("inf"))
    
    if current_high > prior_high and current_close > prior_high:
        return True, "bullish"
    
    if current_low < prior_low and current_close < prior_low:
        return True, "bearish"
    
    return False, "none"


def detect_consolidation(candles: List[Dict], lookback: int = 10, atr_threshold: float = 0.5) -> Tuple[bool, int, int, float, float]:
    """
    Detect sideways consolidation/range before an impulse move.
    
    A consolidation is characterized by:
    - Low ATR relative to recent average (price not moving much)
    - Well-defined high and low boundaries
    
    Args:
        candles: OHLCV candles
        lookback: Number of bars to check for consolidation
        atr_threshold: ATR must be below this multiple of average ATR
    
    Returns:
        Tuple of (is_consolidation, start_idx, end_idx, range_high, range_low)
    """
    if len(candles) < lookback + 14:
        return False, -1, -1, 0, 0
    
    recent = candles[-lookback:]
    
    avg_atr = _atr(candles[:-lookback], 14) if len(candles) > lookback + 14 else _atr(candles, 14)
    recent_atr = _atr(recent, min(lookback, 14))
    
    if avg_atr <= 0:
        return False, -1, -1, 0, 0
    
    atr_ratio = recent_atr / avg_atr
    is_low_volatility = atr_ratio < atr_threshold
    
    range_high = max(c.get("high", 0) for c in recent)
    range_low = min(c.get("low", float("inf")) for c in recent)
    range_size = range_high - range_low
    
    if avg_atr > 0 and range_size < avg_atr * 3:
        is_tight_range = True
    else:
        is_tight_range = False
    
    is_consolidation = is_low_volatility or is_tight_range
    
    start_idx = len(candles) - lookback
    end_idx = len(candles) - 1
    
    return is_consolidation, start_idx, end_idx, range_high, range_low


def detect_impulse_from_consolidation(
    candles: List[Dict],
    consolidation_high: float,
    consolidation_low: float,
    direction: str,
    body_multiplier: float = 1.2,
) -> Tuple[bool, Optional[int], float, float, float]:
    """
    Detect impulse breakout from a consolidation range.
    
    For bullish: Price breaks above consolidation high with strong candle
    For bearish: Price breaks below consolidation low with strong candle
    
    Args:
        candles: OHLCV candles (most recent should be potential impulse)
        consolidation_high: Upper bound of consolidation range
        consolidation_low: Lower bound of consolidation range
        direction: "bullish" or "bearish"
        body_multiplier: Body size must be this multiple of recent average
    
    Returns:
        Tuple of (is_impulse, impulse_idx, impulse_high, impulse_low, impulse_range)
    """
    if len(candles) < 5:
        return False, None, 0, 0, 0
    
    current = candles[-1]
    c_open = current.get("open", 0)
    c_close = current.get("close", 0)
    c_high = current.get("high", 0)
    c_low = current.get("low", float("inf"))
    body = abs(c_close - c_open)
    
    recent_bodies = []
    for c in candles[-20:-1]:
        recent_bodies.append(abs(c.get("close", 0) - c.get("open", 0)))
    avg_body = sum(recent_bodies) / len(recent_bodies) if recent_bodies else body
    
    is_strong_body = body > avg_body * body_multiplier
    
    if direction == "bullish":
        breaks_structure = c_high > consolidation_high and c_close > consolidation_high
        is_bullish_candle = c_close > c_open
        is_impulse = breaks_structure and is_bullish_candle and is_strong_body
        
        if is_impulse:
            impulse_range = c_high - consolidation_low
            return True, len(candles) - 1, c_high, consolidation_low, impulse_range
    else:
        breaks_structure = c_low < consolidation_low and c_close < consolidation_low
        is_bearish_candle = c_close < c_open
        is_impulse = breaks_structure and is_bearish_candle and is_strong_body
        
        if is_impulse:
            impulse_range = consolidation_high - c_low
            return True, len(candles) - 1, consolidation_high, c_low, impulse_range
    
    return False, None, 0, 0, 0


def check_daily_structure(candles: List[Dict], direction: str, lookback: int = 20) -> Tuple[bool, str]:
    """
    Check if daily structure supports the trade direction.
    
    For bullish: Should be making Higher Highs and Higher Lows
    For bearish: Should be making Lower Lows and Lower Highs
    
    Args:
        candles: Daily OHLCV data
        direction: "bullish" or "bearish"
        lookback: Number of bars to analyze
    
    Returns:
        Tuple of (is_aligned, structure_note)
    """
    if len(candles) < lookback + 10:
        return False, "Insufficient data"
    
    recent = candles[-lookback:]
    
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(recent) - 2):
        high = recent[i].get("high", 0)
        low = recent[i].get("low", float("inf"))
        
        left_highs = [recent[j].get("high", 0) for j in range(i-2, i)]
        right_highs = [recent[j].get("high", 0) for j in range(i+1, min(i+3, len(recent)))]
        
        left_lows = [recent[j].get("low", float("inf")) for j in range(i-2, i)]
        right_lows = [recent[j].get("low", float("inf")) for j in range(i+1, min(i+3, len(recent)))]
        
        if high > max(left_highs) and high > max(right_highs) if right_highs else True:
            swing_highs.append((i, high))
        if low < min(left_lows) and low < min(right_lows) if right_lows else True:
            swing_lows.append((i, low))
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, "Insufficient swing points"
    
    recent_highs = [h for _, h in swing_highs[-3:]]
    recent_lows = [l for _, l in swing_lows[-3:]]
    
    if direction == "bullish":
        hh = len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[-2]
        hl = len(recent_lows) >= 2 and recent_lows[-1] > recent_lows[-2]
        
        if hh and hl:
            return True, "HH+HL structure confirmed"
        elif hh or hl:
            return True, "Partial bullish structure"
        else:
            return False, "No bullish structure"
    else:
        ll = len(recent_lows) >= 2 and recent_lows[-1] < recent_lows[-2]
        lh = len(recent_highs) >= 2 and recent_highs[-1] < recent_highs[-2]
        
        if ll and lh:
            return True, "LL+LH structure confirmed"
        elif ll or lh:
            return True, "Partial bearish structure"
        else:
            return False, "No bearish structure"


def check_4h_choch(
    h4_candles: List[Dict],
    direction: str,
    gp_high: float,
    gp_low: float,
) -> Tuple[bool, str]:
    """
    Check for 4H Change of Character (CHoCH) within the Golden Pocket zone.
    
    This confirms momentum has returned before entering.
    
    For bullish: Look for bullish engulfing or close back above 0.5 retrace
    For bearish: Look for bearish engulfing or close back below 0.5 retrace
    
    Args:
        h4_candles: 4H candle data
        direction: "bullish" or "bearish"
        gp_high: Upper bound of Golden Pocket
        gp_low: Lower bound of Golden Pocket
    
    Returns:
        Tuple of (has_choch, choch_note)
    """
    if len(h4_candles) < 5:
        return False, "Insufficient 4H data"
    
    recent = h4_candles[-10:] if len(h4_candles) >= 10 else h4_candles
    gp_mid = (gp_high + gp_low) / 2
    
    for i in range(1, len(recent)):
        curr = recent[i]
        prev = recent[i-1]
        
        c_open = curr.get("open", 0)
        c_close = curr.get("close", 0)
        c_high = curr.get("high", 0)
        c_low = curr.get("low", float("inf"))
        c_body = abs(c_close - c_open)
        
        p_open = prev.get("open", 0)
        p_close = prev.get("close", 0)
        p_body = abs(p_close - p_open)
        
        in_zone = gp_low <= c_close <= gp_high or gp_low <= c_low <= gp_high
        
        if not in_zone:
            continue
        
        if direction == "bullish":
            is_bullish_engulfing = (
                c_close > c_open and
                p_close < p_open and
                c_body > p_body * 1.2 and
                c_close > p_open
            )
            
            closes_above_mid = c_close > gp_mid and c_close > c_open
            
            if is_bullish_engulfing:
                return True, "4H bullish engulfing in GP"
            if closes_above_mid:
                return True, "4H bullish close above GP mid"
        else:
            is_bearish_engulfing = (
                c_close < c_open and
                p_close > p_open and
                c_body > p_body * 1.2 and
                c_close < p_open
            )
            
            closes_below_mid = c_close < gp_mid and c_close < c_open
            
            if is_bearish_engulfing:
                return True, "4H bearish engulfing in GP"
            if closes_below_mid:
                return True, "4H bearish close below GP mid"
    
    return False, "No 4H CHoCH confirmation"


def check_demand_supply_zone(
    candles: List[Dict],
    price: float,
    direction: str,
    lookback: int = 50,
    zone_atr_mult: float = 1.0,
) -> Tuple[bool, str]:
    """
    Check if price is near a demand (for longs) or supply (for shorts) zone.
    
    Demand zones: Areas where price previously reversed upward
    Supply zones: Areas where price previously reversed downward
    
    Args:
        candles: OHLCV candles
        price: Current price to check
        direction: "bullish" (check demand) or "bearish" (check supply)
        lookback: Bars to look back for zones
        zone_atr_mult: Tolerance in ATR multiples
    
    Returns:
        Tuple of (is_at_zone, zone_note)
    """
    if len(candles) < lookback:
        return False, "Insufficient data"
    
    atr = _atr(candles, 14)
    tolerance = atr * zone_atr_mult
    
    swing_highs = []
    swing_lows = []
    
    for i in range(3, len(candles) - 3):
        high = candles[i].get("high", 0)
        low = candles[i].get("low", float("inf"))
        
        is_swing_high = all(candles[j].get("high", 0) < high for j in range(i-3, i)) and \
                        all(candles[j].get("high", 0) < high for j in range(i+1, min(i+4, len(candles))))
        is_swing_low = all(candles[j].get("low", float("inf")) > low for j in range(i-3, i)) and \
                       all(candles[j].get("low", float("inf")) > low for j in range(i+1, min(i+4, len(candles))))
        
        if is_swing_high:
            swing_highs.append(high)
        if is_swing_low:
            swing_lows.append(low)
    
    if direction == "bullish":
        for low in swing_lows[-10:]:
            if abs(price - low) < tolerance:
                return True, f"Near demand zone at {low:.5f}"
        return False, "No demand zone nearby"
    else:
        for high in swing_highs[-10:]:
            if abs(price - high) < tolerance:
                return True, f"Near supply zone at {high:.5f}"
        return False, "No supply zone nearby"


def find_4h_entry_in_zone(
    h4_candles: List[Dict],
    golden_pocket_high: float,
    golden_pocket_low: float,
    direction: str,
    impulse_high: float,
    impulse_low: float,
    impulse_range: float,
) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], str]:
    """
    Find precise entry on 4H timeframe within Golden Pocket zone.
    
    This provides tighter stop loss compared to Daily-only entries.
    
    Args:
        h4_candles: 4H candle data
        golden_pocket_high: Upper bound of Golden Pocket (0.618)
        golden_pocket_low: Lower bound of Golden Pocket (0.786)
        direction: Trade direction
        impulse_high: High of the impulse move
        impulse_low: Low of the impulse move
        impulse_range: Range of impulse for Fib extension calculation
    
    Returns:
        Tuple of (is_entry, entry, sl, tp1, tp2, tp3, tp4, tp5, note)
    """
    if not h4_candles or len(h4_candles) < 5:
        return False, None, None, None, None, None, None, None, "Insufficient 4H data"
    
    current = h4_candles[-1]
    recent = h4_candles[-10:] if len(h4_candles) >= 10 else h4_candles
    
    current_close = current.get("close", 0)
    current_high = current.get("high", 0)
    current_low = current.get("low", float("inf"))
    
    in_zone = golden_pocket_low <= current_close <= golden_pocket_high
    touches_zone = (current_low <= golden_pocket_high and current_high >= golden_pocket_low)
    
    if not (in_zone or touches_zone):
        return False, None, None, None, None, None, None, None, "4H not in Golden Pocket"
    
    atr = _atr(h4_candles, 14)
    atr_buffer = atr * 0.3 if atr > 0 else impulse_range * 0.05
    
    if direction == "bullish":
        h4_swing_low = min(c.get("low", float("inf")) for c in recent)
        
        entry = current_close
        sl = h4_swing_low - atr_buffer
        
        tp1 = impulse_high + impulse_range * 0.25
        tp2 = impulse_high + impulse_range * 0.68
        tp3 = impulse_high + impulse_range * 1.0
        tp4 = impulse_high + impulse_range * 1.5
        tp5 = impulse_high + impulse_range * 2.0
    else:
        h4_swing_high = max(c.get("high", 0) for c in recent)
        
        entry = current_close
        sl = h4_swing_high + atr_buffer
        
        tp1 = impulse_low - impulse_range * 0.25
        tp2 = impulse_low - impulse_range * 0.68
        tp3 = impulse_low - impulse_range * 1.0
        tp4 = impulse_low - impulse_range * 1.5
        tp5 = impulse_low - impulse_range * 2.0
    
    risk = abs(entry - sl)
    if risk <= 0:
        return False, None, None, None, None, None, None, None, "Invalid risk"
    
    reward = abs(tp1 - entry)
    rr = reward / risk
    
    note = f"4H Entry: {entry:.5f}, SL: {sl:.5f}, R:R: {rr:.2f}"
    return True, entry, sl, tp1, tp2, tp3, tp4, tp5, note


def _find_pivots(candles: List[Dict], lookback: int = 5) -> Tuple[List[float], List[float]]:
    """
    Find swing highs and swing lows in candle data.
    
    Args:
        candles: List of OHLCV candle dictionaries
        lookback: Number of bars to look back/forward for pivot identification
    
    Returns:
        Tuple of (swing_highs, swing_lows) as lists of price levels
    """
    if len(candles) < lookback * 2 + 1:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i]["high"]
        low = candles[i]["low"]
        
        is_swing_high = True
        is_swing_low = True
        
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if candles[j]["high"] > high:
                is_swing_high = False
            if candles[j]["low"] < low:
                is_swing_low = False
        
        if is_swing_high:
            swing_highs.append(high)
        if is_swing_low:
            swing_lows.append(low)
    
    return swing_highs, swing_lows


def _infer_trend(candles: List[Dict], ema_short: int = 8, ema_long: int = 21) -> str:
    """
    Infer trend direction from candle data using EMA crossover and price action.
    
    Args:
        candles: List of OHLCV candle dictionaries
        ema_short: Short EMA period
        ema_long: Long EMA period
    
    Returns:
        "bullish", "bearish", or "mixed"
    """
    if not candles or len(candles) < ema_long + 5:
        return "mixed"
    
    closes = [c["close"] for c in candles if c.get("close") is not None]
    
    if len(closes) < ema_long + 5:
        return "mixed"
    
    def calc_ema(values: List[float], period: int) -> float:
        if len(values) < period:
            valid_values = [v for v in values if v is not None and v == v]
            return sum(valid_values) / len(valid_values) if valid_values else 0
        k = 2 / (period + 1)
        initial_values = [v for v in values[:period] if v is not None and v == v]
        if not initial_values:
            return 0
        ema = sum(initial_values) / len(initial_values)
        for price in values[period:]:
            if price is not None and price == price:
                ema = price * k + ema * (1 - k)
        return ema
    
    ema_s = calc_ema(closes, ema_short)
    ema_l = calc_ema(closes, ema_long)
    
    current_price = closes[-1]
    recent_high = max(c["high"] for c in candles[-10:])
    recent_low = min(c["low"] for c in candles[-10:])
    
    bullish_signals = 0
    bearish_signals = 0
    
    if ema_s > ema_l:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if current_price > ema_l:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if len(closes) >= 20:
        higher_highs = closes[-1] > max(closes[-10:-1]) if len(closes) > 10 else False
        lower_lows = closes[-1] < min(closes[-10:-1]) if len(closes) > 10 else False
        
        if higher_highs:
            bullish_signals += 1
        if lower_lows:
            bearish_signals += 1
    
    if bullish_signals > bearish_signals:
        return "bullish"
    elif bearish_signals > bullish_signals:
        return "bearish"
    else:
        return "mixed"


def _pick_direction_from_bias(
    mn_trend: str,
    wk_trend: str,
    d_trend: str
) -> Tuple[str, str, bool]:
    """
    Determine trade direction based on multi-timeframe bias.
    
    Args:
        mn_trend: Monthly trend
        wk_trend: Weekly trend
        d_trend: Daily trend
    
    Returns:
        Tuple of (direction, note, htf_aligned)
    """
    trends = [mn_trend, wk_trend, d_trend]
    bullish_count = sum(1 for t in trends if t == "bullish")
    bearish_count = sum(1 for t in trends if t == "bearish")
    
    if bullish_count >= 2:
        direction = "bullish"
        htf_aligned = mn_trend == "bullish" or wk_trend == "bullish"
        note = f"HTF bias: {mn_trend.upper()[0]}/{wk_trend.upper()[0]}/{d_trend.upper()[0]} -> Bullish"
    elif bearish_count >= 2:
        direction = "bearish"
        htf_aligned = mn_trend == "bearish" or wk_trend == "bearish"
        note = f"HTF bias: {mn_trend.upper()[0]}/{wk_trend.upper()[0]}/{d_trend.upper()[0]} -> Bearish"
    else:
        direction = d_trend if d_trend != "mixed" else "bullish"
        htf_aligned = False
        note = f"HTF bias: Mixed ({mn_trend[0].upper()}/{wk_trend[0].upper()}/{d_trend[0].upper()})"
    
    return direction, note, htf_aligned


def _location_context(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    price: float,
    direction: str,
) -> Tuple[str, bool]:
    """
    Check if price is at a key location (support/resistance zone).
    
    Returns:
        Tuple of (note, is_valid_location)
    """
    if not daily_candles or len(daily_candles) < 20:
        return "Location: Insufficient data", False
    
    highs = [c["high"] for c in daily_candles[-50:]] if len(daily_candles) >= 50 else [c["high"] for c in daily_candles]
    lows = [c["low"] for c in daily_candles[-50:]] if len(daily_candles) >= 50 else [c["low"] for c in daily_candles]
    
    recent_high = max(highs[-20:])
    recent_low = min(lows[-20:])
    range_size = recent_high - recent_low
    
    if range_size <= 0:
        return "Location: No range", False
    
    swing_highs, swing_lows = _find_pivots(daily_candles[-50:] if len(daily_candles) >= 50 else daily_candles, lookback=3)
    
    atr = _atr(daily_candles, 14)
    zone_tolerance = atr * 0.5 if atr > 0 else range_size * 0.05
    
    if direction == "bullish":
        near_support = any(abs(price - sl) < zone_tolerance for sl in swing_lows[-5:]) if swing_lows else False
        near_range_low = (price - recent_low) < range_size * 0.3
        
        if near_support or near_range_low:
            return "Location: Near support zone", True
        else:
            return "Location: Not at key support", False
    else:
        near_resistance = any(abs(price - sh) < zone_tolerance for sh in swing_highs[-5:]) if swing_highs else False
        near_range_high = (recent_high - price) < range_size * 0.3
        
        if near_resistance or near_range_high:
            return "Location: Near resistance zone", True
        else:
            return "Location: Not at key resistance", False


def _fib_context(
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    direction: str,
    price: float,
    fib_low: float = 0.382,
    fib_high: float = 0.886,
) -> Tuple[str, bool]:
    """
    Check if price is within a Fibonacci retracement zone.
    
    Returns:
        Tuple of (note, is_in_fib_zone)
    """
    try:
        candles = daily_candles if daily_candles and len(daily_candles) >= 30 else weekly_candles
        
        if not candles or len(candles) < 20:
            return "Fib: Insufficient data", False
        
        leg = _find_last_swing_leg_for_fib(candles, direction)
        
        if not leg:
            return "Fib: No clear swing leg found", False
        
        lo, hi = leg
        span = hi - lo
        
        if span <= 0:
            return "Fib: Invalid swing range", False
        
        if direction == "bullish":
            fib_382 = hi - span * 0.382
            fib_500 = hi - span * 0.5
            fib_618 = hi - span * 0.618
            fib_786 = hi - span * 0.786
            
            if fib_786 <= price <= fib_382:
                level = round((hi - price) / span, 3)
                return f"Fib: Price at {level:.1%} retracement (Golden Pocket zone)", True
            elif fib_618 <= price <= fib_500:
                return "Fib: Price at 50-61.8% zone", True
            else:
                return "Fib: Price outside retracement zone", False
        else:
            fib_382 = lo + span * 0.382
            fib_500 = lo + span * 0.5
            fib_618 = lo + span * 0.618
            fib_786 = lo + span * 0.786
            
            if fib_382 <= price <= fib_786:
                level = round((price - lo) / span, 3)
                return f"Fib: Price at {level:.1%} retracement (Golden Pocket zone)", True
            elif fib_500 <= price <= fib_618:
                return "Fib: Price at 50-61.8% zone", True
            else:
                return "Fib: Price outside retracement zone", False
    except Exception as e:
        return f"Fib: Error calculating ({type(e).__name__})", False


def _find_last_swing_leg_for_fib(candles: List[Dict], direction: str) -> Optional[Tuple[float, float]]:
    """
    Find the last swing leg for Fibonacci calculation.
    
    Returns:
        Tuple of (swing_low, swing_high) or None
    """
    if not candles or len(candles) < 20:
        return None
    
    try:
        swing_highs, swing_lows = _find_pivots(candles, lookback=3)
    except Exception:
        swing_highs, swing_lows = [], []
    
    if not swing_highs or not swing_lows:
        try:
            highs = [c["high"] for c in candles[-30:] if "high" in c]
            lows = [c["low"] for c in candles[-30:] if "low" in c]
            if highs and lows:
                return (min(lows), max(highs))
        except Exception:
            pass
        return None
    
    try:
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        if recent_highs and recent_lows:
            hi = max(recent_highs)
            lo = min(recent_lows)
            return (lo, hi)
    except Exception:
        pass
    
    return None


def _daily_liquidity_context(candles: List[Dict], price: float) -> Tuple[str, bool]:
    """
    Check for liquidity sweep or proximity to liquidity pools.
    
    Returns:
        Tuple of (note, is_near_liquidity)
    """
    try:
        if not candles or len(candles) < 10:
            return "Liquidity: Insufficient data", False
        
        lookback = min(20, len(candles))
        recent = candles[-lookback:]
        
        recent_highs = [c["high"] for c in recent if "high" in c]
        recent_lows = [c["low"] for c in recent if "low" in c]
        
        if not recent_highs or not recent_lows:
            return "Liquidity: Invalid data", False
        
        equal_highs = []
        equal_lows = []
        
        atr = _atr(candles, 14)
        tolerance = atr * 0.2 if atr > 0 else (max(recent_highs) - min(recent_lows)) * 0.02
        
        for i, h in enumerate(recent_highs):
            for j, h2 in enumerate(recent_highs):
                if i != j and abs(h - h2) < tolerance:
                    equal_highs.append(h)
                    break
        
        for i, l in enumerate(recent_lows):
            for j, l2 in enumerate(recent_lows):
                if i != j and abs(l - l2) < tolerance:
                    equal_lows.append(l)
                    break
        
        near_equal_high = any(abs(price - h) < tolerance * 2 for h in equal_highs)
        near_equal_low = any(abs(price - l) < tolerance * 2 for l in equal_lows)
        
        current = candles[-1]
        prev = candles[-2] if len(candles) >= 2 else None
        
        swept_high = False
        swept_low = False
        
        if prev:
            prev_high = max(c["high"] for c in candles[-10:-1] if "high" in c)
            prev_low = min(c["low"] for c in candles[-10:-1] if "low" in c)
            
            if current.get("high", 0) > prev_high and current.get("close", 0) < prev_high:
                swept_high = True
            if current.get("low", float("inf")) < prev_low and current.get("close", float("inf")) > prev_low:
                swept_low = True
        
        if swept_high or swept_low:
            return "Liquidity: Sweep detected", True
        elif near_equal_high or near_equal_low:
            return "Liquidity: Near equal highs/lows", True
        else:
            return "Liquidity: No clear liquidity zone", False
    except Exception as e:
        return f"Liquidity: Error ({type(e).__name__})", False


def _structure_context(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    direction: str,
) -> Tuple[bool, str]:
    """
    Check market structure alignment (BOS/CHoCH).
    
    Returns:
        Tuple of (is_aligned, note)
    """
    if not daily_candles or len(daily_candles) < 10:
        return False, "Structure: Insufficient data"
    
    swing_highs, swing_lows = _find_pivots(daily_candles[-30:], lookback=3)
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, "Structure: Not enough swing points"
    
    if direction == "bullish":
        higher_low = swing_lows[-1] > swing_lows[-2] if len(swing_lows) >= 2 else False
        higher_high = swing_highs[-1] > swing_highs[-2] if len(swing_highs) >= 2 else False
        
        bos_up = daily_candles[-1]["close"] > max(swing_highs[-3:]) if swing_highs else False
        
        if bos_up:
            return True, "Structure: BOS up confirmed"
        elif higher_low and higher_high:
            return True, "Structure: HH/HL pattern (bullish)"
        elif higher_low:
            return True, "Structure: Higher low formed"
        else:
            return False, "Structure: No bullish structure"
    else:
        lower_high = swing_highs[-1] < swing_highs[-2] if len(swing_highs) >= 2 else False
        lower_low = swing_lows[-1] < swing_lows[-2] if len(swing_lows) >= 2 else False
        
        bos_down = daily_candles[-1]["close"] < min(swing_lows[-3:]) if swing_lows else False
        
        if bos_down:
            return True, "Structure: BOS down confirmed"
        elif lower_high and lower_low:
            return True, "Structure: LH/LL pattern (bearish)"
        elif lower_high:
            return True, "Structure: Lower high formed"
        else:
            return False, "Structure: No bearish structure"


def _h4_confirmation(
    h4_candles: List[Dict],
    direction: str,
    daily_candles: List[Dict],
) -> Tuple[str, bool]:
    """
    Check for 4H timeframe confirmation (entry trigger).
    
    Returns:
        Tuple of (note, is_confirmed)
    """
    candles = h4_candles if h4_candles and len(h4_candles) >= 5 else daily_candles[-10:]
    
    if not candles or len(candles) < 3:
        return "4H: Insufficient data", False
    
    last = candles[-1]
    prev = candles[-2]
    
    body_last = abs(last["close"] - last["open"])
    range_last = last["high"] - last["low"]
    body_ratio = body_last / range_last if range_last > 0 else 0
    
    if direction == "bullish":
        bullish_candle = last["close"] > last["open"]
        engulfing = (
            last["close"] > last["open"] and
            prev["close"] < prev["open"] and
            last["close"] > prev["open"] and
            last["open"] < prev["close"]
        )
        
        lower_wick = last["open"] - last["low"] if last["close"] > last["open"] else last["close"] - last["low"]
        upper_wick = last["high"] - last["close"] if last["close"] > last["open"] else last["high"] - last["open"]
        pin_bar = lower_wick > body_last * 2 and upper_wick < body_last * 0.5
        
        bos_check = last["high"] > max(c["high"] for c in candles[-5:-1]) if len(candles) >= 5 else False
        
        if engulfing:
            return "4H: Bullish engulfing confirmed", True
        elif pin_bar:
            return "4H: Bullish pin bar (rejection)", True
        elif bos_check and bullish_candle:
            return "4H: Break of structure up", True
        elif bullish_candle and body_ratio > 0.6:
            return "4H: Strong bullish candle", True
        else:
            return "4H: Awaiting bullish confirmation", False
    else:
        bearish_candle = last["close"] < last["open"]
        engulfing = (
            last["close"] < last["open"] and
            prev["close"] > prev["open"] and
            last["close"] < prev["open"] and
            last["open"] > prev["close"]
        )
        
        upper_wick = last["high"] - last["open"] if last["close"] < last["open"] else last["high"] - last["close"]
        lower_wick = last["close"] - last["low"] if last["close"] < last["open"] else last["open"] - last["low"]
        pin_bar = upper_wick > body_last * 2 and lower_wick < body_last * 0.5
        
        bos_check = last["low"] < min(c["low"] for c in candles[-5:-1]) if len(candles) >= 5 else False
        
        if engulfing:
            return "4H: Bearish engulfing confirmed", True
        elif pin_bar:
            return "4H: Bearish pin bar (rejection)", True
        elif bos_check and bearish_candle:
            return "4H: Break of structure down", True
        elif bearish_candle and body_ratio > 0.6:
            return "4H: Strong bearish candle", True
        else:
            return "4H: Awaiting bearish confirmation", False


def _find_structure_sl(candles: List[Dict], direction: str, lookback: int = 35) -> Optional[float]:
    """
    Find structure-based stop loss level.
    
    Returns:
        Stop loss price level or None
    """
    if not candles or len(candles) < 5:
        return None
    
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    swing_highs, swing_lows = _find_pivots(recent, lookback=3)
    
    if direction == "bullish":
        if swing_lows:
            return min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
        else:
            return min(c["low"] for c in recent[-10:])
    else:
        if swing_highs:
            return max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
        else:
            return max(c["high"] for c in recent[-10:])


def _compute_confluence_flags(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
) -> Tuple[Dict[str, bool], Dict[str, str], Tuple]:
    """
    Compute all confluence flags for a trading setup.
    
    This is the main entry point for confluence calculation,
    used by both backtests and live scanning.
    
    Returns:
        Tuple of (flags dict, notes dict, trade_levels tuple)
    """
    return compute_confluence(
        monthly_candles, weekly_candles, daily_candles, h4_candles, direction, params
    )


def compute_confluence(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
) -> Tuple[Dict[str, bool], Dict[str, str], Tuple]:
    """
    Compute confluence flags for a given setup.
    
    Uses the same core logic as strategy.py but with parameterization.
    
    Args:
        monthly_candles: Monthly OHLCV data
        weekly_candles: Weekly OHLCV data
        daily_candles: Daily OHLCV data
        h4_candles: 4H OHLCV data
        direction: Trade direction ("bullish" or "bearish")
        params: Strategy parameters (uses defaults if None)
    
    Returns:
        Tuple of (flags dict, notes dict, trade_levels tuple)
    """
    if params is None:
        params = StrategyParams()
    
    price = daily_candles[-1]["close"] if daily_candles else float("nan")
    
    mn_trend = _infer_trend(monthly_candles) if monthly_candles else "mixed"
    wk_trend = _infer_trend(weekly_candles) if weekly_candles else "mixed"
    d_trend = _infer_trend(daily_candles) if daily_candles else "mixed"
    _, htf_note_text, htf_ok = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
    
    if params.use_htf_filter:
        loc_note, loc_ok = _location_context(
            monthly_candles, weekly_candles, daily_candles, price, direction
        )
    else:
        loc_note, loc_ok = "Location filter disabled", True
    
    if params.use_fib_filter:
        fib_note, fib_ok = _fib_context(weekly_candles, daily_candles, direction, price)
    else:
        fib_note, fib_ok = "Fib filter disabled", True
    
    if params.use_liquidity_filter:
        liq_note, liq_ok = _daily_liquidity_context(daily_candles, price)
    else:
        liq_note, liq_ok = "Liquidity filter disabled", True
    
    if params.use_structure_filter:
        struct_ok, struct_note = _structure_context(
            monthly_candles, weekly_candles, daily_candles, direction
        )
    else:
        struct_ok, struct_note = True, "Structure filter disabled"
    
    if params.use_confirmation_filter:
        conf_note, conf_ok = _h4_confirmation(h4_candles, direction, daily_candles)
    else:
        conf_note, conf_ok = "Confirmation filter disabled", True
    
    rr_note, rr_ok, entry, sl, tp1, tp2, tp3, tp4, tp5 = compute_trade_levels(
        daily_candles, direction, params
    )
    
    flags = {
        "htf_bias": htf_ok,
        "location": loc_ok,
        "fib": fib_ok,
        "liquidity": liq_ok,
        "structure": struct_ok,
        "confirmation": conf_ok,
        "rr": rr_ok,
    }
    
    notes = {
        "htf_bias": htf_note_text,
        "location": loc_note,
        "fib": fib_note,
        "liquidity": liq_note,
        "structure": struct_note,
        "confirmation": conf_note,
        "rr": rr_note,
    }
    
    trade_levels = (entry, sl, tp1, tp2, tp3, tp4, tp5)
    return flags, notes, trade_levels


def compute_trade_levels(
    daily_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
) -> Tuple[str, bool, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute entry, SL, and TP levels using Fibonacci extensions.
    
    Blueprint HTF Confluence Strategy:
    - Entry: At 0.618-0.786 Fibonacci retracement of the swing leg
    - SL: Below swing low (bullish) or above swing high (bearish) with ATR buffer
    - TP levels: Fibonacci extensions beyond swing high/low
      - TP1: -0.25 extension (small target for quick profit)
      - TP2: -0.68 extension (medium target)
      - TP3: -1.0 extension (full measured move)
      - TP4: -1.5 extension (extended target)
      - TP5: -2.0 extension (maximum target)
    
    Args:
        daily_candles: Daily OHLCV data
        direction: Trade direction
        params: Strategy parameters
    
    Returns:
        Tuple of (note, is_valid, entry, sl, tp1, tp2, tp3, tp4, tp5)
    """
    if params is None:
        params = StrategyParams()
    
    if not daily_candles:
        return "R/R: no data.", False, None, None, None, None, None, None, None
    
    current = daily_candles[-1]["close"]
    atr = _atr(daily_candles, 14)
    
    if atr <= 0:
        return "R/R: ATR too small.", False, None, None, None, None, None, None, None
    
    leg = _find_last_swing_leg_for_fib(daily_candles, direction)
    
    if leg:
        lo, hi = leg
        span = hi - lo
        
        if span > 0:
            if direction == "bullish":
                fib_618 = hi - span * 0.618
                fib_786 = hi - span * 0.786
                
                if fib_786 <= current <= fib_618:
                    entry = current
                elif abs(current - fib_618) < atr * 0.5:
                    entry = current
                else:
                    entry = fib_618
                
                sl = lo - atr * 0.3
                risk = entry - sl
                
                if risk > 0:
                    tp1 = hi + span * 0.25
                    tp2 = hi + span * 0.68
                    tp3 = hi + span * 1.0
                    tp4 = hi + span * 1.5
                    tp5 = hi + span * 2.0
                    
                    rr_ratio = (tp1 - entry) / risk
                    note = f"R/R: Fib extension levels (Entry={entry:.5f}, SL={sl:.5f}, R:R={rr_ratio:.2f})"
                    return note, rr_ratio >= 1.0, entry, sl, tp1, tp2, tp3, tp4, tp5
            else:
                fib_618 = lo + span * 0.618
                fib_786 = lo + span * 0.786
                
                if fib_618 <= current <= fib_786:
                    entry = current
                elif abs(current - fib_618) < atr * 0.5:
                    entry = current
                else:
                    entry = fib_618
                
                sl = hi + atr * 0.3
                risk = sl - entry
                
                if risk > 0:
                    tp1 = lo - span * 0.25
                    tp2 = lo - span * 0.68
                    tp3 = lo - span * 1.0
                    tp4 = lo - span * 1.5
                    tp5 = lo - span * 2.0
                    
                    rr_ratio = (entry - tp1) / risk
                    note = f"R/R: Fib extension levels (Entry={entry:.5f}, SL={sl:.5f}, R:R={rr_ratio:.2f})"
                    return note, rr_ratio >= 1.0, entry, sl, tp1, tp2, tp3, tp4, tp5
    
    entry = current
    sl_mult = params.atr_sl_multiplier
    
    if direction == "bullish":
        sl = entry - atr * sl_mult
        risk = atr * sl_mult
        tp1 = entry + risk * params.atr_tp1_multiplier
        tp2 = entry + risk * params.atr_tp2_multiplier
        tp3 = entry + risk * params.atr_tp3_multiplier
        tp4 = entry + risk * 2.5
        tp5 = entry + risk * 3.5
    else:
        sl = entry + atr * sl_mult
        risk = atr * sl_mult
        tp1 = entry - risk * params.atr_tp1_multiplier
        tp2 = entry - risk * params.atr_tp2_multiplier
        tp3 = entry - risk * params.atr_tp3_multiplier
        tp4 = entry - risk * 2.5
        tp5 = entry - risk * 3.5
    
    note = f"R/R: ATR-based fallback (risk={risk:.5f})"
    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5


def generate_signals(
    candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
    monthly_candles: Optional[List[Dict]] = None,
    weekly_candles: Optional[List[Dict]] = None,
    h4_candles: Optional[List[Dict]] = None,
    use_n_pattern: bool = True,
) -> List[Signal]:
    """
    Generate trading signals from historical candles using N-pattern detection.
    
    This function uses the Blueprint N-pattern strategy:
    1. Find impulse candles (strong moves that break structure)
    2. Wait for price to retrace into the Golden Pocket (0.618-0.786)
    3. Enter when price touches or closes in the Golden Pocket
    4. Set TPs at Fibonacci extensions from the impulse leg
    
    The function walks through candles sequentially (no look-ahead bias).
    
    Args:
        candles: Daily OHLCV candles (oldest to newest)
        symbol: Asset symbol
        params: Strategy parameters
        monthly_candles: Optional monthly data
        weekly_candles: Optional weekly data
        h4_candles: Optional 4H data
        use_n_pattern: Whether to use N-pattern detection (default True)
    
    Returns:
        List of Signal objects
    """
    if params is None:
        params = StrategyParams()
    
    if len(candles) < 50:
        return []
    
    signals = []
    used_impulse_indices: set = set()
    
    for i in range(50, len(candles)):
        try:
            daily_slice = candles[:i+1]
            
            weekly_slice = weekly_candles[:i//5+1] if weekly_candles else None
            monthly_slice = monthly_candles[:i//20+1] if monthly_candles else None
            h4_slice = h4_candles[:i*6+1] if h4_candles else None
            
            mn_trend = _infer_trend(monthly_slice) if monthly_slice else _infer_trend(daily_slice[-60:])
            wk_trend = _infer_trend(weekly_slice) if weekly_slice else _infer_trend(daily_slice[-20:])
            d_trend = _infer_trend(daily_slice[-10:])
            
            direction, htf_note, htf_aligned = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
            
            n_pattern_entry = False
            n_pattern_levels = None
            n_pattern_impulse = None
            
            if use_n_pattern:
                impulses = find_all_impulse_candles(
                    daily_slice,
                    direction,
                    lookback=20,
                    body_multiplier=1.5,
                    structure_lookback=5,
                    max_age_bars=30,
                )
                
                for impulse in impulses:
                    if impulse.index in used_impulse_indices:
                        continue
                    
                    bars_since_impulse = i - impulse.index
                    if bars_since_impulse < 1 or bars_since_impulse > 20:
                        continue
                    
                    is_entry, entry, sl, tp1, tp2, tp3, tp4, tp5, note = check_golden_pocket_entry(
                        daily_slice,
                        impulse,
                        i,
                    )
                    
                    if is_entry and entry is not None and sl is not None and tp1 is not None:
                        n_pattern_entry = True
                        n_pattern_levels = (entry, sl, tp1, tp2, tp3, tp4, tp5)
                        n_pattern_impulse = impulse
                        used_impulse_indices.add(impulse.index)
                        break
            
            if n_pattern_entry and n_pattern_levels:
                entry, sl, tp1, tp2, tp3, tp4, tp5 = n_pattern_levels
                
                flags = {
                    "htf_bias": htf_aligned,
                    "location": True,
                    "fib": True,
                    "liquidity": True,
                    "structure": True,
                    "confirmation": True,
                    "rr": True,
                    "n_pattern": True,
                }
                
                risk = abs(entry - sl) if entry and sl else 0
                reward = abs(tp1 - entry) if tp1 and entry else 0
                rr = reward / risk if risk > 0 else 0
                
                impulse_bar = n_pattern_impulse.index if n_pattern_impulse else "N/A"
                notes = {
                    "htf_bias": htf_note,
                    "location": "N-Pattern: Golden Pocket entry",
                    "fib": f"Fib: Entry at 0.618-0.786 zone",
                    "liquidity": "N-Pattern: Impulse broke structure",
                    "structure": f"Structure: Impulse at bar {impulse_bar}",
                    "confirmation": "N-Pattern entry confirmed",
                    "rr": f"R/R: {rr:.2f} (TP1 at -0.25 ext)",
                }
                
                confluence_score = sum(1 for v in flags.values() if v)
                quality_factors = 5
                
                candle = candles[i]
                timestamp = candle.get("time") or candle.get("timestamp") or candle.get("date")
                
                signal = Signal(
                    symbol=symbol,
                    direction=direction,
                    bar_index=i,
                    timestamp=timestamp,
                    confluence_score=confluence_score,
                    quality_factors=quality_factors,
                    entry=entry,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    is_active=True,
                    is_watching=False,
                    flags=flags,
                    notes=notes,
                )
                signals.append(signal)
                continue
            
            flags, notes, trade_levels = compute_confluence(
                monthly_slice or [],
                weekly_slice or [],
                daily_slice,
                h4_slice or daily_slice[-20:],
                direction,
                params,
            )
            
            entry, sl, tp1, tp2, tp3, tp4, tp5 = trade_levels
            
            confluence_score = sum(1 for v in flags.values() if v)
            
            quality_factors = sum([
                flags.get("location", False),
                flags.get("fib", False),
                flags.get("liquidity", False),
                flags.get("structure", False),
                flags.get("htf_bias", False),
            ])
            
            has_rr = flags.get("rr", False)
            has_confirmation = flags.get("confirmation", False)
            
            is_active = False
            is_watching = False
            
            if confluence_score >= params.min_confluence and quality_factors >= params.min_quality_factors:
                if params.require_rr_for_active and not has_rr:
                    is_watching = True
                elif params.require_confirmation_for_active and not has_confirmation:
                    is_watching = True
                else:
                    is_active = True
            elif confluence_score >= params.min_confluence - 1:
                is_watching = True
            
            if is_active or is_watching:
                candle = candles[i]
                timestamp = candle.get("time") or candle.get("timestamp") or candle.get("date")
                
                signal = Signal(
                    symbol=symbol,
                    direction=direction,
                    bar_index=i,
                    timestamp=timestamp,
                    confluence_score=confluence_score,
                    quality_factors=quality_factors,
                    entry=entry,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    is_active=is_active,
                    is_watching=is_watching,
                    flags=flags,
                    notes=notes,
                )
                signals.append(signal)
        except Exception:
            continue
    
    return signals


def generate_signals_n_pattern_only(
    candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
    monthly_candles: Optional[List[Dict]] = None,
    weekly_candles: Optional[List[Dict]] = None,
    h4_candles: Optional[List[Dict]] = None,
) -> List[Signal]:
    """
    Generate trading signals using ONLY the N-pattern detection system.
    
    This is a pure implementation of the Blueprint N-pattern strategy:
    1. Find impulse candles (body > 1.5x avg, breaks structure)
    2. Wait for Golden Pocket retracement (0.618-0.786)
    3. Entry on close/touch of Golden Pocket
    4. SL below retracement low + 0.3 ATR buffer
    5. TPs at Fib extensions (-0.25, -0.68, -1.0, -1.5, -2.0)
    
    This function is designed to generate ~50+ trades per year per asset.
    
    Args:
        candles: Daily OHLCV candles (oldest to newest)
        symbol: Asset symbol
        params: Strategy parameters
        monthly_candles: Optional monthly data for trend
        weekly_candles: Optional weekly data for trend
        h4_candles: Optional 4H data
    
    Returns:
        List of Signal objects (all ACTIVE, ready to trade)
    """
    if params is None:
        params = StrategyParams()
    
    if len(candles) < 50:
        return []
    
    signals = []
    used_impulse_indices: set = set()
    cooldown_until = -1
    
    for i in range(50, len(candles)):
        if i <= cooldown_until:
            continue
        
        try:
            daily_slice = candles[:i+1]
            
            weekly_slice = weekly_candles[:i//5+1] if weekly_candles else None
            monthly_slice = monthly_candles[:i//20+1] if monthly_candles else None
            
            mn_trend = _infer_trend(monthly_slice) if monthly_slice else _infer_trend(daily_slice[-60:])
            wk_trend = _infer_trend(weekly_slice) if weekly_slice else _infer_trend(daily_slice[-20:])
            d_trend = _infer_trend(daily_slice[-10:])
            
            direction, htf_note, htf_aligned = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
            
            bullish_impulses = find_all_impulse_candles(
                daily_slice, "bullish",
                lookback=20, body_multiplier=1.5,
                structure_lookback=5, max_age_bars=25,
            )
            
            bearish_impulses = find_all_impulse_candles(
                daily_slice, "bearish",
                lookback=20, body_multiplier=1.5,
                structure_lookback=5, max_age_bars=25,
            )
            
            all_impulses = []
            if direction == "bullish" or direction == "mixed":
                all_impulses.extend(bullish_impulses)
            if direction == "bearish" or direction == "mixed":
                all_impulses.extend(bearish_impulses)
            
            all_impulses.sort(key=lambda x: x.index, reverse=True)
            
            for impulse in all_impulses:
                if impulse.index in used_impulse_indices:
                    continue
                
                bars_since = i - impulse.index
                if bars_since < 1 or bars_since > 15:
                    continue
                
                is_entry, entry, sl, tp1, tp2, tp3, tp4, tp5, note = check_golden_pocket_entry(
                    daily_slice, impulse, i,
                )
                
                if not is_entry:
                    continue
                
                if entry is None or sl is None or tp1 is None:
                    continue
                
                risk = abs(entry - sl)
                reward = abs(tp1 - entry)
                rr = reward / risk if risk > 0 else 0
                
                if rr < 1.0:
                    continue
                
                used_impulse_indices.add(impulse.index)
                cooldown_until = i + params.cooldown_bars
                
                flags = {
                    "htf_bias": htf_aligned,
                    "location": True,
                    "fib": True,
                    "liquidity": True,
                    "structure": True,
                    "confirmation": True,
                    "rr": rr >= 1.0,
                    "n_pattern": True,
                }
                
                notes_dict = {
                    "htf_bias": htf_note,
                    "location": f"Golden Pocket: 0.618-0.786 zone",
                    "fib": f"Entry at retracement, TPs at extensions",
                    "liquidity": f"Impulse body: {impulse.body_size:.5f} ({impulse.body_size/impulse.avg_body:.1f}x avg)",
                    "structure": f"BOS at bar {impulse.index}, {impulse.direction}",
                    "confirmation": note,
                    "rr": f"R:R = {rr:.2f} (Entry={entry:.5f}, SL={sl:.5f}, TP1={tp1:.5f})",
                }
                
                candle = candles[i]
                timestamp = candle.get("time") or candle.get("timestamp") or candle.get("date")
                
                signal = Signal(
                    symbol=symbol,
                    direction=impulse.direction,
                    bar_index=i,
                    timestamp=timestamp,
                    confluence_score=7,
                    quality_factors=5,
                    entry=entry,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    is_active=True,
                    is_watching=False,
                    flags=flags,
                    notes=notes_dict,
                )
                signals.append(signal)
                break
                
        except Exception:
            continue
    
    return signals


def simulate_trades(
    candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
    monthly_candles: Optional[List[Dict]] = None,
    weekly_candles: Optional[List[Dict]] = None,
    h4_candles: Optional[List[Dict]] = None,
) -> List[Trade]:
    """
    Simulate trades through historical candles using the Blueprint strategy.
    
    This is a walk-forward simulation with no look-ahead bias.
    Uses the same logic as live trading but runs through historical data.
    
    Args:
        candles: Daily OHLCV candles (oldest to newest)
        symbol: Asset symbol
        params: Strategy parameters
        monthly_candles: Optional monthly data
        weekly_candles: Optional weekly data
        h4_candles: Optional 4H data
    
    Returns:
        List of completed Trade objects
    """
    if params is None:
        params = StrategyParams()
    
    signals = generate_signals(
        candles, symbol, params,
        monthly_candles, weekly_candles, h4_candles
    )
    
    active_signals = [s for s in signals if s.is_active]
    
    trades = []
    open_trade = None
    last_trade_bar = -params.cooldown_bars - 1
    
    for signal in active_signals:
        if signal.bar_index <= last_trade_bar + params.cooldown_bars:
            continue
        
        if signal.entry is None or signal.stop_loss is None or signal.tp1 is None:
            continue
        
        if open_trade is not None:
            continue
        
        entry_bar = signal.bar_index
        entry_price = signal.entry
        sl = signal.stop_loss
        tp1 = signal.tp1
        tp2 = signal.tp2
        tp3 = signal.tp3
        direction = signal.direction
        
        risk = abs(entry_price - sl)
        if risk <= 0:
            continue
        
        trade_closed = False
        tp1_hit = False
        trailing_sl = sl
        
        reward = 0.0
        rr = 0.0
        is_winner = False
        exit_reason = ""
        
        for exit_bar in range(entry_bar + 1, len(candles)):
            c = candles[exit_bar]
            high = c["high"]
            low = c["low"]
            exit_timestamp = c.get("time") or c.get("timestamp") or c.get("date")
            
            if direction == "bullish":
                if low <= trailing_sl:
                    if tp1_hit:
                        reward = trailing_sl - entry_price
                        rr = reward / risk
                        exit_reason = "TP1+Trail"
                        is_winner = rr >= 0
                    else:
                        reward = trailing_sl - entry_price
                        rr = -1.0
                        exit_reason = "SL"
                        is_winner = False
                    
                    trade_closed = True
                elif tp1 is not None and high >= tp1 and not tp1_hit:
                    reward = tp1 - entry_price
                    rr = reward / risk
                    exit_reason = "TP1"
                    is_winner = True
                    trade_closed = True
            else:
                if high >= trailing_sl:
                    if tp1_hit:
                        reward = entry_price - trailing_sl
                        rr = reward / risk
                        exit_reason = "TP1+Trail"
                        is_winner = rr >= 0
                    else:
                        reward = entry_price - trailing_sl
                        rr = -1.0
                        exit_reason = "SL"
                        is_winner = False
                    
                    trade_closed = True
                elif tp1 is not None and low <= tp1 and not tp1_hit:
                    reward = entry_price - tp1
                    rr = reward / risk
                    exit_reason = "TP1"
                    is_winner = True
                    trade_closed = True
            
            if trade_closed:
                entry_timestamp = signal.timestamp
                
                trade = Trade(
                    symbol=symbol,
                    direction=direction,
                    entry_date=entry_timestamp,
                    exit_date=exit_timestamp,
                    entry_price=entry_price,
                    exit_price=entry_price + reward if direction == "bullish" else entry_price - reward,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    risk=risk,
                    reward=reward,
                    rr=rr,
                    is_winner=is_winner,
                    exit_reason=exit_reason,
                    confluence_score=signal.confluence_score,
                )
                trades.append(trade)
                last_trade_bar = exit_bar
                break
        
        if not trade_closed:
            pass
    
    return trades


def get_default_params(asset: str = "") -> StrategyParams:
    """
    Get strategy parameters with asset-specific overrides.
    
    Each forex pair has parameters tuned for its volatility characteristics
    to achieve target metrics:
    - Win Rate: 70-100%
    - Annual Return: ≥50%
    - Trades per Year: ≥50
    - Max Drawdown: ≤30%
    
    Strategy approach:
    - Balanced confluence (2-3) for enough trade opportunities
    - Tight SL (1.0 ATR) for precise entries
    - Wide TPs (1.5R, 2.5R, 4.0R) to capture big moves
    - Asset-specific adjustments for volatility characteristics
    """
    params = StrategyParams()
    
    if asset == "EUR_USD":
        params.min_confluence = 2
        params.min_quality_factors = 1
        params.atr_sl_multiplier = 1.0
        params.atr_tp1_multiplier = 1.5
        params.atr_tp2_multiplier = 2.5
        params.atr_tp3_multiplier = 4.0
        params.require_confirmation_for_active = False
        params.require_rr_for_active = False
        params.min_rr_ratio = 1.0
        params.liquidity_sweep_lookback = 15
        params.cooldown_bars = 2
        
    elif asset == "GBP_USD":
        params.min_confluence = 2
        params.min_quality_factors = 1
        params.atr_sl_multiplier = 1.2
        params.atr_tp1_multiplier = 1.8
        params.atr_tp2_multiplier = 3.0
        params.atr_tp3_multiplier = 4.5
        params.require_confirmation_for_active = False
        params.require_rr_for_active = False
        params.min_rr_ratio = 1.0
        params.liquidity_sweep_lookback = 12
        params.cooldown_bars = 2
        
    elif asset == "USD_JPY":
        params.min_confluence = 2
        params.min_quality_factors = 1
        params.atr_sl_multiplier = 1.0
        params.atr_tp1_multiplier = 1.5
        params.atr_tp2_multiplier = 2.5
        params.atr_tp3_multiplier = 4.0
        params.require_confirmation_for_active = False
        params.require_rr_for_active = False
        params.min_rr_ratio = 1.0
        params.liquidity_sweep_lookback = 10
        params.cooldown_bars = 2
        
    elif asset == "USD_CHF":
        params.min_confluence = 2
        params.min_quality_factors = 1
        params.atr_sl_multiplier = 1.0
        params.atr_tp1_multiplier = 1.5
        params.atr_tp2_multiplier = 2.5
        params.atr_tp3_multiplier = 4.0
        params.require_confirmation_for_active = False
        params.require_rr_for_active = False
        params.min_rr_ratio = 1.0
        params.liquidity_sweep_lookback = 12
        params.cooldown_bars = 2
        
    elif asset == "USD_CAD":
        params.min_confluence = 2
        params.min_quality_factors = 1
        params.atr_sl_multiplier = 1.0
        params.atr_tp1_multiplier = 1.5
        params.atr_tp2_multiplier = 2.5
        params.atr_tp3_multiplier = 4.0
        params.require_confirmation_for_active = False
        params.require_rr_for_active = False
        params.min_rr_ratio = 1.0
        params.liquidity_sweep_lookback = 12
        params.cooldown_bars = 2
        
    elif asset == "AUD_USD":
        params.min_confluence = 2
        params.min_quality_factors = 1
        params.atr_sl_multiplier = 1.0
        params.atr_tp1_multiplier = 1.5
        params.atr_tp2_multiplier = 2.5
        params.atr_tp3_multiplier = 4.0
        params.require_confirmation_for_active = False
        params.require_rr_for_active = False
        params.min_rr_ratio = 1.0
        params.liquidity_sweep_lookback = 12
        params.cooldown_bars = 2
        
    elif asset == "NZD_USD":
        params.min_confluence = 2
        params.min_quality_factors = 1
        params.atr_sl_multiplier = 1.0
        params.atr_tp1_multiplier = 1.5
        params.atr_tp2_multiplier = 2.5
        params.atr_tp3_multiplier = 4.0
        params.require_confirmation_for_active = False
        params.require_rr_for_active = False
        params.min_rr_ratio = 1.0
        params.liquidity_sweep_lookback = 12
        params.cooldown_bars = 2
    
    return params


def get_aggressive_params() -> StrategyParams:
    """
    Get aggressive (more trades) strategy parameters.
    
    Uses wider TPs for better R:R while keeping confluence low
    to generate more trade opportunities.
    """
    return StrategyParams(
        min_confluence=1,
        min_quality_factors=0,
        require_confirmation_for_active=False,
        require_rr_for_active=False,
        atr_sl_multiplier=1.0,
        atr_tp1_multiplier=1.5,
        atr_tp2_multiplier=2.5,
        atr_tp3_multiplier=4.0,
        cooldown_bars=1,
    )


def get_conservative_params() -> StrategyParams:
    """
    Get conservative (higher quality) strategy parameters.
    
    Uses higher confluence for quality setups with wider TPs
    to maximize reward-to-risk on high-probability trades.
    """
    return StrategyParams(
        min_confluence=3,
        min_quality_factors=2,
        require_htf_alignment=True,
        require_confirmation_for_active=True,
        require_rr_for_active=True,
        atr_sl_multiplier=1.0,
        atr_tp1_multiplier=2.0,
        atr_tp2_multiplier=3.5,
        atr_tp3_multiplier=5.0,
        cooldown_bars=3,
    )


def generate_continuation_signals(
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    weekly_candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
) -> List[Signal]:
    """
    Generate V-pattern continuation signals using swing structure.
    
    Core pattern:
    1. Find confirmed swing structure (swing low → swing high for bullish)
    2. Price makes new high (impulse/BoS)
    3. Wait for retracement into GP zone (0.5-0.786)
    4. Enter on bullish/bearish confirmation candle
    5. SL below swing, TP at swing and extensions
    
    Args:
        daily_candles: Daily OHLCV data
        h4_candles: 4H OHLCV data (unused)
        weekly_candles: Weekly data for trend
        symbol: Asset symbol
        params: Strategy parameters
    
    Returns:
        List of Signal objects
    """
    if params is None:
        params = StrategyParams()
    
    if len(daily_candles) < 60:
        return []
    
    signals = []
    cooldown_until = -1
    active_setups: List[Dict] = []
    
    for i in range(40, len(daily_candles)):
        if i <= cooldown_until:
            continue
        
        try:
            current = daily_candles[i]
            c_open = current.get("open", 0)
            c_close = current.get("close", 0)
            c_high = current.get("high", 0)
            c_low = current.get("low", float("inf"))
            
            lookback = daily_candles[max(0, i-30):i]
            if len(lookback) < 15:
                continue
            
            daily_slice = daily_candles[:i+1]
            atr = _atr(daily_slice, 14)
            
            swing_highs = []
            swing_lows = []
            
            for j in range(2, len(lookback) - 2):
                h = lookback[j].get("high", 0)
                l = lookback[j].get("low", float("inf"))
                
                left_h = [lookback[k].get("high", 0) for k in range(max(0, j-2), j)]
                right_h = [lookback[k].get("high", 0) for k in range(j+1, min(len(lookback), j+3))]
                
                if left_h and right_h and all(x < h for x in left_h) and all(x < h for x in right_h):
                    swing_highs.append((j, h))
                
                left_l = [lookback[k].get("low", float("inf")) for k in range(max(0, j-2), j)]
                right_l = [lookback[k].get("low", float("inf")) for k in range(j+1, min(len(lookback), j+3))]
                
                if left_l and right_l and all(x > l for x in left_l) and all(x > l for x in right_l):
                    swing_lows.append((j, l))
            
            if swing_highs and swing_lows:
                sh_idx, swing_high = swing_highs[-1]
                sl_idx, swing_low = swing_lows[-1]
                
                swing_range = abs(swing_high - swing_low)
                
                if swing_range >= atr * 2:
                    if sl_idx < sh_idx:
                        is_new_high = c_high >= swing_high * 0.998
                        
                        if is_new_high:
                            fib_50 = swing_high - swing_range * 0.5
                            fib_618 = swing_high - swing_range * 0.618
                            fib_786 = swing_high - swing_range * 0.786
                            
                            setup = {
                                "direction": "bullish",
                                "swing_high": max(swing_high, c_high),
                                "swing_low": swing_low,
                                "swing_range": swing_range,
                                "fib_50": fib_50,
                                "fib_618": fib_618,
                                "fib_786": fib_786,
                                "impulse_bar": i,
                                "expires_bar": i + 15,
                            }
                            
                            existing = [s for s in active_setups if s["direction"] == "bullish" and s["swing_low"] == swing_low]
                            if not existing:
                                active_setups.append(setup)
                    
                    elif sh_idx < sl_idx:
                        is_new_low = c_low <= swing_low * 1.002
                        
                        if is_new_low:
                            fib_50 = swing_low + swing_range * 0.5
                            fib_618 = swing_low + swing_range * 0.618
                            fib_786 = swing_low + swing_range * 0.786
                            
                            setup = {
                                "direction": "bearish",
                                "swing_high": swing_high,
                                "swing_low": min(swing_low, c_low),
                                "swing_range": swing_range,
                                "fib_50": fib_50,
                                "fib_618": fib_618,
                                "fib_786": fib_786,
                                "impulse_bar": i,
                                "expires_bar": i + 15,
                            }
                            
                            existing = [s for s in active_setups if s["direction"] == "bearish" and s["swing_high"] == swing_high]
                            if not existing:
                                active_setups.append(setup)
            
            active_setups = [s for s in active_setups if s["expires_bar"] > i]
            
            for setup in active_setups[:]:
                if i <= setup.get("impulse_bar", i):
                    continue
                
                direction = setup["direction"]
                fib_50 = setup["fib_50"]
                fib_618 = setup["fib_618"]
                fib_786 = setup["fib_786"]
                swing_high = setup["swing_high"]
                swing_low = setup["swing_low"]
                swing_range = setup["swing_range"]
                
                if "swept" not in setup:
                    setup["swept"] = False
                    setup["deepest_retrace"] = None
                
                if direction == "bullish":
                    if c_low <= fib_786:
                        setup["swept"] = True
                        if setup["deepest_retrace"] is None or c_low < setup["deepest_retrace"]:
                            setup["deepest_retrace"] = c_low
                    
                    if setup["swept"]:
                        body = abs(c_close - c_open)
                        is_strong_bullish = c_close > c_open and body >= atr * 0.5
                        reclaimed_618 = c_close > fib_618
                        
                        if is_strong_bullish and reclaimed_618:
                            entry = c_close
                            deepest = setup["deepest_retrace"] if setup["deepest_retrace"] else fib_786
                            sl = deepest - atr * 0.3
                            risk = entry - sl
                            
                            if risk > 0:
                                tp1 = swing_high
                                tp2 = swing_high + swing_range * 0.382
                                tp3 = swing_high + swing_range * 0.618
                                
                                rr = (tp1 - entry) / risk
                                
                                if rr >= 1.2:
                                    timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                                    
                                    signal = Signal(
                                        symbol=symbol,
                                        direction="bullish",
                                        bar_index=i,
                                        timestamp=timestamp,
                                        confluence_score=6,
                                        quality_factors=4,
                                        entry=entry,
                                        stop_loss=sl,
                                        tp1=tp1,
                                        tp2=tp2,
                                        tp3=tp3,
                                        is_active=True,
                                        is_watching=False,
                                        flags={"swing": True, "golden_pocket": True, "swept": True, "reclaim": True},
                                        notes={"pattern": f"Bullish GP sweep+reclaim: {fib_618:.5f}"},
                                    )
                                    signals.append(signal)
                                    active_setups.remove(setup)
                                    cooldown_until = i + 3
                                    break
                
                elif direction == "bearish":
                    if c_high >= fib_786:
                        setup["swept"] = True
                        if setup["deepest_retrace"] is None or c_high > setup["deepest_retrace"]:
                            setup["deepest_retrace"] = c_high
                    
                    if setup["swept"]:
                        body = abs(c_close - c_open)
                        is_strong_bearish = c_close < c_open and body >= atr * 0.5
                        reclaimed_618 = c_close < fib_618
                        
                        if is_strong_bearish and reclaimed_618:
                            entry = c_close
                            deepest = setup["deepest_retrace"] if setup["deepest_retrace"] else fib_786
                            sl = deepest + atr * 0.3
                            risk = sl - entry
                            
                            if risk > 0:
                                tp1 = swing_low
                                tp2 = swing_low - swing_range * 0.382
                                tp3 = swing_low - swing_range * 0.618
                                
                                rr = (entry - tp1) / risk
                                
                                if rr >= 1.2:
                                    timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                                    
                                    signal = Signal(
                                        symbol=symbol,
                                        direction="bearish",
                                        bar_index=i,
                                        timestamp=timestamp,
                                        confluence_score=6,
                                        quality_factors=4,
                                        entry=entry,
                                        stop_loss=sl,
                                        tp1=tp1,
                                        tp2=tp2,
                                        tp3=tp3,
                                        is_active=True,
                                        is_watching=False,
                                        flags={"swing": True, "golden_pocket": True, "swept": True, "reclaim": True},
                                        notes={"pattern": f"Bearish GP sweep+reclaim: {fib_618:.5f}"},
                                    )
                                    signals.append(signal)
                                    active_setups.remove(setup)
                                    cooldown_until = i + 3
                                    break
                
        except Exception:
            continue
    
    return signals


def generate_signals_mtf(
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    weekly_candles: List[Dict],
    monthly_candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
) -> List[Signal]:
    """
    Generate signals using Multi-Timeframe (MTF) approach.
    
    TWO SIGNAL TYPES:
    1. CONTINUATION (V-patterns): Primary signal source, trade with weekly trend
       - Sideways → Impulse → Golden Pocket entry
       - More frequent, generates bulk of trades
    
    2. REVERSAL (at S/R): Secondary signal source, trade against trend at key levels
       - Only at Weekly/Monthly S/R zones
       - Requires BOS confirmation
       - Less frequent, higher conviction
    
    Args:
        daily_candles: Daily OHLCV data
        h4_candles: 4H OHLCV data for precise entry
        weekly_candles: Weekly data for S/R zones and trend
        monthly_candles: Monthly data for S/R zones
        symbol: Asset symbol
        params: Strategy parameters
    
    Returns:
        List of Signal objects combining both continuation and reversal signals
    """
    if params is None:
        params = StrategyParams()
    
    if len(daily_candles) < 50:
        return []
    
    continuation_signals = generate_continuation_signals(
        daily_candles=daily_candles,
        h4_candles=h4_candles,
        weekly_candles=weekly_candles,
        symbol=symbol,
        params=params,
    )
    
    reversal_signals = generate_reversal_signals(
        daily_candles=daily_candles,
        h4_candles=h4_candles,
        weekly_candles=weekly_candles,
        monthly_candles=monthly_candles,
        symbol=symbol,
        params=params,
    )
    
    all_signals = continuation_signals + reversal_signals
    all_signals.sort(key=lambda s: s.bar_index)
    
    return all_signals


def generate_reversal_signals(
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    weekly_candles: List[Dict],
    monthly_candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
) -> List[Signal]:
    """
    Generate REVERSAL signals at Weekly/Monthly S/R zones.
    
    This is for head & shoulders / inverse H&S patterns at key levels.
    
    Requirements:
    1. Price at Weekly/Monthly S/R zone
    2. Daily BOS (Break of Structure) in reversal direction
    3. Golden Pocket entry opportunity
    
    Args:
        daily_candles: Daily OHLCV data
        h4_candles: 4H OHLCV data
        weekly_candles: Weekly data for S/R zones
        monthly_candles: Monthly data for S/R zones
        symbol: Asset symbol
        params: Strategy parameters
    
    Returns:
        List of Signal objects
    """
    if params is None:
        params = StrategyParams()
    
    if len(daily_candles) < 50:
        return []
    
    monthly_res, monthly_sup = find_sr_zones(monthly_candles, lookback=2) if monthly_candles else ([], [])
    weekly_res, weekly_sup = find_sr_zones(weekly_candles, lookback=3) if weekly_candles else ([], [])
    
    all_resistance = monthly_res + weekly_res
    all_support = monthly_sup + weekly_sup
    
    signals = []
    used_bars: set = set()
    
    for i in range(50, len(daily_candles)):
        if i in used_bars:
            continue
        
        try:
            daily_slice = daily_candles[:i+1]
            current_candle = daily_candles[i]
            current_close = current_candle.get("close", 0)
            daily_atr = _atr(daily_slice, 14)
            
            is_at_sr, zone_type, sr_direction = check_at_sr_zone(
                current_close, all_resistance, all_support, daily_atr, zone_tolerance_atr=2.0
            )
            
            if not is_at_sr:
                continue
            
            has_bos, bos_direction = detect_daily_bos(daily_slice, lookback=10)
            
            if not has_bos:
                continue
            
            if zone_type == "resistance" and bos_direction == "bearish":
                direction = "bearish"
            elif zone_type == "support" and bos_direction == "bullish":
                direction = "bullish"
            else:
                continue
            
            impulses = find_all_impulse_candles(
                daily_slice, direction,
                lookback=15, body_multiplier=1.0,
                structure_lookback=5, max_age_bars=20,
            )
            
            if not impulses:
                continue
            
            impulse = impulses[-1]
            imp_range = impulse.impulse_range
            
            if imp_range <= 0:
                continue
            
            if direction == "bullish":
                gp_high = impulse.high - imp_range * 0.618
                gp_low = impulse.high - imp_range * 0.786
                recent_low = min(c.get("low", float("inf")) for c in daily_slice[-5:])
                entry = current_close
                sl = recent_low - daily_atr * 0.5
                tp1 = impulse.high + imp_range * 0.25
                tp2 = impulse.high + imp_range * 0.68
                tp3 = impulse.high + imp_range * 1.0
            else:
                gp_low = impulse.low + imp_range * 0.618
                gp_high = impulse.low + imp_range * 0.786
                recent_high = max(c.get("high", 0) for c in daily_slice[-5:])
                entry = current_close
                sl = recent_high + daily_atr * 0.5
                tp1 = impulse.low - imp_range * 0.25
                tp2 = impulse.low - imp_range * 0.68
                tp3 = impulse.low - imp_range * 1.0
            
            risk = abs(entry - sl)
            if risk <= 0:
                continue
            
            reward = abs(tp1 - entry)
            rr = reward / risk
            
            if rr < 1.0:
                continue
            
            flags = {
                "htf_bias": True,
                "location": True,
                "fib": True,
                "liquidity": True,
                "structure": True,
                "confirmation": True,
                "rr": rr >= 1.5,
                "reversal": True,
            }
            
            notes = {
                "htf_bias": f"At {zone_type} with {bos_direction} BOS",
                "location": f"W/M S/R zone",
                "fib": f"Golden Pocket: {gp_low:.5f} - {gp_high:.5f}",
                "liquidity": f"Reversal pattern",
                "structure": f"BOS: {bos_direction}",
                "confirmation": f"S/R + BOS alignment",
                "rr": f"R:R = {rr:.2f}",
            }
            
            confluence_score = sum(1 for v in flags.values() if v)
            
            timestamp = current_candle.get("time") or current_candle.get("timestamp") or current_candle.get("date")
            
            signal = Signal(
                symbol=symbol,
                direction=direction,
                bar_index=i,
                timestamp=timestamp,
                confluence_score=confluence_score,
                quality_factors=5,
                entry=entry,
                stop_loss=sl,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                is_active=True,
                is_watching=False,
                flags=flags,
                notes=notes,
            )
            signals.append(signal)
            used_bars.add(i)
            
            for j in range(1, 6):
                used_bars.add(i + j)
                
        except Exception:
            continue
    
    return signals


@dataclass
class HTFLevel:
    """Represents a Higher Timeframe Support/Resistance level."""
    price: float
    level_type: str
    timeframe: str
    strength: int
    last_touch_bar: int
    touch_count: int = 1
    breached: bool = False
    breached_direction: Optional[str] = None


@dataclass
class HTFConfluenceParams:
    """Parameters for HTF confluence strategy.
    
    Tuned for higher trade frequency while maintaining quality:
    - Widened sr_tolerance_atr (0.5 → 1.0) for broader S/R zone entries
    - Reduced cooldown_bars (2 → 1) to allow more frequent signals
    - Lowered min_confluence thresholds to increase trade count
    - Recalibrated TP ratios for earlier partials and tighter stops
    """
    fib_low: float = 0.55
    fib_high: float = 0.79
    min_reversal_confluence: int = 3
    min_continuation_confluence: int = 2
    max_concurrent_setups: int = 10
    cooldown_bars: int = 1
    pivot_lookback: int = 3
    sr_tolerance_atr: float = 1.0
    body_min_atr_ratio: float = 0.4
    sl_buffer_risk_ratio: float = 0.33
    sl_buffer_atr_ratio: float = 0.3
    tp1_rr: float = 1.2
    tp2_rr: float = 2.0
    tp3_rr: float = 3.5
    ema_short: int = 8
    ema_long: int = 21


def _detect_fractal_pivots_htf(
    candles: List[Dict],
    lookback: int = 3,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Detect fractal pivot highs and lows for HTF S/R detection.
    
    A fractal pivot is a swing point where price is higher/lower than
    surrounding bars on both sides.
    
    Args:
        candles: OHLCV candle data (Monthly or Weekly)
        lookback: Number of bars to check on each side
    
    Returns:
        Tuple of (pivot_highs, pivot_lows) as lists of (bar_index, price)
    """
    if not candles or len(candles) < lookback * 2 + 1:
        return [], []
    
    pivot_highs = []
    pivot_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i].get("high", 0)
        low = candles[i].get("low", float("inf"))
        
        is_pivot_high = True
        is_pivot_low = True
        
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if candles[j].get("high", 0) >= high:
                is_pivot_high = False
            if candles[j].get("low", float("inf")) <= low:
                is_pivot_low = False
        
        if is_pivot_high:
            pivot_highs.append((i, high))
        if is_pivot_low:
            pivot_lows.append((i, low))
    
    return pivot_highs, pivot_lows


def _build_htf_sr_levels(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    current_price: float,
    atr: float,
    pivot_lookback: int = 3,
    tolerance_atr: float = 0.5,
) -> List[HTFLevel]:
    """
    Build a list of HTF Support/Resistance levels from Monthly and Weekly data.
    
    Args:
        monthly_candles: Monthly OHLCV data
        weekly_candles: Weekly OHLCV data
        current_price: Current market price
        atr: ATR value for tolerance calculation
        pivot_lookback: Bars to check for pivot detection
        tolerance_atr: Tolerance in ATR multiples for level clustering
    
    Returns:
        List of HTFLevel objects sorted by distance from current price
    """
    levels = []
    tolerance = atr * tolerance_atr
    
    if monthly_candles and len(monthly_candles) >= pivot_lookback * 2 + 1:
        mn_highs, mn_lows = _detect_fractal_pivots_htf(monthly_candles, pivot_lookback)
        
        for idx, price in mn_highs[-10:]:
            levels.append(HTFLevel(
                price=price,
                level_type="resistance",
                timeframe="monthly",
                strength=3,
                last_touch_bar=idx,
            ))
        
        for idx, price in mn_lows[-10:]:
            levels.append(HTFLevel(
                price=price,
                level_type="support",
                timeframe="monthly",
                strength=3,
                last_touch_bar=idx,
            ))
    
    if weekly_candles and len(weekly_candles) >= pivot_lookback * 2 + 1:
        wk_highs, wk_lows = _detect_fractal_pivots_htf(weekly_candles, pivot_lookback)
        
        for idx, price in wk_highs[-15:]:
            levels.append(HTFLevel(
                price=price,
                level_type="resistance",
                timeframe="weekly",
                strength=2,
                last_touch_bar=idx,
            ))
        
        for idx, price in wk_lows[-15:]:
            levels.append(HTFLevel(
                price=price,
                level_type="support",
                timeframe="weekly",
                strength=2,
                last_touch_bar=idx,
            ))
    
    clustered_levels = []
    sorted_levels = sorted(levels, key=lambda x: x.price)
    
    for level in sorted_levels:
        merged = False
        for existing in clustered_levels:
            if abs(level.price - existing.price) < tolerance:
                if level.strength > existing.strength:
                    existing.price = (existing.price + level.price) / 2
                    existing.strength = max(existing.strength, level.strength)
                    existing.timeframe = level.timeframe if level.strength > existing.strength else existing.timeframe
                existing.touch_count += 1
                merged = True
                break
        if not merged:
            clustered_levels.append(level)
    
    clustered_levels.sort(key=lambda x: abs(x.price - current_price))
    return clustered_levels


def _detect_bos_into_level(
    candles: List[Dict],
    level: HTFLevel,
    current_idx: int,
    lookback: int = 10,
) -> Tuple[bool, str]:
    """
    Detect Break of Structure (BOS) into an S/R level.
    
    For resistance: Price makes higher highs approaching the level
    For support: Price makes lower lows approaching the level
    
    Args:
        candles: Daily OHLCV data
        level: The HTF S/R level to check
        current_idx: Current bar index
        lookback: Bars to check for structure break
    
    Returns:
        Tuple of (has_bos, bos_type)
    """
    if not candles or current_idx < lookback:
        return False, "insufficient_data"
    
    recent = candles[max(0, current_idx - lookback):current_idx + 1]
    if len(recent) < 5:
        return False, "insufficient_data"
    
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(recent) - 2):
        h = recent[i].get("high", 0)
        l = recent[i].get("low", float("inf"))
        
        is_high = all(recent[j].get("high", 0) < h for j in [i-2, i-1, i+1, i+2] if 0 <= j < len(recent))
        is_low = all(recent[j].get("low", float("inf")) > l for j in [i-2, i-1, i+1, i+2] if 0 <= j < len(recent))
        
        if is_high:
            swing_highs.append(h)
        if is_low:
            swing_lows.append(l)
    
    if level.level_type == "resistance":
        if len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]:
            return True, "higher_high_into_resistance"
    else:
        if len(swing_lows) >= 2 and swing_lows[-1] < swing_lows[-2]:
            return True, "lower_low_into_support"
    
    return False, "no_bos"


def _detect_liquidity_sweep(
    candles: List[Dict],
    level: HTFLevel,
    current_idx: int,
    atr: float,
    fib_range: float,
    direction: str,
) -> Tuple[bool, Optional[float]]:
    """
    Detect liquidity sweep past fib_786 level.
    
    For bullish reversal at support: Price wicks below 0.786 fib but closes above
    For bearish reversal at resistance: Price wicks above 0.786 fib but closes below
    
    Args:
        candles: Daily OHLCV data
        level: The HTF S/R level
        current_idx: Current bar index
        atr: ATR for range calculation
        fib_range: The swing range for Fibonacci calculation
        direction: Expected trade direction after reversal
    
    Returns:
        Tuple of (has_sweep, sweep_extreme_price)
    """
    if not candles or current_idx < 1 or current_idx >= len(candles):
        return False, None
    
    current = candles[current_idx]
    fib_786_offset = fib_range * 0.786
    
    if direction == "bullish":
        fib_786_level = level.price - fib_786_offset
        
        if current.get("low", float("inf")) < fib_786_level:
            if current.get("close", 0) > fib_786_level:
                return True, current.get("low")
        
        for i in range(max(0, current_idx - 3), current_idx):
            c = candles[i]
            if c.get("low", float("inf")) < fib_786_level:
                if current.get("close", 0) > fib_786_level:
                    return True, c.get("low")
    else:
        fib_786_level = level.price + fib_786_offset
        
        if current.get("high", 0) > fib_786_level:
            if current.get("close", float("inf")) < fib_786_level:
                return True, current.get("high")
        
        for i in range(max(0, current_idx - 3), current_idx):
            c = candles[i]
            if c.get("high", 0) > fib_786_level:
                if current.get("close", float("inf")) < fib_786_level:
                    return True, c.get("high")
    
    return False, None


def _check_fib_reclaim(
    candles: List[Dict],
    level: HTFLevel,
    current_idx: int,
    fib_range: float,
    direction: str,
) -> Tuple[bool, Optional[float]]:
    """
    Check if price has reclaimed through fib_618 level.
    
    For bullish: Close above the 0.618 retracement from level
    For bearish: Close below the 0.618 retracement from level
    
    Args:
        candles: Daily OHLCV data
        level: The HTF S/R level
        current_idx: Current bar index
        fib_range: The swing range for Fibonacci calculation
        direction: Expected trade direction
    
    Returns:
        Tuple of (has_reclaim, fib_618_level)
    """
    if not candles or current_idx >= len(candles):
        return False, None
    
    current = candles[current_idx]
    fib_618_offset = fib_range * 0.618
    
    if direction == "bullish":
        fib_618_level = level.price - fib_618_offset
        if current.get("close", 0) > fib_618_level:
            return True, fib_618_level
    else:
        fib_618_level = level.price + fib_618_offset
        if current.get("close", float("inf")) < fib_618_level:
            return True, fib_618_level
    
    return False, None


def _detect_4h_momentum_confirmation(
    h4_candles: List[Dict],
    direction: str,
    ema_short: int = 8,
    ema_long: int = 21,
) -> Tuple[bool, str]:
    """
    Detect 4H momentum confirmation via CHOCH or EMA slope.
    
    CHOCH (Change of Character): Swing structure shift
    EMA Slope: Short EMA crosses above/below long EMA
    
    Args:
        h4_candles: 4H OHLCV data
        direction: Expected trade direction
        ema_short: Short EMA period
        ema_long: Long EMA period
    
    Returns:
        Tuple of (has_confirmation, confirmation_type)
    """
    if not h4_candles or len(h4_candles) < ema_long + 5:
        return False, "insufficient_data"
    
    closes = [c.get("close", 0) for c in h4_candles if c.get("close")]
    if len(closes) < ema_long + 5:
        return False, "insufficient_closes"
    
    def calc_ema_series(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return []
        k = 2 / (period + 1)
        ema_values = [sum(values[:period]) / period]
        for price in values[period:]:
            ema_values.append(price * k + ema_values[-1] * (1 - k))
        return ema_values
    
    ema_s = calc_ema_series(closes, ema_short)
    ema_l = calc_ema_series(closes, ema_long)
    
    if len(ema_s) >= 2 and len(ema_l) >= 2:
        ema_l_aligned = ema_l[ema_short - 1:] if len(ema_l) > ema_short - 1 else ema_l
        
        if len(ema_s) >= 2 and len(ema_l_aligned) >= 2:
            min_len = min(len(ema_s), len(ema_l_aligned))
            
            if direction == "bullish":
                if ema_s[-1] > ema_l_aligned[-1] and ema_s[-2] <= ema_l_aligned[-2]:
                    return True, "ema_crossover_bullish"
                if ema_s[-1] > ema_l_aligned[-1]:
                    slope_s = (ema_s[-1] - ema_s[-3]) if len(ema_s) >= 3 else 0
                    if slope_s > 0:
                        return True, "ema_slope_bullish"
            else:
                if ema_s[-1] < ema_l_aligned[-1] and ema_s[-2] >= ema_l_aligned[-2]:
                    return True, "ema_crossover_bearish"
                if ema_s[-1] < ema_l_aligned[-1]:
                    slope_s = (ema_s[-1] - ema_s[-3]) if len(ema_s) >= 3 else 0
                    if slope_s < 0:
                        return True, "ema_slope_bearish"
    
    recent_h4 = h4_candles[-10:] if len(h4_candles) >= 10 else h4_candles
    
    swing_highs = []
    swing_lows = []
    
    for i in range(1, len(recent_h4) - 1):
        h = recent_h4[i].get("high", 0)
        l = recent_h4[i].get("low", float("inf"))
        
        if h > recent_h4[i-1].get("high", 0) and h > recent_h4[i+1].get("high", 0):
            swing_highs.append((i, h))
        if l < recent_h4[i-1].get("low", float("inf")) and l < recent_h4[i+1].get("low", float("inf")):
            swing_lows.append((i, l))
    
    if direction == "bullish":
        if len(swing_lows) >= 2:
            if swing_lows[-1][1] > swing_lows[-2][1]:
                return True, "choch_higher_low"
    else:
        if len(swing_highs) >= 2:
            if swing_highs[-1][1] < swing_highs[-2][1]:
                return True, "choch_lower_high"
    
    return False, "no_confirmation"


def _check_session_bias(
    candles: List[Dict],
    current_idx: int,
    direction: str,
) -> bool:
    """
    Check session bias alignment.
    
    Compares current session momentum to overall daily trend.
    
    Args:
        candles: Daily OHLCV data
        current_idx: Current bar index
        direction: Expected trade direction
    
    Returns:
        True if session bias aligns with direction
    """
    if not candles or current_idx < 5:
        return False
    
    recent = candles[max(0, current_idx - 5):current_idx + 1]
    if len(recent) < 3:
        return False
    
    opens = [c.get("open", 0) for c in recent]
    closes = [c.get("close", 0) for c in recent]
    
    bullish_days = sum(1 for i in range(len(recent)) if closes[i] > opens[i])
    bearish_days = len(recent) - bullish_days
    
    if direction == "bullish":
        return bullish_days >= len(recent) * 0.6
    else:
        return bearish_days >= len(recent) * 0.6


def _check_volume_confirmation(
    candles: List[Dict],
    current_idx: int,
    lookback: int = 10,
) -> bool:
    """
    Check volume confirmation (current volume above average).
    
    Args:
        candles: Daily OHLCV data with volume
        current_idx: Current bar index
        lookback: Period for average volume calculation
    
    Returns:
        True if volume is above average
    """
    if not candles or current_idx < lookback:
        return False
    
    volumes = []
    for i in range(max(0, current_idx - lookback), current_idx):
        vol = candles[i].get("volume", 0)
        if vol > 0:
            volumes.append(vol)
    
    if not volumes:
        return True
    
    avg_volume = sum(volumes) / len(volumes)
    current_volume = candles[current_idx].get("volume", 0)
    
    if current_volume == 0:
        return True
    
    return current_volume >= avg_volume * 0.8


def _detect_sr_break(
    candles: List[Dict],
    level: HTFLevel,
    current_idx: int,
    lookback: int = 5,
) -> Tuple[bool, str]:
    """
    Detect if an S/R level has been broken (for continuation mode).
    
    A level is broken when:
    - Price closes decisively through it
    - Subsequent bars stay on the breakout side
    
    Args:
        candles: Daily OHLCV data
        level: The HTF S/R level
        current_idx: Current bar index
        lookback: Bars to check for break confirmation
    
    Returns:
        Tuple of (is_broken, break_direction)
    """
    if not candles or current_idx < lookback + 2:
        return False, "insufficient_data"
    
    break_direction = None
    break_bar = None
    
    for i in range(max(0, current_idx - lookback * 2), current_idx - 1):
        prev_close = candles[i].get("close", 0) if i > 0 else 0
        curr_close = candles[i + 1].get("close", 0)
        
        if level.level_type == "resistance":
            if prev_close < level.price and curr_close > level.price:
                body = abs(curr_close - candles[i + 1].get("open", 0))
                range_size = candles[i + 1].get("high", 0) - candles[i + 1].get("low", 0)
                if range_size > 0 and body / range_size > 0.5:
                    break_direction = "bullish"
                    break_bar = i + 1
        else:
            if prev_close > level.price and curr_close < level.price:
                body = abs(curr_close - candles[i + 1].get("open", 0))
                range_size = candles[i + 1].get("high", 0) - candles[i + 1].get("low", 0)
                if range_size > 0 and body / range_size > 0.5:
                    break_direction = "bearish"
                    break_bar = i + 1
    
    if break_bar is not None and break_direction is not None:
        closes_after = [candles[j].get("close", 0) for j in range(break_bar + 1, current_idx + 1)]
        
        if break_direction == "bullish":
            if all(c > level.price * 0.998 for c in closes_after):
                return True, break_direction
        else:
            if all(c < level.price * 1.002 for c in closes_after):
                return True, break_direction
    
    return False, "no_break"


def _check_retest_pullback(
    candles: List[Dict],
    level: HTFLevel,
    current_idx: int,
    atr: float,
    direction: str,
) -> Tuple[bool, str]:
    """
    Check for retest/pullback to breached S/R level.
    
    After a break, look for price to pull back and touch the level
    before continuing in the break direction.
    
    Args:
        candles: Daily OHLCV data
        level: The breached HTF S/R level
        current_idx: Current bar index
        atr: ATR for tolerance
        direction: Break direction (continuation direction)
    
    Returns:
        Tuple of (is_retest, retest_type)
    """
    if not candles or current_idx < 3:
        return False, "insufficient_data"
    
    current = candles[current_idx]
    tolerance = atr * 0.5
    
    if direction == "bullish":
        low = current.get("low", float("inf"))
        close = current.get("close", 0)
        
        if abs(low - level.price) <= tolerance:
            if close > level.price:
                return True, "bullish_retest"
        
        for i in range(max(0, current_idx - 3), current_idx):
            c = candles[i]
            if abs(c.get("low", float("inf")) - level.price) <= tolerance:
                if close > level.price:
                    return True, "bullish_retest_prior"
    else:
        high = current.get("high", 0)
        close = current.get("close", float("inf"))
        
        if abs(high - level.price) <= tolerance:
            if close < level.price:
                return True, "bearish_retest"
        
        for i in range(max(0, current_idx - 3), current_idx):
            c = candles[i]
            if abs(c.get("high", 0) - level.price) <= tolerance:
                if close < level.price:
                    return True, "bearish_retest_prior"
    
    return False, "no_retest"


def _find_daily_supply_demand_zones(
    candles: List[Dict],
    current_idx: int,
    lookback: int = 50,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Find Daily supply and demand zones.
    
    Demand zones: Strong bullish moves originating from a base
    Supply zones: Strong bearish moves originating from a base
    
    Args:
        candles: Daily OHLCV data
        current_idx: Current bar index
        lookback: Bars to look back for zone detection
    
    Returns:
        Tuple of (demand_zones, supply_zones) as lists of (zone_low, zone_high)
    """
    demand_zones = []
    supply_zones = []
    
    if not candles or current_idx < lookback:
        return demand_zones, supply_zones
    
    recent = candles[max(0, current_idx - lookback):current_idx + 1]
    
    if len(recent) < 10:
        return demand_zones, supply_zones
    
    avg_range = sum(c.get("high", 0) - c.get("low", 0) for c in recent) / len(recent)
    
    for i in range(3, len(recent) - 3):
        c = recent[i]
        body = abs(c.get("close", 0) - c.get("open", 0))
        range_size = c.get("high", 0) - c.get("low", 0)
        
        if body > avg_range * 1.5:
            prior_range = sum(
                recent[j].get("high", 0) - recent[j].get("low", 0) 
                for j in range(max(0, i-3), i)
            ) / 3 if i >= 3 else avg_range
            
            is_bullish = c.get("close", 0) > c.get("open", 0)
            
            if prior_range < avg_range * 0.7:
                if is_bullish:
                    zone_low = min(recent[j].get("low", float("inf")) for j in range(max(0, i-3), i+1))
                    zone_high = max(recent[j].get("open", 0) for j in range(max(0, i-3), i))
                    if zone_high > zone_low:
                        demand_zones.append((zone_low, zone_high))
                else:
                    zone_high = max(recent[j].get("high", 0) for j in range(max(0, i-3), i+1))
                    zone_low = min(recent[j].get("open", float("inf")) for j in range(max(0, i-3), i))
                    if zone_high > zone_low:
                        supply_zones.append((zone_low, zone_high))
    
    return demand_zones[-5:], supply_zones[-5:]


def _check_confirmation_candle(
    candles: List[Dict],
    current_idx: int,
    direction: str,
    atr: float,
    min_body_ratio: float = 0.5,
) -> Tuple[bool, str]:
    """
    Check for strong confirmation candle.
    
    A confirmation candle has:
    - Body >= 0.5 * ATR
    - Direction matches expected trade direction
    
    Args:
        candles: Daily OHLCV data
        current_idx: Current bar index
        direction: Expected trade direction
        atr: ATR value
        min_body_ratio: Minimum body to ATR ratio
    
    Returns:
        Tuple of (is_confirmation, candle_type)
    """
    if not candles or current_idx >= len(candles) or atr <= 0:
        return False, "invalid_data"
    
    current = candles[current_idx]
    open_price = current.get("open", 0)
    close = current.get("close", 0)
    body = abs(close - open_price)
    
    if body < atr * min_body_ratio:
        return False, "weak_body"
    
    if direction == "bullish":
        if close > open_price:
            return True, "bullish_confirmation"
        return False, "wrong_direction"
    else:
        if close < open_price:
            return True, "bearish_confirmation"
        return False, "wrong_direction"


def _calculate_htf_trade_levels(
    entry: float,
    direction: str,
    sweep_extreme: Optional[float],
    level_price: float,
    atr: float,
    params: HTFConfluenceParams,
) -> Tuple[float, float, float, float]:
    """
    Calculate SL and TP levels for HTF confluence trades.
    
    SL: Beyond liquidity sweep + max(0.33*Risk, 0.4*ATR)
    TP1: 1.5R
    TP2: 2.5R
    TP3: Extension (4R)
    
    Args:
        entry: Entry price
        direction: Trade direction
        sweep_extreme: Liquidity sweep extreme price
        level_price: The S/R level price
        atr: ATR value
        params: HTF confluence parameters
    
    Returns:
        Tuple of (sl, tp1, tp2, tp3)
    """
    if direction == "bullish":
        if sweep_extreme is not None:
            base_sl = sweep_extreme
        else:
            base_sl = level_price - atr * 0.5
        
        initial_risk = entry - base_sl
        buffer = max(initial_risk * params.sl_buffer_risk_ratio, atr * params.sl_buffer_atr_ratio)
        sl = base_sl - buffer
        
        risk = entry - sl
        if risk <= 0:
            risk = atr
            sl = entry - risk
        
        tp1 = entry + risk * params.tp1_rr
        tp2 = entry + risk * params.tp2_rr
        tp3 = entry + risk * params.tp3_rr
    else:
        if sweep_extreme is not None:
            base_sl = sweep_extreme
        else:
            base_sl = level_price + atr * 0.5
        
        initial_risk = base_sl - entry
        buffer = max(initial_risk * params.sl_buffer_risk_ratio, atr * params.sl_buffer_atr_ratio)
        sl = base_sl + buffer
        
        risk = sl - entry
        if risk <= 0:
            risk = atr
            sl = entry + risk
        
        tp1 = entry - risk * params.tp1_rr
        tp2 = entry - risk * params.tp2_rr
        tp3 = entry - risk * params.tp3_rr
    
    return sl, tp1, tp2, tp3


def _calculate_confluence_score(
    htf_level_touch: bool,
    daily_zone_alignment: bool,
    h4_momentum: bool,
    liquidity_sweep: bool,
    session_bias: bool,
    volume_confirmation: bool,
) -> int:
    """
    Calculate confluence score on 0-10 scale.
    
    Scoring:
    - +2: Monthly/Weekly level touch
    - +2: Daily supply/demand zone alignment
    - +2: 4H momentum confirmation (CHOCH or EMA slope)
    - +2: Liquidity sweep occurred
    - +1: Session bias alignment
    - +1: Volume confirmation
    
    Args:
        htf_level_touch: Price is at Monthly/Weekly S/R level
        daily_zone_alignment: Price aligns with Daily supply/demand zone
        h4_momentum: 4H momentum confirmation present
        liquidity_sweep: Liquidity sweep detected
        session_bias: Session bias aligns with direction
        volume_confirmation: Volume above average
    
    Returns:
        Confluence score (0-10)
    """
    score = 0
    
    if htf_level_touch:
        score += 2
    if daily_zone_alignment:
        score += 2
    if h4_momentum:
        score += 2
    if liquidity_sweep:
        score += 2
    if session_bias:
        score += 1
    if volume_confirmation:
        score += 1
    
    return min(score, 10)


def generate_htf_confluence_signals(
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    weekly_candles: List[Dict],
    monthly_candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[HTFConfluenceParams] = None,
) -> List[Signal]:
    """
    Generate trading signals using comprehensive two-track HTF confluence strategy.
    
    TRACK 1 - REVERSAL MODE:
    - Detect Monthly and Weekly S/R levels using fractal pivots
    - Entry when price rejects at these levels with:
      - Break of Structure (BOS) into the level
      - Liquidity sweep past fib_786
      - Reclaim close through fib_618
      - Confluence score >= 4
    
    TRACK 2 - CONTINUATION MODE:
    - When price BREAKS through Monthly/Weekly S/R level:
      - The breached level becomes support (for bullish) or resistance (for bearish)
      - Wait for retest/pullback to the breached level
      - Entry on reclaim with momentum confirmation
    - Also use Daily supply/demand zones for continuation when price is trending
    
    Confluence Scoring (0-10):
    - +2: Monthly/Weekly level touch
    - +2: Daily supply/demand zone alignment
    - +2: 4H momentum confirmation (CHOCH or EMA slope)
    - +2: Liquidity sweep occurred
    - +1: Session bias alignment
    - +1: Volume confirmation
    
    Args:
        daily_candles: Daily OHLCV data (oldest to newest)
        h4_candles: 4H OHLCV data
        weekly_candles: Weekly OHLCV data
        monthly_candles: Monthly OHLCV data
        symbol: Asset symbol
        params: Strategy parameters
    
    Returns:
        List of Signal objects with proper fields for both tracks
    """
    if params is None:
        params = HTFConfluenceParams()
    
    if not daily_candles or len(daily_candles) < 60:
        return []
    
    signals: List[Signal] = []
    used_bars: set = set()
    cooldown_until = -1
    
    for i in range(50, len(daily_candles)):
        if i in used_bars:
            continue
        
        if i <= cooldown_until:
            continue
        
        try:
            daily_slice = daily_candles[:i + 1]
            current = daily_candles[i]
            current_close = current.get("close", 0)
            current_high = current.get("high", 0)
            current_low = current.get("low", float("inf"))
            
            daily_atr = _atr(daily_slice, 14)
            if daily_atr <= 0:
                continue
            
            weekly_slice = weekly_candles[:max(1, i // 5)] if weekly_candles else []
            monthly_slice = monthly_candles[:max(1, i // 20)] if monthly_candles else []
            
            h4_idx = min(len(h4_candles), (i + 1) * 6) if h4_candles else 0
            h4_slice = h4_candles[:h4_idx] if h4_candles else []
            
            sr_levels = _build_htf_sr_levels(
                monthly_slice,
                weekly_slice,
                current_close,
                daily_atr,
                params.pivot_lookback,
                params.sr_tolerance_atr,
            )
            
            demand_zones, supply_zones = _find_daily_supply_demand_zones(daily_slice, i)
            
            signal_found = False
            
            for level in sr_levels[:10]:
                if signal_found:
                    break
                
                distance = abs(current_close - level.price)
                if distance > daily_atr * 2.0:
                    continue
                
                is_broken, break_direction = _detect_sr_break(daily_slice, level, i)
                
                if is_broken:
                    is_retest, retest_type = _check_retest_pullback(
                        daily_slice, level, i, daily_atr, break_direction
                    )
                    
                    if is_retest:
                        direction = break_direction
                        
                        h4_confirmed, h4_type = _detect_4h_momentum_confirmation(
                            h4_slice, direction, params.ema_short, params.ema_long
                        )
                        
                        is_confirm_candle, confirm_type = _check_confirmation_candle(
                            daily_slice, i, direction, daily_atr, params.body_min_atr_ratio
                        )
                        
                        session_aligned = _check_session_bias(daily_slice, i, direction)
                        volume_ok = _check_volume_confirmation(daily_slice, i)
                        
                        daily_zone_aligned = False
                        if direction == "bullish":
                            for zone_low, zone_high in demand_zones:
                                if zone_low <= current_low <= zone_high:
                                    daily_zone_aligned = True
                                    break
                        else:
                            for zone_low, zone_high in supply_zones:
                                if zone_low <= current_high <= zone_high:
                                    daily_zone_aligned = True
                                    break
                        
                        confluence_score = _calculate_confluence_score(
                            htf_level_touch=True,
                            daily_zone_alignment=daily_zone_aligned,
                            h4_momentum=h4_confirmed,
                            liquidity_sweep=False,
                            session_bias=session_aligned,
                            volume_confirmation=volume_ok,
                        )
                        
                        if confluence_score >= params.min_continuation_confluence and is_confirm_candle:
                            entry = current_close
                            
                            sl, tp1, tp2, tp3 = _calculate_htf_trade_levels(
                                entry, direction, None, level.price, daily_atr, params
                            )
                            
                            risk = abs(entry - sl)
                            reward = abs(tp1 - entry)
                            rr = reward / risk if risk > 0 else 0
                            
                            flags = {
                                "htf_level": True,
                                "continuation": True,
                                "reversal": False,
                                "h4_momentum": h4_confirmed,
                                "daily_zone": daily_zone_aligned,
                                "session_bias": session_aligned,
                                "volume": volume_ok,
                                "confirmation": is_confirm_candle,
                                "rr": rr >= 1.5,
                            }
                            
                            notes = {
                                "track": "CONTINUATION",
                                "htf_level": f"{level.timeframe.upper()} {level.level_type} at {level.price:.5f}",
                                "retest": retest_type,
                                "h4_momentum": h4_type,
                                "confirmation": confirm_type,
                                "rr": f"R:R = {rr:.2f}",
                            }
                            
                            timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                            
                            signal = Signal(
                                symbol=symbol,
                                direction=direction,
                                bar_index=i,
                                timestamp=timestamp,
                                confluence_score=confluence_score,
                                quality_factors=sum(1 for v in flags.values() if v),
                                entry=entry,
                                stop_loss=sl,
                                tp1=tp1,
                                tp2=tp2,
                                tp3=tp3,
                                is_active=True,
                                is_watching=False,
                                flags=flags,
                                notes=notes,
                            )
                            signals.append(signal)
                            signal_found = True
                            cooldown_until = i + params.cooldown_bars
                
                else:
                    if level.level_type == "support" and distance <= daily_atr * 1.0:
                        direction = "bullish"
                    elif level.level_type == "resistance" and distance <= daily_atr * 1.0:
                        direction = "bearish"
                    else:
                        continue
                    
                    has_bos, bos_type = _detect_bos_into_level(daily_slice, level, i)
                    
                    fib_range = daily_atr * 3
                    has_sweep, sweep_extreme = _detect_liquidity_sweep(
                        daily_slice, level, i, daily_atr, fib_range, direction
                    )
                    
                    has_reclaim, fib_618 = _check_fib_reclaim(
                        daily_slice, level, i, fib_range, direction
                    )
                    
                    h4_confirmed, h4_type = _detect_4h_momentum_confirmation(
                        h4_slice, direction, params.ema_short, params.ema_long
                    )
                    
                    is_confirm_candle, confirm_type = _check_confirmation_candle(
                        daily_slice, i, direction, daily_atr, params.body_min_atr_ratio
                    )
                    
                    session_aligned = _check_session_bias(daily_slice, i, direction)
                    volume_ok = _check_volume_confirmation(daily_slice, i)
                    
                    daily_zone_aligned = False
                    if direction == "bullish":
                        for zone_low, zone_high in demand_zones:
                            if zone_low <= current_low <= zone_high:
                                daily_zone_aligned = True
                                break
                    else:
                        for zone_low, zone_high in supply_zones:
                            if zone_low <= current_high <= zone_high:
                                daily_zone_aligned = True
                                break
                    
                    confluence_score = _calculate_confluence_score(
                        htf_level_touch=True,
                        daily_zone_alignment=daily_zone_aligned,
                        h4_momentum=h4_confirmed,
                        liquidity_sweep=has_sweep,
                        session_bias=session_aligned,
                        volume_confirmation=volume_ok,
                    )
                    
                    if (confluence_score >= params.min_reversal_confluence and 
                        has_reclaim and is_confirm_candle):
                        
                        entry = current_close
                        
                        sl, tp1, tp2, tp3 = _calculate_htf_trade_levels(
                            entry, direction, sweep_extreme, level.price, daily_atr, params
                        )
                        
                        risk = abs(entry - sl)
                        reward = abs(tp1 - entry)
                        rr = reward / risk if risk > 0 else 0
                        
                        flags = {
                            "htf_level": True,
                            "reversal": True,
                            "continuation": False,
                            "bos": has_bos,
                            "liquidity_sweep": has_sweep,
                            "fib_reclaim": has_reclaim,
                            "h4_momentum": h4_confirmed,
                            "daily_zone": daily_zone_aligned,
                            "session_bias": session_aligned,
                            "volume": volume_ok,
                            "confirmation": is_confirm_candle,
                            "rr": rr >= 1.5,
                        }
                        
                        notes = {
                            "track": "REVERSAL",
                            "htf_level": f"{level.timeframe.upper()} {level.level_type} at {level.price:.5f}",
                            "bos": bos_type,
                            "sweep": f"Sweep to {sweep_extreme:.5f}" if sweep_extreme else "No sweep",
                            "fib_reclaim": f"Reclaimed 0.618 at {fib_618:.5f}" if fib_618 else "No reclaim",
                            "h4_momentum": h4_type,
                            "confirmation": confirm_type,
                            "rr": f"R:R = {rr:.2f}",
                        }
                        
                        timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                        
                        signal = Signal(
                            symbol=symbol,
                            direction=direction,
                            bar_index=i,
                            timestamp=timestamp,
                            confluence_score=confluence_score,
                            quality_factors=sum(1 for v in flags.values() if v),
                            entry=entry,
                            stop_loss=sl,
                            tp1=tp1,
                            tp2=tp2,
                            tp3=tp3,
                            is_active=True,
                            is_watching=False,
                            flags=flags,
                            notes=notes,
                        )
                        signals.append(signal)
                        signal_found = True
                        cooldown_until = i + params.cooldown_bars
            
            if not signal_found and (demand_zones or supply_zones):
                trend = _infer_trend(daily_slice[-20:])
                
                if trend == "bullish":
                    for zone_low, zone_high in demand_zones:
                        if zone_low <= current_low <= zone_high * 1.005:
                            direction = "bullish"
                            
                            h4_confirmed, h4_type = _detect_4h_momentum_confirmation(
                                h4_slice, direction, params.ema_short, params.ema_long
                            )
                            
                            is_confirm_candle, confirm_type = _check_confirmation_candle(
                                daily_slice, i, direction, daily_atr, params.body_min_atr_ratio
                            )
                            
                            session_aligned = _check_session_bias(daily_slice, i, direction)
                            volume_ok = _check_volume_confirmation(daily_slice, i)
                            
                            confluence_score = _calculate_confluence_score(
                                htf_level_touch=False,
                                daily_zone_alignment=True,
                                h4_momentum=h4_confirmed,
                                liquidity_sweep=False,
                                session_bias=session_aligned,
                                volume_confirmation=volume_ok,
                            )
                            
                            if confluence_score >= params.min_continuation_confluence and is_confirm_candle:
                                entry = current_close
                                sl = zone_low - daily_atr * params.sl_buffer_atr_ratio
                                risk = entry - sl
                                
                                tp1 = entry + risk * params.tp1_rr
                                tp2 = entry + risk * params.tp2_rr
                                tp3 = entry + risk * params.tp3_rr
                                
                                rr = (tp1 - entry) / risk if risk > 0 else 0
                                
                                flags = {
                                    "htf_level": False,
                                    "daily_zone": True,
                                    "continuation": True,
                                    "reversal": False,
                                    "h4_momentum": h4_confirmed,
                                    "session_bias": session_aligned,
                                    "volume": volume_ok,
                                    "confirmation": is_confirm_candle,
                                    "rr": rr >= 1.5,
                                }
                                
                                notes = {
                                    "track": "CONTINUATION_DEMAND",
                                    "zone": f"Demand zone: {zone_low:.5f} - {zone_high:.5f}",
                                    "h4_momentum": h4_type,
                                    "confirmation": confirm_type,
                                    "rr": f"R:R = {rr:.2f}",
                                }
                                
                                timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                                
                                signal = Signal(
                                    symbol=symbol,
                                    direction=direction,
                                    bar_index=i,
                                    timestamp=timestamp,
                                    confluence_score=confluence_score,
                                    quality_factors=sum(1 for v in flags.values() if v),
                                    entry=entry,
                                    stop_loss=sl,
                                    tp1=tp1,
                                    tp2=tp2,
                                    tp3=tp3,
                                    is_active=True,
                                    is_watching=False,
                                    flags=flags,
                                    notes=notes,
                                )
                                signals.append(signal)
                                cooldown_until = i + params.cooldown_bars
                                break
                
                elif trend == "bearish":
                    for zone_low, zone_high in supply_zones:
                        if zone_low * 0.995 <= current_high <= zone_high:
                            direction = "bearish"
                            
                            h4_confirmed, h4_type = _detect_4h_momentum_confirmation(
                                h4_slice, direction, params.ema_short, params.ema_long
                            )
                            
                            is_confirm_candle, confirm_type = _check_confirmation_candle(
                                daily_slice, i, direction, daily_atr, params.body_min_atr_ratio
                            )
                            
                            session_aligned = _check_session_bias(daily_slice, i, direction)
                            volume_ok = _check_volume_confirmation(daily_slice, i)
                            
                            confluence_score = _calculate_confluence_score(
                                htf_level_touch=False,
                                daily_zone_alignment=True,
                                h4_momentum=h4_confirmed,
                                liquidity_sweep=False,
                                session_bias=session_aligned,
                                volume_confirmation=volume_ok,
                            )
                            
                            if confluence_score >= params.min_continuation_confluence and is_confirm_candle:
                                entry = current_close
                                sl = zone_high + daily_atr * params.sl_buffer_atr_ratio
                                risk = sl - entry
                                
                                tp1 = entry - risk * params.tp1_rr
                                tp2 = entry - risk * params.tp2_rr
                                tp3 = entry - risk * params.tp3_rr
                                
                                rr = (entry - tp1) / risk if risk > 0 else 0
                                
                                flags = {
                                    "htf_level": False,
                                    "daily_zone": True,
                                    "continuation": True,
                                    "reversal": False,
                                    "h4_momentum": h4_confirmed,
                                    "session_bias": session_aligned,
                                    "volume": volume_ok,
                                    "confirmation": is_confirm_candle,
                                    "rr": rr >= 1.5,
                                }
                                
                                notes = {
                                    "track": "CONTINUATION_SUPPLY",
                                    "zone": f"Supply zone: {zone_low:.5f} - {zone_high:.5f}",
                                    "h4_momentum": h4_type,
                                    "confirmation": confirm_type,
                                    "rr": f"R:R = {rr:.2f}",
                                }
                                
                                timestamp = current.get("time") or current.get("timestamp") or current.get("date")
                                
                                signal = Signal(
                                    symbol=symbol,
                                    direction=direction,
                                    bar_index=i,
                                    timestamp=timestamp,
                                    confluence_score=confluence_score,
                                    quality_factors=sum(1 for v in flags.values() if v),
                                    entry=entry,
                                    stop_loss=sl,
                                    tp1=tp1,
                                    tp2=tp2,
                                    tp3=tp3,
                                    is_active=True,
                                    is_watching=False,
                                    flags=flags,
                                    notes=notes,
                                )
                                signals.append(signal)
                                cooldown_until = i + params.cooldown_bars
                                break
        
        except Exception:
            continue
    
    return signals
