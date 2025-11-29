"""
Strategy module - Uses the 5%ers Challenge-validated strategy
Bollinger Band + RSI(2) Mean Reversion with 4R targets
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Dict

from data import get_ohlcv
from config import (
    FOREX_PAIRS,
    METALS,
    INDICES,
    ENERGIES,
    CRYPTO_ASSETS,
)

from strategy_highwin import (
    generate_bb_signal,
    calculate_atr,
    ASSET_PARAMS,
    STRATEGY_CONFIG,
)

MAX_SIGNAL_AGE_DAYS = 5


@dataclass
class ScanResult:
    symbol: str
    direction: str
    confluence_score: int
    htf_bias: str
    location_note: str
    fib_note: str
    liquidity_note: str
    structure_note: str
    confirmation_note: str
    rr_note: str
    summary_reason: str
    status: str = "scan_only"
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tp4: Optional[float] = None
    tp5: Optional[float] = None
    setup_type: str = ""
    what_to_look_for: str = ""


def _signal_to_scan_result(signal: Dict, symbol: str) -> ScanResult:
    """Convert a strategy_highwin signal dict to ScanResult for Discord."""
    direction = signal['direction'].upper()
    
    rsi_val = signal.get('rsi', 50)
    adx_val = signal.get('adx', 20)
    signal_type = signal.get('signal_type', 'bb_reversal')
    
    if direction == "LONG":
        dir_text = "BULLISH"
        what_to_look = "Long entry - BB bounce from lower band with RSI oversold"
    else:
        dir_text = "BEARISH"
        what_to_look = "Short entry - BB rejection from upper band with RSI overbought"
    
    summary = (
        f"{dir_text} | BB+RSI Mean Reversion | "
        f"RSI(2)={rsi_val:.1f}, ADX={adx_val:.1f} | "
        f"4R Target"
    )
    
    return ScanResult(
        symbol=symbol,
        direction=signal['direction'],
        confluence_score=5,
        htf_bias=f"ADX={adx_val:.1f} (range mode)" if adx_val < 35 else f"ADX={adx_val:.1f} (trending)",
        location_note=f"Bollinger Band {signal_type}",
        fib_note="",
        liquidity_note="",
        structure_note=f"RSI(2) = {rsi_val:.1f}",
        confirmation_note="BB + RSI confirmed",
        rr_note="4:1 R:R target",
        summary_reason=summary,
        status="active",
        entry=signal['entry'],
        stop_loss=signal['sl'],
        tp1=signal['tp1'],
        tp2=signal['tp2'],
        tp3=signal['tp3'],
        setup_type="BB Mean Reversion",
        what_to_look_for=what_to_look,
    )


def scan_single_asset(symbol: str) -> Optional[ScanResult]:
    """Scan a single asset using the challenge-validated BB+RSI strategy."""
    try:
        candles = get_ohlcv(symbol, "H4", count=100)
        if not candles or len(candles) < 30:
            return None
        
        signal = generate_bb_signal(candles, symbol)
        
        if signal:
            return _signal_to_scan_result(signal, symbol)
        
        return None
        
    except Exception as e:
        print(f"[scan] Error scanning {symbol}: {e}")
        return None


def _scan_list(symbols: List[str], category: str) -> List[ScanResult]:
    """Scan a list of symbols."""
    results = []
    print(f"[scan] Scanning {category}...")
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] Scanning {symbol}...")
        result = scan_single_asset(symbol)
        if result:
            results.append(result)
    print(f"[scan] {category} done: {len(symbols)} scanned, {len(results)} signals")
    return results


def scan_forex() -> List[ScanResult]:
    """Scan all forex pairs."""
    return _scan_list(FOREX_PAIRS, "Forex")


def scan_crypto() -> List[ScanResult]:
    """Scan all crypto assets."""
    return _scan_list(CRYPTO_ASSETS, "Crypto")


def scan_metals() -> List[ScanResult]:
    """Scan all metals."""
    return _scan_list(METALS, "Metals")


def scan_indices() -> List[ScanResult]:
    """Scan all indices."""
    return _scan_list(INDICES, "Indices")


def scan_energies() -> List[ScanResult]:
    """Scan all energies."""
    return _scan_list(ENERGIES, "Energies")


def scan_all_markets() -> Dict[str, List[ScanResult]]:
    """Scan all markets and return results by category."""
    return {
        "forex": scan_forex(),
        "metals": scan_metals(),
        "indices": scan_indices(),
        "energies": scan_energies(),
        "crypto": scan_crypto(),
    }
