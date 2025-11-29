"""
Strategy wrapper for Discord bot - uses V3 strategy.

This file provides the interface between the Discord bot and strategy_v3.py.
All scanning and signal generation goes through strategy_v3.

V3 Strategy: HTF S/R + Break of Structure + Structural TPs
NO RSI, NO SMC, NO Fibonacci TPs
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

from data import get_ohlcv
from config import (
    FOREX_PAIRS, METALS, INDICES, ENERGIES, CRYPTO_ASSETS
)
from strategy_v3 import generate_signal, TradeSignal


@dataclass 
class ScanResult:
    """Result from scanning an instrument."""
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    r_multiple: float
    status: str
    confluence_score: int
    reasoning: str
    timestamp: datetime
    htf_zone_low: float = 0.0
    htf_zone_high: float = 0.0
    bos_level: float = 0.0
    
    @property
    def tp1(self) -> float:
        return self.take_profit
    
    @property
    def htf_bias(self) -> str:
        return self.direction.upper()
    
    @property
    def summary_reason(self) -> str:
        return self.reasoning


def convert_signal_to_scan_result(signal: TradeSignal) -> ScanResult:
    """Convert a TradeSignal to ScanResult for Discord display."""
    return ScanResult(
        symbol=signal.symbol,
        direction=signal.direction,
        entry=signal.entry,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        r_multiple=signal.r_multiple,
        status=signal.status,
        confluence_score=signal.confluence_score,
        reasoning=signal.reasoning,
        timestamp=signal.timestamp,
        htf_zone_low=signal.htf_zone[0] if signal.htf_zone else 0.0,
        htf_zone_high=signal.htf_zone[1] if signal.htf_zone else 0.0,
        bos_level=signal.bos_level
    )


def scan_single_asset(symbol: str) -> Optional[ScanResult]:
    """
    Scan a single instrument using V3 strategy.
    Fetches H4, Daily, and Weekly data from OANDA.
    """
    try:
        h4_candles = get_ohlcv(symbol, timeframe="H4", count=200)
        daily_candles = get_ohlcv(symbol, timeframe="D", count=200)
        weekly_candles = get_ohlcv(symbol, timeframe="W", count=50)
        
        if len(h4_candles) < 50 or len(daily_candles) < 50 or len(weekly_candles) < 10:
            return None
        
        signal = generate_signal(
            symbol=symbol,
            h4_candles=h4_candles,
            daily_candles=daily_candles,
            weekly_candles=weekly_candles,
            min_rr=2.0,
            min_confluence=2
        )
        
        if signal:
            return convert_signal_to_scan_result(signal)
        
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
        "Forex": scan_forex(),
        "Metals": scan_metals(),
        "Indices": scan_indices(),
        "Energies": scan_energies(),
        "Crypto": scan_crypto(),
    }
