"""
Strategy wrapper for Discord bot - uses V3 Pro strategy.

This file provides the interface between the Discord bot and strategy_v3_pro.py.
All scanning and signal generation goes through strategy_v3_pro.

V3 Pro Strategy: 
- Daily S/D Zones + Golden Pocket (0.618-0.65) entries
- Wyckoff Spring/Upthrust patterns
- EMA crossover trend confirmation
- Structural TPs (2-8 day hold time)
- NO RSI, NO MACD, NO SMC, NO Fibonacci for TPs
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

from data import get_ohlcv
from config import (
    FOREX_PAIRS, METALS, INDICES, ENERGIES, CRYPTO_ASSETS
)
from strategy_v3_pro import generate_v3_pro_signal, TradeSignal


OPTIMAL_ASSET_CONFIGS = {
    'EUR_USD': {'conf': 2, 'rr': 2.0},
    'GBP_USD': {'conf': 3, 'rr': 3.0},
    'USD_JPY': {'conf': 3, 'rr': 3.0},
    'USD_CHF': {'conf': 2, 'rr': 1.5},
    'USD_CAD': {'conf': 3, 'rr': 3.0},
    'AUD_USD': {'conf': 2, 'rr': 1.5},
    'NZD_USD': {'conf': 3, 'rr': 3.0},
    'EUR_GBP': {'conf': 3, 'rr': 2.5},
    'EUR_JPY': {'conf': 2, 'rr': 1.5},
    'GBP_JPY': {'conf': 3, 'rr': 3.0},
    'XAU_USD': {'conf': 2, 'rr': 1.5},
    'XAG_USD': {'conf': 3, 'rr': 2.5},
    'WTICO_USD': {'conf': 3, 'rr': 3.0},
    'BCO_USD': {'conf': 2, 'rr': 1.5},
    'NAS100_USD': {'conf': 2, 'rr': 2.0},
    'SPX500_USD': {'conf': 3, 'rr': 2.5},
    'BTC_USD': {'conf': 3, 'rr': 3.0},
    'ETH_USD': {'conf': 3, 'rr': 2.5},
}


@dataclass 
class ScanResult:
    """Result from scanning an instrument (V3 Pro format)."""
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
    zone_low: float = 0.0
    zone_high: float = 0.0
    entry_type: str = "zone_entry"
    
    @property
    def tp1(self) -> float:
        return self.take_profit
    
    @property
    def htf_bias(self) -> str:
        return self.direction.upper()
    
    @property
    def summary_reason(self) -> str:
        return self.reasoning
    
    @property
    def htf_zone_low(self) -> float:
        return self.zone_low
    
    @property
    def htf_zone_high(self) -> float:
        return self.zone_high
    
    @property
    def bos_level(self) -> float:
        return self.entry


def convert_signal_to_scan_result(signal: TradeSignal) -> ScanResult:
    """Convert a V3 Pro TradeSignal to ScanResult for Discord display."""
    ts = signal.timestamp
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except:
            ts = datetime.now()
    
    return ScanResult(
        symbol=signal.symbol,
        direction=signal.direction,
        entry=signal.entry,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        r_multiple=signal.r_multiple,
        status=signal.status,
        confluence_score=signal.confluence,
        reasoning=signal.reasoning,
        timestamp=ts,
        zone_low=signal.zone.low if signal.zone else 0.0,
        zone_high=signal.zone.high if signal.zone else 0.0,
        entry_type=signal.entry_type or "zone_entry"
    )


def scan_single_asset(symbol: str) -> Optional[ScanResult]:
    """
    Scan a single instrument using V3 Pro strategy.
    Fetches Daily and Weekly data from OANDA.
    Uses asset-specific optimal configurations.
    """
    try:
        daily_candles = get_ohlcv(symbol, timeframe="D", count=200)
        weekly_candles = get_ohlcv(symbol, timeframe="W", count=50)
        
        if len(daily_candles) < 60 or len(weekly_candles) < 12:
            return None
        
        cfg = OPTIMAL_ASSET_CONFIGS.get(symbol, {'conf': 3, 'rr': 2.5})
        
        signal = generate_v3_pro_signal(
            symbol=symbol,
            daily_candles=daily_candles,
            weekly_candles=weekly_candles,
            min_rr=cfg['rr'],
            min_confluence=cfg['conf']
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
