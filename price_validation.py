"""
Price Validation Module for Blueprint Trader AI.

This module validates that OANDA price data matches reference prices
and ensures trade entries/exits are consistent with real candle data.
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
import csv
import os


def load_reference_prices(filepath: str = "eurusd_reference_prices.csv") -> Dict[date, float]:
    """
    Load reference prices from CSV file.
    
    Args:
        filepath: Path to the reference prices CSV
        
    Returns:
        Dictionary mapping dates to reference close prices
    """
    reference = {}
    
    if not os.path.exists(filepath):
        print(f"[price_validation] Reference file not found: {filepath}")
        return reference
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row.get('Date', '').strip()
                price_str = row.get('Price', '').strip()
                
                if not date_str or not price_str:
                    continue
                
                try:
                    d = datetime.strptime(date_str, '%Y-%m-%d').date()
                    price = float(price_str)
                    reference[d] = price
                except (ValueError, TypeError) as e:
                    print(f"[price_validation] Error parsing row: {row}, {e}")
                    continue
    except Exception as e:
        print(f"[price_validation] Error loading reference prices: {e}")
    
    return reference


def validate_price_against_reference(
    check_date: date,
    oanda_close: float,
    instrument: str = "EUR_USD",
    tolerance: float = 0.005,
) -> Tuple[bool, str]:
    """
    Validate an OANDA close price against reference data.
    
    Args:
        check_date: The date to validate
        oanda_close: The close price from OANDA
        instrument: The instrument name (only EUR_USD has reference data)
        tolerance: Maximum allowed difference (default 0.5%)
        
    Returns:
        Tuple of (is_valid, message)
    """
    if instrument != "EUR_USD":
        return True, f"No reference data for {instrument}, skipping validation"
    
    reference = load_reference_prices()
    
    if not reference:
        return True, "No reference data loaded, skipping validation"
    
    if check_date not in reference:
        return True, f"No reference price for {check_date}, skipping validation"
    
    ref_price = reference[check_date]
    diff = abs(oanda_close - ref_price)
    diff_pct = diff / ref_price * 100
    
    if diff_pct <= tolerance * 100:
        return True, f"Price validated: OANDA={oanda_close:.4f}, Ref={ref_price:.4f}, Diff={diff_pct:.3f}%"
    else:
        return False, f"Price mismatch: OANDA={oanda_close:.4f}, Ref={ref_price:.4f}, Diff={diff_pct:.3f}% (exceeds {tolerance*100}%)"


def validate_entry_price(
    entry_price: float,
    candle_high: float,
    candle_low: float,
    direction: str,
) -> Tuple[bool, str]:
    """
    Validate that an entry price is realistic within the candle's range.
    
    Args:
        entry_price: The proposed entry price
        candle_high: The candle's high price
        candle_low: The candle's low price
        direction: 'bullish' or 'bearish'
        
    Returns:
        Tuple of (is_valid, message)
    """
    if entry_price > candle_high:
        return False, f"Entry {entry_price} above candle high {candle_high}"
    
    if entry_price < candle_low:
        return False, f"Entry {entry_price} below candle low {candle_low}"
    
    return True, f"Entry {entry_price} within candle range [{candle_low}, {candle_high}]"


def validate_exit_price(
    exit_price: float,
    exit_type: str,
    candle_high: float,
    candle_low: float,
    direction: str,
) -> Tuple[bool, str]:
    """
    Validate that an exit price (TP or SL) could have been hit.
    
    Args:
        exit_price: The exit price
        exit_type: 'TP' or 'SL'
        candle_high: The candle's high
        candle_low: The candle's low
        direction: 'bullish' or 'bearish'
        
    Returns:
        Tuple of (is_valid, message)
    """
    if direction == "bullish":
        if exit_type == "TP":
            if candle_high >= exit_price:
                return True, f"TP {exit_price} hit (candle high {candle_high})"
            else:
                return False, f"TP {exit_price} not reached (candle high {candle_high})"
        else:
            if candle_low <= exit_price:
                return True, f"SL {exit_price} hit (candle low {candle_low})"
            else:
                return False, f"SL {exit_price} not hit (candle low {candle_low})"
    else:
        if exit_type == "TP":
            if candle_low <= exit_price:
                return True, f"TP {exit_price} hit (candle low {candle_low})"
            else:
                return False, f"TP {exit_price} not reached (candle low {candle_low})"
        else:
            if candle_high >= exit_price:
                return True, f"SL {exit_price} hit (candle high {candle_high})"
            else:
                return False, f"SL {exit_price} not hit (candle high {candle_high})"


def validate_candles_against_reference(
    candles: List[Dict],
    instrument: str = "EUR_USD",
    tolerance: float = 0.005,
) -> Dict:
    """
    Validate a batch of candles against reference prices.
    
    Args:
        candles: List of candle dicts with 'time' and 'close' keys
        instrument: The instrument name
        tolerance: Maximum allowed price difference
        
    Returns:
        Dictionary with validation results
    """
    if instrument != "EUR_USD":
        return {
            "instrument": instrument,
            "validated": 0,
            "passed": 0,
            "failed": 0,
            "skipped": len(candles),
            "errors": [],
            "note": "No reference data available for this instrument"
        }
    
    reference = load_reference_prices()
    
    if not reference:
        return {
            "instrument": instrument,
            "validated": 0,
            "passed": 0,
            "failed": 0,
            "skipped": len(candles),
            "errors": [],
            "note": "No reference file loaded"
        }
    
    validated = 0
    passed = 0
    failed = 0
    skipped = 0
    errors = []
    
    for candle in candles:
        candle_time = candle.get('time')
        candle_close = candle.get('close')
        
        if candle_time is None or candle_close is None:
            skipped += 1
            continue
        
        if isinstance(candle_time, datetime):
            candle_date = candle_time.date()
        elif isinstance(candle_time, date):
            candle_date = candle_time
        else:
            try:
                candle_date = datetime.fromisoformat(str(candle_time).replace('Z', '')).date()
            except:
                skipped += 1
                continue
        
        if candle_date not in reference:
            skipped += 1
            continue
        
        validated += 1
        ref_price = reference[candle_date]
        diff_pct = abs(candle_close - ref_price) / ref_price * 100
        
        if diff_pct <= tolerance * 100:
            passed += 1
        else:
            failed += 1
            errors.append({
                "date": candle_date.isoformat(),
                "oanda_close": candle_close,
                "reference": ref_price,
                "diff_pct": round(diff_pct, 4),
            })
    
    return {
        "instrument": instrument,
        "validated": validated,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors[:10],
        "note": f"Validated {validated} candles, {passed} passed, {failed} failed"
    }


def get_realistic_entry_price(
    candle: Dict,
    direction: str,
    target_price: Optional[float] = None,
) -> float:
    """
    Get a realistic entry price within the candle's range.
    
    If target_price is provided and is within the candle range, use it.
    Otherwise, use a reasonable default based on direction.
    
    Args:
        candle: The candle dict with high, low, open, close
        direction: 'bullish' or 'bearish'
        target_price: Optional target entry price from strategy
        
    Returns:
        A realistic entry price
    """
    high = candle['high']
    low = candle['low']
    open_price = candle['open']
    close = candle['close']
    
    if target_price is not None:
        if low <= target_price <= high:
            return target_price
    
    if direction == "bullish":
        return (low + close) / 2
    else:
        return (high + close) / 2
