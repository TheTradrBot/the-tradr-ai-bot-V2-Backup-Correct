#!/usr/bin/env python3
"""
Download 1-hour historical data from Dukascopy for backtesting.
"""
import os
from datetime import datetime
import dukascopy_python
from dukascopy_python import instruments
import pandas as pd

DATA_DIR = 'data'

INSTRUMENTS = {
    'EUR_USD': instruments.INSTRUMENT_FX_MAJORS_EUR_USD,
    'GBP_USD': instruments.INSTRUMENT_FX_MAJORS_GBP_USD,
    'USD_JPY': instruments.INSTRUMENT_FX_MAJORS_USD_JPY,
    'USD_CHF': instruments.INSTRUMENT_FX_MAJORS_USD_CHF,
    'AUD_USD': instruments.INSTRUMENT_FX_MAJORS_AUD_USD,
    'USD_CAD': instruments.INSTRUMENT_FX_MAJORS_USD_CAD,
    'NZD_USD': instruments.INSTRUMENT_FX_MAJORS_NZD_USD,
    'EUR_GBP': instruments.INSTRUMENT_FX_CROSSES_EUR_GBP,
    'EUR_JPY': instruments.INSTRUMENT_FX_CROSSES_EUR_JPY,
    'GBP_JPY': instruments.INSTRUMENT_FX_CROSSES_GBP_JPY,
    'XAU_USD': instruments.INSTRUMENT_FX_METALS_XAU_USD,
    'XAG_USD': instruments.INSTRUMENT_FX_METALS_XAG_USD,
    'SPX500_USD': instruments.INSTRUMENT_IDX_AMERICA_E_SANDP_500,
    'NAS100_USD': instruments.INSTRUMENT_IDX_AMERICA_E_NQ_100,
}

def download_instrument(symbol: str, start_year: int = 2003, end_year: int = 2025):
    """Download 1-hour data for a single instrument."""
    if symbol not in INSTRUMENTS:
        print(f"  Unknown instrument: {symbol}")
        return None
    
    instrument = INSTRUMENTS[symbol]
    all_data = []
    
    for year in range(start_year, end_year + 1):
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        
        if year == end_year:
            end = datetime.now()
        
        try:
            print(f"  Fetching {symbol} {year}...", end=' ', flush=True)
            df = dukascopy_python.fetch(
                instrument=instrument,
                interval=dukascopy_python.INTERVAL_HOUR_1,
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=start,
                end=end
            )
            if df is not None and len(df) > 0:
                all_data.append(df)
                print(f"{len(df)} candles")
            else:
                print("no data")
        except Exception as e:
            print(f"error: {str(e)[:50]}")
    
    if all_data:
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        return combined
    return None


def save_to_csv(df: pd.DataFrame, symbol: str):
    """Save DataFrame to CSV in the data directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    df_save = df.reset_index()
    df_save = df_save.rename(columns={
        'timestamp': 'Date',
        'index': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    if 'Date' in df_save.columns:
        df_save['Date'] = pd.to_datetime(df_save['Date']).dt.tz_localize(None)
    
    filepath = os.path.join(DATA_DIR, f'{symbol}_1H.csv')
    df_save.to_csv(filepath, index=False)
    print(f"  Saved: {filepath} ({len(df_save)} rows)")
    return filepath


def download_all():
    """Download all instruments."""
    print("="*60)
    print("DUKASCOPY 1-HOUR DATA DOWNLOAD")
    print("="*60)
    
    results = {}
    
    for symbol in INSTRUMENTS.keys():
        print(f"\n[{symbol}]")
        df = download_instrument(symbol, start_year=2010, end_year=2025)
        if df is not None:
            filepath = save_to_csv(df, symbol)
            results[symbol] = len(df)
        else:
            results[symbol] = 0
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for symbol, count in results.items():
        status = "OK" if count > 0 else "FAILED"
        print(f"  {symbol}: {count:,} candles [{status}]")
    
    total = sum(results.values())
    print(f"\nTotal: {total:,} candles downloaded")
    return results


if __name__ == '__main__':
    download_all()
