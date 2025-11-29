"""
Historical Forex Data Downloader

Downloads free historical forex data from HistData.com
Data available from 2000-present for major forex pairs

Usage:
    python download_histdata.py

The data will be saved to the /data folder and can be used by the bot's
backtest system automatically.
"""

import os
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"

PAIRS = {
    'EURUSD': 'EUR_USD',
    'GBPUSD': 'GBP_USD',
    'USDJPY': 'USD_JPY',
    'USDCHF': 'USD_CHF',
    'AUDUSD': 'AUD_USD',
    'USDCAD': 'USD_CAD',
    'NZDUSD': 'NZD_USD',
    'EURGBP': 'EUR_GBP',
    'EURJPY': 'EUR_JPY',
    'GBPJPY': 'GBP_JPY',
    'XAUUSD': 'XAU_USD',
}

def download_histdata_ascii(pair: str, year: int) -> pd.DataFrame:
    """
    Download historical data from HistData.com ASCII format.
    
    Note: HistData.com requires manual download due to captcha.
    This function provides instructions for manual download.
    """
    url = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair.lower()}/{year}"
    print(f"  Manual download required for {pair} {year}")
    print(f"  Visit: {url}")
    return pd.DataFrame()


def download_from_dukascopy(pair: str, year: int, month: int) -> pd.DataFrame:
    """
    Download data from Dukascopy (requires their API or manual download).
    Dukascopy provides tick data from 2003.
    """
    print(f"Dukascopy requires their JForex platform or manual CSV export")
    print(f"Visit: https://www.dukascopy.com/swiss/english/marketwatch/historical/")
    return pd.DataFrame()


def create_sample_csv_format():
    """Create a sample CSV to show the expected format."""
    sample_data = {
        'timestamp': ['2003-01-02 00:00:00', '2003-01-02 04:00:00', '2003-01-02 08:00:00'],
        'open': [1.0446, 1.0450, 1.0455],
        'high': [1.0460, 1.0465, 1.0470],
        'low': [1.0440, 1.0445, 1.0450],
        'close': [1.0450, 1.0455, 1.0460],
        'volume': [1000, 1200, 1100]
    }
    df = pd.DataFrame(sample_data)
    
    sample_path = DATA_DIR / "SAMPLE_FORMAT.csv"
    df.to_csv(sample_path, index=False)
    print(f"Sample CSV format saved to: {sample_path}")
    return sample_path


def print_download_instructions():
    """Print instructions for downloading historical data."""
    print("\n" + "="*70)
    print("HISTORICAL FOREX DATA DOWNLOAD GUIDE")
    print("="*70)
    
    print("\n[OPTION 1] HISTDATA.COM (FREE - 2000 to Present)")
    print("-" * 50)
    print("1. Visit: https://www.histdata.com/download-free-forex-data/")
    print("2. Select 'ASCII' format")
    print("3. Choose timeframe: 1 Minute Bars")
    print("4. Select pair (e.g., EURUSD)")
    print("5. Download each year's data (2003-2024)")
    print("6. Extract ZIP files and combine into single CSV per pair")
    print("7. Place in: " + str(DATA_DIR))
    
    print("\n[OPTION 2] DUKASCOPY (FREE - 2003 to Present)")
    print("-" * 50)
    print("1. Visit: https://www.dukascopy.com/swiss/english/marketwatch/historical/")
    print("2. Select instrument (e.g., EUR/USD)")
    print("3. Set date range: 2003-01-01 to today")
    print("4. Select timeframe: Hourly or 4-Hour")
    print("5. Download CSV")
    print("6. Rename to EURUSD.csv and place in: " + str(DATA_DIR))
    
    print("\n[OPTION 3] TRUEFX (FREE - 2009 to Present)")
    print("-" * 50)
    print("1. Visit: https://www.truefx.com/dev/data/")
    print("2. Download historical tick data")
    print("3. Convert to H4 OHLCV format")
    
    print("\n" + "="*70)
    print("CSV FILE FORMAT REQUIRED:")
    print("="*70)
    print("""
File naming: EURUSD.csv, GBPUSD.csv, XAUUSD.csv, etc.
Place files in: """ + str(DATA_DIR) + """

Columns required:
  - timestamp (or time/date): ISO format like 2003-01-02 00:00:00
  - open: Opening price
  - high: Highest price
  - low: Lowest price  
  - close: Closing price
  - volume: Volume (optional, can be 0)

Example CSV content:
timestamp,open,high,low,close,volume
2003-01-02 00:00:00,1.0446,1.0460,1.0440,1.0450,1000
2003-01-02 04:00:00,1.0450,1.0465,1.0445,1.0455,1200
...
""")
    
    print("\n[RECOMMENDED PAIRS TO DOWNLOAD]")
    print("-" * 50)
    for histdata_name, oanda_name in PAIRS.items():
        print(f"  {histdata_name}.csv -> {oanda_name}")
    
    print("\n" + "="*70)


def check_existing_data():
    """Check what data files already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[CURRENT DATA STATUS]")
    print("-" * 50)
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in data folder.")
        print(f"Data folder: {DATA_DIR}")
    else:
        for csv_file in csv_files:
            if csv_file.name == "SAMPLE_FORMAT.csv":
                continue
            try:
                df = pd.read_csv(csv_file, nrows=5)
                full_df = pd.read_csv(csv_file)
                
                time_col = None
                for col in ['timestamp', 'time', 'date', 'Date', 'Time']:
                    if col in full_df.columns:
                        time_col = col
                        break
                
                if time_col:
                    full_df[time_col] = pd.to_datetime(full_df[time_col])
                    min_date = full_df[time_col].min()
                    max_date = full_df[time_col].max()
                    print(f"  {csv_file.name}: {len(full_df):,} rows | {min_date.date()} to {max_date.date()}")
                else:
                    print(f"  {csv_file.name}: {len(full_df):,} rows | No timestamp column found")
            except Exception as e:
                print(f"  {csv_file.name}: Error reading - {str(e)[:40]}")
    
    print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BLUEPRINT TRADER AI - HISTORICAL DATA SETUP")
    print("="*70)
    
    check_existing_data()
    print_download_instructions()
    create_sample_csv_format()
    
    print("\nAfter downloading data, run: python data_loader.py")
    print("to verify the data is loaded correctly.\n")
