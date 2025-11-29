import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import os

pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCAD=X', 'AUDUSD=X', 'NZDUSD=X']

start_date = '2003-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

all_data = []

print(f"Downloading 4H forex data from {start_date} to {end_date}")
print("=" * 60)

for pair in pairs:
    print(f"\nDownloading data for {pair}...")
    try:
        ticker = yf.Ticker(pair)
        df = ticker.history(start=start_date, end=end_date, interval='1h')
        
        if df.empty:
            print(f"  No 1h data for {pair}, trying daily...")
            df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            print(f"  No data available for {pair}")
            continue
        
        df.reset_index(inplace=True)
        
        if 'Datetime' in df.columns:
            time_col = 'Datetime'
        elif 'Date' in df.columns:
            time_col = 'Date'
        else:
            time_col = df.columns[0]
        
        df['Pair'] = pair.replace('=X', '')
        all_data.append(df)
        print(f"  Downloaded {len(df)} rows for {pair}")
        print(f"  Date range: {df[time_col].min()} to {df[time_col].max()}")
        
        time.sleep(1)
        
    except Exception as e:
        print(f"  Error downloading {pair}: {e}")

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if 'Datetime' in combined_df.columns:
        combined_df.sort_values(['Datetime', 'Pair'], inplace=True)
    elif 'Date' in combined_df.columns:
        combined_df.sort_values(['Date', 'Pair'], inplace=True)
    
    combined_df.to_csv('main_forex_4h_2003_2025.csv', index=False)
    print(f"\n{'=' * 60}")
    print(f"Combined data saved to 'main_forex_4h_2003_2025.csv'")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    
    print("\nPreview of first 10 rows:")
    print(combined_df.head(10))
    
    print("\nData summary by pair:")
    for pair in combined_df['Pair'].unique():
        pair_data = combined_df[combined_df['Pair'] == pair]
        print(f"  {pair}: {len(pair_data)} rows")
else:
    print("\nNo data was downloaded.")
