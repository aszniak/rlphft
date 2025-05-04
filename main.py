import os
import pandas as pd
from dotenv import load_dotenv
from data_fetcher import (
    fetch_and_cache_klines,
    process_klines_data
)

# Load environment variables
load_dotenv("config.env")
API_KEY = os.getenv("API_KEY")


def main():
    # Define parameters
    symbol = "BTCUSDT"
    interval = "1m"  # Using 1-minute data for everything
    lookback_days = 30  # A full month of data (60*24*30 = 43,200 samples)

    # Fetch data (or load from cache)
    klines_data = fetch_and_cache_klines(
        symbol, interval=interval, lookback_days=lookback_days
    )

    # Process into DataFrame
    df = process_klines_data(klines_data)
    print(f"Working with {len(df)} {interval} candles for {symbol}")
    print(f"Period: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")


    # Data is now ready for use
    print("\nData fetching completed. Data available in:")
    print("- df: Raw price data")
    print("- df_returns: Calculated returns")

    # Return data for potential further use
    return df


if __name__ == "__main__":
    main()
