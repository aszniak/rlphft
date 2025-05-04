import os
import time
import pickle
import requests
import pandas as pd


def fetch_and_cache_klines(
    symbol,
    interval="1m",
    limit=1000,
    lookback_days=7,
    force_refresh=False,
    cache_dir="data_cache",
):
    """Fetch klines data with caching to avoid repeated API calls"""
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Define cache filename
    cache_file = f"{cache_dir}/{symbol}_{interval}_{lookback_days}days.pkl"

    # Check if cache file exists and we're not forcing a refresh
    if os.path.exists(cache_file) and not force_refresh:
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # If we're here, we need to fetch the data
    print(
        f"Fetching fresh data for {symbol} ({interval} interval, {lookback_days} days)"
    )
    all_klines = []

    # Calculate end time (now) and start time (X days ago)
    end_time = int(time.time() * 1000)
    start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)

    current_end = end_time

    # Make multiple API calls to get all the data
    while current_end > start_time:
        # API endpoint
        base_url = "https://api.binance.com"
        endpoint = "/api/v3/klines"
        url = f"{base_url}{endpoint}"

        # Parameters
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": current_end,
        }

        # Make the request
        response = requests.get(url, params=params)
        klines = response.json()

        if not klines:
            break

        # Prepend the new klines to our collection
        all_klines = klines + all_klines

        # Update the end time for the next request
        current_end = int(klines[0][0]) - 1  # Start time of first candle minus 1ms

        # Be nice to the API - add a small delay
        time.sleep(0.1)

        # Status update
        print(f"Retrieved {len(klines)} candles, total: {len(all_klines)}")

    # Save to cache
    print(f"Saving data to cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(all_klines, f)

    return all_klines


def process_klines_data(klines_data):
    """Convert raw klines data to a DataFrame with proper columns"""
    df = pd.DataFrame(
        klines_data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades_count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    # Convert string values to appropriate types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Convert timestamps to datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    return df



