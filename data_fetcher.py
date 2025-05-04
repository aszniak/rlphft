import os
import time
import pickle
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
from technical_indicators import add_technical_indicators, normalize_features


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
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]:
        df[col] = df[col].astype(float)

    # Convert timestamps to datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df


def get_top_coins_by_volume(quote_asset="USDT", limit=5, cache_hours=24):
    """
    Fetch top cryptocurrency pairs by volume

    NOTE: While not currently used directly (we use hardcoded list),
    this function is kept for potential future dynamic symbol selection.

    Args:
        quote_asset: Base currency (e.g., 'USDT')
        limit: Number of top pairs to return
        cache_hours: Hours to cache the results

    Returns:
        List of symbol strings (e.g., ['BTCUSDT', 'ETHUSDT'])
    """
    cache_dir = "data_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file = f"{cache_dir}/top_coins_{quote_asset}_{limit}.pkl"

    # Check if we have a recent cache
    if os.path.exists(cache_file):
        file_mod_time = os.path.getmtime(cache_file)
        if (time.time() - file_mod_time) < cache_hours * 3600:
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    # Fetch 24hr ticker data
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)
    all_tickers = response.json()

    # Filter for quote asset and sort by volume
    quote_pairs = [t for t in all_tickers if t["symbol"].endswith(quote_asset)]
    sorted_by_volume = sorted(
        quote_pairs, key=lambda x: float(x["quoteVolume"]), reverse=True
    )

    # Get the top N symbols
    top_symbols = [ticker["symbol"] for ticker in sorted_by_volume[:limit]]

    # Cache the results
    with open(cache_file, "wb") as f:
        pickle.dump(top_symbols, f)

    return top_symbols


def fetch_training_data(
    symbols: Union[List[str], None] = None,
    interval: str = "1m",
    lookback_days: int = 30,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch training data for multiple symbols

    Args:
        symbols: List of symbol pairs (e.g., ['BTCUSDT', 'ETHUSDT']).
                If None, fetches top 5 by volume.
        interval: Candlestick interval ('1m', '5m', '15m', '1h', etc.)
        lookback_days: How many days of historical data to fetch
        force_refresh: Whether to force refresh cached data

    Returns:
        Dictionary mapping symbols to their respective DataFrames
    """
    # If symbols not provided, get top coins by volume
    if symbols is None:
        symbols = get_top_coins_by_volume(limit=5)
        print(f"Using top 5 coins by volume: {symbols}")

    result = {}

    # Fetch data for each symbol
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        klines = fetch_and_cache_klines(
            symbol=symbol,
            interval=interval,
            lookback_days=lookback_days,
            force_refresh=force_refresh,
        )

        df = process_klines_data(klines)
        result[symbol] = df

    return result


def prepare_multi_asset_dataset(
    data_dict: Dict[str, pd.DataFrame], add_indicators: bool = True
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Prepare a multi-asset dataset suitable for RL training.

    Args:
        data_dict: Dictionary mapping symbols to DataFrames
        add_indicators: Whether to add basic technical indicators

    Returns:
        Tuple containing:
        - Dictionary mapping symbols to enhanced DataFrames
        - Combined DataFrame with all assets aligned by timestamp
    """
    enhanced_data = {}

    # Process each asset's data
    for symbol, df in data_dict.items():
        # Make a copy to avoid modifying the original
        df_enhanced = df.copy()

        # Calculate returns
        df_enhanced["returns"] = df_enhanced["close"].pct_change()

        # Calculate log returns
        df_enhanced["log_returns"] = np.log(
            df_enhanced["close"] / df_enhanced["close"].shift(1)
        )

        # Add technical indicators if requested
        if add_indicators:
            print(f"Adding technical indicators for {symbol}...")
            df_enhanced = add_technical_indicators(df_enhanced)

            # Normalize features to make them suitable for RL
            print(f"Normalizing features for {symbol}...")
            df_enhanced = normalize_features(df_enhanced)

        # Drop any rows with NaN values (first row will have NaN returns)
        df_enhanced = df_enhanced.dropna()

        enhanced_data[symbol] = df_enhanced

    # Create a combined dataframe aligned by timestamp
    # This will be useful for analyzing correlations and multi-asset strategies
    combined_df = None
    for symbol, df in enhanced_data.items():
        # Set the index to open_time temporarily for the merge
        temp_df = df.set_index("open_time")

        # For the combined DF, include only the essential features
        if add_indicators:
            # If we have technical indicators, include the normalized versions
            norm_cols = [col for col in temp_df.columns if col.endswith("_norm")]
            cols_to_include = ["close", "volume", "returns", "log_returns"] + norm_cols
        else:
            cols_to_include = ["close", "volume", "returns", "log_returns"]

        renamed_cols = {col: f"{symbol}_{col}" for col in cols_to_include}
        temp_df = temp_df[cols_to_include].rename(columns=renamed_cols)

        if combined_df is None:
            combined_df = temp_df
        else:
            combined_df = combined_df.join(temp_df, how="outer")

    # Reset the index to make open_time a column again
    if combined_df is not None:
        combined_df = combined_df.reset_index()

    return enhanced_data, combined_df
