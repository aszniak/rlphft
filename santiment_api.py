"""
Santiment API Integration Module

This module provides functions to fetch sentiment and social metrics from Santiment API
and process them for integration with the trading environment.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from dotenv import load_dotenv
import time

# Load API key from environment variables
load_dotenv("config.env")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")


# Initialize GraphQL client
def get_santiment_client():
    """Initialize and return a Santiment GraphQL client."""
    if not SANTIMENT_API_KEY:
        raise ValueError("SANTIMENT_API_KEY not found in environment variables.")

    transport = RequestsHTTPTransport(
        url="https://api.santiment.net/graphql",
        headers={"Authorization": f"Apikey {SANTIMENT_API_KEY}"},
    )
    return Client(transport=transport, fetch_schema_from_transport=True)


def _get_santiment_cache_path(
    cache_type, slug, from_date, to_date, interval="1h", cache_dir="data_cache"
):
    """
    Generate cache file path for Santiment data.

    Args:
        cache_type (str): Type of data ("sentiment" or "market")
        slug (str): Cryptocurrency slug
        from_date (str): Start date
        to_date (str): End date
        interval (str): Data interval
        cache_dir (str): Cache directory

    Returns:
        str: Cache file path
    """
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Create a simple hash of the date range for the filename
    date_hash = abs(hash(f"{from_date}_{to_date}_{interval}")) % 100000
    cache_file = f"{cache_dir}/santiment_{cache_type}_{slug}_{date_hash}.pkl"
    return cache_file


def _load_santiment_cache(cache_file, max_age_hours=24):
    """
    Load Santiment data from cache if it exists and is recent enough.

    Args:
        cache_file (str): Path to cache file
        max_age_hours (int): Maximum age of cache in hours

    Returns:
        dict or None: Cached data if available and recent, None otherwise
    """
    if not os.path.exists(cache_file):
        return None

    # Check cache age
    file_mod_time = os.path.getmtime(cache_file)
    if (time.time() - file_mod_time) > max_age_hours * 3600:
        print(f"📂 Cache expired for {cache_file}")
        return None

    try:
        with open(cache_file, "rb") as f:
            print(f"📂 Loading cached Santiment data from {cache_file}")
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Error loading cache {cache_file}: {e}")
        return None


def _save_santiment_cache(data, cache_file):
    """
    Save Santiment data to cache.

    Args:
        data (dict): Data to cache
        cache_file (str): Path to cache file
    """
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Saved Santiment data to cache: {cache_file}")
    except Exception as e:
        print(f"⚠️ Error saving cache {cache_file}: {e}")


def get_sentiment_metrics(
    slug, from_date, to_date, interval="1h", force_refresh=False, cache_hours=24
):
    """
    Fetch sentiment metrics for a cryptocurrency with caching.

    Args:
        slug (str): The cryptocurrency slug (e.g., "bitcoin")
        from_date (str): Start date in ISO format
        to_date (str): End date in ISO format
        interval (str): Data interval (e.g., "1h", "1d")
        force_refresh (bool): Whether to force refresh cached data
        cache_hours (int): Hours to keep cache valid

    Returns:
        dict: Dictionary containing DataFrames with sentiment metrics
    """
    # Try to load from cache first
    cache_file = _get_santiment_cache_path(
        "sentiment", slug, from_date, to_date, interval
    )

    if not force_refresh:
        cached_data = _load_santiment_cache(cache_file, max_age_hours=cache_hours)
        if cached_data is not None:
            return cached_data

    print(f"🔄 Fetching fresh sentiment data for {slug}")

    client = get_santiment_client()

    # Define the GraphQL query for sentiment metrics
    query = gql(
        """
    {
      sentimentVolume: getMetric(metric: "sentiment_volume_consumed_twitter") {
        timeseriesData(
          slug: "%s"
          from: "%s"
          to: "%s"
          interval: "%s"
        ) {
          datetime
          value
        }
      }
      
      socialVolume: getMetric(metric: "social_volume_twitter") {
        timeseriesData(
          slug: "%s"
          from: "%s"
          to: "%s"
          interval: "%s"
        ) {
          datetime
          value
        }
      }
      
      socialDominance: getMetric(metric: "social_dominance_twitter") {
        timeseriesData(
          slug: "%s"
          from: "%s"
          to: "%s"
          interval: "%s"
        ) {
          datetime
          value
        }
      }
    }
    """
        % (
            slug,
            from_date,
            to_date,
            interval,
            slug,
            from_date,
            to_date,
            interval,
            slug,
            from_date,
            to_date,
            interval,
        )
    )

    # Execute the query
    result = client.execute(query)

    # Process the results into DataFrames
    metrics = {}

    # Extract sentiment volume
    sentiment_data = result["sentimentVolume"]["timeseriesData"]
    if sentiment_data:
        df_sentiment = pd.DataFrame(sentiment_data)
        df_sentiment["datetime"] = pd.to_datetime(df_sentiment["datetime"])
        df_sentiment.set_index("datetime", inplace=True)
        df_sentiment.rename(columns={"value": "sentiment_score"}, inplace=True)
        metrics["sentiment"] = df_sentiment

    # Extract social volume
    social_volume_data = result["socialVolume"]["timeseriesData"]
    if social_volume_data:
        df_volume = pd.DataFrame(social_volume_data)
        df_volume["datetime"] = pd.to_datetime(df_volume["datetime"])
        df_volume.set_index("datetime", inplace=True)
        df_volume.rename(columns={"value": "social_volume"}, inplace=True)
        metrics["social_volume"] = df_volume

    # Extract social dominance
    social_dominance_data = result["socialDominance"]["timeseriesData"]
    if social_dominance_data:
        df_dominance = pd.DataFrame(social_dominance_data)
        df_dominance["datetime"] = pd.to_datetime(df_dominance["datetime"])
        df_dominance.set_index("datetime", inplace=True)
        df_dominance.rename(columns={"value": "social_dominance"}, inplace=True)
        metrics["social_dominance"] = df_dominance

    # Save to cache
    _save_santiment_cache(metrics, cache_file)

    return metrics


def get_market_metrics(
    slug, from_date, to_date, interval="1h", force_refresh=False, cache_hours=24
):
    """
    Fetch market metrics like dev activity, github activity with caching.

    Args:
        slug (str): The cryptocurrency slug (e.g., "bitcoin")
        from_date (str): Start date in ISO format
        to_date (str): End date in ISO format
        interval (str): Data interval (e.g., "1h", "1d")
        force_refresh (bool): Whether to force refresh cached data
        cache_hours (int): Hours to keep cache valid

    Returns:
        dict: Dictionary containing DataFrames with market metrics
    """
    # Try to load from cache first
    cache_file = _get_santiment_cache_path("market", slug, from_date, to_date, interval)

    if not force_refresh:
        cached_data = _load_santiment_cache(cache_file, max_age_hours=cache_hours)
        if cached_data is not None:
            return cached_data

    print(f"🔄 Fetching fresh market data for {slug}")

    client = get_santiment_client()

    # Define the GraphQL query for market metrics
    query = gql(
        """
    {
      devActivity: getMetric(metric: "dev_activity") {
        timeseriesData(
          slug: "%s"
          from: "%s"
          to: "%s"
          interval: "%s"
        ) {
          datetime
          value
        }
      }
      
      githubActivity: getMetric(metric: "github_activity") {
        timeseriesData(
          slug: "%s"
          from: "%s"
          to: "%s"
          interval: "%s"
        ) {
          datetime
          value
        }
      }
      
      exchangeInflow: getMetric(metric: "exchange_inflow") {
        timeseriesData(
          slug: "%s"
          from: "%s"
          to: "%s"
          interval: "%s"
        ) {
          datetime
          value
        }
      }
      
      exchangeOutflow: getMetric(metric: "exchange_outflow") {
        timeseriesData(
          slug: "%s"
          from: "%s"
          to: "%s"
          interval: "%s"
        ) {
          datetime
          value
        }
      }
    }
    """
        % (
            slug,
            from_date,
            to_date,
            interval,
            slug,
            from_date,
            to_date,
            interval,
            slug,
            from_date,
            to_date,
            interval,
            slug,
            from_date,
            to_date,
            interval,
        )
    )

    # Execute the query
    result = client.execute(query)

    # Process the results into DataFrames
    metrics = {}

    # Process each metric using aliases
    alias_to_metric = {
        "devActivity": "dev_activity",
        "githubActivity": "github_activity",
        "exchangeInflow": "exchange_inflow",
        "exchangeOutflow": "exchange_outflow",
    }

    for alias, metric_name in alias_to_metric.items():
        data = result[alias]["timeseriesData"]
        if data:
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df.rename(columns={"value": metric_name}, inplace=True)
            metrics[metric_name] = df

    # Save to cache
    _save_santiment_cache(metrics, cache_file)

    return metrics


def merge_sentiment_with_price(price_df, sentiment_metrics):
    """
    Merge sentiment metrics with price data.

    Args:
        price_df (pd.DataFrame): DataFrame with price data (with 'open_time' column)
        sentiment_metrics (dict): Dictionary of sentiment metric DataFrames

    Returns:
        pd.DataFrame: Merged DataFrame with price and sentiment data
    """
    # Create a copy of the price DataFrame
    merged_df = price_df.copy()

    # Ensure open_time is datetime type
    if "open_time" in merged_df.columns:
        merged_df["open_time"] = pd.to_datetime(merged_df["open_time"])
        # Ensure timezone-naive for compatibility
        if merged_df["open_time"].dt.tz is not None:
            merged_df["open_time"] = merged_df["open_time"].dt.tz_localize(None)
    else:
        raise ValueError("Expected 'open_time' column in price DataFrame")

    # Merge each sentiment metric
    for metric_name, metric_df in sentiment_metrics.items():
        if metric_df is None or metric_df.empty:
            continue

        # Ensure metric_df has datetime index
        if not isinstance(metric_df.index, pd.DatetimeIndex):
            metric_df.index = pd.to_datetime(metric_df.index)

        # Ensure timezone-naive for compatibility
        if metric_df.index.tz is not None:
            metric_df.index = metric_df.index.tz_localize(None)

        # Use pandas merge_asof for efficient time-based merging
        for col in metric_df.columns:
            # Reset indices to use merge_asof
            sentiment_temp = metric_df[[col]].reset_index()
            sentiment_temp.columns = ["datetime", col]

            # Sort by datetime for merge_asof
            merged_df = merged_df.sort_values("open_time")
            sentiment_temp = sentiment_temp.sort_values("datetime")

            # Merge using merge_asof (backward search)
            merged_temp = pd.merge_asof(
                merged_df[["open_time"]].reset_index(),
                sentiment_temp,
                left_on="open_time",
                right_on="datetime",
                direction="backward",
            )

            # Add the sentiment column to our main dataframe
            if len(merged_temp) == len(merged_df):
                merged_df[col] = merged_temp[col].values
            else:
                # Fallback: fill with NaN and then forward fill
                merged_df[col] = np.nan

            # Forward fill missing values
            merged_df[col] = merged_df[col].ffill()
            # Fill any remaining NaNs with 0
            merged_df[col] = merged_df[col].fillna(0)

    return merged_df


def fetch_santiment_data(
    symbol, start_date, end_date, force_refresh=False, cache_hours=24
):
    """
    Fetch sentiment and market metrics for a cryptocurrency with caching.

    Args:
        symbol (str): Trading symbol (e.g., "BTCUSDT")
        start_date (str): Start date in ISO format
        end_date (str): End date in ISO format
        force_refresh (bool): Whether to force refresh cached data
        cache_hours (int): Hours to keep cache valid

    Returns:
        dict: Dictionary containing DataFrames with sentiment and market metrics
    """
    # Convert trading symbol to Santiment slug
    slug_mapping = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "BNBUSDT": "binance-coin",
        "SOLUSDT": "solana",
        "XRPUSDT": "ripple",
        # Add more mappings as needed
    }

    slug = slug_mapping.get(symbol)
    if not slug:
        raise ValueError(f"No Santiment slug mapping found for symbol: {symbol}")

    # Fetch sentiment metrics with caching
    sentiment_metrics = get_sentiment_metrics(
        slug, start_date, end_date, force_refresh=force_refresh, cache_hours=cache_hours
    )

    # Fetch market metrics with caching
    market_metrics = get_market_metrics(
        slug, start_date, end_date, force_refresh=force_refresh, cache_hours=cache_hours
    )

    # Combine all metrics
    all_metrics = {**sentiment_metrics, **market_metrics}

    return all_metrics


def ensure_consistent_sentiment_features(df):
    """
    Ensure DataFrame has all expected sentiment columns, filling missing ones with zeros.
    This maintains consistent observation space dimensions across all symbols.

    Args:
        df (pd.DataFrame): DataFrame that may or may not have sentiment features

    Returns:
        pd.DataFrame: DataFrame with all expected sentiment columns
    """
    # Define all expected sentiment columns (both raw and normalized)
    expected_sentiment_columns = [
        "sentiment_score",
        "social_volume",
        "social_dominance",
        "dev_activity",
        "github_activity",
        "exchange_inflow",
        "exchange_outflow",
    ]

    expected_normalized_columns = [f"{col}_norm" for col in expected_sentiment_columns]
    all_expected_columns = expected_sentiment_columns + expected_normalized_columns

    # Make a copy to avoid modifying the original
    df_consistent = df.copy()

    # Add missing sentiment columns with zeros
    for col in all_expected_columns:
        if col not in df_consistent.columns:
            df_consistent[col] = 0.0

    return df_consistent


def normalize_sentiment_features(df):
    """
    Normalize sentiment features.

    Args:
        df (pd.DataFrame): DataFrame with sentiment features

    Returns:
        pd.DataFrame: DataFrame with normalized sentiment features
    """
    sentiment_columns = [
        "sentiment_score",
        "social_volume",
        "social_dominance",
        "dev_activity",
        "github_activity",
        "exchange_inflow",
        "exchange_outflow",
    ]

    normalized_df = df.copy()

    for col in sentiment_columns:
        if col in df.columns:
            # Get rolling mean and std for normalization
            rolling_mean = df[col].rolling(window=24).mean()
            rolling_std = df[col].rolling(window=24).std()

            # Handle initial NaN values
            rolling_mean.fillna(df[col].mean(), inplace=True)
            rolling_std.fillna(df[col].std(), inplace=True)

            # Avoid division by zero
            rolling_std = rolling_std.replace(0, 1)

            # Z-score normalization
            normalized_df[f"{col}_norm"] = (df[col] - rolling_mean) / rolling_std

    return normalized_df
