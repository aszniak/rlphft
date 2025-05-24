"""
Santiment API Integration Module

This module provides functions to fetch sentiment and social metrics from Santiment API
and process them for integration with the trading environment.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from dotenv import load_dotenv

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


def get_sentiment_metrics(slug, from_date, to_date, interval="1h"):
    """
    Fetch sentiment metrics for a cryptocurrency.

    Args:
        slug (str): The cryptocurrency slug (e.g., "bitcoin")
        from_date (str): Start date in ISO format
        to_date (str): End date in ISO format
        interval (str): Data interval (e.g., "1h", "1d")

    Returns:
        dict: Dictionary containing DataFrames with sentiment metrics
    """
    client = get_santiment_client()

    # Define the GraphQL query for sentiment metrics
    query = gql(
        """
    {
      getMetric(metric: "sentiment_volume_consumed_twitter") {
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
      
      getMetric(metric: "social_volume_twitter") {
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
      
      getMetric(metric: "social_dominance_twitter") {
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
    sentiment_data = result["getMetric"]["timeseriesData"]
    if sentiment_data:
        df_sentiment = pd.DataFrame(sentiment_data)
        df_sentiment["datetime"] = pd.to_datetime(df_sentiment["datetime"])
        df_sentiment.set_index("datetime", inplace=True)
        df_sentiment.rename(columns={"value": "sentiment_score"}, inplace=True)
        metrics["sentiment"] = df_sentiment

    # Extract social volume
    social_volume_data = result["getMetric"][1]["timeseriesData"]
    if social_volume_data:
        df_volume = pd.DataFrame(social_volume_data)
        df_volume["datetime"] = pd.to_datetime(df_volume["datetime"])
        df_volume.set_index("datetime", inplace=True)
        df_volume.rename(columns={"value": "social_volume"}, inplace=True)
        metrics["social_volume"] = df_volume

    # Extract social dominance
    social_dominance_data = result["getMetric"][2]["timeseriesData"]
    if social_dominance_data:
        df_dominance = pd.DataFrame(social_dominance_data)
        df_dominance["datetime"] = pd.to_datetime(df_dominance["datetime"])
        df_dominance.set_index("datetime", inplace=True)
        df_dominance.rename(columns={"value": "social_dominance"}, inplace=True)
        metrics["social_dominance"] = df_dominance

    return metrics


def get_market_metrics(slug, from_date, to_date, interval="1h"):
    """
    Fetch market metrics like dev activity, github activity.

    Args:
        slug (str): The cryptocurrency slug (e.g., "bitcoin")
        from_date (str): Start date in ISO format
        to_date (str): End date in ISO format
        interval (str): Data interval (e.g., "1h", "1d")

    Returns:
        dict: Dictionary containing DataFrames with market metrics
    """
    client = get_santiment_client()

    # Define the GraphQL query for market metrics
    query = gql(
        """
    {
      getMetric(metric: "dev_activity") {
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
      
      getMetric(metric: "github_activity") {
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
      
      getMetric(metric: "exchange_inflow") {
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
      
      getMetric(metric: "exchange_outflow") {
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

    metric_names = [
        "dev_activity",
        "github_activity",
        "exchange_inflow",
        "exchange_outflow",
    ]

    for i, metric_name in enumerate(metric_names):
        data = result["getMetric"][i]["timeseriesData"]
        if data:
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df.rename(columns={"value": metric_name}, inplace=True)
            metrics[metric_name] = df

    return metrics


def merge_sentiment_with_price(price_df, sentiment_metrics):
    """
    Merge sentiment metrics with price data.

    Args:
        price_df (pd.DataFrame): DataFrame with price data
        sentiment_metrics (dict): Dictionary of sentiment metric DataFrames

    Returns:
        pd.DataFrame: Merged DataFrame with price and sentiment data
    """
    # Create a copy of the price DataFrame
    merged_df = price_df.copy()

    # Convert index to datetime if it's not already
    if not isinstance(merged_df.index, pd.DatetimeIndex):
        merged_df.index = pd.to_datetime(merged_df.index)

    # Merge each sentiment metric
    for metric_name, metric_df in sentiment_metrics.items():
        # Resample to match price data frequency
        price_freq = pd.infer_freq(merged_df.index)
        if price_freq:
            metric_df = metric_df.resample(price_freq).ffill()

        # Align indices
        for col in metric_df.columns:
            merged_df[col] = np.nan

            # Map sentiment data to price data timestamps
            for timestamp, row in metric_df.iterrows():
                # Find the closest timestamp in price data
                idx = merged_df.index.get_indexer([timestamp], method="nearest")[0]
                if idx >= 0 and idx < len(merged_df):
                    merged_df.iloc[idx, merged_df.columns.get_loc(col)] = row[col]

            # Forward fill missing values
            merged_df[col].fillna(method="ffill", inplace=True)
            # Backward fill any remaining NaNs at the beginning
            merged_df[col].fillna(method="bfill", inplace=True)
            # Fill any remaining NaNs with 0
            merged_df[col].fillna(0, inplace=True)

    return merged_df


def fetch_santiment_data(symbol, start_date, end_date):
    """
    Fetch sentiment and market metrics for a cryptocurrency.

    Args:
        symbol (str): Trading symbol (e.g., "BTCUSDT")
        start_date (str): Start date in ISO format
        end_date (str): End date in ISO format

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

    # Fetch sentiment metrics
    sentiment_metrics = get_sentiment_metrics(slug, start_date, end_date)

    # Fetch market metrics
    market_metrics = get_market_metrics(slug, start_date, end_date)

    # Combine all metrics
    all_metrics = {**sentiment_metrics, **market_metrics}

    return all_metrics


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
