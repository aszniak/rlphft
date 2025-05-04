import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from data_fetcher import (
    fetch_and_cache_klines,
    process_klines_data,
    fetch_training_data,
    prepare_multi_asset_dataset,
)

# Load environment variables
load_dotenv("config.env")
API_KEY = os.getenv("API_KEY")

# Hardcoded list of top cryptocurrencies by volume
# These are the major coins with high liquidity
TOP_CRYPTOCURRENCIES = [
    "BTCUSDT",  # Bitcoin
    "ETHUSDT",  # Ethereum
    "BNBUSDT",  # Binance Coin
    "SOLUSDT",  # Solana
    "XRPUSDT",  # Ripple
]


def main():
    # Define parameters for training data
    interval = "1m"  # Using 1-minute data
    lookback_days = 30  # A full month of data

    print("Fetching training data for top cryptocurrencies...")

    # Fetch data for the hardcoded list of top cryptocurrencies
    data_dict = fetch_training_data(
        symbols=TOP_CRYPTOCURRENCIES,
        interval=interval,
        lookback_days=lookback_days,
        force_refresh=False,  # Use cached data if available
    )

    # Prepare dataset with additional features
    enhanced_data, combined_df = prepare_multi_asset_dataset(
        data_dict, add_indicators=True  # Enable technical indicators
    )

    # Print summary of the data
    print("\nTraining Data Summary:")
    for symbol, df in enhanced_data.items():
        print(
            f"{symbol}: {len(df)} candles from {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}"
        )
        print(f"  - Features: {len(df.columns)} columns")

        # Print sample of technical indicator columns
        tech_indicators = [
            col
            for col in df.columns
            if any(
                indicator in col for indicator in ["sma", "ema", "rsi", "macd", "bb_"]
            )
        ]
        if tech_indicators:
            print(
                f"  - Technical indicators include: {', '.join(tech_indicators[:5])}..."
            )

        # Print sample of normalized columns
        norm_cols = [col for col in df.columns if col.endswith("_norm")]
        if norm_cols:
            print(f"  - Normalized features include: {', '.join(norm_cols[:5])}...")

    print("\nCombined DataFrame Shape:", combined_df.shape)
    print(
        f"Combined DataFrame Columns (first 10): {combined_df.columns[:10].tolist()}..."
    )

    # Data is now ready for use in RL training
    print("\nData preparation completed. Available objects:")
    print("- data_dict: Raw price data for each symbol")
    print("- enhanced_data: Processed data with technical indicators for each symbol")
    print("- combined_df: Combined data for all symbols aligned by timestamp")

    # Return data for potential further use
    return data_dict, enhanced_data, combined_df


if __name__ == "__main__":
    main()
