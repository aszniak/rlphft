import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.sandbox.stats.runs import runstest_1samp
from dotenv import load_dotenv
from data_fetcher import (
    fetch_and_cache_klines,
    process_klines_data,
    calculate_relative_changes,
)
import plotting as plt_utils  # Import our plotting utilities

# Load environment variables
load_dotenv("config.env")
API_KEY = os.getenv("API_KEY")


def create_synthetic_datasets(size):
    """Create synthetic datasets for comparison"""
    np.random.seed(42)  # For reproducibility

    # Create random Gaussian noise as returns
    random_returns = np.random.normal(0, 0.001, size)
    # Convert to "price" by cumulative product
    random_price = 100 * np.cumprod(1 + random_returns)

    # Create sinusoidal pattern with 10-point cycle for returns
    t = np.arange(size)
    # Base sinusoidal with 10-point period + mean drift
    sin_returns = 0.0001 + 0.001 * np.sin(2 * np.pi * t / 10)
    # Add some noise
    sin_returns_with_noise = sin_returns + np.random.normal(0, 0.0005, size)
    # Convert to "price" by cumulative product
    sin_price = 100 * np.cumprod(1 + sin_returns_with_noise)

    return {
        "random": {"prices": random_price, "returns": random_returns},
        "sinusoidal": {"prices": sin_price, "returns": sin_returns_with_noise},
    }


def run_randomness_tests(prices, returns=None, title="Price Data"):
    """Run a series of tests to determine if the price data is random or has structure"""
    # Convert to numpy array if it's a Series
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Calculate returns if not provided
    if returns is None:
        returns = np.diff(prices) / prices[:-1]
    elif isinstance(returns, pd.Series):
        returns = returns.values

    results = {}

    # 1. Basic statistics
    results["mean"] = np.mean(returns)
    results["std"] = np.std(returns)
    results["skew"] = stats.skew(returns)
    results["kurtosis"] = stats.kurtosis(returns)

    # 2. Autocorrelation test (Ljung-Box test)
    # H0: The data are independently distributed
    lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
    results["ljung_box_stat"] = lb_result["lb_stat"].iloc[0]
    results["ljung_box_pvalue"] = lb_result["lb_pvalue"].iloc[0]

    # 3. Runs test for randomness (using statsmodels implementation)
    # H0: The sequence is random
    runs, runs_pvalue = runstest_1samp(returns, cutoff="median", correction=True)
    results["runs_test_stat"] = runs
    results["runs_test_pvalue"] = runs_pvalue

    # 4. Augmented Dickey-Fuller test for stationarity
    # H0: Unit root exists (non-stationary)
    adf_result = adfuller(returns)
    results["adf_stat"] = adf_result[0]
    results["adf_pvalue"] = adf_result[1]

    # 5. KPSS test for stationarity (complementary to ADF)
    # H0: Series is stationary
    kpss_result = kpss(returns, regression="c")
    results["kpss_stat"] = kpss_result[0]
    results["kpss_pvalue"] = kpss_result[1]

    # Display results
    print(f"\n--- Randomness Analysis for {title} ---")
    print(f"Number of data points: {len(returns)}")
    print("\nBasic Statistics:")
    print(f"Mean: {results['mean']:.8f}")
    print(f"Standard Deviation: {results['std']:.8f}")
    print(f"Skewness: {results['skew']:.4f}")
    print(f"Kurtosis: {results['kurtosis']:.4f}")

    print("\nAutocorrelation (Ljung-Box Test):")
    print(f"Statistic: {results['ljung_box_stat']:.4f}")
    print(f"p-value: {results['ljung_box_pvalue']:.6f}")
    print(
        f"Interpretation: {'Random (no autocorrelation)' if results['ljung_box_pvalue'] > 0.05 else 'Not random (has autocorrelation)'}"
    )

    print("\nRuns Test:")
    print(f"Statistic: {results['runs_test_stat']:.4f}")
    print(f"p-value: {results['runs_test_pvalue']:.6f}")
    print(
        f"Interpretation: {'Random' if results['runs_test_pvalue'] > 0.05 else 'Not random'}"
    )

    print("\nStationarity Tests:")
    print("ADF Test:")
    print(f"Statistic: {results['adf_stat']:.4f}")
    print(f"p-value: {results['adf_pvalue']:.6f}")
    print(
        f"Interpretation: {'Stationary' if results['adf_pvalue'] < 0.05 else 'Non-stationary'}"
    )

    print("KPSS Test:")
    print(f"Statistic: {results['kpss_stat']:.4f}")
    print(f"p-value: {results['kpss_pvalue']:.6f}")
    print(
        f"Interpretation: {'Non-stationary' if results['kpss_pvalue'] < 0.05 else 'Stationary'}"
    )

    # Create visualizations using our plotting utility
    plt_utils.plot_randomness_analysis(prices, returns, title)

    return results, returns, prices


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
    print(
        f"Using all {len(df)} data points (60*24*30 = 43,200 expected for a full month)"
    )

    # Calculate returns for BTC
    df_returns = calculate_relative_changes(df)
    btc_returns = df_returns["pct_change"].dropna().values

    # Create synthetic datasets for comparison
    synthetic_data = create_synthetic_datasets(len(df))

    # Store results from all datasets
    all_results = {}
    all_returns = {}
    all_prices = {}

    # Run randomness tests on real BTC data - explicitly using returns
    results, returns, prices = run_randomness_tests(
        df["close"].values, btc_returns, title=f"{symbol}_{interval}"
    )
    all_results["BTC"] = results
    all_returns["BTC"] = returns
    all_prices["BTC"] = prices

    # Run randomness tests on synthetic datasets
    for name, data in synthetic_data.items():
        results, returns, prices = run_randomness_tests(
            data["prices"], data["returns"], title=f"Synthetic_{name}"
        )
        all_results[name] = results
        all_returns[name] = returns
        all_prices[name] = prices

    # Create comparison plots using our plotting utility
    plt_utils.plot_dataset_comparison(all_results, all_returns, all_prices)

    # Print comparison summary
    plt_utils.print_comparison_summary(all_results)

    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()
