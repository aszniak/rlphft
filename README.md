# Crypto Trading Research

A research project for algorithmic cryptocurrency trading and signal analysis using the Binance API.

## Features

- Fetch real-time cryptocurrency price data
- Retrieve historical candlestick (OHLCV) data
- Get 24-hour market statistics
- Public API access (no authentication required)
- Statistical analysis of price randomness
- Compare real data against synthetic benchmarks
- Framework for adding trading strategies

## Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the main script to fetch data and perform analysis:

   ```
   python main.py
   ```

## Analysis Features

### Signal Randomness Analysis

This project includes a comprehensive suite of statistical tests to determine whether cryptocurrency price movements exhibit randomness or contain patterns:

- **Basic Statistics**: Mean, standard deviation, skewness, kurtosis
- **Autocorrelation Tests**: Ljung-Box test for serial correlation
- **Runs Test**: Tests for randomness in the sequence of price changes
- **Stationarity Tests**: ADF and KPSS tests to check for unit roots and stationarity

### Synthetic Data Comparison

The analysis includes comparison of real cryptocurrency data against two synthetic datasets:

1. **Pure Random**: Generated from Gaussian noise
2. **Sinusoidal Pattern**: Contains a 10-point cycle plus noise

This comparison helps identify whether real price movements behave more like random noise or contain detectable patterns.

### Visualization

The project generates several visualization outputs:

- Price and returns time series plots
- Returns distribution histograms
- Autocorrelation and partial autocorrelation plots
- Q-Q plots for normality assessment
- Heatmaps of statistical test results

## Authentication

The current version works with public endpoints only. To access account data and trading functionality:

1. Create an account on [Binance](https://www.binance.com/)
2. Go to API Management in your account settings
3. Create a new API key with appropriate permissions
4. Copy `config.env.example` to `config.env` and update the `API_KEY` and `API_SECRET` variables

## Notes

- The code uses Binance's testnet by default, which is a sandbox environment
- To use real Binance API, set `testnet=False` when creating the client
- For production use, never hardcode API keys/secrets in your code
- Generated images and audio files are automatically excluded from git
