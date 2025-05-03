# Crypto Trading Research

A research project for algorithmic cryptocurrency trading using Binance API.

## Features

- Fetch real-time cryptocurrency price data
- Retrieve historical candlestick (OHLCV) data
- Get 24-hour market statistics
- Public API access (no authentication required)
- Framework for adding trading strategies

## Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the main script to fetch basic market data:

   ```
   python main.py
   ```

## Authentication

The current version works with public endpoints only. To access account data and trading functionality:

1. Create an account on [Binance](https://www.binance.com/)
2. Go to API Management in your account settings
3. Create a new API key with appropriate permissions
4. Update the `API_KEY` and `API_SECRET` variables in the code

## Notes

- The code uses Binance's testnet by default, which is a sandbox environment
- To use real Binance API, set `testnet=False` when creating the client
- For production use, never hardcode API keys/secrets in your code
