# Reinforcement Learning for Cryptocurrency Trading

This project applies reinforcement learning techniques to cryptocurrency trading using Binance market data.

## Python Setup

The project uses Python 3.10 for all scripts, as it has the necessary packages installed.

### Running Scripts

Instead of using `python script.py`, use the provided shell script:

```bash
./run.sh script.py [arguments]
```

This ensures all scripts run with the correct Python version and environment.

### Visualization

To visualize the technical indicators:

```bash
# Visualize a random cryptocurrency
./run_visualization.sh

# Visualize a specific cryptocurrency
./run_visualization.sh BTCUSDT

# Visualize a specific cryptocurrency with a custom timeframe (days)
./run_visualization.sh ETHUSDT 3
```

This will generate a PNG file showing the selected cryptocurrency with all technical indicators, including:

- Price chart with SMA, EMA, and Bollinger Bands
- Volume with SMA
- RSI and Stochastic oscillators
- MACD
- ATR (Average True Range) and Z-Score

### Updating Packages

If you need to install or update packages, use:

```bash
./update_packages.sh
```

## Project Structure

- `main.py` - Main script to fetch and process data
- `data_fetcher.py` - Functions to fetch crypto data from Binance API
- `technical_indicators.py` - Implementations of technical indicators
- `visualize_indicators.py` - Visualization of technical indicators
- `run.sh` - Helper script to run Python scripts with the correct version
- `update_packages.sh` - Helper script to install/update required packages

## Features

- Fetch real-time cryptocurrency price data from Binance
- Retrieve historical candlestick (OHLCV) data for multiple major cryptocurrencies
- Calculate over 30 technical indicators for market analysis
- Normalize data for machine learning applications
- Visualize price data with technical indicators
- Detect potential market inefficiencies (e.g., volume spikes, gaps, divergences)
- Framework for developing RL-based trading strategies

## Reinforcement Learning Approach

This project aims to develop RL agents that can:

1. **Detect Market Inefficiencies**: Use technical indicators to identify potential trading opportunities
2. **Learn Optimal Strategies**: Optimize entry/exit points and position sizing
3. **Adapt to Market Conditions**: Dynamically adjust to changing market regimes

The RL environment will be built on top of the data processing pipeline, using preprocessed and normalized technical indicators as input features.

## Authentication

This project uses SSH key-based authentication for Binance API access instead of traditional API key/secret pairs:

1. The system utilizes Ed25519 SSH keys for request signing
2. The SSH key from `~/.ssh/id_ed25519` is used by default
3. The API key is configured in the `config.env` file which is gitignored

This approach provides enhanced security compared to standard API key/secret authentication methods.

**Note:** The `config.env` file is included in `.gitignore` and should never be committed to the repository.

## Notes

- The code uses Binance's public API endpoints by default
- The SSH key-signing method is implemented in the `binance_api.py` module
- Generated image files are automatically excluded from git
