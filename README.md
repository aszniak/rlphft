# Reinforcement Learning Powered High Frequency Trading

This project applies reinforcement learning techniques to cryptocurrency trading using Binance market data.

## Prerequisites

### Trading Components

- Python 3.10 or higher
- Requests, NumPy, Pandas, Matplotlib
- PyTorch for reinforcement learning models
- Gymnasium for RL environments
- SSH key for Binance API authentication (optional)
- Weights & Biases for experiment tracking (optional)
- Santiment API for sentiment analysis (optional)

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

## Cryptocurrency Trading Implementation

This project implements a Proximal Policy Optimization (PPO) agent for cryptocurrency trading. The agent learns to make optimal trading decisions based on historical price data, technical indicators, and sentiment analysis.

### Running the Trading Agent

To run the cryptocurrency trading system:

```bash
# Run with default settings (will load saved model if available)
python main.py

# Train the model with multiple symbols
python main.py --symbols BTCUSDT ETHUSDT SOLUSDT --train

# Force training a new model even if one exists
python main.py --train

# Use fast training mode (fewer epochs for quick iterations)
python main.py --train --fast_train

# Use full training mode (more epochs for complete training)
python main.py --train --full_train

# Specify the number of days of historical data to use
python main.py --lookback_days 60

# Force refresh of data cache
python main.py --force_refresh

# Evaluate the model (loads existing model if available)
python main.py --evaluate

# Evaluate on a specific symbol
python main.py --eval_symbol ETHUSDT

# Evaluate on a random symbol
python main.py --eval_random

# Evaluate on all available symbols
python main.py --eval_all

# Set number of evaluation episodes
python main.py --episodes 10

# Specify initial crypto allocation for evaluation (0.0 to 1.0)
python main.py --initial_allocation 0.5

# Enable sentiment analysis (requires Santiment API key)
python main.py --sentiment

# Disable sentiment analysis
python main.py --no_sentiment

# Disable hardware optimization
python main.py --no_optimize

# Disable mixed precision training
python main.py --no_mixed_precision

# Enable Weights & Biases tracking
python main.py --wandb

# Test with Binance testnet (requires API setup)
python main.py --testnet
```

### Features of the Trading Implementation

- **Trading Environment**: A Gymnasium-compatible environment simulating cryptocurrency trading
- **PPO Agent**: Implementation of the PPO algorithm for trading optimization
- **Multi-Asset Support**: Support for training and evaluating on multiple cryptocurrencies
- **Configurable Parameters**: Comprehensive configuration system through `config.py`
- **Technical Indicators**: Automated calculation of technical indicators as features
- **Sentiment Analysis**: Integration with Santiment API for social and market sentiment data
- **Evaluation Framework**: Comparison with baseline strategies (Random, Buy & Hold)
- **Hardware Optimization**: Automatic GPU utilization and mixed precision training
- **Flexible Evaluation**: Options to evaluate on specific, random, or all available symbols
- **W&B Integration**: Optional tracking of training/evaluation metrics with Weights & Biases

## Python Setup

The project is compatible with Python 3.10 or higher. Make sure you have all the required dependencies installed.

## Sentiment Analysis with Santiment

The project integrates with the Santiment API to incorporate sentiment and social metrics into the trading model. This provides the agent with additional context beyond just price data and technical indicators.

### Setting Up Santiment API

1. Create an account at [Santiment](https://santiment.net/)
2. Get your API key from the dashboard
3. Add the API key to your `config.env` file:
   ```
   SANTIMENT_API_KEY=your_api_key_here
   ```

### Available Sentiment Metrics

The system collects the following metrics from Santiment:

- **Twitter Sentiment**: Overall sentiment of Twitter discussions about each cryptocurrency
- **Social Volume**: Amount of social media discussion about each cryptocurrency
- **Social Dominance**: Share of discussions relative to other cryptocurrencies
- **Developer Activity**: Code commit activity in GitHub repositories
- **Exchange Flow**: Cryptocurrency moving in and out of exchanges

These metrics are normalized and added as features to the agent's observation space.

## Project Structure

- **Main Scripts:**

  - `main.py` - Main script for cryptocurrency trading with RL

- **Agents and Environments:**

  - `agent.py` - Implementation of the PPO and Random agents
  - `trading_env.py` - Gymnasium-compatible trading environment with TradingEnv and BuyAndHoldEnv

- **Training and Evaluation:**

  - `training.py` - Functions for training the RL agent
  - `evaluation.py` - Functions for evaluating agents and testing with Binance

- **Data Processing:**

  - `utils.py` - Utility functions including data fetching and processing
  - `santiment_api.py` - Functions for fetching and processing sentiment data

- **Configuration:**
  - `config.py` - Centralized configuration system for the trading agent
  - `config.env` - Environment variables and API keys (gitignored)
  - `requirements.txt` - List of required Python packages

## Configuration System

The project uses a centralized configuration system in `config.py` that allows for:

- **Default Configuration**: Sensible defaults for all parameters
- **Environment Overrides**: Load values from `config.env` file
- **Command Line Arguments**: Override configuration via command line
- **Runtime Modifications**: Programmatically modify configuration as needed

The configuration includes settings for:

- **Data Fetching**: Trading symbols, timeframes, lookback periods
- **Environment Parameters**: Game length, window size, commission rates
- **Training Settings**: Episodes, epochs, steps per epoch
- **Evaluation Settings**: Number of episodes, initial balance
- **Sentiment Analysis**: API key, enabled metrics
- **Experiment Tracking**: W&B project and team settings

## Features

- Fetch real-time cryptocurrency price data from Binance
- Retrieve historical candlestick (OHLCV) data for multiple major cryptocurrencies
- Calculate over 30 technical indicators for market analysis
- Incorporate sentiment and social metrics from Santiment API
- Normalize data for machine learning applications
- Visualize price data with technical indicators
- Train RL agents to detect potential market inefficiencies and execute trades
- Evaluate trading strategies against baselines
- Track experiments with W&B (optional)

## Reinforcement Learning Approach

This project aims to develop RL agents that can:

1. **Detect Market Inefficiencies**: Use technical indicators to identify potential trading opportunities
2. **Analyze Sentiment**: Incorporate social and developer metrics to gauge market mood
3. **Learn Optimal Strategies**: Optimize entry/exit points and position sizing
4. **Adapt to Market Conditions**: Dynamically adjust to changing market regimes

The RL environment is built on top of the data processing pipeline, using preprocessed and normalized technical indicators as input features.

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
