# Reinforcement Learning Powered High Frequency Trading

This project applies reinforcement learning techniques to cryptocurrency trading using Binance market data.

## Prerequisites

### Cryptocurrency Trading Components

- Python 3.10 or higher
- Requests, NumPy, Pandas, Matplotlib
- SSH key for Binance API authentication (optional)

### CartPole RL Components

- PyTorch 2.0+
- Gymnasium 0.29+
- OpenCV Python
- tqdm for progress tracking

You can install all dependencies with:

```bash
# Install for system Python
./update_packages.sh

# Or use a virtual environment for the RL components
./update_packages.sh --venv
```

## CartPole Implementation

Before applying RL to the more complex domain of cryptocurrency trading, we've implemented a Proximal Policy Optimization (PPO) agent to solve the classic CartPole control problem. This serves as a testbed for our RL algorithms and visualization techniques.

### Running the CartPole Environment

For convenience, use the provided shell script:

```bash
# Run with default settings (will load saved model if available)
./run_cartpole.sh

# Force training a new model even if one exists
./run_cartpole.sh --train

# Specify a custom path for saving/loading the model
./run_cartpole.sh --model_path my_model

# Set the number of training epochs
./run_cartpole.sh --train --epochs 100

# Set the number of evaluation episodes
./run_cartpole.sh --eval_episodes 30
```

Alternatively, you can run the Python script directly:

```bash
python cartpole.py [arguments]
```

### Features of the CartPole Implementation

- **PPO Implementation**: Efficient implementation of the PPO algorithm
- **Model Saving/Loading**: Save trained models and load them later to avoid retraining
- **Visualization**: Three-panel visualization showing:
  - Random agent performance
  - Training progress of the PPO agent
  - Trained agent performance
- **Progress Tracking**: Uses tqdm for clear progress visualization during training

### Files for CartPole

- `cartpole.py` - Main file for running the CartPole environment with visualization
- `agent.py` - Implementation of the PPO agent and policy/value networks

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

- **Main Scripts:**

  - `main.py` - Main script to fetch and process cryptocurrency data
  - `cartpole.py` - Main file for running the CartPole environment with visualization

- **Agents and Models:**

  - `agent.py` - Implementation of the PPO agent and policy/value networks for CartPole

- **Data Processing:**

  - `data_fetcher.py` - Functions to fetch crypto data from Binance API
  - `technical_indicators.py` - Implementations of technical indicators
  - `visualize_indicators.py` - Visualization of technical indicators

- **Helper Scripts:**

  - `run.sh` - Helper script to run Python scripts with the correct version
  - `run_cartpole.sh` - Helper script for running the CartPole environment
  - `run_visualization.sh` - Script for visualizing technical indicators
  - `update_packages.sh` - Helper script to install/update required packages

- **Configuration:**
  - `requirements.txt` - List of required Python packages
  - `config.env` - Configuration file for API keys (gitignored)

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
