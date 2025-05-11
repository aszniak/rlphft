import os
import argparse
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import time

from data_fetcher import (
    fetch_training_data,
    prepare_multi_asset_dataset,
)
from agent import PPOAgent, RandomAgent
from trading_env import TradingEnv, BuyAndHoldEnv, TradingAction

# Load environment variables
load_dotenv("config.env")
API_KEY = os.getenv("API_KEY")

# ===== CONSTANTS =====
# W&B settings
WANDB_PROJECT = "crypto-trading-rl"
WANDB_TEAM = "aszczesniak-aszczesniak"

# Data settings
DEFAULT_INTERVAL = "1m"  # 1-minute candles

# Environment settings
GAME_LENGTH = 1440  # 24 hours of minute candles
WINDOW_SIZE = 30  # How many past candles to include in state
COMMISSION_RATE = 0.001  # 0.1% commission rate (Binance standard)

# Training parameters
EVAL_EPISODES = 5
STEPS_PER_EPOCH = 2000
DEFAULT_EPOCHS = 100
INITIAL_BALANCE = 10000.0

# Hardcoded list of top cryptocurrencies by volume
# These are the major coins with high liquidity
TOP_CRYPTOCURRENCIES = [
    "BTCUSDT",  # Bitcoin
    "ETHUSDT",  # Ethereum
    "BNBUSDT",  # Binance Coin
    "SOLUSDT",  # Solana
    "XRPUSDT",  # Ripple
]


def fetch_data(
    symbols=None, interval=DEFAULT_INTERVAL, lookback_days=30, force_refresh=False
):
    """
    Fetch and prepare data for training or evaluation.

    Args:
        symbols: List of cryptocurrency symbols to fetch
        interval: Data interval (e.g., "1m" for 1-minute candles)
        lookback_days: Days of historical data to fetch
        force_refresh: Whether to force refresh cached data

    Returns:
        Dictionary of processed data with technical indicators
    """
    if symbols is None:
        symbols = TOP_CRYPTOCURRENCIES

    print("Fetching training data for cryptocurrencies...")
    data_dict = fetch_training_data(
        symbols=symbols,
        interval=interval,
        lookback_days=lookback_days,
        force_refresh=force_refresh,
    )

    # Prepare dataset with additional features
    print("Adding technical indicators and normalizing features...")
    enhanced_data, combined_df = prepare_multi_asset_dataset(
        data_dict, add_indicators=True
    )

    # Print summary of the data
    print("\nData Summary:")
    for symbol, df in enhanced_data.items():
        print(
            f"{symbol}: {len(df)} candles from {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}"
        )

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

    return enhanced_data


def train_agent(
    data_dict,
    symbol,
    num_epochs=DEFAULT_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    initial_balance=INITIAL_BALANCE,
    model_save_path=None,
    display_plots=True,
    use_wandb=False,
):
    """
    Train a PPO agent for trading with optional W&B tracking.

    Args:
        data_dict: Dictionary with processed price data
        symbol: Symbol to trade
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        initial_balance: Initial trading balance
        model_save_path: Path to save the trained model
        display_plots: Whether to display performance plots
        use_wandb: Whether to use Weights & Biases for tracking

    Returns:
        Trained agent
    """
    # Initialize W&B if requested
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_TEAM,
            name=f"ppo-{symbol}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "symbol": symbol,
                "epochs": num_epochs,
                "steps_per_epoch": steps_per_epoch,
                "initial_balance": initial_balance,
                "game_length": GAME_LENGTH,
                "model_type": "PPO",
                "window_size": WINDOW_SIZE,
                "commission_rate": COMMISSION_RATE,
            },
        )

    # Check if we have enough data
    if len(data_dict[symbol]) < GAME_LENGTH * 2:
        raise ValueError(
            f"Not enough data for {symbol}. Need at least {GAME_LENGTH * 2} candles."
        )

    # Setup environment
    env = TradingEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=GAME_LENGTH,
        window_size=WINDOW_SIZE,
        commission_rate=COMMISSION_RATE,
    )

    # Create agents
    feature_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPOAgent(feature_dim, action_dim)

    # Setup visualization if needed
    if display_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Training Progress")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Reward")
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.ion()  # Turn on interactive mode

    # Training loop
    print(f"Training on {symbol} for {num_epochs} epochs...")
    epoch_rewards = []

    for epoch in tqdm(range(num_epochs)):
        # Collect trajectories and update agent
        states, actions, rewards, dones, next_states, log_probs, _ = (
            ppo_agent.collect_trajectories(env, steps_per_epoch)
        )

        ppo_agent.update(states, actions, rewards, dones, next_states, log_probs)

        # Track performance
        avg_reward = np.mean(
            ppo_agent.all_episode_rewards[-len(ppo_agent.last_episode_rewards) :]
        )
        epoch_rewards.append(avg_reward)

        # Get final portfolio value for this epoch
        portfolio_value = (
            ppo_agent.last_info["total_portfolio_value"]
            if hasattr(ppo_agent, "last_info")
            else None
        )

        # Log to W&B
        if use_wandb:
            metrics = {
                "epoch": epoch,
                "avg_reward": avg_reward,
                "episode_count": len(ppo_agent.last_episode_rewards),
            }

            # Add portfolio value if available
            if portfolio_value:
                metrics["portfolio_value"] = portfolio_value

            # Add action distribution
            if len(actions) > 0:
                action_counts = np.bincount(actions, minlength=env.action_space.n)
                action_freq = action_counts / len(actions)
                for i, freq in enumerate(action_freq):
                    action_name = TradingAction(i).name.lower()
                    metrics[f"action_freq_{action_name}"] = freq

            wandb.log(metrics)

        # Display progress
        if display_plots and (epoch + 1) % 10 == 0:
            ax.clear()
            ax.plot(range(1, len(epoch_rewards) + 1), epoch_rewards, "b-")
            ax.set_title(f"Training Progress - {symbol}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Reward")
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.draw()
            plt.pause(0.01)

    # Save the trained model
    if model_save_path:
        ppo_agent.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")

        # Upload model to W&B
        if use_wandb:
            wandb.save(f"{model_save_path}.pt")

    # Finish W&B run
    if use_wandb:
        wandb.finish()

    if display_plots:
        plt.ioff()

    return ppo_agent


def evaluate_agents(
    data_dict,
    symbol,
    trained_agent=None,
    trained_model_path=None,
    episodes=EVAL_EPISODES,
    initial_balance=INITIAL_BALANCE,
    display_plots=True,
    use_wandb=False,
):
    """
    Evaluate and compare agents: trained agent vs random agent vs buy and hold.

    Args:
        data_dict: Dictionary with processed price data
        symbol: Symbol to trade
        trained_agent: Pre-trained agent (if None, will load from trained_model_path)
        trained_model_path: Path to load a trained model
        episodes: Number of evaluation episodes
        initial_balance: Initial trading balance
        display_plots: Whether to display performance plots
        use_wandb: Whether to use Weights & Biases for tracking

    Returns:
        Dictionary with performance results
    """
    # Setup environments with fixed start for fair comparison
    env = TradingEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=GAME_LENGTH,
        window_size=WINDOW_SIZE,
        commission_rate=COMMISSION_RATE,
        random_start=False,
    )

    buyhold_env = BuyAndHoldEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=GAME_LENGTH,
        window_size=WINDOW_SIZE,
        commission_rate=COMMISSION_RATE,
        random_start=False,
    )

    # Load or use trained agent
    if trained_agent is None:
        if trained_model_path is None:
            trained_model_path = f"saved_model_{symbol}"

        if os.path.exists(f"{trained_model_path}.pt"):
            print(f"Loading trained model from {trained_model_path}")
            feature_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            trained_agent = PPOAgent.load_model(
                trained_model_path, feature_dim, action_dim
            )
        else:
            raise ValueError(
                f"No trained model found at {trained_model_path}. Please train first."
            )

    # Create random agent
    random_agent = RandomAgent(env.action_space.n)

    # Setup visualization
    if display_plots:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.tight_layout(pad=5.0)

        # Setup plot titles and labels
        axs[0].set_title("Random Agent Performance")
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Portfolio Value ($)")

        axs[1].set_title("Buy and Hold Performance")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("Portfolio Value ($)")

        axs[2].set_title("Trained Agent Performance")
        axs[2].set_xlabel("Step")
        axs[2].set_ylabel("Portfolio Value ($)")

        # Add grid to all plots
        for ax in axs:
            ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.ion()
    else:
        axs = [None, None, None]

    # Evaluate agents
    print("\n--- Evaluation ---")

    # Use fixed seeds for reproducibility
    random_seed = 42

    # Random agent
    print("\nRandom Agent:")
    env.reset(seed=random_seed)
    random_perf = evaluate_agent(random_agent, env, episodes=episodes, plot_ax=axs[0])
    random_return = (random_perf[-1] - random_perf[0]) / random_perf[0] * 100

    # Buy and hold
    print("\nBuy and Hold Strategy:")
    buyhold_env.reset(seed=random_seed)
    buyhold_perf = evaluate_agent(
        random_agent, buyhold_env, episodes=episodes, plot_ax=axs[1]
    )
    buyhold_return = (buyhold_perf[-1] - buyhold_perf[0]) / buyhold_perf[0] * 100

    # Trained agent
    print("\nTrained PPO Agent:")
    env.reset(seed=random_seed)
    ppo_perf = evaluate_agent(trained_agent, env, episodes=episodes, plot_ax=axs[2])
    ppo_return = (ppo_perf[-1] - ppo_perf[0]) / ppo_perf[0] * 100

    # Print summary
    print("\n--- Performance Summary ---")
    print(f"Random Agent: Final return = {random_return:.2f}%")
    print(f"Buy and Hold: Final return = {buyhold_return:.2f}%")
    print(f"Trained Agent: Final return = {ppo_return:.2f}%")

    if display_plots:
        plt.ioff()
        plt.show()

    # Log comparison to W&B
    if use_wandb and not wandb.run:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_TEAM,
            name=f"eval-{symbol}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={"symbol": symbol, "episodes": episodes},
        )

    # Log final evaluation metrics
    if use_wandb:
        # Log returns
        wandb.log(
            {
                "random_agent_return": random_return,
                "buy_hold_return": buyhold_return,
                "trained_agent_return": ppo_return,
            }
        )

        # Create a comparison plot and log it
        fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
        ax.plot(range(len(random_perf)), random_perf, "r-", label="Random")
        ax.plot(range(len(buyhold_perf)), buyhold_perf, "y-", label="Buy & Hold")
        ax.plot(range(len(ppo_perf)), ppo_perf, "g-", label="Trained Agent")
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        wandb.log({"performance_comparison": wandb.Image(fig)})

        wandb.finish()

    return {
        "random": {"values": random_perf, "return": random_return},
        "buyhold": {"values": buyhold_perf, "return": buyhold_return},
        "trained": {"values": ppo_perf, "return": ppo_return},
    }


def evaluate_agent(agent, env, episodes=1, plot_ax=None):
    """
    Evaluate an agent's performance over multiple episodes.

    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        episodes: Number of episodes to run
        plot_ax: Matplotlib axis for plotting results

    Returns:
        List of portfolio values at each step for each episode
    """
    all_portfolio_values = []
    all_timestamps = []

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        portfolio_values = [info["total_portfolio_value"]]
        timestamps = [info["timestamp"]]

        while not done:
            action, _ = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            portfolio_values.append(info["total_portfolio_value"])
            timestamps.append(info["timestamp"])

        all_portfolio_values.append(portfolio_values)
        all_timestamps.append(timestamps)

        print(f"Episode {ep+1}: Final portfolio value = ${portfolio_values[-1]:.2f}")

    # Calculate average portfolio value across episodes
    max_length = max(len(values) for values in all_portfolio_values)
    padded_values = []

    for values in all_portfolio_values:
        # Pad shorter episodes with their last value
        if len(values) < max_length:
            padded = values + [values[-1]] * (max_length - len(values))
        else:
            padded = values
        padded_values.append(padded)

    avg_portfolio_values = np.mean(padded_values, axis=0)

    # Plot results if axis provided
    if plot_ax is not None:
        plot_ax.clear()

        # Set title based on agent type
        if isinstance(agent, RandomAgent):
            plot_ax.set_title("Random Agent Performance")
        elif isinstance(env, BuyAndHoldEnv):
            plot_ax.set_title("Buy and Hold Performance")
        else:
            plot_ax.set_title("Trained Agent Performance")

        # Plot individual episodes in light color
        for i, values in enumerate(all_portfolio_values):
            plot_ax.plot(range(len(values)), values, alpha=0.3)

        # Plot average in bold
        plot_ax.plot(
            range(len(avg_portfolio_values)), avg_portfolio_values, "b-", linewidth=2
        )

        plot_ax.set_xlabel("Step")
        plot_ax.set_ylabel("Portfolio Value ($)")
        plot_ax.grid(True, linestyle="--", alpha=0.7)

    # Return the average portfolio values for comparison
    return avg_portfolio_values


def test_with_binance(
    agent, symbol, test_duration=GAME_LENGTH, initial_balance=INITIAL_BALANCE
):
    """
    Test the agent with Binance testnet.
    This is a placeholder for future implementation.

    Args:
        agent: The trained agent
        symbol: Symbol to trade
        test_duration: Duration of the test in minutes
        initial_balance: Initial balance

    Returns:
        Test results (placeholder)
    """
    print("Binance testnet integration not implemented yet.")
    print(
        "This will allow you to test your agent on the Binance testnet with simulated trading."
    )
    print(
        f"Planned test: Trading {symbol} for {test_duration} minutes with ${initial_balance} initial balance."
    )
    return {"status": "not_implemented"}


def main():
    parser = argparse.ArgumentParser(description="RL Trading Agent")

    # Data fetching arguments
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbol to trade (default: BTCUSDT)",
    )
    parser.add_argument(
        "--lookback_days", type=int, default=30, help="Days of historical data to use"
    )
    parser.add_argument(
        "--force_refresh", action="store_true", help="Force refresh data cache"
    )

    # Mode arguments
    parser.add_argument(
        "--train", action="store_true", help="Train a new model even if one exists"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model (default if model exists)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Test with Binance testnet (not implemented yet)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--steps_per_epoch", type=int, default=STEPS_PER_EPOCH, help="Steps per epoch"
    )
    parser.add_argument(
        "--initial_balance", type=float, default=INITIAL_BALANCE, help="Initial balance"
    )

    # Evaluation arguments
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVAL_EPISODES,
        help="Number of evaluation episodes",
    )

    # Misc arguments
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Custom model path (default: saved_model_{symbol})",
    )

    # Add W&B arg
    parser.add_argument(
        "--wandb", action="store_true", help="Track metrics using Weights & Biases"
    )

    args = parser.parse_args()

    # Determine model path
    model_path = args.model_path if args.model_path else f"saved_model_{args.symbol}"
    model_exists = os.path.exists(f"{model_path}.pt")

    # Determine operation mode
    mode_train = args.train or not model_exists
    mode_evaluate = args.evaluate or (
        model_exists and not args.train and not args.testnet
    )
    mode_testnet = args.testnet

    # Always fetch data first
    enhanced_data = fetch_data(
        symbols=[args.symbol],
        lookback_days=args.lookback_days,
        force_refresh=args.force_refresh,
    )

    # Train if needed
    trained_agent = None
    if mode_train:
        print(f"\n{'=' * 40}")
        print(f"Training agent for {args.symbol}")
        print(f"{'=' * 40}")
        trained_agent = train_agent(
            enhanced_data,
            args.symbol,
            num_epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            initial_balance=args.initial_balance,
            model_save_path=model_path,
            display_plots=not args.no_plots,
            use_wandb=args.wandb,
        )

    # Evaluate
    if mode_evaluate:
        print(f"\n{'=' * 40}")
        print(f"Evaluating performance for {args.symbol}")
        print(f"{'=' * 40}")
        results = evaluate_agents(
            enhanced_data,
            args.symbol,
            trained_agent=trained_agent,
            trained_model_path=model_path,
            episodes=args.episodes,
            initial_balance=args.initial_balance,
            display_plots=not args.no_plots,
            use_wandb=args.wandb,
        )

    # Test with Binance testnet
    if mode_testnet:
        print(f"\n{'=' * 40}")
        print(f"Testing with Binance testnet for {args.symbol}")
        print(f"{'=' * 40}")
        if trained_agent is None:
            feature_dim = TradingEnv(
                enhanced_data, args.symbol, args.initial_balance
            ).observation_space.shape[0]
            action_dim = len(TradingAction)
            trained_agent = PPOAgent.load_model(model_path, feature_dim, action_dim)

        testnet_results = test_with_binance(
            trained_agent,
            args.symbol,
            test_duration=GAME_LENGTH,
            initial_balance=args.initial_balance,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
