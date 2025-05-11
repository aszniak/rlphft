import os
import argparse
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv
import matplotlib.pyplot as plt  # Keep for wandb visualization only
from tqdm import tqdm
import wandb
import time
import colorama
from colorama import Fore, Style

from config import Config, default_config
from data_fetcher import (
    fetch_training_data,
    prepare_multi_asset_dataset,
)
from agent import PPOAgent, RandomAgent
from trading_env import TradingEnv, BuyAndHoldEnv, TradingAction

# Initialize colorama for cross-platform color support
colorama.init()

# Load environment variables from config
load_dotenv("config.env")


def fetch_data(
    config=default_config,
    symbols=None,
    interval=None,
    lookback_days=None,
    force_refresh=None,
):
    """
    Fetch and prepare data for training or evaluation.

    Args:
        config: Configuration object
        symbols: List of cryptocurrency symbols to fetch (overrides config)
        interval: Data interval (overrides config)
        lookback_days: Days of historical data to fetch (overrides config)
        force_refresh: Whether to force refresh cached data (overrides config)

    Returns:
        Dictionary of processed data with technical indicators
    """
    # Use provided parameters or fall back to config values
    symbols = symbols or config.symbols
    interval = interval or config.interval
    lookback_days = lookback_days if lookback_days is not None else config.lookback_days
    force_refresh = force_refresh if force_refresh is not None else config.force_refresh

    print(
        f"{Fore.CYAN}🔍 Fetching training data for cryptocurrencies...{Style.RESET_ALL}"
    )
    data_dict = fetch_training_data(
        symbols=symbols,
        interval=interval,
        lookback_days=lookback_days,
        force_refresh=force_refresh,
    )

    # Prepare dataset with additional features
    print(
        f"{Fore.CYAN}📊 Adding technical indicators and normalizing features...{Style.RESET_ALL}"
    )
    enhanced_data, combined_df = prepare_multi_asset_dataset(
        data_dict, add_indicators=True
    )

    # Print summary of the data
    print(f"\n{Fore.GREEN}📈 Data Summary:{Style.RESET_ALL}")
    for symbol, df in enhanced_data.items():
        print(
            f"{Fore.YELLOW}{symbol}{Style.RESET_ALL}: {len(df)} candles from {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}"
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
    config=default_config,
    num_epochs=None,
    steps_per_epoch=None,
    initial_balance=None,
    model_save_path=None,
    use_wandb=None,
):
    """
    Train a PPO agent for trading with optional W&B tracking.

    Args:
        data_dict: Dictionary with processed price data
        symbol: Symbol to trade
        config: Configuration object
        num_epochs: Number of training epochs (overrides config)
        steps_per_epoch: Steps per epoch (overrides config)
        initial_balance: Initial trading balance (overrides config)
        model_save_path: Path to save the trained model
        use_wandb: Whether to use Weights & Biases for tracking (overrides config)

    Returns:
        Trained agent
    """
    # Use provided parameters or fall back to config values
    num_epochs = num_epochs if num_epochs is not None else config.epochs
    steps_per_epoch = (
        steps_per_epoch if steps_per_epoch is not None else config.steps_per_epoch
    )
    initial_balance = (
        initial_balance if initial_balance is not None else config.initial_balance
    )
    use_wandb = use_wandb if use_wandb is not None else config.use_wandb

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_team,
            name=f"ppo-{symbol}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "symbol": symbol,
                "epochs": num_epochs,
                "steps_per_epoch": steps_per_epoch,
                "initial_balance": initial_balance,
                "game_length": config.game_length,
                "model_type": "PPO",
                "window_size": config.window_size,
                "commission_rate": config.commission_rate,
            },
        )

    # Check if we have enough data
    if len(data_dict[symbol]) < config.game_length * 2:
        raise ValueError(
            f"Not enough data for {symbol}. Need at least {config.game_length * 2} candles."
        )

    # Setup environment
    env = TradingEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=config.game_length,
        window_size=config.window_size,
        commission_rate=config.commission_rate,
    )

    # Create agents
    feature_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPOAgent(feature_dim, action_dim)

    # Training loop
    print(
        f"{Fore.GREEN}🚀 Training on {symbol} for {num_epochs} epochs...{Style.RESET_ALL}"
    )
    epoch_rewards = []

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Collect trajectories and update agent
        states, actions, rewards, dones, next_states, log_probs, _ = (
            ppo_agent.collect_trajectories(env, steps_per_epoch)
        )

        ppo_agent.update(states, actions, rewards, dones, next_states, log_probs)

        # Track performance - handle empty episode lists safely
        if (
            hasattr(ppo_agent, "last_episode_rewards")
            and len(ppo_agent.last_episode_rewards) > 0
        ):
            avg_reward = np.mean(ppo_agent.last_episode_rewards)
        else:
            avg_reward = 0.0  # Default if no episodes completed

        epoch_rewards.append(avg_reward)

        # Get final portfolio value for this epoch
        portfolio_value = (
            ppo_agent.last_info["total_portfolio_value"]
            if hasattr(ppo_agent, "last_info") and ppo_agent.last_info is not None
            else None
        )

        # Log to W&B
        if use_wandb:
            metrics = {
                "epoch": epoch,
                "avg_reward": avg_reward,
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

    # Save the trained model
    if model_save_path:
        ppo_agent.save_model(model_save_path)
        print(f"{Fore.GREEN}💾 Model saved to {model_save_path}{Style.RESET_ALL}")

        # Upload model to W&B
        if use_wandb:
            wandb.save(f"{model_save_path}.pt")

    # Finish W&B run
    if use_wandb:
        wandb.finish()

    return ppo_agent


def evaluate_agents(
    data_dict,
    symbol,
    config=default_config,
    trained_agent=None,
    trained_model_path=None,
    episodes=None,
    initial_balance=None,
    use_wandb=None,
):
    """
    Evaluate and compare agents: trained agent vs random agent vs buy and hold.

    Args:
        data_dict: Dictionary with processed price data
        symbol: Symbol to trade
        config: Configuration object
        trained_agent: Pre-trained agent (if None, will load from trained_model_path)
        trained_model_path: Path to load a trained model
        episodes: Number of evaluation episodes (overrides config)
        initial_balance: Initial trading balance (overrides config)
        use_wandb: Whether to use Weights & Biases for tracking (overrides config)

    Returns:
        Dictionary with performance results
    """
    # Use provided parameters or fall back to config values
    episodes = episodes if episodes is not None else config.eval_episodes
    initial_balance = (
        initial_balance if initial_balance is not None else config.initial_balance
    )
    use_wandb = use_wandb if use_wandb is not None else config.use_wandb

    # Setup environments with fixed start for fair comparison
    env = TradingEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=config.game_length,
        window_size=config.window_size,
        commission_rate=config.commission_rate,
        random_start=False,
    )

    buyhold_env = BuyAndHoldEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=config.game_length,
        window_size=config.window_size,
        commission_rate=config.commission_rate,
        random_start=False,
    )

    # Load or use trained agent
    if trained_agent is None:
        if trained_model_path is None:
            trained_model_path = f"saved_model_{symbol}"

        if os.path.exists(f"{trained_model_path}.pt"):
            print(
                f"{Fore.CYAN}📂 Loading trained model from {trained_model_path}{Style.RESET_ALL}"
            )
            feature_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            trained_agent = PPOAgent.load_model(trained_model_path)
        else:
            raise ValueError(
                f"No trained model found at {trained_model_path}. Please train first."
            )

    # Create random agent
    random_agent = RandomAgent(env.action_space.n)

    # Evaluate agents
    print(f"\n{Fore.GREEN}🔍 Evaluation{Style.RESET_ALL}")

    # Use fixed seeds for reproducibility
    random_seed = 42

    # Random agent
    print(f"\n{Fore.YELLOW}🎲 Random Agent:{Style.RESET_ALL}")
    env.reset(seed=random_seed)
    random_perf = evaluate_agent(random_agent, env, episodes=episodes)
    random_return = (random_perf[-1] - random_perf[0]) / random_perf[0] * 100

    # Buy and hold
    print(f"\n{Fore.YELLOW}📈 Buy and Hold Strategy:{Style.RESET_ALL}")
    buyhold_env.reset(seed=random_seed)
    buyhold_perf = evaluate_agent(random_agent, buyhold_env, episodes=episodes)
    buyhold_return = (buyhold_perf[-1] - buyhold_perf[0]) / buyhold_perf[0] * 100

    # Trained agent
    print(f"\n{Fore.YELLOW}🤖 Trained PPO Agent:{Style.RESET_ALL}")
    env.reset(seed=random_seed)
    ppo_perf = evaluate_agent(trained_agent, env, episodes=episodes)
    ppo_return = (ppo_perf[-1] - ppo_perf[0]) / ppo_perf[0] * 100

    # Print summary
    print(f"\n{Fore.GREEN}📊 Performance Summary{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}🎲 Random Agent:{Style.RESET_ALL} Final return = {Fore.CYAN}{random_return:.2f}%{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}📈 Buy and Hold:{Style.RESET_ALL} Final return = {Fore.CYAN}{buyhold_return:.2f}%{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}🤖 Trained Agent:{Style.RESET_ALL} Final return = {Fore.CYAN}{ppo_return:.2f}%{Style.RESET_ALL}"
    )

    # Log comparison to W&B
    if use_wandb and not wandb.run:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_team,
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

        # Create a comparison plot and log it to wandb
        fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
        ax.plot(range(len(random_perf)), random_perf, "r-", label="Random")
        ax.plot(range(len(buyhold_perf)), buyhold_perf, "y-", label="Buy & Hold")
        ax.plot(range(len(ppo_perf)), ppo_perf, "g-", label="Trained Agent")
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        wandb.log({"performance_comparison": wandb.Image(fig)})
        plt.close(fig)  # Close the figure after logging to wandb

        wandb.finish()

    return {
        "random": {"values": random_perf, "return": random_return},
        "buyhold": {"values": buyhold_perf, "return": buyhold_return},
        "trained": {"values": ppo_perf, "return": ppo_return},
    }


def evaluate_agent(agent, env, episodes=1):
    """
    Evaluate an agent's performance over multiple episodes.

    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        episodes: Number of episodes to run

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

    # Return the average portfolio values for comparison
    return avg_portfolio_values


def test_with_binance(
    agent, symbol, config=default_config, test_duration=None, initial_balance=None
):
    """
    Test the agent with Binance testnet.
    This is a placeholder for future implementation.

    Args:
        agent: The trained agent
        symbol: Symbol to trade
        config: Configuration object
        test_duration: Duration of the test in minutes (overrides config)
        initial_balance: Initial balance (overrides config)

    Returns:
        Test results (placeholder)
    """
    # Use provided parameters or fall back to config values
    test_duration = test_duration if test_duration is not None else config.game_length
    initial_balance = (
        initial_balance if initial_balance is not None else config.initial_balance
    )

    print(
        f"{Fore.YELLOW}⚠️ Binance testnet integration not implemented yet.{Style.RESET_ALL}"
    )
    print(
        f"{Fore.CYAN}This will allow you to test your agent on the Binance testnet with simulated trading.{Style.RESET_ALL}"
    )
    print(
        f"{Fore.CYAN}Planned test: Trading {symbol} for {test_duration} minutes with ${initial_balance} initial balance.{Style.RESET_ALL}"
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
        "--lookback_days", type=int, help="Days of historical data to use"
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
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, help="Steps per epoch")
    parser.add_argument("--initial_balance", type=float, help="Initial balance")

    # Evaluation arguments
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of evaluation episodes",
    )

    # Misc arguments
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

    # Create config from defaults and override with args
    config = Config()

    # Override config with args if provided
    if args.symbol:
        config.symbols = [args.symbol]
    if args.lookback_days:
        config.lookback_days = args.lookback_days
    if args.force_refresh:
        config.force_refresh = args.force_refresh
    if args.epochs:
        config.epochs = args.epochs
    if args.steps_per_epoch:
        config.steps_per_epoch = args.steps_per_epoch
    if args.initial_balance:
        config.initial_balance = args.initial_balance
    if args.episodes:
        config.eval_episodes = args.episodes
    if args.wandb:
        config.use_wandb = True

    # Symbol to focus on (first in list)
    symbol = config.symbols[0]

    # Determine model path
    model_path = args.model_path if args.model_path else f"saved_model_{symbol}"
    model_exists = os.path.exists(f"{model_path}.pt")

    # Determine operation mode
    mode_train = args.train or not model_exists
    mode_evaluate = args.evaluate or (
        model_exists and not args.train and not args.testnet
    )
    mode_testnet = args.testnet

    # Always fetch data first
    enhanced_data = fetch_data(
        config=config,
        symbols=[symbol],
    )

    # Train if needed
    trained_agent = None
    if mode_train:
        print(f"\n{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🏋️ Training agent for {symbol}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        trained_agent = train_agent(
            enhanced_data,
            symbol,
            config=config,
            model_save_path=model_path,
        )

    # Evaluate
    if mode_evaluate:
        print(f"\n{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📊 Evaluating performance for {symbol}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        results = evaluate_agents(
            enhanced_data,
            symbol,
            config=config,
            trained_agent=trained_agent,
            trained_model_path=model_path,
        )

    # Test with Binance testnet
    if mode_testnet:
        print(f"\n{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        print(
            f"{Fore.GREEN}🌐 Testing with Binance testnet for {symbol}{Style.RESET_ALL}"
        )
        print(f"{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        if trained_agent is None:
            feature_dim = TradingEnv(
                enhanced_data, symbol, config.initial_balance
            ).observation_space.shape[0]
            action_dim = len(TradingAction)
            trained_agent = PPOAgent.load_model(model_path)

        testnet_results = test_with_binance(
            trained_agent,
            symbol,
            config=config,
        )

    print(f"\n{Fore.GREEN}✅ Done!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
