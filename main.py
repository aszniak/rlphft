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
import gym

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
    for_training=True,  # New parameter to indicate training vs evaluation
):
    """
    Fetch and prepare data for training or evaluation.

    Args:
        config: Configuration object
        symbols: List of cryptocurrency symbols to fetch (overrides config)
        interval: Data interval (overrides config)
        lookback_days: Days of historical data to fetch (overrides config)
        force_refresh: Whether to force refresh cached data (overrides config)
        for_training: Whether data is for training (True) or evaluation (False)

    Returns:
        Dictionary of processed data with technical indicators
    """
    # Use provided parameters or fall back to config values
    symbols = symbols or config.symbols
    interval = interval or config.interval

    # Choose appropriate lookback period based on whether we're training or evaluating
    if lookback_days is None:
        lookback_days = (
            config.training_lookback_days if for_training else config.eval_lookback_days
        )

    force_refresh = force_refresh if force_refresh is not None else config.force_refresh

    # For evaluation, we can select a random period if configured
    random_start_date = None
    if not for_training and config.random_eval_period:
        # Calculate a random start date within the last year (adjust as needed)
        # This gives us a random 30-day window for evaluation
        import datetime
        import random

        now = datetime.datetime.now()
        max_days_ago = 365 * 2  # Expand to 2 years maximum (was 365 days)
        min_days_ago = (
            config.eval_lookback_days + 30
        )  # At least 30 days after minimum lookback
        random_days = random.randint(min_days_ago, max_days_ago)
        random_start_date = now - datetime.timedelta(days=random_days)
        print(
            f"{Fore.CYAN}🎲 Using random evaluation period starting {random_days} days ago ({random_start_date.strftime('%Y-%m-%d')}){Style.RESET_ALL}"
        )

    print(
        f"{Fore.CYAN}🔍 Fetching {'training' if for_training else 'evaluation'} data for cryptocurrencies...{Style.RESET_ALL}"
    )

    # Update the fetch_training_data function call to include the random_start_date parameter
    data_dict = fetch_training_data(
        symbols=symbols,
        interval=interval,
        lookback_days=lookback_days,
        force_refresh=force_refresh
        or (
            not for_training and config.random_eval_period
        ),  # Force refresh for random eval periods
        start_date=random_start_date,
        cache_suffix=(
            f"_{random_days}days_ago" if random_start_date else None
        ),  # Add date-specific suffix to cache
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
    symbols,  # Changed from single symbol to list of symbols
    config=default_config,
    num_epochs=None,
    steps_per_epoch=None,
    initial_balance=None,
    model_save_path=None,
    use_wandb=None,
):
    """
    Train a PPO agent for trading with optional W&B tracking.
    Now supports training on multiple assets simultaneously.

    Args:
        data_dict: Dictionary with processed price data
        symbols: List of symbols to trade (e.g., ['BTCUSDT', 'ETHUSDT'])
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

    # Ensure symbols is a list
    if isinstance(symbols, str):
        symbols = [symbols]

    # Create a descriptive name for the model that includes all symbols
    symbols_string = "_".join([s.replace("USDT", "") for s in symbols])
    model_name = f"ppo-multi-{symbols_string}"

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_team,
            name=f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "symbols": symbols,
                "epochs": num_epochs,
                "steps_per_epoch": steps_per_epoch,
                "initial_balance": initial_balance,
                "game_length": config.game_length,
                "model_type": "PPO",
                "window_size": config.window_size,
                "commission_rate": config.commission_rate,
                "num_parallel_envs": config.num_parallel_envs,
            },
        )

    # Check if we have enough data for each symbol
    for symbol in symbols:
        if len(data_dict[symbol]) < config.game_length * 2:
            raise ValueError(
                f"Not enough data for {symbol}. Need at least {config.game_length * 2} candles."
            )

    # Create a factory for environments that randomly selects a symbol for each environment
    def make_env():
        # Randomly select a symbol from the list for this environment
        symbol = np.random.choice(symbols)
        return TradingEnv(
            data_dict=data_dict,
            symbol=symbol,
            initial_balance=initial_balance,
            game_length=config.game_length,
            window_size=config.window_size,
            commission_rate=config.commission_rate,
            random_start=True,
            full_data_evaluation=False,  # Training mode uses random segments
            max_episode_steps=config.max_episode_steps,  # Add maximum steps per episode
        )

    # Create one environment to get dimensions - use the first symbol as reference
    # All environments should have the same observation and action spaces
    env = TradingEnv(
        data_dict=data_dict,
        symbol=symbols[0],
        initial_balance=initial_balance,
        game_length=config.game_length,
        window_size=config.window_size,
        commission_rate=config.commission_rate,
    )
    feature_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent with config
    ppo_agent = PPOAgent(feature_dim, action_dim, config=config)

    # Training loop
    print(
        f"{Fore.GREEN}🚀 Training on {len(symbols)} assets ({', '.join(symbols)}) for {num_epochs} epochs using {config.num_parallel_envs} parallel environments...{Style.RESET_ALL}"
    )
    epoch_rewards = []

    # Create a tqdm progress bar
    progress_bar = tqdm(range(num_epochs), desc="Training Progress")
    action_names = [a.name for a in TradingAction]

    for epoch in progress_bar:
        # Collect trajectories using parallel environments with random asset selection
        states, actions, rewards, dones, next_states, log_probs, _ = (
            ppo_agent.collect_trajectories_parallel(
                make_env,
                n_envs=config.num_parallel_envs,
                steps_per_env=max(
                    config.max_episode_steps,
                    steps_per_epoch // config.num_parallel_envs,
                ),  # Ensure we collect at least one full episode
                disable_progress=True,  # Disable the inner progress bar
            )
        )

        # Update agent with collected data
        ppo_agent.update(
            states,
            actions,
            rewards,
            dones,
            next_states,
            log_probs,
            batch_size=config.batch_size,
        )

        # Track performance - handle empty episode lists safely
        if (
            hasattr(ppo_agent, "last_episode_rewards")
            and len(ppo_agent.last_episode_rewards) > 0
        ):
            avg_episode_reward = float(np.mean(ppo_agent.last_episode_rewards))
        else:
            avg_episode_reward = 0.0  # Default if no episodes completed

        # Also track average step reward since episodes may not complete in an epoch
        avg_step_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0

        epoch_rewards.append(avg_step_reward)

        # Get final portfolio value for this epoch
        portfolio_value = None
        if hasattr(ppo_agent, "last_info") and ppo_agent.last_info is not None:
            if "total_portfolio_value" in ppo_agent.last_info:
                portfolio_value = ppo_agent.last_info["total_portfolio_value"]
                # Convert to native Python float if it's a numpy value
                if isinstance(portfolio_value, (np.ndarray, np.number)):
                    portfolio_value = float(portfolio_value)

        # Calculate action distribution
        action_dist = {}
        if len(actions) > 0:
            action_counts = np.bincount(actions, minlength=env.action_space.n)
            action_freq = action_counts / len(actions)
            # Create a short representation of action distribution
            action_dist = {
                action_names[i][:3]: f"{float(freq):.1%}"
                for i, freq in enumerate(action_freq)
            }

        # Calculate completed episodes count
        completed_episodes = (
            int(len(ppo_agent.last_episode_rewards))
            if hasattr(ppo_agent, "last_episode_rewards")
            else 0
        )

        # Update progress bar with portfolio value and avg reward
        # We'll include just enough information to be useful without cluttering the display
        postfix_dict = {
            "port": (
                f"${float(np.mean(portfolio_value)):.2f}"
                if portfolio_value is not None
                else "N/A"
            ),
            "step_r": f"{avg_step_reward:.4f}",
            "eps": completed_episodes,
        }
        # Add action distribution to postfix
        postfix_dict.update(action_dist)

        # Update the progress bar
        progress_bar.set_postfix(postfix_dict)

        # Log to W&B
        if use_wandb:
            # Prepare metrics with explicit Python native types
            metrics = {
                "epoch": int(epoch),
                "avg_step_reward": float(avg_step_reward),
                "avg_episode_reward": float(avg_episode_reward),
                "completed_episodes": int(completed_episodes),
            }

            # Add portfolio value if available (convert to Python float)
            if portfolio_value is not None:
                # Handle both single value and array cases
                if isinstance(portfolio_value, (list, np.ndarray)):
                    # If we have multiple values from parallel envs, use the mean
                    metrics["portfolio_value"] = float(np.mean(portfolio_value))
                else:
                    metrics["portfolio_value"] = float(portfolio_value)

            # Add action distribution (convert to Python float)
            if len(actions) > 0:
                action_counts = np.bincount(actions, minlength=env.action_space.n)
                action_freq = action_counts / len(actions)
                for i, freq in enumerate(action_freq):
                    action_name = TradingAction(i).name.lower()
                    metrics[f"action_freq_{action_name}"] = float(freq)

            # Add total rewards (sum of all rewards in this batch)
            metrics["total_reward"] = (
                float(np.sum(rewards)) if len(rewards) > 0 else 0.0
            )

            # Add extra debug info
            metrics["rewards_min"] = float(np.min(rewards)) if len(rewards) > 0 else 0.0
            metrics["rewards_max"] = float(np.max(rewards)) if len(rewards) > 0 else 0.0
            metrics["rewards_nonzero"] = (
                int(np.count_nonzero(rewards)) if len(rewards) > 0 else 0
            )

            # Log explicitly converted metrics to W&B
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
    initial_allocation=0.0,  # Default 0% allocation to crypto
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
        initial_allocation: Initial allocation to crypto (0.0 to 1.0)

    Returns:
        Dictionary with performance results
    """
    # Use provided parameters or fall back to config values
    episodes = episodes if episodes is not None else config.eval_episodes
    initial_balance = (
        initial_balance if initial_balance is not None else config.initial_balance
    )
    use_wandb = use_wandb if use_wandb is not None else config.use_wandb

    # Calculate total dataset length for progress reporting
    dataset_length = len(data_dict[symbol]) - config.window_size
    print(
        f"Full dataset evaluation: {dataset_length} steps (approx. {dataset_length/1440:.1f} days)"
    )

    # Creating new environment for evaluation with appropriate state_dim
    env = TradingEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance
        * (1.0 - initial_allocation),  # Adjusted for allocation
        game_length=dataset_length,  # Use full data length
        window_size=config.window_size,
        commission_rate=config.commission_rate,
        random_start=False,
        full_data_evaluation=True,  # Evaluation mode uses full dataset
    )

    # If we have initial allocation, add crypto holdings after reset
    if initial_allocation > 0.0:
        print(
            f"Setting initial allocation: {initial_allocation*100:.1f}% in crypto, {(1.0-initial_allocation)*100:.1f}% in cash"
        )
        state, info = env.reset()

        # Calculate how much crypto to buy
        crypto_value = initial_balance * initial_allocation
        price = data_dict[symbol].iloc[env.current_step]["close"]
        crypto_amount = crypto_value / price

        # Manually set the environment state
        env.crypto_holdings = crypto_amount
        env.crypto_value = crypto_value
        env._update_portfolio_value()  # Recalculate portfolio value

        # Reset the environment state
        env.portfolio_history = [env.total_portfolio_value]
        env.previous_portfolio_value = env.total_portfolio_value

        # We need to reset env again to get clean state
        env.reset()

    # Create buy-and-hold environment with same settings
    buyhold_env = BuyAndHoldEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=len(data_dict[symbol]) - config.window_size,  # Use full data length
        window_size=config.window_size,
        commission_rate=config.commission_rate,
        random_start=False,
        full_data_evaluation=True,  # Evaluation mode uses full dataset
    )

    # Load or use trained agent
    if trained_agent is None:
        if trained_model_path is None:
            trained_model_path = f"saved_model_{symbol}"

        if os.path.exists(f"{trained_model_path}.pt"):
            print(
                f"{Fore.CYAN}📂 Loading trained model from {trained_model_path}{Style.RESET_ALL}"
            )

            # Check state dimensions and adjust if needed
            feature_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            # Load the model to check its state_dim
            checkpoint = torch.load(f"{trained_model_path}.pt", weights_only=False)
            model_state_dim = checkpoint["state_dim"]

            if model_state_dim != feature_dim:
                print(
                    f"{Fore.YELLOW}⚠️ State dimension mismatch: model expects {model_state_dim}, environment has {feature_dim}{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.YELLOW}Adjusting environment window_size to match model...{Style.RESET_ALL}"
                )

                # Recalculate window_size based on feature columns count
                feature_columns_count = len(env.feature_columns)

                if model_state_dim % feature_columns_count == 0:
                    # Perfect division case
                    new_window_size = model_state_dim // feature_columns_count
                    print(
                        f"{Fore.CYAN}Calculated new window_size: {new_window_size} based on {feature_columns_count} features{Style.RESET_ALL}"
                    )
                else:
                    # Imperfect division - we need to be flexible
                    print(
                        f"{Fore.YELLOW}Feature count ({feature_columns_count}) doesn't divide model dimension ({model_state_dim}) evenly{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.YELLOW}Using a flexible approach to match dimensions...{Style.RESET_ALL}"
                    )

                    # Try to find the closest window size
                    closest_window_size = round(model_state_dim / feature_columns_count)
                    expected_dim = closest_window_size * feature_columns_count
                    print(
                        f"{Fore.CYAN}Using window_size: {closest_window_size} which gives dimension: {expected_dim}{Style.RESET_ALL}"
                    )
                    new_window_size = closest_window_size

                # Recreate environments with adjusted window size
                env = TradingEnv(
                    data_dict=data_dict,
                    symbol=symbol,
                    initial_balance=initial_balance * (1.0 - initial_allocation),
                    game_length=dataset_length,
                    window_size=new_window_size,  # Updated window size
                    commission_rate=config.commission_rate,
                    random_start=False,
                    full_data_evaluation=True,
                )

                buyhold_env = BuyAndHoldEnv(
                    data_dict=data_dict,
                    symbol=symbol,
                    initial_balance=initial_balance,
                    game_length=len(data_dict[symbol]) - new_window_size,
                    window_size=new_window_size,  # Updated window size
                    commission_rate=config.commission_rate,
                    random_start=False,
                    full_data_evaluation=True,
                )

                # Verify adjustment worked
                new_feature_dim = env.observation_space.shape[0]
                if new_feature_dim != model_state_dim:
                    print(
                        f"{Fore.RED}❌ Dimension mismatch persists. Model: {model_state_dim}, Environment: {new_feature_dim}{Style.RESET_ALL}"
                    )

                    # Fall back to forcing exact dimension match by modifying the environment
                    print(
                        f"{Fore.YELLOW}Attempting to force dimension compatibility...{Style.RESET_ALL}"
                    )

                    # Create a new environment class that can adapt to the model's dimension
                    class DimensionAdaptedEnv(TradingEnv):
                        _adaptation_message_shown = (
                            False  # Class variable instead of instance variable
                        )

                        def _get_state(self):
                            """Modified state function that adjusts dimensions to match the model"""
                            original_state = super()._get_state()

                            # If dimensions match exactly, return as is
                            if len(original_state) == model_state_dim:
                                return original_state

                            # If original state is larger, truncate
                            if len(original_state) > model_state_dim:
                                if not DimensionAdaptedEnv._adaptation_message_shown:
                                    print(
                                        f"{Fore.GREEN}✅ Adapting state by truncating from {len(original_state)} to {model_state_dim}{Style.RESET_ALL}"
                                    )
                                    DimensionAdaptedEnv._adaptation_message_shown = True
                                return original_state[:model_state_dim]

                            # If original state is smaller, pad with zeros
                            if len(original_state) < model_state_dim:
                                padding = np.zeros(
                                    model_state_dim - len(original_state)
                                )
                                if not DimensionAdaptedEnv._adaptation_message_shown:
                                    print(
                                        f"{Fore.GREEN}✅ Adapting state by padding from {len(original_state)} to {model_state_dim}{Style.RESET_ALL}"
                                    )
                                    DimensionAdaptedEnv._adaptation_message_shown = True
                                return np.concatenate([original_state, padding])

                    # Create adapted environments
                    env = DimensionAdaptedEnv(
                        data_dict=data_dict,
                        symbol=symbol,
                        initial_balance=initial_balance * (1.0 - initial_allocation),
                        game_length=dataset_length,
                        window_size=new_window_size,
                        commission_rate=config.commission_rate,
                        random_start=False,
                        full_data_evaluation=True,
                    )

                    class DimensionAdaptedBuyHoldEnv(
                        DimensionAdaptedEnv, BuyAndHoldEnv
                    ):
                        pass

                    buyhold_env = DimensionAdaptedBuyHoldEnv(
                        data_dict=data_dict,
                        symbol=symbol,
                        initial_balance=initial_balance,
                        game_length=len(data_dict[symbol]) - new_window_size,
                        window_size=new_window_size,
                        commission_rate=config.commission_rate,
                        random_start=False,
                        full_data_evaluation=True,
                    )

                    # Override observation space to match model
                    env.observation_space = gym.spaces.Box(
                        low=0, high=1, shape=(model_state_dim,), dtype=np.float32
                    )
                    buyhold_env.observation_space = gym.spaces.Box(
                        low=0, high=1, shape=(model_state_dim,), dtype=np.float32
                    )

                    print(
                        f"{Fore.GREEN}✅ Successfully forced dimension compatibility{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"{Fore.GREEN}✅ Successfully adjusted window size to {new_window_size}{Style.RESET_ALL}"
                    )

            # Now load the model with correct dimensions
            trained_agent = PPOAgent.load_model(trained_model_path)
            if trained_agent is None:
                raise ValueError(f"Failed to load the model from {trained_model_path}")
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
    Evaluate an agent's performance.
    For full data evaluation, only runs one episode through the entire dataset.
    """
    if env.full_data_evaluation:
        # For full dataset evaluation, we only run one episode
        state, info = env.reset()
        done = False
        portfolio_values = [info["total_portfolio_value"]]
        timestamps = [info["timestamp"]]

        # For debugging - track actions
        action_counts = [0, 0, 0, 0, 0]  # Counts for each action type
        balance_history = [info["account_balance"]]
        holdings_history = [info["crypto_holdings"]]

        # Create progress bar for long evaluation
        total_steps = env.end_index - env.current_step
        progress_bar = tqdm(
            total=total_steps, desc=f"Evaluating on full dataset", unit="step"
        )

        step_count = 0
        while not done:
            action, _ = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track statistics
            action_counts[action] += 1
            balance_history.append(info["account_balance"])
            holdings_history.append(info["crypto_holdings"])

            portfolio_values.append(info["total_portfolio_value"])
            timestamps.append(info["timestamp"])

            step_count += 1
            progress_bar.update(1)

            # Update progress bar with current portfolio value
            if step_count % 100 == 0:  # Update less frequently to reduce overhead
                progress_bar.set_postfix({"portfolio": f"${portfolio_values[-1]:.2f}"})

        progress_bar.close()

        # Print action distribution
        total_actions = sum(action_counts)
        action_names = ["DO_NOTHING", "SELL_20%", "SELL_10%", "BUY_10%", "BUY_20%"]
        print("\nAction distribution:")
        for i, count in enumerate(action_counts):
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            print(f"  {action_names[i]}: {count} ({percentage:.1f}%)")

        # Calculate trading statistics
        avg_balance = (
            sum(balance_history) / len(balance_history) if balance_history else 0
        )
        final_balance = balance_history[-1] if balance_history else 0
        max_holdings = max(holdings_history) if holdings_history else 0
        final_holdings = holdings_history[-1] if holdings_history else 0

        print(f"Trading statistics:")
        print(f"  Initial balance: ${balance_history[0]:.2f}")
        print(f"  Final balance: ${final_balance:.2f}")
        print(f"  Final holdings: {final_holdings:.6f}")
        print(f"  Max holdings: {max_holdings:.6f}")

        print(
            f"Full dataset evaluation: Final portfolio value = ${portfolio_values[-1]:.2f} after {step_count} steps"
        )
        return portfolio_values
    else:
        # Original code for multiple episode evaluation
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

            print(
                f"Episode {ep+1}: Final portfolio value = ${portfolio_values[-1]:.2f}"
            )

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
        "--symbols",
        type=str,
        nargs="+",
        default=["BTCUSDT"],
        help="Symbols to trade (default: BTCUSDT). Can provide multiple symbols separated by spaces.",
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
    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        help="Number of parallel environments for training",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for PPO updates")
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--gamma", type=float, help="Discount factor for future rewards"
    )
    parser.add_argument(
        "--ppo_epochs", type=int, help="Number of PPO epochs per update"
    )

    # Evaluation arguments
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--random_eval",
        action="store_true",
        help="Use a random 30-day period for evaluation (different from training data)",
    )
    parser.add_argument(
        "--fixed_eval",
        action="store_true",
        help="Use the most recent 30-day period for evaluation (disables random evaluation)",
    )
    parser.add_argument(
        "--eval_symbol",
        type=str,
        help="Symbol to use for evaluation (if different from training symbols)",
    )
    parser.add_argument(
        "--eval_random",
        action="store_true",
        help="Evaluate on a random symbol from the available symbols",
    )
    parser.add_argument(
        "--eval_all",
        action="store_true",
        help="Evaluate on all available symbols",
    )

    # Misc arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Custom model path (default: saved_model_crypto)",
    )

    # Add W&B arg
    parser.add_argument(
        "--wandb", action="store_true", help="Track metrics using Weights & Biases"
    )

    # Add arg to disable parallel environments
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Disable parallel environments and use a single environment",
    )

    # Add arg for initial allocation
    parser.add_argument(
        "--initial_allocation",
        type=float,
        default=0.0,
        help="Initial allocation to crypto (0.0 to 1.0) for evaluation",
    )

    args = parser.parse_args()

    # Create config from defaults and override with args
    config = Config()

    # Override config with args if provided
    if args.symbols:
        config.symbols = args.symbols
    if args.lookback_days:
        config.training_lookback_days = args.lookback_days
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
    if args.num_parallel_envs:
        config.num_parallel_envs = args.num_parallel_envs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.gamma:
        config.gamma = args.gamma
    if args.ppo_epochs:
        config.ppo_epochs = args.ppo_epochs
    # Handle evaluation period options
    if args.random_eval:
        config.random_eval_period = True
    if args.fixed_eval:
        config.random_eval_period = False
    # Disable parallel environments if requested
    if args.no_parallel:
        config.num_parallel_envs = 1

    # Determine which symbols to use for training
    training_symbols = config.symbols

    # Available symbols for evaluation (use the ones specified in args or config)
    available_symbols = config.symbols

    # For evaluation, determine which symbols to evaluate on
    eval_symbols = []

    if args.eval_symbol:
        # Specific symbol provided via command line
        eval_symbols = [args.eval_symbol]
    elif args.eval_random:
        # Random symbol selection
        import random

        eval_symbols = [random.choice(available_symbols)]
        print(
            f"{Fore.CYAN}🎲 Randomly selected {eval_symbols[0]} for evaluation{Style.RESET_ALL}"
        )
    elif args.eval_all:
        # All available symbols
        eval_symbols = available_symbols.copy()
    else:
        # Default to first training symbol
        eval_symbols = [training_symbols[0]]

    # Generate model path based on symbols if not provided
    if not args.model_path:
        # Use a generic model name instead of making it currency-dependent
        model_path = "saved_model_crypto"
    else:
        model_path = args.model_path

    model_exists = os.path.exists(f"{model_path}.pt")

    # Initialize trained_agent to None
    trained_agent = None

    # Determine operation mode - prioritize explicit flags
    # First check for explicit evaluation flags
    mode_evaluate = (
        args.evaluate or args.eval_random or args.eval_all or args.eval_symbol
    )

    # Only train if explicitly requested or if no model exists AND no evaluation flags
    mode_train = args.train or (not model_exists and not mode_evaluate)

    mode_testnet = args.testnet

    # Print selected modes for debugging
    if mode_train:
        print(f"{Fore.CYAN}ℹ️ Mode: Training enabled{Style.RESET_ALL}")
    if mode_evaluate:
        print(
            f"{Fore.CYAN}ℹ️ Mode: Evaluation enabled on {len(eval_symbols)} symbol(s): {', '.join(eval_symbols)}{Style.RESET_ALL}"
        )
    if mode_testnet:
        print(f"{Fore.CYAN}ℹ️ Mode: Testnet enabled{Style.RESET_ALL}")

    # Fetch training data for all symbols
    if mode_train:
        print(f"\n{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        print(
            f"{Fore.GREEN}🏋️ Training agent on {len(training_symbols)} symbols: {', '.join(training_symbols)}{Style.RESET_ALL}"
        )
        print(f"{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")

        training_data = fetch_data(
            config=config, symbols=training_symbols, for_training=True
        )

        trained_agent = train_agent(
            training_data,
            training_symbols,
            config=config,
            model_save_path=model_path,
        )

    # Fetch evaluation data separately - just for the eval symbol
    if mode_evaluate:
        print(f"\n{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        print(
            f"{Fore.GREEN}📊 Evaluating performance on {', '.join(eval_symbols)}{Style.RESET_ALL}"
        )
        print(f"{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")

        # Check if model exists before proceeding
        if not model_exists:
            print(
                f"{Fore.RED}❌ Error: No trained model found at {model_path}. Cannot evaluate.{Style.RESET_ALL}"
            )
            print(
                f"{Fore.YELLOW}Hint: Use --train flag to train a model first, or specify a different model with --model_path.{Style.RESET_ALL}"
            )
            return  # Exit the function early

        # Fetch data only for the symbols we need to evaluate
        print(
            f"{Fore.CYAN}Fetching data specifically for evaluation on: {', '.join(eval_symbols)}{Style.RESET_ALL}"
        )
        eval_data = fetch_data(config=config, symbols=eval_symbols, for_training=False)

        # Store results for all symbols
        all_results = {}

        # Evaluate on each symbol
        for current_symbol in eval_symbols:
            print(f"\n{Fore.YELLOW}Evaluating on {current_symbol}:{Style.RESET_ALL}")

            # Evaluate the current symbol
            results = evaluate_agents(
                eval_data,
                current_symbol,
                config=config,
                trained_agent=trained_agent,
                trained_model_path=model_path,
                initial_allocation=args.initial_allocation,
            )

            # Store results
            all_results[current_symbol] = results

        # Print comparative summary if evaluating on multiple symbols
        if len(eval_symbols) > 1:
            print(f"\n{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}📊 Comparative Performance Summary{Style.RESET_ALL}")

            print(
                f"{'Symbol':<10} {'Random %':<10} {'Buy & Hold %':<15} {'Trained Agent %':<15} {'vs Buy & Hold':<12}"
            )
            print(f"{'-'*70}")

            for symbol, result in all_results.items():
                random_return = result["random"]["return"]
                buyhold_return = result["buyhold"]["return"]
                trained_return = result["trained"]["return"]
                performance_vs_buyhold = trained_return - buyhold_return

                # Color code the comparison
                comparison_color = (
                    Fore.GREEN if performance_vs_buyhold > 0 else Fore.RED
                )

                print(
                    f"{symbol:<10} {random_return:>8.2f}% {buyhold_return:>13.2f}% {trained_return:>13.2f}% {comparison_color}{performance_vs_buyhold:>+10.2f}%{Style.RESET_ALL}"
                )

    # Test with Binance testnet
    if mode_testnet:
        print(f"\n{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        print(
            f"{Fore.GREEN}🌐 Testing with Binance testnet for {eval_symbols[0]}{Style.RESET_ALL}"
        )
        print(f"{Fore.GREEN}{'=' * 40}{Style.RESET_ALL}")
        if trained_agent is None:
            feature_dim = TradingEnv(
                eval_data, eval_symbols[0], config.initial_balance
            ).observation_space.shape[0]
            action_dim = len(TradingAction)
            trained_agent = PPOAgent.load_model(model_path)

        testnet_results = test_with_binance(
            trained_agent,
            eval_symbols[0],
            config=config,
        )

    print(f"\n{Fore.GREEN}✅ Done!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
