import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import gymnasium as gym

from agent import PPOAgent, RandomAgent
from trading_env import TradingEnv, BuyAndHoldEnv, TradingAction
from data_fetcher import fetch_training_data, prepare_multi_asset_dataset

# Hyperparameters
STEPS_PER_EPOCH = 2000  # Number of environment steps per training epoch
NUM_EPOCHS = 100  # Default number of training epochs
EVAL_EPISODES = 5  # Number of episodes for evaluation


def setup_plots():
    """Initialize plots for visualization"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.tight_layout(pad=5.0)

    # Random agent performance
    axs[0].set_title("Random Agent Performance")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Portfolio Value ($)")

    # Buy and hold performance
    axs[1].set_title("Buy and Hold Performance")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Portfolio Value ($)")

    # Trained agent performance
    axs[2].set_title("Trained Agent Performance")
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Portfolio Value ($)")

    # Add grid to all plots
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig, axs


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


def train_trading_agent(
    data_dict,
    symbol="BTCUSDT",
    num_epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    game_length=1440,  # 24 hours of minute candles
    initial_balance=10000.0,
    display_plots=True,
):
    """
    Train a PPO agent for trading.

    Args:
        data_dict: Dictionary with price data from data_fetcher
        symbol: Symbol to trade
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        game_length: Length of each trading episode
        initial_balance: Initial account balance
        display_plots: Whether to display performance plots

    Returns:
        Trained agent and evaluation results
    """
    # Check if we have enough data
    if len(data_dict[symbol]) < game_length * 2:
        raise ValueError(
            f"Not enough data for {symbol}. Need at least {game_length * 2} candles."
        )

    # Setup environment
    env = TradingEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=game_length,
    )

    # Create baseline environments
    random_env = TradingEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=game_length,
        random_start=False,  # Fixed start for fair comparison
    )

    buyhold_env = BuyAndHoldEnv(
        data_dict=data_dict,
        symbol=symbol,
        initial_balance=initial_balance,
        game_length=game_length,
        random_start=False,  # Fixed start for fair comparison
    )

    # Create agents
    feature_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPOAgent(feature_dim, action_dim)
    random_agent = RandomAgent(action_dim)

    # Setup visualization
    if display_plots:
        fig, axs = setup_plots()
        plt.ion()  # Turn on interactive mode
    else:
        fig, axs = None, [None, None, None]

    # Training loop
    print(f"Training on {symbol} for {num_epochs} epochs...")

    for epoch in tqdm(range(num_epochs)):
        # Collect trajectories and update agent
        states, actions, rewards, dones, next_states, log_probs, _ = (
            ppo_agent.collect_trajectories(env, steps_per_epoch)
        )

        ppo_agent.update(states, actions, rewards, dones, next_states, log_probs)

        # Periodically evaluate performance
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"\nEvaluation after epoch {epoch + 1}:")

            # Evaluate random agent
            print("\nRandom Agent:")
            random_perf = evaluate_agent(
                random_agent,
                random_env.reset()[0],
                episodes=EVAL_EPISODES,
                plot_ax=axs[0],
            )

            # Evaluate buy and hold
            print("\nBuy and Hold Strategy:")
            buyhold_perf = evaluate_agent(
                random_agent,
                buyhold_env.reset()[0],
                episodes=EVAL_EPISODES,
                plot_ax=axs[1],
            )

            # Evaluate trained agent
            print("\nTrained PPO Agent:")
            ppo_perf = evaluate_agent(
                ppo_agent, random_env.reset()[0], episodes=EVAL_EPISODES, plot_ax=axs[2]
            )

            if display_plots:
                plt.draw()
                plt.pause(0.1)

    # Final evaluation with more episodes
    print("\n--- Final Evaluation ---")

    # Random agent
    print("\nRandom Agent:")
    random_env.reset(seed=42)[0]  # Use fixed seed for final evaluation
    random_perf = evaluate_agent(
        random_agent, random_env, episodes=EVAL_EPISODES, plot_ax=axs[0]
    )
    random_return = (random_perf[-1] - random_perf[0]) / random_perf[0] * 100

    # Buy and hold
    print("\nBuy and Hold Strategy:")
    buyhold_env.reset(seed=42)[0]  # Use fixed seed for final evaluation
    buyhold_perf = evaluate_agent(
        random_agent, buyhold_env, episodes=EVAL_EPISODES, plot_ax=axs[1]
    )
    buyhold_return = (buyhold_perf[-1] - buyhold_perf[0]) / buyhold_perf[0] * 100

    # Trained agent
    print("\nTrained PPO Agent:")
    random_env.reset(seed=42)[0]  # Use fixed seed for final evaluation
    ppo_perf = evaluate_agent(
        ppo_agent, random_env, episodes=EVAL_EPISODES, plot_ax=axs[2]
    )
    ppo_return = (ppo_perf[-1] - ppo_perf[0]) / ppo_perf[0] * 100

    # Print summary
    print("\n--- Performance Summary ---")
    print(f"Random Agent: Final return = {random_return:.2f}%")
    print(f"Buy and Hold: Final return = {buyhold_return:.2f}%")
    print(f"Trained Agent: Final return = {ppo_return:.2f}%")

    if display_plots:
        plt.ioff()
        plt.show()

    return ppo_agent, (random_perf, buyhold_perf, ppo_perf)


def main():
    parser = argparse.ArgumentParser(description="Train a trading agent using RL")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to trade")
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--lookback_days", type=int, default=30, help="Days of historical data to use"
    )
    parser.add_argument(
        "--balance", type=float, default=10000.0, help="Initial balance"
    )
    parser.add_argument(
        "--force_refresh", action="store_true", help="Force refresh data cache"
    )
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    # Fetch training data
    print(f"Fetching data for {args.symbol}...")
    data_dict = fetch_training_data(
        symbols=[args.symbol],
        interval="1m",
        lookback_days=args.lookback_days,
        force_refresh=args.force_refresh,
    )

    # Prepare dataset with additional features
    print("Preparing dataset with technical indicators...")
    enhanced_data, _ = prepare_multi_asset_dataset(data_dict, add_indicators=True)

    # Train agent
    trained_agent, performance = train_trading_agent(
        enhanced_data,
        symbol=args.symbol,
        num_epochs=args.epochs,
        initial_balance=args.balance,
        display_plots=not args.no_plots,
    )

    # Save trained model
    trained_agent.save_model(f"saved_model_{args.symbol}")
    print(f"Model saved as saved_model_{args.symbol}")


if __name__ == "__main__":
    main()
