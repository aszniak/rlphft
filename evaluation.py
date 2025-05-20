import os
import numpy as np
import torch
import wandb
from tqdm import tqdm
import colorama
from colorama import Fore, Style
import gym

from trading_env import TradingEnv, BuyAndHoldEnv, TradingAction
from agent import PPOAgent, RandomAgent
from visualization import (
    init_wandb_for_evaluation,
    log_evaluation_results,
    finish_wandb,
)


def evaluate_agents(
    data_dict,
    symbol,
    config,
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
            try:
                # First try to load the model with PPOAgent.load_model
                trained_agent = PPOAgent.load_model(trained_model_path)

                if trained_agent is None:
                    raise ValueError(f"PPOAgent.load_model returned None")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating a new model with matching architecture...")

                # Extract hidden_dim from the checkpoint
                if "policy_state_dict" in checkpoint:
                    # Get the shape of the first layer weight
                    first_layer_shape = checkpoint["policy_state_dict"][
                        "net.0.weight"
                    ].shape
                    if len(first_layer_shape) > 0:
                        hidden_dim = first_layer_shape[
                            0
                        ]  # Hidden dim is output size of first layer
                        print(f"Detected hidden_dim={hidden_dim} from checkpoint")
                else:
                    # Default to config
                    hidden_dim = config.hidden_dim

                # Manually create PPOAgent with the correct dimensions
                trained_agent = PPOAgent(
                    state_dim=model_state_dim,
                    action_dim=env.action_space.n,
                    hidden_dim=hidden_dim,
                    config=config,
                )

                # Manually load state dictionaries
                try:
                    trained_agent.policy.load_state_dict(
                        checkpoint["policy_state_dict"]
                    )
                    trained_agent.value.load_state_dict(checkpoint["value_state_dict"])
                    print("Successfully loaded model with matching architecture")
                except Exception as load_error:
                    print(f"Failed to load state dictionaries: {load_error}")
                    raise ValueError(f"Could not load model from {trained_model_path}")

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

    # Initialize wandb for evaluation if enabled
    if use_wandb:
        init_wandb_for_evaluation(config, symbol, episodes)

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

    # Log evaluation metrics using our visualization module
    if use_wandb:
        log_evaluation_results(
            random_return,
            buyhold_return,
            ppo_return,
            random_perf,
            buyhold_perf,
            ppo_perf,
        )
        finish_wandb()

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


def test_with_binance(agent, symbol, config, test_duration=None, initial_balance=None):
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
