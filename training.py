import numpy as np
import torch
import time
import wandb
from tqdm import tqdm
import multiprocessing
import colorama
from colorama import Fore, Style

from trading_env import TradingEnv, TradingAction
from agent import PPOAgent
from visualization import init_wandb_for_training, log_training_metrics, finish_wandb


def train_agent(
    data_dict,
    symbols,
    config,
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

    # Optimize number of parallel environments based on available CPU cores
    cpu_count = multiprocessing.cpu_count()

    # Track if batch size was manually specified
    batch_size_manually_set = (
        hasattr(config, "batch_size_manually_set") and config.batch_size_manually_set
    )

    # Use at least 8 environments, but scale up based on available CPUs
    # Leave some cores for system processes
    recommended_envs = min(max(8, cpu_count - 2), 32)

    # Only override if the config value is smaller and no manual override
    if config.num_parallel_envs < recommended_envs:
        print(
            f"{Fore.CYAN}Optimizing: Increasing parallel environments from {config.num_parallel_envs} to {recommended_envs} based on your {cpu_count} CPU cores{Style.RESET_ALL}"
        )
        num_parallel_envs = recommended_envs
    else:
        num_parallel_envs = config.num_parallel_envs

    # Optimize batch size based on data size BUT ONLY if batch_size was not manually set
    # If steps_per_epoch is large, use a larger batch size to improve GPU utilization
    if (
        not batch_size_manually_set
        and config.batch_size < 256
        and steps_per_epoch >= 2048
    ):
        original_batch = config.batch_size
        recommended_batch = min(512, steps_per_epoch // 8)
        config.batch_size = recommended_batch
        print(
            f"{Fore.CYAN}Optimizing: Increasing batch size from {original_batch} to {recommended_batch} for better GPU utilization{Style.RESET_ALL}"
        )
    else:
        if batch_size_manually_set:
            print(
                f"{Fore.GREEN}Using manually specified batch size: {config.batch_size}{Style.RESET_ALL}"
            )

    # For very large batch sizes, make sure PPO epochs is reasonable
    if config.batch_size >= 1024 and config.ppo_epochs > 5:
        print(
            f"{Fore.CYAN}Optimizing: Reducing PPO epochs from {config.ppo_epochs} to 5 for very large batch size{Style.RESET_ALL}"
        )
        config.ppo_epochs = 5

    # Calculate ideal steps per env based on steps_per_epoch and number of environments
    # This ensures we collect enough data while keeping batch sizes efficient
    steps_per_env = max(
        config.max_episode_steps,
        (steps_per_epoch + num_parallel_envs - 1) // num_parallel_envs,
    )

    # If we have a GPU, make sure we collect enough samples to utilize it well
    if torch.cuda.is_available():
        min_samples_for_gpu = 8192  # A reasonable minimum for good GPU utilization
        if steps_per_env * num_parallel_envs < min_samples_for_gpu:
            steps_per_env = (
                min_samples_for_gpu + num_parallel_envs - 1
            ) // num_parallel_envs
            print(
                f"{Fore.CYAN}Optimizing: Increasing steps_per_env to {steps_per_env} for better GPU utilization{Style.RESET_ALL}"
            )

    # Initialize W&B if requested
    if use_wandb:
        init_wandb_for_training(
            config,
            symbols,
            num_epochs,
            steps_per_epoch,
            initial_balance,
            num_parallel_envs,
            steps_per_env,
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
        f"{Fore.GREEN}🚀 Training on {len(symbols)} assets ({', '.join(symbols)}) for {num_epochs} epochs using {num_parallel_envs} parallel environments...{Style.RESET_ALL}"
    )
    epoch_rewards = []

    # Create a tqdm progress bar
    progress_bar = tqdm(range(num_epochs), desc="Training Progress")
    action_names = [a.name for a in TradingAction]

    # Track training start time
    training_start = time.time()

    for epoch in progress_bar:
        # Collect trajectories using parallel environments with random asset selection
        states, actions, rewards, dones, next_states, log_probs, _ = (
            ppo_agent.collect_trajectories_parallel(
                make_env,
                n_envs=num_parallel_envs,
                steps_per_env=steps_per_env,
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

        # Log to W&B using our visualization module
        if use_wandb:
            # Prepare metrics that change during training (exclude constant parameters)
            metrics = {
                "epoch": int(epoch),
                "avg_step_reward": float(avg_step_reward),
                "avg_episode_reward": float(avg_episode_reward),
                "completed_episodes": int(completed_episodes),
            }

            # Handle portfolio value and convert appropriately
            if portfolio_value is not None:
                metrics["portfolio_value"] = (
                    float(np.mean(portfolio_value))
                    if isinstance(portfolio_value, (list, np.ndarray))
                    else float(portfolio_value)
                )
            else:
                metrics["portfolio_value"] = 0.0

            # Handle rewards with proper tensor conversion
            if isinstance(rewards, torch.Tensor):
                rewards_np = rewards.cpu().detach().numpy()
                metrics["total_reward"] = (
                    float(rewards_np.sum()) if len(rewards_np) > 0 else 0.0
                )
                metrics["rewards_min"] = (
                    float(rewards_np.min()) if len(rewards_np) > 0 else 0.0
                )
                metrics["rewards_max"] = (
                    float(rewards_np.max()) if len(rewards_np) > 0 else 0.0
                )
            else:
                # Handle numpy or list type rewards
                metrics["total_reward"] = (
                    float(np.sum(rewards)) if len(rewards) > 0 else 0.0
                )
                metrics["rewards_min"] = (
                    float(np.min(rewards)) if len(rewards) > 0 else 0.0
                )
                metrics["rewards_max"] = (
                    float(np.max(rewards)) if len(rewards) > 0 else 0.0
                )

            # Add action distribution (convert to Python float)
            if len(actions) > 0:
                action_counts = np.bincount(actions, minlength=env.action_space.n)
                action_freq = action_counts / len(actions)
                for i, freq in enumerate(action_freq):
                    action_name = TradingAction(i).name.lower()
                    metrics[f"action_freq_{action_name}"] = float(freq)

            # Add more meaningful metrics about learning progress
            if hasattr(ppo_agent, "policy_optimizer"):
                # Track learning rate if scheduler is used
                current_lr = ppo_agent.policy_optimizer.param_groups[0]["lr"]
                metrics["learning_rate"] = current_lr

            # Add value loss and policy loss if available
            if hasattr(ppo_agent, "last_value_loss"):
                metrics["value_loss"] = float(ppo_agent.last_value_loss)
            if hasattr(ppo_agent, "last_policy_loss"):
                metrics["policy_loss"] = float(ppo_agent.last_policy_loss)
            if hasattr(ppo_agent, "last_entropy"):
                metrics["policy_entropy"] = float(ppo_agent.last_entropy)
            if hasattr(ppo_agent, "last_clip_fraction"):
                metrics["gradient_clip_fraction"] = float(ppo_agent.last_clip_fraction)

            # Track only GPU memory that changes
            if torch.cuda.is_available():
                metrics["gpu_memory_allocated_MB"] = float(
                    torch.cuda.memory_allocated() / 1024**2
                )

            # Add crypto to cash ratio from the environment
            if hasattr(env, "crypto_value") and hasattr(env, "account_balance"):
                total = env.crypto_value + env.account_balance
                if total > 0:
                    metrics["crypto_allocation"] = float(env.crypto_value / total)

            # Log metrics using our visualization module
            log_training_metrics(metrics)

    # Calculate total training time
    training_time = time.time() - training_start
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"\n{Fore.GREEN}Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s{Style.RESET_ALL}"
    )

    # Display profiling information if enabled
    if hasattr(config, "enable_profiling") and config.enable_profiling:
        ppo_agent.profiler.print_stats()

        # Print hardware optimization recommendations based on profiling
        print(f"\n{Fore.YELLOW}Hardware Utilization Analysis:{Style.RESET_ALL}")

        # Get profiling stats
        stats = ppo_agent.profiler.get_stats()

        if stats:
            # Check for environment step bottlenecks
            if "env_step" in stats and stats["env_step"]["percentage"] > 40:
                print(
                    f"{Fore.RED}Environment stepping is a major bottleneck ({stats['env_step']['percentage']:.1f}% of time){Style.RESET_ALL}"
                )
                print(
                    f"  - Recommendation: Increase number of parallel environments (--num_parallel_envs)"
                )

            # Check for data transfer bottlenecks
            if (
                "tensor_to_numpy" in stats
                and stats["tensor_to_numpy"]["percentage"] > 10
            ) or (
                "data_to_device" in stats and stats["data_to_device"]["percentage"] > 10
            ):
                print(
                    f"{Fore.RED}CPU-GPU data transfer appears to be a bottleneck{Style.RESET_ALL}"
                )
                print(
                    f"  - Recommendation: Use larger batch sizes (--cuda_batch_size) and activate mixed precision (--mixed_precision)"
                )

            # Check for policy inference bottlenecks
            if (
                "policy_inference" in stats
                and stats["policy_inference"]["percentage"] > 20
            ):
                print(
                    f"{Fore.YELLOW}Policy network inference is taking significant time{Style.RESET_ALL}"
                )
                print(
                    f"  - Recommendation: Enable mixed precision (--mixed_precision) if not already enabled"
                )

            # Check GPU utilization
            if torch.cuda.is_available():
                print(f"\nCUDA Memory Usage:")
                print(
                    f"  - Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"
                )
                print(f"  - Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
                print(
                    f"  - Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB"
                )

                # If memory usage is low, recommend larger batches
                if (
                    torch.cuda.max_memory_allocated()
                    / torch.cuda.get_device_properties(0).total_memory
                    < 0.5
                ):
                    print(
                        f"{Fore.YELLOW}GPU memory utilization is below 50%{Style.RESET_ALL}"
                    )
                    print(
                        f"  - Recommendation: Increase batch size (--cuda_batch_size) for better GPU utilization"
                    )

            # Provide sample command with recommended settings
            print(
                f"\n{Fore.GREEN}Try running with these optimized settings:{Style.RESET_ALL}"
            )
            cmd = f"python main.py --train --optimize_hardware --mixed_precision"

            if "env_step" in stats and stats["env_step"]["percentage"] > 40:
                recommended_envs = min(num_parallel_envs * 2, 32)
                cmd += f" --num_parallel_envs {recommended_envs}"

            if (
                torch.cuda.is_available()
                and torch.cuda.max_memory_allocated()
                / torch.cuda.get_device_properties(0).total_memory
                < 0.4
            ):
                recommended_batch = min(config.batch_size * 2, 1024)
                cmd += f" --cuda_batch_size {recommended_batch}"

            print(cmd)

    # Save the trained model
    if model_save_path:
        ppo_agent.save_model(model_save_path)
        print(f"{Fore.GREEN}💾 Model saved to {model_save_path}{Style.RESET_ALL}")

        # Upload model to W&B
        if use_wandb:
            wandb.save(f"{model_save_path}.pt")

    # Finish W&B run
    if use_wandb:
        finish_wandb()

    return ppo_agent
