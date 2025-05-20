import wandb
import time
import numpy as np


def init_wandb_for_training(
    config,
    symbols,
    num_epochs,
    steps_per_epoch,
    initial_balance,
    num_parallel_envs,
    steps_per_env,
):
    """
    Initialize Weights & Biases for training.

    Args:
        config: The configuration object
        symbols: List of trading symbols
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        initial_balance: Initial balance for trading
        num_parallel_envs: Number of parallel environments
        steps_per_env: Steps per environment

    Returns:
        None
    """
    if not config.use_wandb:
        return

    # Create a descriptive name for the model that includes all symbols
    symbols_string = "_".join([s.replace("USDT", "") for s in symbols])
    model_name = f"ppo-multi-{symbols_string}"

    # Add more detailed config parameters that don't change during training
    wandb_config = {
        "symbols": symbols,
        "epochs": num_epochs,
        "steps_per_epoch": steps_per_epoch,
        "initial_balance": initial_balance,
        "game_length": config.game_length,
        "model_type": "PPO",
        "window_size": config.window_size,
        "commission_rate": config.commission_rate,
        "num_parallel_envs": num_parallel_envs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "hidden_dim": getattr(config, "hidden_dim", 512),
        "ppo_epochs": config.ppo_epochs,
        "gamma": config.gamma,
        "clip_epsilon": config.clip_epsilon,
        "steps_per_env": steps_per_env,
        "mixed_precision": (
            config.use_mixed_precision
            if hasattr(config, "use_mixed_precision")
            else False
        ),
        "total_samples_per_epoch": num_parallel_envs * steps_per_env,
    }

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_team,
        name=f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
        config=wandb_config,
    )


def log_training_metrics(metrics):
    """
    Log training metrics to Weights & Biases.

    Args:
        metrics: Dictionary of metrics to log

    Returns:
        None
    """
    if wandb.run:
        wandb.log(metrics)


def init_wandb_for_evaluation(config, symbol, episodes):
    """
    Initialize Weights & Biases for evaluation.

    Args:
        config: The configuration object
        symbol: The trading symbol being evaluated
        episodes: Number of evaluation episodes

    Returns:
        None
    """
    if not config.use_wandb:
        return

    if not wandb.run:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_team,
            name=f"eval-{symbol}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={"symbol": symbol, "episodes": episodes},
        )


def log_evaluation_results(
    random_return, buyhold_return, ppo_return, random_perf, buyhold_perf, ppo_perf
):
    """
    Log evaluation results to Weights & Biases.

    Args:
        random_return: Return percentage of the random agent
        buyhold_return: Return percentage of the buy and hold strategy
        ppo_return: Return percentage of the trained PPO agent
        random_perf: Portfolio values over time for the random agent
        buyhold_perf: Portfolio values over time for the buy and hold strategy
        ppo_perf: Portfolio values over time for the trained PPO agent

    Returns:
        None
    """
    if not wandb.run:
        return

    # Log returns
    wandb.log(
        {
            "random_agent_return": random_return,
            "buy_hold_return": buyhold_return,
            "trained_agent_return": ppo_return,
        }
    )

    # Log performance comparison
    # We'll use wandb's built-in line chart by providing the data
    # Convert numpy arrays to lists if necessary
    if isinstance(random_perf, np.ndarray):
        random_perf = random_perf.tolist()
    if isinstance(buyhold_perf, np.ndarray):
        buyhold_perf = buyhold_perf.tolist()
    if isinstance(ppo_perf, np.ndarray):
        ppo_perf = ppo_perf.tolist()

    # Create data for the line chart
    wandb.log(
        {
            "performance_comparison": {
                "random": random_perf,
                "buyhold": buyhold_perf,
                "trained": ppo_perf,
            }
        }
    )


def finish_wandb():
    """
    Finish the current Weights & Biases run.

    Returns:
        None
    """
    if wandb.run:
        wandb.finish()
