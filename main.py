import os
import argparse
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
import time
import colorama
from colorama import Fore, Style
import wandb

from config import Config, default_config
from utils import fetch_data
from training import train_agent
from evaluation import evaluate_agents, test_with_binance
from agent import PPOAgent, RandomAgent
from trading_env import TradingEnv, BuyAndHoldEnv, TradingAction

# Initialize colorama for cross-platform color support
colorama.init()

# Load environment variables from config
load_dotenv("config.env")


def main():
    parser = argparse.ArgumentParser(description="RL Trading Agent")

    # Data fetching arguments
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=[
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
        ],  # Updated default with popular cryptos
        help="Symbols to trade (default: BTCUSDT ETHUSDT SOLUSDT). Can provide multiple symbols separated by spaces.",
    )
    parser.add_argument(
        "--lookback_days",
        type=int,
        help="Days of historical data to use (default: 100)",
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

    # Training arguments (simplified with better defaults)
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--fast_train",
        action="store_true",
        help="Use fast training mode (fewer epochs and steps)",
    )
    parser.add_argument(
        "--full_train",
        action="store_true",
        help="Use full training mode (more epochs for complete training)",
    )
    parser.add_argument(
        "--no_optimize",
        action="store_true",
        help="Disable automatic hardware optimization (not recommended)",
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Disable mixed precision training (not recommended for GPUs)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--random_eval",
        action="store_true",
        help="Use a random 30-day period for evaluation (default)",
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

    # Add arg for initial allocation
    parser.add_argument(
        "--initial_allocation",
        type=float,
        default=0.0,
        help="Initial allocation to crypto (0.0 to 1.0) for evaluation",
    )

    # Add sentiment data arguments
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Enable sentiment data from Santiment API (requires API key)",
    )
    parser.add_argument(
        "--no_sentiment",
        action="store_true",
        help="Disable sentiment data even if API key is available",
    )

    args = parser.parse_args()

    # Create config from defaults and override with args
    config = Config()

    # Override config with args if provided
    if args.symbols:
        config.symbols = args.symbols
    if args.lookback_days:
        config.training_lookback_days = args.lookback_days
        # Also apply to evaluation lookback for longer evaluations
        config.eval_lookback_days = args.lookback_days
    if args.force_refresh:
        config.force_refresh = args.force_refresh
    if args.epochs:
        config.epochs = args.epochs
    if args.episodes:
        config.eval_episodes = args.episodes
    if args.wandb:
        config.use_wandb = True
    if args.no_optimize:
        config.optimize_hardware = False
    if args.no_mixed_precision:
        config.use_mixed_precision = False

    # Handle sentiment data options
    if args.sentiment:
        config.use_sentiment = True
        print(f"{Fore.CYAN}🔍 Sentiment analysis enabled{Style.RESET_ALL}")
    if args.no_sentiment:
        config.use_sentiment = False
        print(f"{Fore.CYAN}🔍 Sentiment analysis disabled{Style.RESET_ALL}")

    # Handle fast/full training modes
    if args.fast_train:
        print(
            f"{Fore.CYAN}Using fast training mode with fewer epochs and steps{Style.RESET_ALL}"
        )
        config.epochs = min(config.epochs, 10)  # Maximum 10 epochs for fast training
        config.steps_per_epoch = 1024  # Fewer steps per epoch

    if args.full_train:
        print(f"{Fore.CYAN}Using full training mode with more epochs{Style.RESET_ALL}")
        config.epochs = max(config.epochs, 500)  # At least 500 epochs for full training

    # Display hardware optimization settings
    if config.optimize_hardware:
        print(f"{Fore.GREEN}Hardware optimization enabled{Style.RESET_ALL}")

        # Get GPU information if available
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"{Fore.CYAN}Detected GPU: {gpu_name}{Style.RESET_ALL}")

    if config.use_mixed_precision and torch.cuda.is_available():
        print(
            f"{Fore.CYAN}Mixed precision training enabled for faster GPU computation{Style.RESET_ALL}"
        )

    # Handle evaluation period options
    if args.random_eval:
        config.random_eval_period = True
    if args.fixed_eval:
        config.random_eval_period = False

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

            # Create a more compact table header with colors
            header = f"{Fore.YELLOW}{'Symbol':<10} {'Random':<10} {'Buy & Hold':<15} {'PPO Agent':<15} {'vs BuyHold':<12}{Style.RESET_ALL}"
            print(header)
            print(f"{'-'*65}")

            # Print each symbol's performance as a row in the table
            for symbol, result in all_results.items():
                random_return = result["random"]["return"]
                buyhold_return = result["buyhold"]["return"]
                trained_return = result["trained"]["return"]
                performance_vs_buyhold = trained_return - buyhold_return

                # Color code the comparison
                comparison_color = (
                    Fore.GREEN if performance_vs_buyhold > 0 else Fore.RED
                )

                # Make returns right-aligned with sign (+ or -) and color-coded
                print(
                    f"{symbol:<10} {Fore.CYAN}{random_return:>+8.2f}%{Style.RESET_ALL} {Fore.CYAN}{buyhold_return:>+13.2f}%{Style.RESET_ALL} "
                    + f"{Fore.CYAN}{trained_return:>+13.2f}%{Style.RESET_ALL} {comparison_color}{performance_vs_buyhold:>+10.2f}%{Style.RESET_ALL}"
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
