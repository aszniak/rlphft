import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")


@dataclass
class Config:
    """Central configuration for the trading agent system"""

    # W&B settings
    use_wandb: bool = False
    wandb_project: str = os.getenv("WANDB_PROJECT", "")
    wandb_team: str = os.getenv("WANDB_TEAM", "")

    # API settings
    api_key: str = os.getenv("BINANCE_API_KEY", "")

    # Santiment API settings
    santiment_api_key: str = os.getenv("SANTIMENT_API_KEY", "")
    use_sentiment: bool = (
        False  # Temporarily disable sentiment data to reduce feature complexity
    )
    sentiment_metrics: List[str] = field(
        default_factory=lambda: [
            "sentiment_volume_consumed_twitter",
            "social_volume_twitter",
            "social_dominance_twitter",
            "dev_activity",
            "github_activity",
            "exchange_inflow",
            "exchange_outflow",
        ]
    )

    # Data settings
    interval: str = "1h"  # 1-hour candles for position trading
    training_lookback_days: int = 365  # More historical data for longer-term patterns
    eval_lookback_days: int = 90  # 90-day evaluation periods
    force_refresh: bool = False
    random_eval_period: bool = True  # Use random 90-day window for evaluation

    # Trading symbols
    symbols: List[str] = field(
        default_factory=lambda: [
            "BTCUSDT",  # Bitcoin
            "ETHUSDT",  # Ethereum
            "BNBUSDT",  # Binance Coin
            "SOLUSDT",  # Solana
            "XRPUSDT",  # Ripple
        ]
    )

    # Environment settings
    game_length: int = 168  # 1 week of hourly candles (7 days * 24 hours)
    window_size: int = 24  # 24 hours of past data for state
    commission_rate: float = 0.001  # 0.1% commission rate (Binance standard)
    initial_balance: float = 10000.0  # Updated from default value
    max_episode_steps: int = 168  # Maximum steps per episode (1 week of hourly candles)

    # Training parameters
    eval_episodes: int = 5
    steps_per_epoch: int = (
        168
        * 8  # 8 weeks of hourly data per epoch (more realistic for position trading)
    )
    epochs: int = 500  # Increased from 100 for more thorough training
    num_parallel_envs: int = 16  # Reduced since episodes are longer
    batch_size: int = 1024  # Reduced for longer episodes

    # Hardware optimization flags
    optimize_hardware: bool = True  # Enable hardware optimization by default
    use_mixed_precision: bool = True  # Enable mixed precision training by default

    # PPO Agent parameters
    hidden_dim: int = 512  # Increased from 512 to 768 for better representation
    gamma: float = 0.99  # Discount factor for future rewards
    clip_epsilon: float = 0.2  # PPO clipping parameter
    learning_rate: float = 3e-4  # Increased from 1e-5 for better learning
    ppo_epochs: int = 5  # Reduced for large batch sizes
    entropy_coef: float = 0.05  # Increased from 0.015 to encourage more exploration
    value_coef: float = 0.5  # Value loss coefficient
    clip_grad_norm: float = (
        5.0  # Higher gradient clipping threshold for large batch size
    )

    # Paths
    model_path: str = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.api_key and not os.path.exists("config.env"):
            print("Warning: API key not found. Some features may not work.")

        if not self.santiment_api_key and self.use_sentiment:
            print(
                "Warning: Santiment API key not found. Sentiment features will be disabled."
            )
            self.use_sentiment = False

        if self.window_size >= self.game_length:
            raise ValueError("window_size must be smaller than game_length")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items()}


# Default configuration
default_config = Config()
