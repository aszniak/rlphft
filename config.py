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
    wandb_project: str = "crypto-trading-rl"
    wandb_team: str = "aszczesniak-aszczesniak"
    use_wandb: bool = False

    # API settings
    api_key: str = os.getenv("BINANCE_API_KEY", "")

    # Data settings
    interval: str = "1m"  # 1-minute candles
    training_lookback_days: int = 100  # 144,000 candles (100 days)
    eval_lookback_days: int = 30  # Standard 30-day period
    force_refresh: bool = False
    random_eval_period: bool = True  # Use random 30-day window for evaluation

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
    game_length: int = 1440  # 24 hours of minute candles
    window_size: int = 30  # How many past candles to include in state
    commission_rate: float = 0.001  # 0.1% commission rate (Binance standard)
    initial_balance: float = 10000.0  # Updated from default value
    max_episode_steps: int = (
        1440  # Maximum steps per episode (updated to match game_length)
    )

    # Training parameters
    eval_episodes: int = 5
    steps_per_epoch: int = (
        2880 * 2
    )  # Increased to ensure complete episodes (4 days per epoch)
    epochs: int = 100  # Reduced from 2000 to a more reasonable default
    num_parallel_envs: int = 32  # Updated to use more parallel environments
    batch_size: int = 2048  # Updated to a larger batch size for better GPU utilization

    # Hardware optimization flags
    optimize_hardware: bool = True  # Enable hardware optimization by default
    use_mixed_precision: bool = True  # Enable mixed precision training by default

    # PPO Agent parameters
    hidden_dim: int = 768  # Increased from 512 to 768 for better representation
    gamma: float = 0.99  # Discount factor for future rewards
    clip_epsilon: float = 0.2  # PPO clipping parameter
    learning_rate: float = 1e-5  # Updated to a smaller learning rate
    ppo_epochs: int = 5  # Reduced for large batch sizes
    entropy_coef: float = 0.015  # Increased from 0.01 to encourage more exploration
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
