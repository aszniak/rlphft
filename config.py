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
    lookback_days: int = 30
    force_refresh: bool = False

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
    initial_balance: float = 10000.0

    # Training parameters
    eval_episodes: int = 5
    steps_per_epoch: int = 2000
    epochs: int = 100

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
