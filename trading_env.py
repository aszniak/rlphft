import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class TradingAction(Enum):
    """Enumeration of discrete trading actions"""

    DO_NOTHING = 0
    SELL_2_PERCENT = 1
    SELL_1_PERCENT = 2
    BUY_1_PERCENT = 3
    BUY_2_PERCENT = 4


class TradingEnv(gym.Env):
    """
    A cryptocurrency trading environment compatible with Gymnasium interface

    The environment simulates trading on 24-hour periods of minute candles (1440 steps).
    Actions are discrete: do nothing, buy/sell 1% or 2% of current capital.
    """

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbol: str,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1% commission rate (Binance standard)
        window_size: int = 30,  # How many past candles to include in state
        game_length: int = 1440,  # One day of minute candles
        random_start: bool = True,
    ):
        """
        Initialize the trading environment.

        Args:
            data_dict: Dictionary of DataFrames with crypto data from data_fetcher
            symbol: Symbol to trade (e.g., 'BTCUSDT')
            initial_balance: Starting balance in USDT
            commission_rate: Trading fee as a fraction (e.g., 0.001 = 0.1%)
            window_size: Number of past candles included in the state
            game_length: Length of each game in candles
            random_start: If True, random starting point for each episode
        """
        super().__init__()

        # Store parameters
        self.data = data_dict[symbol].copy()
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.window_size = window_size
        self.game_length = game_length
        self.random_start = random_start

        # Define state and action spaces
        # For each time step we get normalized features from the dataframe
        # Count how many normalized (feature_norm) columns we have
        self.feature_columns = [
            col for col in self.data.columns if col.endswith("_norm")
        ]
        self.state_dim = len(self.feature_columns) * window_size

        # Action space: 5 discrete actions
        self.action_space = gym.spaces.Discrete(len(TradingAction))

        # Observation space: normalized values
        # All normalized features for window_size time steps
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )

        # Trading state variables
        self.current_step = 0
        self.start_index = 0
        self.end_index = 0
        self.account_balance = 0.0
        self.crypto_holdings = 0.0
        self.crypto_value = 0.0
        self.total_portfolio_value = 0.0
        self.transaction_history = []

        # Initial state
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Reset trading state
        self.account_balance = self.initial_balance
        self.crypto_holdings = 0.0
        self.transaction_history = []

        # Determine valid start and end indices
        min_idx = self.window_size  # Need window_size previous candles
        max_idx = (
            len(self.data) - self.game_length - 1
        )  # Need game_length candles ahead

        # Random or fixed starting point
        if self.random_start:
            self.start_index = (
                self.np_random.integers(min_idx, max_idx)
                if max_idx > min_idx
                else min_idx
            )
        else:
            self.start_index = min_idx

        self.current_step = self.start_index
        self.end_index = self.start_index + self.game_length

        # Update portfolio value
        self._update_portfolio_value()

        # Get initial state
        state = self._get_state()

        return state, self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a trading action and move to the next time step.

        Args:
            action: Integer representing the action to take (enum TradingAction)

        Returns:
            state: New state observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Execute trading action
        self._take_action(action)

        # Record portfolio value before moving to the next step
        portfolio_value_before = self.total_portfolio_value

        # Move to next time step
        self.current_step += 1

        # Update portfolio based on new prices
        self._update_portfolio_value()

        # Calculate reward (change in portfolio value)
        reward = (
            self.total_portfolio_value - portfolio_value_before
        ) / portfolio_value_before

        # Get new state
        state = self._get_state()

        # Check if episode is done
        done = self.current_step >= self.end_index

        return state, reward, done, False, self._get_info()

    def _take_action(self, action: int) -> None:
        """
        Execute a trading action.

        Args:
            action: Integer representing action from TradingAction enum
        """
        current_price = self.data.iloc[self.current_step]["close"]

        # Extract the action type from the enum
        action_type = TradingAction(action)

        if action_type == TradingAction.DO_NOTHING:
            # No action
            pass

        elif action_type == TradingAction.BUY_1_PERCENT:
            # Buy with 1% of current balance
            buy_amount_usd = self.account_balance * 0.01
            if buy_amount_usd > 0:
                self._execute_buy(buy_amount_usd, current_price)

        elif action_type == TradingAction.BUY_2_PERCENT:
            # Buy with 2% of current balance
            buy_amount_usd = self.account_balance * 0.02
            if buy_amount_usd > 0:
                self._execute_buy(buy_amount_usd, current_price)

        elif action_type == TradingAction.SELL_1_PERCENT:
            # Sell 1% of crypto holdings
            sell_amount_crypto = self.crypto_holdings * 0.01
            if sell_amount_crypto > 0:
                self._execute_sell(sell_amount_crypto, current_price)

        elif action_type == TradingAction.SELL_2_PERCENT:
            # Sell 2% of crypto holdings
            sell_amount_crypto = self.crypto_holdings * 0.02
            if sell_amount_crypto > 0:
                self._execute_sell(sell_amount_crypto, current_price)

    def _execute_buy(self, amount_usd: float, price: float) -> None:
        """
        Execute a buy order.

        Args:
            amount_usd: Amount in USD to spend
            price: Current crypto price
        """
        # Calculate commission
        commission = amount_usd * self.commission_rate

        # Calculate actual amount to buy after commission
        actual_amount_usd = amount_usd - commission

        # Convert to crypto amount
        crypto_amount = actual_amount_usd / price

        # Update balances
        self.account_balance -= amount_usd
        self.crypto_holdings += crypto_amount

        # Record transaction
        self.transaction_history.append(
            {
                "timestamp": self.data.iloc[self.current_step]["open_time"],
                "action": "buy",
                "price": price,
                "amount_usd": amount_usd,
                "amount_crypto": crypto_amount,
                "commission": commission,
            }
        )

    def _execute_sell(self, amount_crypto: float, price: float) -> None:
        """
        Execute a sell order.

        Args:
            amount_crypto: Amount of crypto to sell
            price: Current crypto price
        """
        # Calculate USD value before commission
        amount_usd = amount_crypto * price

        # Calculate commission
        commission = amount_usd * self.commission_rate

        # Calculate actual USD after commission
        actual_amount_usd = amount_usd - commission

        # Update balances
        self.account_balance += actual_amount_usd
        self.crypto_holdings -= amount_crypto

        # Record transaction
        self.transaction_history.append(
            {
                "timestamp": self.data.iloc[self.current_step]["open_time"],
                "action": "sell",
                "price": price,
                "amount_usd": actual_amount_usd,
                "amount_crypto": amount_crypto,
                "commission": commission,
            }
        )

    def _update_portfolio_value(self) -> None:
        """Update the current value of the portfolio."""
        current_price = self.data.iloc[self.current_step]["close"]
        self.crypto_value = self.crypto_holdings * current_price
        self.total_portfolio_value = self.account_balance + self.crypto_value

    def _get_state(self) -> np.ndarray:
        """
        Create a state representation from the current window of data.
        Returns normalized feature values for the window_size previous time steps.
        """
        # Get the window of data
        window_start = max(0, self.current_step - self.window_size + 1)
        window_end = self.current_step + 1
        window_data = self.data.iloc[window_start:window_end]

        # If we don't have enough history, pad with the oldest available data
        if len(window_data) < self.window_size:
            padding_needed = self.window_size - len(window_data)
            padding_data = self.data.iloc[0:1].copy()
            padding_data = pd.concat([padding_data] * padding_needed, ignore_index=True)
            window_data = pd.concat([padding_data, window_data], ignore_index=True)

        # Extract only the normalized feature columns
        features = window_data[self.feature_columns].values

        # Flatten to 1D array
        flat_state = features.flatten()

        return flat_state

    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        return {
            "timestamp": self.data.iloc[self.current_step]["open_time"],
            "current_price": self.data.iloc[self.current_step]["close"],
            "account_balance": self.account_balance,
            "crypto_holdings": self.crypto_holdings,
            "crypto_value": self.crypto_value,
            "total_portfolio_value": self.total_portfolio_value,
            "step": self.current_step - self.start_index,
            "total_steps": self.game_length,
        }

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state for visualization.
        Returns a visualization frame (to be implemented for GUI).
        """
        # For now, just return basic text representation
        info = self._get_info()
        print(f"Step: {info['step']} / {info['total_steps']}")
        print(f"Time: {info['timestamp']}")
        print(f"Price: {info['current_price']:.2f}")
        print(f"Portfolio: ${info['total_portfolio_value']:.2f}")
        print(f"  Cash: ${info['account_balance']:.2f}")
        print(f"  Crypto: {info['crypto_holdings']:.6f} (${info['crypto_value']:.2f})")
        print("---")
        return None


class BuyAndHoldEnv(TradingEnv):
    """
    A modified trading environment that implements a simple buy and hold strategy.
    Used for baseline comparison.
    """

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and execute buy and hold strategy."""
        state, info = super().reset(seed=seed, options=options)

        # Buy with all initial balance at the start
        current_price = self.data.iloc[self.current_step]["close"]
        self._execute_buy(self.account_balance, current_price)

        return state, self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step without taking any actions (just hold).

        Returns:
            state: New state observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Record portfolio value before moving to the next step
        portfolio_value_before = self.total_portfolio_value

        # Move to next time step
        self.current_step += 1

        # Update portfolio based on new prices
        self._update_portfolio_value()

        # Calculate reward (change in portfolio value)
        reward = (
            self.total_portfolio_value - portfolio_value_before
        ) / portfolio_value_before

        # Get new state
        state = self._get_state()

        # Check if episode is done
        done = self.current_step >= self.end_index

        return state, reward, done, False, self._get_info()
