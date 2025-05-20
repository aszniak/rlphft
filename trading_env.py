import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class TradingAction(Enum):
    """Enumeration of discrete trading actions"""

    DO_NOTHING = 0
    SELL_20_PERCENT = 1
    SELL_10_PERCENT = 2
    BUY_10_PERCENT = 3
    BUY_20_PERCENT = 4


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
        full_data_evaluation: bool = False,
        max_episode_steps: int = 240,  # Force episode termination every 4 hours (240 minutes)
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
            full_data_evaluation: If True, use full data for evaluation
            max_episode_steps: Maximum steps before forcing episode termination
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
        self.full_data_evaluation = full_data_evaluation
        self.max_episode_steps = max_episode_steps
        self.steps_in_current_episode = 0

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

        # For tracking rewards
        self.portfolio_history = []
        self.last_action = None
        self.previous_portfolio_value = self.initial_balance

        # Parameters for reward calculation
        self.reward_volatility_window = 30  # Window for volatility calculation
        self.sharpe_weight = 0.5
        self.alpha_weight = 0.3
        self.trend_weight = 0.1
        self.trade_weight = 0.1

        # Initial state
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Reset trading state
        self.account_balance = self.initial_balance
        self.crypto_holdings = 0.0
        self.transaction_history = []

        # Reset reward tracking variables
        self.portfolio_history = [self.initial_balance]
        self.last_action = None
        self.previous_portfolio_value = self.initial_balance
        self.steps_in_current_episode = 0  # Reset episode step counter

        # For full data evaluation, start from the beginning
        if self.full_data_evaluation:
            self.current_step = self.window_size
            self.start_index = 0
            self.end_index = len(self.data) - 1
        else:
            # Training mode - randomly select a segment of data
            if self.random_start:
                max_start = len(self.data) - self.game_length - self.window_size
                self.start_index = np.random.randint(0, max_start)
                self.end_index = self.start_index + self.game_length
                self.current_step = self.start_index + self.window_size
            else:
                # Fixed start for reproducible evaluation
                self.start_index = self.window_size
                self.end_index = self.start_index + self.game_length
                self.current_step = self.start_index

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
        # Store current state for reward calculation
        self.last_action = action
        portfolio_value_before = self.total_portfolio_value
        self.previous_portfolio_value = portfolio_value_before

        # Execute trading action
        self._take_action(action)

        # Move to next time step
        self.current_step += 1
        self.steps_in_current_episode += 1

        # Update portfolio based on new prices
        self._update_portfolio_value()

        # Add current portfolio value to history
        self.portfolio_history.append(self.total_portfolio_value)

        # Calculate sophisticated reward
        reward = self._calculate_reward()

        # Get new state
        state = self._get_state()

        # Check if episode is done due to reaching the end of data
        done_end_of_data = self.current_step >= self.end_index

        # Check if episode is done due to reaching maximum episode steps
        # Only apply step limit during training, not during full data evaluation
        done_max_steps = False
        if not self.full_data_evaluation:
            done_max_steps = self.steps_in_current_episode >= self.max_episode_steps

        # Episode is done if either condition is met
        terminated = done_end_of_data or done_max_steps
        truncated = False

        return state, reward, terminated, truncated, self._get_info()

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

        elif action_type == TradingAction.BUY_10_PERCENT:
            # Buy with 10% of current balance
            buy_amount_usd = self.account_balance * 0.10
            if buy_amount_usd > 0:
                self._execute_buy(buy_amount_usd, current_price)

        elif action_type == TradingAction.BUY_20_PERCENT:
            # Buy with 20% of current balance
            buy_amount_usd = self.account_balance * 0.20
            if buy_amount_usd > 0:
                self._execute_buy(buy_amount_usd, current_price)

        elif action_type == TradingAction.SELL_10_PERCENT:
            # Sell 10% of crypto holdings
            sell_amount_crypto = self.crypto_holdings * 0.10
            if sell_amount_crypto > 0:
                self._execute_sell(sell_amount_crypto, current_price)

        elif action_type == TradingAction.SELL_20_PERCENT:
            # Sell 20% of crypto holdings
            sell_amount_crypto = self.crypto_holdings * 0.20
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

    def _calculate_reward(self):
        """Calculate a sophisticated reward using multiple components"""
        # Basic return calculation
        raw_return = (
            self.total_portfolio_value - self.previous_portfolio_value
        ) / self.previous_portfolio_value

        # Calculate volatility component if we have enough history
        if len(self.portfolio_history) >= self.reward_volatility_window:
            # Calculate returns for the recent window
            recent_values = self.portfolio_history[-self.reward_volatility_window :]
            returns = np.array(
                [
                    (recent_values[i + 1] - recent_values[i]) / recent_values[i]
                    for i in range(len(recent_values) - 1)
                ]
            )

            volatility = np.std(returns) + 1e-6  # Avoid division by zero

            # Higher reward for positive returns with low volatility
            if raw_return > 0:
                sharpe_component = raw_return / (volatility * 10)  # Scale factor
            else:
                sharpe_component = raw_return  # Keep negative returns as penalty
        else:
            sharpe_component = raw_return

        # Market benchmark comparison (alpha)
        current_price = self.data.iloc[self.current_step]["close"]
        previous_price = self.data.iloc[self.current_step - 1]["close"]
        market_return = (current_price - previous_price) / previous_price
        alpha_component = raw_return - market_return

        # Trend alignment reward - significantly increased
        trend_component = self._calculate_trend_alignment(self.last_action)

        # Trading frequency penalty - dramatically increased to discourage excessive trading
        # This is critical - with 0.1% commission per trade, frequent trading erodes capital
        trade_penalty = (
            -0.005 if self.last_action != TradingAction.DO_NOTHING.value else 0.001
        )

        # Extra reward for holding during profitable market moves
        holding_reward = 0.0
        if (
            self.last_action == TradingAction.DO_NOTHING.value
            and market_return > 0
            and self.crypto_holdings > 0
        ):
            # Reward for holding crypto during uptrends
            holding_reward = 0.002

        # Capital preservation component - add significant penalty for dropping below initial balance
        # This provides a strong incentive to preserve capital and avoid excessive risk
        capital_preservation = 0.0
        if self.total_portfolio_value < self.initial_balance:
            # Calculate how far below initial balance we are (as a percentage)
            loss_percentage = (
                self.initial_balance - self.total_portfolio_value
            ) / self.initial_balance

            # Progressive penalty that grows exponentially as losses increase
            # Small losses (~5%) get a minor penalty, but larger losses (>10%) receive severe penalties
            capital_preservation = (
                -0.01 * (loss_percentage**2) * 10
            )  # Quadratic penalty

            # Additional severe penalty if portfolio drops below 80% of initial balance
            if self.total_portfolio_value < (self.initial_balance * 0.8):
                capital_preservation -= 0.02  # Extra fixed penalty for severe drawdown

        # Combine all reward components with updated weighting
        # Increase the penalties for trading and rewards for trend alignment
        total_reward = (
            0.35 * sharpe_component  # Reduced weight for Sharpe
            + 0.15 * alpha_component  # Reduced weight for alpha
            + 0.15 * trend_component  # Reduced weight for trend alignment
            + 0.15 * trade_penalty  # Reduced weight for trading penalty
            + 0.10
            * holding_reward  # Reduced weight for holding during profitable trends
            + 0.10 * capital_preservation  # New capital preservation component
        )

        return total_reward

    def _calculate_trend_alignment(self, action):
        """Calculate bonus for trading aligned with market trend"""
        # Default - no bonus
        alignment_bonus = 0.0

        # Use stationary indicators for trend detection - prefer crossovers and distances over raw SMAs
        if "sma_5_sma_20_crossover" in self.data.columns:
            # Use the crossover indicator directly (1 = uptrend, 0 = downtrend)
            is_uptrend = (
                self.data.iloc[self.current_step]["sma_5_sma_20_crossover"] == 1
            )

            # Use distance between MAs to gauge trend strength
            if "sma_5_sma_20_distance" in self.data.columns:
                trend_strength = abs(
                    self.data.iloc[self.current_step]["sma_5_sma_20_distance"]
                )
            else:
                trend_strength = 0.01  # Default strength if distance not available

            # Check if action aligns with trend - increased rewards
            if is_uptrend:  # Uptrend
                if action in [
                    TradingAction.BUY_10_PERCENT.value,
                    TradingAction.BUY_20_PERCENT.value,
                ]:
                    alignment_bonus = 0.005 * (
                        1 + trend_strength
                    )  # Higher bonus for buying in uptrend
                elif (
                    action == TradingAction.DO_NOTHING.value
                    and self.crypto_holdings > 0
                ):
                    alignment_bonus = 0.003 * (
                        1 + trend_strength
                    )  # Reward for holding crypto in uptrend
                elif action in [
                    TradingAction.SELL_10_PERCENT.value,
                    TradingAction.SELL_20_PERCENT.value,
                ]:
                    alignment_bonus = -0.003 * (
                        1 + trend_strength
                    )  # Penalty for selling in uptrend
            else:  # Downtrend
                if action in [
                    TradingAction.SELL_10_PERCENT.value,
                    TradingAction.SELL_20_PERCENT.value,
                ]:
                    alignment_bonus = 0.005 * (
                        1 + trend_strength
                    )  # Higher bonus for selling in downtrend
                elif (
                    action == TradingAction.DO_NOTHING.value
                    and self.account_balance > 0
                ):
                    alignment_bonus = 0.002 * (
                        1 + trend_strength
                    )  # Reward for holding cash in downtrend
                elif action in [
                    TradingAction.BUY_10_PERCENT.value,
                    TradingAction.BUY_20_PERCENT.value,
                ]:
                    alignment_bonus = -0.003 * (
                        1 + trend_strength
                    )  # Penalty for buying in downtrend

        # Fallback to original SMA comparison if stationary indicators not available
        elif all(indicator in self.data.columns for indicator in ["sma_5", "sma_20"]):
            sma_short = self.data.iloc[self.current_step]["sma_5"]
            sma_long = self.data.iloc[self.current_step]["sma_20"]

            # Check if action aligns with trend - increased rewards
            if sma_short > sma_long:  # Uptrend
                if action in [
                    TradingAction.BUY_10_PERCENT.value,
                    TradingAction.BUY_20_PERCENT.value,
                ]:
                    alignment_bonus = 0.005  # Higher bonus for buying in uptrend
                elif (
                    action == TradingAction.DO_NOTHING.value
                    and self.crypto_holdings > 0
                ):
                    alignment_bonus = 0.003  # Reward for holding crypto in uptrend
                elif action in [
                    TradingAction.SELL_10_PERCENT.value,
                    TradingAction.SELL_20_PERCENT.value,
                ]:
                    alignment_bonus = -0.003  # Penalty for selling in uptrend
            elif sma_short < sma_long:  # Downtrend
                if action in [
                    TradingAction.SELL_10_PERCENT.value,
                    TradingAction.SELL_20_PERCENT.value,
                ]:
                    alignment_bonus = 0.005  # Higher bonus for selling in downtrend
                elif (
                    action == TradingAction.DO_NOTHING.value
                    and self.account_balance > 0
                ):
                    alignment_bonus = 0.002  # Reward for holding cash in downtrend
                elif action in [
                    TradingAction.BUY_10_PERCENT.value,
                    TradingAction.BUY_20_PERCENT.value,
                ]:
                    alignment_bonus = -0.003  # Penalty for buying in downtrend

        # If no SMAs available, check simple price movement
        else:
            current_price = self.data.iloc[self.current_step]["close"]
            prev_price = self.data.iloc[max(0, self.current_step - 5)][
                "close"
            ]  # 5-step trend

            if current_price > prev_price:  # Uptrend
                if action in [
                    TradingAction.BUY_10_PERCENT.value,
                    TradingAction.BUY_20_PERCENT.value,
                ]:
                    alignment_bonus = 0.005
                elif (
                    action == TradingAction.DO_NOTHING.value
                    and self.crypto_holdings > 0
                ):
                    alignment_bonus = 0.003  # Reward for holding crypto in uptrend
            elif current_price < prev_price:  # Downtrend
                if action in [
                    TradingAction.SELL_10_PERCENT.value,
                    TradingAction.SELL_20_PERCENT.value,
                ]:
                    alignment_bonus = 0.005
                elif (
                    action == TradingAction.DO_NOTHING.value
                    and self.account_balance > 0
                ):
                    alignment_bonus = 0.002  # Reward for holding cash in downtrend

        return alignment_bonus


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
