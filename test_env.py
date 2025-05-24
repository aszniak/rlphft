#!/usr/bin/env python3
"""
Test script to verify the trading environment works correctly.
This script checks that portfolio changes, actions, and state transitions
work as expected.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules but handle missing dependencies gracefully
try:
    from trading_env import TradingEnv, TradingAction
    from config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def create_test_data():
    """Create synthetic test data for environment verification"""
    # Create 1000 candles with predictable price movements
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")

    # Create a simple uptrend followed by a downtrend
    base_price = 50000
    prices = []

    for i in range(1000):
        if i < 500:
            # Uptrend: price increases gradually
            price = base_price + (i * 10) + np.random.normal(0, 50)
        else:
            # Downtrend: price decreases
            price = base_price + (500 * 10) - ((i - 500) * 5) + np.random.normal(0, 50)
        prices.append(max(price, 1000))  # Ensure price doesn't go too low

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price + np.random.uniform(0, 100)
        low = price - np.random.uniform(0, 100)
        open_price = price + np.random.uniform(-50, 50)
        volume = np.random.uniform(1000, 10000)

        data.append(
            {
                "open_time": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)

    # Add simple normalized features for testing
    df["price_norm"] = (df["close"] - df["close"].min()) / (
        df["close"].max() - df["close"].min()
    )
    df["volume_norm"] = (df["volume"] - df["volume"].min()) / (
        df["volume"].max() - df["volume"].min()
    )

    return {"TESTCOIN": df}


def test_basic_functionality():
    """Test basic environment functionality"""
    print("=" * 60)
    print("TESTING BASIC ENVIRONMENT FUNCTIONALITY")
    print("=" * 60)

    # Create test data
    data_dict = create_test_data()

    # Create environment
    env = TradingEnv(
        data_dict=data_dict,
        symbol="TESTCOIN",
        initial_balance=10000.0,
        window_size=10,
        game_length=100,
        random_start=False,
        full_data_evaluation=False,
    )

    print(f"✅ Environment created successfully")
    print(f"   Initial balance: ${env.initial_balance}")
    print(f"   Window size: {env.window_size}")
    print(f"   State dimension: {env.observation_space.shape[0]}")
    print(f"   Action space: {env.action_space.n}")

    # Test reset
    state, info = env.reset()
    print(f"✅ Environment reset successfully")
    print(f"   State shape: {state.shape}")
    print(f"   Portfolio value: ${info['total_portfolio_value']:.2f}")
    print(f"   Current price: ${info['current_price']:.2f}")

    return env, state, info


def test_action_execution():
    """Test that actions are executed correctly and portfolio changes reflect properly"""
    print("\n" + "=" * 60)
    print("TESTING ACTION EXECUTION AND PORTFOLIO CHANGES")
    print("=" * 60)

    # Create test data
    data_dict = create_test_data()

    # Create environment
    env = TradingEnv(
        data_dict=data_dict,
        symbol="TESTCOIN",
        initial_balance=10000.0,
        window_size=10,
        game_length=50,
        random_start=False,
        full_data_evaluation=False,
    )

    state, info = env.reset()
    initial_portfolio = info["total_portfolio_value"]

    print(f"Starting portfolio: ${initial_portfolio:.2f}")
    print(f"Starting cash: ${info['account_balance']:.2f}")
    print(f"Starting crypto: {info['crypto_holdings']:.6f}")
    print(f"Current price: ${info['current_price']:.2f}")

    # Test OPEN_LONG action (buy 25% of balance)
    print("\n--- Testing OPEN_LONG action ---")
    action = TradingAction.OPEN_LONG.value
    state, reward, done, truncated, info = env.step(action)

    print(f"After OPEN_LONG:")
    print(f"  Portfolio: ${info['total_portfolio_value']:.2f}")
    print(f"  Cash: ${info['account_balance']:.2f}")
    print(f"  Crypto: {info['crypto_holdings']:.6f}")
    print(f"  Crypto value: ${info['crypto_value']:.2f}")
    print(f"  Price: ${info['current_price']:.2f}")
    print(f"  Reward: {reward:.6f}")

    # Verify that cash decreased and crypto increased
    if info["account_balance"] < initial_portfolio and info["crypto_holdings"] > 0:
        print("✅ OPEN_LONG action executed correctly")
    else:
        print("❌ OPEN_LONG action failed")
        return False

    # Test HOLD action
    print("\n--- Testing HOLD action ---")
    portfolio_before_hold = info["total_portfolio_value"]
    cash_before_hold = info["account_balance"]
    crypto_before_hold = info["crypto_holdings"]

    action = TradingAction.HOLD.value
    state, reward, done, truncated, info = env.step(action)

    print(f"After HOLD:")
    print(f"  Portfolio: ${info['total_portfolio_value']:.2f}")
    print(f"  Cash: ${info['account_balance']:.2f}")
    print(f"  Crypto: {info['crypto_holdings']:.6f}")
    print(f"  Price: ${info['current_price']:.2f}")
    print(f"  Reward: {reward:.6f}")

    # Verify that cash and crypto holdings didn't change (but portfolio value might due to price change)
    if (
        abs(info["account_balance"] - cash_before_hold) < 0.01
        and abs(info["crypto_holdings"] - crypto_before_hold) < 1e-8
    ):
        print("✅ HOLD action executed correctly")
    else:
        print("❌ HOLD action failed")
        return False

    # Test CLOSE_LONG action
    print("\n--- Testing CLOSE_LONG action ---")
    action = TradingAction.CLOSE_LONG.value
    state, reward, done, truncated, info = env.step(action)

    print(f"After CLOSE_LONG:")
    print(f"  Portfolio: ${info['total_portfolio_value']:.2f}")
    print(f"  Cash: ${info['account_balance']:.2f}")
    print(f"  Crypto: {info['crypto_holdings']:.6f}")
    print(f"  Price: ${info['current_price']:.2f}")
    print(f"  Reward: {reward:.6f}")

    # Verify that crypto holdings are now zero and cash increased
    if info["crypto_holdings"] < 1e-8 and info["account_balance"] > cash_before_hold:
        print("✅ CLOSE_LONG action executed correctly")
    else:
        print("❌ CLOSE_LONG action failed")
        return False

    return True


def test_portfolio_value_tracking():
    """Test that portfolio value changes are correctly tracked"""
    print("\n" + "=" * 60)
    print("TESTING PORTFOLIO VALUE TRACKING")
    print("=" * 60)

    # Create test data
    data_dict = create_test_data()

    # Create environment
    env = TradingEnv(
        data_dict=data_dict,
        symbol="TESTCOIN",
        initial_balance=10000.0,
        window_size=10,
        game_length=20,
        random_start=False,
        full_data_evaluation=False,
    )

    state, info = env.reset()

    # Buy some crypto
    action = TradingAction.OPEN_LONG.value
    state, reward, done, truncated, info = env.step(action)

    portfolio_values = []
    prices = []

    print("Tracking portfolio value changes:")
    print("Step | Price     | Portfolio | Crypto    | Cash     ")
    print("-" * 50)

    for step in range(10):
        portfolio_values.append(info["total_portfolio_value"])
        prices.append(info["current_price"])

        print(
            f"{step:4} | ${info['current_price']:8.2f} | ${info['total_portfolio_value']:8.2f} | "
            f"{info['crypto_holdings']:8.4f} | ${info['account_balance']:7.2f}"
        )

        # Take HOLD action to just move forward in time
        action = TradingAction.HOLD.value
        state, reward, done, truncated, info = env.step(action)

        if done:
            break

    # Verify that portfolio value changes correlate with price changes
    # when holding crypto positions
    price_changes = np.diff(prices)
    portfolio_changes = np.diff(portfolio_values)

    # Check if portfolio changes in the same direction as price changes
    correlation = np.corrcoef(price_changes, portfolio_changes)[0, 1]

    print(f"\nPrice-Portfolio correlation: {correlation:.4f}")

    if correlation > 0.5:  # Strong positive correlation expected
        print("✅ Portfolio value tracking works correctly")
        return True
    else:
        print("❌ Portfolio value tracking may have issues")
        return False


def test_commission_impact():
    """Test that commissions are correctly applied"""
    print("\n" + "=" * 60)
    print("TESTING COMMISSION IMPACT")
    print("=" * 60)

    # Create test data
    data_dict = create_test_data()

    # Create environment with higher commission for easier testing
    env = TradingEnv(
        data_dict=data_dict,
        symbol="TESTCOIN",
        initial_balance=10000.0,
        commission_rate=0.01,  # 1% commission for easier testing
        window_size=10,
        game_length=20,
        random_start=False,
        full_data_evaluation=False,
    )

    state, info = env.reset()
    initial_balance = info["total_portfolio_value"]

    print(f"Initial balance: ${initial_balance:.2f}")
    print(f"Commission rate: {env.commission_rate*100:.1f}%")

    # Buy crypto
    action = TradingAction.OPEN_LONG.value
    state, reward, done, truncated, info = env.step(action)

    # Calculate expected commission
    buy_amount = initial_balance * 0.25  # OPEN_LONG uses 25%
    expected_commission = buy_amount * env.commission_rate

    print(f"\nAfter buying 25% of balance:")
    print(f"  Trade amount: ${buy_amount:.2f}")
    print(f"  Expected commission: ${expected_commission:.2f}")
    print(f"  Portfolio: ${info['total_portfolio_value']:.2f}")
    print(f"  Transaction history: {len(env.transaction_history)} trades")

    if len(env.transaction_history) > 0:
        actual_commission = env.transaction_history[-1]["commission"]
        print(f"  Actual commission: ${actual_commission:.2f}")

        if abs(actual_commission - expected_commission) < 0.01:
            print("✅ Commission calculation is correct")
        else:
            print("❌ Commission calculation may be incorrect")
            return False

    # Now sell all crypto
    action = TradingAction.CLOSE_LONG.value
    state, reward, done, truncated, info = env.step(action)

    print(f"\nAfter selling all crypto:")
    print(f"  Portfolio: ${info['total_portfolio_value']:.2f}")
    print(f"  Transaction history: {len(env.transaction_history)} trades")

    if len(env.transaction_history) > 1:
        sell_commission = env.transaction_history[-1]["commission"]
        print(f"  Sell commission: ${sell_commission:.2f}")

    # Total portfolio should be less than initial due to commissions
    total_loss = initial_balance - info["total_portfolio_value"]
    print(f"  Total loss from commissions: ${total_loss:.2f}")

    if total_loss > 0:
        print("✅ Commissions are being applied correctly")
        return True
    else:
        print("❌ Commissions may not be applied correctly")
        return False


def run_all_tests():
    """Run all environment tests"""
    print("TRADING ENVIRONMENT VERIFICATION TESTS")
    print("=" * 60)

    try:
        # Test basic functionality
        env, state, info = test_basic_functionality()

        # Test action execution
        action_test_passed = test_action_execution()

        # Test portfolio tracking
        portfolio_test_passed = test_portfolio_value_tracking()

        # Test commission impact
        commission_test_passed = test_commission_impact()

        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"✅ Basic functionality: PASSED")
        print(
            f"{'✅' if action_test_passed else '❌'} Action execution: {'PASSED' if action_test_passed else 'FAILED'}"
        )
        print(
            f"{'✅' if portfolio_test_passed else '❌'} Portfolio tracking: {'PASSED' if portfolio_test_passed else 'FAILED'}"
        )
        print(
            f"{'✅' if commission_test_passed else '❌'} Commission calculation: {'PASSED' if commission_test_passed else 'FAILED'}"
        )

        all_passed = (
            action_test_passed and portfolio_test_passed and commission_test_passed
        )

        if all_passed:
            print(
                f"\n🎉 ALL TESTS PASSED! The environment appears to be working correctly."
            )
        else:
            print(f"\n⚠️  SOME TESTS FAILED. There may be issues with the environment.")

        return all_passed

    except Exception as e:
        print(f"\n❌ ERROR during testing: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()
