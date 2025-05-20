import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to a DataFrame containing OHLCV data.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns

    Returns:
        DataFrame with additional technical indicator columns
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Dictionary to collect all indicator values
    indicators = {}

    # Simple Moving Averages
    indicators["sma_5"] = df["close"].rolling(window=5).mean()
    indicators["sma_20"] = df["close"].rolling(window=20).mean()
    indicators["sma_50"] = df["close"].rolling(window=50).mean()

    # Exponential Moving Averages
    indicators["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    indicators["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    indicators["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Moving Average Convergence Divergence (MACD)
    indicators["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    indicators["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    indicators["macd"] = indicators["ema_12"] - indicators["ema_26"]
    indicators["macd_signal"] = indicators["macd"].ewm(span=9, adjust=False).mean()
    indicators["macd_hist"] = indicators["macd"] - indicators["macd_signal"]

    # Relative Strength Index (RSI)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Calculate RS with handling for zero avg_loss
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rs = rs.fillna(100)  # When avg_loss is zero, RSI should be 100

    # Calculate RSI
    indicators["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    indicators["bb_middle"] = df["close"].rolling(window=20).mean()
    indicators["bb_std"] = df["close"].rolling(window=20).std()
    indicators["bb_upper"] = indicators["bb_middle"] + 2 * indicators["bb_std"]
    indicators["bb_lower"] = indicators["bb_middle"] - 2 * indicators["bb_std"]

    # Average True Range (ATR)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    indicators["atr_14"] = true_range.rolling(window=14).mean()

    # Volume indicators
    indicators["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    indicators["volume_ratio"] = df["volume"] / indicators["volume_sma_20"]

    # Price change rate
    indicators["close_pct_change_1"] = df["close"].pct_change(periods=1)
    indicators["close_pct_change_5"] = df["close"].pct_change(periods=5)

    # Volatility (standard deviation of returns)
    indicators["volatility_5"] = (
        indicators["close_pct_change_1"].rolling(window=5).std()
    )
    indicators["volatility_20"] = (
        indicators["close_pct_change_1"].rolling(window=20).std()
    )

    # Momentum indicators
    indicators["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    indicators["momentum_10"] = df["close"] / df["close"].shift(10) - 1

    # Stochastic Oscillator
    n = 14
    indicators["stoch_k"] = (
        (df["close"] - df["low"].rolling(window=n).min())
        / (df["high"].rolling(window=n).max() - df["low"].rolling(window=n).min())
    ) * 100
    indicators["stoch_d"] = indicators["stoch_k"].rolling(window=3).mean()

    # On-Balance Volume (OBV)
    indicators["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # Add market inefficiency indicators that might be useful for RL agent

    # Mean reversion potential (z-score of price)
    indicators["price_zscore_20"] = (df["close"] - indicators["sma_20"]) / indicators[
        "bb_std"
    ]

    # Momentum divergence (price making new highs but RSI not)
    indicators["close_5d_high"] = df["close"].rolling(window=5).max()
    indicators["rsi_5d_high"] = indicators["rsi_14"].rolling(window=5).max()
    indicators["price_rsi_divergence"] = (
        (df["close"] == indicators["close_5d_high"])
        & (indicators["rsi_14"] < indicators["rsi_5d_high"] * 0.95)
    ).astype(int)

    # Volatility expansion (ATR increasing)
    indicators["atr_expansion"] = indicators["atr_14"] / indicators["atr_14"].shift(5)

    # Volume spike detection
    indicators["volume_spike"] = (
        df["volume"] > indicators["volume_sma_20"] * 2
    ).astype(int)

    # Gap detection
    indicators["gap_up"] = (df["open"] > df["close"].shift(1) * 1.01).astype(int)
    indicators["gap_down"] = (df["open"] < df["close"].shift(1) * 0.99).astype(int)

    # Create a DataFrame from all indicators
    indicators_df = pd.DataFrame(indicators, index=df.index)

    # Join with the original DataFrame in one operation
    result = pd.concat([df_copy, indicators_df], axis=1)

    return result


def make_indicators_stationary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make technical indicators more stationary by calculating:
    - For price-based indicators: Percent changes (returns)
    - For oscillator indicators (already bounded): Use changes
    - For volatility indicators: Use relative changes

    Args:
        df: DataFrame with technical indicators

    Returns:
        DataFrame with additional stationary features
    """
    # Create a copy of the original dataframe
    df_copy = df.copy()

    # Dictionary to collect all stationary transformations
    stationary_columns = {}

    # 1. For moving averages and price-based indicators, calculate percent changes
    price_based_indicators = [
        "sma_5",
        "sma_20",
        "sma_50",
        "ema_5",
        "ema_12",
        "ema_20",
        "ema_26",
        "ema_50",
        "bb_middle",
        "bb_upper",
        "bb_lower",
    ]

    for col in price_based_indicators:
        if col in df.columns:
            # Calculate percent change (return)
            stationary_columns[f"{col}_pct"] = df[col].pct_change(periods=1)

            # Calculate difference between current and previous (for more direct stationarity)
            stationary_columns[f"{col}_diff"] = df[col].diff()

            # Calculate ratio to price (removes trend component) - this is stationary
            stationary_columns[f"{col}_ratio"] = df[col] / df["close"] - 1

            # Calculate crossovers (binary indicator of relative position - highly stationary)
            if "sma" in col or "ema" in col:
                # Extract the window size from the column name (e.g., "sma_20" -> 20)
                window_size = int(col.split("_")[1])

                # Only create crossover if we have both short and long term moving averages
                if window_size > 10:  # Only for longer-term MAs
                    # Check if shorter-term MA exists
                    short_ma = f"{col.split('_')[0]}_5"  # e.g., "sma_5" or "ema_5"
                    if short_ma in df.columns:
                        # Create crossover indicator (1 when short > long, 0 otherwise)
                        stationary_columns[f"{short_ma}_{col}_crossover"] = (
                            df[short_ma] > df[col]
                        ).astype(int)

                        # Create distance between MAs (normalized by price for stationarity)
                        stationary_columns[f"{short_ma}_{col}_distance"] = (
                            df[short_ma] - df[col]
                        ) / df["close"]

    # 2. For oscillators (already bounded), calculate changes
    oscillator_indicators = ["rsi_14", "stoch_k", "stoch_d"]

    for col in oscillator_indicators:
        if col in df.columns:
            # Simple change is more appropriate since these are already bounded
            stationary_columns[f"{col}_change"] = df[col].diff()

    # 3. For MACD components, use raw values or small changes
    macd_indicators = ["macd", "macd_signal", "macd_hist"]

    for col in macd_indicators:
        if col in df.columns:
            # MACD is somewhat stationary already as it oscillates around zero
            # but we can still calculate changes for added stationarity
            stationary_columns[f"{col}_change"] = df[col].diff()

    # 4. For volatility indicators, calculate relative changes
    volatility_indicators = ["atr_14", "volatility_5", "volatility_20", "bb_std"]

    for col in volatility_indicators:
        if col in df.columns:
            # Relative change (percent)
            stationary_columns[f"{col}_rel_change"] = df[col].pct_change()

    # 5. For volume indicators
    volume_indicators = ["volume_ratio", "obv"]

    for col in volume_indicators:
        if col in df.columns:
            if col == "obv":
                # For OBV, calculate change
                stationary_columns[f"{col}_change"] = df[col].diff()
            else:
                # For volume ratio, take raw value as it's already relatively stationary
                pass

    # 6. For momentum indicators, they are already stationary (percent changes)
    # For binary indicators, they are already stationary (0s and 1s)

    # Create a DataFrame from all stationary columns
    stationary_df = pd.DataFrame(stationary_columns, index=df.index)

    # Join with the original DataFrame in one operation
    result = pd.concat([df_copy, stationary_df], axis=1)

    return result


def normalize_features(df: pd.DataFrame, feature_cols=None) -> pd.DataFrame:
    """
    Normalize features to [0, 1] or [-1, 1] range depending on the feature.

    Args:
        df: DataFrame with features
        feature_cols: List of columns to normalize (if None, normalize all except time columns)

    Returns:
        DataFrame with normalized features
    """
    # Make a copy of the original dataframe
    df_copy = df.copy()

    # Create a dictionary to collect all normalized columns
    normalized_columns = {}

    if feature_cols is None:
        # Exclude time-related columns and non-numeric columns
        feature_cols = []
        for col in df.columns:
            # Skip time-related columns
            if any(time_col in col.lower() for time_col in ["time", "date"]):
                continue

            # Skip non-numeric columns
            if pd.api.types.is_numeric_dtype(df[col]) == False:
                continue

            feature_cols.append(col)

    for col in feature_cols:
        # Skip if the column doesn't exist
        if col not in df.columns:
            continue

        # Ensure we're working with numeric data
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # For inherently bounded features like RSI (0-100) or stochastic oscillators
        if "rsi" in col.lower() or "stoch" in col.lower():
            normalized_columns[f"{col}_norm"] = df[col] / 100

        # For z-score features that are already normalized
        elif "zscore" in col.lower():
            # Already normalized but clip to reasonable range
            normalized_columns[f"{col}_norm"] = np.clip(df[col], -3, 3) / 3

        # For percentage changes
        elif (
            "pct" in col.lower() or "momentum" in col.lower() or "change" in col.lower()
        ):
            # Normalize to [-1, 1] range with reasonable bounds
            normalized_columns[f"{col}_norm"] = np.clip(df[col], -0.1, 0.1) / 0.1

        # For binary indicators
        elif set(df[col].unique()).issubset({0, 1, 0.0, 1.0}):
            normalized_columns[f"{col}_norm"] = df[col]

        # For other features, use Min-Max scaling with rolling window
        else:
            # Handle special case for volume-related features
            if "volume" in col.lower():
                # Volume can spike dramatically, so use log transformation
                log_values = np.log1p(df[col])
                max_val = np.log1p(df[col].rolling(window=50).max())
                # Avoid division by zero
                max_val = max_val.replace(0, np.nan)
                normalized_value = log_values / max_val
                normalized_columns[f"{col}_norm"] = normalized_value.fillna(0.5)
            else:
                # Use rolling min-max to adapt to changing scales
                rolling_min = df[col].rolling(window=100, min_periods=20).min()
                rolling_max = df[col].rolling(window=100, min_periods=20).max()

                # Handle cases where min equals max
                denom = rolling_max - rolling_min
                denom = denom.replace(0, np.nan)  # Replace zeros with NaN

                # Compute normalized value, handling NaNs
                normalized = (df[col] - rolling_min) / denom
                normalized_columns[f"{col}_norm"] = normalized.fillna(
                    0.5
                )  # Default to middle if undefined

    # Create a DataFrame from all normalized columns
    normalized_df = pd.DataFrame(normalized_columns, index=df.index)

    # Join with the original DataFrame in one operation
    result = pd.concat([df_copy, normalized_df], axis=1)

    return result
