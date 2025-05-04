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
    df = df.copy()

    # Simple Moving Averages
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()

    # Exponential Moving Averages
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Moving Average Convergence Divergence (MACD)
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

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
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

    # Average True Range (ATR)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14).mean()

    # Volume indicators
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # Price change rate
    df["close_pct_change_1"] = df["close"].pct_change(periods=1)
    df["close_pct_change_5"] = df["close"].pct_change(periods=5)

    # Volatility (standard deviation of returns)
    df["volatility_5"] = df["close_pct_change_1"].rolling(window=5).std()
    df["volatility_20"] = df["close_pct_change_1"].rolling(window=20).std()

    # Momentum indicators
    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

    # Stochastic Oscillator
    n = 14
    df["stoch_k"] = (
        (df["close"] - df["low"].rolling(window=n).min())
        / (df["high"].rolling(window=n).max() - df["low"].rolling(window=n).min())
    ) * 100
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # On-Balance Volume (OBV)
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # Add market inefficiency indicators that might be useful for RL agent

    # Mean reversion potential (z-score of price)
    df["price_zscore_20"] = (df["close"] - df["sma_20"]) / df["bb_std"]

    # Momentum divergence (price making new highs but RSI not)
    df["close_5d_high"] = df["close"].rolling(window=5).max()
    df["rsi_5d_high"] = df["rsi_14"].rolling(window=5).max()
    df["price_rsi_divergence"] = (
        (df["close"] == df["close_5d_high"]) & (df["rsi_14"] < df["rsi_5d_high"] * 0.95)
    ).astype(int)

    # Volatility expansion (ATR increasing)
    df["atr_expansion"] = df["atr_14"] / df["atr_14"].shift(5)

    # Volume spike detection
    df["volume_spike"] = (df["volume"] > df["volume_sma_20"] * 2).astype(int)

    # Gap detection
    df["gap_up"] = (df["open"] > df["close"].shift(1) * 1.01).astype(int)
    df["gap_down"] = (df["open"] < df["close"].shift(1) * 0.99).astype(int)

    return df


def normalize_features(df: pd.DataFrame, feature_cols=None) -> pd.DataFrame:
    """
    Normalize features to [0, 1] or [-1, 1] range depending on the feature.

    Args:
        df: DataFrame with features
        feature_cols: List of columns to normalize (if None, normalize all except time columns)

    Returns:
        DataFrame with normalized features
    """
    df_normalized = df.copy()

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
            df_normalized[f"{col}_norm"] = df[col] / 100

        # For z-score features that are already normalized
        elif "zscore" in col.lower():
            # Already normalized but clip to reasonable range
            df_normalized[f"{col}_norm"] = np.clip(df[col], -3, 3) / 3

        # For percentage changes
        elif "pct" in col.lower() or "momentum" in col.lower():
            # Normalize to [-1, 1] range with reasonable bounds
            df_normalized[f"{col}_norm"] = np.clip(df[col], -0.1, 0.1) / 0.1

        # For binary indicators
        elif set(df[col].unique()).issubset({0, 1, 0.0, 1.0}):
            df_normalized[f"{col}_norm"] = df[col]

        # For other features, use Min-Max scaling with rolling window
        else:
            # Handle special case for volume-related features
            if "volume" in col.lower():
                # Volume can spike dramatically, so use log transformation
                log_values = np.log1p(df[col])
                max_val = np.log1p(df[col].rolling(window=50).max())
                # Avoid division by zero
                max_val = max_val.replace(0, np.nan)
                df_normalized[f"{col}_norm"] = log_values / max_val
                df_normalized[f"{col}_norm"] = df_normalized[f"{col}_norm"].fillna(0.5)
            else:
                # Use rolling min-max to adapt to changing scales
                rolling_min = df[col].rolling(window=100, min_periods=20).min()
                rolling_max = df[col].rolling(window=100, min_periods=20).max()

                # Handle cases where min equals max
                denom = rolling_max - rolling_min
                denom = denom.replace(0, np.nan)  # Replace zeros with NaN

                # Compute normalized value, handling NaNs
                normalized = (df[col] - rolling_min) / denom
                df_normalized[f"{col}_norm"] = normalized.fillna(
                    0.5
                )  # Default to middle if undefined

    return df_normalized
