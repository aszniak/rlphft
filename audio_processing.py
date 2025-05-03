import numpy as np
from scipy.io import wavfile
import pandas as pd


def create_audio_from_prices_no_resample(
    price_df, filename="bitcoin_audio_no_resample.wav", sample_rate=44100
):
    """Convert price data to audio file without resampling

    This version uses exactly sample_rate data points to create 1 second of audio
    """
    print(f"Original data has {len(price_df)} points")

    # Determine how many points to use
    if len(price_df) >= sample_rate:
        # If we have more points than needed, use the most recent ones
        df = price_df.tail(sample_rate).copy()
        print(f"Using the most recent {sample_rate} points")
    else:
        # If we have fewer points than needed, use what we have
        df = price_df.copy()
        print(f"Warning: Only have {len(df)} points, less than ideal {sample_rate}")

    # Calculate percentage changes
    df["pct_change"] = df["close"].pct_change()

    # Calculate relative changes (starting at 1)
    df["rel_change"] = 1.0
    for i in range(1, len(df)):
        df.loc[df.index[i], "rel_change"] = 1 + df["pct_change"].iloc[i]

    # Shift to start at 0
    df["rel_change_shifted"] = df["rel_change"] - 1

    # Use the rel_change_shifted column for audio
    audio_data = df["rel_change_shifted"].fillna(0).values

    # Normalize the data to range -1 to 1 for audio
    max_val = max(abs(audio_data.min()), abs(audio_data.max()))
    audio_data = audio_data / max_val if max_val > 0 else audio_data

    # Convert to 16-bit PCM
    audio_samples = np.int16(audio_data * 32767)

    # Save as WAV file
    wavfile.write(filename, sample_rate, audio_samples)

    print(f"Created audio file: {filename}")
    print(f"Duration: {len(audio_samples)/sample_rate:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")

    return filename


def create_5sec_audio_sample(
    price_df, filename="bitcoin_5sec_sample.wav", sample_rate=44100
):
    """Create a 5-second audio sample with 44.1kHz sample rate from price data

    Uses exactly 5*sample_rate data points to create 5 seconds of audio
    """
    # Calculate how many points we need for 5 seconds
    points_needed = 5 * sample_rate
    print(
        f"Creating a 5-second sample at {sample_rate}Hz (need {points_needed} points)"
    )

    # Verify we have enough data
    total_points = len(price_df)
    print(f"Original data has {total_points} points")

    if total_points < points_needed:
        print(
            f"Warning: Not enough data points. Have {total_points}, need {points_needed}"
        )
        # Use what we have
        df = price_df.copy()
    else:
        # Select the most recent points_needed data points
        df = price_df.tail(points_needed).copy()
        print(f"Using the most recent {points_needed} points")

    # Calculate percentage changes
    df["pct_change"] = df["close"].pct_change()

    # Calculate relative changes (starting at 1)
    df["rel_change"] = 1.0
    for i in range(1, len(df)):
        df.loc[df.index[i], "rel_change"] = 1 + df["pct_change"].iloc[i]

    # Shift to start at 0
    df["rel_change_shifted"] = df["rel_change"] - 1

    # Use the rel_change_shifted column for audio
    audio_data = df["rel_change_shifted"].fillna(0).values

    # Normalize the data to range -1 to 1 for audio
    max_val = max(abs(audio_data.min()), abs(audio_data.max()))
    audio_data = audio_data / max_val if max_val > 0 else audio_data

    # Convert to 16-bit PCM
    audio_samples = np.int16(audio_data * 32767)

    # Save as WAV file
    wavfile.write(filename, sample_rate, audio_samples)

    print(f"Created audio file: {filename}")
    print(f"Duration: {len(audio_samples)/sample_rate:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")

    return filename
