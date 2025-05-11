import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import random
import argparse
from main import main as fetch_data
from matplotlib.gridspec import GridSpec

# Set style for plots
plt.style.use("seaborn-v0_8-darkgrid")


def plot_technical_indicators(df, days=6, symbol=None):
    """
    Plot technical indicators for a specific symbol and time window.

    Args:
        df: DataFrame with price data and technical indicators
        days: Number of days to plot
        symbol: Symbol name for the title
    """
    # Calculate the number of minutes in the requested days
    minutes_in_day = 60 * 24
    window_size = minutes_in_day * days

    # If df has more than window_size rows, select a random window
    if len(df) > window_size:
        start_idx = random.randint(0, len(df) - window_size - 1)
        end_idx = start_idx + window_size
        plot_df = df.iloc[start_idx:end_idx].copy()
    else:
        plot_df = df.copy()

    # Create figure and grid layout for subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1])

    # Subplot 1: Price with SMA, EMA, and Bollinger Bands
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(
        plot_df["open_time"],
        plot_df["close"],
        label="Close Price",
        color="blue",
        linewidth=1.5,
    )
    ax1.plot(
        plot_df["open_time"],
        plot_df["sma_20"],
        label="SMA(20)",
        color="red",
        linestyle="--",
        alpha=0.7,
    )
    ax1.plot(
        plot_df["open_time"],
        plot_df["ema_20"],
        label="EMA(20)",
        color="orange",
        linestyle="--",
        alpha=0.7,
    )
    ax1.plot(
        plot_df["open_time"],
        plot_df["bb_upper"],
        label="BB Upper",
        color="green",
        linestyle=":",
        alpha=0.6,
    )
    ax1.plot(
        plot_df["open_time"],
        plot_df["bb_middle"],
        label="BB Middle",
        color="green",
        linestyle="-",
        alpha=0.6,
    )
    ax1.plot(
        plot_df["open_time"],
        plot_df["bb_lower"],
        label="BB Lower",
        color="green",
        linestyle=":",
        alpha=0.6,
    )

    # Highlight volume spikes, gaps, and divergence points
    volume_spike_idx = plot_df[plot_df["volume_spike"] == 1].index
    gap_up_idx = plot_df[plot_df["gap_up"] == 1].index
    gap_down_idx = plot_df[plot_df["gap_down"] == 1].index
    divergence_idx = plot_df[plot_df["price_rsi_divergence"] == 1].index

    # Plot markers for these events
    ax1.scatter(
        plot_df.loc[volume_spike_idx, "open_time"],
        plot_df.loc[volume_spike_idx, "close"],
        color="purple",
        marker="^",
        s=100,
        label="Volume Spike",
    )
    ax1.scatter(
        plot_df.loc[gap_up_idx, "open_time"],
        plot_df.loc[gap_up_idx, "close"],
        color="darkgreen",
        marker=">",
        s=100,
        label="Gap Up",
    )
    ax1.scatter(
        plot_df.loc[gap_down_idx, "open_time"],
        plot_df.loc[gap_down_idx, "close"],
        color="darkred",
        marker="v",
        s=100,
        label="Gap Down",
    )
    ax1.scatter(
        plot_df.loc[divergence_idx, "open_time"],
        plot_df.loc[divergence_idx, "close"],
        color="magenta",
        marker="*",
        s=150,
        label="RSI Divergence",
    )

    ax1.set_title(f"{symbol} Price and Indicators - {days} Day Window", fontsize=16)
    ax1.set_ylabel("Price", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Volume with SMA
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    # Calculate time delta between points for width calculation
    time_delta = (
        plot_df["open_time"].iloc[1] - plot_df["open_time"].iloc[0]
    ).total_seconds()
    width_fraction = 0.8  # Use 80% of the time interval for each bar
    width = pd.Timedelta(seconds=time_delta * width_fraction)

    # Plot volume with more visible bars
    ax2.bar(
        plot_df["open_time"],
        plot_df["volume"],
        label="Volume",
        alpha=0.7,
        color="blue",
        width=width,  # Use calculated width
    )

    ax2.plot(
        plot_df["open_time"],
        plot_df["volume_sma_20"],
        label="Volume SMA(20)",
        color="red",
        linewidth=1.5,
    )
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Adjust y-axis to show volume clearly
    ymin, ymax = ax2.get_ylim()
    if ymax > 0:
        ax2.set_ylim(0, ymax * 1.1)  # Add 10% padding to top

    # Subplot 3: RSI and Stochastic
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(plot_df["open_time"], plot_df["rsi_14"], label="RSI(14)", color="purple")
    ax3.plot(
        plot_df["open_time"],
        plot_df["stoch_k"],
        label="Stoch %K",
        color="blue",
        alpha=0.7,
    )
    ax3.plot(
        plot_df["open_time"],
        plot_df["stoch_d"],
        label="Stoch %D",
        color="red",
        alpha=0.7,
    )
    ax3.axhline(y=70, color="red", linestyle="--", alpha=0.5)
    ax3.axhline(y=30, color="green", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Oscillator Values", fontsize=12)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Subplot 4: MACD
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # Create copy of MACD data with more reasonable scaling for plotting
    macd = plot_df["macd"].copy()
    macd_signal = plot_df["macd_signal"].copy()
    macd_hist = plot_df["macd_hist"].copy()

    # Scale MACD if it's unusually large
    max_abs_macd = max(
        abs(macd.max()), abs(macd.min()), abs(macd_signal.max()), abs(macd_signal.min())
    )

    if max_abs_macd > 100:  # If MACD values are very large, scale them
        scale_factor = max_abs_macd / 10  # Scale to keep max around 10
        macd = macd / scale_factor
        macd_signal = macd_signal / scale_factor
        macd_hist = macd_hist / scale_factor
        ax4.set_ylabel(f"MACD (scaled ÷{scale_factor:.1f})", fontsize=12)
    else:
        ax4.set_ylabel("MACD", fontsize=12)

    # Plot MACD components with better scaling
    ax4.plot(plot_df["open_time"], macd, label="MACD", color="blue")
    ax4.plot(plot_df["open_time"], macd_signal, label="Signal", color="red")
    ax4.bar(
        plot_df["open_time"],
        macd_hist,
        label="Histogram",
        alpha=0.5,
        color="green",
        width=width,  # Use same width as volume bars for consistency
    )

    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    # Subplot 5: ATR and Z-Score
    ax5 = fig.add_subplot(gs[4], sharex=ax1)

    # Scale ATR if it's unusually large
    atr = plot_df["atr_14"].copy()
    z_score = plot_df["price_zscore_20"].copy()

    max_atr = atr.max()
    if max_atr > 100:  # If ATR values are very large, scale them
        scale_factor = max_atr / 20  # Scale to keep max around 20
        atr = atr / scale_factor
        ax5.plot(
            plot_df["open_time"],
            atr,
            label=f"ATR(14) (scaled ÷{scale_factor:.1f})",
            color="brown",
        )
    else:
        ax5.plot(plot_df["open_time"], atr, label="ATR(14)", color="brown")

    # Z-score usually doesn't need scaling as it's already normalized
    ax5.plot(
        plot_df["open_time"],
        z_score,
        label="Price Z-Score",
        color="green",
    )

    ax5.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax5.axhline(y=2, color="red", linestyle="--", alpha=0.5)
    ax5.axhline(y=-2, color="green", linestyle="--", alpha=0.5)
    ax5.set_ylabel("Volatility", fontsize=12)
    ax5.legend(loc="upper left")
    ax5.grid(True, alpha=0.3)

    # Set reasonable y-limits for z-score plot
    z_min, z_max = min(-3, z_score.min()), max(3, z_score.max())
    atr_max = atr.max()
    ax5.set_ylim(min(-3, z_min * 1.1), max(atr_max * 1.1, z_max * 1.1))

    # Format x-axis for readability
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    return fig


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Visualize technical indicators for cryptocurrency data"
    )
    parser.add_argument(
        "--symbol", type=str, help="Symbol to visualize (e.g., BTCUSDT, ETHUSDT)"
    )
    parser.add_argument(
        "--days", type=int, default=6, help="Number of days to visualize"
    )
    args = parser.parse_args()

    # Fetch data
    data_dict, enhanced_data, _ = fetch_data()

    # Pick a symbol - either user specified or random
    symbols = list(enhanced_data.keys())
    if not symbols:
        print("No data available. Please run main.py first.")
        return

    if args.symbol and args.symbol in symbols:
        symbol = args.symbol
    else:
        if args.symbol and args.symbol not in symbols:
            print(f"Symbol {args.symbol} not found. Available symbols: {symbols}")
        symbol = random.choice(symbols)

    print(f"Plotting technical indicators for {symbol}...")

    # Plot the indicators
    fig = plot_technical_indicators(
        enhanced_data[symbol], days=args.days, symbol=symbol
    )

    # Save the plot
    output_file = f"technical_indicators_{symbol}_{args.days}days.png"
    fig.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
