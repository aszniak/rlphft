import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats


def plot_stationary_with_fft(price_df, symbol, interval, sampling_freq=1):
    """Create a dual plot with stationary price changes and FFT analysis"""
    # Make a copy to avoid modifying original
    df = price_df.copy()

    # Calculate percentage changes
    df["pct_change"] = df["close"].pct_change()

    # Calculate relative changes (starting at 1)
    df["rel_change"] = 1.0
    for i in range(1, len(df)):
        df.loc[df.index[i], "rel_change"] = 1 + df["pct_change"].iloc[i]

    # Shift to start at 0
    df["rel_change_shifted"] = df["rel_change"] - 1

    # Create figure with two subplots (one above the other)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=False, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top subplot - Stationary price data
    ax1.plot(df["open_time"], df["rel_change_shifted"])
    ax1.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    ax1.set_title(f"{symbol} Relative Price Changes ({interval})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Relative Change (0 = no change)")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()  # Format dates on x-axis

    # Bottom subplot - FFT analysis
    # Get the FFT of the stationary data
    fft_data = fft.fft(df["rel_change_shifted"].values)

    # Get the absolute values of the FFT (magnitude)
    fft_magnitude = np.abs(fft_data)

    # Calculate frequency axis
    n = len(fft_data)
    freq = fft.fftfreq(n, d=1 / sampling_freq)

    # Only plot the positive frequencies (first half)
    pos_mask = freq >= 0
    pos_freq = freq[pos_mask]
    pos_magnitude = fft_magnitude[pos_mask]

    # Plot the FFT
    ax2.plot(pos_freq, pos_magnitude)
    ax2.set_title("Frequency Analysis (FFT)")
    ax2.set_xlabel("Frequency (cycles per data point)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, alpha=0.3)

    # Annotate the top frequencies (skip the DC component)
    top_n = 5
    top_indices = np.argsort(pos_magnitude)[
        -top_n - 1 : -1
    ]  # Skip the highest (likely DC)
    for idx in top_indices:
        if pos_freq[idx] > 0:  # Skip DC component
            period = 1 / pos_freq[idx] if pos_freq[idx] > 0 else float("inf")
            ax2.annotate(
                f"Period: {period:.1f} points",
                xy=(pos_freq[idx], pos_magnitude[idx]),
                xytext=(pos_freq[idx], pos_magnitude[idx] * 1.1),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                ha="center",
            )

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(f"{symbol}_{interval}_stationary_with_fft.png")
    plt.close()

    return df


def plot_randomness_analysis(prices, returns, title="Price Data"):
    """Create visualizations for randomness analysis"""
    plt.figure(figsize=(15, 12))

    # Price plot
    plt.subplot(3, 2, 1)
    plt.plot(prices)
    plt.title("Price Series")
    plt.grid(True)

    # Returns histogram
    plt.subplot(3, 2, 2)
    plt.hist(returns, bins=50, alpha=0.75)
    plt.title("Returns Distribution")
    plt.grid(True)

    # Returns plot
    plt.subplot(3, 2, 3)
    plt.plot(returns)
    plt.title("Returns Series")
    plt.grid(True)

    # Autocorrelation function
    plt.subplot(3, 2, 4)
    plot_acf(returns, lags=40, ax=plt.gca())
    plt.title("Autocorrelation Function")

    # Partial autocorrelation function
    plt.subplot(3, 2, 5)
    plot_pacf(returns, lags=40, ax=plt.gca())
    plt.title("Partial Autocorrelation Function")

    # QQ plot to check for normality
    plt.subplot(3, 2, 6)
    stats.probplot(returns, dist="norm", plot=plt)
    plt.title("Q-Q Plot (Testing for Normality)")

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_randomness_analysis.png")
    plt.close()


def plot_dataset_comparison(dataset_results, dataset_returns, dataset_prices):
    """Create comparison plots for all datasets"""
    # 1. Price series comparison
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)

    # Normalize prices for better comparison
    for name, prices in dataset_prices.items():
        norm_prices = prices / prices[0]
        plt.plot(norm_prices, label=name)

    plt.title("Normalized Price Series Comparison")
    plt.legend()
    plt.grid(True)

    # 2. Returns comparison
    plt.subplot(2, 1, 2)
    for name, returns in dataset_returns.items():
        plt.plot(returns, label=name, alpha=0.7)

    plt.title("Returns Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("price_returns_comparison.png")
    plt.close()

    # 3. Autocorrelation comparison
    plt.figure(figsize=(15, 10))

    for i, (name, returns) in enumerate(dataset_returns.items(), 1):
        plt.subplot(len(dataset_returns), 1, i)
        plot_acf(returns, lags=40, title=f"ACF for {name}", ax=plt.gca())
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("acf_comparison.png")
    plt.close()

    # 4. Statistical test results comparison
    test_names = ["Ljung-Box", "Runs Test", "ADF Test", "KPSS Test"]
    stat_names = ["ljung_box_stat", "runs_test_stat", "adf_stat", "kpss_stat"]
    pval_names = ["ljung_box_pvalue", "runs_test_pvalue", "adf_pvalue", "kpss_pvalue"]

    # Create a table for p-values
    plt.figure(figsize=(12, 8))

    # Collect p-values for all datasets
    datasets = list(dataset_results.keys())
    pvalues = []

    for test_idx, pval_name in enumerate(pval_names):
        test_pvals = []
        for dataset in datasets:
            test_pvals.append(dataset_results[dataset][pval_name])
        pvalues.append(test_pvals)

    # Plot as a heatmap
    plt.imshow(pvalues, cmap="coolwarm", aspect="auto", vmin=0, vmax=0.2)
    plt.colorbar(label="p-value")

    # Add grid, labels, and values
    plt.xticks(np.arange(len(datasets)), datasets, rotation=45)
    plt.yticks(np.arange(len(test_names)), test_names)

    # Add text annotations
    for i in range(len(test_names)):
        for j in range(len(datasets)):
            plt.text(
                j,
                i,
                f"{pvalues[i][j]:.4f}",
                ha="center",
                va="center",
                color="white" if pvalues[i][j] < 0.1 else "black",
            )

    plt.title("Statistical Test P-Values Comparison")
    plt.tight_layout()
    plt.savefig("statistical_tests_comparison.png")
    plt.close()


def print_comparison_summary(dataset_results):
    """Print a summary of the statistical test results for all datasets"""
    # Print summary table of results
    print("\n--- Comparison Summary ---")
    print(f"{'Test':<15} | {'Interpretation':<40}")
    print("-" * 60)

    test_names = ["Ljung-Box", "Runs Test", "ADF Test", "KPSS Test"]
    pval_names = ["ljung_box_pvalue", "runs_test_pvalue", "adf_pvalue", "kpss_pvalue"]

    interpretations = {
        "ljung_box_pvalue": {
            True: "Random (no autocorrelation)",
            False: "Not random (has autocorrelation)",
        },
        "runs_test_pvalue": {True: "Random", False: "Not random"},
        "adf_pvalue": {True: "Non-stationary", False: "Stationary"},
        "kpss_pvalue": {True: "Stationary", False: "Non-stationary"},
    }

    # Reverse the pass condition for these tests
    reverse_pass = {"adf_pvalue": True, "kpss_pvalue": False}

    datasets = list(dataset_results.keys())
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        for test_idx, (pval_name, test_name) in enumerate(zip(pval_names, test_names)):
            pval = dataset_results[dataset][pval_name]
            passed = pval > 0.05

            # Reverse interpretation for some tests
            if pval_name in reverse_pass:
                passed = not passed

            interp = interpretations[pval_name][passed]
            print(f"{test_name:<15} | {interp:<40} (p={pval:.6f})")
