import colorama
from colorama import Fore, Style
import datetime
import random

from data_fetcher import (
    fetch_training_data,
    prepare_multi_asset_dataset,
)


def fetch_data(
    config,
    symbols=None,
    interval=None,
    lookback_days=None,
    force_refresh=None,
    for_training=True,  # Parameter to indicate training vs evaluation
    add_sentiment=None,  # Parameter to control sentiment data inclusion
):
    """
    Fetch and prepare data for training or evaluation.

    Args:
        config: Configuration object
        symbols: List of cryptocurrency symbols to fetch (overrides config)
        interval: Data interval (overrides config)
        lookback_days: Days of historical data to fetch (overrides config)
        force_refresh: Whether to force refresh cached data (overrides config)
        for_training: Whether data is for training (True) or evaluation (False)
        add_sentiment: Whether to add sentiment data (overrides config)

    Returns:
        Dictionary of processed data with technical indicators
    """
    # Use provided parameters or fall back to config values
    symbols = symbols or config.symbols
    interval = interval or config.interval

    # Use provided sentiment setting or fall back to config
    if add_sentiment is None:
        add_sentiment = getattr(config, "use_sentiment", False)

    # Choose appropriate lookback period based on whether we're training or evaluating
    if lookback_days is None:
        lookback_days = (
            config.training_lookback_days if for_training else config.eval_lookback_days
        )

    force_refresh = force_refresh if force_refresh is not None else config.force_refresh

    # For evaluation, we can select a random period if configured
    random_start_date = None
    if not for_training and config.random_eval_period:
        # Calculate a random start date within the last year (adjust as needed)
        # This gives us a random 30-day window for evaluation
        now = datetime.datetime.now()
        max_days_ago = 365 * 2  # Expand to 2 years maximum (was 365 days)
        min_days_ago = (
            config.eval_lookback_days + 30
        )  # At least 30 days after minimum lookback
        random_days = random.randint(min_days_ago, max_days_ago)
        random_start_date = now - datetime.timedelta(days=random_days)
        print(
            f"{Fore.CYAN}🎲 Using random evaluation period starting {random_days} days ago ({random_start_date.strftime('%Y-%m-%d')}){Style.RESET_ALL}"
        )

    print(
        f"{Fore.CYAN}🔍 Fetching {'training' if for_training else 'evaluation'} data for cryptocurrencies...{Style.RESET_ALL}"
    )

    # Update the fetch_training_data function call to include the random_start_date parameter
    data_dict = fetch_training_data(
        symbols=symbols,
        interval=interval,
        lookback_days=lookback_days,
        force_refresh=force_refresh
        or (
            not for_training and config.random_eval_period
        ),  # Force refresh for random eval periods
        start_date=random_start_date,
        cache_suffix=(
            f"_{random_days}days_ago" if random_start_date else None
        ),  # Add date-specific suffix to cache
    )

    # Prepare dataset with additional features
    print(
        f"{Fore.CYAN}📊 Adding technical indicators and normalizing features...{Style.RESET_ALL}"
    )
    enhanced_data, combined_df = prepare_multi_asset_dataset(
        data_dict,
        add_indicators=True,
        add_sentiment=add_sentiment,
        force_refresh=force_refresh,
    )

    # Print summary of the data
    print(f"\n{Fore.GREEN}📈 Data Summary:{Style.RESET_ALL}")
    for symbol, df in enhanced_data.items():
        print(
            f"{Fore.YELLOW}{symbol}{Style.RESET_ALL}: {len(df)} candles from {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}"
        )

        # Print sample of technical indicator columns
        tech_indicators = [
            col
            for col in df.columns
            if any(
                indicator in col for indicator in ["sma", "ema", "rsi", "macd", "bb_"]
            )
        ]
        if tech_indicators:
            print(
                f"  - Technical indicators include: {', '.join(tech_indicators[:5])}..."
            )

        # Print sample of normalized columns
        norm_cols = [col for col in df.columns if col.endswith("_norm")]
        if norm_cols:
            print(f"  - Normalized features include: {', '.join(norm_cols[:5])}...")

        # Print sentiment indicators if available
        sentiment_cols = [
            col
            for col in df.columns
            if any(
                s in col
                for s in ["sentiment", "social", "github", "dev_activity", "exchange"]
            )
        ]
        if sentiment_cols:
            print(f"  - Sentiment features include: {', '.join(sentiment_cols[:5])}...")

    return enhanced_data
