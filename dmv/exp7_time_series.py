"""Experiment 13 (DMV Lab 13) - Stock Market Time Series Analysis.

Aim
    Analyse daily stock prices: plot the trend, smooth it with moving
    averages, decompose trend/seasonality/residual, check the Close-Volume
    correlation, and forecast the next 30 days with ARIMA.

Dataset
    A deterministic synthetic daily stock series generated in this script
    (420 business days). No external file is needed.

Outputs
    outputs/exp7_stock_prices.csv
    outputs/exp7_closing_price.png
    outputs/exp7_moving_averages.png
    outputs/exp7_seasonal_decomposition.png
    outputs/exp7_arima_forecast.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:  # pragma: no cover - only hit when statsmodels is missing
    ARIMA = None
    seasonal_decompose = None

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def make_stock_data(periods: int = 420, seed: int = 42) -> pd.DataFrame:
    """Build a realistic-enough daily series: trend + monthly cycle + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=periods)

    trend = np.linspace(100.0, 180.0, periods)
    seasonality = 8.0 * np.sin(2 * np.pi * np.arange(periods) / 21)
    noise = rng.normal(0.0, 4.0, periods)
    close = np.maximum(10.0, trend + seasonality + noise).round(2)

    # Volume rises with the price level, plus random trading noise.
    volume = (
        1_000_000 + 5_000 * (close - close.min()) + rng.normal(0, 250_000, periods)
    ).clip(100_000).astype(int)

    return pd.DataFrame({"Date": dates, "Close": close, "Volume": volume})


def main() -> None:
    if ARIMA is None or seasonal_decompose is None:
        print("statsmodels is required for this experiment.")
        print("Install it with: pip install statsmodels")
        return

    df = make_stock_data()
    df.to_csv(OUTPUT_DIR / "exp7_stock_prices.csv", index=False)

    # A DatetimeIndex is what makes .rolling, .plot and ARIMA time-aware.
    df = df.set_index("Date").sort_index()

    print("Rows:", len(df), "| Date range:", df.index.min().date(), "to", df.index.max().date())
    print("\nFirst five rows:\n", df.head().to_string())
    print("\nSummary:\n", df.describe().round(2).to_string())

    # 1. Closing price trend
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], color="tab:blue", linewidth=1)
    ax.set_title("Closing price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp7_closing_price.png", dpi=150)
    plt.close(fig)

    # 2. Moving averages smooth short-term noise.
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["MA90"] = df["Close"].rolling(window=90).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], label="Close", alpha=0.45, linewidth=1)
    ax.plot(df.index, df["MA30"], label="30-day MA", linewidth=1.5)
    ax.plot(df.index, df["MA90"], label="90-day MA", linewidth=1.5)
    ax.set_title("Closing price with moving averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp7_moving_averages.png", dpi=150)
    plt.close(fig)

    # 3. Split the series into trend, seasonal cycle and residual.
    decomposition = seasonal_decompose(df["Close"], model="additive", period=21)
    fig = decomposition.plot()
    fig.set_size_inches(10, 7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp7_seasonal_decomposition.png", dpi=150)
    plt.close(fig)

    # 4. Correlation between closing price and volume.
    correlation = df["Close"].corr(df["Volume"])
    print(f"\nCorrelation Close vs Volume: {correlation:.3f}")

    # 5. Time-ordered train/test split - never shuffle time series data.
    train = df["Close"].iloc[:-30]
    test = df["Close"].iloc[-30:]
    model = ARIMA(train, order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=len(test))

    mae = float(np.mean(np.abs(test.to_numpy() - forecast.to_numpy())))
    rmse = float(np.sqrt(np.mean((test.to_numpy() - forecast.to_numpy()) ** 2)))
    print(f"\nTrain size: {len(train)} | Test size: {len(test)}")
    print(f"ARIMA(5,1,0) forecast - MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    print("First five forecasts:\n", forecast.head().round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train.index[-120:], train.iloc[-120:], label="Train")
    ax.plot(test.index, test, label="Actual", color="tab:orange")
    ax.plot(test.index, forecast, label="Forecast", color="tab:green", linestyle="--")
    ax.set_title("ARIMA(5,1,0) - 30-day forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp7_arima_forecast.png", dpi=150)
    plt.close(fig)

    print("\nOutputs written to", OUTPUT_DIR)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
