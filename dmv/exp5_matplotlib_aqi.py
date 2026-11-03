"""Experiment 5: Data Visualization using matplotlib.

Aim: Analyse 180 days of city air-quality readings and visualise AQI and
pollutant trends with line, bar, stacked bar, box, violin, scatter and
bubble charts.

Dataset: synthetic city air-quality records generated in this script.
Run: python exp5_matplotlib_aqi.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_sample_data() -> None:
    np.random.seed(42)
    dates = pd.date_range(start="2026-05-01", periods=180, freq="D")
    n = len(dates)

    seasonal_factor = np.cos(np.linspace(0, 2 * np.pi, n))
    pm25 = np.clip(
        65 + 35 * seasonal_factor + np.random.normal(0, 15, n), 15, 250
    ).round(2)
    pm10 = np.clip(
        pm25 * 1.65 + np.random.normal(15, 20, n), 30, 420
    ).round(2)
    no2 = np.clip(
        32 + 12 * seasonal_factor + np.random.normal(0, 6, n), 8, 95
    ).round(2)
    co = np.clip(
        1.1 + 0.45 * seasonal_factor + np.random.normal(0, 0.2, n), 0.2, 3.5
    ).round(2)

    aqi = np.clip(
        pm25 * 1.45 + pm10 * 0.35 + no2 * 0.25 + np.random.normal(0, 8, n),
        35,
        450,
    ).round(0)

    def get_aqi_bucket(value: float) -> str:
        if value <= 50:
            return "Good"
        if value <= 100:
            return "Satisfactory"
        if value <= 200:
            return "Moderate"
        if value <= 300:
            return "Poor"
        return "Very Poor"

    aqi_category = [get_aqi_bucket(value) for value in aqi]
    dominant_pollutant = np.where(
        pm25 > 90, "PM2.5", np.where(pm10 > 150, "PM10", "NO2")
    )

    frame = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Time": np.random.choice(["08:00:00", "14:00:00", "20:00:00"], n),
            "City": "Pune",
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "CO": co,
            "AQI": aqi,
            "Dominant_Pollutant": dominant_pollutant,
            "AQI_Category": aqi_category,
        }
    )
    frame.to_csv(OUTPUT_DIR / "City_Air_Quality.csv", index=False)
    print("Sample City_Air_Quality.csv generated.\n")


def main() -> None:
    generate_sample_data()
    df = pd.read_csv(OUTPUT_DIR / "City_Air_Quality.csv")

    print("--- Task 1 & 2: Dataset Structure and Summary ---")
    print(f"Dataset shape: {df.shape}")
    print("\nData types and non-null counts:")
    print(df.info())
    print("\nDescriptive statistics:")
    print(df.describe().round(2))
    print("\nFirst 5 rows:")
    print(df.head())

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Month_Name"] = df["Date"].dt.strftime("%b %Y")

    pollutant_columns = ["PM2.5", "PM10", "NO2", "CO", "AQI"]
    df[pollutant_columns] = df[pollutant_columns].interpolate(method="linear")
    df["AQI_7Day_MA"] = df["AQI"].rolling(window=7, min_periods=1).mean()

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        df["Date"], df["AQI"], color="#adb5bd", alpha=0.6, linewidth=1.2,
        label="Daily AQI",
    )
    ax.plot(
        df["Date"], df["AQI_7Day_MA"], color="#d90429", linewidth=2.5,
        label="7-Day Moving Average",
    )
    ax.axhline(100, color="#2b9348", linestyle="--", linewidth=1.2,
               label="Satisfactory Limit (100)")
    ax.axhline(200, color="#f77f00", linestyle="--", linewidth=1.2,
               label="Moderate Limit (200)")
    ax.set_title("Overall Air Quality Index (AQI) Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp5_aqi_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("Individual Pollutant Concentration Trends", fontsize=14,
                 fontweight="bold")
    axes[0].plot(df["Date"], df["PM2.5"], color="#e63946", linewidth=1.5,
                 label="PM2.5")
    axes[0].axhline(60, color="black", linestyle=":", alpha=0.7,
                    label="Safe 24h Standard (60)")
    axes[0].set_ylabel("PM2.5")
    axes[0].set_title("Fine Particulate Matter (PM2.5) Trend")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(df["Date"], df["PM10"], color="#457b9d", linewidth=1.5,
                 label="PM10")
    axes[1].axhline(100, color="black", linestyle=":", alpha=0.7,
                    label="Safe 24h Standard (100)")
    axes[1].set_ylabel("PM10")
    axes[1].set_title("Coarse Particulate Matter (PM10) Trend")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    axes[2].plot(df["Date"], df["CO"], color="#2a9d8f", linewidth=1.5,
                 label="CO")
    axes[2].axhline(2.0, color="black", linestyle=":", alpha=0.7,
                    label="Safe 8h Standard (2.0)")
    axes[2].set_ylabel("CO")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Carbon Monoxide (CO) Trend")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp5_pollutant_trends.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    monthly = (
        df.groupby("Month", sort=True)
        .agg(
            Month_Label=("Month_Name", "first"),
            Avg_AQI=("AQI", "mean"),
            Avg_PM25=("PM2.5", "mean"),
            Avg_PM10=("PM10", "mean"),
            Avg_NO2=("NO2", "mean"),
        )
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    bars = ax1.bar(
        monthly["Month_Label"],
        monthly["Avg_AQI"],
        color=plt.cm.YlOrRd(monthly["Avg_AQI"] / monthly["Avg_AQI"].max()),
        edgecolor="black",
        width=0.6,
    )
    ax1.set_title("Monthly Average AQI Comparison")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Average AQI")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    labels = monthly["Month_Label"]
    ax2.bar(labels, monthly["Avg_PM25"], label="PM2.5", color="#e76f51",
            edgecolor="black", width=0.6)
    ax2.bar(labels, monthly["Avg_PM10"], bottom=monthly["Avg_PM25"],
            label="PM10", color="#f4a261", edgecolor="black", width=0.6)
    ax2.bar(labels, monthly["Avg_NO2"],
            bottom=monthly["Avg_PM25"] + monthly["Avg_PM10"],
            label="NO2", color="#264653", edgecolor="black", width=0.6)
    ax2.set_title("Monthly Pollutant Composition")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Cumulative Concentration")
    ax2.tick_params(axis="x", rotation=25)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp5_monthly_bars.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    pollutant_groups = ["PM2.5", "PM10", "NO2"]
    box_data = [
        df[df["Dominant_Pollutant"] == group]["AQI"].dropna().values
        for group in pollutant_groups
        if group in df["Dominant_Pollutant"].values
    ]
    valid_labels = [
        group
        for group in pollutant_groups
        if group in df["Dominant_Pollutant"].values
    ]
    try:
        box_plot = ax1.boxplot(
            box_data, tick_labels=valid_labels, patch_artist=True,
            boxprops=dict(edgecolor="black", linewidth=1.2),
            medianprops=dict(color="yellow", linewidth=2),
        )
    except TypeError:
        box_plot = ax1.boxplot(
            box_data, labels=valid_labels, patch_artist=True,
            boxprops=dict(edgecolor="black", linewidth=1.2),
            medianprops=dict(color="yellow", linewidth=2),
        )
    box_colors = ["#ef476f", "#118ab2", "#06d6a0"]
    for patch, color in zip(box_plot["boxes"], box_colors[: len(valid_labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax1.set_title("AQI by Dominant Pollutant")
    ax1.set_xlabel("Dominant Pollutant Category")
    ax1.set_ylabel("AQI Distribution")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    violin_columns = ["PM2.5", "PM10", "NO2"]
    violin_data = [df[column].dropna().values for column in violin_columns]
    violin_plot = ax2.violinplot(
        violin_data, showmeans=True, showmedians=True
    )
    for index, body in enumerate(violin_plot["bodies"]):
        body.set_facecolor(box_colors[index])
        body.set_edgecolor("black")
        body.set_alpha(0.75)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(violin_columns)
    ax2.set_title("Pollutant Concentration Distributions")
    ax2.set_xlabel("Pollutant Type")
    ax2.set_ylabel("Concentration Level")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp5_distributions.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    scatter = ax1.scatter(
        df["PM2.5"], df["AQI"], c=df["AQI"], cmap="plasma", alpha=0.8,
        edgecolors="k", linewidth=0.5, s=50,
    )
    slope, intercept = np.polyfit(df["PM2.5"], df["AQI"], 1)
    ax1.plot(
        df["PM2.5"], slope * df["PM2.5"] + intercept, color="red",
        linestyle="--", linewidth=2, label=f"Trendline (slope={slope:.2f})",
    )
    colorbar = plt.colorbar(scatter, ax=ax1)
    colorbar.set_label("AQI Intensity")
    ax1.set_title("PM2.5 vs AQI")
    ax1.set_xlabel("PM2.5 Concentration")
    ax1.set_ylabel("Air Quality Index (AQI)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    bubble = ax2.scatter(
        df["PM10"], df["AQI"], s=df["CO"] * 75, c=df["PM2.5"], cmap="viridis",
        alpha=0.75, edgecolors="black", linewidth=0.8,
    )
    colorbar = plt.colorbar(bubble, ax=ax2)
    colorbar.set_label("PM2.5 Level")
    for co_value in [0.5, 1.5, 2.5]:
        ax2.scatter([], [], c="gray", alpha=0.6, s=co_value * 75,
                    edgecolors="black", label=f"CO = {co_value} mg/m3")
    ax2.set_title("PM10 vs AQI (size = CO, color = PM2.5)")
    ax2.set_xlabel("PM10 Concentration")
    ax2.set_ylabel("Air Quality Index (AQI)")
    ax2.legend(scatterpoints=1, frameon=True, title="Bubble Size (CO)",
               loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp5_scatter_bubble.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    df.to_csv(OUTPUT_DIR / "exp5_aqi_cleaned.csv", index=False)
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
