"""Experiment 2: Interacting with Web APIs.

Aim: Request a 5-day / 3-hour weather forecast from OpenWeatherMap,
flatten the JSON into a DataFrame, clean it, aggregate it and visualise it.

Dataset: live OpenWeatherMap API if OPENWEATHER_API_KEY is set; otherwise a
synthetic response with the same schema so the experiment runs offline.

Run: python exp2_web_api_weather.py
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_CITY = "Pune,IN"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def generate_fallback_json(city: str) -> dict:
    """Replicate the OpenWeatherMap 5-day / 3-hour response schema."""
    np.random.seed(42)
    # Fixed start keeps the offline fallback reproducible and consistent with the notes.
    base_time = datetime(2026, 10, 19, 0, 0, 0)
    forecast_list = []

    for index in range(40):
        timestamp = base_time + timedelta(hours=3 * index)
        hour_factor = np.sin((timestamp.hour - 6) * np.pi / 12)
        temperature = round(26 + 5 * hour_factor + np.random.normal(0, 0.8), 2)
        humidity = int(
            np.clip(65 - 20 * hour_factor + np.random.normal(0, 5), 25, 98)
        )
        rain_value = (
            round(np.random.exponential(1.8), 2)
            if humidity > 75 and np.random.rand() > 0.4
            else 0.0
        )

        entry = {
            "dt_txt": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "main": {
                "temp": temperature,
                "feels_like": round(temperature + (0.05 * humidity - 2), 2),
                "temp_min": round(temperature - np.random.uniform(0.5, 1.2), 2),
                "temp_max": round(temperature + np.random.uniform(0.5, 1.2), 2),
                "pressure": int(np.random.normal(1011, 3)),
                "humidity": humidity,
            },
            "wind": {"speed": round(np.random.uniform(1.2, 7.5), 2)},
            "clouds": {"all": int(np.clip(humidity * 0.9, 0, 100))},
            "weather": [
                {
                    "main": "Rain" if rain_value > 0 else "Clouds",
                    "description": (
                        "light rain" if rain_value > 0 else "scattered clouds"
                    ),
                }
            ],
        }
        if rain_value > 0:
            entry["rain"] = {"3h": rain_value}
        forecast_list.append(entry)

    return {
        "city": {
            "name": city.split(",")[0],
            "coord": {"lat": 18.5204, "lon": 73.8567},
        },
        "list": forecast_list,
    }


def fetch_weather_forecast(city: str, api_key: str | None) -> dict:
    if not api_key:
        print("OPENWEATHER_API_KEY not set; using synthetic forecast data.")
        return generate_fallback_json(city)

    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=10)
        if response.status_code == 200:
            print(f"Retrieved live API data for {city}.")
            return response.json()
        print(f"API returned status {response.status_code}; using synthetic data.")
    except requests.RequestException as exc:
        print(f"Network error ({exc}); using synthetic data.")
    return generate_fallback_json(city)


def main() -> None:
    raw_json = fetch_weather_forecast(
        TARGET_CITY, os.getenv("OPENWEATHER_API_KEY")
    )

    records = []
    for item in raw_json["list"]:
        records.append(
            {
                "Timestamp": item.get("dt_txt"),
                "Temperature_C": item["main"].get("temp"),
                "Feels_Like_C": item["main"].get("feels_like"),
                "Temp_Min_C": item["main"].get("temp_min"),
                "Temp_Max_C": item["main"].get("temp_max"),
                "Humidity_Pct": item["main"].get("humidity"),
                "Pressure_hPa": item["main"].get("pressure"),
                "Wind_Speed_ms": item["wind"].get("speed"),
                "Cloudiness_Pct": item["clouds"].get("all"),
                "Precipitation_mm": item.get("rain", {}).get("3h", np.nan),
                "Weather_Condition": item["weather"][0].get("main"),
            }
        )

    weather_df = pd.DataFrame(records)
    weather_df["Timestamp"] = pd.to_datetime(weather_df["Timestamp"])
    weather_df["Precipitation_mm"] = weather_df["Precipitation_mm"].fillna(0.0)

    numeric_cols = [
        "Temperature_C",
        "Feels_Like_C",
        "Humidity_Pct",
        "Wind_Speed_ms",
        "Pressure_hPa",
    ]
    weather_df[numeric_cols] = weather_df[numeric_cols].interpolate(
        method="linear"
    )
    weather_df = (
        weather_df.sort_values("Timestamp")
        .drop_duplicates(subset=["Timestamp"])
        .reset_index(drop=True)
    )

    print("\n--- Cleaned Weather DataFrame ---")
    print(weather_df.info())
    print(weather_df.head(3))

    weather_df["Temp_Rolling_24h_Avg"] = (
        weather_df["Temperature_C"].rolling(window=8, min_periods=1).mean()
    )
    time_index = np.arange(len(weather_df))
    slope, intercept = np.polyfit(
        time_index, weather_df["Temperature_C"], 1
    )
    weather_df["Linear_Trend"] = slope * time_index + intercept

    print("\n--- Overall Weather Pattern Summary ---")
    print(f"Average Temperature : {weather_df['Temperature_C'].mean():.2f} C")
    print(f"Maximum Temperature : {weather_df['Temperature_C'].max():.2f} C")
    print(f"Minimum Temperature : {weather_df['Temperature_C'].min():.2f} C")
    print(f"Average Humidity    : {weather_df['Humidity_Pct'].mean():.1f} %")
    print(f"Total Precipitation : {weather_df['Precipitation_mm'].sum():.2f} mm")
    print(f"Max Wind Speed      : {weather_df['Wind_Speed_ms'].max():.2f} m/s")

    weather_df["Date"] = weather_df["Timestamp"].dt.date
    weather_df["Hour"] = weather_df["Timestamp"].dt.hour

    def assign_time_of_day(hour: int) -> str:
        if 6 <= hour < 12:
            return "1_Morning"
        if 12 <= hour < 17:
            return "2_Afternoon"
        if 17 <= hour < 21:
            return "3_Evening"
        return "4_Night"

    weather_df["Time_Of_Day"] = weather_df["Hour"].apply(assign_time_of_day)

    daily_summary = (
        weather_df.groupby("Date")
        .agg(
            Avg_Temp_C=("Temperature_C", "mean"),
            Min_Temp_C=("Temp_Min_C", "min"),
            Max_Temp_C=("Temp_Max_C", "max"),
            Avg_Humidity_Pct=("Humidity_Pct", "mean"),
            Total_Rain_mm=("Precipitation_mm", "sum"),
            Max_Wind_ms=("Wind_Speed_ms", "max"),
        )
        .round(2)
        .reset_index()
    )
    print("\n--- Daily Aggregated Weather Statistics ---")
    print(daily_summary.to_string(index=False))

    tod_summary = (
        weather_df.groupby("Time_Of_Day")[
            ["Temperature_C", "Humidity_Pct", "Wind_Speed_ms"]
        ]
        .mean()
        .round(2)
    )
    print("\n--- Time-of-Day Aggregation ---")
    print(tod_summary)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"OpenWeatherMap Data Modeling and Analysis ({TARGET_CITY})",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].plot(
        weather_df["Timestamp"],
        weather_df["Temperature_C"],
        marker="o",
        label="3-Hourly Temp (C)",
        color="#e63946",
        alpha=0.8,
    )
    axes[0, 0].plot(
        weather_df["Timestamp"],
        weather_df["Temp_Rolling_24h_Avg"],
        linestyle="--",
        linewidth=2.5,
        label="24h Rolling Mean",
        color="#1d3557",
    )
    axes[0, 0].set_title("Temperature Trend Over Time")
    axes[0, 0].set_xlabel("Date and Time")
    axes[0, 0].set_ylabel("Temperature (C)")
    axes[0, 0].tick_params(axis="x", rotation=30)
    axes[0, 0].legend()

    sns.barplot(
        data=daily_summary,
        x="Date",
        y="Total_Rain_mm",
        hue="Date",
        palette="Blues_d",
        ax=axes[0, 1],
        legend=False,
    )
    axes[0, 1].set_title("Daily Accumulated Precipitation (mm)")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Total Rainfall (mm)")
    axes[0, 1].tick_params(axis="x", rotation=15)

    scatter = axes[1, 0].scatter(
        weather_df["Temperature_C"],
        weather_df["Humidity_Pct"],
        c=weather_df["Wind_Speed_ms"],
        s=weather_df["Cloudiness_Pct"] + 30,
        cmap="viridis",
        alpha=0.85,
        edgecolors="k",
    )
    colorbar = plt.colorbar(scatter, ax=axes[1, 0])
    colorbar.set_label("Wind Speed (m/s)")
    axes[1, 0].set_title("Temperature vs Humidity")
    axes[1, 0].set_xlabel("Temperature (C)")
    axes[1, 0].set_ylabel("Relative Humidity (%)")

    corr_features = [
        "Temperature_C",
        "Feels_Like_C",
        "Humidity_Pct",
        "Pressure_hPa",
        "Wind_Speed_ms",
        "Precipitation_mm",
    ]
    sns.heatmap(
        weather_df[corr_features].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Correlation Matrix of Weather Attributes")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp2_weather_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    weather_df.to_csv(OUTPUT_DIR / "exp2_weather_cleaned.csv", index=False)

    try:
        import folium

        city_info = raw_json.get("city", {})
        coords = city_info.get("coord", {})
        weather_map = folium.Map(
            location=[coords.get("lat", 18.5204), coords.get("lon", 73.8567)],
            zoom_start=10,
        )
        weather_map.save(OUTPUT_DIR / "exp2_weather_map.html")
        print("Folium map saved.")
    except ImportError:
        print("folium not installed; skipping the HTML map.")

    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
