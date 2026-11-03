"""Experiment 2 - Uber fare prediction with Linear, Ridge and Lasso regression.

Aim:
    Clean the Uber fares data, detect fare outliers with the IQR rule, inspect
    correlations, and compare Linear Regression, Ridge (L2) and Lasso (L1).

Dataset:
    Kaggle "Uber Fares" CSV (200,000 rows). The file is not committed to this
    repository. Download it from:
    https://www.kaggle.com/datasets/yasserh/uber-fares-dataset
    Then either place it at data/uber.csv or set the UBER_CSV environment
    variable to its full path.

Run:
    python exp2_uber_regression.py

Outputs:
    Console: cleaning counts, correlation with fare, model comparison table.
    Files:   outputs/exp2_fare_boxplot.png, outputs/exp2_correlation_heatmap.png
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "uber.csv"
DATA_PATH = Path(os.environ.get("UBER_CSV", DEFAULT_DATA_PATH))
TARGET = "fare_amount"


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    print("Original shape:", df.shape)

    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET}'. Found: {list(df.columns)}"
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    df = df[df[TARGET] >= 0]

    for col, lo, hi in [
        ("pickup_longitude", -180, 180),
        ("dropoff_longitude", -180, 180),
        ("pickup_latitude", -90, 90),
        ("dropoff_latitude", -90, 90),
    ]:
        if col in df.columns:
            df = df[df[col].between(lo, hi)]

    if "passenger_count" in df.columns:
        df = df[df["passenger_count"].between(1, 6)]

    if "pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
        df["pickup_hour"] = dt.dt.hour
        df["pickup_day"] = dt.dt.day
        df["pickup_month"] = dt.dt.month
        df["pickup_weekday"] = dt.dt.dayofweek
        df = df.drop(columns=["pickup_datetime"])

    df = df.select_dtypes(include="number").dropna(subset=[TARGET])
    return df


def main() -> None:
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Download the Uber Fares CSV from:")
        print("  https://www.kaggle.com/datasets/yasserh/uber-fares-dataset")
        print("Then place it at data/uber.csv or set UBER_CSV to its full path.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_and_clean()

    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_mask = (df[TARGET] < q1 - 1.5 * iqr) | (df[TARGET] > q3 + 1.5 * iqr)
    print("Rows after basic cleaning:", len(df))
    print("Fare outliers by IQR rule:", int(outlier_mask.sum()))
    print("\nSummary statistics:")
    print(df.describe().T.round(2))

    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df[TARGET])
    plt.title("Fare amount - outlier inspection")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp2_fare_boxplot.png", dpi=150)
    plt.close()

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp2_correlation_heatmap.png", dpi=150)
    plt.close()

    print("\nCorrelation with fare:")
    print(corr[TARGET].sort_values(ascending=False).round(6).to_string())

    # Sample large files for a practical runtime.
    if len(df) > 100_000:
        df = df.sample(100_000, random_state=42)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
    }

    rows = []
    for name, estimator in models.items():
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append(
            {
                "Model": name,
                "R2": r2_score(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "MAE": mean_absolute_error(y_test, pred),
            }
        )

    results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    print("\nModel comparison:")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
