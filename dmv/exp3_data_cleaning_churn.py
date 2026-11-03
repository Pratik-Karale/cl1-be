"""Experiment 3: Data Cleaning and Preparation.

Aim: Clean a messy telecom customer-churn dataset: remove duplicates,
standardise text, fix data types, impute missing values, cap outliers,
engineer features, encode, scale and split into train/test sets.

Dataset: synthetic telecom records generated in this script.
Run: python exp3_data_cleaning_churn.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 200

    sample_data = {
        "customerID": [f"CUST-{1000 + i}" for i in range(n)],
        "gender": np.random.choice(
            ["Male", "Female", "male", "female", "M", "F"],
            n,
            p=[0.4, 0.4, 0.05, 0.05, 0.05, 0.05],
        ),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "Partner": np.random.choice(["Yes", "No"], n),
        "tenure": np.random.randint(0, 72, n).astype(float),
        "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "InternetService": np.random.choice(
            ["DSL", "Fiber optic", "fiber optic", "No", "dsl "], n
        ),
        "OnlineSecurity": np.random.choice(
            ["Yes", "No", "No internet service"], n
        ),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract": np.random.choice(
            ["Month-to-month", "One year", "Two year", "month to month"], n
        ),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            n,
        ),
        "MonthlyCharges": np.round(np.random.uniform(199.0, 1499.0, n), 2),
        "Churn": np.random.choice(
            ["Yes", "No", "yes", "no"], n, p=[0.25, 0.65, 0.05, 0.05]
        ),
    }

    frame = pd.DataFrame(sample_data)
    frame["TotalCharges"] = (
        frame["tenure"] * frame["MonthlyCharges"] + np.random.normal(0, 50, n)
    ).round(2).astype(str)

    frame.loc[frame["tenure"] == 0, "TotalCharges"] = " "
    frame.loc[5:10, "MonthlyCharges"] = np.nan
    frame.loc[15:18, "tenure"] = np.nan
    frame.loc[20, "MonthlyCharges"] = 9500.00
    frame.loc[21, "tenure"] = 240.0
    frame = pd.concat([frame, frame.iloc[:5]], ignore_index=True)

    frame.to_csv(OUTPUT_DIR / "Telecom_Customer_Churn.csv", index=False)
    print("Sample Telecom_Customer_Churn.csv generated.\n")
    return frame


def main() -> None:
    generate_sample_data()
    df = pd.read_csv(OUTPUT_DIR / "Telecom_Customer_Churn.csv")

    print("--- Task 2: Dataset Exploration ---")
    print(f"Initial shape: {df.shape}")
    print("\nData types and non-null counts:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe().round(2))
    print("\nFirst 3 rows:")
    print(df.head(3))

    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if "customerID" in df.columns:
        df.drop_duplicates(subset=["customerID"], keep="first", inplace=True)
    print(
        f"\n--- Task 4: Duplicates Removed ---\n"
        f"Removed {initial_rows - len(df)} duplicate rows. "
        f"Remaining rows: {len(df)}"
    )

    print("\n--- Task 5 & 6: Standardising Formats and Data Types ---")
    string_columns = df.select_dtypes(include="object").columns
    for column in string_columns:
        df[column] = df[column].astype(str).str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .str.lower()
            .map({"male": "Male", "m": "Male", "female": "Female", "f": "Female"})
            .fillna(df["gender"])
        )

    if "InternetService" in df.columns:
        df["InternetService"] = df["InternetService"].str.lower().replace(
            {"dsl": "DSL", "fiber optic": "Fiber optic", "no": "No"}
        )

    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].str.lower().replace(
            {
                "month-to-month": "Month-to-month",
                "month to month": "Month-to-month",
                "one year": "One year",
                "two year": "Two year",
            }
        )

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].str.title()

    addon_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for column in addon_columns:
        if column in df.columns:
            df[column] = df[column].replace({"No internet service": "No"})

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    print("\n--- Task 3: Handling Missing Values ---")
    print("Missing values before imputation:\n", df.isnull().sum()[df.isnull().sum() > 0])

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].median())

    categorical_columns = df.select_dtypes(include="object").columns
    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].mode()[0])

    print("Missing values after imputation:", int(df.isnull().sum().sum()))

    print("\n--- Task 7: Identifying and Capping Outliers (IQR) ---")
    continuous_vars = [
        column
        for column in ["tenure", "MonthlyCharges", "TotalCharges"]
        if column in df.columns
    ]
    for column in continuous_vars:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = (
            (df[column] < lower_bound) | (df[column] > upper_bound)
        ).sum()
        print(
            f"{column}: {outlier_count} outliers outside "
            f"[{lower_bound:.2f}, {upper_bound:.2f}]"
        )
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    print("\n--- Task 8: Feature Engineering ---")
    df["Tenure_Group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=[
            "0-1_Year",
            "1-2_Years",
            "2-4_Years",
            "4-5_Years",
            "Over_5_Years",
        ],
    )

    available_services = [
        column
        for column in ["PhoneService", "OnlineSecurity", "TechSupport", "StreamingTV"]
        if column in df.columns
    ]
    df["Total_Services_Count"] = (df[available_services] == "Yes").sum(axis=1)

    historical_average = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )
    df["Charge_Increase_Ratio"] = (
        df["MonthlyCharges"] / (historical_average + 1e-5)
    ).round(2)
    df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(
        df[
            [
                "tenure",
                "Tenure_Group",
                "Total_Services_Count",
                "Charge_Increase_Ratio",
                "Churn_Binary",
            ]
        ].head(3)
    )

    print("\n--- Task 9: Encoding and Scaling ---")
    columns_to_drop = [column for column in ["customerID", "Churn"] if column in df.columns]
    df_model = df.drop(columns=columns_to_drop)
    df_encoded = pd.get_dummies(df_model, drop_first=True, dtype=int)

    scaler = StandardScaler()
    features_to_scale = [
        column
        for column in [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Total_Services_Count",
            "Charge_Increase_Ratio",
        ]
        if column in df_encoded.columns
    ]
    df_encoded[features_to_scale] = scaler.fit_transform(
        df_encoded[features_to_scale]
    )
    print(df_encoded[features_to_scale].describe().loc[["mean", "std"]].round(2))

    print("\n--- Task 10: Train-Test Split ---")
    X = df_encoded.drop(columns=["Churn_Binary"])
    y = df_encoded["Churn_Binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"X_test={X_test.shape}, y_test={y_test.shape}")

    df.to_csv(OUTPUT_DIR / "exp3_cleaned_churn.csv", index=False)
    df_encoded.to_csv(OUTPUT_DIR / "exp3_model_ready_churn.csv", index=False)
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
