"""Experiment 4: Data Wrangling.

Aim: Clean a messy real-estate dataset, merge neighbourhood amenities,
filter recent residential sales, cap outliers, aggregate prices and encode
categorical variables.

Dataset: synthetic real-estate and amenities tables generated in this script.
Run: python exp4_data_wrangling_real_estate.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_sample_data() -> None:
    np.random.seed(42)
    n = 150

    neighborhoods = [
        "Koregaon Park",
        "Baner",
        "Wakad",
        "Viman Nagar",
        "Kothrud",
        "Hinjewadi",
    ]
    property_types = ["Apartment", "Villa", "Row House", "Commercial"]
    furnishing = ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]

    raw_data = {
        "Property ID #": [f"PROP-{1000 + i}" for i in range(n)],
        "Neighborhood / Locality": np.random.choice(neighborhoods, n),
        "Property Type!": np.random.choice(
            property_types, n, p=[0.60, 0.15, 0.15, 0.10]
        ),
        "Area (Sq.Ft.)": np.random.randint(650, 3200, n).astype(float),
        "BHK_Count": np.random.choice(
            [1, 2, 3, 4], n, p=[0.15, 0.45, 0.30, 0.10]
        ).astype(float),
        "Furnishing Status": np.random.choice(furnishing, n),
        "Property Age (Yrs)": np.random.randint(0, 20, n).astype(float),
        "Sale Date": pd.date_range(
            start="2023-01-01", periods=n, freq="7D"
        ).strftime("%Y-%m-%d"),
        "Sale Price (INR Lakhs)": np.round(
            np.random.uniform(45.0, 280.0, n), 2
        ),
    }

    df_raw = pd.DataFrame(raw_data)
    df_raw.loc[5:12, "Area (Sq.Ft.)"] = np.nan
    df_raw.loc[15:20, "BHK_Count"] = np.nan
    df_raw.loc[25:28, "Furnishing Status"] = np.nan
    df_raw.loc[30:32, "Sale Price (INR Lakhs)"] = np.nan
    df_raw.loc[40, "Sale Price (INR Lakhs)"] = 1450.0
    df_raw.loc[41, "Area (Sq.Ft.)"] = 15000.0

    df_raw.to_csv(OUTPUT_DIR / "RealEstate_Prices.csv", index=False)

    amenities_data = {
        "Neighborhood": neighborhoods,
        "Metro_Distance_km": [2.5, 4.0, 5.5, 1.2, 0.8, 3.0],
        "Livability_Index": [9.2, 8.6, 8.1, 8.9, 8.7, 7.9],
        "Nearby_Schools_Count": [12, 15, 18, 14, 16, 10],
    }
    pd.DataFrame(amenities_data).to_csv(
        OUTPUT_DIR / "Neighborhood_Amenities.csv", index=False
    )
    print("Sample real-estate and amenities files generated.\n")


def main() -> None:
    generate_sample_data()
    df = pd.read_csv(OUTPUT_DIR / "RealEstate_Prices.csv")

    print("--- Task 1: Cleaning Column Names ---")
    print("Original columns:", df.columns.tolist())
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"[^\w\s]", "", column).strip().replace(" ", "_"))
        for column in df.columns
    ]
    rename_map = {
        "Property_ID": "Property_ID",
        "Neighborhood_Locality": "Neighborhood",
        "Property_Type": "Property_Type",
        "Area_SqFt": "Area_SqFt",
        "BHK_Count": "BHK",
        "Furnishing_Status": "Furnishing_Status",
        "Property_Age_Yrs": "Property_Age_Yrs",
        "Sale_Date": "Sale_Date",
        "Sale_Price_INR_Lakhs": "Sale_Price_Lakhs",
    }
    df.rename(columns=rename_map, inplace=True)
    print("Cleaned columns:", df.columns.tolist())

    print("\n--- Task 2: Handling Missing Values ---")
    print("Missing values before cleaning:\n", df.isnull().sum()[df.isnull().sum() > 0])
    df.dropna(subset=["Sale_Price_Lakhs"], inplace=True)
    df["Area_SqFt"] = df.groupby("Property_Type")["Area_SqFt"].transform(
        lambda values: values.fillna(values.median())
    )

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].median())

    categorical_columns = df.select_dtypes(include="object").columns
    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].mode()[0])

    print("Total missing values after cleaning:", int(df.isnull().sum().sum()))

    print("\n--- Task 3: Merging Neighbourhood Amenities ---")
    amenities_df = pd.read_csv(OUTPUT_DIR / "Neighborhood_Amenities.csv")
    df_merged = pd.merge(df, amenities_df, on="Neighborhood", how="left")
    print(f"Shape before merge: {df.shape} -> after merge: {df_merged.shape}")
    print(
        df_merged[
            [
                "Property_ID",
                "Neighborhood",
                "Sale_Price_Lakhs",
                "Metro_Distance_km",
                "Livability_Index",
            ]
        ].head(3)
    )

    print("\n--- Task 4: Filtering and Subsetting ---")
    df_merged["Sale_Date"] = pd.to_datetime(df_merged["Sale_Date"])
    residential_types = ["Apartment", "Villa", "Row House"]
    df_filtered = df_merged[
        (df_merged["Property_Type"].isin(residential_types))
        & (df_merged["Sale_Date"] >= "2024-01-01")
        & (df_merged["Area_SqFt"] >= 600)
    ].copy().reset_index(drop=True)
    print(
        f"Rows before filtering: {len(df_merged)} | after filtering: "
        f"{len(df_filtered)}"
    )

    print("\n--- Task 7: Identifying and Handling Outliers (IQR) ---")
    for column in ["Area_SqFt", "Sale_Price_Lakhs"]:
        q1 = df_filtered[column].quantile(0.25)
        q3 = df_filtered[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (
            (df_filtered[column] < lower_bound)
            | (df_filtered[column] > upper_bound)
        ).sum()
        print(
            f"{column}: {outliers} outliers outside "
            f"[{lower_bound:.2f}, {upper_bound:.2f}]"
        )
        df_filtered[column] = df_filtered[column].clip(
            lower=lower_bound, upper=upper_bound
        )

    df_filtered["Price_Per_SqFt_INR"] = (
        (df_filtered["Sale_Price_Lakhs"] * 100000) / df_filtered["Area_SqFt"]
    ).round(2)

    print("\n--- Task 6: Data Aggregation ---")
    neighborhood_summary = (
        df_filtered.groupby("Neighborhood")
        .agg(
            Total_Listings=("Property_ID", "count"),
            Avg_Sale_Price_Lakhs=("Sale_Price_Lakhs", "mean"),
            Median_Sale_Price_Lakhs=("Sale_Price_Lakhs", "median"),
            Avg_Price_Per_SqFt_INR=("Price_Per_SqFt_INR", "mean"),
            Avg_Livability=("Livability_Index", "first"),
        )
        .round(2)
        .reset_index()
        .sort_values(by="Avg_Sale_Price_Lakhs", ascending=False)
    )
    print("\nSummary statistics by neighbourhood:")
    print(neighborhood_summary.to_string(index=False))

    price_pivot = df_filtered.pivot_table(
        values="Sale_Price_Lakhs",
        index="Neighborhood",
        columns="Property_Type",
        aggfunc="mean",
    ).round(2)
    print("\nAverage sale price pivot table:")
    print(price_pivot)

    print("\n--- Task 5: Encoding Categorical Variables ---")
    df_encoded = df_filtered.copy()
    furnishing_order = {
        "Unfurnished": 0,
        "Semi-Furnished": 1,
        "Fully-Furnished": 2,
    }
    df_encoded["Furnishing_Encoded"] = df_encoded["Furnishing_Status"].map(
        furnishing_order
    )

    label_encoder = LabelEncoder()
    df_encoded["Property_Type_Label"] = label_encoder.fit_transform(
        df_encoded["Property_Type"]
    )

    df_encoded = pd.get_dummies(
        df_encoded,
        columns=["Neighborhood", "Property_Type"],
        prefix=["Loc", "Type"],
        drop_first=True,
        dtype=int,
    )
    print("Encoded columns:", df_encoded.columns.tolist())

    df_encoded.to_csv(OUTPUT_DIR / "exp4_wrangled_real_estate.csv", index=False)
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
