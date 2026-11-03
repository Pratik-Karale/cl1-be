"""Experiment 6: Data Aggregation.

Aim: Aggregate 300 retail transactions by region and by region x category,
rank regions, compute revenue shares and visualise the results.

Dataset: synthetic retail sales records generated in this script.
Run: python exp6_data_aggregation_retail.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_sample_data() -> None:
    np.random.seed(42)
    n = 300

    regions = ["West", "South", "North", "East", "Central"]
    categories = [
        "Electronics",
        "Apparel",
        "Home & Kitchen",
        "Groceries",
        "Beauty & Personal Care",
    ]
    payment_modes = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"]

    region_choices = np.random.choice(regions, n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    category_choices = np.random.choice(
        categories, n, p=[0.25, 0.25, 0.20, 0.20, 0.10]
    )
    quantities = np.random.randint(1, 8, n)

    base_prices = {
        "Electronics": (8000, 45000),
        "Apparel": (800, 4500),
        "Home & Kitchen": (1200, 12000),
        "Groceries": (250, 2500),
        "Beauty & Personal Care": (300, 3000),
    }
    unit_prices = [
        np.round(np.random.uniform(*base_prices[category]), 2)
        for category in category_choices
    ]
    sales_amounts = np.round(quantities * np.array(unit_prices), 2)

    frame = pd.DataFrame(
        {
            "Transaction_ID": [f"TXN-{10000 + i}" for i in range(n)],
            "Transaction_Date": pd.date_range(
                start="2026-01-01", periods=n, freq="18h"
            ).strftime("%Y-%m-%d"),
            "Region": region_choices,
            "Product_Category": category_choices,
            "Quantity_Sold": quantities,
            "Unit_Price": unit_prices,
            "Sales_Amount": sales_amounts,
            "Payment_Mode": np.random.choice(
                payment_modes, n, p=[0.45, 0.25, 0.15, 0.10, 0.05]
            ),
        }
    )
    frame.loc[10:12, "Sales_Amount"] = np.nan
    frame.to_csv(OUTPUT_DIR / "Retail_Sales_Data.csv", index=False)
    print("Sample Retail_Sales_Data.csv generated.\n")


def main() -> None:
    generate_sample_data()
    df = pd.read_csv(OUTPUT_DIR / "Retail_Sales_Data.csv")

    print("--- Task 1 & 2: Dataset Structure and Exploration ---")
    print(f"Dataset shape: {df.shape}")
    print("\nColumn data types and non-null counts:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe().round(2))
    print("\nFirst 5 records:")
    print(df.head())

    print("\n--- Task 3: Checking Relevant Variables ---")
    relevant_columns = [
        "Region",
        "Product_Category",
        "Quantity_Sold",
        "Sales_Amount",
    ]
    print("Missing values in key columns:\n", df[relevant_columns].isnull().sum())

    if df["Sales_Amount"].isnull().sum() > 0:
        if "Unit_Price" in df.columns:
            df["Sales_Amount"] = df["Sales_Amount"].fillna(
                df["Quantity_Sold"] * df["Unit_Price"]
            )
        else:
            df["Sales_Amount"] = df["Sales_Amount"].fillna(
                df["Sales_Amount"].median()
            )

    df["Region"] = df["Region"].astype(str).str.strip().str.title()
    df["Product_Category"] = (
        df["Product_Category"].astype(str).str.strip().str.title()
    )

    regional_sales = (
        df.groupby("Region", as_index=False)
        .agg(
            Total_Transactions=("Transaction_ID", "count"),
            Total_Units_Sold=("Quantity_Sold", "sum"),
            Total_Sales_Amount=("Sales_Amount", "sum"),
            Average_Order_Value=("Sales_Amount", "mean"),
        )
        .round(2)
        .sort_values(by="Total_Sales_Amount", ascending=False)
        .reset_index(drop=True)
    )
    total_company_sales = regional_sales["Total_Sales_Amount"].sum()
    regional_sales["Revenue_Share_Pct"] = (
        (regional_sales["Total_Sales_Amount"] / total_company_sales) * 100
    ).round(2)

    print("\n--- Task 4: Sales Aggregated by Region ---")
    print(regional_sales.to_string(index=False))

    print("\n--- Task 6: Top-Performing Regions ---")
    top_region = regional_sales.iloc[0]
    print(
        f"Top region: {top_region['Region']} "
        f"({top_region['Total_Sales_Amount']:,.2f} | "
        f"{top_region['Revenue_Share_Pct']}% of total revenue)"
    )
    print(
        regional_sales[
            ["Region", "Total_Sales_Amount", "Revenue_Share_Pct"]
        ].head(3).to_string(index=False)
    )

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Regional Sales Performance Distribution", fontsize=14,
                 fontweight="bold")

    sns.barplot(
        data=regional_sales,
        x="Region",
        y="Total_Sales_Amount",
        hue="Region",
        palette="viridis",
        edgecolor="black",
        ax=ax1,
        legend=False,
    )
    ax1.set_title("Total Sales Amount by Region")
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Total Sales Amount")
    for bar in ax1.patches:
        height = bar.get_height()
        ax1.annotate(
            f"{height / 1e5:.2f}L",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    explode_values = [
        0.08 if index == 0 else 0 for index in range(len(regional_sales))
    ]
    ax2.pie(
        regional_sales["Total_Sales_Amount"],
        labels=regional_sales["Region"],
        autopct="%1.1f%%",
        startangle=140,
        explode=explode_values,
        colors=sns.color_palette("pastel"),
        wedgeprops={"edgecolor": "black", "linewidth": 1},
    )
    ax2.set_title("Percentage Share of Total Sales by Region")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp6_regional_sales.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    region_category_sales = (
        df.groupby(["Region", "Product_Category"], as_index=False)
        .agg(
            Total_Units_Sold=("Quantity_Sold", "sum"),
            Total_Sales_Amount=("Sales_Amount", "sum"),
        )
        .round(2)
    )
    print("\n--- Task 7A: Region x Product Category (long format) ---")
    print(region_category_sales.head(10).to_string(index=False))

    sales_pivot = df.pivot_table(
        index="Region",
        columns="Product_Category",
        values="Sales_Amount",
        aggfunc="sum",
        fill_value=0,
    ).round(2)
    sales_pivot = sales_pivot.loc[regional_sales["Region"]]
    print("\n--- Task 7B: Region x Product Category Pivot Table ---")
    print(sales_pivot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Sales Comparison Across Regions and Product Categories",
                 fontsize=14, fontweight="bold")

    sns.barplot(
        data=region_category_sales,
        x="Region",
        y="Total_Sales_Amount",
        hue="Product_Category",
        order=regional_sales["Region"],
        palette="Set2",
        edgecolor="black",
        ax=ax1,
    )
    ax1.set_title("Grouped Bar Plot (Region x Category)")
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Total Sales Amount")
    ax1.legend(title="Product Category", loc="upper right")

    sales_pivot.plot(
        kind="bar",
        stacked=True,
        colormap="tab20c",
        edgecolor="black",
        width=0.65,
        ax=ax2,
    )
    ax2.set_title("Stacked Bar Plot of Category Composition")
    ax2.set_xlabel("Region")
    ax2.set_ylabel("Cumulative Sales Amount")
    ax2.tick_params(axis="x", rotation=0)
    ax2.legend(title="Product Category", loc="upper right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp6_region_category.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    df.to_csv(OUTPUT_DIR / "exp6_retail_cleaned.csv", index=False)
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
