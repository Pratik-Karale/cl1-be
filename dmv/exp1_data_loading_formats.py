"""Experiment 1: Data Loading, Storage and File Formats.

Aim: Load the same sales data from CSV, Excel and JSON, unify the frames,
clean inconsistencies and summarise revenue by product category.

Dataset: synthetic sales records generated in this script.
Run: python exp1_data_loading_formats.py
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


def generate_sample_files() -> None:
    csv_data = {
        "Order_ID": ["ORD101", "ORD102", "ORD103", "ORD104", "ORD102"],
        "Customer_Info": [
            "Rahul Sharma - Mumbai",
            "Priya Patel - Pune",
            "Amit Kumar - Delhi",
            "Sneha Reddy - Hyderabad",
            "Priya Patel - Pune",
        ],
        "Product_Category": [
            "Electronics",
            "Apparel",
            "Electronics",
            "Home & Kitchen",
            "Apparel",
        ],
        "Order_Date": ["2026-01-15", "2026-01-16", "2026-01-18", "2026-01-20", "2026-01-16"],
        "Quantity": [1, 2, np.nan, 3, 2],
        "Unit_Price": [45000, 1500, 12000, 800, 1500],
    }
    pd.DataFrame(csv_data).to_csv(OUTPUT_DIR / "sales_data.csv", index=False)

    excel_data = {
        "Order_ID": ["ORD201", "ORD202", "ORD203", "ORD204"],
        "Customer_Info": [
            "Vikram Singh - Jaipur",
            "Ananya Das - Kolkata",
            "Rohan Mehta - Pune",
            "Kavita Joshi - Mumbai",
        ],
        "Category": ["electronics", "Books", "Apparel", "Home & Kitchen"],
        "Order_Date": ["2026-02-01", "2026-02-03", "2026-02-05", "2026-02-10"],
        "Quantity": [1, 4, 2, 1],
        "Unit_Price": [55000, 450, 2200, np.nan],
    }
    pd.DataFrame(excel_data).to_excel(OUTPUT_DIR / "sales_data.xlsx", index=False)

    json_data = {
        "Order_ID": ["ORD301", "ORD302", "ORD303", "ORD304"],
        "Customer_Info": [
            "Arjun Nair - Kochi",
            "Meera Iyer - Chennai",
            "Zoya Khan - Bengaluru",
            "Kabir Verma - Delhi",
        ],
        "Product_Category": ["Books", "Electronics", "Apparel", "Electronics"],
        "Order_Date": ["2026-03-01", "2026-03-04", "2026-03-08", "2026-03-12"],
        "Quantity": [2, 1, 3, 1],
        "Unit_Price": [600, 28000, 1800, 62000],
    }
    pd.DataFrame(json_data).to_json(
        OUTPUT_DIR / "sales_data.json", orient="records", indent=4
    )
    print("Sample CSV, Excel and JSON files generated.\n")


def main() -> None:
    generate_sample_files()

    df_csv = pd.read_csv(OUTPUT_DIR / "sales_data.csv")
    df_excel = pd.read_excel(OUTPUT_DIR / "sales_data.xlsx")
    df_json = pd.read_json(OUTPUT_DIR / "sales_data.json")

    df_csv["Source"] = "CSV_Store"
    df_excel["Source"] = "Excel_Store"
    df_json["Source"] = "JSON_Online"

    print("--- Task 2: Data Exploration ---")
    for name, frame in [("CSV", df_csv), ("Excel", df_excel), ("JSON", df_json)]:
        print(f"\n[{name} Dataset] Shape: {frame.shape}")
        print("Columns:", frame.columns.tolist())
        print("Missing values:\n", frame.isnull().sum()[frame.isnull().sum() > 0])
        print("Duplicates:", frame.duplicated().sum())

    df_excel.rename(columns={"Category": "Product_Category"}, inplace=True)
    unified = pd.concat([df_csv, df_excel, df_json], ignore_index=True)

    unified.drop_duplicates(inplace=True)
    unified["Product_Category"] = (
        unified["Product_Category"].str.strip().str.title()
    )
    unified["Quantity"] = unified["Quantity"].fillna(unified["Quantity"].median())
    unified["Unit_Price"] = unified.groupby("Product_Category")[
        "Unit_Price"
    ].transform(lambda values: values.fillna(values.median()))
    unified["Order_Date"] = pd.to_datetime(unified["Order_Date"])

    print("\n--- Task 3 & 4: Cleaned Unified DataFrame ---")
    print(unified.info())

    customer_parts = unified["Customer_Info"].str.split(" - ", expand=True)
    unified["Customer_Name"] = customer_parts[0]
    unified["City"] = customer_parts[1]
    unified.drop(columns=["Customer_Info"], inplace=True)
    unified["Total_Sales"] = unified["Quantity"] * unified["Unit_Price"]
    unified["Order_Month"] = unified["Order_Date"].dt.month_name()

    print("\n--- Task 5: Transformed Data Preview ---")
    print(unified.head())

    print("\n--- Task 6: Sales Analysis Metrics ---")
    print(unified[["Quantity", "Unit_Price", "Total_Sales"]].describe().round(2))
    total_revenue = unified["Total_Sales"].sum()
    average_order_value = unified["Total_Sales"].mean()
    print(f"\nTotal Revenue: {total_revenue:,.2f}")
    print(f"Average Order Value (AOV): {average_order_value:,.2f}")

    category_summary = (
        unified.groupby("Product_Category")
        .agg(
            Total_Orders=("Order_ID", "count"),
            Total_Units_Sold=("Quantity", "sum"),
            Total_Revenue=("Total_Sales", "sum"),
            Avg_Order_Value=("Total_Sales", "mean"),
        )
        .reset_index()
    )
    print("\nSales Aggregation by Product Category:")
    print(category_summary.to_string(index=False))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Unified Sales Data Analysis Dashboard", fontsize=16, fontweight="bold"
    )

    sns.barplot(
        data=category_summary,
        x="Product_Category",
        y="Total_Revenue",
        hue="Product_Category",
        palette="viridis",
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Total Revenue by Category")
    axes[0].set_xlabel("Product Category")
    axes[0].set_ylabel("Total Revenue")
    axes[0].tick_params(axis="x", rotation=15)

    category_counts = unified["Product_Category"].value_counts()
    axes[1].pie(
        category_counts,
        labels=category_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    axes[1].set_title("Share of Orders by Category")

    sns.boxplot(
        data=unified,
        x="Source",
        y="Total_Sales",
        hue="Source",
        palette="Set2",
        ax=axes[2],
        legend=False,
    )
    axes[2].set_title("Order Value Distribution by Source")
    axes[2].set_xlabel("Data Source")
    axes[2].set_ylabel("Order Value")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp1_sales_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    unified.to_csv(OUTPUT_DIR / "exp1_unified_sales.csv", index=False)

    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
