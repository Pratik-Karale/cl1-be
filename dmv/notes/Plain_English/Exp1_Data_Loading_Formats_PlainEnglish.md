# Experiment 1 — Data Loading, Storage and File Formats in Plain English

> **In one sentence:** We take sales data saved in three different file types (CSV, Excel, JSON), load all of them with pandas, put them into one clean table, and then summarise how much each product category sold.

## The problem in everyday words

Imagine three shop branches. One keeps its sales in a plain text file (CSV), one in an Excel workbook, and one in a JSON file from an online store. Each uses slightly different column names, and some rows are missing values or appear twice. Before you can answer "which product category earns the most?", you need all three lists in one clean table that uses the same column names and has no repeats.

## What the main idea is, without jargon

Think of CSV, Excel and JSON as three different containers for the same shopping list. CSV is comma-separated plain text, Excel is a workbook of cells, and JSON is a set of key-value records. Pandas has one reader for each container — `read_csv`, `read_excel`, `read_json` — and all three give back the same object: a DataFrame, which is just a table.

Loading is the easy part. The real work is making the three tables agree: renaming columns, removing duplicate rows, filling gaps, and only then stacking them into one table.

## What the code does, step by step

1. Load each file with its own reader.
2. Add a `Source` column saying which file each row came from.
3. Look at the table's shape, columns, missing-value counts and duplicate count.
4. Rename Excel's `Category` column to `Product_Category` so all three files match.
5. Stack the rows with `concat(..., ignore_index=True)`.
6. Remove repeated rows with `drop_duplicates()`.
7. Standardise category text (`Electronics`, `electronics`, ` ELECTRONICS`) with strip and title-case.
8. Fill a missing `Quantity` with the overall median. Fill a missing `Unit_Price` with the median of its own category using `groupby(...).transform(...)`.
9. Convert `Order_Date` to a real date, split `Customer_Info` into name and city, and create `Total_Sales = Quantity × Unit_Price`.
10. Summarise by category with `groupby().agg()` and draw bar, pie and box charts.

## What the result means

- After stacking there are 13 rows; one duplicate is removed, leaving **12 clean rows and 7 columns**.
- Total revenue is **₹2,33,000**, and the average order value is **₹19,416.67**.
- Electronics: 5 orders, **₹2,14,000** (about 92% of revenue), average order ₹42,800.
- Books: ₹12,800. Apparel: ₹3,000. Home & Kitchen: ₹3,200.
- The lesson: Electronics earns the most because its unit price is high, not because it has the most orders.

## Why these choices?

- Rename columns before stacking; otherwise pandas creates two columns with half the values missing in each.
- Use the category median for price, because a single overall median would misprice expensive and cheap categories.
- Remove duplicates so the same order is not counted twice in revenue.
- Fill missing quantity with the median instead of deleting the row, so no data is thrown away.
- Convert dates so the data can be sorted and grouped by month or day.

## If the examiner asks (plain answers)

1. **Why use three formats?** Real data arrives in different containers; the analysis should not depend on the container.
2. **Which library reads Excel?** `openpyxl`; pandas uses it behind the scenes.
3. **How do the three readers differ?** They all return a DataFrame; only the parser is different.
4. **What does `ignore_index=True` do?** It gives the combined table fresh row numbers 0, 1, 2… instead of repeating old ones.
5. **Why add a `Source` column?** So each row can be traced back to its file and stores can be compared.
6. **How was the Excel column fixed?** `Category` was renamed to `Product_Category` before combining.
7. **Why the category median for unit price?** Prices differ sharply by category; one global median would distort them.
8. **Why remove duplicate rows?** A repeated order would inflate the revenue total.
9. **What if the JSON were nested?** `pd.json_normalize` would flatten it into columns.
10. **Which category dominates?** Electronics, because of high prices rather than order count.

**One-line takeaway:** Load each format, make the column names agree, stack the rows, clean missing and duplicate values, then group and summarise.
