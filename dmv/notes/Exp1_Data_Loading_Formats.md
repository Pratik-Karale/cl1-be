# Experiment 1 — Data Loading, Storage and File Formats (Viva Theory)

> **30-second answer:** Load the same sales data from **CSV, Excel and JSON** with `pd.read_csv/read_excel/read_json`, add a `Source` column, `concat` them into one unified DataFrame, clean duplicates/missing values/column-name mismatches, derive `Total_Sales`, and summarise revenue by product category. Headline result: **12 clean rows, total revenue ₹2,33,000, AOV ₹19,416.67; Electronics earns ₹2,14,000 (92%)**.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

- **Aim:** Analyse sales data stored in multiple file formats and unify it into one clean DataFrame for analysis.
- **Dataset:** three generated files — `sales_data.csv` (5 rows × 7 cols, 1 missing `Quantity`, 1 duplicate), `sales_data.xlsx` (4 × 7, 1 missing `Unit_Price`, column named `Category`), `sales_data.json` (4 × 7, clean).
- **Pipeline:**
  1. Load each format with its reader.
  2. Add `Source` = `CSV_Store` / `Excel_Store` / `JSON_Online`.
  3. Explore shape, columns, `isnull().sum()`, `duplicated().sum()`.
  4. Rename Excel's `Category` → `Product_Category`.
  5. `pd.concat(..., ignore_index=True)` to stack rows.
  6. `drop_duplicates()`, standardise category text with `.str.strip().str.title()`.
  7. Impute `Quantity` with median; impute `Unit_Price` with the **median of its own category** (`groupby(...).transform`).
  8. Convert `Order_Date` with `pd.to_datetime`; split `Customer_Info` into name/city; derive `Total_Sales = Quantity × Unit_Price`.
  9. Describe, group by category, and plot bar/pie/box charts.

**Code skeleton**
```python
df_csv   = pd.read_csv('sales_data.csv')
df_excel = pd.read_excel('sales_data.xlsx')   # needs openpyxl
df_json  = pd.read_json('sales_data.json')
df = pd.concat([df_csv, df_excel, df_json], ignore_index=True)
df.drop_duplicates(inplace=True)
df['Product_Category'] = df['Product_Category'].str.strip().str.title()
df['Quantity'] = df['Quantity'].fillna(df['Quantity'].median())
df['Unit_Price'] = df.groupby('Product_Category')['Unit_Price'].transform(
    lambda x: x.fillna(x.median()))
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Total_Sales'] = df['Quantity'] * df['Unit_Price']
df.groupby('Product_Category').agg(
    Total_Orders=('Order_ID','count'),
    Total_Revenue=('Total_Sales','sum'),
    Avg_Order_Value=('Total_Sales','mean'))
```

**Expected output**
- Unified clean shape: **12 rows × 7 columns** (13 after concat, 1 duplicate dropped).
- Total revenue **₹2,33,000.00**; AOV **₹19,416.67**.
- Electronics: 5 orders, ₹2,14,000 (AOV ₹42,800); Books ₹12,800; Apparel ₹3,000; Home & Kitchen ₹3,200.
- Charts: bar (revenue/category), pie (order share), box (order value by source).

**Viva Q&A**
1. **Why use three formats?** To show that real data arrives in CSV, Excel and JSON and must be normalised before analysis.
2. **Which library reads `.xlsx`?** `openpyxl` (pandas delegates to it).
3. **`read_csv` vs `read_excel` vs `read_json`?** Same goal — return a DataFrame; only the parser differs.
4. **What does `ignore_index=True` do?** Drops the old row indices and assigns a fresh 0…n−1 index, avoiding duplicate labels after concatenation.
5. **Why add a `Source` column?** To trace each row back to its file and later compare distributions by source.
6. **How did you unify the Excel column?** `df_excel.rename(columns={'Category':'Product_Category'}, inplace=True)` before `concat`.
7. **Why impute `Unit_Price` by category median, not global median?** Prices differ sharply by category; a global median would distort expensive categories.
8. **`transform` vs `agg` here?** `transform` returns a value for every row (same index), so it can fill NaNs in place; `agg` collapses each group to one value.
9. **Why `pd.to_datetime`?** It converts strings to `datetime64`, enabling `.dt.month_name()`, sorting and time filtering.
10. **What if a file is missing?** Raise a clear `FileNotFoundError` or use `try/except`; never silently continue.
11. **What if JSON is nested?** Use `pd.json_normalize()` instead of `read_json` to flatten it.
12. **Which category dominates and why?** Electronics — few orders but high unit price, so revenue ≠ order count.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit IV — pandas I/O and data structures
- **Series** = 1-D labelled array; **DataFrame** = 2-D table of Series sharing an index.
- **Tidy data:** one row per observation, one column per variable, one table per unit of analysis.
- **`inplace=True`** mutates the original object and returns `None`; otherwise assign the result.

| Format | Reader / Writer | Best for |
|---|---|---|
| CSV | `read_csv` / `to_csv` | plain, portable, large data |
| Excel | `read_excel` / `to_excel` | formatted business sheets |
| JSON | `read_json` / `to_json` | nested/API-style records |

**Q&A**
1. **`concat` vs `merge`?** `concat` stacks or aligns along an axis; `merge` joins on common keys (SQL-style).
2. **`drop_duplicates(subset=...)`?** Considers only the listed columns when deciding duplication.
3. **Why `str.title()`?** Normalises `electronics` and `Electronics` into one category so groupby does not split them.
4. **How to inspect data quickly?** `.shape`, `.info()`, `.describe()`, `.head()`, `.isnull().sum()`, `.duplicated().sum()`.
5. **What is `NaN`?** pandas' missing-value marker (`np.nan`), float by default; `pd.NA` is the nullable alternative.
6. **`fillna(median)` vs `dropna()`?** Imputation preserves rows but adds assumption; dropping loses information.
7. **What does `reset_index()` do?** Moves the index back into a column (or discards it) and creates a default RangeIndex.
8. **Why is `Order_Date` an object before conversion?** `read_csv` parses text; dates are not auto-detected unless `parse_dates` is used.

## 3. LEAST LIKELY — Stretch / Advanced / Edge Cases

1. **Heterogeneous schemas:** CSV/JSON had `Product_Category`, Excel had `Category`; unifying names is mandatory before `concat`, otherwise pandas creates two columns with NaNs.
2. **Silent type coercion:** `TotalCharges`-style numeric columns can arrive as strings with blanks — always inspect `dtypes` and use `pd.to_numeric(errors='coerce')`.
3. **`concat` with different column sets:** missing columns become NaN; use `join='inner'` if only common columns are wanted.
4. **Large files:** `chunksize=` streams CSV in blocks; `dtype=` and `usecols=` reduce memory.
5. **JSON orientations:** `orient='records'` produces a list of row objects; `orient='columns'` stores column-wise — the wrong orientation transposes the table.
6. **Imputation leakage:** if this cleaned data feeds a model, compute medians on the training split only, not the full dataset.
7. **Cross-unit link (Unit V):** the `groupby().agg()` used here is the same split-apply-combine pattern formalised in time-series aggregation.
8. **Reproducibility trap:** `drop_duplicates()` keeps the first occurrence by default; use `keep='last'` or a key column when the newest record should win.

**If you remember only one thing:** load each format → rename columns → `concat` → clean/impute → `groupby` — the clean unified DataFrame is the whole point.
