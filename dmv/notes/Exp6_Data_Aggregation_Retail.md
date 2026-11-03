# Experiment 6 — Data Aggregation (Viva Theory)

> **30-second answer:** Load 300 retail transactions, repair 3 missing `Sales_Amount` values as `Quantity × Unit_Price`, **group by region** to compute transactions, units, revenue and AOV, rank regions, then **group by region × category** and build a pivot table for grouped/stacked bar charts. Headline: **West is the top region — ₹39.34 lakh, 33.91% of revenue**.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

- **Aim:** Aggregate retail sales by region (and region × category) to identify top-performing regions.
- **Dataset:** `Retail_Sales_Data.csv` — **300 rows × 8 columns**: Transaction_ID, Transaction_Date, Region, Product_Category, Quantity_Sold, Unit_Price, Sales_Amount, Payment_Mode. `Sales_Amount` has **3 missing** values.
- **Pipeline:**
  1. `read_csv`; explore `shape` (300, 8), `info()`, `describe()`, `head()`.
  2. Check missing values in key columns (`Region`, `Product_Category`, `Quantity_Sold`, `Sales_Amount`).
  3. Fill missing `Sales_Amount` with `Quantity_Sold × Unit_Price` (business rule, not a blind median).
  4. Standardise grouping columns: `str.strip().str.title()`.
  5. `groupby('Region', as_index=False).agg(...)` → transactions, units, total sales, AOV.
  6. Sort descending and add `Revenue_Share_Pct`.
  7. Identify top region / top 3 with `.iloc[0]` and `.head(3)`.
  8. `groupby(['Region','Product_Category'])` and `pivot_table(index='Region', columns='Product_Category', values='Sales_Amount', aggfunc='sum')`.
  9. Plot bar + pie for regional share; grouped bar + stacked bar for region × category.

**Code skeleton**
```python
df = pd.read_csv("Retail_Sales_Data.csv")
df['Sales_Amount'] = df['Sales_Amount'].fillna(df['Quantity_Sold'] * df['Unit_Price'])
df['Region'] = df['Region'].str.strip().str.title()

regional = (df.groupby('Region', as_index=False)
    .agg(Total_Transactions=('Transaction_ID','count'),
         Total_Units_Sold=('Quantity_Sold','sum'),
         Total_Sales_Amount=('Sales_Amount','sum'),
         Average_Order_Value=('Sales_Amount','mean'))
    .sort_values('Total_Sales_Amount', ascending=False))

regional['Revenue_Share_Pct'] = (
    regional.Total_Sales_Amount / regional.Total_Sales_Amount.sum() * 100).round(2)

pivot = df.pivot_table(index='Region', columns='Product_Category',
                       values='Sales_Amount', aggfunc='sum', fill_value=0)
pivot.plot(kind='bar', stacked=True, ax=ax)
```

**Expected output**
- Missing `Sales_Amount`: **3** before, **0** after.
- Regional results:

| Region | Txns | Units | Total Sales (₹) | AOV (₹) | Share |
|---|---:|---:|---:|---:|---:|
| West | 95 | 406 | 39,33,559.22 | 41,405.89 | 33.91% |
| South | 70 | 277 | 24,69,182.68 | 35,274.04 | 21.29% |
| North | 59 | 229 | 18,92,976.70 | 32,084.35 | 16.32% |
| East | 47 | 202 | 17,33,804.53 | 36,889.46 | 14.95% |
| Central | 29 | 113 | 15,69,625.60 | 54,125.02 | 13.53% |

- Total company sales ≈ **₹1,15,99,148.73** (₹1.16 crore).
- Pivot table region × 5 categories; grouped and stacked bar charts.

**Viva Q&A**
1. **What is aggregation?** Reducing many rows to summary statistics per group (sum, mean, count, min, max).
2. **Why group by region?** It answers "which region performs best?" — a business question at the right granularity.
3. **`as_index=False`?** Keeps group keys as columns instead of the index, simplifying downstream calculations/plots.
4. **`agg` with named tuples?** `NewName=('Column','func')` creates clean, self-documenting output columns.
5. **Why fill missing sales from Quantity × Unit_Price?** It recomputes the true value from components instead of guessing a summary statistic.
6. **What is AOV?** Average Order Value = total sales ÷ number of orders (here the mean of `Sales_Amount`).
7. **Why `Revenue_Share_Pct`?** Absolute revenue hides relative importance; share normalises against the company total.
8. **What is a pivot table?** A 2-D summary crossing one variable against another, with an aggregation function.
9. **`pivot_table` vs `groupby`?** Same computation, different shape: pivot = wide matrix, groupby = long table.
10. **`fill_value=0` in pivot?** Replaces NaN for missing region–category combinations with zero, which is correct when no sale occurred.
11. **Grouped vs stacked bar?** Grouped compares categories side by side; stacked shows composition of each total.
12. **Why standardise region text?** ` west`, `West`, `WEST` would otherwise become separate groups.
13. **Which region has the highest AOV?** Central (₹54,125) despite the lowest total — few but high-value orders.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit V — data aggregation and group operations
- **Split-apply-combine:** split data into groups → apply a function to each → combine results.
- **GroupBy mechanics:** `df.groupby(key)` creates a grouped object; `.agg/.sum/.mean` materialise results.
- **Pivot tables** and **cross-tabulation** (`pd.crosstab`) summarise multi-dimensional relationships.
- **Time-series aggregation:** `resample('M')`, `rolling(window).mean()`, date ranges, frequency conversion.

| Operation | Use |
|---|---|
| `groupby().agg()` | flexible multi-metric summaries |
| `pivot_table` | 2-D matrix summary |
| `crosstab` | frequency counts between two categoricals |
| `resample` | time-based aggregation |
| `rolling` | moving-window statistics |

**Q&A**
1. **What is split-apply-combine?** The conceptual model behind groupby: partition → compute per partition → stitch back.
2. **`sum` vs `mean` vs `count`?** Total, average, and number of non-null observations respectively.
3. **Why `sort_values(ascending=False)`?** Ranks regions so the top performer is first.
4. **What is cross-tabulation?** A table of counts/frequencies for combinations of two categorical variables.
5. **What is a MultiIndex result?** Grouping by two keys returns a hierarchical index; `reset_index()` flattens it.
6. **Time-series resampling?** Changing the frequency of a time series (daily → monthly) with an aggregation rule.
7. **What is a moving window?** A statistic computed over a sliding fixed-length window to smooth/trend a series.
8. **Why aggregate before visualising?** Plotting 300 raw rows is noisy; regional totals make comparison clear.
9. **What does `df.groupby('Region')['Sales_Amount'].transform('sum')` do?** Returns each row's group total (no reduction), useful for per-row share calculations.

## 3. LEAST LIKELY — Stretch / Advanced / Edge Cases

1. **Aggregation hides distribution:** two regions with equal totals can have very different order-size distributions; pair totals with box plots.
2. **NaN-aware aggregation:** `sum()` skips NaN by default; `count()` counts non-null values only — use `size()` for row counts including nulls.
3. **Named aggregation pitfalls:** column names collide if the same alias is reused; keep aliases unique.
4. **Multi-key ordering:** after `groupby(['Region','Product_Category'])`, results are sorted by keys; `sort=False` preserves first-appearance order.
5. **Small-group bias:** Central's high AOV comes from only 29 transactions — a small sample; report counts alongside averages.
6. **Currency scaling:** ₹1.16 crore totals need `height/1e5` formatting for readable axis labels (as done with "L" suffixes).
7. **Cross-unit link (Unit V time series):** wrapping `Transaction_Date` as datetime enables monthly revenue trends via `resample`, extending this regional analysis over time.

**If you remember only one thing:** repair missing values → standardise group keys → `groupby().agg()` → add share/rank → `pivot_table` for two-way summaries.
