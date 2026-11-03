# Experiment 4 — Data Wrangling (Viva Theory)

> **30-second answer:** Start with messy real-estate data (150 rows, ugly column names, missing values, injected outliers), **clean column names with regex**, drop missing targets, impute by group, **left-merge** neighbourhood amenities, filter to residential sales from 2024, cap IQR outliers, aggregate/pivot price by neighbourhood and property type, and encode categories (ordinal map, `LabelEncoder`, one-hot).

## 1. MOST LIKELY — Core of the Experiment (High Yield)

- **Aim:** Wrangle a messy real-estate dataset into a clean, analysis-ready table and uncover price drivers.
- **Dataset:** `RealEstate_Prices.csv` — **150 rows × 9 columns**, messy names like `Property ID #`, `Area (Sq.Ft.)`, `Sale Price (INR Lakhs)`; plus `Neighborhood_Amenities.csv` (6 neighbourhoods).
- **Pipeline:**
  1. `read_csv`; normalise columns with regex: remove non-word chars, spaces/slashes → `_`, collapse repeated `_`.
  2. Rename to concise names (`Neighborhood_Locality` → `Neighborhood`, `Sale_Price_INR_Lakhs` → `Sale_Price_Lakhs`).
  3. Missing: **drop rows where the target is missing** (`dropna(subset=['Sale_Price_Lakhs'])`, 3 rows); impute `Area_SqFt` by **median per Property_Type**; other numerics by median, categoricals by mode.
  4. `pd.merge(df, amenities, on='Neighborhood', how='left')` — enriches 9 → 12 columns.
  5. Filter: `Property_Type.isin(residential)` & `Sale_Date >= '2024-01-01'` & `Area_SqFt >= 600` → **83 rows**.
  6. IQR outlier capping on area and price; derive `Price_Per_SqFt_INR`.
  7. Aggregate by neighbourhood; `pivot_table` neighbourhood × property type.
  8. Encode: furnishing ordinal map; `LabelEncoder` for property type; `get_dummies` for nominal columns.

**Code skeleton**
```python
df = pd.read_csv("RealEstate_Prices.csv")
df.columns = [re.sub(r'_+','_', re.sub(r'[^\w\s]','',c).strip().replace(' ','_'))
              for c in df.columns]
df = df.rename(columns={'Neighborhood_Locality':'Neighborhood',
                        'Sale_Price_INR_Lakhs':'Sale_Price_Lakhs'})
df.dropna(subset=['Sale_Price_Lakhs'], inplace=True)
df['Area_SqFt'] = df.groupby('Property_Type')['Area_SqFt'].transform(
    lambda x: x.fillna(x.median()))
df = pd.merge(df, amenities_df, on='Neighborhood', how='left')
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
df = df[(df.Property_Type.isin(['Apartment','Villa','Row House'])) &
        (df.Sale_Date >= '2024-01-01') & (df.Area_SqFt >= 600)].copy()
df['Price_Per_SqFt_INR'] = df.Sale_Price_Lakhs * 100000 / df.Area_SqFt
df.pivot_table(values='Sale_Price_Lakhs', index='Neighborhood',
               columns='Property_Type', aggfunc='mean')
```

**Expected output**
- Missing before: `Area_SqFt` **8**, `BHK` **6**, `Furnishing_Status` **4**, `Sale_Price_Lakhs` **3** → **0 after**.
- Merge: **147 × 9 → 147 × 12**.
- Filter: **147 → 83 rows**.
- Outlier fences after filtering: Area **0** outside [−112.25, 3781.75]; Price **0** outside [−73.28, 366.16].
- Neighbourhood summary (avg price): Baner **161.70**, Hinjewadi **154.43**, Wakad **153.66**, Kothrud **151.60**, Koregaon Park **148.95**, Viman Nagar **147.29** lakhs.
- Pivot table of mean price by neighbourhood × {Apartment, Row House, Villa}.
- Final encoded columns: 20, including `Loc_*`, `Type_*`, `Furnishing_Encoded`, `Property_Type_Label`.

**Viva Q&A**
1. **What is data wrangling?** Transforming raw data into a clean, structured, analysis-ready form — cleaning, merging, reshaping, enriching.
2. **Why regex on column names?** Spaces/special characters break attribute access and make code error-prone; normalised names are machine-friendly.
3. **What does `re.sub(r'[^\w\s]','',c)` do?** Removes characters that are neither word characters nor whitespace (e.g. `#`, `!`, `(`, `)`).
4. **Why drop rows with missing target?** You cannot train/summarise a price model without the price; imputing the target would fabricate outcomes.
5. **Why median area per property type?** Villas and apartments have different area scales; a global median misrepresents both.
6. **`how='left'` meaning?** Keep all left rows and attach matching right values; non-matches get NaN.
7. **What if a neighbourhood were missing from amenities?** Left join keeps the property with NaN amenity fields; an inner join would silently drop it.
8. **Why filter before aggregating?** Answers a specific business question (recent residential sales) and removes commercial/noise records.
9. **Why cap outliers rather than delete?** Preserves sample size and avoids selection bias; extreme but real luxury sales still inform the model.
10. **What is `Price_Per_SqFt`?** A derived metric enabling fair comparison across differently sized properties.
11. **`pivot_table` vs `groupby`?** `pivot_table` reshapes into a matrix (index × columns); `groupby` returns a long/tabular result.
12. **Why three encoding methods?** Ordinal data keeps order; tree models can use labels; nominal data needs one-hot to avoid false ordering.
13. **What does `drop_first=True` prevent?** Multicollinearity from redundant dummy columns.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit IV — combining, merging, reshaping
- **`merge`** is a database-style join (`inner`, `left`, `right`, `outer`) on keys.
- **`concat`** stacks along an axis; **`join`** merges on the index.
- **Hierarchical indexing** (MultiIndex) allows grouping by more than one key without losing structure.
- **Reshaping:** `stack` → long format, `unstack` → wide format, `pivot`/`pivot_table` → matrix summaries.

| Join | Keeps |
|---|---|
| inner | only matching keys |
| left | all left rows |
| right | all right rows |
| outer | all rows from both |

**Q&A**
1. **Label vs one-hot encoding?** Label assigns integers (suitable for ordinal/tree models); one-hot creates binary columns (suitable for nominal/linear models).
2. **When is label encoding harmful?** When a linear model interprets 1 < 2 < 3 as a real order that does not exist.
3. **What is a MultiIndex?** An index with multiple levels, created by `set_index(['Region','Category'])`; enables `.loc` slicing per level.
4. **`pivot` vs `pivot_table`?** `pivot` cannot aggregate duplicates; `pivot_table` accepts `aggfunc`.
5. **Why `reset_index()` after groupby?** Converts the group keys from index into regular columns for plotting/export.
6. **What is a derived variable?** A new column computed from existing ones (e.g. price per sq ft).
7. **What is feature encoding?** Converting categorical values to numbers a model can consume.
8. **Why clean column names?** Avoids syntax errors and makes merges/joins reliable across datasets.

## 3. LEAST LIKELY — Stretch / Advanced / Edge Cases

1. **Dtype surprises:** after `dropna`, pandas may keep object dtype; explicitly cast with `astype(float)`.
2. **Merge explosion:** many-to-many joins can duplicate rows; check `len()` before/after and validate key uniqueness.
3. **Outlier order matters:** detecting outliers after filtering changes the fences; document whether fences are global or subset-based.
4. **Encoding leakage:** fit `LabelEncoder` on the training set only in a modelling pipeline; fitting on all data leaks category knowledge.
5. **Target leakage:** `Price_Per_SqFt` is derived from the target — never use it as a predictor when predicting price.
6. **Skewed prices:** real estate prices are right-skewed; a log transform often improves linear models.
7. **Cross-unit link (Unit VI):** a logical data model (entities/relationships) is the conceptual ancestor of these joins — properties ↔ neighbourhoods is a many-to-one relationship.

**If you remember only one thing:** clean names → handle missing by group → merge → filter → cap outliers → aggregate/pivot → encode.
