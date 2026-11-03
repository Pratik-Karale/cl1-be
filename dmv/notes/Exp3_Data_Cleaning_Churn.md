# Experiment 3 — Data Cleaning and Preparation (Viva Theory)

> **30-second answer:** Load the telecom **customer-churn** CSV, explore it, remove **5 duplicate rows** (205 → 200), standardise inconsistent text/casing, convert `TotalCharges` to numeric, impute missing tenure/charges with median, **cap IQR outliers**, engineer churn-related features, one-hot encode, scale, and split **80/20 stratified** (160 train / 40 test).

## 1. MOST LIKELY — Core of the Experiment (High Yield)

- **Aim:** Clean and prepare raw telecom data so it can be used to model customer churn.
- **Dataset:** `Telecom_Customer_Churn.csv`, **205 rows × 15 columns** (200 generated + 5 injected duplicates). Target `Churn` ∈ {Yes, No} with inconsistent casing.
- **Pipeline:**
  1. `read_csv`, inspect `shape`, `info()`, `describe()`, `head()`.
  2. `drop_duplicates()` and `drop_duplicates(subset=['customerID'], keep='first')` → **200 rows**.
  3. Strip whitespace on all object columns; lower-case then map/replace inconsistent categories (`M/male` → Male, `month to month` → Month-to-month).
  4. `pd.to_numeric(TotalCharges, errors='coerce')` — blanks became NaN.
  5. Missing values: tenure 4, MonthlyCharges 6, TotalCharges 4 → **0 after imputation** (numeric = median, categorical = mode; tenure 0 ⇒ TotalCharges 0).
  6. Outliers via **IQR rule**; **cap/clip** to fences instead of deleting rows.
  7. Feature engineering: `Tenure_Group` (`pd.cut`), `Total_Services_Count`, `Charge_Increase_Ratio`, `Churn_Binary`.
  8. `pd.get_dummies(drop_first=True)` + `StandardScaler`; export cleaned and model-ready CSVs.
  9. `train_test_split(test_size=0.20, random_state=42, stratify=y)`.

**Code skeleton**
```python
df = pd.read_csv("Telecom_Customer_Churn.csv")
df.drop_duplicates(inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
for c in df.select_dtypes(include=[np.number]):
    df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes(include='object'):
    df[c] = df[c].fillna(df[c].mode()[0])
Q1, Q3 = df['MonthlyCharges'].quantile([.25, .75])
IQR = Q3 - Q1
df['MonthlyCharges'] = df['MonthlyCharges'].clip(Q1-1.5*IQR, Q3+1.5*IQR)
df = pd.get_dummies(df.drop(columns=['customerID','Churn']), drop_first=True, dtype=int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y)
```

**Expected output**
- Duplicates removed: **5**; remaining **200 rows**.
- Missing before: tenure 4, MonthlyCharges 6, TotalCharges 4 → **0 after**.
- IQR outliers capped: tenure **1** (fence −41 to 111), MonthlyCharges **1** (−471.26 to 2119.12), TotalCharges **1**.
- Scaled features: mean ≈ 0, std ≈ 1.
- Train `X_train=(160, 23)`, test `X_test=(40, 23)`.
- Exports: `Cleaned_Telecom_Customer_Churn.csv`, `Model_Ready_Telecom_Churn.csv`.

**Viva Q&A**
1. **Why clean data before modelling?** Garbage in → garbage out; duplicates bias counts, NaNs break estimators, outliers distort means/scaling.
2. **`drop_duplicates()` vs `subset=`?** Full-row duplicate vs duplicate by key (`customerID`), keeping the first record.
3. **Why is `TotalCharges` an object?** It contains blank strings for new customers; `to_numeric(errors='coerce')` turns blanks into NaN.
4. **Mean or median imputation?** Median — robust to skew and outliers, which this data has.
5. **Mode imputation?** For categorical columns; fills with the most frequent category.
6. **What is the IQR rule?** `IQR = Q3−Q1`; values outside `Q1−1.5·IQR` and `Q3+1.5·IQR` are flagged as outliers.
7. **Why cap instead of drop?** Clipping preserves every row and the sample size; dropping could remove genuine customers.
8. **Why `np.where(tenure > 0, TotalCharges/tenure, MonthlyCharges)`?** Avoids division by zero when computing historical average spend.
9. **Why one-hot encode?** Most models need numeric input and nominal categories have no order.
10. **Why `drop_first=True`?** Avoids the **dummy-variable trap** (perfect multicollinearity) for linear models.
11. **Why `StandardScaler`?** Puts features on the same scale so distance/gradient-based models are not dominated by large-value columns.
12. **Why `stratify=y`?** Keeps the churn/non-churn proportion identical in train and test, important for imbalanced targets.
13. **What if you scale before splitting?** Data leakage: test statistics influence training; fit the scaler on train only.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit I — descriptive statistics
- **Mean/median/mode**, **variance**, **standard deviation** summarise centre and spread.
- **Skewness** measures asymmetry; **kurtosis** measures tail heaviness. Median is preferred when skew ≠ 0.

### Unit IV — data preparation concepts
- **Missing data mechanisms:** MCAR (completely at random), MAR (explained by other variables), MNAR (depends on the missing value itself).
- **Outlier detection:** IQR (univariate), z-score (>3σ), isolation forest (multivariate).

| Issue | Typical fix |
|---|---|
| Missing numeric | median/mean, model-based imputation |
| Missing categorical | mode, "Unknown" category |
| Duplicates | `drop_duplicates` |
| Inconsistent text | `strip`, `lower`, mapping |
| Outliers | cap, remove, transform |
| Different scales | StandardScaler / MinMaxScaler |
| Nominal categories | one-hot encoding |

**Q&A**
1. **Feature engineering — why?** Domain-derived columns (tenure buckets, service count) expose signals the raw columns hide.
2. **`pd.cut` vs `pd.qcut`?** `cut` uses fixed bin edges; `qcut` uses quantiles so bins are equal-sized.
3. **Standardisation vs normalisation?** Standard = `(x−μ)/σ` (mean 0, std 1); Min-Max = `(x−min)/(max−min)` (range 0–1).
4. **Why is churn an imbalanced classification problem?** Far fewer churners than stayers, so accuracy alone is misleading; use precision/recall/F1.
5. **What is data leakage?** Information from the test set (or the target's future) leaks into training, inflating scores.
6. **Why convert `SeniorCitizen` to int?** It is already binary 0/1; keeping it numeric is correct.
7. **What does `errors='coerce'` do?** Turns unparseable values into NaN instead of raising.
8. **Why export two files?** One human-readable cleaned dataset and one fully encoded/scaled model-ready dataset.

## 3. LEAST LIKELY — Stretch / Advanced / Edge Cases

1. **Imputation bias:** median imputation shrinks variance and can hide subgroup differences; consider group-wise or model-based imputation.
2. **MNAR danger:** if high spenders disproportionately omit `TotalCharges`, imputation silently biases the result.
3. **Scaling only continuous features:** the practical scales a list of numeric columns; binary dummies should normally be left unscaled.
4. **Class imbalance fixes:** class weights, SMOTE/undersampling, threshold tuning — not just accuracy.
5. **IQR on skewed data:** the fences can be negative (as seen for tenure); for counts, lower bound 0 is more sensible.
6. **Pipeline hygiene:** put imputer + scaler inside a `Pipeline` so cross-validation refits them per fold.
7. **Cross-unit link (Unit II):** before claiming churn drivers, test group differences with a t-test/chi-square rather than eyeballing means.

**If you remember only one thing:** dedupe → standardise text/types → impute (median/mode) → cap IQR outliers → engineer → encode/scale → **stratified split**.
