# Common Syntax & Viva Cheat Sheet
### Machine Learning (417521) + Data Modeling & Visualization (417522) — Computer Laboratory I

> **How to use this file (3-hour plan):** 2 min on Section 0, then read the table for the subject you are being examined on. Section 3 is the shared viva bank — skim the bold question, read only the answers you cannot say in one line. Every answer below is already in "exam answer" form.

**Companion file:** `Evaluation_Metrics_Explained.md` — every metric (classification, regression, clustering, statistics, RL) with plain-text formulas and one-line explanations.

---

## 0. The one pattern behind every ML practical

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

model.fit(X_train, y_train)      # learn
pred = model.predict(X_test)     # infer
score = metric(y_test, pred)     # evaluate
```

| Call | Meaning | Viva one-liner |
|---|---|---|
| `fit(X, y)` | learn parameters from training data | "Trains the model." |
| `transform(X)` | apply learned preprocessing | "Scales/encodes using **training** statistics." |
| `fit_transform(X)` | learn + apply in one step | "Only on **training** data — using it on test data leaks information." |
| `predict(X)` | output labels/values | "Returns the predicted class or number." |
| `predict_proba(X)` | class probabilities | "Returns probability per class; `predict` takes the argmax." |
| `Pipeline([...])` | chain preprocessing + model | "Prevents data leakage and keeps train/test identical." |

**Shape rule:** `X` is always 2-D `(n_samples, n_features)`; `y` is 1-D `(n_samples,)`. A single sample passed to `predict` must still be 2-D: `model.predict([[5.1, 3.5, 1.4, 0.2]])`.

---

## 1. Machine Learning syntax nuances (417521)

### 1.1 Loading and splitting
```python
from sklearn.datasets import load_iris, load_digits
iris = load_iris()                 # .data, .target, .target_names
digits = load_digits()             # 8x8 images flattened to 64 features
df = pd.read_csv("uber.csv")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
```
- `random_state=42` → reproducible split.
- `stratify=y` → keeps class proportions in train and test (important for classification).
- `test_size=0.25` → 25% test, 75% train.

### 1.2 Preprocessing
```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   estimator)])
```
| Task | Code | Note |
|---|---|---|
| Standardize | `StandardScaler().fit_transform(X_train)` | mean 0, std 1; `transform(X_test)` only |
| Impute | `SimpleImputer(strategy="median")` | median is robust to outliers |
| Reduce dimensions (unsupervised) | `PCA(n_components=k).fit_transform(X)` | use `.explained_variance_ratio_` |
| Reduce dimensions (supervised) | `LinearDiscriminantAnalysis().fit(X, y)` | needs labels; maximizes class separation |

### 1.3 Models used in the practicals
| Experiment | Code | Key parameters |
|---|---|---|
| LDA | `LinearDiscriminantAnalysis()` | no real hyperparameters; supervised |
| Linear regression | `LinearRegression()` | none |
| Ridge | `Ridge(alpha=1.0)` | `alpha` = L2 penalty strength |
| Lasso | `Lasso(alpha=0.01, max_iter=10000)` | `alpha` = L1 penalty; can zero coefficients |
| ElasticNet | `ElasticNet(alpha=..., l1_ratio=...)` | mix of L1 + L2 |
| SVM | `SVC(kernel="rbf", C=5, gamma="scale")` | `C` = margin softness; `gamma` = kernel width |
| K-Means | `KMeans(n_clusters=k, random_state=42, n_init=10)` | `n_init` = restarts; `.inertia_` after fit |
| Voting | `VotingClassifier(estimators=[...], voting="hard")` | hard = majority label; soft = average probability |
| AdaBoost | `AdaBoostClassifier(n_estimators=100, random_state=42)` | sequential reweighting |
| Gradient Boosting | `GradientBoostingClassifier(random_state=42)` | fits residuals |
| Random Forest | `RandomForestClassifier(n_estimators=150, random_state=42)` | bagged trees |
| XGBoost | `XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, objective="multi:softprob", num_class=3, eval_metric="mlogloss")` | labels must be `0..n-1` |

### 1.4 Metrics — exact calls
```python
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             r2_score, mean_squared_error, mean_absolute_error,
                             silhouette_score)

accuracy_score(y_test, pred)
confusion_matrix(y_test, pred)                    # rows = actual, cols = predicted
classification_report(y_test, pred, target_names=..., zero_division=0)
precision_score(y_test, pred, average="macro", zero_division=0)
r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))  # MSE is squared — take sqrt
mean_absolute_error(y_test, pred)
silhouette_score(X, labels)                       # clustering quality, -1 to +1
```

### 1.5 Gotchas that examiners love
| Trap | Correct statement |
|---|---|
| `mean_squared_error` vs RMSE | MSE is in squared units; **RMSE = √MSE** |
| `fit_transform` on test data | Wrong — leaks test statistics. Use `transform`. |
| Accuracy on imbalanced data | Misleading; use **macro F1**, precision, recall. |
| `macro` vs `weighted` average | Macro = unweighted mean per class (treats rare classes equally); weighted = support-weighted. |
| PCA vs LDA | PCA is **unsupervised** (max variance); LDA is **supervised** (max class separation). |
| K-Means without scaling | Distance-based — **always scale first**. |
| K-Means `n_init` | Runs the algorithm several times with different centroids and keeps the best inertia. |
| `inertia_` | Within-cluster sum of squared distances; lower is tighter, but always falls as k rises — use the elbow. |
| Precision vs recall | Precision = of predicted positives, how many are right; recall = of actual positives, how many were found. |
| When to prefer recall | Disease/fraud detection — missing a positive is costly. |
| When to prefer precision | Spam filter — false alarms are costly. |
| `predict` vs `predict_proba` | Label vs probability vector. |
| XGBoost error on labels | Encode labels to `0..n-1` first. |

---

## 2. Data Modeling & Visualization syntax nuances (417522)

### 2.1 Loading and inspecting
```python
df_csv   = pd.read_csv("sales_data.csv")
df_excel = pd.read_excel("sales_data.xlsx")      # needs openpyxl
df_json  = pd.read_json("sales_data.json")

df.shape; df.info(); df.describe().round(2); df.head()
df.dtypes; df.columns.tolist()
df.isnull().sum()[df.isnull().sum() > 0]
df.duplicated().sum()
```
| Task | Code |
|---|---|
| Write CSV / Excel / JSON | `to_csv(..., index=False)`, `to_excel(..., index=False)`, `to_json(..., orient="records", indent=4)` |
| Rename / clean columns | `df.columns = [c.strip().lower() for c in df.columns]`; `df.rename(columns={...}, inplace=True)` |
| Clean names by regex | `re.sub(r"_+", "_", re.sub(r"[^\w\s]", "", col).strip().replace(" ", "_"))` |

### 2.2 Cleaning
```python
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=["customerID"], keep="first", inplace=True)

df[col] = df[col].fillna(df[col].median())       # numeric, robust
df[col] = df[col].fillna(df[col].mode()[0])      # categorical
df = df.dropna(subset=["fare_amount"])           # drop only where target missing
df["Quantity"].fillna(df["Quantity"].median())
df.groupby("Product_Category")["Unit_Price"].transform(
    lambda x: x.fillna(x.median()))              # group-wise imputation

df = df.replace([np.inf, -np.inf], np.nan)
```

**IQR outlier rule**
```python
q1, q3 = df[col].quantile([.25, .75])
iqr = q3 - q1
lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
outliers = ((df[col] < lower) | (df[col] > upper)).sum()
df[col] = df[col].clip(lower, upper)             # cap (Winsorize), or filter rows
```

**Encoding and scaling**
```python
df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})
df_encoded = pd.get_dummies(df_model, drop_first=True, dtype=int)   # one-hot
df["col"] = LabelEncoder().fit_transform(df["col"])                # label encoding
df[num_cols] = StandardScaler().fit_transform(df[num_cols])
```

### 2.3 Dates and time series
```python
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"], errors="coerce", utc=True)
df["Hour"]    = df["Timestamp"].dt.hour
df["Month"]   = df["Timestamp"].dt.month
df["Weekday"] = df["Timestamp"].dt.dayofweek
df["Date"]    = df["Timestamp"].dt.date

df["Temp_Rolling_24h_Avg"] = df["Temperature_C"].rolling(window=8, min_periods=1).mean()
daily = df.groupby("Date").agg(avg_temp=("Temperature_C", "mean")).reset_index()
```
- `rolling(window=8)` on 3-hourly data = 24-hour moving average.
- `resample("D")` needs a **DatetimeIndex**: `df.set_index("Date").resample("D")["AQI"].mean()`.

### 2.4 Combining, grouping, reshaping
```python
unified = pd.concat([df_csv, df_excel, df_json], ignore_index=True)
merged  = pd.merge(df, amenities_df, on="Neighborhood", how="left")

df.groupby("Region")["Sales_Amount"].sum()
df.groupby("Region").agg(total=("Sales_Amount","sum"),
                         avg=("Sales_Amount","mean")).reset_index()

df.pivot_table(values="Sale_Price_Lakhs", index="Neighborhood",
               columns="Property_Type", aggfunc="mean")
pd.crosstab(df["Region"], df["Product_Category"])
```
| Operation | Purpose | One-liner |
|---|---|---|
| `concat` | stack rows/columns | "Glues tables together by axis." |
| `merge` | SQL-style join on a key | "`how='left'` keeps every left row." |
| `groupby` | split → apply → combine | "One row per group after aggregation." |
| `pivot_table` | spreadsheet-style summary | "Default `aggfunc='mean'`; use `sum` for totals." |
| `crosstab` | frequency table of two categoricals | "Counts combinations." |
| `sort_values` | order rows | `ascending=False` for top-first. |

### 2.5 Plotting (matplotlib + seaborn)
```python
import matplotlib.pyplot as plt, matplotlib.dates as mdates, seaborn as sns
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["Date"], df["AQI"], label="AQI")
ax.set_xlabel("Date"); ax.set_ylabel("AQI"); ax.set_title("AQI trend")
ax.legend(); ax.grid(alpha=.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

sns.boxplot(x=df["fare_amount"])
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.savefig("chart.png", dpi=300, bbox_inches="tight")
```
| Chart | Use when |
|---|---|
| Line | trend over time |
| Bar / stacked bar | compare categories / parts of a whole |
| Histogram | distribution of one variable |
| Box / violin | distribution + outliers across groups |
| Scatter / bubble | relationship between two/three variables |
| Heatmap | correlation matrix or 2-D intensity |
| Pie | share of whole (only a few categories) |

### 2.6 Web API
```python
import requests
resp = requests.get(url, params={"q": "Pune,IN", "appid": KEY, "units": "metric"}, timeout=10)
if resp.status_code == 200:
    data = resp.json()                 # dict, like a parsed JSON file
record = {
    "temp":     item["main"]["temp"],
    "humidity": item["main"]["humidity"],
    "rain":     item.get("rain", {}).get("3h", np.nan),
}
```
- `params` → query string; `timeout` → don't hang forever; `resp.json()` → Python dict.
- Nested access: `item["main"]["temp"]`; `.get(...)` avoids `KeyError`.

### 2.7 Gotchas that examiners love
| Trap | Correct statement |
|---|---|
| `inplace=True` return value | It returns `None`; never write `df = df.dropna(inplace=True)`. |
| `axis=0` vs `axis=1` | 0 = rows, 1 = columns. |
| `loc` vs `iloc` | `loc` = label-based; `iloc` = integer-position-based. |
| `read_excel` fails | Install `openpyxl`; pass `engine="openpyxl"` if needed. |
| `to_csv` adds an index | Always `index=False`. |
| `groupby` result index | Group keys become the index; call `.reset_index()` to flatten. |
| `pivot_table` default | `aggfunc="mean"` — pass `"sum"` for totals. |
| Correlation heatmap error | Use `df.corr(numeric_only=True)`; non-numeric columns break it. |
| Time column is a string | Convert with `pd.to_datetime` before `.dt` or `.resample`. |
| Missing values | Median for numeric with outliers; mode for categorical; drop only if rare. |
| Outlier methods | IQR = robust, no normality assumption; Z-score assumes roughly normal data. |
| One-hot vs label | One-hot for nominal (no order); label for ordinal or tree models. |
| Data cleaning vs wrangling | Cleaning fixes quality (missing, duplicates, types); wrangling reshapes/merges/derives. |

---

## 3. Common viva questions (both subjects)

### 3.1 Fundamentals
| Question | Answer |
|---|---|
| What is machine learning? | Algorithms that learn patterns from data instead of being explicitly programmed with rules. |
| Supervised vs unsupervised vs reinforcement | Supervised has labels; unsupervised finds structure; RL learns from reward and trial-and-error. |
| Classification vs regression | Classification predicts a class; regression predicts a continuous number. |
| What is a feature and a target? | Feature = input variable (`X`); target = output to predict (`y`). |
| Why split into train and test? | To measure generalization on unseen data, not memorization. |
| Overfitting vs underfitting | Overfit = fits training noise, poor on test (high variance); underfit = too simple, poor on both (high bias). |
| Bias–variance trade-off | Increasing model complexity lowers bias but raises variance; the optimum minimizes total error. |
| What does `random_state` do? | Fixes the random seed so splits/initializations are reproducible. |
| What does `stratify` do? | Keeps class proportions the same in train and test. |
| Why scale features? | Distance/gradient-based methods (KNN, SVM, K-Means, PCA, regularized regression) are dominated by large-scale features otherwise. |
| What is data leakage? | Using test or future information during training; makes scores look better than reality. |

### 3.2 Evaluation
| Question | Answer |
|---|---|
| Confusion matrix terms | TP, TN, FP (false alarm), FN (missed positive). |
| Precision | TP / (TP + FP) — trustworthiness of positive predictions. |
| Recall / sensitivity | TP / (TP + FN) — coverage of actual positives. |
| F1-score | Harmonic mean of precision and recall; good when classes are imbalanced. |
| Accuracy | (TP + TN) / total; misleading on imbalanced data. |
| Macro vs weighted average | Macro treats every class equally; weighted accounts for class frequency. |
| R² | Fraction of variance in the target explained by the model (1 = perfect). |
| RMSE vs MAE | RMSE penalizes large errors more (squared); MAE is the average absolute error. |
| When is a high R² suspicious? | Leakage, overfitting, or too few test points. |

### 3.3 Algorithms
| Question | Answer |
|---|---|
| PCA vs LDA | PCA: unsupervised, maximum variance. LDA: supervised, maximum between-class separation. |
| Ridge vs Lasso | Ridge = L2, shrinks coefficients; Lasso = L1, can set coefficients exactly to zero (feature selection). |
| Why regularize? | To reduce overfitting and stabilize coefficients when features are correlated. |
| SVM intuition | Finds the maximum-margin boundary; the kernel trick maps data to a higher dimension for non-linear separation. |
| What is `C` in SVM? | Regularization: small C = wider margin, more tolerance; large C = fits training points more strictly. |
| What is `gamma` in RBF SVM? | Reach of a single training point: small = far/smooth, large = local/complex. |
| Bagging vs boosting | Bagging trains models in parallel on bootstrap samples and averages (reduces variance); boosting trains sequentially on errors (reduces bias). |
| Random Forest | Bagged decision trees + random feature subsets; reduces variance. |
| Voting hard vs soft | Hard = majority label; soft = average predicted probabilities then argmax. |
| XGBoost vs AdaBoost | XGBoost uses regularized gradient boosting; AdaBoost reweights misclassified samples. |
| K-Means vs hierarchical vs DBSCAN | K-Means = centroid, needs k; hierarchical = tree/dendrogram, no k; DBSCAN = density, finds arbitrary shapes and noise. |
| Elbow vs silhouette | Elbow looks for the bend in inertia vs k; silhouette measures separation and cohesion (-1 to +1). |
| Why does K-Means fail sometimes? | Non-spherical clusters, different densities, outliers, poor initialization, unscaled features. |
| Q-Learning vs supervised learning | Q-Learning learns action values from delayed rewards with no labeled correct action. |
| What is epsilon-greedy? | With probability ε explore a random action; otherwise exploit the best known Q-value. |
| Why decay epsilon? | Explore early, exploit later as the Q-table becomes reliable. |
| Bellman equation (Q form) | `Q(s,a) = r + γ · max Q(s',a')` — current reward plus discounted best future value. |
| Role of `alpha` and `gamma` in Q-Learning | `alpha` = learning rate; `gamma` = discount factor for future rewards. |

### 3.4 Data handling (DMV)
| Question | Answer |
|---|---|
| DataFrame vs Series | DataFrame = 2-D table; Series = 1-D labeled column. |
| `loc` vs `iloc` | Label vs integer position. |
| `merge` vs `concat` | Merge joins on a key; concat stacks tables by axis. |
| Inner vs left vs outer join | Inner = matching keys only; left = all left rows; outer = all rows from both. |
| Split-apply-combine | Split data by group, apply an aggregation, combine into a result. |
| `pivot_table` vs `crosstab` | Pivot aggregates numeric values; crosstab counts frequencies of two categoricals. |
| Handling missing values | Drop if rare; impute mean/median (numeric) or mode (categorical); median is robust to outliers; group-wise median preserves subgroup behavior. |
| Handling duplicates | `drop_duplicates()`, optionally `subset=[id]`, `keep="first"`. |
| Handling inconsistent text | Lowercase, strip whitespace, map spelling variants, then re-check unique values. |
| Outlier detection | IQR fences (`Q1 − 1.5·IQR`, `Q3 + 1.5·IQR`) or Z-scores; cap/filter/drop. |
| One-hot vs label encoding | One-hot for unordered categories; label for ordered categories or trees. |
| Why normalize before modeling? | Puts features on a comparable scale, required for distance-based and regularized models. |
| What is hierarchical indexing? | Multi-level index (`MultiIndex`) enabling `stack`/`unstack` and grouped selection. |
| Reshaping operations | `pivot`/`melt` (wide ↔ long), `stack`/`unstack` (columns ↔ row index). |
| Time series components | Trend, seasonality, cyclical variation, noise. |
| Moving average purpose | Smooths short-term noise to reveal trend/seasonality. |
| Why convert dates to datetime? | Enables `.dt` accessors, correct sorting, resampling, and date-axis plots. |
| API request/response | Client sends HTTP request; server returns status code + body; `200 OK` means success. |
| JSON parsing | `response.json()` → dict/list; index into nested keys. |
| Descriptive vs inferential statistics | Descriptive summarizes the sample; inferential generalizes to the population. |
| Correlation vs causation | Correlation measures co-movement; it does not prove one variable causes the other. |
| Type I vs Type II error | Type I = false positive (reject a true null); Type II = false negative (fail to reject a false null). |
| Which chart for which data? | Time → line; category comparison → bar; distribution → histogram/box; relationship → scatter; correlation → heatmap. |
| Why not a pie chart with many slices? | Human perception of angle is poor; bars compare better. |
| 3-D plot caution | Perspective can distort values and hide points; use 2-D unless the third dimension is essential. |

---

## 4. Last-minute "say this" lines

- **Any sklearn practical:** "I split with stratification, fit only on training data inside a pipeline, predict on test, and report accuracy plus a confusion matrix and macro F1."
- **Any regression practical:** "I compare R², RMSE and MAE; RMSE is √MSE, and regularization controls overfitting."
- **Any clustering practical:** "I scale first, sweep k, pick the elbow, and confirm with the silhouette score."
- **Any ensemble practical:** "Bagging reduces variance by averaging; boosting reduces bias by fitting sequential weak learners."
- **Any RL practical:** "The agent updates `Q(s,a) += α[r + γ·max Q(s') − Q(s,a)]` and uses ε-greedy exploration."
- **Any DMV practical:** "I load → inspect shape/dtypes/missing values → clean and standardize → transform/merge/group → visualize and interpret."
- **Any API practical:** "`requests.get` with params and a timeout, check `status_code == 200`, parse `response.json()`, then flatten nested keys into a DataFrame."
- **Any visualization practical:** "I choose the chart from the data type: line for time, bar for categories, box for distribution, scatter for relationships, heatmap for correlation."
