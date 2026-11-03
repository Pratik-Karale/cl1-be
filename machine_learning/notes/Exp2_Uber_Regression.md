# Experiment 2 — Uber Fare Prediction with Linear, Ridge and Lasso Regression (Viva Theory)

> **30-second answer:** Clean the Uber fares dataset, detect outliers with the IQR rule, inspect correlations, then compare **Linear Regression, Ridge (L2) and Lasso (L1)** for predicting `fare_amount`. All three score **R² ≈ 0.0013**, showing the available features barely explain fare.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

**Aim:** Pre-process the Uber fares data (clean, detect outliers, check correlation) and fit Linear, Ridge and Lasso regression models; compare with **R², RMSE and MAE**.

**Dataset:** `uber.csv` — original shape **(200,000 × 9)**. Columns include `fare_amount` (target), `pickup_datetime`, pickup/dropoff latitude–longitude, `passenger_count`.

**Algorithm in 5 bullets**
- Remove infinities, drop missing target values, keep `fare_amount ≥ 0`.
- Validate coordinates (`lon` −180…180, `lat` −90…90) and `passenger_count` 1–6.
- Parse datetime → `pickup_hour`, `pickup_day`, `pickup_month`, `pickup_weekday`; drop raw datetime.
- Detect fare outliers with the **IQR rule**: outlier if value < Q1 − 1.5·IQR or > Q3 + 1.5·IQR.
- Model pipeline: **SimpleImputer(median) → StandardScaler → Linear / Ridge(α=1.0) / Lasso(α=0.01, max_iter=10000)**.

**Code skeleton**
```python
df = pd.read_csv("uber.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["fare_amount"])
df = df[df["fare_amount"] >= 0]
dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
df["pickup_hour"] = dt.dt.hour
q1, q3 = df["fare_amount"].quantile([.25, .75]); iqr = q3 - q1
outliers = (df["fare_amount"] < q1 - 1.5*iqr) | (df["fare_amount"] > q3 + 1.5*iqr)

pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler()),
                 ("model", Ridge(alpha=1.0))])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(r2_score(y_test, pred), np.sqrt(mean_squared_error(y_test, pred)))
```

**Expected output**
- Rows after basic cleaning: **199,261**; fare outliers by IQR: **17,120**.
- Correlation with fare is very weak: `pickup_month 0.0235`, `dropoff_longitude 0.0119`, `pickup_longitude 0.0116`, `passenger_count 0.0102`, `pickup_hour −0.0215`.
- Model comparison:

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Lasso | 0.0013 | 10.0735 | 6.0905 |
| Ridge | 0.0013 | 10.0738 | 6.0910 |
| Linear Regression | 0.0013 | 10.0738 | 6.0910 |

**Viva Q&A**
1. **Why is R² so low?** Raw coordinates and timestamps do not encode trip distance or traffic; the model is under-specified, not broken.
2. **What does R² = 0.0013 mean?** The model explains only **0.13%** of fare variance — barely better than predicting the mean.
3. **Why standardise before regularisation?** Ridge/Lasso penalise coefficients; features on different scales would be penalised unfairly.
4. **What does Ridge do?** Adds an **L2 penalty** = λ × (sum of squared weights) — shrinks coefficients smoothly, handles multicollinearity.
5. **What does Lasso do?** Adds an **L1 penalty** = λ × (sum of absolute weights) — can drive coefficients exactly to **zero** (feature selection).
6. **Why identical scores?** With weak, mostly uncorrelated features and small α, all three converge to nearly the same near-zero model.
7. **What is the IQR rule?** A robust outlier test based on quartiles; it does not assume normality.
8. **Why impute with the median?** Median is robust to skewed fares and outliers; mean would be pulled by extreme values.
9. **What if you added trip distance?** R² would rise sharply — distance is the dominant fare driver.
10. **RMSE vs MAE?** RMSE penalises large errors more (squares them); MAE is the average absolute error.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit II: Regression
- **Regression** predicts a **continuous** target; **correlation** measures linear association between two variables (no prediction direction).
- **Types:** univariate vs multivariate; linear vs nonlinear; simple vs multiple linear.
- **Bias–variance trade-off:** high bias = underfitting; high variance = overfitting.
- **Ridge (L2):** Loss = MSE + λ × (sum of squared weights) — shrinks, keeps all features.
- **Lasso (L1):** Loss = MSE + λ × (sum of absolute weights) — sparse solution.
- **ElasticNet:** combines L1 + L2: λ₁ × (sum of absolute weights) + λ₂ × (sum of squared weights).
- **Metrics:**

| Metric | In plain words | Meaning |
|---|---|---|
| MSE | average of (actual − predicted)² | squared error |
| RMSE | square root of MSE | error in target units |
| MAE | average of \|actual − predicted\| | average absolute error |
| R² | 1 − (unexplained variation ÷ total variation) | fraction of variance explained |
| Adjusted R² | R² adjusted for the number of predictors | penalises extra predictors |

**Likely Q&A**
- *Why is R² preferred over MSE in reporting?* It is scale-free and interpretable (0–1; can be negative for bad models).
- *When does Ridge beat Lasso?* When many features each contribute a little; Lasso wins when only a few features matter.
- *What is stepwise regression?* Automatic forward/backward selection of predictors.
- *Difference between linear and polynomial regression?* Polynomial adds powers/interactions, still linear in coefficients.
- *What is Bayesian linear regression?* Places a prior on weights; returns a distribution over coefficients.

## 3. LEAST LIKELY — Stretch / Edge Cases

- **Outlier handling choices:** removal vs capping (Winsorisation) vs robust loss — removal can bias the model if outliers are genuine.
- **Data leakage:** fitting the scaler/imputer on the full dataset instead of only the training set leaks test information.
- **Target leakage:** including a column derived from fare would inflate R².
- **Multicollinearity:** ridge stabilises unstable coefficients; Lasso arbitrarily picks one of correlated features.
- **Nonlinearity:** tree-based or polynomial features capture distance effects better.
- **Cross-unit link:** residual analysis feeds Unit III classification metrics and Unit V feature importance.
- **Trap question:** "Does a lower RMSE always mean a better model?" No — compare on the same test set and check whether the improvement is meaningful.

**If you remember only one thing:** Ridge shrinks all coefficients (L2), Lasso can zero them out (L1), and this Uber model's low R² proves feature choice matters more than the regressor.
