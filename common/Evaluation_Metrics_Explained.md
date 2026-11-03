# Evaluation Metrics — Quick Explanation
### For Machine Learning (417521) and Data Modeling & Visualization (417522)

> **How to use this file:** Find your task in the 30-second map, then read only that section. Every formula is written in plain text, and every symbol is explained in the line below it. No LaTeX, no images.

---

## 0. The 30-second map — which metric belongs to which task

| Task | Main metrics | Use when |
|---|---|---|
| Binary classification, balanced | Accuracy, F1, AUC | Classes are roughly equal |
| Binary classification, imbalanced | Precision, Recall, F1, PR-AUC | One class is rare (fraud, disease) |
| Multiclass classification | Macro F1, weighted F1, confusion matrix | 3+ classes (Iris, digits) |
| Regression | RMSE, MAE, R² | Predicting a number (fare, price) |
| Clustering | Inertia, silhouette | No labels, finding groups |
| Reinforcement learning | Success rate, steps to goal, average return | Agent learning a task |
| Data quality (DMV) | Missing %, duplicate count, outlier count | Before modelling |
| Statistics (DMV) | Mean, median, variance, correlation, IQR | Describing and cleaning data |

---

## 1. Classification metrics

### 1.1 Start with the confusion matrix

For a two-class problem, the confusion matrix has four cells:

```
                        Predicted positive     Predicted negative
Actual positive              TP                     FN
Actual negative              FP                     TN
```

| Term | Plain meaning |
|---|---|
| **TP** (true positive) | It was positive, and we said positive |
| **TN** (true negative) | It was negative, and we said negative |
| **FP** (false positive) | It was negative, but we said positive — a false alarm |
| **FN** (false negative) | It was positive, but we said negative — a missed case |

Rows = actual class; columns = predicted class. The diagonal (TP and TN) is correct; off-diagonal is wrong.

### 1.2 The formulas, decoded

**Accuracy** — how many guesses were correct overall.
```
Accuracy = (TP + TN) ÷ (TP + TN + FP + FN)
```
The bottom is simply the total number of predictions. Good only when classes are balanced.

**Error rate** — the opposite of accuracy.
```
Error rate = (FP + FN) ÷ total = 1 − Accuracy
```

**Precision** — when we say "positive", how often are we right?
```
Precision = TP ÷ (TP + FP)
```
Bottom = everyone we *predicted* positive. High precision means few false alarms.

**Recall** (also called sensitivity) — of all the real positives, how many did we catch?
```
Recall = TP ÷ (TP + FN)
```
Bottom = everyone who *actually* was positive. High recall means few missed cases.

**Specificity** — of all the real negatives, how many did we correctly reject?
```
Specificity = TN ÷ (TN + FP)
```

**F1-score** — one number that balances precision and recall.
```
F1 = 2 × (Precision × Recall) ÷ (Precision + Recall)
```
It is the harmonic mean. It is high only when *both* precision and recall are high. It ignores TN.

**Balanced accuracy** — average of the two class recalls; fairer than accuracy when classes are imbalanced.
```
Balanced accuracy = (Recall of positive class + Recall of negative class) ÷ 2
```

### 1.3 Worked example

Suppose the confusion matrix is TP = 40, FN = 10, FP = 5, TN = 45. Total = 100.

| Metric | Calculation | Result |
|---|---|---|
| Accuracy | (40 + 45) ÷ 100 | **0.85** |
| Precision | 40 ÷ (40 + 5) | **0.889** |
| Recall | 40 ÷ (40 + 10) | **0.80** |
| Specificity | 45 ÷ (45 + 5) | **0.90** |
| F1 | 2 × (0.889 × 0.80) ÷ (0.889 + 0.80) | **0.842** |

Say it in a viva: *"Accuracy is 85%, but recall is only 80%, so we are missing one in five real positives."*

### 1.4 Multiclass versions

In multiclass problems, the confusion matrix is class × class (for example, 10 × 10 for digits).

- **Per-class precision** = correct predictions for that class ÷ all predictions for that class.
- **Per-class recall** = correct predictions for that class ÷ all actual examples of that class.
- **Macro average** = plain average of the per-class scores. Every class counts equally, so rare classes matter.
- **Weighted average** = average weighted by how many examples each class has. Big classes dominate.
- **Micro average** = add up TP, FP, FN across classes first, then compute one score. In single-label multiclass, micro F1 equals accuracy.

Rule of thumb: use **macro F1** when every class matters; use **weighted F1** when you care about overall performance on the actual class mix.

### 1.5 AUC-ROC (if asked)

AUC measures how well the model *ranks* positives above negatives.
- 0.5 = random guessing
- 1.0 = perfect ranking
- It is threshold-independent: it judges the scores, not one chosen cutoff.

### 1.6 Which one should you quote?

| Situation | Quote this |
|---|---|
| Balanced classes | Accuracy + F1 |
| Imbalanced classes | Precision, recall, F1 (not accuracy) |
| Missing a positive is dangerous | **Recall** (disease, fraud) |
| A false alarm is expensive | **Precision** (spam filter) |
| Need one balanced number | **F1** |
| Comparing many classes | Macro F1 + confusion matrix |

### 1.7 Where these appear in your practicals

- **Exp1 (LDA):** accuracy 1.00; confusion matrix is perfectly diagonal.
- **Exp3 (SVM):** accuracy 0.98; class 8 has the weakest recall at 0.93; macro F1 ≈ 0.98.
- **Exp5 (Ensemble):** models compared with accuracy, macro precision, macro recall and macro F1; Gradient Boosting wins at 0.9737.

---

## 2. Regression metrics

Here "actual" means the true value and "predicted" means the model's output.

**MSE (Mean Squared Error)** — average of the squared mistakes.
```
MSE = average of (actual − predicted)²
```
Squaring makes big errors count much more. Units are squared (rupees², for example), so it is hard to interpret directly.

**RMSE (Root Mean Squared Error)** — square root of MSE.
```
RMSE = √MSE
```
Back in the original units (rupees, years, etc.), so it is easy to explain. Still punishes large errors heavily.

**MAE (Mean Absolute Error)** — average size of the mistake, ignoring direction.
```
MAE = average of |actual − predicted|
```
Easy to interpret: "on average, we are off by 6 rupees." Less sensitive to extreme errors than RMSE.

**R² (R-squared)** — how much of the target's variation the model explains.
```
R² = 1 − (unexplained variation ÷ total variation)
   = 1 − [sum of (actual − predicted)²] ÷ [sum of (actual − mean)²]
```
- 1.0 = perfect
- 0.0 = no better than always predicting the mean
- Negative = worse than predicting the mean

**Adjusted R²** — R² with a penalty for adding useless features.
```
Adjusted R² = 1 − (1 − R²) × (n − 1) ÷ (n − p − 1)
```
`n` = number of samples, `p` = number of predictors. It can go down when a new feature does not help.

**MAPE (Mean Absolute Percentage Error)** — average percentage mistake.
```
MAPE = average of (|actual − predicted| ÷ |actual|) × 100%
```
Scale-free and easy to explain, but breaks when actual values are zero or near zero.

### Which regression metric to quote

| Question | Metric |
|---|---|
| "How large is a typical error, in real units?" | MAE |
| "Do big mistakes matter more?" | RMSE |
| "How much variation did we explain?" | R² |
| "Is the extra feature worth it?" | Adjusted R² |
| "Compare across different scales" | MAPE (careful with zeros) |

### Where these appear in your practicals

- **Exp2 (Uber):** R² ≈ 0.0013, RMSE ≈ 10.07, MAE ≈ 6.09. The model explains only 0.13% of fare variation — it is barely better than guessing the mean. That is the honest conclusion, not a code bug.

---

## 3. Clustering metrics

There are no labels, so we measure how tight and how separated the clusters are.

**Inertia (WCSS — Within-Cluster Sum of Squares)** — how compact the clusters are.
```
Inertia = sum over all points of (squared distance from the point to its own cluster centre)
```
Lower is tighter. But inertia always falls as you add more clusters, so never compare raw inertia across different k. Use the **elbow method**: plot inertia vs k and look for the bend.

**Silhouette score (per point)** — how well a point fits its own cluster compared with the nearest other cluster.
```
s = (b − a) ÷ max(a, b)
```
- `a` = average distance from the point to other points in its own cluster (want small)
- `b` = average distance from the point to points in the nearest other cluster (want large)
- Range: −1 to +1. Near +1 = well inside its cluster; near 0 = on a boundary; negative = probably in the wrong cluster.

**Average silhouette** — mean of the per-point scores. Rough guide: above 0.5 is reasonable, above 0.7 is strong.

**Davies–Bouldin index** — average ratio of within-cluster spread to between-cluster separation. **Lower is better.** (Less common in viva.)

**Calinski–Harabasz index** — between-cluster separation divided by within-cluster spread. **Higher is better.** (Less common in viva.)

### Where these appear in your practicals

- **Exp4 (K-Means):** inertia falls from 600 at k = 1 to 222 at k = 2 and 140 at k = 3; the elbow and the known 3 species give k = 3. Silhouette = 0.4599, which means moderate but real structure.

### Elbow vs silhouette

| Method | What it looks at | How to read it |
|---|---|---|
| Elbow | Inertia vs k | Look for the bend where extra clusters stop helping much |
| Silhouette | Separation vs cohesion per point | Higher average is better; no bend needed |

---

## 4. Statistics and data-quality metrics (DMV)

**Mean** — the average. Sensitive to extreme values.
```
Mean = sum of values ÷ number of values
```

**Median** — the middle value when sorted. Robust to outliers. Use it for skewed data such as income or fare.
```
Median = middle value (or average of the two middle values)
```

**Mode** — the most frequent value. Use it for categorical columns.

**Variance** — average squared distance from the mean. Measures spread, but in squared units.
```
Variance = average of (value − mean)²
```

**Standard deviation** — square root of variance, back in the original units.
```
Standard deviation = √Variance
```

**Covariance** — do two variables move together? Positive = together, negative = opposite. The size depends on units, so the sign is the useful part.

**Correlation (Pearson)** — covariance scaled to −1 … +1, so it is comparable across variables.
```
Correlation = covariance ÷ (standard deviation of x × standard deviation of y)
```
- +1 = perfect positive straight-line relationship
- 0 = no linear relationship
- −1 = perfect negative straight-line relationship
- Correlation does **not** prove causation.

**IQR (Interquartile Range)** — spread of the middle half of the data.
```
IQR = Q3 − Q1        (Q1 = 25th percentile, Q3 = 75th percentile)
Lower fence = Q1 − 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
```
Values outside the fences are flagged as outliers. This is the rule used in Exp2, Exp3 and Exp4.

**Z-score** — how many standard deviations a value is from the mean.
```
Z = (value − mean) ÷ standard deviation
```
Roughly, |Z| > 3 is an outlier. This assumes a roughly normal distribution; IQR does not.

**Skewness** — asymmetry. Positive = long right tail; negative = long left tail.
**Kurtosis** — how heavy the tails are compared with a normal distribution.

**Data-quality counts** — always cheap and useful in a viva:
- Missing % = missing cells ÷ total cells × 100
- Duplicate count = number of repeated rows
- Outlier count = number of values outside the IQR fences

### Where these appear in your practicals

- **Exp3 (Churn):** 5 duplicate rows removed (205 → 200); missing values imputed; outliers capped with IQR.
- **Exp5 (AQI):** mean AQI 146.62, range 35–289; PM2.5 is the dominant pollutant.
- **Exp6 (Retail):** West is the top region with 33.91% of revenue.

---

## 5. Reinforcement-learning metrics

RL is not scored like classification or regression.

| Metric | Meaning |
|---|---|
| **Return / cumulative reward** | Sum of all rewards collected in one episode |
| **Average return** | Mean return over many episodes; should rise during training |
| **Success rate** | Percentage of episodes that reach the goal |
| **Steps to goal / episode length** | Lower is better once the agent has learned |
| **Exploration rate (epsilon)** | Probability of taking a random action; usually decayed over time |

### Where this appears in your practical

- **Exp6 (Q-Learning maze):** after 5,000 episodes, the agent reaches the goal in **8 steps** and the learned path is valid. A perfect success rate on the training maze is not proof of generalisation to a new maze.

---

## 6. Formula sheet — all in one place

```
Classification
  Accuracy          = (TP + TN) ÷ (TP + TN + FP + FN)
  Error rate        = 1 − Accuracy
  Precision         = TP ÷ (TP + FP)
  Recall            = TP ÷ (TP + FN)
  Specificity       = TN ÷ (TN + FP)
  F1                = 2 × (Precision × Recall) ÷ (Precision + Recall)
  Balanced accuracy = (Recall+ + Recall−) ÷ 2

Regression
  MSE               = average of (actual − predicted)²
  RMSE              = √MSE
  MAE               = average of |actual − predicted|
  R²                = 1 − (unexplained variation ÷ total variation)
  Adjusted R²       = 1 − (1 − R²) × (n − 1) ÷ (n − p − 1)
  MAPE              = average of (|actual − predicted| ÷ |actual|) × 100%

Clustering
  Inertia           = sum of squared distances to own cluster centre
  Silhouette        = (b − a) ÷ max(a, b)

Statistics
  Variance          = average of (value − mean)²
  Standard deviation= √Variance
  Correlation       = covariance ÷ (std_x × std_y)
  IQR               = Q3 − Q1
  Outlier fences    = Q1 − 1.5×IQR  and  Q3 + 1.5×IQR
  Z-score           = (value − mean) ÷ standard deviation
```

---

## 7. Traps examiners love

| Trap | Correct answer |
|---|---|
| "Accuracy is 95%, so the model is good." | On imbalanced data, a model that always predicts the majority class can score 95%. Check precision/recall/F1. |
| "Lower MSE is always better." | Compare on the **same test set**. MSE is in squared units, so it is not directly interpretable. |
| "RMSE and MAE are the same." | RMSE punishes large errors more; MAE is the average absolute error. |
| "R² can never be negative." | It can — a model can be worse than predicting the mean. |
| "F1 uses all four confusion-matrix cells." | F1 uses TP, FP and FN; it ignores TN. |
| "Macro and weighted averages are the same." | Macro treats every class equally; weighted follows class frequency. |
| "Silhouette works for k = 1." | It needs at least two clusters. |
| "High correlation means causation." | Correlation only measures co-movement. |
| "Inertia should be compared across different k." | Inertia always falls as k rises; use the elbow, not the raw value. |
| "A perfect training score means a good model." | It may be overfitting; check the test score. |

---

## 8. Last-minute "say this" lines

- **Classification:** "I report accuracy only if classes are balanced; otherwise precision, recall and F1, with a confusion matrix."
- **Regression:** "I report RMSE for large-error sensitivity, MAE for interpretability, and R² for explained variation."
- **Clustering:** "I choose k with the elbow and confirm with silhouette; inertia alone always improves as k rises."
- **Statistics:** "I use median and IQR when the data is skewed or has outliers, and mean and standard deviation when it is roughly symmetric."
- **RL:** "I report success rate and steps to goal, not accuracy, because the agent learns from reward rather than labels."
