# Experiment 5 — Ensemble Learning on Iris: Voting, AdaBoost, Gradient Boosting and XGBoost (Viva Theory)

> **30-second answer:** Combine multiple weak classifiers into stronger ensembles — **hard Voting, AdaBoost, Gradient Boosting and XGBoost** — and compare them on Iris. **Gradient Boosting wins with 97.37% accuracy**; AdaBoost and XGBoost reach 92.11%, hard Voting 89.47%.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

**Aim:** Train ensemble classifiers (hard Voting, AdaBoost, Gradient Boosting, XGBoost) on Iris and compare accuracy, macro precision, macro recall and macro F1.

**Dataset:** Iris — **150 samples × 4 features**, **3 classes**, 50 each. Split 75/25 stratified → **38 test samples**.

**Algorithm in 5 bullets**
- **Voting (hard):** majority vote of three different models — Logistic Regression, Decision Tree (`max_depth=3`), Random Forest (`n_estimators=150`).
- **AdaBoost:** sequential boosting; each new stump focuses on previously misclassified points; `n_estimators=100`.
- **Gradient Boosting:** fits each new tree to the **negative gradient of the loss** (pseudo-residuals); additive model.
- **XGBoost:** regularised gradient boosting with `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `objective='multi:softprob'`.
- Compare with **accuracy, macro precision, macro recall, macro F1** and per-class classification reports.

**Code skeleton**
```python
models = {
  "Voting (hard)": VotingClassifier(estimators=[
      ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
      ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
      ("rf", RandomForestClassifier(n_estimators=150, random_state=42))],
      voting="hard"),
  "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
  "Gradient Boosting": GradientBoostingClassifier(random_state=42),
  "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                           objective="multi:softprob", num_class=3,
                           eval_metric="mlogloss", random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # score with accuracy, macro precision/recall/F1
```

**Expected output**

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| **Gradient Boosting** | **0.9737** | 0.9762 | 0.9744 | 0.9743 |
| AdaBoost | 0.9211 | 0.9246 | 0.9231 | 0.9230 |
| XGBoost | 0.9211 | 0.9246 | 0.9231 | 0.9230 |
| Voting (hard) | 0.8947 | 0.9030 | 0.8974 | 0.8968 |

**Viva Q&A**
1. **Why ensemble learning?** Combining models reduces variance/bias and usually beats any single weak learner.
2. **What is a weak learner?** A model only slightly better than random (e.g., a shallow decision stump).
3. **Hard vs soft voting?** Hard = majority of predicted labels; soft = average predicted probabilities (needs `predict_proba`).
4. **Bagging vs boosting?** Bagging trains models **independently** on bootstrap samples and averages; boosting trains **sequentially**, each model fixing prior errors.
5. **How does AdaBoost work?** Increase weights of misclassified samples, add a new weighted weak learner, combine with weights based on accuracy.
6. **How does Gradient Boosting work?** Each tree fits the residual/negative gradient of the loss; learning rate shrinks each contribution.
7. **What makes XGBoost different?** Regularised objective (L1/L2), parallel tree building, handling of missing values, and column subsampling.
8. **Why did Gradient Boosting win here?** Iris is small and clean; boosting quickly fits the near-linear boundaries.
9. **What is Random Forest?** Bagging of decision trees plus random feature subsets; reduces variance.
10. **What is stacking?** Train a meta-model on the predictions of several base models.
11. **Why `average='macro'`?** It treats all classes equally regardless of support — important for multiclass comparison.
12. **What if classes were imbalanced?** Use `class_weight='balanced'`, stratified splits, or resampling.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit V: Ensemble Learning
- **Homogeneous ensemble:** same algorithm, different samples/parameters (e.g., Random Forest).
- **Heterogeneous ensemble:** different algorithms combined (e.g., Voting).
- **Voting types:** max voting (classification), averaging (regression), weighted average.
- **Bagging:** **bootstrap** sampling + **aggregate** (average/vote) → variance reduction.
- **Boosting:** sequential; reduce **bias**; AdaBoost, Gradient Boosting, XGBoost.
- **Stacking:** base learners + meta-learner; can use blending (hold-out predictions).
- **Random Forest:** many decorrelated trees; feature randomness + bagging.

| Technique | Training | Main effect | Example |
|---|---|---|---|
| Bagging | parallel | ↓ variance | Random Forest |
| Boosting | sequential | ↓ bias | AdaBoost, XGBoost |
| Voting | parallel | combines diverse models | hard/soft voting |
| Stacking | two-stage | learns optimal combination | meta-classifier |

**Likely Q&A**
- *Advantages of ensembles?* Higher accuracy, robustness, reduced overfitting. *Limitations?* More compute, harder to interpret.
- *What is bootstrapping?* Sampling with replacement to create varied training subsets.
- *Why does Random Forest reduce variance?* Averaging many decorrelated trees cancels individual errors.
- *What is the learning rate in boosting?* Shrinkage applied to each tree's contribution; lower = more trees needed but better generalisation.
- *Applications of ensemble learning?* Fraud detection, medical diagnosis, credit scoring, competitions (Kaggle).

## 3. LEAST LIKELY — Stretch / Edge Cases

- **Overfitting in boosting:** too many estimators / too high learning rate; use early stopping.
- **XGBoost dependency:** may be unavailable; the code catches the import error and skips it.
- **Hard voting limitation:** ignores confidence; soft voting often performs better.
- **Correlated base models:** ensembles gain little if all models make the same errors — diversity matters.
- **Interpretability:** ensemble feature importance (RF/XGBoost) is approximate; use SHAP for detail.
- **Cross-unit link:** ensembles combine the classifiers from Unit III and can use clustering outputs (Unit IV) as features.
- **Trap question:** "Is the most complex ensemble always best?" No — on tiny datasets, simpler models can generalise better.

**If you remember only one thing:** Bagging averages independent models to cut variance, boosting builds models sequentially to cut bias — Gradient Boosting gave the best 97.37% here.
