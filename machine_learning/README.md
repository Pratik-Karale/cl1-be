# Machine Learning — 417521

Six runnable practicals. Run each from the repository root:

```bash
python machine_learning/exp1_lda_iris.py
python machine_learning/exp2_uber_regression.py
python machine_learning/exp3_svm_digits.py
python machine_learning/exp4_kmeans_elbow.py
python machine_learning/exp5_ensemble_iris.py
python machine_learning/exp6_qlearning_maze.py
```

| Script | What it demonstrates |
|---|---|
| `exp1_lda_iris.py` | Supervised dimensionality reduction and classification with LDA |
| `exp2_uber_regression.py` | Preprocessing, IQR outliers, correlation, Linear/Ridge/Lasso, R²/RMSE/MAE |
| `exp3_svm_digits.py` | RBF-kernel SVM, multiclass one-vs-one, precision/recall/F1 |
| `exp4_kmeans_elbow.py` | K-Means, inertia/elbow, silhouette score |
| `exp5_ensemble_iris.py` | Hard voting, AdaBoost, Gradient Boosting, optional XGBoost |
| `exp6_qlearning_maze.py` | Q-Learning, epsilon-greedy, Q-table, learned path |

`exp2_uber_regression.py` needs the Uber fares CSV. See [`../data/README.md`](../data/README.md).

All plots are saved to `outputs/`.

## Notes

- `notes/ExpN_*.md` — code, exact outputs, syllabus theory, likely viva Q&A.
- `notes/Plain_English/ExpN_*_PlainEnglish.md` — the same experiment without math or notation.
