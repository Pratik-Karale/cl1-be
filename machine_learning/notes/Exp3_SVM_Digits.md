# Experiment 3 — SVM with RBF Kernel for Handwritten Digit Classification (Viva Theory)

> **30-second answer:** Classify 8×8 handwritten digit images (0–9) from scikit-learn's `digits` dataset using a **Support Vector Machine with an RBF kernel**. The pipeline scales features then fits `SVC(C=5, gamma='scale')`, achieving **98% test accuracy** on 450 test images.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

**Aim:** Implement an SVM with an RBF kernel to classify handwritten digits (0–9) from the 8×8 scikit-learn digits dataset, and evaluate with accuracy, confusion matrix and classification report.

**Dataset:** `load_digits()` — **1,797 samples**, each an **8×8 image = 64 pixel features**, **10 classes** (digits 0–9). `digits.data` is flattened to 64 columns; `digits.images` retains 8×8 shape.

**Algorithm in 5 bullets**
- Split with `test_size=0.25`, `random_state=42`, `stratify=digits.target` → **450 test samples**.
- **StandardScaler** puts all 64 pixel features on the same scale (SVM is distance-based).
- **RBF kernel** measures how similar two points are: `similarity = exp(−γ × squared distance between the points)`. Nearby points get high similarity; far points get near zero.
- **C = 5** controls the soft-margin trade-off; **gamma = 'scale'** = `1 ÷ (number of features × variance of X)`.
- Multiclass is handled internally by **one-vs-one** (45 binary classifiers for 10 classes).

**Code skeleton**
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=42, stratify=digits.target)

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale"))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test, pred))        # 0.98
print(confusion_matrix(y_test, pred))      # 10×10
print(classification_report(y_test, pred))
```

**Expected output**
- Accuracy: **0.98** (≈441/450 correct).
- Confusion matrix: **10×10**, mostly diagonal; small confusions like 8→1 (2 cases), 5→9, 9→6/7.
- Per-class F1 mostly **0.97–1.00**; class **8 recall = 0.93** is the weakest; macro avg ≈ **0.98**.

**Viva Q&A**
1. **Why SVM for images?** SVM handles high-dimensional feature spaces well (64 features) and the RBF kernel captures nonlinear boundaries.
2. **What is the margin?** The distance between the decision boundary and the nearest training points (support vectors); SVM maximises it.
3. **What are support vectors?** Training samples closest to the boundary; only they determine the classifier.
4. **What does C do?** Large C = little tolerance for misclassification (narrow margin, overfit risk); small C = wider margin, more regularisation.
5. **What does gamma do?** Controls RBF reach: high gamma = each point influences only nearby points (overfit); low gamma = smoother, broader boundary.
6. **Why `gamma='scale'`?** Automatically sets `1 ÷ (number of features × variance of X)` — adapts to feature scale.
7. **Why scale before SVM?** The RBF distance is scale-sensitive; unscaled pixels would dominate.
8. **How does SVM do 10-class problems?** scikit-learn's `SVC` uses **one-vs-one** (45 pairwise classifiers).
9. **Why is class 8 harder?** Handwritten 8s resemble 1s/9s/0s in low-resolution 8×8 images.
10. **What if you used a linear kernel?** Faster but likely lower accuracy — digit boundaries are nonlinear.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit III: Classification
- **Binary vs multiclass:** binary = 2 classes; multiclass = >2 (here 10).
- **One-vs-One (OvO):** train one classifier per pair; predict by majority vote. **One-vs-All (OvA):** one classifier per class vs the rest.
- **Linear SVM:** finds a straight boundary `(weights · features) + bias = 0`, and maximises the margin, which equals `2 ÷ length of the weight vector`.
- **Soft margin:** allows a few mistakes ("slack") so the boundary is not forced to fit every point perfectly. Objective = minimise `(½ × length of weights²) + C × (total allowed mistakes)`.
- **Kernel trick:** computes dot products in a higher-dimensional space without actually mapping the points there.

| Kernel | In plain words | Use |
|---|---|---|
| Linear | dot product of the two points | linearly separable |
| RBF/Gaussian | `exp(−γ × squared distance between points)` | general nonlinear |
| Polynomial | `(dot product + c) raised to power d` | polynomial boundaries |
| Sigmoid | `tanh(α × dot product + c)` | neural-network-like |

- **Metrics:** Accuracy = (TP+TN)/total; Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = `2 × (precision × recall) ÷ (precision + recall)`.
- **Confusion matrix:** rows = actual, columns = predicted.

**Likely Q&A**
- *Why is accuracy misleading on imbalanced data?* A majority-class predictor can score high while ignoring the minority class.
- *Precision vs recall?* Precision = how many predicted positives are correct; recall = how many actual positives were found.
- *What is the kernel trick?* Implicitly mapping to high dimensions via a kernel function, avoiding expensive explicit computation.
- *What is a linear classifier?* A decision boundary that is linear in the input features.
- *What is a balanced vs imbalanced classification problem?* Class frequencies are roughly equal vs heavily skewed.
- *What is KNN?* Classifies by majority label of the k nearest training points; non-parametric, no training phase.

## 3. LEAST LIKELY — Stretch / Edge Cases

- **Training cost:** SVM training grows roughly with the square-to-cube of the number of samples; large datasets need `LinearSVC` or SGD.
- **Parameter tuning:** `C` and `gamma` are best chosen with grid search + cross-validation.
- **Feature scaling leakage:** fit the scaler on train only, then transform test.
- **Probability output:** `SVC` needs `probability=True` to expose `predict_proba` (slower).
- **Class imbalance:** use `class_weight='balanced'` or resampling.
- **Cross-unit link:** ensemble SVMs (Unit V) or clustering of digit embeddings (Unit IV).
- **Trap question:** "Is a higher C always better?" No — it can overfit; tune it.

**If you remember only one thing:** SVM maximises the margin between classes; the RBF kernel handles nonlinearity, and C/gamma control the fit–generalisation trade-off.
