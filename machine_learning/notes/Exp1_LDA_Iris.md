# Experiment 1 — Linear Discriminant Analysis on Iris (Viva Theory)

> **30-second answer:** LDA is a **supervised** dimensionality-reduction and classification technique. It projects the 4 Iris features onto directions that maximise **between-class separation** while minimising **within-class scatter**, then classifies the flower species. Test accuracy = **1.00** (38/38 correct).

## 1. MOST LIKELY — Core of the Experiment (High Yield)

### Plain-English: what LDA actually does

Imagine plotting all 150 flowers using their 4 measurements. The 3 species form 3 groups, but the groups overlap when you look at any single measurement. LDA asks one question:

> **"Which new direction makes the 3 groups sit as far apart as possible, while each group stays as tightly packed as possible?"**

It then:

1. **Finds those directions** (new axes, built from the original 4 measurements).
2. **Projects every flower onto them** — 4 measurements become at most 2 new numbers.
3. **Classifies** a new flower by which group it lands closest to in that new space.

Because it uses the species labels while finding the directions, it is **supervised**. That is the whole idea — no math needed to say it in a viva.

**Analogy:** PCA asks *"which direction shows the most spread in the crowd?"* LDA asks *"which direction best separates the groups in the crowd?"* PCA ignores labels; LDA uses them.

**Aim:** Apply LDA on the Iris dataset to find class-separating projections, classify flower species, and evaluate with accuracy, confusion matrix and classification report.

**Dataset:** `load_iris()` — **150 samples × 4 features** (sepal length, sepal width, petal length, petal width), **3 classes** (setosa, versicolor, virginica), 50 each.

**Algorithm in 5 bullets**
- Compute the **class means**, the **within-class scatter** (S_W: how spread each species is around its own mean), and the **between-class scatter** (S_B: how far the species means are from each other).
- Find the new directions **w** that make this ratio as large as possible: **between-class scatter ÷ within-class scatter**. (The textbook writes this as `S_W⁻¹ × S_B × w = λ × w`, but that is the same sentence.)
- Keep the top directions — at most **classes − 1 = 2** discriminant components.
- Project the data onto those directions (this maximises the Fisher score `J = between-class scatter ÷ within-class scatter`).
- Classify a new point by the nearest class mean / decision boundary in the projected space.

**Math words decoded (if the examiner uses them)**

| Term | Plain meaning | Which way do we want it? |
|---|---|---|
| Class mean | The average flower of each species | — |
| Within-class scatter (S_W) | How spread out each species is around its own average | **Small** (tight groups) |
| Between-class scatter (S_B) | How far apart the species averages are | **Large** (separated groups) |
| Eigenvector (w) | A new direction/axis made from the original features | — |
| Eigenvalue (λ) | How good that direction is at separating classes | **Large** |
| Fisher criterion (J) | The ratio `between ÷ within` | **Large** |

So the scary formula `S_W⁻¹ × S_B × w = λ × w` is just: *"find the directions where between-class separation is high compared with within-class spread."* You can say that sentence in the viva instead of reading the formula.

**Code skeleton**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target)
model = LinearDiscriminantAnalysis()      # supervised; learns class structure
model.fit(X_train, y_train)
y_pred = model.predict(X_test)            # class labels 0/1/2
print(accuracy_score(y_test, y_pred))     # 1.0
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Expected output**
- Test accuracy: **1.0000**
- Confusion matrix: diagonal `[[12,0,0],[0,13,0],[0,0,13]]` — **38 test samples**, no errors.
- Precision / recall / F1 = **1.00** for all three classes.
- Sample `[5.1, 3.5, 1.4, 0.2]` → predicted **setosa**.

**Viva Q&A**
1. **Why LDA and not PCA here?** LDA is supervised — it uses class labels to maximise separation; PCA is unsupervised and only maximises variance.
2. **What does `n_components` mean in LDA?** Maximum useful components = `n_classes − 1` (here 2); it cannot exceed that because between-class scatter has rank ≤ k−1.
3. **Why `stratify=iris.target`?** Keeps the 50/50/50 class ratio in both train and test splits.
4. **What assumption does LDA make?** Features are Gaussian and all classes share the same covariance matrix.
5. **What is the Fisher criterion?** Ratio of between-class to within-class scatter; LDA maximises it.
6. **Can LDA reduce to 3 components for 3 classes?** No — at most `k−1 = 2`.
7. **Why is accuracy 1.0?** Iris is a small, well-separated dataset; setosa is linearly separable and the other two are nearly so.
8. **What if classes had unequal covariance?** Use **QDA** (Quadratic Discriminant Analysis) instead.
9. **Does LDA need feature scaling?** Not strictly (it is affine-invariant), but scaling is good practice.
10. **What does the confusion matrix show?** Rows = actual class, columns = predicted class; diagonal = correct predictions.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit I: Feature Transformation
- **Dimensionality reduction:** reduce feature count while retaining useful information — faster models, less overfitting, easier visualisation.
- **PCA vs LDA:**

| Point | PCA | LDA |
|---|---|---|
| Supervision | Unsupervised | Supervised |
| Uses labels | No | Yes |
| Question it asks | "Which direction shows the most **spread**?" | "Which direction best **separates the groups**?" |
| Objective | Maximise variance | Maximise class separation |
| Math goal | Keep the largest-variance axes | Maximise `between ÷ within` scatter |
| Max components | `min(n_samples, n_features)` | `n_classes − 1` |
| Classifies by itself? | No — it only transforms | Yes — it can predict a class |
| Best for | Compression / visualisation | Classification |

**How they are related:** both are **linear projections** — they build new axes as combinations of the old features, and both can reduce dimensions. They differ only in *what they are trying to preserve*: PCA preserves **spread**, LDA preserves **separation between labelled groups**. PCA can be run before LDA, but PCA alone never looks at `y`.

- **Supervised vs unsupervised:** supervised has labelled `y`; unsupervised finds structure without labels.
- **Parametric vs non-parametric:** LDA is parametric (assumes a Gaussian form); KNN is non-parametric.

**Likely Q&A**
- *What is the curse of dimensionality?* As dimensions grow, data becomes sparse and distance-based learning degrades.
- *Is LDA a classifier or a reducer?* Both — it projects and can classify in the projected space.
- *Difference between LDA and logistic regression?* LDA models class-conditional Gaussians; logistic regression directly models "probability of the class given the features" and makes fewer distribution assumptions.
- *What is the role of eigenvalues?* They rank discriminant directions by separability; the largest eigenvalue is the most discriminative.
- *When would you prefer PCA?* When labels are unavailable or the task is compression/denoising.

## 3. LEAST LIKELY — Stretch / Edge Cases

- **Limitation:** LDA fails when classes are not linearly separable or covariance differs strongly.
- **Small-sample problem:** the within-class scatter (S_W) cannot be inverted reliably when features ≫ samples; use shrinkage LDA (`solver='lsqr'`, `shrinkage='auto'`).
- **Outliers:** class means and covariance are sensitive to outliers — clean/scale first.
- **Multicollinearity:** correlated features make the within-class scatter (S_W) unstable to invert.
- **Alternatives:** QDA, regularised LDA, kernel LDA, or SVM for pure classification.
- **Cross-unit link:** the same projection idea reappears in Unit III (linear classifiers) and Unit IV (cluster visualisation).
- **Trap question:** "Does LDA always improve accuracy?" No — it reduces dimensions and can discard useful variance if classes overlap.

**If you remember only one thing:** LDA is supervised Fisher projection — maximise between-class scatter over within-class scatter, giving at most `classes − 1` components.
