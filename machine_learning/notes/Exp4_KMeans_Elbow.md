# Experiment 4 — K-Means Clustering on Iris with the Elbow Method (Viva Theory)

> **30-second answer:** Group the unlabeled Iris data into clusters with **K-Means**, using **inertia** to find the elbow and choosing **k = 3**, then evaluate with **silhouette score = 0.4599**. Cluster sizes: **53, 50, 47**.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

**Aim:** Implement K-Means on Iris, use the elbow method on within-cluster inertia to choose *k*, and evaluate with cluster counts and silhouette score.

**Dataset:** Iris — **150 samples × 4 features**, no labels used during clustering. True species count = 3 (only used to interpret/choose k).

**Algorithm in 5 bullets**
- **Scale** features with `StandardScaler` (K-Means uses Euclidean distance).
- Initialise **k centroids** (default k-means++ in scikit-learn).
- **Assign** each point to the nearest centroid; **update** each centroid to the mean of its points; repeat until convergence.
- Compute **inertia** (within-cluster sum of squares) for k = 1…10.
- Pick k at the **elbow**; final run reports cluster sizes and **silhouette score**.

**Code skeleton**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = StandardScaler().fit_transform(iris.data)
inertias = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(model.inertia_)          # WCSS for this k

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)                # cluster ids, not species
print({int(i): int((labels == i).sum()) for i in set(labels)})
print(silhouette_score(X, labels))           # 0.4599
```

**Expected output**
- Inertia: **[600.0, 222.36, 139.82, 114.09, 90.93, 81.54, 72.63, 62.54, 55.12, 47.39]** for k = 1…10.
- Elbow visible around **k = 3**; chosen **k = 3**.
- Cluster counts: **{0: 53, 1: 50, 2: 47}**.
- Silhouette score: **0.4599**.

**Viva Q&A**
1. **Why K-Means?** Simple, fast, and works well for compact, roughly spherical clusters like Iris.
2. **Why scale first?** Distance-based algorithm — a large-range feature would dominate.
3. **What is inertia?** Sum of squared distances of each point to its centroid (WCSS); lower is tighter.
4. **Why use the elbow method?** Inertia always falls as k rises; the elbow is where extra clusters stop giving meaningful improvement.
5. **What does the silhouette score mean?** Values near 1 = well-separated clusters; near 0 = overlapping; negative = likely wrong assignment. 0.46 ≈ moderate structure.
6. **Why is k = 3 chosen?** The elbow flattens after 3 and Iris has 3 species; the data itself is unlabeled.
7. **What is `n_init=10`?** Runs 10 independent initialisations and keeps the best inertia — guards against bad random seeds.
8. **What does `random_state` do?** Makes centroid initialisation reproducible.
9. **Are clusters the same as species?** No — K-Means is unsupervised; clusters may or may not align with true labels.
10. **What if k is too large?** Over-segmentation; inertia keeps dropping but clusters lose meaning.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit IV: Clustering
- **Clustering:** grouping unlabeled data so points within a group are more similar than across groups. **No labels are used.**
- **Types:**
  - **Centroid/partitioning:** K-Means, K-Medoids, K-Prototype.
  - **Hierarchical/connectivity:** Agglomerative (bottom-up), Divisive (top-down).
  - **Density:** DBSCAN.
  - **Distribution:** Gaussian Mixture Model (GMM).
- **K-Means steps:** choose k → initialise centroids → assign → update → repeat.
- **Inertia (WCSS):** for every point, measure its squared distance from its own cluster centre; add all those up. Lower = tighter clusters.
- **Silhouette:** `s = (b − a) ÷ max(a, b)`, where `a` = average distance to points in the same cluster and `b` = average distance to the nearest other cluster. Range −1…1; closer to 1 = better separated.
- **K-Medoids:** uses actual data points as centres (robust to outliers).
- **DBSCAN:** density-based; finds arbitrary shapes and labels noise; needs `eps` and `min_samples`.
- **GMM:** soft, probabilistic assignment using Gaussian components (EM algorithm).

| Method | Cluster shape | Needs k? | Handles noise? |
|---|---|---|---|
| K-Means | spherical | yes | no |
| K-Medoids | spherical-ish | yes | more robust |
| DBSCAN | arbitrary | no | yes |
| GMM | elliptical | yes (components) | probabilistic |

**Likely Q&A**
- *Advantages of K-Means?* Fast, simple, scalable; *disadvantages?* Must choose k, sensitive to initialisation/outliers, assumes spherical clusters.
- *Elbow vs silhouette?* Elbow uses inertia; silhouette measures separation quality and can validate k.
- *What are applications of clustering?* Market segmentation, social-network analysis, image segmentation, anomaly detection.
- *Difference between clustering and classification?* Clustering is unsupervised (no labels); classification is supervised.
- *What is hierarchical clustering?* Builds a dendrogram by merging (AHC) or splitting (DHC) clusters.

## 3. LEAST LIKELY — Stretch / Edge Cases

- **Initialisation sensitivity:** k-means++ and `n_init` mitigate local minima.
- **Outliers:** can drag centroids; use K-Medoids or remove outliers.
- **Non-spherical clusters:** K-Means fails; use DBSCAN or GMM.
- **Choosing k objectively:** silhouette, Calinski–Harabasz, Davies–Bouldin; domain knowledge.
- **Feature scaling leakage:** fit scaler on training data in supervised pipelines.
- **Cross-unit link:** cluster labels can become features for classification (Unit III) or segments for ensembling (Unit V).
- **Trap question:** "Does a lower inertia always mean better clustering?" No — inertia falls monotonically with k; it must be balanced against interpretability.

**If you remember only one thing:** K-Means alternates assign/update to minimise inertia; the elbow picks k and silhouette (0.4599 here) measures how good the resulting clusters are.
