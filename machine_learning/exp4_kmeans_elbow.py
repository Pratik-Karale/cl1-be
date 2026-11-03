"""Experiment 4 - K-Means clustering on Iris with the elbow method.

Aim:
    Cluster the unlabeled Iris data with K-Means, use within-cluster inertia
    to choose k (elbow method), and evaluate the final clustering with cluster
    counts and the silhouette score.

Dataset:
    scikit-learn built-in Iris dataset (150 samples, 4 features).

Run:
    python exp4_kmeans_elbow.py

Outputs:
    Console: inertia values for k = 1..10, chosen k, cluster counts,
             silhouette score.
    Files:   outputs/exp4_elbow.png, outputs/exp4_clusters.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)

    k_values = range(1, 11)
    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(model.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_values), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow method - Iris")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp4_elbow.png", dpi=150)
    plt.close()

    print("Inertia values:", [round(v, 2) for v in inertias])

    # Iris has three known species; choose k = 3 for the final clustering.
    k = 3
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    print("Chosen k:", k)
    print(
        "Cluster counts:",
        {int(i): int((labels == i).sum()) for i in set(labels)},
    )
    print("Silhouette score:", round(silhouette_score(X, labels), 4))

    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
    plt.xlabel("Scaled sepal length")
    plt.ylabel("Scaled sepal width")
    plt.title("K-Means clusters (first two scaled features)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp4_clusters.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
