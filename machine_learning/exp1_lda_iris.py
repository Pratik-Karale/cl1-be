"""Experiment 1 - Linear Discriminant Analysis (LDA) on the Iris dataset.

Aim:
    Project the four Iris measurements onto class-separating directions with
    LDA and classify the three flower species.

Dataset:
    scikit-learn built-in Iris dataset (150 samples, 4 features, 3 classes).

Run:
    python exp1_lda_iris.py

Outputs:
    Console: accuracy, confusion matrix, classification report, sample prediction.
    File:    outputs/exp1_lda_confusion_matrix.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )

    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(
        classification_report(
            y_test, y_pred, target_names=iris.target_names, zero_division=0
        )
    )

    sample = [[5.1, 3.5, 1.4, 0.2]]
    print("\nSample measurements:", sample[0])
    print("Predicted species:", iris.target_names[model.predict(sample)[0]])

    display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=iris.target_names
    )
    display.ax_.set_title("LDA - Iris confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp1_lda_confusion_matrix.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
