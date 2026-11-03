"""Experiment 3 - Support Vector Machine (RBF kernel) for handwritten digits.

Aim:
    Classify 8x8 handwritten digit images (0-9) with an RBF-kernel SVM and
    evaluate with accuracy, a confusion matrix and a classification report.

Dataset:
    scikit-learn built-in digits dataset (1,797 samples, 64 pixel features).

Run:
    python exp3_svm_digits.py

Outputs:
    Console: accuracy, 10x10 confusion matrix, classification report.
    Files:   outputs/exp3_svm_confusion_matrix.png,
             outputs/exp3_digit_samples.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.25,
        random_state=42,
        stratify=digits.target,
    )

    model = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale")
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))

    display = ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
    display.ax_.set_title("SVM - handwritten digits")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp3_svm_confusion_matrix.png", dpi=150)
    plt.close()

    # Recreate the same split on row indices to display ten held-out images.
    _, test_idx, _, _ = train_test_split(
        range(len(digits.data)),
        digits.target,
        test_size=0.25,
        random_state=42,
        stratify=digits.target,
    )

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, idx in zip(axes.ravel(), test_idx[:10]):
        ax.imshow(digits.images[idx], cmap="gray")
        ax.set_title(f"True: {digits.target[idx]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp3_digit_samples.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
