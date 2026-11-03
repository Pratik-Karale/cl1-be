"""Experiment 5 - Ensemble learning on Iris.

Aim:
    Compare hard Voting, AdaBoost, Gradient Boosting and (if installed)
    XGBoost on the Iris dataset using accuracy and macro precision/recall/F1.

Dataset:
    scikit-learn built-in Iris dataset (150 samples, 4 features, 3 classes).

Run:
    python exp5_ensemble_iris.py

Note:
    XGBoost is optional. If it is not installed, the script prints a note and
    compares the remaining three models.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_models() -> dict:
    models = {
        "Voting (hard)": VotingClassifier(
            estimators=[
                (
                    "lr",
                    make_pipeline(
                        StandardScaler(), LogisticRegression(max_iter=2000)
                    ),
                ),
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
                (
                    "rf",
                    RandomForestClassifier(n_estimators=150, random_state=42),
                ),
            ],
            voting="hard",
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
    except Exception as exc:  # pragma: no cover - depends on environment
        print("XGBoost unavailable; skipped:", exc)

    return models


def main() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Macro precision": precision_score(
                    y_test, pred, average="macro", zero_division=0
                ),
                "Macro recall": recall_score(
                    y_test, pred, average="macro", zero_division=0
                ),
                "Macro F1": f1_score(
                    y_test, pred, average="macro", zero_division=0
                ),
            }
        )
        print(f"\n{name}")
        print(
            classification_report(
                y_test,
                pred,
                target_names=iris.target_names,
                zero_division=0,
            )
        )

    results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    print("\nModel comparison:")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
