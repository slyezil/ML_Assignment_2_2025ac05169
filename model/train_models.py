
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "target"

MODEL_FILES = OrderedDict(
    {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "kNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Random Forest (Ensemble)": "random_forest.joblib",
    }
)


def get_project_root() -> Path:
    """Return the repository root regardless of the current working directory."""
    return Path(__file__).resolve().parent.parent


def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load the digits data and return features, target, and dataset metadata."""
    digits = load_digits()
    feature_names = [
        f"pixel_r{row}_c{column}" for row in range(8) for column in range(8)
    ]

    X = pd.DataFrame(digits.data, columns=feature_names, dtype=float)
    y = pd.Series(digits.target.astype(int), name=TARGET_COLUMN)

    dataset_info = {
        "name": "Optical Recognition of Handwritten Digits",
        "implementation_source": "scikit-learn load_digits",
        "repository_source": "UCI Machine Learning Repository",
        "instances": int(X.shape[0]),
        "features": int(X.shape[1]),
        "classes": [int(value) for value in sorted(y.unique())],
        "class_count": int(y.nunique()),
        "image_shape": [8, 8],
        "feature_value_range": [float(X.min().min()), float(X.max().max())],
        "target_column": TARGET_COLUMN,
        "feature_names": feature_names,
    }
    return X, y, dataset_info


def build_models() -> OrderedDict[str, Any]:

    return OrderedDict(
        {
            "Logistic Regression": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=3000,
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                min_samples_leaf=1,
            ),
            "kNN": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        KNeighborsClassifier(
                            n_neighbors=5,
                            weights="distance",
                        ),
                    ),
                ]
            ),
            "Naive Bayes": Pipeline(
                steps=[
                    ("scaler", MinMaxScaler()),
                    ("classifier", MultinomialNB(alpha=0.1)),
                ]
            ),
            "Random Forest (Ensemble)": RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }
    )


def calculate_metrics(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)
    model_classes = np.asarray(model.classes_)

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(
            roc_auc_score(
                y_test,
                y_probability,
                labels=model_classes,
                multi_class="ovr",
                average="weighted",
            )
        ),
        "Precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
    }


def main() -> None:

    project_root = get_project_root()
    model_directory = project_root / "model"
    model_directory.mkdir(parents=True, exist_ok=True)

    X, y, dataset_info = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_data = X_test.reset_index(drop=True).copy()
    test_data[TARGET_COLUMN] = y_test.reset_index(drop=True)
    test_data.to_csv(project_root / "test_data.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    models = build_models()

    print("Training and evaluating five models...\n")
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = calculate_metrics(model, X_test, y_test)

        model_path = model_directory / MODEL_FILES[model_name]
        joblib.dump(model, model_path, compress=3)

        metric_rows.append({"ML Model Name": model_name, **metrics})
        print(
            f"{model_name:31s} "
            f"Accuracy={metrics['Accuracy']:.4f}  "
            f"AUC={metrics['AUC']:.4f}  "
            f"F1={metrics['F1']:.4f}  "
            f"MCC={metrics['MCC']:.4f}"
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(model_directory / "metrics.csv", index=False)

    winner = metrics_frame.sort_values(
        by=["F1", "Accuracy", "MCC", "AUC"], ascending=False
    ).iloc[0]["ML Model Name"]

    metadata = {
        **dataset_info,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "model_files": MODEL_FILES,
        "winner": winner,
        "metric_average": "weighted",
        "auc_method": "one-vs-rest, weighted",
    }
    with (model_directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nSaved artifacts:")
    print(f"- {project_root / 'test_data.csv'}")
    print(f"- {model_directory / 'metrics.csv'}")
    print(f"- {model_directory / 'metadata.json'}")
    print(f"- {len(models)} serialized model files in {model_directory}")
    print(f"\nBest model: {winner}")


if __name__ == "__main__":
    main()
