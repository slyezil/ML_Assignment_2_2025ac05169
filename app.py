
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Digit Classification Model Lab",
    page_icon="🔢",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"
TARGET_COLUMN = "target"

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.15rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #5c6370;
        margin-bottom: 1.25rem;
    }
    .info-card {
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 0.75rem;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        background: rgba(242, 246, 252, 0.55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metadata() -> dict[str, Any]:
    with (MODEL_DIR / "metadata.json").open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_baseline_metrics() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "metrics.csv")


@st.cache_data
def load_included_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model(model_file: str) -> Any:
    return joblib.load(MODEL_DIR / model_file)


def normalize_target_column(data: pd.DataFrame) -> pd.DataFrame:
    """Accept a few common target aliases while keeping 'target' as the standard."""
    if TARGET_COLUMN in data.columns:
        return data

    aliases = ["Target", "label", "Label", "class", "Class"]
    for alias in aliases:
        if alias in data.columns:
            return data.rename(columns={alias: TARGET_COLUMN})
    return data


def validate_and_prepare_data(
    data: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame | None, pd.Series | None, list[str], list[str]]:
    """Validate an uploaded CSV and return features, optional target, errors, warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    data = normalize_target_column(data.copy())

    if data.empty:
        errors.append("The CSV file contains no rows.")
        return None, None, errors, warnings

    missing_features = [name for name in feature_names if name not in data.columns]
    if missing_features:
        preview = ", ".join(missing_features[:8])
        suffix = "..." if len(missing_features) > 8 else ""
        errors.append(
            f"The CSV is missing {len(missing_features)} required feature columns: "
            f"{preview}{suffix}"
        )
        return None, None, errors, warnings

    X = data.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_cells = int(X.isna().sum().sum())
        errors.append(
            f"The feature data contains {bad_cells} blank or non-numeric values. "
            "Replace them with valid numbers before evaluation."
        )
        return None, None, errors, warnings

    minimum = float(X.min().min())
    maximum = float(X.max().max())
    if minimum < 0 or maximum > 16:
        warnings.append(
            f"Pixel values are normally between 0 and 16, but this file ranges "
            f"from {minimum:.2f} to {maximum:.2f}. Predictions will still be attempted."
        )

    y: pd.Series | None = None
    if TARGET_COLUMN in data.columns:
        numeric_target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
        if numeric_target.isna().any():
            errors.append("The target column contains blank or non-numeric values.")
            return None, None, errors, warnings
        y = numeric_target.astype(int)
    else:
        warnings.append(
            "No target column was found. Predictions can be generated, but evaluation "
            "metrics, confusion matrix, and classification report require true labels."
        )

    extra_columns = [
        column
        for column in data.columns
        if column not in feature_names and column != TARGET_COLUMN
    ]
    if extra_columns:
        warnings.append(
            f"{len(extra_columns)} extra column(s) will be ignored: "
            + ", ".join(extra_columns[:6])
            + ("..." if len(extra_columns) > 6 else "")
        )

    return X, y, errors, warnings


def calculate_live_metrics(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
) -> tuple[dict[str, float], np.ndarray, np.ndarray | None, str | None]:
    """Calculate the assignment metrics on uploaded test data."""
    y_pred = model.predict(X)
    probabilities: np.ndarray | None = None
    auc_value = float("nan")
    auc_note: str | None = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        model_classes = np.asarray(model.classes_)
        present_classes = np.asarray(sorted(pd.unique(y_true)))

        if set(present_classes.tolist()) == set(model_classes.tolist()):
            auc_value = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=model_classes,
                    multi_class="ovr",
                    average="weighted",
                )
            )
        else:
            auc_note = (
                "AUC is shown as N/A because the uploaded target column does not "
                "contain every class used to train the model."
            )

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc_value,
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    return metrics, y_pred, probabilities, auc_note


def metric_text(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.4f}"


metadata = load_metadata()
baseline_metrics = load_baseline_metrics()
feature_names = list(metadata["feature_names"])
model_files = dict(metadata["model_files"])
model_names = list(model_files.keys())

st.markdown('<div class="main-title">🔢 Digit Classification Model Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload test data, choose a model, and compare multiclass classification performance.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Evaluation controls")
    selected_model_name = st.selectbox("Select a classification model", model_names)
    selected_model = load_model(model_files[selected_model_name])

    st.markdown("---")
    st.subheader("Dataset")
    st.write(f"**Instances:** {metadata['instances']:,}")
    st.write(f"**Features:** {metadata['features']}")
    st.write(f"**Classes:** {metadata['class_count']} digits (0-9)")
    st.write(f"**Held-out test rows:** {metadata['test_rows']}")

    st.markdown("---")
    st.caption("The application includes the five classification models named in the assignment.")

summary_tab, evaluation_tab, comparison_tab = st.tabs(
    ["Project summary", "Evaluate test data", "Model comparison"]
)

with summary_tab:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Problem statement")
        st.write(
            "Classify grayscale images of handwritten digits from 0 to 9 using multiple "
            "machine-learning algorithms, then compare their predictive performance using "
            "Accuracy, AUC, Precision, Recall, F1 score, and Matthews Correlation Coefficient."
        )
        st.subheader("Experiment design")
        st.write(
            "The 1,797 rows are split into 80% training data and 20% test data using a "
            "stratified split with random_state=42. Precision, Recall, and F1 use weighted "
            "multiclass averaging. AUC uses weighted one-vs-rest averaging."
        )
    with right:
        st.markdown(
            f"""
            <div class="info-card">
            <strong>Overall best model</strong><br>
            {metadata['winner']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            "Each digit is represented as an 8 × 8 grid. The 64 pixel-intensity features "
            "have values from 0 to 16."
        )

with evaluation_tab:
    st.subheader("1. Choose the test-data source")
    source_choice = st.radio(
        "Data source",
        ["Upload a CSV file", "Use the included test_data.csv"],
        horizontal=True,
        label_visibility="collapsed",
    )

    included_test_data = load_included_test_data()
    st.download_button(
        "Download the required test_data.csv",
        data=TEST_DATA_PATH.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

    active_data: pd.DataFrame | None = None
    if source_choice == "Upload a CSV file":
        uploaded_file = st.file_uploader(
            "Upload test data in CSV format",
            type=["csv"],
            help=(
                "Use the 64 feature columns from test_data.csv. Include a target column "
                "to calculate evaluation metrics."
            ),
        )
        if uploaded_file is not None:
            try:
                active_data = pd.read_csv(uploaded_file)
            except Exception as error:
                st.error(f"The CSV could not be read: {error}")
    else:
        active_data = included_test_data
        st.success("The included held-out test data is ready for evaluation.")

    if active_data is not None:
        active_data = normalize_target_column(active_data)
        X_active, y_active, validation_errors, validation_warnings = validate_and_prepare_data(
            active_data, feature_names
        )

        if y_active is not None:
            allowed_targets = set(int(value) for value in metadata["classes"])
            uploaded_targets = set(int(value) for value in pd.unique(y_active))
            invalid_targets = sorted(uploaded_targets - allowed_targets)
            if invalid_targets:
                validation_errors.append(
                    "The target column contains labels that were not used during training: "
                    + ", ".join(str(value) for value in invalid_targets)
                )

        for warning in validation_warnings:
            st.warning(warning)
        for error in validation_errors:
            st.error(error)

        if not validation_errors and X_active is not None:
            st.subheader("2. Dataset preview")
            st.write(
                f"Rows: **{len(active_data):,}** | Feature columns used: **{len(feature_names)}** "
                f"| Target available: **{'Yes' if y_active is not None else 'No'}**"
            )
            st.dataframe(active_data.head(12), use_container_width=True, hide_index=True)

            with st.expander("Visualize one uploaded digit", expanded=False):
                max_index = max(0, len(X_active) - 1)
                selected_row = st.number_input(
                    "Row number",
                    min_value=0,
                    max_value=max_index,
                    value=0,
                    step=1,
                )
                digit_pixels = X_active.iloc[int(selected_row)].to_numpy().reshape(8, 8)
                figure, axis = plt.subplots(figsize=(3, 3))
                axis.imshow(digit_pixels, cmap="gray_r", vmin=0, vmax=16)
                axis.set_title(f"Uploaded row {int(selected_row)}")
                axis.axis("off")
                st.pyplot(figure, clear_figure=True)

            st.subheader(f"3. Results for {selected_model_name}")

            if y_active is not None:
                metrics, predictions, probabilities, auc_note = calculate_live_metrics(
                    selected_model, X_active, y_active
                )

                metric_columns = st.columns(3)
                for column, metric_name in zip(
                    metric_columns, ["Accuracy", "AUC", "Precision"]
                ):
                    column.metric(metric_name, metric_text(metrics[metric_name]))
                metric_columns = st.columns(3)
                for column, metric_name in zip(metric_columns, ["Recall", "F1", "MCC"]):
                    column.metric(metric_name, metric_text(metrics[metric_name]))
                if auc_note:
                    st.info(auc_note)

                confusion_column, report_column = st.columns([1, 1.15])
                with confusion_column:
                    st.markdown("#### Confusion matrix")
                    figure, axis = plt.subplots(figsize=(7.2, 6.1))
                    ConfusionMatrixDisplay.from_predictions(
                        y_active,
                        predictions,
                        labels=np.asarray(selected_model.classes_),
                        ax=axis,
                        colorbar=False,
                    )
                    axis.set_title(selected_model_name)
                    st.pyplot(figure, clear_figure=True)

                with report_column:
                    st.markdown("#### Classification report")
                    report = classification_report(
                        y_active,
                        predictions,
                        labels=np.asarray(selected_model.classes_),
                        output_dict=True,
                        zero_division=0,
                    )
                    report_frame = pd.DataFrame(report).transpose()
                    st.dataframe(
                        report_frame.style.format(
                            {
                                "precision": "{:.4f}",
                                "recall": "{:.4f}",
                                "f1-score": "{:.4f}",
                                "support": "{:.0f}",
                            }
                        ),
                        use_container_width=True,
                    )

                prediction_frame = active_data.copy()
                prediction_frame["predicted_target"] = predictions
                prediction_frame["correct_prediction"] = (
                    prediction_frame[TARGET_COLUMN].astype(int) == predictions
                )
                if probabilities is not None:
                    prediction_frame["prediction_confidence"] = probabilities.max(axis=1)
            else:
                predictions = selected_model.predict(X_active)
                prediction_frame = active_data.copy()
                prediction_frame["predicted_target"] = predictions
                if hasattr(selected_model, "predict_proba"):
                    prediction_frame["prediction_confidence"] = selected_model.predict_proba(
                        X_active
                    ).max(axis=1)

            st.markdown("#### Prediction preview")
            preview_columns = [
                column
                for column in [
                    TARGET_COLUMN,
                    "predicted_target",
                    "prediction_confidence",
                    "correct_prediction",
                ]
                if column in prediction_frame.columns
            ]
            st.dataframe(
                prediction_frame.loc[:, preview_columns].head(30),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download predictions as CSV",
                data=prediction_frame.to_csv(index=False).encode("utf-8"),
                file_name="model_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "Upload test_data.csv, or select the included test data, to calculate model results."
        )

with comparison_tab:
    st.subheader("Held-out test-set comparison")
    display_frame = baseline_metrics.copy()
    numeric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    st.dataframe(
        display_frame.style.format({column: "{:.4f}" for column in numeric_columns}).highlight_max(
            subset=numeric_columns, axis=0
        ),
        use_container_width=True,
        hide_index=True,
    )

    winner_row = baseline_metrics.sort_values(
        by=["F1", "Accuracy", "MCC", "AUC"], ascending=False
    ).iloc[0]
    st.success(
        f"Overall winner: {winner_row['ML Model Name']} "
        f"(Accuracy {winner_row['Accuracy']:.4f}, "
        f"F1 {winner_row['F1']:.4f}, "
        f"MCC {winner_row['MCC']:.4f})."
    )
