from __future__ import annotations

import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.error_analysis import (
    build_failure_insights,
    build_prediction_frame,
    error_rate_by_feature,
    misclassified_samples,
)
from utils.evaluation import classification_report_df, compute_metrics, confusion_matrix_figure
from utils.explainability import (
    compute_shap_explanation,
    get_global_importance,
    plot_feature_importance,
    plot_local_shap_bars,
    plot_shap_summary,
)
from utils.preprocessing import (
    TrainingArtifacts,
    get_available_model_names,
    get_builtin_dataset_default_target,
    get_builtin_dataset_names,
    load_builtin_dataset,
    train_model,
)


APP_TITLE = "ML Failure Analyzer + Explainable AI Dashboard"
APP_SUBTITLE = "Train a model, inspect failures, and explain predictions with a clean dashboard."


st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")


def inject_custom_styles():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            .app-header {
                padding: 1.25rem 1.5rem;
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.94));
                color: white;
                margin-bottom: 1rem;
                border: 1px solid rgba(148,163,184,0.18);
            }
            .app-header h1 {
                margin-bottom: 0.25rem;
                font-size: 2.1rem;
            }
            .app-header p {
                margin: 0;
                opacity: 0.86;
                font-size: 1rem;
            }
            div[data-testid="metric-container"] {
                background: #ffffff;
                border: 1px solid rgba(148,163,184,0.25);
                padding: 0.75rem 0.85rem;
                border-radius: 16px;
                box-shadow: 0 10px 24px rgba(15,23,42,0.05);
            }
            .small-note {
                color: #64748b;
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset_from_source(source: str, uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    if source in get_builtin_dataset_names():
        return load_builtin_dataset(source)
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes))
    return pd.DataFrame()


def render_metric_cards(metrics: dict, is_binary: bool):
    columns = st.columns(5 if is_binary and np.isfinite(metrics.get("roc_auc", np.nan)) else 4)
    card_specs = [
        ("Accuracy", metrics.get("accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1-score", metrics.get("f1")),
    ]
    if is_binary and np.isfinite(metrics.get("roc_auc", np.nan)):
        card_specs.append(("ROC-AUC", metrics.get("roc_auc")))

    for col, (label, value) in zip(columns, card_specs):
        col.metric(label, f"{value:.3f}" if value is not None and np.isfinite(value) else "N/A")


def render_confidence_comparison(prediction_frame: pd.DataFrame):
    if "prediction_confidence" not in prediction_frame.columns:
        st.info("Prediction confidence is unavailable for this model.")
        return

    correct_conf = prediction_frame.loc[prediction_frame["correct"], "prediction_confidence"].dropna()
    wrong_conf = prediction_frame.loc[~prediction_frame["correct"], "prediction_confidence"].dropna()

    if correct_conf.empty or wrong_conf.empty:
        st.info("Confidence comparison requires both correct and incorrect predictions.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Average confidence when correct", f"{correct_conf.mean():.3f}")
    c2.metric("Average confidence when wrong", f"{wrong_conf.mean():.3f}")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(["Correct", "Wrong"], [correct_conf.mean(), wrong_conf.mean()], color=["#10b981", "#ef4444"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average confidence")
    ax.set_title("Confidence Comparison")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_feature_failure_table(feature_name: str, table: pd.DataFrame):
    if table.empty:
        st.info("No failure pattern data available for this feature.")
        return

    st.dataframe(table, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8.5, max(3.8, len(table) * 0.35)))
    value_col = "bin" if "bin" in table.columns else feature_name
    top_table = table.head(10).copy()
    labels = top_table[value_col].astype(str).tolist()
    ax.barh(labels[::-1], top_table["error_rate"].tolist()[::-1], color="#f97316")
    ax.set_xlabel("Error rate")
    ax.set_title(f"Error rate by {feature_name}")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


@st.cache_data(show_spinner=False)
def cached_model_comparison(df: pd.DataFrame, target_col: str, test_size: float, model_names: tuple[str, ...], exclude_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    rows = []
    for name in model_names:
        artifacts = train_model(
            df=df,
            target_col=target_col,
            model_name=name,
            test_size=test_size,
            exclude_columns=list(exclude_columns) if exclude_columns else None,
        )
        metrics = compute_metrics(artifacts.y_test, artifacts.y_pred, artifacts.y_proba, artifacts.is_binary)
        rows.append({
            "model": name,
            "accuracy": metrics.get("accuracy", np.nan),
            "precision": metrics.get("precision", np.nan),
            "recall": metrics.get("recall", np.nan),
            "f1": metrics.get("f1", np.nan),
            "roc_auc": metrics.get("roc_auc", np.nan),
        })
    return pd.DataFrame(rows)


def save_model_bytes(artifacts: TrainingArtifacts) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(artifacts.pipeline, buffer)
    return buffer.getvalue()


def save_model_to_outputs(artifacts: TrainingArtifacts) -> Path:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "trained_model.pkl"
    joblib.dump(artifacts.pipeline, model_path)
    return model_path


inject_custom_styles()

st.markdown(
    f"""
    <div class="app-header">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("A beginner-friendly ML dashboard with advanced failure analysis, explainability, and export tools.")


if "artifacts" not in st.session_state:
    st.session_state.artifacts = None
if "prediction_frame" not in st.session_state:
    st.session_state.prediction_frame = None
if "feature_error_tables" not in st.session_state:
    st.session_state.feature_error_tables = None
if "failure_insights" not in st.session_state:
    st.session_state.failure_insights = None
if "shap_values" not in st.session_state:
    st.session_state.shap_values = None
if "shap_feature_matrix" not in st.session_state:
    st.session_state.shap_feature_matrix = None
if "shap_message" not in st.session_state:
    st.session_state.shap_message = None


tabs = st.tabs(
    [
        "Dataset",
        "Model Training",
        "Evaluation",
        "Failure Analysis",
        "Explainability",
        "Export",
    ]
)


with tabs[0]:
    st.subheader("Dataset Section")
    dataset_source = st.radio(
        "Choose a dataset source",
        [*get_builtin_dataset_names(), "Upload CSV"],
        horizontal=True,
    )

    uploaded_file = None
    uploaded_bytes = None
    if dataset_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()

    loading_status = st.empty()
    loading_status.info("Loading the dataset and preparing the dashboard...")
    try:
        df = load_dataset_from_source(dataset_source, uploaded_bytes)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        df = pd.DataFrame()
    finally:
        loading_status.empty()

    if df.empty:
        st.info("Load a dataset to begin.")
    else:
        st.success(f"Loaded dataset with shape {df.shape}.")

        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown("#### Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        with right:
            st.markdown("#### Dataset Summary")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            missing_summary = df.isna().sum().sort_values(ascending=False)
            st.dataframe(
                missing_summary[missing_summary > 0].rename("missing_count").to_frame(),
                use_container_width=True,
            )

        st.markdown("#### Column Types")
        dtype_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        if dataset_source in get_builtin_dataset_names():
            target_default = get_builtin_dataset_default_target(dataset_source)
        else:
            target_default = "survived" if "survived" in df.columns else df.columns[-1]

        target_col = st.selectbox(
            "Select the target column",
            df.columns,
            index=list(df.columns).index(target_default) if target_default in df.columns else 0,
        )
        target_unique = df[target_col].nunique(dropna=True)
        unique_ratio = target_unique / max(len(df), 1)

        if target_unique < 2:
            st.error("The selected target column needs at least two unique values for classification.")
        elif target_unique > 20 or unique_ratio > 0.3:
            st.warning(
                "This target has many unique values. The app is designed for classification, so make sure this column represents classes rather than a continuous value."
            )

        if dataset_source == "Adult Income":
            st.info("Default Adult Income target: `income`.")
        elif dataset_source == "Loan Default":
            st.info("Default Loan Default target: `default_status`.")

        st.session_state.dataset = df
        st.session_state.target_col = target_col
        st.session_state.dataset_source = dataset_source


with tabs[1]:
    st.subheader("Model Training")

    if "dataset" not in st.session_state or st.session_state.dataset.empty:
        st.info("Load a dataset in the Dataset tab first.")
    else:
        df = st.session_state.dataset
        target_col = st.session_state.target_col
        available_models = get_available_model_names()

        c1, c2 = st.columns([1, 1])
        with c1:
            model_name = st.selectbox("Choose a model", available_models, index=0)
        with c2:
            test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

        if model_name == "XGBoost":
            st.caption("XGBoost is available as an additional tabular model option.")

        train_clicked = st.button("Train Model", type="primary")

        if train_clicked:
            try:
                with st.spinner("Training the model and preparing analysis artifacts..."):
                    artifacts = train_model(
                        df=df,
                        target_col=target_col,
                        model_name=model_name,
                        test_size=test_size,
                    )
                    st.session_state.artifacts = artifacts
                    st.session_state.prediction_frame = build_prediction_frame(
                        artifacts.X_test,
                        artifacts.y_test,
                        artifacts.y_pred,
                        artifacts.label_encoder,
                        artifacts.y_proba,
                    )
                    st.session_state.test_size = test_size
                    st.session_state.feature_error_tables = error_rate_by_feature(
                        artifacts.X_test,
                        artifacts.y_test != artifacts.y_pred,
                        artifacts.numeric_features,
                        artifacts.categorical_features,
                    )
                    st.session_state.failure_insights = build_failure_insights(
                        st.session_state.prediction_frame,
                        st.session_state.feature_error_tables,
                    )
                    st.session_state.shap_values = None
                    st.session_state.shap_feature_matrix = None
                    st.session_state.shap_message = None
                st.success("Training completed successfully. Go to Evaluation and Failure Analysis for results.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        st.markdown("#### Current Setup")
        st.write(f"**Dataset:** {st.session_state.get('dataset_source', 'Unknown')}")
        st.write(f"**Target:** {target_col}")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Test split:** {test_size:.2f}")
        st.caption("Evaluation metrics, confusion matrix, and the classification report now live only in the Evaluation tab.")


with tabs[2]:
    st.subheader("Evaluation")
    artifacts = st.session_state.artifacts
    prediction_frame = st.session_state.prediction_frame

    if artifacts is None or prediction_frame is None:
        st.info("Train a model first to unlock evaluation insights.")
    else:
        metrics = compute_metrics(
            artifacts.y_test,
            artifacts.y_pred,
            artifacts.y_proba,
            artifacts.is_binary,
        )
        render_metric_cards(metrics, artifacts.is_binary)

        st.markdown("#### Confusion Matrix")
        st.pyplot(
            confusion_matrix_figure(
                artifacts.y_test,
                artifacts.y_pred,
                list(artifacts.label_encoder.classes_),
            ),
            clear_figure=True,
        )
        st.caption(f"Current model: {artifacts.model_name}")

        with st.expander("Classification report", expanded=False):
            st.dataframe(
                classification_report_df(
                    artifacts.y_test,
                    artifacts.y_pred,
                    list(artifacts.label_encoder.classes_),
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Quick model comparison", expanded=False):
            st.caption("Compare the supported models on the current dataset without cluttering the main view.")
            compare_models = st.button("Run comparison", key="run_model_comparison")
            if compare_models:
                try:
                    compare_df = cached_model_comparison(
                        st.session_state.dataset,
                        st.session_state.target_col,
                        float(st.session_state.get('test_size', 0.2)),
                        tuple(get_available_model_names()),
                    )
                    st.dataframe(compare_df.sort_values("f1", ascending=False), use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.warning(f"Model comparison could not be completed: {exc}")


with tabs[3]:
    st.subheader("Failure Analysis Section")
    artifacts = st.session_state.artifacts
    prediction_frame = st.session_state.prediction_frame
    feature_error_tables = st.session_state.feature_error_tables or {}

    if artifacts is None or prediction_frame is None:
        st.info("Train a model first to analyze misclassifications.")
    else:
        total_test = len(prediction_frame)
        wrong_predictions = int((~prediction_frame["correct"]).sum())
        error_rate = wrong_predictions / max(total_test, 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total test samples", total_test)
        c2.metric("Wrong predictions", wrong_predictions)
        c3.metric("Error rate", f"{error_rate:.1%}")

        st.markdown("#### Misclassified Samples")
        mis_df = misclassified_samples(prediction_frame)
        if mis_df.empty:
            st.success("No misclassifications found on the test split.")
        else:
            display_cols = [c for c in mis_df.columns if c not in {"correct"}]
            st.dataframe(
                mis_df[display_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Error Patterns")
        if feature_error_tables:
            feature_names = list(feature_error_tables.keys())
            selected_feature = st.selectbox(
                "Inspect a feature",
                feature_names,
                index=0,
                key="failure_feature_select",
            )
            render_feature_failure_table(selected_feature, feature_error_tables[selected_feature].head(8))
        else:
            st.info("No feature-level failure analysis could be generated.")

        with st.expander("Confidence comparison", expanded=False):
            render_confidence_comparison(prediction_frame)

        with st.expander("Failure insights", expanded=True):
            for insight in st.session_state.failure_insights or []:
                st.markdown(f"- {insight}")


with tabs[4]:
    st.subheader("Explainability")
    artifacts = st.session_state.artifacts
    prediction_frame = st.session_state.prediction_frame

    if artifacts is None or prediction_frame is None:
        st.info("Train a model first to unlock explainability tools.")
    else:
        with st.expander("Global feature importance", expanded=True):
            global_importance = get_global_importance(artifacts.pipeline, artifacts.feature_names)
            if global_importance is not None:
                st.pyplot(
                    plot_feature_importance(
                        artifacts.feature_names,
                        global_importance,
                        top_n=min(12, len(artifacts.feature_names)),
                    ),
                    clear_figure=True,
                )
            else:
                st.info("Feature importance fallback is not available for this model.")

        if st.session_state.shap_values is None and st.session_state.shap_message is None:
            with st.spinner("Computing SHAP explanations..."):
                shap_values, base_values, feature_matrix, shap_message = compute_shap_explanation(
                    artifacts.pipeline,
                    artifacts.X_train,
                    artifacts.X_test,
                    artifacts.feature_names,
                    artifacts.model_name,
                )
                st.session_state.shap_values = shap_values
                st.session_state.shap_feature_matrix = feature_matrix
                st.session_state.shap_message = shap_message

        if st.session_state.shap_message:
            st.warning(st.session_state.shap_message)

        shap_values = st.session_state.shap_values
        feature_matrix = st.session_state.shap_feature_matrix
        with st.expander("Explain one misclassified row", expanded=True):
            mis_df = misclassified_samples(prediction_frame)
            if mis_df.empty:
                st.success("There are no misclassified samples to explain.")
            elif shap_values is not None and feature_matrix is not None:
                row_options = mis_df.index.tolist()
                selected_row = st.selectbox(
                    "Select a misclassified row",
                    row_options,
                    format_func=lambda idx: f"Row {idx}",
                    key="shap_mis_row",
                )
                selected_values = prediction_frame.loc[selected_row]
                st.dataframe(selected_values.to_frame().T, use_container_width=True, hide_index=True)
                try:
                    st.pyplot(
                        plot_local_shap_bars(shap_values, artifacts.feature_names, selected_row),
                        clear_figure=True,
                    )
                except Exception as exc:
                    st.warning(f"Could not render local SHAP explanation: {exc}")
            else:
                st.info("SHAP was not available for this model, so only global importance is shown.")


with tabs[5]:
    st.subheader("Export Section")
    artifacts = st.session_state.artifacts
    prediction_frame = st.session_state.prediction_frame

    if artifacts is None or prediction_frame is None:
        st.info("Train a model first to enable downloads.")
    else:
        st.markdown("#### Download Misclassified Samples")
        mis_df = misclassified_samples(prediction_frame)
        csv_bytes = mis_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download misclassified samples as CSV",
            data=csv_bytes,
            file_name="misclassified_samples.csv",
            mime="text/csv",
        )

        st.markdown("#### Download Trained Model")
        model_path = save_model_to_outputs(artifacts)
        model_bytes = save_model_bytes(artifacts)
        st.download_button(
            "Download trained model as .pkl",
            data=model_bytes,
            file_name="trained_model.pkl",
            mime="application/octet-stream",
        )
        st.caption(f"A copy of the model was also saved locally at `{model_path.as_posix()}`.")
