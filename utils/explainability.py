from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _extract_transformed_data(pipeline, X):
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return transformed


def get_global_importance(pipeline, feature_names):
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.mean(np.abs(coef), axis=0)
    else:
        return None

    if len(values) != len(feature_names):
        return None

    importance = (
        np.abs(values)
        if np.any(values < 0)
        else values
    )

    return importance


def plot_feature_importance(feature_names, importance, top_n: int = 15):
    order = np.argsort(importance)[::-1][:top_n]
    names = [feature_names[i] for i in order][::-1]
    scores = importance[order][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    ax.barh(names, scores, color="#3b82f6")
    ax.set_title("Global Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def _normalize_shap_output(shap_values):
    if hasattr(shap_values, "values"):
        values = shap_values.values
        base_values = getattr(shap_values, "base_values", None)
        data = getattr(shap_values, "data", None)
        return values, base_values, data

    return shap_values, None, None


def compute_shap_explanation(pipeline, X_train, X_test, feature_names, model_name: str):
    try:
        import shap
    except Exception:
        return None, None, None, "SHAP is not available in this environment."

    try:
        background = X_train.sample(min(len(X_train), 100), random_state=42)
        X_train_transformed = _extract_transformed_data(pipeline, background)
        X_test_transformed = _extract_transformed_data(pipeline, X_test)
        model = pipeline.named_steps["model"]

        if model_name in {"Random Forest Classifier", "XGBoost"}:
            explainer = shap.TreeExplainer(model)
            raw_shap = explainer.shap_values(X_test_transformed)
        else:
            explainer = shap.LinearExplainer(model, X_train_transformed)
            raw_shap = explainer.shap_values(X_test_transformed)

        shap_values, base_values, data = _normalize_shap_output(raw_shap)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        return shap_values, base_values, X_test_transformed, None
    except Exception as exc:
        return None, None, None, f"SHAP failed gracefully: {exc}"


def plot_shap_summary(shap_values, X_values, feature_names):
    import shap

    fig = plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(shap_values, X_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        return plt.gcf()
    except Exception:
        plt.close(fig)
        raise


def plot_local_shap_bars(shap_values, feature_names, row_idx: int):
    values = np.asarray(shap_values[row_idx])
    order = np.argsort(np.abs(values))[::-1][:10]
    names = [feature_names[i] for i in order][::-1]
    scores = values[order][::-1]

    colors = ["#ef4444" if score < 0 else "#10b981" for score in scores]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, scores, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Local SHAP Explanation")
    ax.set_xlabel("SHAP value")
    fig.tight_layout()
    return fig

