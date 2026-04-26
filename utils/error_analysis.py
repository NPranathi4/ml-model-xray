from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_prediction_frame(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_encoder,
    y_proba: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    actual_labels = label_encoder.inverse_transform(y_test)
    predicted_labels = label_encoder.inverse_transform(y_pred)

    result = X_test.copy().reset_index(drop=True)
    result["actual_label"] = actual_labels
    result["predicted_label"] = predicted_labels
    result["correct"] = actual_labels == predicted_labels

    if y_proba is not None:
        confidence = np.max(y_proba, axis=1)
        result["prediction_confidence"] = confidence
        if y_proba.shape[1] == 2:
            result["predicted_probability"] = np.where(
                y_pred == 1,
                y_proba[:, 1],
                y_proba[:, 0],
            )
        else:
            result["predicted_probability"] = confidence
    else:
        result["prediction_confidence"] = np.nan
        result["predicted_probability"] = np.nan

    return result


def misclassified_samples(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    return prediction_frame.loc[~prediction_frame["correct"]].copy()


def _safe_bin_series(series: pd.Series, bins: int = 4) -> pd.Series:
    clean = series.copy()
    non_null = clean.dropna()

    if non_null.empty:
        return pd.Series(["Missing"] * len(clean), index=clean.index)

    if non_null.nunique() == 1:
        return pd.Series(["Single value"] * len(clean), index=clean.index)

    if non_null.nunique() <= bins:
        try:
            binned = pd.cut(clean, bins=max(1, non_null.nunique()), duplicates="drop")
        except Exception:
            return pd.Series(["Single value"] * len(clean), index=clean.index)
    else:
        try:
            binned = pd.qcut(clean, q=bins, duplicates="drop")
        except Exception:
            binned = pd.cut(clean, bins=bins, duplicates="drop")

    return binned.astype(str).replace("nan", "Missing")


def error_rate_by_feature(
    X_test: pd.DataFrame,
    misclassified_mask: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    bins: int = 4,
) -> Dict[str, pd.DataFrame]:
    analyses: Dict[str, pd.DataFrame] = {}
    error_series = pd.Series(misclassified_mask, index=X_test.index)

    for feature in categorical_features:
        temp = pd.DataFrame({
            feature: X_test[feature].astype(str).fillna("Missing"),
            "is_error": error_series.values,
        })
        grouped = (
            temp.groupby(feature)["is_error"]
            .agg(["count", "sum"])
            .rename(columns={"sum": "errors"})
            .reset_index()
        )
        grouped["error_rate"] = grouped["errors"] / grouped["count"]
        grouped = grouped.sort_values(["error_rate", "count"], ascending=[False, False])
        analyses[feature] = grouped

    for feature in numeric_features:
        temp = pd.DataFrame({
            feature: X_test[feature],
            "is_error": error_series.values,
        })
        temp["bin"] = _safe_bin_series(temp[feature], bins=bins)
        grouped = (
            temp.groupby("bin")["is_error"]
            .agg(["count", "sum"])
            .rename(columns={"sum": "errors"})
            .reset_index()
        )
        grouped["error_rate"] = grouped["errors"] / grouped["count"]
        grouped = grouped.sort_values(["error_rate", "count"], ascending=[False, False])
        analyses[feature] = grouped

    return analyses


def build_failure_insights(
    prediction_frame: pd.DataFrame,
    feature_error_tables: Dict[str, pd.DataFrame],
) -> List[str]:
    insights: List[str] = []

    total = len(prediction_frame)
    wrong = int((~prediction_frame["correct"]).sum())
    if total:
        insights.append(f"The model made {wrong} wrong predictions out of {total} test samples.")

    if wrong == 0:
        insights.append(
            "No misclassifications were found on this split, so failure patterns are not informative. Try a different test split or remove leakage-prone features."
        )
        return insights

    if "prediction_confidence" in prediction_frame.columns:
        correct_conf = prediction_frame.loc[prediction_frame["correct"], "prediction_confidence"].dropna()
        wrong_conf = prediction_frame.loc[~prediction_frame["correct"], "prediction_confidence"].dropna()
        if len(correct_conf) and len(wrong_conf):
            insights.append(
                f"The model is more confident when correct ({correct_conf.mean():.3f}) than when incorrect ({wrong_conf.mean():.3f})."
            )

    for feature, table in feature_error_tables.items():
        if table.empty:
            continue
        best_rows = table.loc[table["error_rate"] > 0]
        if best_rows.empty:
            continue
        worst = best_rows.iloc[0]
        label = worst.get("bin", worst.iloc[0] if len(worst) else "Unknown")
        error_rate = worst["error_rate"]
        if feature in prediction_frame.columns and prediction_frame[feature].dtype.name == "object":
            insights.append(
                f"The model has the highest error rate for {feature} = {label} ({error_rate:.1%})."
            )
        else:
            insights.append(
                f"The model fails more for {feature} in {label} ({error_rate:.1%})."
            )

    return insights
