from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


try:
    from xgboost import XGBClassifier  # type: ignore

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None  # type: ignore
    XGBOOST_AVAILABLE = False


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    label_encoder: LabelEncoder
    feature_names: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    is_binary: bool
    target_name: str
    model_name: str


BUILTIN_DATASETS = {
    "Adult Income": {
        "file": "adult_income_sample.csv",
        "target": "income",
        "description": "Mixed categorical and numeric classification for income prediction.",
    },
    "Loan Default": {
        "file": "loan_default_sample.csv",
        "target": "default_status",
        "description": "Credit-risk style classification for default prediction.",
    },
}


def get_builtin_dataset_names() -> List[str]:
    return list(BUILTIN_DATASETS.keys())


def get_builtin_dataset_default_target(dataset_name: str) -> str:
    return BUILTIN_DATASETS.get(dataset_name, {}).get("target", "target")


def load_builtin_dataset(dataset_name: str) -> pd.DataFrame:
    """Load one of the bundled sample datasets used by the dashboard."""
    dataset = BUILTIN_DATASETS.get(dataset_name)
    if dataset is None:
        raise ValueError(f"Unknown built-in dataset: {dataset_name}")

    sample_path = Path(__file__).resolve().parents[1] / "sample_data" / dataset["file"]
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing bundled dataset file: {sample_path}")
    return pd.read_csv(sample_path)


# Backward-compatible alias for older notebooks or cached imports.
def load_default_titanic_data() -> pd.DataFrame:
    return load_builtin_dataset("Adult Income")


def infer_feature_columns(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target_col]).copy()
    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model(model_name: str, random_state: int = 42):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1500, solver="lbfgs")
    if model_name == "Random Forest Classifier":
        return RandomForestClassifier(
            n_estimators=250,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        return XGBClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def get_available_model_names() -> List[str]:
    # Always surface XGBoost in the UI; availability is checked again at train time.
    return ["Random Forest Classifier", "Logistic Regression", "XGBoost"]


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names: List[str] = []

        if numeric_features:
            feature_names.extend(numeric_features)

        if categorical_features:
            cat_transformer = preprocessor.named_transformers_.get("cat")
            if cat_transformer is not None:
                try:
                    encoder = cat_transformer.named_steps["onehot"]
                    feature_names.extend(
                        list(encoder.get_feature_names_out(categorical_features))
                    )
                    return feature_names
                except Exception:
                    feature_names.extend(categorical_features)

        return feature_names


def train_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str = "Random Forest Classifier",
    test_size: float = 0.2,
    random_state: int = 42,
    exclude_columns: Optional[List[str]] = None,
) -> TrainingArtifacts:
    clean_df = df.dropna(subset=[target_col]).copy()
    X = clean_df.drop(columns=[target_col]).copy()

    if exclude_columns:
        removable = [col for col in exclude_columns if col in X.columns]
        if removable:
            X = X.drop(columns=removable)

    y_raw = clean_df[target_col].astype(str).fillna("Missing")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    numeric_features, categorical_features = infer_feature_columns(clean_df, target_col)
    scale_numeric = model_name == "Logistic Regression"
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric)
    model = build_model(model_name, random_state=random_state)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    stratify = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_proba = None
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_proba = None

    feature_names = get_feature_names_from_preprocessor(
        pipeline.named_steps["preprocessor"],
        numeric_features,
        categorical_features,
    )

    return TrainingArtifacts(
        pipeline=pipeline,
        label_encoder=label_encoder,
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        is_binary=len(label_encoder.classes_) == 2,
        target_name=target_col,
        model_name=model_name,
    )
