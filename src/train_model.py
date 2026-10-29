"""Model training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from .preprocess import clean_text
from .utils import ensure_parent_dir


def _resolve_cv(
    labels: pd.Series,
    max_folds: int = 5,
    random_state: int = 42,
) -> StratifiedKFold:
    """Create a stratified CV splitter that respects the smallest class size."""
    class_counts = labels.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_class_count < 2:
        raise ValueError("Need at least two samples in each class for cross-validation.")

    folds = min(max_folds, min_class_count)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)


def _evaluate_baselines(
    texts: pd.Series,
    labels: pd.Series,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Compare a few common text classifiers with cross-validation."""
    cv = _resolve_cv(labels, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
    }

    candidates = {
        "linear_svc": LinearSVC(class_weight="balanced"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "naive_bayes": MultinomialNB(),
    }

    comparison: dict[str, dict[str, float]] = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), estimator)
        scores = cross_validate(pipeline, texts, labels, scoring=scoring, cv=cv, n_jobs=None)
        comparison[name] = {
            metric: float(np.mean(scores[f"test_{metric}"])) for metric in scoring.keys()
        }
        comparison[name]["cv_folds"] = float(cv.get_n_splits())

    return comparison


def train_model(
    data_path: Path,
    model_path: Path = Path("models/text_model.pkl"),
    vectorizer_path: Path = Path("models/vectorizer.pkl"),
    metrics_path: Path = Path("metrics.json"),
    random_state: int = 42,
) -> dict[str, float]:
    """Train, evaluate, and persist a TF-IDF + calibrated LinearSVC classifier."""
    df = pd.read_csv(data_path)

    if {"text", "label"}.difference(df.columns):
        raise ValueError("Training data must include 'text' and 'label' columns")

    cleaned = df.dropna(subset=["text", "label"]).copy()
    cleaned["text"] = cleaned["text"].apply(clean_text)
    cleaned = cleaned[cleaned["text"] != ""]
    input_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["text"], keep="first")
    removed_duplicate_rows = input_rows - len(cleaned)

    if cleaned.empty:
        raise ValueError("No valid rows remain after preprocessing")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned["text"],
            cleaned["label"],
            test_size=0.2,
            random_state=random_state,
            stratify=cleaned["label"],
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned["text"],
            cleaned["label"],
            test_size=0.2,
            random_state=random_state,
            stratify=None,
        )

    eval_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = eval_vectorizer.fit_transform(X_train)
    X_test_vec = eval_vectorizer.transform(X_test)

    eval_model = CalibratedClassifierCV(
        estimator=LinearSVC(class_weight="balanced"),
        cv=5,
        method="sigmoid",
    )
    eval_model.fit(X_train_vec, y_train)

    y_pred = eval_model.predict(X_test_vec)
    y_prob = eval_model.predict_proba(X_test_vec)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    positive_precision, positive_recall, positive_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average=None,
        zero_division=0,
    )

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred).tolist()

    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        roc_auc = 0.0

    brier = float(brier_score_loss(y_test, y_prob))

    baseline_comparison = _evaluate_baselines(
        cleaned["text"],
        cleaned["label"],
        random_state=random_state,
    )
    class_counts = cleaned["label"].value_counts(dropna=False).sort_index()

    metrics: dict[str, float | dict[str, object]] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "brier_score": brier,
        "confusion_matrix": confusion,
        "classification_report": report,
        "class_0_precision": float(positive_precision[0]) if len(positive_precision) > 0 else 0.0,
        "class_0_recall": float(positive_recall[0]) if len(positive_recall) > 0 else 0.0,
        "class_0_f1": float(positive_f1[0]) if len(positive_f1) > 0 else 0.0,
        "class_1_precision": float(positive_precision[1]) if len(positive_precision) > 1 else 0.0,
        "class_1_recall": float(positive_recall[1]) if len(positive_recall) > 1 else 0.0,
        "class_1_f1": float(positive_f1[1]) if len(positive_f1) > 1 else 0.0,
        "dataset_summary": {
            "total_samples": int(len(cleaned)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "input_samples_before_deduplication": int(input_rows),
            "removed_duplicate_rows": int(removed_duplicate_rows),
            "class_distribution": {str(label): int(count) for label, count in class_counts.items()},
        },
        "holdout_metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": roc_auc,
            "brier_score": brier,
        },
        "baseline_comparison": baseline_comparison,
    }

    final_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    final_features = final_vectorizer.fit_transform(cleaned["text"])
    final_model = CalibratedClassifierCV(
        estimator=LinearSVC(class_weight="balanced"),
        cv=5,
        method="sigmoid",
    )
    final_model.fit(final_features, cleaned["label"])

    ensure_parent_dir(model_path)
    ensure_parent_dir(vectorizer_path)
    ensure_parent_dir(metrics_path)

    joblib.dump(final_model, model_path)
    joblib.dump(final_vectorizer, vectorizer_path)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return metrics
