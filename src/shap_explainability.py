"""SHAP explanations for the project's TF-IDF linear text model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from .preprocess import clean_text
from .utils import ensure_parent_dir


class _AveragedLinearModel:
    """Linear decision model formed from calibrated classifier fold estimators."""

    def __init__(self, model: Any) -> None:
        estimators = [
            calibrated.estimator
            for calibrated in getattr(model, "calibrated_classifiers_", [])
            if hasattr(getattr(calibrated, "estimator", None), "coef_")
        ]
        if not estimators:
            raise TypeError("SHAP explanations require a calibrated linear classifier")
        self.coef_ = np.mean(
            np.stack([np.asarray(estimator.coef_) for estimator in estimators]),
            axis=0,
        )
        self.intercept_ = np.mean(
            np.stack([np.asarray(estimator.intercept_) for estimator in estimators]),
            axis=0,
        )
        self.classes_ = np.asarray(getattr(model, "classes_", [0, 1]))

    def decision_function(self, features: Any) -> np.ndarray:
        scores = np.asarray(features @ self.coef_.T) + self.intercept_
        return scores[:, 0] if scores.shape[1] == 1 else scores


def _top_shap_features(
    values: np.ndarray,
    feature_names: np.ndarray,
    top_k: int,
) -> list[dict[str, float | str]]:
    order = np.argsort(np.abs(values))[::-1]
    return [
        {
            "feature": str(feature_names[index]),
            "shap_value": float(values[index]),
            "absolute_shap_value": float(abs(values[index])),
        }
        for index in order[:top_k]
        if values[index] != 0
    ]


def explain_text_with_shap(
    text: str,
    model_path: Path,
    vectorizer_path: Path,
    *,
    background: Any,
    top_k: int = 10,
) -> dict[str, Any]:
    """Return exact linear SHAP attributions for one preprocessed text."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text is empty after preprocessing")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    linear_model = _AveragedLinearModel(model)
    features = vectorizer.transform([cleaned])
    explainer = shap.LinearExplainer(linear_model, background)
    explanation = explainer(features)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[0, :, 0]
    else:
        values = values[0]
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    decision_value = float(linear_model.decision_function(features)[0])
    feature_names = vectorizer.get_feature_names_out()

    return {
        "text": text,
        "cleaned_text": cleaned,
        "method": "SHAP LinearExplainer on the averaged calibrated LinearSVC fold coefficients",
        "explained_output": "uncalibrated LinearSVC decision function",
        "base_value": base_value,
        "decision_value": decision_value,
        "additivity_error": float(base_value + values.sum() - decision_value),
        "top_features": _top_shap_features(values, feature_names, top_k),
    }


def run_shap_audit(
    data_path: Path,
    model_path: Path,
    vectorizer_path: Path,
    output_path: Path,
    *,
    max_samples: int = 25,
    background_size: int = 100,
    top_k: int = 10,
) -> dict[str, Any]:
    """Generate a reproducible SHAP audit report for labeled text data."""
    frame = pd.read_csv(data_path).dropna(subset=["text", "label"]).head(max_samples).copy()
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    background_texts = frame["text"].astype(str).map(clean_text).replace("", np.nan).dropna()
    if background_texts.empty:
        raise ValueError("No non-empty text is available for the SHAP background")
    background = vectorizer.transform(background_texts.head(background_size))
    linear_model = _AveragedLinearModel(model)
    explainer = shap.LinearExplainer(linear_model, background)
    feature_names = vectorizer.get_feature_names_out()

    samples: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        cleaned = clean_text(str(row["text"]))
        if not cleaned:
            continue
        features = vectorizer.transform([cleaned])
        explanation = explainer(features)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[0, :, 0]
        else:
            values = values[0]
        base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
        decision_value = float(np.asarray(linear_model.decision_function(features))[0])
        samples.append(
            {
                "label": int(row["label"]),
                "text": str(row["text"])[:500],
                "base_value": base_value,
                "decision_value": decision_value,
                "additivity_error": float(base_value + values.sum() - decision_value),
                "top_features": _top_shap_features(values, feature_names, top_k),
            }
        )

    report: dict[str, Any] = {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "vectorizer_path": str(vectorizer_path),
        "method": {
            "explainer": "shap.LinearExplainer",
            "model": "averaged LinearSVC fold coefficients from CalibratedClassifierCV",
            "output": "uncalibrated LinearSVC decision function",
            "background_samples": int(background.shape[0]),
            "feature_count": int(background.shape[1]),
        },
        "sample_count": len(samples),
        "max_absolute_additivity_error": max(
            (abs(float(sample["additivity_error"])) for sample in samples),
            default=0.0,
        ),
        "samples": samples,
    }
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
