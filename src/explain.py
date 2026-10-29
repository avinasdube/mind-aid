"""Simple explainability hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .preprocess import clean_text
from .predict_text import predict_emotion


_NOISE_KEYWORDS = {
    "reddit",
    "tldr",
    "amp",
    "http",
    "https",
    "url",
    "img",
    "www",
    "haven",
    "havent",
}


def _resolve_linear_coefficients(model: Any) -> np.ndarray | None:
    """Return linear coefficients from plain or calibrated linear classifiers."""
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_)

    if hasattr(model, "calibrated_classifiers_"):
        fold_coefs: list[np.ndarray] = []
        for calibrated in getattr(model, "calibrated_classifiers_", []):
            estimator = getattr(calibrated, "estimator", None)
            if estimator is not None and hasattr(estimator, "coef_"):
                fold_coefs.append(np.asarray(estimator.coef_))

        if fold_coefs:
            return np.mean(np.stack(fold_coefs, axis=0), axis=0)

    if hasattr(model, "estimator") and hasattr(model.estimator, "coef_"):
        return np.asarray(model.estimator.coef_)

    return None


def get_top_keywords(
    text: str,
    vectorizer: Any,
    model: Any,
    top_k: int = 10,
) -> list[dict[str, float | str]]:
    """Return top positively contributing input keywords for the model decision."""
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []

    features = vectorizer.transform([cleaned_text])
    if features.nnz == 0:
        return []

    prediction = model.predict(features)[0]
    class_labels = np.asarray(getattr(model, "classes_", []))
    class_index = int(np.where(class_labels == prediction)[0][0]) if class_labels.size else 0

    coefficients = _resolve_linear_coefficients(model)
    if coefficients is None:
        return []

    if coefficients.ndim == 1:
        class_coef = coefficients
    elif coefficients.shape[0] == 1:
        class_coef = coefficients[0]
    else:
        class_coef = coefficients[class_index]

    feature_names = vectorizer.get_feature_names_out()
    contributions: list[dict[str, float | str]] = []

    row = features[0]
    for feature_idx, tfidf_value in zip(row.indices, row.data):
        contribution = float(tfidf_value * class_coef[feature_idx])
        if contribution > 0:
            contributions.append(
                {
                    "keyword": str(feature_names[feature_idx]),
                    "contribution": contribution,
                }
            )

    contributions.sort(key=lambda item: float(item["contribution"]), reverse=True)

    # Prefer semantically meaningful terms for UI/readability.
    filtered: list[dict[str, float | str]] = []
    for item in contributions:
        keyword = str(item["keyword"]).strip().lower()
        if len(keyword) <= 2:
            continue
        if keyword in ENGLISH_STOP_WORDS:
            continue
        if keyword in _NOISE_KEYWORDS:
            continue
        if keyword.isdigit():
            continue
        filtered.append(item)

    selected = filtered if filtered else contributions
    return selected[: max(1, top_k)]


def explain_prediction(
    model_path: Path,
    text: str,
    vectorizer_path: Path = Path("models/vectorizer.pkl"),
) -> str:
    """Return a lightweight explanation for a prediction."""
    label = predict_emotion(model_path=model_path, vectorizer_path=vectorizer_path, text=text)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    top_keywords = get_top_keywords(text=text, vectorizer=vectorizer, model=model, top_k=5)
    keyword_text = ", ".join(str(item["keyword"]) for item in top_keywords) if top_keywords else "none"

    return (
        f"Prediction: {label}. "
        f"Top contributing keywords: {keyword_text}. "
        "Explanation method: TF-IDF feature value x LinearSVC learned coefficient."
    )
