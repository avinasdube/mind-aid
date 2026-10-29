"""Inference helpers for text and risk prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .preprocess import clean_text
from .safety import detect_crisis_language


def _load_artifacts(model_path: Path, vectorizer_path: Path) -> tuple[Any, Any]:
    """Load trained model and vectorizer artifacts from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def _label_to_int(label: Any, classes: np.ndarray) -> int:
    """Convert model label to int, falling back to class index when needed."""
    try:
        return int(label)
    except (TypeError, ValueError):
        for idx, class_label in enumerate(classes):
            if class_label == label:
                return idx
    return 0


def _confidence_from_decision_scores(decision_scores: np.ndarray) -> float:
    """Convert decision scores to a calibrated-style confidence value in [0, 1]."""
    scores = np.asarray(decision_scores)

    # Binary LinearSVC may return shape (n_samples,); multiclass returns (n_samples, n_classes).
    if scores.ndim == 1:
        margin = float(scores[0])
        return float(1.0 / (1.0 + np.exp(-margin)))

    row = scores[0]
    shifted = row - np.max(row)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores)
    return float(np.max(probs))


def _risk_label(raw_prediction: int, positive_probability: float, n_classes: int) -> str:
    """Map numeric prediction and confidence to risk tiers."""
    if positive_probability >= 0.75:
        return "High Risk"
    if positive_probability >= 0.45:
        return "Moderate Risk"

    return "Low Risk"


def _risk_score(raw_prediction: int, confidence: float, n_classes: int) -> float:
    """Convert prediction + confidence into a normalized risk score in [0, 1]."""
    return float(max(0.0, min(1.0, confidence)))


def predict_emotion(
    model_path: Path,
    vectorizer_path: Path = Path("models/vectorizer.pkl"),
    text: str = "",
) -> str:
    """Predict emotion label for a single text input."""
    model, vectorizer = _load_artifacts(model_path, vectorizer_path)
    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("Input text is empty after preprocessing")

    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]
    return str(prediction)


def predict_risk(
    text: str,
    model_path: Path = Path("models/text_model.pkl"),
    vectorizer_path: Path = Path("models/vectorizer.pkl"),
) -> dict[str, str | float | int]:
    """Predict risk tier with confidence and normalized risk score."""
    model, vectorizer = _load_artifacts(model_path, vectorizer_path)
    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("Input text is empty after preprocessing")

    crisis = detect_crisis_language(cleaned_text)
    if crisis.flagged:
        return {
            "label": "High Risk",
            "score": 1.0,
            "confidence": 1.0,
            "raw_prediction": 1,
            "positive_probability": 1.0,
            "crisis_flag": True,
            "crisis_message": crisis.message,
            "matched_phrases": crisis.matched_phrases,
        }

    features = vectorizer.transform([cleaned_text])
    predicted_label = model.predict(features)[0]

    classes = np.asarray(model.classes_)
    raw_prediction = _label_to_int(predicted_label, classes)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))[0]
        positive_index = int(np.where(classes == (1 if 1 in classes else classes[-1]))[0][0])
        positive_probability = float(probabilities[positive_index])
    else:
        decision_scores = np.asarray(model.decision_function(features))
        positive_probability = _confidence_from_decision_scores(decision_scores)

    confidence = float(max(positive_probability, 1.0 - positive_probability))
    label = _risk_label(raw_prediction, positive_probability, len(classes))
    risk_score = _risk_score(raw_prediction, positive_probability, len(classes))

    return {
        "label": label,
        "score": risk_score,
        "confidence": confidence,
        "raw_prediction": raw_prediction,
        "positive_probability": positive_probability,
        "crisis_flag": False,
        "crisis_message": "",
        "matched_phrases": [],
    }
