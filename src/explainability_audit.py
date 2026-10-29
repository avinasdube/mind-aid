"""Reproducible validation for local text attribution explanations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .explain import get_top_keywords
from .preprocess import clean_text
from .utils import ensure_parent_dir


def _predict(model: Any, vectorizer: Any, text: str) -> tuple[int, float]:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text is empty after preprocessing")
    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    probabilities = np.asarray(model.predict_proba(features))[0]
    classes = np.asarray(model.classes_)
    positive_index = int(np.where(classes == 1)[0][0]) if 1 in classes else len(classes) - 1
    return prediction, float(probabilities[positive_index])


def _mask_keywords(text: str, keywords: list[str]) -> str:
    masked = text
    for keyword in sorted(keywords, key=len, reverse=True):
        terms = [re.escape(term) for term in keyword.split()]
        pattern = r"\b" + r"\s+".join(terms) + r"\b"
        masked = re.sub(pattern, " ", masked, flags=re.IGNORECASE)
    return " ".join(masked.split())


def run_explainability_audit(
    data_path: Path,
    model_path: Path,
    vectorizer_path: Path,
    output_path: Path,
    *,
    max_samples: int = 715,
    top_k: int = 5,
) -> dict[str, Any]:
    """Measure attribution stability, masking sensitivity, and example cases."""
    frame = pd.read_csv(data_path).dropna(subset=["text", "label"]).head(max_samples).copy()
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    audits: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = {"correct": [], "misclassified": []}

    for _, row in frame.iterrows():
        text = str(row["text"])
        label = int(row["label"])
        prediction, probability = _predict(model, vectorizer, text)
        explanation = get_top_keywords(text, vectorizer, model, top_k=top_k)
        keywords = [str(item["keyword"]) for item in explanation]
        masked_text = _mask_keywords(text, keywords)
        masked_prediction, masked_probability = _predict(model, vectorizer, masked_text)
        punctuation_variant = re.sub(r"[^\w\s]", " ", text)
        variant_keywords = [
            str(item["keyword"])
            for item in get_top_keywords(punctuation_variant, vectorizer, model, top_k=top_k)
        ]
        left = set(keywords)
        right = set(variant_keywords)
        stability = len(left & right) / len(left | right) if left | right else 1.0
        audits.append(
            {
                "label": label,
                "prediction": prediction,
                "positive_probability": probability,
                "masked_positive_probability": masked_probability,
                "masked_prediction": masked_prediction,
                "top_keywords": keywords,
                "keyword_jaccard_after_punctuation_variant": stability,
                "absolute_probability_change_after_masking": abs(probability - masked_probability),
            }
        )
        bucket = "correct" if label == prediction else "misclassified"
        if len(examples[bucket]) < 3:
            examples[bucket].append(
                {
                    "label": label,
                    "prediction": prediction,
                    "positive_probability": probability,
                    "text": text[:500],
                    "top_keywords": keywords,
                }
            )

    audit_frame = pd.DataFrame(audits)
    report: dict[str, Any] = {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "vectorizer_path": str(vectorizer_path),
        "method": {
            "attribution": "TF-IDF value multiplied by learned linear coefficient",
            "stability": "Top-keyword Jaccard similarity after punctuation perturbation",
            "faithfulness_proxy": "Absolute positive-probability change after masking top keywords",
            "clinical_interpretation": "Statistical feature influence only; not a clinically validated indicator",
        },
        "sample_count": int(len(audit_frame)),
        "summary": {
            "mean_keyword_stability": float(audit_frame["keyword_jaccard_after_punctuation_variant"].mean()),
            "mean_absolute_probability_change_after_masking": float(
                audit_frame["absolute_probability_change_after_masking"].mean()
            ),
            "prediction_flip_rate_after_masking": float(
                (audit_frame["prediction"] != audit_frame["masked_prediction"]).mean()
            ),
            "misclassified_count": int((audit_frame["label"] != audit_frame["prediction"]).sum()),
        },
        "examples": examples,
    }
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
