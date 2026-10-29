"""Fusion module for combining multiple signals."""

from __future__ import annotations

from typing import Iterable


def average_confidence(scores: Iterable[float]) -> float:
    """Return the arithmetic mean of confidence scores."""
    score_list = list(scores)
    if not score_list:
        return 0.0
    return sum(score_list) / len(score_list)


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value into a closed interval."""
    return max(lower, min(upper, value))


def _risk_label_from_score(final_score: float) -> str:
    """Map fused score into a risk tier."""
    if final_score <= 0.35:
        return "Low Risk"
    if final_score <= 0.65:
        return "Moderate Risk"
    return "High Risk"


def fuse_scores(
    text_score: float,
    face_score: float,
    *,
    text_confidence: float | None = None,
    face_confidence: float | None = None,
    face_available: bool = True,
) -> dict[str, float | str]:
    """Fuse text and face risk scores into a final score and risk label.

    The fusion is confidence-aware so that low-confidence or unavailable
    facial analysis does not distort the text signal.
    """
    text_score = _clamp(float(text_score), 0.0, 1.0)
    face_score = _clamp(float(face_score), 0.0, 1.0)

    if not face_available:
        final_score = text_score
        text_weight = 1.0
        face_weight = 0.0
        fusion_mode = "text_only"
    else:
        text_weight = 0.7
        face_weight = 0.3

        if text_confidence is not None:
            text_weight = _clamp(0.55 + 0.25 * float(text_confidence), 0.55, 0.8)
        if face_confidence is not None:
            face_weight = _clamp(0.15 + 0.35 * float(face_confidence), 0.15, 0.45)

        if text_weight + face_weight == 0:
            text_weight = 1.0
            face_weight = 0.0

        weight_total = text_weight + face_weight
        text_weight /= weight_total
        face_weight /= weight_total
        final_score = (text_weight * text_score) + (face_weight * face_score)
        fusion_mode = "confidence_weighted"

    final_score = _clamp(final_score, 0.0, 1.0)

    return {
        "final_score": final_score,
        "risk_label": _risk_label_from_score(final_score),
        "text_weight": text_weight,
        "face_weight": face_weight,
        "fusion_mode": fusion_mode,
    }
