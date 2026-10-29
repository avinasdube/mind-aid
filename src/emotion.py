"""Emotion taxonomy and related helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


EMOTION_RISK_MAP: dict[str, float] = {
    "happy": 0.1,
    "neutral": 0.3,
    "sad": 0.8,
    "fear": 0.7,
    "angry": 0.6,
    "surprise": 0.4,
    "disgust": 0.7,
}


@dataclass(frozen=True)
class EmotionResult:
    """A normalized emotion output."""

    label: str
    confidence: float


def normalize_label(label: str) -> str:
    """Normalize labels to lowercase snake-like text."""
    return label.strip().lower().replace(" ", "_")


def _extract_emotion_payload(result: Any, note: str) -> dict[str, str | float]:
    """Normalize DeepFace result payload into API-friendly fields."""
    if isinstance(result, list):
        if not result:
            return {"emotion": "unknown", "risk_score": 0.3, "confidence": 0.0, "note": f"{note}_empty"}
        result = result[0]

    if not isinstance(result, dict):
        return {"emotion": "unknown", "risk_score": 0.3, "confidence": 0.0, "note": f"{note}_invalid"}

    dominant = str(result.get("dominant_emotion", "unknown")).strip().lower()
    emotion_scores = result.get("emotion", {})
    dominant_score = float(emotion_scores.get(dominant, 0.0)) if isinstance(emotion_scores, dict) else 0.0
    risk_score = float(EMOTION_RISK_MAP.get(dominant, 0.3))

    return {
        "emotion": dominant,
        "risk_score": risk_score,
        "confidence": max(0.0, min(1.0, dominant_score / 100.0)),
        "note": note,
    }


def analyze_emotion(image_path: str | Path) -> dict[str, str | float]:
    """Detect dominant facial emotion from an image and map it to risk."""
    default_response = {"emotion": "unknown", "risk_score": 0.3, "confidence": 0.0, "note": "fallback"}

    try:
        from deepface import DeepFace
    except Exception as exc:
        fallback = dict(default_response)
        fallback["note"] = f"deepface_import_failed: {exc}"
        return fallback

    try:
        image_path = Path(image_path)
        if not image_path.exists():
            fallback = dict(default_response)
            fallback["note"] = "image_path_not_found"
            return fallback

        try:
            result: Any = DeepFace.analyze(
                img_path=str(image_path),
                actions=["emotion"],
                enforce_detection=True,
                detector_backend="retinaface",
                silent=True,
            )
            return _extract_emotion_payload(result, note="ok")
        except Exception as strict_exc:
            # Retry with relaxed detection for partially occluded faces.
            result = DeepFace.analyze(
                img_path=str(image_path),
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )
            payload = _extract_emotion_payload(result, note="ok_relaxed")
            if payload.get("emotion") == "unknown":
                payload["note"] = f"deepface_inference_failed: {strict_exc}"
            return payload
    except Exception as exc:
        fallback = dict(default_response)
        fallback["note"] = f"deepface_inference_failed: {exc}"
        return fallback
