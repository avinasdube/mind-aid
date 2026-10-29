"""FastAPI backend for the React MIND-AID frontend."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import joblib
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.emotion import analyze_emotion
from src.explain import get_top_keywords
from src.fusion import fuse_scores
from src.predict_text import predict_risk
from src.safety import detect_crisis_language
from src.distilbert_explainability import (
    explain_distilbert_text,
    predict_distilbert_text,
)

def _resolve_artifact_paths() -> tuple[Path, Path]:
    """Resolve model paths with environment override and strict-model preference."""
    model_env = os.getenv("MIND_AID_MODEL_PATH")
    vectorizer_env = os.getenv("MIND_AID_VECTORIZER_PATH")
    if model_env and vectorizer_env:
        return Path(model_env), Path(vectorizer_env)

    strict_model = Path("models/text_model_dreaddit_strict.pkl")
    strict_vectorizer = Path("models/vectorizer_dreaddit_strict.pkl")
    if strict_model.exists() and strict_vectorizer.exists():
        return strict_model, strict_vectorizer

    return Path("models/text_model.pkl"), Path("models/vectorizer.pkl")


MODEL_PATH, VECTORIZER_PATH = _resolve_artifact_paths()
DISTILBERT_MODEL_PATH = Path("models/distilbert_dreaddit_strict")
TEMP_UPLOAD_DIR = Path("outputs/temp_uploads")
LOG_PATH = Path("outputs/logs.csv")


@lru_cache(maxsize=1)
def load_text_artifacts() -> tuple[object, object]:
    """Load model artifacts once and reuse for future requests."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@lru_cache(maxsize=1)
def load_distilbert_artifacts() -> tuple[object, object, torch.device]:
    """Load the fine-tuned DistilBERT model and tokenizer once."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not DISTILBERT_MODEL_PATH.exists():
        raise FileNotFoundError(f"DistilBERT model directory not found: {DISTILBERT_MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def validate_image_bytes(image_bytes: bytes) -> None:
    """Validate image format from raw bytes."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpeg or is_png):
        raise ValueError("Invalid image file. Please upload a JPG or PNG image.")


def save_uploaded_image(image_bytes: bytes, filename: str) -> Path:
    """Persist uploaded image in temp uploads and return its path."""
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix or ".jpg"
    file_path = TEMP_UPLOAD_DIR / f"upload_{uuid4().hex}{suffix}"
    file_path.write_bytes(image_bytes)
    return file_path


def format_keywords(items: list[dict[str, float | str]]) -> list[str]:
    """Convert keyword dictionaries into keyword strings."""
    return [str(item["keyword"]) for item in items] if items else []


def emotion_intensity_from_score(score: float) -> str:
    """Map emotion risk score to intensity buckets."""
    score = float(max(0.0, min(1.0, score)))
    if score <= 0.35:
        return "low"
    if score <= 0.65:
        return "moderate"
    return "high"


def summarize_input_text(text: str, max_len: int = 220) -> str:
    """Create concise one-line summary for report output."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    return f"{collapsed[:max_len].rstrip()}..."


def build_text_report(
    input_text: str,
    text_score: float,
    text_label: str,
    face_emotion: str,
    final_score: float,
    final_label: str,
    keywords: list[str],
    screening_label: str = "",
    screening_detail: str = "",
    safety_note: str = "",
) -> str:
    """Build downloadable report for API responses."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = summarize_input_text(input_text)
    keyword_text = ", ".join(keywords) if keywords else "none"

    return (
        "MIND-AID Analysis Report\n"
        "========================\n\n"
        f"date: {timestamp}\n"
        f"input summary: {summary}\n"
        f"text risk: {text_label} ({text_score:.3f})\n"
        f"face emotion: {face_emotion}\n"
        f"final result: {final_label} ({final_score:.3f})\n"
        f"screening summary: {screening_label}\n"
        f"screening note: {screening_detail}\n"
        f"safety note: {safety_note}\n"
        f"keywords: {keyword_text}\n"
        "disclaimer: This tool is not a medical diagnosis.\n"
    )


def screening_interpretation(risk_label: str) -> dict[str, str]:
    """Translate the model label into research-friendly plain language."""
    label = risk_label.lower()
    if "high" in label:
        return {
            "short": "High concern",
            "long": "Strong emotional distress signal; immediate support or follow-up is recommended.",
        }
    if "moderate" in label:
        return {
            "short": "Possible concern",
            "long": "Noticeable distress signal; follow-up screening and self-care support may help.",
        }
    return {
        "short": "Low concern",
        "long": "No strong distress signal detected in this input; continue monitoring if symptoms change.",
    }


def append_run_log(text_score: float, face_score: float, final_score: float, risk_label: str) -> None:
    """Append anonymous run record to outputs/logs.csv."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", encoding="utf-8", newline="") as log_file:
        writer = csv.writer(log_file)
        if not file_exists:
            writer.writerow(["timestamp", "text_score", "face_score", "final_score", "risk_label"])

        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{float(text_score):.6f}",
                f"{float(face_score):.6f}",
                f"{float(final_score):.6f}",
                str(risk_label),
            ]
        )


app = FastAPI(title="MIND-AID API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    text: str = Form(...),
    image: UploadFile | None = File(default=None),
    model_name: str = Form("tfidf"),
) -> dict[str, object]:
    """Run text and optional face analysis and return combined results."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required.")

    selected_model = model_name.strip().lower()
    if selected_model not in {"tfidf", "distilbert"}:
        raise HTTPException(status_code=400, detail="model_name must be 'tfidf' or 'distilbert'.")

    try:
        if selected_model == "tfidf":
            model, vectorizer = load_text_artifacts()
            text_result = predict_risk(text, MODEL_PATH, VECTORIZER_PATH)
            keywords = format_keywords(get_top_keywords(text, vectorizer, model, top_k=8))
            model_label = "TF-IDF + calibrated LinearSVC"
            explanation_method = "TF-IDF feature value × learned linear coefficient"
        else:
            distilbert_model, tokenizer, device = load_distilbert_artifacts()
            positive_probability = predict_distilbert_text(
                text,
                distilbert_model,
                tokenizer,
                device=device,
            )
            crisis = detect_crisis_language(text)
            if positive_probability >= 0.75:
                text_risk_label = "High Risk"
            elif positive_probability >= 0.45:
                text_risk_label = "Moderate Risk"
            else:
                text_risk_label = "Low Risk"
            text_result = {
                "label": text_risk_label,
                "score": positive_probability,
                "confidence": max(positive_probability, 1.0 - positive_probability),
                "positive_probability": positive_probability,
                "crisis_flag": crisis.flagged,
                "crisis_message": crisis.message,
                "matched_phrases": crisis.matched_phrases,
            }
            keywords_raw = explain_distilbert_text(
                text,
                distilbert_model,
                tokenizer,
                device=device,
            )
            keywords = [str(item["token"]) for item in keywords_raw]
            model_label = "Fine-tuned DistilBERT"
            explanation_method = "Token-level SHAP Partition attribution"
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Selected model artifacts not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{selected_model} prediction failed: {exc}") from exc

    crisis_flag = bool(text_result.get("crisis_flag", False))
    crisis_message = str(text_result.get("crisis_message", ""))
    crisis_phrases = list(text_result.get("matched_phrases", []))

    emotion_name = "image_not_uploaded"
    face_score = float(text_result["score"])
    face_note = "No image uploaded. Text-only mode used."
    face_signal_state = "not_uploaded"
    face_confidence = None

    if crisis_flag:
        face_note = crisis_message
        face_signal_state = "skipped_crisis"

    if image is not None and not crisis_flag:
        try:
            image_bytes = await image.read()
            validate_image_bytes(image_bytes)
            image_path = save_uploaded_image(image_bytes, image.filename or "upload.jpg")
            emotion_result = analyze_emotion(image_path)
            emotion_name = str(emotion_result.get("emotion", "unknown"))
            emotion_confidence = float(emotion_result.get("confidence", 0.0))
            candidate_face_score = float(emotion_result.get("risk_score", face_score))
            raw_note = str(emotion_result.get("note", "ok"))
            face_signal_valid = (
                emotion_name != "unknown"
                and candidate_face_score >= 0.0
                and emotion_confidence >= 0.2
            )

            if face_signal_valid:
                face_score = candidate_face_score
                face_confidence = emotion_confidence
                face_note = "Face analysis completed." if raw_note in {"ok", "ok_relaxed"} else f"Face analysis fallback: {raw_note}"
                face_signal_state = "used"
            else:
                face_score = float(text_result["score"])
                face_note = "Face signal ignored due to low confidence or undetected face."
                face_signal_state = "ignored"

            if emotion_name == "unknown":
                if raw_note and raw_note != "ok":
                    face_note = f"Face emotion not detected; fallback used ({raw_note})."
                else:
                    face_note = "Face emotion could not be confidently detected; fallback values applied."
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            emotion_name = "analysis_unavailable"
            face_note = "Face analysis unavailable; text score used for face score fallback."

    fusion_result = fuse_scores(
        float(text_result["score"]),
        face_score,
        text_confidence=float(text_result.get("confidence", text_result["score"])),
        face_confidence=face_confidence,
        face_available=face_signal_state == "used",
    )

    if crisis_flag:
        fusion_result = {"final_score": 1.0, "risk_label": "High Risk"}

    append_run_log(
        text_score=float(text_result["score"]),
        face_score=face_score,
        final_score=float(fusion_result["final_score"]),
        risk_label=str(fusion_result["risk_label"]),
    )

    confidence_pct = float(text_result.get("confidence", text_result["score"])) * 100.0
    interpretation = screening_interpretation(str(fusion_result["risk_label"]))

    emotion_display = emotion_name.replace("_", " ").title()
    if emotion_name == "image_not_uploaded":
        emotion_display = "Image Not Uploaded"
    elif emotion_name == "analysis_unavailable":
        emotion_display = "Analysis Unavailable"
    elif emotion_name == "unknown":
        emotion_display = "Not Detected"

    emotion_score = float(max(0.0, min(1.0, face_score)))
    emotion_intensity = emotion_intensity_from_score(emotion_score)
    if emotion_name in {"image_not_uploaded", "analysis_unavailable", "unknown"}:
        emotion_intensity = "none"

    report_text = build_text_report(
        input_text=text,
        text_score=float(text_result["score"]),
        text_label=str(text_result["label"]),
        face_emotion=emotion_display,
        final_score=float(fusion_result["final_score"]),
        final_label=str(fusion_result["risk_label"]),
        keywords=keywords,
        screening_label=str(interpretation["short"]),
        screening_detail=str(interpretation["long"]),
        safety_note=crisis_message if crisis_flag else "No crisis language detected.",
    )

    return {
        "risk_label": str(fusion_result["risk_label"]),
        "screening_label": interpretation["short"],
        "screening_detail": interpretation["long"],
        "final_score": float(fusion_result["final_score"]),
        "text_score": float(text_result["score"]),
        "depression_probability": float(text_result.get("positive_probability", text_result["score"])),
        "confidence_pct": confidence_pct,
        "emotion": emotion_display,
        "emotion_score": emotion_score,
        "emotion_intensity": emotion_intensity,
        "fusion_mode": fusion_result.get("fusion_mode", "confidence_weighted"),
        "fusion_text_weight": float(fusion_result.get("text_weight", 1.0)),
        "fusion_face_weight": float(fusion_result.get("face_weight", 0.0)),
        "keywords": keywords,
        "model": selected_model,
        "model_label": model_label,
        "explanation_method": explanation_method,
        "face_note": face_note,
        "face_signal_state": face_signal_state,
        "crisis_flag": crisis_flag,
        "crisis_message": crisis_message,
        "crisis_phrases": crisis_phrases,
        "report_text": report_text,
    }
