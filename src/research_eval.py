"""Research-oriented evaluation helpers for internal ablation and external validation."""

from __future__ import annotations

import json
import string
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.patches import FancyBboxPatch
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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.calibration import calibration_curve

from .emotion import analyze_emotion
from .fusion import fuse_scores
from .preprocess import clean_text
from .utils import ensure_parent_dir


LABEL_COLUMNS = ("label", "is_depression", "target", "y")
TEXT_COLUMNS = ("text", "clean_text", "content", "sentence", "body")
FACE_SCORE_COLUMNS = ("face_score", "emotion_score", "image_risk_score")
FACE_CONFIDENCE_COLUMNS = ("face_confidence", "emotion_confidence")
IMAGE_COLUMNS = ("image_path", "image", "photo_path", "filepath")


def _apply_chart_theme() -> None:
    """Use a consistent professional chart style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "axes.titleweight": "normal",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _rounded_bars(ax: Any, x_positions: list[float], heights: list[float], *, width: float, color: str, edge_color: str) -> None:
    """Draw rounded bars with clean edges."""
    for x, h in zip(x_positions, heights):
        rect = FancyBboxPatch(
            (x - (width / 2), 0),
            width,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.03",
            linewidth=1.0,
            edgecolor=edge_color,
            facecolor=color,
            mutation_aspect=1,
        )
        ax.add_patch(rect)


def _annotate_values(ax: Any, x_positions: list[float], heights: list[float], *, dy: float = 0.015) -> None:
    for x, h in zip(x_positions, heights):
        ax.text(x, h + dy, f"{h:.3f}", ha="center", va="bottom", fontsize=7, color="#0f172a")


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Model artifacts used during evaluation."""

    model: Any
    vectorizer: Any


@dataclass(frozen=True)
class TransformerBaselineArtifacts:
    """Transformer embedding baseline artifacts for side-by-side comparison."""

    encoder: Any
    classifier: LogisticRegression
    model_name: str


def _first_existing_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _coerce_binary_label(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Encountered a missing label value.")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return int(round(float(value)))

    text = str(value).strip().lower()
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "negative": 0,
        "positive": 1,
        "non_depression": 0,
        "not_depressed": 0,
        "depressed": 1,
    }

    if text in mapping:
        return mapping[text]

    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Unable to coerce label '{value}' into a binary class.") from exc


def _load_model_artifacts(model_path: Path, vectorizer_path: Path) -> EvaluationArtifacts:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return EvaluationArtifacts(model=model, vectorizer=vectorizer)


def _normalize_evaluation_frame(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    columns = list(df.columns)

    text_column = _first_existing_column(columns, TEXT_COLUMNS)
    label_column = _first_existing_column(columns, LABEL_COLUMNS)

    if text_column is None or label_column is None:
        raise ValueError(
            "Evaluation data must include a text column and a binary label column. "
            f"Found columns: {', '.join(columns)}"
        )

    normalized = df.copy().dropna(subset=[text_column, label_column])
    normalized["text"] = normalized[text_column].astype(str).map(clean_text)
    normalized["label"] = normalized[label_column].map(_coerce_binary_label)
    normalized = normalized[normalized["text"] != ""]

    face_score_column = _first_existing_column(columns, FACE_SCORE_COLUMNS)
    face_confidence_column = _first_existing_column(columns, FACE_CONFIDENCE_COLUMNS)
    image_column = _first_existing_column(columns, IMAGE_COLUMNS)

    if face_score_column is not None:
        normalized["face_score"] = pd.to_numeric(normalized[face_score_column], errors="coerce")
    else:
        normalized["face_score"] = np.nan

    if face_confidence_column is not None:
        normalized["face_confidence"] = pd.to_numeric(normalized[face_confidence_column], errors="coerce")
    else:
        normalized["face_confidence"] = np.nan

    if image_column is not None:
        normalized["image_path"] = normalized[image_column].astype(str)
    else:
        normalized["image_path"] = ""

    if normalized.empty:
        raise ValueError("No valid rows remain after evaluation preprocessing.")

    return normalized[["text", "label", "face_score", "face_confidence", "image_path"]].reset_index(drop=True)


def _positive_probability(model: Any, features: Any) -> float:
    classes = np.asarray(model.classes_)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))[0]
        if 1 in classes:
            positive_index = int(np.where(classes == 1)[0][0])
        else:
            positive_index = int(len(classes) - 1)
        return float(probabilities[positive_index])

    decision_scores = np.asarray(model.decision_function(features))
    if decision_scores.ndim == 1:
        margin = float(decision_scores[0])
        return float(1.0 / (1.0 + np.exp(-margin)))

    row = decision_scores[0]
    shifted = row - np.max(row)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores)
    return float(np.max(probs))


def _predict_from_artifacts(text: str, artifacts: EvaluationArtifacts) -> tuple[int, float]:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("Input text is empty after preprocessing")

    features = artifacts.vectorizer.transform([cleaned_text])
    prediction = int(artifacts.model.predict(features)[0])
    probability = _positive_probability(artifacts.model, features)
    return prediction, probability


def _predict_many_from_artifacts(texts: pd.Series, artifacts: EvaluationArtifacts) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and positive probabilities for many texts at once."""
    cleaned_texts = texts.map(clean_text)
    valid_mask = cleaned_texts != ""
    valid_texts = cleaned_texts[valid_mask]

    if valid_texts.empty:
        raise ValueError("Input text is empty after preprocessing")

    features = artifacts.vectorizer.transform(valid_texts.tolist())
    predictions = np.asarray(artifacts.model.predict(features), dtype=int)
    probabilities = np.asarray(artifacts.model.predict_proba(features))
    classes = np.asarray(artifacts.model.classes_)
    if 1 in classes:
        positive_index = int(np.where(classes == 1)[0][0])
    else:
        positive_index = int(len(classes) - 1)
    positive_probabilities = probabilities[:, positive_index].astype(float)

    full_predictions = np.zeros(len(texts), dtype=int)
    full_probabilities = np.zeros(len(texts), dtype=float)
    valid_indices = valid_mask.to_numpy()
    full_predictions[valid_indices] = predictions
    full_probabilities[valid_indices] = positive_probabilities
    return full_predictions, full_probabilities


def _build_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0
        metrics["brier_score"] = 0.0

    return metrics


def _build_transformer_baseline(train_frame: pd.DataFrame) -> tuple[TransformerBaselineArtifacts | None, str | None]:
    """Train a sentence-embedding logistic baseline on the internal training frame."""
    if train_frame.empty:
        return None, "Internal frame is empty."

    unique_labels = np.unique(train_frame["label"].to_numpy(dtype=int))
    if len(unique_labels) < 2:
        return None, "Need at least two classes to train transformer baseline."

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"sentence-transformers unavailable: {exc}"

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        encoder = SentenceTransformer(model_name)
        embeddings = encoder.encode(
            train_frame["text"].tolist(),
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        classifier.fit(embeddings, train_frame["label"].to_numpy(dtype=int))
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        return None, f"Unable to initialize transformer baseline: {exc}"

    return (
        TransformerBaselineArtifacts(
            encoder=encoder,
            classifier=classifier,
            model_name=model_name,
        ),
        None,
    )


def _evaluate_transformer_baseline(
    frame: pd.DataFrame,
    baseline_artifacts: TransformerBaselineArtifacts,
) -> dict[str, Any]:
    """Evaluate a transformer sentence-embedding baseline on a frame."""
    embeddings = baseline_artifacts.encoder.encode(
        frame["text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    probabilities = baseline_artifacts.classifier.predict_proba(embeddings)[:, 1].astype(float)
    predictions = np.asarray(probabilities >= 0.5, dtype=int)
    metrics = _build_metrics(frame["label"].to_numpy(dtype=int), predictions, probabilities)
    metrics["backbone"] = baseline_artifacts.model_name
    return metrics


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute a simple expected calibration error."""
    if len(y_true) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(len(y_true))
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        if not np.any(mask):
            continue

        bin_true = float(np.mean(y_true[mask]))
        bin_prob = float(np.mean(y_prob[mask]))
        ece += (float(np.sum(mask)) / total) * abs(bin_true - bin_prob)

    return float(ece)


def _bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Estimate 95% bootstrap confidence intervals for key classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return {}

    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "f1": [],
        "roc_auc": [],
        "brier_score": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        bt_true = y_true[idx]
        bt_pred = y_pred[idx]
        bt_prob = y_prob[idx]

        stats["accuracy"].append(float(accuracy_score(bt_true, bt_pred)))
        stats["balanced_accuracy"].append(float(balanced_accuracy_score(bt_true, bt_pred)))
        stats["f1"].append(float(precision_recall_fscore_support(bt_true, bt_pred, average="weighted", zero_division=0)[2]))
        stats["brier_score"].append(float(brier_score_loss(bt_true, bt_prob)))
        if len(np.unique(bt_true)) > 1:
            stats["roc_auc"].append(float(roc_auc_score(bt_true, bt_prob)))

    confidence_intervals: dict[str, dict[str, float]] = {}
    for metric, values in stats.items():
        if not values:
            continue
        confidence_intervals[metric] = {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
        }

    return confidence_intervals


def _extract_predictions_for_named_model(
    frame: pd.DataFrame,
    artifacts: EvaluationArtifacts,
    model_name: str,
    transformer_baseline_artifacts: TransformerBaselineArtifacts | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return predictions and probabilities for a named model block."""
    if model_name == "text_only_model":
        y_pred, y_prob = _predict_many_from_artifacts(frame["text"], artifacts)
        return np.asarray(y_pred, dtype=int), np.asarray(y_prob, dtype=float)

    if model_name == "text_threshold_045":
        _, y_prob = _predict_many_from_artifacts(frame["text"], artifacts)
        y_pred = np.asarray(y_prob >= 0.45, dtype=int)
        return y_pred, np.asarray(y_prob, dtype=float)

    if model_name == "text_threshold_050":
        _, y_prob = _predict_many_from_artifacts(frame["text"], artifacts)
        y_pred = np.asarray(y_prob >= 0.50, dtype=int)
        return y_pred, np.asarray(y_prob, dtype=float)

    if model_name == "transformer_sentence_embedding_baseline" and transformer_baseline_artifacts is not None:
        embeddings = transformer_baseline_artifacts.encoder.encode(
            frame["text"].tolist(),
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        y_prob = transformer_baseline_artifacts.classifier.predict_proba(embeddings)[:, 1].astype(float)
        y_pred = np.asarray(y_prob >= 0.5, dtype=int)
        return y_pred, np.asarray(y_prob, dtype=float)

    return None


def _add_confidence_intervals_to_results(
    results: dict[str, Any],
    frame: pd.DataFrame,
    artifacts: EvaluationArtifacts,
    transformer_baseline_artifacts: TransformerBaselineArtifacts | None,
) -> None:
    """Attach bootstrap CI blocks to key result entries in-place."""
    y_true = frame["label"].to_numpy(dtype=int)
    target_models = [
        "text_only_model",
        "text_threshold_045",
        "text_threshold_050",
        "transformer_sentence_embedding_baseline",
    ]

    for model_name in target_models:
        block = results.get(model_name)
        if not isinstance(block, dict) or block.get("status") == "skipped":
            continue

        extracted = _extract_predictions_for_named_model(
            frame,
            artifacts,
            model_name,
            transformer_baseline_artifacts,
        )
        if extracted is None:
            continue

        y_pred, y_prob = extracted
        block["confidence_intervals"] = _bootstrap_confidence_intervals(y_true, y_pred, y_prob)


def _save_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> Path | None:
    """Save a reliability diagram when probabilities are available."""
    if len(np.unique(y_true)) < 2 or len(y_prob) == 0:
        return None

    _apply_chart_theme()
    output_path = output_dir / "calibration_curve.png"
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_facecolor("#ffffff")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.2, label="Perfect calibration")
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        color="#0ea5e9",
        label="Model",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(axis="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _perturb_text(text: str, mode: str) -> str:
    """Create lightweight text perturbations for robustness checks."""
    if mode == "punctuation_removed":
        translation = str.maketrans("", "", string.punctuation)
        return text.translate(translation)

    if mode == "whitespace_collapsed":
        return " ".join(text.split())

    if mode == "word_dropout":
        words = text.split()
        if len(words) <= 4:
            return text
        keep_mask = np.array([(idx % 5) != 0 for idx in range(len(words))])
        kept = [word for word, keep in zip(words, keep_mask) if keep]
        return " ".join(kept) if kept else text

    if mode == "spelling_noise":
        replacements = {"the": "teh", "really": "realy", "feel": "feal", "help": "hlep"}
        return " ".join(replacements.get(word.lower(), word) for word in text.split())

    if mode == "slang_abbreviation":
        replacements = {"you": "u", "are": "r", "because": "bc", "people": "ppl"}
        return " ".join(replacements.get(word.lower(), word) for word in text.split())

    if mode == "negation_removed":
        negations = {"no", "not", "never", "neither", "nor", "cannot", "cant", "dont", "isnt", "wasnt"}
        return " ".join(word for word in text.split() if word.lower() not in negations)

    if mode == "stopwords_removed_preserving_negation":
        preserved = {"no", "not", "never", "neither", "nor", "cannot", "cant", "dont", "isnt", "wasnt"}
        return " ".join(
            word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS or word.lower() in preserved
        )

    if mode == "short_truncation":
        return " ".join(text.split()[:20])

    return text


def _evaluate_robustness(frame: pd.DataFrame, artifacts: EvaluationArtifacts) -> dict[str, Any]:
    """Measure how much predictions shift under simple text noise."""
    max_samples = 1500
    if len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=42).reset_index(drop=True)

    variants = {
        "punctuation_removed": [],
        "whitespace_collapsed": [],
        "word_dropout": [],
        "spelling_noise": [],
        "slang_abbreviation": [],
        "negation_removed": [],
        "stopwords_removed_preserving_negation": [],
        "short_truncation": [],
    }

    original_predictions, original_probabilities = _predict_many_from_artifacts(frame["text"], artifacts)

    for mode in variants:
        perturbed_texts = frame["text"].map(lambda text: _perturb_text(text, mode))
        perturbed_predictions, _ = _predict_many_from_artifacts(perturbed_texts, artifacts)
        variants[mode] = perturbed_predictions.tolist()

    y_true = frame["label"].to_numpy(dtype=int)
    original_predictions_arr = np.asarray(original_predictions, dtype=int)
    original_probabilities_arr = np.asarray(original_probabilities, dtype=float)

    robustness: dict[str, Any] = {
        "baseline": _build_metrics(y_true, original_predictions_arr, original_probabilities_arr),
        "variants": {},
    }

    for mode, predictions in variants.items():
        predictions_arr = np.asarray(predictions, dtype=int)
        robustness["variants"][mode] = {
            "accuracy": float(accuracy_score(y_true, predictions_arr)),
            "f1": float(precision_recall_fscore_support(y_true, predictions_arr, average="weighted", zero_division=0)[2]),
            "prediction_flip_rate": float(np.mean(predictions_arr != original_predictions_arr)),
        }

    return robustness


def _evaluate_text_only(frame: pd.DataFrame, artifacts: EvaluationArtifacts) -> dict[str, Any]:
    predictions, probabilities = _predict_many_from_artifacts(frame["text"], artifacts)

    return _build_metrics(
        frame["label"].to_numpy(dtype=int),
        predictions,
        probabilities,
    )


def _evaluate_risk_threshold(frame: pd.DataFrame, artifacts: EvaluationArtifacts, threshold: float) -> dict[str, Any]:
    _, probabilities = _predict_many_from_artifacts(frame["text"], artifacts)
    predictions = np.asarray(probabilities >= threshold, dtype=int)

    return _build_metrics(
        frame["label"].to_numpy(dtype=int),
        predictions,
        probabilities,
    )


def _resolve_face_signal(row: pd.Series, image_base_dir: Path) -> tuple[float, float, bool, str]:
    if pd.notna(row["face_score"]):
        confidence = float(row["face_confidence"]) if pd.notna(row["face_confidence"]) else 0.5
        return float(row["face_score"]), confidence, True, "provided_face_score"

    image_path = str(row.get("image_path", "")).strip()
    if image_path:
        resolved_path = Path(image_path)
        if not resolved_path.is_absolute():
            resolved_path = image_base_dir / resolved_path
        if resolved_path.exists():
            emotion_result = analyze_emotion(resolved_path)
            return (
                float(emotion_result.get("risk_score", 0.3)),
                float(emotion_result.get("confidence", 0.0)),
                True,
                str(emotion_result.get("note", "image_analyzed")),
            )

    return 0.0, 0.0, False, "face_unavailable"


def _evaluate_fusion(
    frame: pd.DataFrame,
    artifacts: EvaluationArtifacts,
    *,
    image_base_dir: Path,
    use_confidence: bool,
) -> dict[str, Any] | None:
    fused_probabilities: list[float] = []
    fused_predictions: list[int] = []
    face_sources: list[str] = []

    for _, row in frame.iterrows():
        _, text_probability = _predict_from_artifacts(str(row["text"]), artifacts)
        text_score = float(text_probability)
        text_confidence = float(max(text_probability, 1.0 - text_probability))
        face_score, face_confidence, face_available, source = _resolve_face_signal(row, image_base_dir)

        if not face_available:
            return None

        face_sources.append(source)

        if use_confidence:
            fused = fuse_scores(
                text_score,
                face_score,
                text_confidence=text_confidence,
                face_confidence=face_confidence,
                face_available=True,
            )
            final_score = float(fused["final_score"])
        else:
            final_score = float((0.7 * text_score) + (0.3 * face_score))

        fused_probabilities.append(final_score)
        fused_predictions.append(int(final_score >= 0.5))

    metrics = _build_metrics(
        frame["label"].to_numpy(dtype=int),
        np.asarray(fused_predictions, dtype=int),
        np.asarray(fused_probabilities, dtype=float),
    )
    metrics["face_source_notes"] = sorted(set(face_sources))
    return metrics


def _save_strategy_chart(results: dict[str, dict[str, Any]], output_path: Path) -> Path | None:
    rows = [
        (
            name,
            float(metrics.get("accuracy", 0.0)),
            float(metrics.get("balanced_accuracy", 0.0)),
            float(metrics.get("f1", 0.0)),
        )
        for name, metrics in results.items()
        if metrics.get("status", "ok") == "ok"
    ]

    if not rows:
        return None

    names = [row[0] for row in rows]
    accuracy = [row[1] for row in rows]
    balanced = [row[2] for row in rows]
    f1_scores = [row[3] for row in rows]

    _apply_chart_theme()
    x = np.arange(len(names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    bars1_x = [float(idx - width) for idx in x]
    bars2_x = [float(idx) for idx in x]
    bars3_x = [float(idx + width) for idx in x]

    _rounded_bars(ax, bars1_x, accuracy, width=width, color="#0ea5e9", edge_color="#0369a1")
    _rounded_bars(ax, bars2_x, balanced, width=width, color="#14b8a6", edge_color="#0f766e")
    _rounded_bars(ax, bars3_x, f1_scores, width=width, color="#f59e0b", edge_color="#b45309")

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "\n".join(textwrap.wrap(name.replace("_", " ").title(), width=16)) for name in names
    ])
    ax.grid(axis="y")
    ax.legend(
        handles=[
            Patch(facecolor="#0ea5e9", edgecolor="#0369a1", label="Accuracy"),
            Patch(facecolor="#14b8a6", edgecolor="#0f766e", label="Balanced Accuracy"),
            Patch(facecolor="#f59e0b", edgecolor="#b45309", label="F1"),
        ],
        loc="upper right",
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    _annotate_values(ax, bars1_x, accuracy)
    _annotate_values(ax, bars2_x, balanced)
    _annotate_values(ax, bars3_x, f1_scores)

    fig.tight_layout(pad=1.2)
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _evaluate_frame(
    frame: pd.DataFrame,
    artifacts: EvaluationArtifacts,
    data_path: Path,
    transformer_baseline_artifacts: TransformerBaselineArtifacts | None = None,
    transformer_skip_reason: str | None = None,
) -> dict[str, Any]:
    internal_results: dict[str, dict[str, Any]] = {
        "text_only_model": _evaluate_text_only(frame, artifacts),
        "text_threshold_045": _evaluate_risk_threshold(frame, artifacts, threshold=0.45),
        "text_threshold_050": _evaluate_risk_threshold(frame, artifacts, threshold=0.50),
    }

    fixed_fusion = _evaluate_fusion(
        frame,
        artifacts,
        image_base_dir=data_path.parent,
        use_confidence=False,
    )
    if fixed_fusion is not None:
        internal_results["fixed_text_face_fusion"] = fixed_fusion
    else:
        internal_results["fixed_text_face_fusion"] = {
            "status": "skipped",
            "reason": "No usable face score or image path columns were available.",
        }

    confidence_fusion = _evaluate_fusion(
        frame,
        artifacts,
        image_base_dir=data_path.parent,
        use_confidence=True,
    )
    if confidence_fusion is not None:
        internal_results["confidence_aware_fusion"] = confidence_fusion
    else:
        internal_results["confidence_aware_fusion"] = {
            "status": "skipped",
            "reason": "No usable face score or image path columns were available.",
        }

    internal_results["robustness_checks"] = _evaluate_robustness(frame, artifacts)

    if transformer_baseline_artifacts is not None:
        internal_results["transformer_sentence_embedding_baseline"] = _evaluate_transformer_baseline(
            frame,
            transformer_baseline_artifacts,
        )
    else:
        internal_results["transformer_sentence_embedding_baseline"] = {
            "status": "skipped",
            "reason": transformer_skip_reason or "Transformer baseline unavailable.",
        }

    _add_confidence_intervals_to_results(
        internal_results,
        frame,
        artifacts,
        transformer_baseline_artifacts,
    )

    return internal_results


def run_research_evaluation(
    data_path: Path,
    model_path: Path = Path("models/text_model.pkl"),
    vectorizer_path: Path = Path("models/vectorizer.pkl"),
    output_path: Path = Path("outputs/reports/research_evaluation.json"),
    chart_path: Path = Path("outputs/charts/ablation_comparison.png"),
    external_data_path: Path | None = None,
) -> dict[str, Any]:
    """Run in-sample diagnostics and optional external validation for a labeled dataset."""
    artifacts = _load_model_artifacts(model_path, vectorizer_path)
    internal_frame = _normalize_evaluation_frame(data_path)

    transformer_baseline_artifacts, transformer_skip_reason = _build_transformer_baseline(internal_frame)
    internal_results = _evaluate_frame(
        internal_frame,
        artifacts,
        data_path,
        transformer_baseline_artifacts=transformer_baseline_artifacts,
        transformer_skip_reason=transformer_skip_reason,
    )

    external_results: dict[str, Any] | None = None
    external_samples = 0
    if external_data_path is not None:
        external_frame = _normalize_evaluation_frame(external_data_path)
        external_results = _evaluate_frame(
            external_frame,
            artifacts,
            external_data_path,
            transformer_baseline_artifacts=transformer_baseline_artifacts,
            transformer_skip_reason=transformer_skip_reason,
        )
        external_results["robustness_checks"] = _evaluate_robustness(external_frame, artifacts)
        external_samples = int(len(external_frame))

    report: dict[str, Any] = {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "vectorizer_path": str(vectorizer_path),
        "evaluation_scope": {
            "internal": "in_sample_diagnostic",
            "external": "unseen_validation",
            "primary_claim_block": "external_validation",
        },
        "internal_ablation": internal_results,
        "internal_diagnostic_in_sample": internal_results,
        "external_validation": external_results,
        "dataset_summary": {
            "internal_samples": int(len(internal_frame)),
            "external_samples": external_samples,
        },
        "notes": [
            "Internal metrics are in-sample diagnostics and can be optimistic.",
            "Use external_validation as the primary evidence for generalization claims.",
            "Text-only metrics are always available.",
            "Fusion metrics require either face_score columns or image paths that can be analyzed locally.",
            "Transformer baseline uses sentence-transformers/all-MiniLM-L6-v2 embeddings with LogisticRegression when available.",
        ],
    }

    chart_results = {
        name: metrics for name, metrics in internal_results.items() if metrics.get("status", "ok") == "ok"
    }
    chart_created = _save_strategy_chart(chart_results, chart_path)

    _, internal_text_probabilities = _predict_many_from_artifacts(internal_frame["text"], artifacts)
    calibration_chart_path = _save_calibration_curve(
        internal_frame["label"].to_numpy(dtype=int),
        internal_text_probabilities,
        chart_path.parent,
    )

    if chart_created is not None:
        report["ablation_chart"] = str(chart_created)

    if calibration_chart_path is not None:
        report["calibration_chart"] = str(calibration_chart_path)

    report["internal_ablation"]["text_only_model"]["expected_calibration_error"] = _expected_calibration_error(
        internal_frame["label"].to_numpy(dtype=int),
        internal_text_probabilities,
    )

    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report
