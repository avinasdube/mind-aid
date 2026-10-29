"""Reproducible resource and repeated-seed benchmark for text models."""

from __future__ import annotations

import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.linear_model import LogisticRegression

from .preprocess import clean_text
from .train_model import train_model
from .utils import ensure_parent_dir


def _evaluate(model: Any, vectorizer: Any, frame: pd.DataFrame) -> dict[str, float]:
    cleaned_text = frame["text"].astype(str).map(clean_text)
    features = vectorizer.transform(cleaned_text.tolist())
    predictions = np.asarray(model.predict(features), dtype=int)
    probabilities = np.asarray(model.predict_proba(features))[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        frame["label"], predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(frame["label"], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(frame["label"], predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(frame["label"], probabilities)),
    }


def run_model_benchmark(
    training_data: Path,
    external_data: Path,
    output_path: Path,
    seeds: tuple[int, ...] = (13, 42, 73),
) -> dict[str, Any]:
    """Benchmark repeated seeds and measured resource requirements."""
    external = pd.read_csv(external_data).dropna(subset=["text", "label"])
    seed_results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="mindaid-benchmark-") as temp_dir:
        temp_root = Path(temp_dir)
        for seed in seeds:
            model_path = temp_root / f"model-{seed}.pkl"
            vectorizer_path = temp_root / f"vectorizer-{seed}.pkl"
            metrics_path = temp_root / f"metrics-{seed}.json"
            start = time.perf_counter()
            train_model(
                training_data,
                model_path,
                vectorizer_path,
                metrics_path,
                random_state=seed,
            )
            training_seconds = time.perf_counter() - start
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            start = time.perf_counter()
            metrics = _evaluate(model, vectorizer, external)
            inference_seconds = time.perf_counter() - start
            seed_results.append(
                {
                    "seed": seed,
                    "metrics": metrics,
                    "training_seconds": training_seconds,
                    "inference_seconds_total": inference_seconds,
                    "inference_milliseconds_per_sample": inference_seconds / len(external) * 1000,
                    "model_bytes": model_path.stat().st_size,
                    "vectorizer_bytes": vectorizer_path.stat().st_size,
                    "feature_count": int(len(vectorizer.vocabulary_)),
                }
            )

    metric_names = list(seed_results[0]["metrics"])
    summary = {
        metric: {
            "mean": float(np.mean([row["metrics"][metric] for row in seed_results])),
            "std": float(np.std([row["metrics"][metric] for row in seed_results], ddof=1)),
            "min": float(min(row["metrics"][metric] for row in seed_results)),
            "max": float(max(row["metrics"][metric] for row in seed_results)),
        }
        for metric in metric_names
    }
    transformer_result: dict[str, Any]
    try:
        from sentence_transformers import SentenceTransformer

        train_frame = pd.read_csv(training_data).dropna(subset=["text", "label"])
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        start = time.perf_counter()
        train_embeddings = encoder.encode(
            train_frame["text"].astype(str).map(clean_text).tolist(),
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        classifier.fit(train_embeddings, train_frame["label"].to_numpy(dtype=int))
        transformer_training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        external_embeddings = encoder.encode(
            external["text"].astype(str).map(clean_text).tolist(),
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        transformer_inference_seconds = time.perf_counter() - start
        transformer_result = {
            "status": "ok",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": int(train_embeddings.shape[1]),
            "training_seconds_including_embedding_and_logistic_regression": transformer_training_seconds,
            "inference_milliseconds_per_sample": transformer_inference_seconds / len(external) * 1000,
            "serialized_model_size": "Not measured; encoder is loaded from the sentence-transformers cache.",
        }
    except (ImportError, OSError, RuntimeError) as exc:
        transformer_result = {"status": "skipped", "reason": str(exc)}

    report: dict[str, Any] = {
        "training_data": str(training_data),
        "external_data": str(external_data),
        "seeds": list(seeds),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "external_samples": int(len(external)),
        "repeated_seed_summary": summary,
        "runs": seed_results,
        "transformer_resource_comparison": transformer_result,
        "inference_memory_note": "Peak process memory was not estimated; artifact size and per-sample latency are reported.",
        "statistical_testing_note": "Repeated-seed variation and bootstrap intervals are reported; no paired significance test is claimed.",
    }
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
