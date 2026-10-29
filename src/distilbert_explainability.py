"""Fine-tuned DistilBERT evaluation and SHAP text attribution."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from .preprocess import clean_text
from .utils import ensure_parent_dir


class _TextDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).map(clean_text)
    frame = frame[frame["text"] != ""].drop_duplicates(subset=["text"], keep="first")
    if frame.empty:
        raise ValueError(f"No usable text rows found in {path}")
    return frame


def _select_explanatory_tokens(
    tokens: list[str],
    values: np.ndarray,
    *,
    top_k: int = 10,
) -> list[dict[str, float | str]]:
    """Return readable, deduplicated content tokens ranked by SHAP magnitude."""
    selected: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for token_index in np.argsort(np.abs(values))[::-1]:
        token = " ".join(str(tokens[token_index]).split()).strip()
        normalized = token.lower()
        if (
            not normalized
            or normalized in seen
            or normalized in ENGLISH_STOP_WORDS
            or normalized in {"[cls]", "[sep]", "[pad]", "[mask]", "##"}
            or len(normalized) < 3
            or not any(character.isalnum() for character in normalized)
        ):
            continue
        seen.add(normalized)
        selected.append(
            {
                "token": token,
                "shap_value": float(values[token_index]),
                "absolute_shap_value": float(abs(values[token_index])),
            }
        )
        if len(selected) == top_k:
            break
    return selected


def _predict_probabilities(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probabilities.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def predict_distilbert_text(
    text: str,
    model: Any,
    tokenizer: Any,
    *,
    max_length: int = 256,
    device: torch.device | None = None,
) -> float:
    """Return the fine-tuned DistilBERT class-1 probability for one text."""
    active_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probability = _predict_probabilities(
        model,
        tokenizer,
        [clean_text(text)],
        batch_size=1,
        max_length=max_length,
        device=active_device,
    )
    return float(probability[0])


def explain_distilbert_text(
    text: str,
    model: Any,
    tokenizer: Any,
    *,
    max_length: int = 256,
    max_evals: int = 100,
    top_k: int = 8,
    device: torch.device | None = None,
) -> list[dict[str, float | str]]:
    """Return readable token-level SHAP contributions for one text."""
    import shap

    active_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_proba(texts: list[str]) -> np.ndarray:
        probabilities = _predict_probabilities(
            model,
            tokenizer,
            [clean_text(value) for value in texts],
            batch_size=8,
            max_length=max_length,
            device=active_device,
        )
        return np.column_stack([1.0 - probabilities, probabilities])

    explainer = shap.Explainer(
        predict_proba,
        shap.maskers.Text(tokenizer),
        algorithm="partition",
    )
    explanation = explainer([clean_text(text)], max_evals=max_evals)
    values = np.asarray(explanation.values[0])
    class_values = values[:, 1] if values.ndim == 2 else values
    tokens = [str(token) for token in explanation.data[0]]
    return _select_explanatory_tokens(tokens, class_values, top_k=top_k)


def train_distilbert(
    training_data: Path,
    external_data: Path,
    output_dir: Path,
    metrics_path: Path,
    *,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 8,
    max_length: int = 256,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune DistilBERT and evaluate it on a separate external dataset."""
    _set_seed(seed)
    train_frame = _load_frame(training_data)
    external_frame = _load_frame(external_data)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = _TextDataset(
        train_frame["text"].tolist(),
        train_frame["label"].astype(int).tolist(),
        tokenizer,
        max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()

    external_texts = external_frame["text"].tolist()
    probabilities = _predict_probabilities(
        model,
        tokenizer,
        external_texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    predictions = (probabilities >= 0.5).astype(int)
    labels = external_frame["label"].astype(int).to_numpy()
    metrics: dict[str, Any] = {
        "model": model_name,
        "training_data": str(training_data),
        "external_data": str(external_data),
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "train_samples": int(len(train_frame)),
        "external_samples": int(len(external_frame)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted")),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }
    ensure_parent_dir(metrics_path)
    ensure_parent_dir(output_dir / "config.json")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_distilbert_shap_audit(
    data_path: Path,
    model_dir: Path,
    output_path: Path,
    *,
    max_samples: int = 10,
    max_evals: int = 200,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Generate token-level SHAP values for a saved fine-tuned DistilBERT model."""
    import shap

    frame = _load_frame(data_path).head(max_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def predict_proba(texts: list[str]) -> np.ndarray:
        probabilities = _predict_probabilities(
            model,
            tokenizer,
            [clean_text(text) for text in texts],
            batch_size=batch_size,
            max_length=tokenizer.model_max_length
            if tokenizer.model_max_length < 10000
            else 256,
            device=device,
        )
        return np.column_stack([1.0 - probabilities, probabilities])

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker, algorithm="partition")
    explanations = explainer(
        frame["text"].tolist(),
        max_evals=max_evals,
        batch_size=batch_size,
    )
    samples: list[dict[str, Any]] = []
    for index, text in enumerate(frame["text"].tolist()):
        values = np.asarray(explanations.values[index])
        class_values = values[:, 1] if values.ndim == 2 else values
        tokens = [str(token) for token in explanations.data[index]]
        top_tokens = _select_explanatory_tokens(tokens, class_values)
        probabilities = predict_proba([text])[0]
        base_value = float(np.asarray(explanations.base_values[index])[1])
        additivity_error = float(base_value + class_values.sum() - probabilities[1])
        samples.append(
            {
                "label": int(frame.iloc[index]["label"]),
                "text": text[:500],
                "predicted_label": int(np.argmax(probabilities)),
                "class_1_probability": float(probabilities[1]),
                "base_value_class_1": base_value,
                "additivity_error": additivity_error,
                "top_tokens": top_tokens,
            }
        )

    report: dict[str, Any] = {
        "data_path": str(data_path),
        "model_dir": str(model_dir),
        "method": {
            "explainer": "shap.Explainer with shap.maskers.Text",
            "algorithm": "partition",
            "output": "class-1 probability",
            "max_evals": max_evals,
            "device": str(device),
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
