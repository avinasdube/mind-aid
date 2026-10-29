"""Data preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
import re
import string

import pandas as pd

from .utils import ensure_parent_dir


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EXTRA_SPACES_PATTERN = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Normalize free-form text into a safe NLP model input string."""
    if text is None:
        return ""

    normalized = str(text)
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = normalized.lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = EXTRA_SPACES_PATTERN.sub(" ", normalized).strip()
    return normalized


def preprocess_dataset(input_path: Path, output_path: Path) -> None:
    """Load, clean, and save a text dataset."""
    df = pd.read_csv(input_path)

    # Normalize common source schemas to the canonical training schema.
    if "text" not in df.columns and "clean_text" in df.columns:
        df = df.rename(columns={"clean_text": "text"})
    if "label" not in df.columns and "is_depression" in df.columns:
        df = df.rename(columns={"is_depression": "label"})

    required_columns = {"text", "label"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

    cleaned = df.dropna(subset=["text", "label"]).copy()
    cleaned["text"] = cleaned["text"].apply(clean_text)
    cleaned = cleaned[cleaned["text"] != ""]

    ensure_parent_dir(output_path)
    cleaned.to_csv(output_path, index=False)
