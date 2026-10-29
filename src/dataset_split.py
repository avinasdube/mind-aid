"""Dataset splitting helpers for paper-ready training and validation workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import clean_text
from .utils import ensure_parent_dir


def split_labeled_dataset(
    input_path: Path,
    train_output_path: Path,
    validation_output_path: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, int | str]:
    """Split a labeled text dataset into train and unseen validation CSV files."""
    df = pd.read_csv(input_path)

    required_columns = {"text", "label"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Dataset must contain columns: {missing_columns}")

    cleaned = df.dropna(subset=["text", "label"]).copy()
    if cleaned.empty:
        raise ValueError("No valid rows remain after removing missing values.")

    input_rows = len(cleaned)
    cleaned["_normalized_text"] = cleaned["text"].map(clean_text)
    cleaned = cleaned[cleaned["_normalized_text"] != ""].drop_duplicates(
        subset=["_normalized_text"],
        keep="first",
    )
    removed_duplicate_rows = input_rows - len(cleaned)
    cleaned = cleaned.drop(columns=["_normalized_text"])
    if cleaned.empty:
        raise ValueError("No valid rows remain after duplicate removal.")

    try:
        train_df, validation_df = train_test_split(
            cleaned,
            test_size=test_size,
            random_state=random_state,
            stratify=cleaned["label"],
        )
    except ValueError:
        train_df, validation_df = train_test_split(
            cleaned,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    ensure_parent_dir(train_output_path)
    ensure_parent_dir(validation_output_path)
    train_df.to_csv(train_output_path, index=False)
    validation_df.to_csv(validation_output_path, index=False)

    return {
        "input_rows": int(input_rows),
        "removed_duplicate_rows": int(removed_duplicate_rows),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "train_output": str(train_output_path),
        "validation_output": str(validation_output_path),
    }