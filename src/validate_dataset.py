"""Validation utilities for external evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .preprocess import clean_text
from .utils import ensure_parent_dir


TEXT_COLUMNS = ("text", "clean_text", "content", "sentence", "body")
LABEL_COLUMNS = ("label", "is_depression", "target", "y")
IMAGE_COLUMNS = ("image_path", "image", "photo_path", "filepath")


def _first_existing_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _safe_label_distribution(series: pd.Series) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for label, count in series.value_counts(dropna=False).items():
        distribution[str(label)] = int(count)
    return distribution


def _clean_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).map(clean_text)


def validate_external_dataset(
    external_data_path: Path,
    *,
    training_data_path: Path | None = None,
    output_path: Path = Path("outputs/reports/external_dataset_validation.json"),
    require_images: bool = False,
) -> dict[str, Any]:
    """Validate external dataset schema, quality, overlap risk, and image readiness."""
    external_df = pd.read_csv(external_data_path)
    external_columns = list(external_df.columns)

    text_column = _first_existing_column(external_columns, TEXT_COLUMNS)
    label_column = _first_existing_column(external_columns, LABEL_COLUMNS)
    image_column = _first_existing_column(external_columns, IMAGE_COLUMNS)

    errors: list[str] = []
    warnings: list[str] = []

    if text_column is None:
        errors.append("Missing text column. Expected one of: text, clean_text, content, sentence, body.")
    if label_column is None:
        errors.append("Missing label column. Expected one of: label, is_depression, target, y.")

    if errors:
        report = {
            "status": "error",
            "external_data_path": str(external_data_path),
            "errors": errors,
            "warnings": warnings,
            "detected_columns": external_columns,
        }
        ensure_parent_dir(output_path)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        return report

    cleaned_external = external_df.dropna(subset=[text_column, label_column]).copy()
    cleaned_external["_clean_text"] = _clean_text_series(cleaned_external[text_column])
    cleaned_external = cleaned_external[cleaned_external["_clean_text"] != ""]

    duplicate_count = int(cleaned_external["_clean_text"].duplicated().sum())
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate text rows in external dataset.")

    overlap_count = 0
    overlap_ratio = 0.0
    training_duplicate_count = 0
    training_rows = 0
    if training_data_path is not None and training_data_path.exists():
        train_df = pd.read_csv(training_data_path)
        train_columns = list(train_df.columns)
        train_text_column = _first_existing_column(train_columns, TEXT_COLUMNS)

        if train_text_column is None:
            warnings.append("Training dataset has no recognized text column; overlap check skipped.")
        else:
            train_clean = _clean_text_series(train_df[train_text_column].dropna())
            train_clean = train_clean[train_clean != ""]
            training_rows = int(len(train_clean))
            training_duplicate_count = int(train_clean.duplicated().sum())
            train_set = set(train_clean.tolist())
            overlap_series = cleaned_external["_clean_text"].isin(train_set)
            overlap_count = int(overlap_series.sum())
            total = int(len(cleaned_external))
            overlap_ratio = float(overlap_count / total) if total > 0 else 0.0
            if overlap_count > 0:
                warnings.append(
                    f"Detected {overlap_count} overlapping text rows with training data ({overlap_ratio:.2%})."
                )

    image_rows = 0
    existing_image_rows = 0
    if image_column is None:
        if require_images:
            errors.append("Image column is required for multimodal benchmarking but was not found.")
        else:
            warnings.append("No image path column found. Fusion ablations will be skipped.")
    else:
        paths = cleaned_external[image_column].fillna("").astype(str).str.strip()
        image_rows = int((paths != "").sum())
        if image_rows == 0:
            warnings.append("Image path column exists but has no populated rows.")
            if require_images:
                errors.append("Image paths are required for multimodal benchmarking but none were provided.")
        else:
            base_dir = external_data_path.parent

            def _exists(path_str: str) -> bool:
                candidate = Path(path_str)
                if not candidate.is_absolute():
                    candidate = base_dir / candidate
                return candidate.exists()

            existing_image_rows = int(paths[paths != ""].map(_exists).sum())
            if existing_image_rows < image_rows:
                missing = image_rows - existing_image_rows
                warnings.append(f"Found {missing} image paths that do not exist on disk.")
                if require_images:
                    errors.append("Some required image paths are missing on disk.")

    status = "ok"
    if errors:
        status = "error"
    elif warnings:
        status = "warning"

    report: dict[str, Any] = {
        "status": status,
        "external_data_path": str(external_data_path),
        "training_data_path": str(training_data_path) if training_data_path is not None else "",
        "schema": {
            "text_column": text_column,
            "label_column": label_column,
            "image_column": image_column or "",
        },
        "row_counts": {
            "raw_rows": int(len(external_df)),
            "valid_rows": int(len(cleaned_external)),
            "duplicate_text_rows": duplicate_count,
        },
        "label_distribution": _safe_label_distribution(cleaned_external[label_column]),
        "overlap_check": {
            "overlap_rows": overlap_count,
            "overlap_ratio": overlap_ratio,
        },
        "training_data_quality": {
            "valid_rows": training_rows,
            "duplicate_text_rows": training_duplicate_count,
            "duplicates_removed_before_model_fit": True,
        },
        "image_readiness": {
            "image_rows": image_rows,
            "existing_image_rows": existing_image_rows,
            "require_images": require_images,
        },
        "warnings": warnings,
        "errors": errors,
    }

    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report