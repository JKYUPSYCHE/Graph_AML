"""Validation helpers for ML-05 feature build inputs.

Code map:
- Input: split-aware DataFrame metadata columns.
- Output: canonical metadata frame for tx_id/timestamp/label/split.
- Public: existing_split_metadata_frame, validate_time_split, validate_unique_tx_ids.
- Leakage guard: train max timestamp must be before val, and val before test.
- Notes: ML-05 consumes an existing split and never creates a new split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_05_fb_schema import parse_datetime_series_strict, parse_numeric_series_strict


def validate_unique_tx_ids(df: pd.DataFrame) -> None:
    """Ensure tx_id is unique across the combined split frame."""

    duplicated = df["tx_id"].astype("string").duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "tx_id"].astype(str).head(10).tolist()
        raise ValueError(
            "Feature build failed: tx_id values are duplicated across split files. "
            f"duplicated_count={int(duplicated.sum())}, examples={examples}"
        )


def validate_time_split(df: pd.DataFrame) -> None:
    """Ensure train < val < test temporal split ordering."""

    counts = df["split"].value_counts().to_dict()
    missing = {"train", "val", "test"} - set(counts)
    if missing:
        raise ValueError(f"Missing required split values in existing split column: {sorted(missing)}")

    train_max = df.loc[df["split"] == "train", "timestamp"].max()
    val_min = df.loc[df["split"] == "val", "timestamp"].min()
    val_max = df.loc[df["split"] == "val", "timestamp"].max()
    test_min = df.loc[df["split"] == "test", "timestamp"].min()
    if train_max >= val_min:
        raise ValueError(f"Time split boundary violation: train_max={train_max}, val_min={val_min}")
    if val_max >= test_min:
        raise ValueError(f"Time split boundary violation: val_max={val_max}, test_min={test_min}")


def normalize_existing_split_values(series: pd.Series, *, source_path: Path, split_col: str) -> pd.Series:
    """Normalize and validate existing split values."""

    if series.isna().any():
        raise ValueError(
            "Existing split column has missing values. "
            f"path={source_path}, split_col={split_col!r}, missing_count={int(series.isna().sum())}"
        )
    normalized = series.astype("string").str.strip().str.lower()
    blank_mask = normalized == ""
    if blank_mask.any():
        raise ValueError(
            "Existing split column has blank values. "
            f"path={source_path}, split_col={split_col!r}, blank_count={int(blank_mask.sum())}"
        )
    allowed = {"train", "val", "test"}
    invalid = normalized[~normalized.isin(allowed)]
    if not invalid.empty:
        raise ValueError(
            "Existing split column has unsupported values. "
            f"path={source_path}, split_col={split_col!r}, allowed={sorted(allowed)}, "
            f"observed_examples={sorted(invalid.unique().tolist())[:20]}"
        )
    return normalized


def existing_split_metadata_frame(
    df: pd.DataFrame,
    *,
    source_path: Path,
    tx_id_col: str,
    timestamp_col: str,
    label_col: str,
    split_col: str,
) -> pd.DataFrame:
    """Build and validate canonical split metadata."""

    required = {tx_id_col, timestamp_col, label_col, split_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Single parquet input is missing columns required for existing split validation. "
            f"path={source_path}, missing={sorted(missing)}"
        )

    tx_id = df[tx_id_col]
    if tx_id.isna().any():
        raise ValueError(
            "tx_id column has missing values. "
            f"path={source_path}, tx_id_col={tx_id_col!r}, missing_count={int(tx_id.isna().sum())}"
        )

    raw_timestamp = df[timestamp_col]
    timestamp = parse_datetime_series_strict(
        raw_timestamp,
        missing_message=lambda missing_count: (
            "timestamp column has missing values. "
            f"path={source_path}, timestamp_col={timestamp_col!r}, missing_count={missing_count}"
        ),
        failed_message=lambda failed_count, examples: (
            "timestamp parsing failed for existing split validation. "
            f"path={source_path}, timestamp_col={timestamp_col!r}, failed_count={failed_count}, "
            f"example_values={examples}"
        ),
    )

    raw_label = df[label_col]
    label = parse_numeric_series_strict(
        raw_label,
        missing_message=lambda missing_count: (
            "label column has missing values. "
            f"path={source_path}, label_col={label_col!r}, missing_count={missing_count}"
        ),
        failed_message=lambda failed_count, examples: (
            "label parsing failed for existing split validation. "
            f"path={source_path}, label_col={label_col!r}, failed_count={failed_count}, "
            f"example_values={examples}"
        ),
    )
    label_values = sorted(label.dropna().unique().tolist())
    if not set(label_values).issubset({0, 1}):
        raise ValueError(
            "label must be binary 0/1 for existing split validation. "
            f"path={source_path}, label_col={label_col!r}, observed_values={label_values[:20]}"
        )

    metadata = pd.DataFrame(
        {
            "tx_id": tx_id.reset_index(drop=True),
            "timestamp": timestamp.reset_index(drop=True),
            "label": label.astype("int8").reset_index(drop=True),
            "split": normalize_existing_split_values(df[split_col], source_path=source_path, split_col=split_col).reset_index(drop=True),
        }
    )
    validate_unique_tx_ids(metadata)
    validate_time_split(metadata)
    return metadata
