"""ML-06 feature-build schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


META_COLUMNS: tuple[str, ...] = ("tx_id", "timestamp", "split", "label")
REQUIRED_SPLITS: tuple[str, ...] = ("train", "val", "test")


def validate_required_columns(df: pd.DataFrame, columns: tuple[str, ...] | list[str], *, context: str) -> None:
    """Fail when required columns are missing."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def normalize_split_column(df: pd.DataFrame) -> pd.Series:
    """Return normalized split values after validation."""

    validate_required_columns(df, ["split"], context="ML-06 input")
    if df["split"].isna().any():
        raise ValueError(f"split column has missing values: missing_count={int(df['split'].isna().sum())}")
    split = df["split"].astype("string").str.strip().str.lower()
    invalid = sorted(split.loc[~split.isin(REQUIRED_SPLITS)].dropna().unique().tolist())
    if invalid:
        raise ValueError(f"split column has unsupported values: {invalid}")
    return split


def validate_frame_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the core ML input contract."""

    validate_required_columns(df, list(META_COLUMNS), context="ML-06 input")
    out = df.copy(deep=False)
    out["split"] = normalize_split_column(out)

    if out["tx_id"].isna().any():
        raise ValueError(f"tx_id column has missing values: missing_count={int(out['tx_id'].isna().sum())}")
    duplicated = out["tx_id"].astype("string").duplicated(keep=False)
    if bool(duplicated.any()):
        examples = out.loc[duplicated, "tx_id"].astype(str).head(10).tolist()
        raise ValueError(f"tx_id values are duplicated: duplicated_count={int(duplicated.sum())}, examples={examples}")

    timestamp = pd.to_datetime(out["timestamp"], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"timestamp parsing failed: failed_count={int(timestamp.isna().sum())}")
    out["timestamp"] = timestamp

    label = pd.to_numeric(out["label"], errors="coerce")
    if label.isna().any():
        raise ValueError(f"label parsing failed: failed_count={int(label.isna().sum())}")
    observed = sorted(label.dropna().unique().tolist())
    if not set(observed).issubset({0, 1}):
        raise ValueError(f"label must be binary 0/1: observed={observed[:20]}")
    out["label"] = label.astype("int8")

    split_values = set(out["split"].astype(str).unique().tolist())
    missing_splits = sorted(set(REQUIRED_SPLITS) - split_values)
    if missing_splits:
        raise ValueError(f"input is missing required split values: {missing_splits}")
    return out


def numeric_values(df: pd.DataFrame, column: str, *, context: str) -> np.ndarray:
    """Return a finite float64 numpy array for a numeric column."""

    validate_required_columns(df, [column], context=context)
    series = pd.to_numeric(df[column], errors="coerce")
    values = series.to_numpy(dtype="float64", copy=False)
    finite = np.isfinite(values)
    if not bool(finite.all()):
        bad_count = int((~finite).sum())
        raise ValueError(f"{context} column has NaN or inf values: column={column!r}, bad_count={bad_count}")
    return values
