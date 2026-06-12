"""ML-02 feature build 검증 helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_unique_tx_ids(df: pd.DataFrame) -> None:
    """train/val/test 전체 합본에서 tx_id가 중복되지 않는지 확인한다."""

    duplicated = df["tx_id"].astype("string").duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "tx_id"].astype(str).head(10).tolist()
        raise ValueError(
            "Feature build failed: tx_id values are duplicated across split files. "
            f"duplicated_count={int(duplicated.sum())}, examples={examples}"
        )


def validate_time_split(df: pd.DataFrame) -> None:
    """split 결과가 train < val < test 시간 순서를 만족하는지 검사한다."""

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
    """기존 split 컬럼 값을 train/val/test canonical 값으로 정규화하고 검증한다."""

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
    """split-only 검증에 필요한 canonical metadata frame을 만든다."""

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
    if raw_timestamp.isna().any():
        raise ValueError(
            "timestamp column has missing values. "
            f"path={source_path}, timestamp_col={timestamp_col!r}, missing_count={int(raw_timestamp.isna().sum())}"
        )
    timestamp = pd.to_datetime(raw_timestamp, errors="coerce")
    if timestamp.isna().any():
        failed = raw_timestamp.loc[timestamp.isna()].astype(str).head(5).tolist()
        raise ValueError(
            "timestamp parsing failed for existing split validation. "
            f"path={source_path}, timestamp_col={timestamp_col!r}, "
            f"failed_count={int(timestamp.isna().sum())}, example_values={failed}"
        )

    raw_label = df[label_col]
    if raw_label.isna().any():
        raise ValueError(
            "label column has missing values. "
            f"path={source_path}, label_col={label_col!r}, missing_count={int(raw_label.isna().sum())}"
        )
    label = pd.to_numeric(raw_label, errors="coerce")
    if label.isna().any():
        failed = raw_label.loc[label.isna()].astype(str).head(5).tolist()
        raise ValueError(
            "label parsing failed for existing split validation. "
            f"path={source_path}, label_col={label_col!r}, "
            f"failed_count={int(label.isna().sum())}, example_values={failed}"
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
