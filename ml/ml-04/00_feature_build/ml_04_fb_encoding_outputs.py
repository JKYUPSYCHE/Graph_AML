"""Output path and export validation helpers for ML-04 FB encoding.

Code map:
- Input: output directory, artifact prefix, and saved parquet/CSV/JSON paths.
- Output: EncodingOutputPaths and post-save validation errors.
- Public: make_encoding_output_paths, require_no_existing_encoding_outputs, validate_encoding_outputs.
- Leakage guard: selected and materialized non-meta feature names are checked after save.
- Notes: parquet split files must match Xy_all schema and row-count total.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Union

import pandas as pd

from ml_04_fb_io import parquet_columns, parquet_row_count, parquet_schema_types, resolve_path
from ml_04_fb_schema import META_COLUMNS, validate_no_forbidden_feature_columns


OUTPUT_PATH_FIELDS = (
    ("all", "all_path"),
    ("train", "train_path"),
    ("val", "val_path"),
    ("test", "test_path"),
    ("feature_contract", "feature_contract_path"),
    ("encoding_manifest", "encoding_manifest_path"),
    ("feature_types", "feature_types_path"),
    ("category_mapping", "category_mapping_path"),
    ("category_unknown_summary", "category_unknown_summary_path"),
    ("split_summary", "split_summary_path"),
)
SPLIT_PATH_FIELDS = (("train", "train_path"), ("val", "val_path"), ("test", "test_path"))


@dataclass(frozen=True)
class EncodingOutputPaths:
    """Encoding/export output paths."""

    output_dir: Path
    all_path: Path
    train_path: Path
    val_path: Path
    test_path: Path
    feature_contract_path: Path
    encoding_manifest_path: Path
    feature_types_path: Path
    category_mapping_path: Path
    category_unknown_summary_path: Path
    split_summary_path: Path


def make_encoding_output_paths(output_dir: Union[str, Path], artifact_prefix: str) -> EncodingOutputPaths:
    """Create output paths for ML-04 encoding/export artifacts."""

    prefix = str(artifact_prefix).strip()
    if not prefix:
        raise ValueError("artifact_prefix must not be empty.")
    base = resolve_path(output_dir)
    return EncodingOutputPaths(
        output_dir=base,
        all_path=base / f"{prefix}_Xy_all.parquet",
        train_path=base / f"{prefix}_Xy_train.parquet",
        val_path=base / f"{prefix}_Xy_val.parquet",
        test_path=base / f"{prefix}_Xy_test.parquet",
        feature_contract_path=base / f"{prefix}_fb_output_feature_contract.csv",
        encoding_manifest_path=base / f"{prefix}_encoding_manifest.json",
        feature_types_path=base / f"{prefix}_feature_types.json",
        category_mapping_path=base / f"{prefix}_category_mapping_train_only.csv",
        category_unknown_summary_path=base / f"{prefix}_category_unknown_summary.csv",
        split_summary_path=base / f"{prefix}_split_summary.csv",
    )


def output_paths_by_name(paths: EncodingOutputPaths) -> dict[str, Path]:
    """Return required encoding output paths keyed by artifact role."""

    return {name: getattr(paths, field_name) for name, field_name in OUTPUT_PATH_FIELDS}


def require_no_existing_encoding_outputs(paths: EncodingOutputPaths, overwrite: bool) -> None:
    """Fail on existing artifacts unless overwrite=True."""

    existing = [str(path) for path in output_paths_by_name(paths).values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Existing encoding artifacts found. Set overwrite=True or change RUN_ID. existing={existing}")


def validate_category_manifest_values(
    *,
    feature_columns: list[str],
    feature_types: Mapping[str, str],
    category_values: Mapping[str, list[str]],
) -> list[str]:
    """Validate category values in the manifest and return categorical columns."""

    categorical_columns = [column for column in feature_columns if feature_types[column] == "c"]
    missing_category_values = [column for column in categorical_columns if column not in category_values]
    if missing_category_values:
        raise ValueError(f"encoding manifest category_values is missing categorical features: {missing_category_values[:30]}")
    extra_category_values = sorted(set(category_values) - set(categorical_columns))
    if extra_category_values:
        raise ValueError(f"encoding manifest category_values contains non-categorical features: {extra_category_values[:30]}")
    return categorical_columns


def split_encoded_frames(all_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split Xy_all into train/val/test while preserving column order and dtypes."""

    split_frames = {split: all_df.loc[all_df["split"] == split].reset_index(drop=True) for split, _field in SPLIT_PATH_FIELDS}
    train_df, val_df, test_df = split_frames["train"], split_frames["val"], split_frames["test"]
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"encoded split output must not be empty. train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    reference_columns = list(all_df.columns)
    reference_dtypes = {column: str(all_df[column].dtype) for column in reference_columns}
    for split_name, split_frame in split_frames.items():
        if list(split_frame.columns) != reference_columns:
            raise ValueError(f"{split_name} columns do not match Xy_all columns.")
        dtype_mismatch = {
            column: {"all": reference_dtypes[column], "split": str(split_frame[column].dtype)}
            for column in reference_columns
            if reference_dtypes[column] != str(split_frame[column].dtype)
        }
        if dtype_mismatch:
            raise ValueError(f"{split_name} dtypes do not match Xy_all dtypes: {dict(list(dtype_mismatch.items())[:30])}")
    return split_frames


def _duplicated_names(names: list[str]) -> list[str]:
    return sorted({name for name in names if names.count(name) > 1})


def validate_encoding_outputs(
    paths: EncodingOutputPaths,
    *,
    feature_columns: list[str],
    materialized_columns: list[str],
    feature_types: Mapping[str, str],
) -> None:
    """Validate files and core encoding/export contracts after save."""

    missing_files = {name: str(path) for name, path in output_paths_by_name(paths).items() if not path.is_file()}
    if missing_files:
        raise FileNotFoundError(f"encoding export did not create required files: {missing_files}")
    if not feature_columns:
        raise ValueError("encoding export produced no feature columns.")
    if not materialized_columns:
        raise ValueError("encoding export produced no materialized columns.")
    duplicated = _duplicated_names(feature_columns)
    if duplicated:
        raise ValueError(f"encoding export produced duplicated feature columns: {duplicated}")
    duplicated_materialized = _duplicated_names(materialized_columns)
    if duplicated_materialized:
        raise ValueError(f"encoding export produced duplicated materialized columns: {duplicated_materialized}")
    validate_no_forbidden_feature_columns(feature_columns)
    missing_feature_types = [column for column in feature_columns if column not in feature_types]
    if missing_feature_types:
        raise ValueError(f"feature_types is missing exported features: {missing_feature_types[:30]}")

    expected_columns = set(META_COLUMNS) | set(materialized_columns)
    all_columns = parquet_columns(paths.all_path)
    missing_all_columns = sorted(expected_columns - set(all_columns))
    if missing_all_columns:
        raise ValueError(f"encoded all parquet is missing required columns: {missing_all_columns[:30]}")
    extra_all_columns = sorted(set(all_columns) - expected_columns)
    if extra_all_columns:
        raise ValueError(f"encoded all parquet contains columns outside the materialized contract: {extra_all_columns[:30]}")
    all_types = parquet_schema_types(paths.all_path)
    split_row_total = 0
    for split_name, split_field in SPLIT_PATH_FIELDS:
        split_path = getattr(paths, split_field)
        split_columns_ordered = parquet_columns(split_path)
        if split_columns_ordered != all_columns:
            raise ValueError(f"encoded split parquet columns do not match Xy_all columns. split={split_name}, path={split_path}")
        split_types = parquet_schema_types(split_path)
        mismatched_types = {
            column: {"all": all_types.get(column), "split": split_types.get(column)}
            for column in all_columns
            if all_types.get(column) != split_types.get(column)
        }
        if mismatched_types:
            raise ValueError(f"encoded split parquet schema types do not match Xy_all schema types. split={split_name}, mismatched={dict(list(mismatched_types.items())[:30])}")
        split_row_total += parquet_row_count(split_path)
    all_row_count = parquet_row_count(paths.all_path)
    if all_row_count != split_row_total:
        raise ValueError(f"encoded Xy_all row count does not equal train+val+test row count. all={all_row_count}, split_total={split_row_total}")
