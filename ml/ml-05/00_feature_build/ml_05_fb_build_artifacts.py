"""ML-05 feature build artifact assembly helpers.

The ML-05 input is expected to be the ML-04 feature parquet. This module
preserves all input columns and appends only newly generated Stage 4 features.

Code map:
- Input: validated split_df, FeatureSpec tuple, and build config metadata.
- Output: BuildArtifacts with appended feature_frame, split frames, and summary.
- Public: assemble_build_artifacts, append_generated_features, split_feature_frame.
- Leakage guard: preserves existing time split and refuses generated column overwrite.
- Notes: build artifacts stay in memory; encode_split_frame() writes final files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import pandas as pd

from ml_05_fb_io import utc_now_iso
from ml_05_fb_operations import execute_feature_specs
from ml_05_fb_specs import FeatureSpec, feature_columns


@dataclass(frozen=True)
class BuildArtifacts:
    """Internal artifact bundle returned by build assembly."""

    feature_frame: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_info: pd.DataFrame
    operation_artifacts: dict[str, pd.DataFrame]
    selected_feature_columns: list[str]
    row_counts: dict[str, int]
    build_summary: dict[str, Any]


def split_feature_frame(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the final feature frame into train/val/test frames."""

    train_df = feature_frame[feature_frame["split"] == "train"].reset_index(drop=True)
    val_df = feature_frame[feature_frame["split"] == "val"].reset_index(drop=True)
    test_df = feature_frame[feature_frame["split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Feature split output must not be empty. "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    return train_df, val_df, test_df


def preserve_source_columns(raw_df: pd.DataFrame, standardized_df: pd.DataFrame) -> pd.DataFrame:
    """Preserve all source columns and append canonical build columns."""

    output = raw_df.reset_index(drop=True).copy(deep=False)
    for column in standardized_df.columns:
        output[column] = standardized_df[column].reset_index(drop=True)
    return output


def append_generated_features(
    source_frame: pd.DataFrame,
    built_feature_frame: pd.DataFrame,
    generated_columns: list[str],
) -> pd.DataFrame:
    """Append generated ML-05 feature columns without overwriting inputs."""

    source = source_frame.reset_index(drop=True).copy(deep=False)
    built = built_feature_frame.reset_index(drop=True)
    if len(source) != len(built):
        raise ValueError(
            "Feature build failed: source and generated feature row counts differ. "
            f"source_rows={len(source)}, generated_rows={len(built)}"
        )

    for meta_col in ["tx_id", "timestamp", "split", "label"]:
        if meta_col not in source.columns or meta_col not in built.columns:
            raise ValueError(f"Feature build failed: metadata column is missing before append. column={meta_col!r}")
        left = source[meta_col].astype("string").reset_index(drop=True)
        right = built[meta_col].astype("string").reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"Feature build failed: metadata order mismatch before append. column={meta_col!r}")

    collisions = [column for column in generated_columns if column in source.columns]
    if collisions:
        raise ValueError(
            "Feature build refused to overwrite existing columns. "
            f"generated_columns_already_exist={collisions[:30]}, collision_count={len(collisions)}"
        )

    for column in generated_columns:
        if column not in built.columns:
            raise ValueError(f"Feature build failed: generated column is missing. column={column!r}")
        source[column] = built[column]
    source["label"] = built["label"].astype("int8")
    source["split"] = built["split"].astype("string")
    return source


def assemble_build_artifacts(
    split_df: pd.DataFrame,
    *,
    specs: Tuple[FeatureSpec, ...],
    config: Any,
    column_map: Mapping[str, str],
    input_label: Any,
    input_mode: str,
    sample_scan_summary: Mapping[str, Any] | None = None,
) -> BuildArtifacts:
    """Assemble generated feature frame, summaries, and operation artifacts."""

    generated_feature_frame, feature_info, operation_artifacts = execute_feature_specs(split_df, specs)
    selected_feature_columns = feature_columns(specs)
    feature_frame = append_generated_features(split_df, generated_feature_frame, selected_feature_columns)
    train_df, val_df, test_df = split_feature_frame(feature_frame)
    row_counts = feature_row_counts(feature_frame, train_df, val_df, test_df)
    build_summary = make_build_summary(
        config=config,
        column_map=column_map,
        input_label=input_label,
        input_mode=input_mode,
        sample_scan_summary=sample_scan_summary,
        selected_feature_columns=selected_feature_columns,
        specs=specs,
        operation_artifacts=operation_artifacts,
        row_counts=row_counts,
    )
    return BuildArtifacts(
        feature_frame=feature_frame,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_info=feature_info,
        operation_artifacts=operation_artifacts,
        selected_feature_columns=selected_feature_columns,
        row_counts=row_counts,
        build_summary=build_summary,
    )


def feature_row_counts(
    feature_frame: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    """Create shared row-count metadata."""

    return {
        "all": int(len(feature_frame)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }


def make_build_summary(
    *,
    config: Any,
    column_map: Mapping[str, str],
    input_label: Any,
    input_mode: str,
    sample_scan_summary: Mapping[str, Any] | None,
    selected_feature_columns: list[str],
    specs: Tuple[FeatureSpec, ...],
    operation_artifacts: Mapping[str, pd.DataFrame],
    row_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Create reproducibility metadata for the build result."""

    unknown_category_total = unknown_category_total_from_summary(operation_artifacts.get("category_unknown_summary", pd.DataFrame()))
    return {
        "created_at_utc": utc_now_iso(),
        "experiment_id": config.experiment_id,
        "run_name": config.run_name,
        "input_mode": input_mode,
        "input": input_label,
        "output_dir": None,
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,
        "sample_scan_summary": dict(sample_scan_summary) if sample_scan_summary is not None else None,
        "overwrite": config.overwrite,
        "preserve_source_columns": config.preserve_source_columns,
        "train_ratio": None,
        "val_ratio": None,
        "preserve_timestamp_groups": None,
        "configured_column_map": dict(config.column_map) if config.column_map is not None else None,
        "resolved_columns": dict(column_map),
        "feature_columns": selected_feature_columns,
        "feature_count": len(selected_feature_columns),
        "operations": [spec.operation for spec in specs],
        "unique_operations": sorted({spec.operation for spec in specs}),
        "unknown_category_total": unknown_category_total,
        "row_counts": dict(row_counts),
    }


def unknown_category_total_from_summary(unknown_summary: pd.DataFrame) -> int:
    """Sum unknown category counts from operation artifacts."""

    if unknown_summary.empty:
        return 0
    if "unknown_count" not in unknown_summary.columns:
        raise ValueError("category_unknown_summary artifact is missing unknown_count column.")
    return int(unknown_summary["unknown_count"].sum())
