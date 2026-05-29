"""ML-05 feature build entry points.

This module reads a split-aware ML-04 feature parquet or DataFrame, validates
the existing split, and appends only ML-05 Stage 4 flow-balance/pass-through
features.
It does not save files directly; export is handled by ``ml_05_fb_encoding``.

Code map:
- Input: ML-04 split-aware parquet or in-memory DataFrame.
- Output: FeatureBuildResult with in-memory feature_frame and metadata.
- Public: FeatureBuildConfig, FeatureBuildResult, build_features, build_features_from_frame.
- Leakage guard: validates existing time split and sorts by timestamp/tx_id before build.
- Notes: output_dir is rejected here; encode_split_frame() is the only save step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import pandas as pd

import ml_05_fb_build_validation as build_validation
from ml_05_fb_build_artifacts import assemble_build_artifacts, preserve_source_columns
from ml_05_fb_io import DEFAULT_INPUT_PATH, load_parquet_columns, load_parquet_split_sample, parquet_columns, resolve_path
from ml_05_fb_operations import SUPPORTED_BATCH_OPERATIONS
from ml_05_fb_schema import resolve_requested_columns, standardize_input_frame, validate_no_forbidden_input_columns
from ml_05_fb_specs import FeatureSpec, ml05_stage4_feature_specs, required_input_columns, validate_feature_specs


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Feature build execution config."""

    input_path: Optional[Union[str, Path]] = DEFAULT_INPUT_PATH
    output_dir: Optional[Union[str, Path]] = None
    base_dir: Optional[Union[str, Path]] = None
    experiment_id: str = "ML-05"
    run_name: str = "stage4_flowbalance_passflow_full48"
    feature_specs: Optional[Tuple[FeatureSpec, ...]] = None
    column_map: Optional[Mapping[str, str]] = None
    sample_rows: Optional[int] = None
    overwrite: bool = False
    preserve_source_columns: bool = True
    tx_id_col: str = "tx_id"
    timestamp_col: str = "timestamp"
    label_col: str = "label"

    def __post_init__(self) -> None:
        if self.input_path is not None:
            object.__setattr__(self, "input_path", resolve_path(self.input_path, self.base_dir))
        if self.output_dir is not None:
            raise ValueError(
                "FeatureBuildConfig.output_dir is no longer used. "
                "Run build_features() for in-memory feature creation, then save final artifacts with encode_split_frame()."
            )
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer or None.")
        if not str(self.experiment_id).strip():
            raise ValueError("experiment_id must not be empty.")
        if not str(self.run_name).strip():
            raise ValueError("run_name must not be empty.")

        if self.column_map is not None:
            cleaned_column_map: dict[str, str] = {}
            for logical_name, source_column in self.column_map.items():
                logical = str(logical_name).strip()
                source = str(source_column).strip()
                if not logical or not source:
                    raise ValueError(
                        "column_map keys and values must not be blank. "
                        f"logical_name={logical_name!r}, source_column={source_column!r}"
                    )
                if logical in cleaned_column_map:
                    raise ValueError(f"column_map has duplicated logical name after stripping: {logical!r}")
                cleaned_column_map[logical] = source
            object.__setattr__(self, "column_map", cleaned_column_map)


@dataclass(frozen=True)
class FeatureBuildResult:
    """In-memory feature build result."""

    output_paths: None
    feature_columns: list[str]
    row_counts: dict[str, int]
    build_summary: Mapping[str, Any]
    feature_frame: pd.DataFrame
    feature_info: pd.DataFrame


def build_features(config: Optional[FeatureBuildConfig] = None) -> FeatureBuildResult:
    """Build ML-05 Stage 4 features from a split-aware parquet input without saving files."""

    config = FeatureBuildConfig() if config is None else config
    if config.input_path is None:
        raise ValueError("input_path is required for build_features(). Use build_features_from_frame() for DataFrame input.")
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input parquet not found: {input_path}")

    specs = _require_feature_specs(config.feature_specs)
    _validate_specs_for_build(specs)
    requested_columns = required_input_columns(
        specs,
        extra_columns=[config.tx_id_col, config.timestamp_col, config.label_col, "split"],
    )
    source_columns = parquet_columns(input_path)
    column_map = resolve_requested_columns(source_columns, requested_columns, column_map=config.column_map)
    load_columns = _parquet_columns_for_build(
        source_columns,
        requested_columns=requested_columns,
        column_map=column_map,
        preserve_source_columns=config.preserve_source_columns,
    )

    sample_scan_summary: dict[str, Any] | None = None
    if config.sample_rows is None:
        raw_df = load_parquet_columns(input_path, load_columns, sample_rows=None)
    else:
        sample_scan_summary = {}
        raw_df = load_parquet_split_sample(
            input_path,
            load_columns,
            sample_rows=config.sample_rows,
            split_col="split",
            scan_summary=sample_scan_summary,
        )
    return _build_from_raw_frame(
        raw_df,
        column_map=column_map,
        config=config,
        input_label=str(input_path),
        input_mode="single_parquet" if config.preserve_source_columns else "single_parquet_minimal_columns",
        sample_scan_summary=sample_scan_summary,
    )


def build_features_from_frame(
    df: pd.DataFrame,
    *,
    feature_specs: Optional[Tuple[FeatureSpec, ...]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    experiment_id: str = "ML-05",
    run_name: str = "stage4_flowbalance_passflow_full48",
    column_map: Optional[Mapping[str, str]] = None,
    overwrite: bool = False,
    tx_id_col: str = "tx_id",
    timestamp_col: str = "timestamp",
    label_col: str = "label",
) -> FeatureBuildResult:
    """Build ML-05 Stage 4 features from an in-memory split-aware DataFrame without saving files."""

    config = FeatureBuildConfig(
        input_path=None,
        output_dir=output_dir,
        base_dir=base_dir,
        experiment_id=experiment_id,
        run_name=run_name,
        feature_specs=feature_specs,
        column_map=column_map,
        overwrite=overwrite,
        tx_id_col=tx_id_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
    )
    specs = _require_feature_specs(config.feature_specs)
    _validate_specs_for_build(specs)
    requested_columns = required_input_columns(
        specs,
        extra_columns=[config.tx_id_col, config.timestamp_col, config.label_col, "split"],
    )
    resolved_columns = resolve_requested_columns(df.columns, requested_columns, column_map=config.column_map)
    return _build_from_raw_frame(
        df,
        column_map=resolved_columns,
        config=config,
        input_label="dataframe",
        input_mode="dataframe",
        sample_scan_summary=None,
    )


def _validate_specs_for_build(specs: Tuple[FeatureSpec, ...]) -> None:
    """Validate specs and block feature inputs that look label-derived."""

    validate_feature_specs(specs)
    unsupported_specs = [
        {"output_col": spec.output_col, "operation": spec.operation}
        for spec in specs
        if spec.operation not in SUPPORTED_BATCH_OPERATIONS
    ]
    if unsupported_specs:
        raise ValueError(
            "Feature build failed: unsupported ML-05 feature operation. "
            f"unsupported_specs={unsupported_specs[:30]}, unsupported_count={len(unsupported_specs)}, "
            f"supported_operations={sorted(SUPPORTED_BATCH_OPERATIONS)}"
        )
    validate_no_forbidden_input_columns(column for spec in specs for column in spec.required_columns())


def _require_feature_specs(
    feature_specs: Optional[Tuple[FeatureSpec, ...]],
) -> Tuple[FeatureSpec, ...]:
    """Return explicit specs or the fixed ML-05 Stage 4 specs."""

    if feature_specs is None:
        return ml05_stage4_feature_specs()
    return feature_specs


def _validate_resolved_feature_source_columns(
    specs: Tuple[FeatureSpec, ...],
    resolved_columns: Mapping[str, str],
) -> None:
    """Validate resolved source columns for feature inputs only."""

    feature_input_columns = list(dict.fromkeys(column for spec in specs for column in spec.required_columns()))
    missing = [column for column in feature_input_columns if column not in resolved_columns]
    if missing:
        raise ValueError(
            "Feature build failed: resolved source columns are missing feature inputs. "
            f"missing={missing[:30]}, missing_count={len(missing)}"
        )
    validate_no_forbidden_input_columns(resolved_columns[column] for column in feature_input_columns)


def _parquet_columns_for_build(
    source_columns: list[str],
    *,
    requested_columns: list[str],
    column_map: Mapping[str, str],
    preserve_source_columns: bool,
) -> list[str]:
    """Return parquet columns needed by the current build mode."""

    if preserve_source_columns:
        return source_columns
    return list(dict.fromkeys(column_map[column] for column in requested_columns))


def _build_from_raw_frame(
    raw_df: pd.DataFrame,
    *,
    column_map: Mapping[str, str],
    config: FeatureBuildConfig,
    input_label: str,
    input_mode: str,
    sample_scan_summary: Mapping[str, Any] | None,
) -> FeatureBuildResult:
    """Standardize metadata and validate the existing split before computing features."""

    clean_df = standardize_input_frame(
        raw_df,
        column_map,
        tx_id_col=config.tx_id_col,
        timestamp_col=config.timestamp_col,
        label_col=config.label_col,
    )
    if config.preserve_source_columns:
        source_with_meta = preserve_source_columns(raw_df, clean_df)
    else:
        source_with_meta = clean_df.reset_index(drop=True).copy(deep=False)
    if "split" not in source_with_meta.columns:
        raise ValueError(
            "Feature build requires an existing split column in the input parquet/DataFrame. "
            "This ML-05 path does not create a new train/val/test split. "
            f"input={input_label}"
        )

    metadata = build_validation.existing_split_metadata_frame(
        source_with_meta,
        source_path=Path(input_label),
        tx_id_col="tx_id",
        timestamp_col="timestamp",
        label_col="label",
        split_col="split",
    )
    split_df = source_with_meta.copy(deep=False)
    split_df["tx_id"] = metadata["tx_id"]
    split_df["timestamp"] = metadata["timestamp"]
    split_df["label"] = metadata["label"]
    split_df["split"] = metadata["split"].astype("string")
    effective_input_mode = f"{input_mode}_existing_split"
    return _build_from_split_frame(
        split_df,
        column_map=column_map,
        config=config,
        input_label=input_label,
        input_mode=effective_input_mode,
        sample_scan_summary=sample_scan_summary,
    )


def _build_from_split_frame(
    split_df: pd.DataFrame,
    *,
    column_map: Mapping[str, str],
    config: FeatureBuildConfig,
    input_label: Any,
    input_mode: str,
    sample_scan_summary: Mapping[str, Any] | None,
) -> FeatureBuildResult:
    """Compute ML-05 Stage 4 features from a validated split frame."""

    specs = _require_feature_specs(config.feature_specs)
    _validate_resolved_feature_source_columns(specs, column_map)
    split_df = split_df.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    build_validation.validate_time_split(split_df)
    build_artifacts = assemble_build_artifacts(
        split_df,
        specs=specs,
        config=config,
        column_map=column_map,
        input_label=input_label,
        input_mode=input_mode,
        sample_scan_summary=sample_scan_summary,
    )
    return FeatureBuildResult(
        output_paths=None,
        feature_columns=build_artifacts.selected_feature_columns,
        row_counts=build_artifacts.row_counts,
        build_summary=build_artifacts.build_summary,
        feature_frame=build_artifacts.feature_frame,
        feature_info=build_artifacts.feature_info,
    )
