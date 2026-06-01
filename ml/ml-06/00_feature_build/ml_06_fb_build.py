"""ML-06 feature build entry point.

This module reads ML-05 r00 features, applies ML-06 preprocessing policies, and
exports ML-06 r00 artifacts. It never modifies ML-05 artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml_06_fb_catalog import (
    build_contract_variants,
    build_encoding_manifest,
    build_feature_types,
    source_feature_columns_for_variants,
    update_contract_materialization,
    validate_input_contract,
)
from ml_06_fb_io import (
    DEFAULT_FB_OUTPUT_DIR,
    DEFAULT_FEATURE_COLUMNS_PATH,
    DEFAULT_INPUT_CONTRACT_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_ML_INPUT_DIR,
    DEFAULT_SOURCE_ENCODING_MANIFEST_PATH,
    DEFAULT_SOURCE_FEATURE_TYPES_PATH,
    load_parquet_frame,
    require_no_existing_outputs,
    resolve_path,
    save_dataframe_csv,
    save_dataframe_parquet,
    write_json,
)
from ml_06_fb_operations import apply_ratio_transforms, apply_recency_sentinel_policy
from ml_06_fb_schema import validate_frame_contract


@dataclass(frozen=True)
class FeatureBuildConfig:
    """ML-06 feature-build configuration."""

    input_path: str | Path = DEFAULT_INPUT_PATH
    source_contract_path: str | Path = DEFAULT_INPUT_CONTRACT_PATH
    feature_columns_path: str | Path = DEFAULT_FEATURE_COLUMNS_PATH
    source_feature_types_path: str | Path | None = DEFAULT_SOURCE_FEATURE_TYPES_PATH
    source_encoding_manifest_path: str | Path | None = DEFAULT_SOURCE_ENCODING_MANIFEST_PATH
    fb_output_dir: str | Path = DEFAULT_FB_OUTPUT_DIR
    ml_input_dir: str | Path = DEFAULT_ML_INPUT_DIR
    artifact_prefix: str = "ml_06__r00"
    sample_rows: int | None = None
    overwrite: bool = False

    def resolved(self) -> "FeatureBuildConfig":
        """Return a config with normalized paths."""

        return FeatureBuildConfig(
            input_path=resolve_path(self.input_path),
            source_contract_path=resolve_path(self.source_contract_path),
            feature_columns_path=resolve_path(self.feature_columns_path),
            source_feature_types_path=resolve_path(self.source_feature_types_path) if self.source_feature_types_path else None,
            source_encoding_manifest_path=resolve_path(self.source_encoding_manifest_path) if self.source_encoding_manifest_path else None,
            fb_output_dir=resolve_path(self.fb_output_dir),
            ml_input_dir=resolve_path(self.ml_input_dir),
            artifact_prefix=self.artifact_prefix,
            sample_rows=self.sample_rows,
            overwrite=self.overwrite,
        )


@dataclass(frozen=True)
class FeatureBuildResult:
    """ML-06 feature-build output summary."""

    output_paths: dict[str, str]
    row_counts: dict[str, int]
    generated_columns: list[str]
    base_ratio_columns: list[str]
    recency_report: pd.DataFrame
    ratio_reuse_report: pd.DataFrame
    ratio_manifest: dict[str, Any]


def _artifact_path(directory: Path, prefix: str, suffix: str) -> Path:
    return directory / f"{prefix}_{suffix}"


def _split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {"all": df.reset_index(drop=True)}
    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name].reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"ML-06 output has no rows for required split: {split_name}")
        frames[split_name] = split_df
    return frames


def _row_counts(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {name: int(len(frame)) for name, frame in frames.items()}


def _output_paths(config: FeatureBuildConfig) -> dict[str, Path]:
    fb_dir = Path(config.fb_output_dir)
    ml_dir = Path(config.ml_input_dir)
    prefix = config.artifact_prefix
    return {
        "fb_all": _artifact_path(fb_dir, prefix, "Xy_all.parquet"),
        "fb_train": _artifact_path(fb_dir, prefix, "Xy_train.parquet"),
        "fb_val": _artifact_path(fb_dir, prefix, "Xy_val.parquet"),
        "fb_test": _artifact_path(fb_dir, prefix, "Xy_test.parquet"),
        "ml_all": _artifact_path(ml_dir, prefix, "Xy_all.parquet"),
        "ml_train": _artifact_path(ml_dir, prefix, "Xy_train.parquet"),
        "ml_val": _artifact_path(ml_dir, prefix, "Xy_val.parquet"),
        "ml_test": _artifact_path(ml_dir, prefix, "Xy_test.parquet"),
        "recency_report": fb_dir / "recency_sentinel_report.csv",
        "ratio_manifest": fb_dir / "ratio_transform_manifest.json",
        "ratio_reuse_report": fb_dir / "ratio_transform_reuse_report.csv",
        "contract_approve": _artifact_path(ml_dir, prefix, "fb_output_feature_contract_approve.csv"),
        "contract_original": _artifact_path(ml_dir, prefix, "fb_output_feature_contract_ratio_original.csv"),
        "contract_log1p": _artifact_path(ml_dir, prefix, "fb_output_feature_contract_ratio_log1p.csv"),
        "contract_clip": _artifact_path(ml_dir, prefix, "fb_output_feature_contract_ratio_clip_p9999.csv"),
        "feature_types": _artifact_path(ml_dir, prefix, "feature_types.json"),
        "encoding_manifest": _artifact_path(ml_dir, prefix, "encoding_manifest.json"),
    }


def _save_split_outputs(frames: dict[str, pd.DataFrame], paths: dict[str, Path], *, overwrite: bool) -> None:
    for split_name in ("all", "train", "val", "test"):
        save_dataframe_parquet(frames[split_name], paths[f"fb_{split_name}"], overwrite=overwrite)
        save_dataframe_parquet(frames[split_name], paths[f"ml_{split_name}"], overwrite=overwrite)


def build_features(config: FeatureBuildConfig | None = None) -> FeatureBuildResult:
    """Run the ML-06 recency and ratio preprocessing build."""

    resolved = (FeatureBuildConfig() if config is None else config).resolved()
    paths = _output_paths(resolved)
    require_no_existing_outputs(paths.values(), overwrite=resolved.overwrite)

    raw_df = load_parquet_frame(resolved.input_path, sample_rows=resolved.sample_rows)
    source_columns = list(raw_df.columns)
    input_contract = pd.read_csv(resolved.source_contract_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})
    contract_validation = validate_input_contract(
        input_contract,
        artifact_prefix=resolved.artifact_prefix,
        available_columns=source_columns,
    )
    split_df = validate_frame_contract(raw_df)
    recency_df, recency_report = apply_recency_sentinel_policy(split_df)

    feature_df, ratio_manifest, ratio_reuse_report, ratio_specs = apply_ratio_transforms(
        recency_df,
        feature_columns=contract_validation.selected_columns,
        base_ratio_columns=contract_validation.ratio_base_columns,
    )
    frames = _split_frames(feature_df)
    row_counts = _row_counts(frames)

    contract_variants = build_contract_variants(
        input_contract,
        feature_columns=source_feature_columns_for_variants(input_contract),
        ratio_specs=ratio_specs,
        artifact_prefix=resolved.artifact_prefix,
    )
    contract_approve = update_contract_materialization(input_contract, feature_df)
    contract_original = update_contract_materialization(contract_variants["original"], feature_df)
    contract_log1p = update_contract_materialization(contract_variants["log1p"], feature_df)
    contract_clip = update_contract_materialization(contract_variants["clip_p9999"], feature_df)
    generated_columns = []
    for spec in ratio_specs:
        for column in (spec.log1p_column, spec.clip_column):
            if column in feature_df.columns:
                generated_columns.append(column)
    generated_columns = list(dict.fromkeys(generated_columns))

    feature_types = build_feature_types(
        source_feature_types_path=str(resolved.source_feature_types_path) if resolved.source_feature_types_path else None,
        generated_columns=generated_columns,
    )
    encoding_manifest = build_encoding_manifest(
        input_path=Path(resolved.input_path),
        source_contract_path=Path(resolved.source_contract_path),
        feature_columns_path=Path(resolved.feature_columns_path),
        source_encoding_manifest_path=Path(resolved.source_encoding_manifest_path) if resolved.source_encoding_manifest_path else None,
        ratio_manifest=ratio_manifest,
        generated_columns=generated_columns,
        row_counts=row_counts,
        artifact_prefix=resolved.artifact_prefix,
        output_dir=Path(resolved.ml_input_dir),
        output_paths=paths,
        feature_columns=contract_validation.selected_columns,
        materialized_columns=list(feature_df.columns),
    )

    _save_split_outputs(frames, paths, overwrite=resolved.overwrite)
    save_dataframe_csv(recency_report, paths["recency_report"], overwrite=resolved.overwrite)
    save_dataframe_csv(ratio_reuse_report, paths["ratio_reuse_report"], overwrite=resolved.overwrite)
    write_json(paths["ratio_manifest"], ratio_manifest, overwrite=resolved.overwrite)

    save_dataframe_csv(contract_original, paths["contract_original"], overwrite=resolved.overwrite)
    save_dataframe_csv(contract_approve, paths["contract_approve"], overwrite=resolved.overwrite)
    save_dataframe_csv(contract_log1p, paths["contract_log1p"], overwrite=resolved.overwrite)
    save_dataframe_csv(contract_clip, paths["contract_clip"], overwrite=resolved.overwrite)
    write_json(paths["feature_types"], feature_types, overwrite=resolved.overwrite)
    write_json(paths["encoding_manifest"], encoding_manifest, overwrite=resolved.overwrite)

    return FeatureBuildResult(
        output_paths={name: str(path) for name, path in paths.items()},
        row_counts=row_counts,
        generated_columns=generated_columns,
        base_ratio_columns=list(ratio_manifest["base_ratio_columns"]),
        recency_report=recency_report,
        ratio_reuse_report=ratio_reuse_report,
        ratio_manifest=ratio_manifest,
    )


def build_features_from_frame(
    df: pd.DataFrame,
    *,
    source_contract_path: str | Path = DEFAULT_INPUT_CONTRACT_PATH,
    feature_columns_path: str | Path = DEFAULT_FEATURE_COLUMNS_PATH,
    source_feature_types_path: str | Path | None = DEFAULT_SOURCE_FEATURE_TYPES_PATH,
    source_encoding_manifest_path: str | Path | None = DEFAULT_SOURCE_ENCODING_MANIFEST_PATH,
    fb_output_dir: str | Path = DEFAULT_FB_OUTPUT_DIR,
    ml_input_dir: str | Path = DEFAULT_ML_INPUT_DIR,
    artifact_prefix: str = "ml_06__r00",
    overwrite: bool = False,
) -> FeatureBuildResult:
    """Build ML-06 features from an in-memory frame and export artifacts."""

    temp_input = Path("<dataframe>")
    config = FeatureBuildConfig(
        input_path=DEFAULT_INPUT_PATH,
        source_contract_path=source_contract_path,
        feature_columns_path=feature_columns_path,
        source_feature_types_path=source_feature_types_path,
        source_encoding_manifest_path=source_encoding_manifest_path,
        fb_output_dir=fb_output_dir,
        ml_input_dir=ml_input_dir,
        artifact_prefix=artifact_prefix,
        sample_rows=None,
        overwrite=overwrite,
    ).resolved()
    paths = _output_paths(config)
    require_no_existing_outputs(paths.values(), overwrite=config.overwrite)

    source_columns = list(df.columns)
    input_contract = pd.read_csv(config.source_contract_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})
    contract_validation = validate_input_contract(
        input_contract,
        artifact_prefix=config.artifact_prefix,
        available_columns=source_columns,
    )
    split_df = validate_frame_contract(df)
    recency_df, recency_report = apply_recency_sentinel_policy(split_df)
    feature_df, ratio_manifest, ratio_reuse_report, ratio_specs = apply_ratio_transforms(
        recency_df,
        feature_columns=contract_validation.selected_columns,
        base_ratio_columns=contract_validation.ratio_base_columns,
    )
    frames = _split_frames(feature_df)
    row_counts = _row_counts(frames)

    contract_variants = build_contract_variants(
        input_contract,
        feature_columns=source_feature_columns_for_variants(input_contract),
        ratio_specs=ratio_specs,
        artifact_prefix=config.artifact_prefix,
    )
    contract_approve = update_contract_materialization(input_contract, feature_df)
    contract_original = update_contract_materialization(contract_variants["original"], feature_df)
    contract_log1p = update_contract_materialization(contract_variants["log1p"], feature_df)
    contract_clip = update_contract_materialization(contract_variants["clip_p9999"], feature_df)
    generated_columns = list(
        dict.fromkeys(
            column
            for spec in ratio_specs
            for column in (spec.log1p_column, spec.clip_column)
            if column in feature_df.columns
        )
    )
    feature_types = build_feature_types(
        source_feature_types_path=str(config.source_feature_types_path) if config.source_feature_types_path else None,
        generated_columns=generated_columns,
    )
    encoding_manifest = build_encoding_manifest(
        input_path=temp_input,
        source_contract_path=Path(config.source_contract_path),
        feature_columns_path=Path(config.feature_columns_path),
        source_encoding_manifest_path=Path(config.source_encoding_manifest_path) if config.source_encoding_manifest_path else None,
        ratio_manifest=ratio_manifest,
        generated_columns=generated_columns,
        row_counts=row_counts,
        artifact_prefix=config.artifact_prefix,
        output_dir=Path(config.ml_input_dir),
        output_paths=paths,
        feature_columns=contract_validation.selected_columns,
        materialized_columns=list(feature_df.columns),
    )

    _save_split_outputs(frames, paths, overwrite=config.overwrite)
    save_dataframe_csv(recency_report, paths["recency_report"], overwrite=config.overwrite)
    save_dataframe_csv(ratio_reuse_report, paths["ratio_reuse_report"], overwrite=config.overwrite)
    write_json(paths["ratio_manifest"], ratio_manifest, overwrite=config.overwrite)
    save_dataframe_csv(contract_original, paths["contract_original"], overwrite=config.overwrite)
    save_dataframe_csv(contract_approve, paths["contract_approve"], overwrite=config.overwrite)
    save_dataframe_csv(contract_log1p, paths["contract_log1p"], overwrite=config.overwrite)
    save_dataframe_csv(contract_clip, paths["contract_clip"], overwrite=config.overwrite)
    write_json(paths["feature_types"], feature_types, overwrite=config.overwrite)
    write_json(paths["encoding_manifest"], encoding_manifest, overwrite=config.overwrite)

    return FeatureBuildResult(
        output_paths={name: str(path) for name, path in paths.items()},
        row_counts=row_counts,
        generated_columns=generated_columns,
        base_ratio_columns=list(ratio_manifest["base_ratio_columns"]),
        recency_report=recency_report,
        ratio_reuse_report=ratio_reuse_report,
        ratio_manifest=ratio_manifest,
    )
