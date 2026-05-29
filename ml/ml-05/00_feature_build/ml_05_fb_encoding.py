"""ML-05 FB encoding/export helpers.

The generated ML-05 features are numeric, but this module preserves the ML-04
encoding contract surface so existing notebooks can export carry-forward and
encoded columns with the same public entry point.

Code map:
- Input: split-aware feature_frame plus EncodingSpec contract rows.
- Output: train/val/test/all parquet, contract CSV, mapping CSV, manifest JSON.
- Public: encode_split_frame, load_encoding_specs, EncodingResult.
- Leakage guard: category vocab is fit on train only; forbidden feature names fail.
- Notes: build_features() does not save files; final export happens only here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

import pandas as pd

from ml_05_fb_catalog import make_split_summary
from ml_05_fb_io import (
    load_parquet_columns,
    parquet_columns,
    resolve_path,
    save_dataframe_csv,
    save_dataframe_parquet,
    save_json,
    utc_now_iso,
)
from ml_05_fb_encoding_materialize import (
    CATEGORY_MAPPING_CSV_COLUMNS,
    CATEGORY_UNKNOWN_SUMMARY_CSV_COLUMNS,
    categories_with_unknown as _categories_with_unknown,
    category_mapping_row as _category_mapping_row,
    fit_train_categories as _fit_train_categories,
    label_code_series as _label_code_series,
    normalize_input_category_values as _normalize_input_category_values,
    passthrough_series as _passthrough_series,
    split_indexed_category as _split_indexed_category,
    unknown_rows as _unknown_rows,
    xgb_native_series as _xgb_native_series,
)
from ml_05_fb_encoding_outputs import (
    OUTPUT_PATH_FIELDS,
    SPLIT_PATH_FIELDS,
    EncodingOutputPaths,
    make_encoding_output_paths,
    require_no_existing_encoding_outputs,
    split_encoded_frames as _split_encoded_frames,
    validate_category_manifest_values as _validate_category_manifest_values,
    validate_encoding_outputs,
)
from ml_05_fb_encoding_specs import (
    ENCODE_ENCODINGS,
    PASSTHROUGH_BUILD_ACTIONS,
    SUPPORTED_BUILD_ACTIONS,
    SUPPORTED_ENCODINGS,
    UNKNOWN_CATEGORY,
    EncodingSpec,
    load_encoding_specs,
    normalize_encoding_specs,
    parse_used_in_ml_value,
)
from ml_05_fb_schema import META_COLUMNS, validate_no_forbidden_feature_columns


__all__ = [
    "CATEGORY_MAPPING_CSV_COLUMNS",
    "CATEGORY_UNKNOWN_SUMMARY_CSV_COLUMNS",
    "ENCODE_ENCODINGS",
    "META_COLUMNS",
    "OUTPUT_PATH_FIELDS",
    "PASSTHROUGH_BUILD_ACTIONS",
    "SPLIT_PATH_FIELDS",
    "SUPPORTED_BUILD_ACTIONS",
    "SUPPORTED_ENCODINGS",
    "UNKNOWN_CATEGORY",
    "EncodingOutputPaths",
    "EncodingResult",
    "EncodingSpec",
    "encode_split_frame",
    "load_encoding_specs",
    "make_encoding_output_paths",
    "parse_used_in_ml_value",
    "require_no_existing_encoding_outputs",
    "validate_encoding_outputs",
]


@dataclass(frozen=True)
class EncodingResult:
    """Encoding/export result."""

    output_paths: EncodingOutputPaths
    feature_columns: list[str]
    materialized_columns: list[str]
    feature_types: dict[str, str]
    row_counts: dict[str, int]
    encoding_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedEncodingInputs:
    """Validated inputs needed before materialization starts."""

    split_df: pd.DataFrame
    materialized_specs: list[EncodingSpec]
    paths: EncodingOutputPaths
    carry_forward_category_values: dict[str, list[str]]


@dataclass(frozen=True)
class MaterializedEncodingPayload:
    """In-memory encoded frame and side artifacts before file writes."""

    feature_frame: pd.DataFrame
    feature_columns: list[str]
    materialized_columns: list[str]
    feature_types: dict[str, str]
    materialized_feature_types: dict[str, str]
    feature_spec_metadata: dict[str, dict[str, str]]
    category_values: dict[str, list[str]]
    mapping_frame: pd.DataFrame
    unknown_frame: pd.DataFrame


def _duplicated_names(names: list[str]) -> list[str]:
    return sorted({name for name in names if names.count(name) > 1})


def _normalize_specs(specs: Iterable[EncodingSpec]) -> list[EncodingSpec]:
    """Backward-compatible private wrapper around spec normalization."""

    return normalize_encoding_specs(specs)


def _source_columns_for_encoding(materialized_specs: list[EncodingSpec]) -> list[str]:
    """Return source columns required to materialize the encoding contract."""

    return list(dict.fromkeys(spec.source_column for spec in materialized_specs))


def _metadata_mismatch_rows(left: pd.Series, right: pd.Series) -> list[int]:
    """Return first row indexes where metadata values differ."""

    left_values = left.astype("string").fillna("__NA__").reset_index(drop=True)
    right_values = right.astype("string").fillna("__NA__").reset_index(drop=True)
    mismatch_mask = left_values != right_values
    return [int(index) for index in mismatch_mask[mismatch_mask].index[:5]]


def _supplement_missing_source_columns(
    prepared_split_df: pd.DataFrame,
    materialized_specs: list[EncodingSpec],
    *,
    source_data_path: Union[str, Path] | None,
) -> pd.DataFrame:
    """Load carry-forward/export columns from source parquet when build used a minimal column set."""

    missing_meta = set(META_COLUMNS) - set(prepared_split_df.columns)
    if missing_meta:
        raise ValueError(f"encoding input is missing metadata columns: {sorted(missing_meta)}")

    required_source_columns = _source_columns_for_encoding(materialized_specs)
    missing_source_columns = [column for column in required_source_columns if column not in prepared_split_df.columns]
    if not missing_source_columns:
        return prepared_split_df

    if source_data_path is None:
        raise ValueError(
            "encoding input is missing source columns and source_data_path was not provided. "
            f"missing={missing_source_columns[:30]}, missing_count={len(missing_source_columns)}, "
            "fix=pass source_data_path to encode_split_frame() or run build_features() with preserve_source_columns=True."
        )

    source_path = resolve_path(source_data_path)
    available_columns = parquet_columns(source_path)
    load_columns = list(dict.fromkeys([*META_COLUMNS, *missing_source_columns]))
    missing_from_source = [column for column in load_columns if column not in available_columns]
    if missing_from_source:
        raise ValueError(
            "source_data_path cannot supply missing encoding source columns. "
            f"source_data_path={source_path}, missing={missing_from_source[:30]}, "
            f"missing_count={len(missing_from_source)}"
        )

    print(
        "[ML-05 encoding] loading missing source columns "
        f"source={source_path} columns={len(missing_source_columns)}",
        flush=True,
    )
    source_df = load_parquet_columns(source_path, load_columns, sample_rows=None).reset_index(drop=True)
    if len(source_df) != len(prepared_split_df):
        raise ValueError(
            "source_data_path row count does not match encoding input. "
            f"encoding_rows={len(prepared_split_df)}, source_rows={len(source_df)}, source_data_path={source_path}"
        )

    for meta_col in META_COLUMNS:
        mismatch_rows = _metadata_mismatch_rows(prepared_split_df[meta_col], source_df[meta_col])
        if mismatch_rows:
            raise ValueError(
                "source_data_path metadata order/value mismatch. "
                f"column={meta_col!r}, first_mismatch_rows={mismatch_rows}, source_data_path={source_path}"
            )

    supplemented = prepared_split_df.copy(deep=False)
    for column in missing_source_columns:
        supplemented[column] = source_df[column].reset_index(drop=True)
    return supplemented


def _make_output_contract_table(
    *,
    contract_table: pd.DataFrame | None,
    materialized_specs: list[EncodingSpec],
    feature_spec_metadata: Mapping[str, Mapping[str, str]],
    materialized_feature_types: Mapping[str, str],
    feature_types: Mapping[str, str],
    materialized_columns: list[str],
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    if contract_table is None:
        return pd.DataFrame(
            [
                {
                    "column_name": spec.output_column,
                    "used_in_ml": "TRUE" if spec.used_in_ml else "FALSE",
                    "source_column": feature_spec_metadata[spec.output_column]["source_column"],
                    "encoding": feature_spec_metadata[spec.output_column]["encoding"],
                    "feature_group": "encoded",
                    "dtype": str(feature_frame[spec.output_column].dtype),
                    "xgb_feature_type": materialized_feature_types[spec.output_column] if spec.used_in_ml else "",
                    "materialized": "TRUE",
                }
                for spec in materialized_specs
            ]
        )

    feature_columns_table = contract_table.copy()
    feature_columns_table["used_in_ml"] = feature_columns_table["used_in_ml"].map(lambda value: "TRUE" if parse_used_in_ml_value(value) else "FALSE")
    for column in ("dtype", "xgb_feature_type", "materialized", "observed_dtype"):
        if column not in feature_columns_table.columns:
            feature_columns_table[column] = ""
    for column in materialized_columns:
        matched = feature_columns_table["column_name"].astype(str).str.strip() == column
        if not matched.any():
            raise ValueError(f"materialized column is missing from output contract table: {column}")
        feature_columns_table.loc[matched, "source_column"] = feature_spec_metadata[column]["source_column"]
        feature_columns_table.loc[matched, "encoding"] = feature_spec_metadata[column]["encoding"]
        feature_columns_table.loc[matched, "dtype"] = str(feature_frame[column].dtype)
        feature_columns_table.loc[matched, "observed_dtype"] = str(feature_frame[column].dtype)
        feature_columns_table.loc[matched, "materialized"] = "TRUE"
        if column in feature_types:
            feature_columns_table.loc[matched, "xgb_feature_type"] = feature_types[column]
    return feature_columns_table


def _prepare_encoding_inputs(
    split_df: pd.DataFrame,
    specs: Iterable[EncodingSpec],
    *,
    output_dir: Union[str, Path],
    artifact_prefix: str,
    overwrite: bool,
    input_category_values: Mapping[str, Any] | None,
    source_data_path: Union[str, Path] | None,
) -> PreparedEncodingInputs:
    """Normalize specs, validate required columns, and reserve output paths."""

    prepared_split_df = split_df.reset_index(drop=True).copy()
    materialized_specs = _normalize_specs(specs)
    prepared_split_df = _supplement_missing_source_columns(
        prepared_split_df,
        materialized_specs,
        source_data_path=source_data_path,
    )
    carry_forward_category_values = _normalize_input_category_values(input_category_values)
    required_columns = set(META_COLUMNS) | {spec.source_column for spec in materialized_specs}
    missing = required_columns - set(prepared_split_df.columns)
    if missing:
        raise ValueError(f"encoding input is missing required columns: {sorted(missing)}")

    train_mask = prepared_split_df["split"] == "train"
    if not bool(train_mask.any()):
        raise ValueError("encoding input has no train rows.")

    paths = make_encoding_output_paths(output_dir, artifact_prefix)
    require_no_existing_encoding_outputs(paths, overwrite=overwrite)
    return PreparedEncodingInputs(
        split_df=prepared_split_df,
        materialized_specs=materialized_specs,
        paths=paths,
        carry_forward_category_values=carry_forward_category_values,
    )


def _materialize_encoded_columns(prepared: PreparedEncodingInputs) -> MaterializedEncodingPayload:
    """Apply passthrough/category encodings in memory."""

    split_df = prepared.split_df
    feature_frame = split_df.reset_index(drop=True).copy()
    feature_columns: list[str] = []
    materialized_columns: list[str] = []
    feature_types: dict[str, str] = {}
    materialized_feature_types: dict[str, str] = {}
    feature_spec_metadata: dict[str, dict[str, str]] = {}
    category_values: dict[str, list[str]] = {}
    mapping_rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []

    def record_materialized_output(spec: EncodingSpec, xgb_feature_type: str) -> None:
        materialized_columns.append(spec.output_column)
        materialized_feature_types[spec.output_column] = xgb_feature_type
        feature_spec_metadata[spec.output_column] = {"source_column": spec.source_column, "encoding": spec.encoding}
        if spec.used_in_ml:
            feature_columns.append(spec.output_column)
            feature_types[spec.output_column] = xgb_feature_type

    for spec in prepared.materialized_specs:
        source = split_df[spec.source_column]
        if spec.encoding == "passthrough":
            if spec.used_in_ml and spec.xgb_feature_type == "c" and spec.build_action != "encode":
                if spec.output_column not in prepared.carry_forward_category_values:
                    raise ValueError(f"manual categorical carry-forward is missing category values. column={spec.output_column!r}")
                category_values[spec.output_column] = _categories_with_unknown(prepared.carry_forward_category_values[spec.output_column])
            feature_frame[spec.output_column] = _passthrough_series(split_df, spec)
            record_materialized_output(spec, spec.xgb_feature_type)
            continue

        normalized = _split_indexed_category(source, split_df["split"], spec.source_column)
        categories = _fit_train_categories(normalized, spec.source_column)
        unknown_rows.extend(
            _unknown_rows(
                output_column=spec.output_column,
                source_column=spec.source_column,
                encoding=spec.encoding,
                normalized=normalized,
                categories=categories,
            )
        )
        if spec.encoding == "label_code":
            mapping = {category: code for code, category in enumerate(categories)}
            feature_frame[spec.output_column] = _label_code_series(normalized, categories)
            record_materialized_output(spec, "q")
            for category, code in mapping.items():
                mapping_rows.append(_category_mapping_row(spec, category, int(code)))
            continue
        if spec.encoding == "xgb_native":
            categories_with_unknown = _categories_with_unknown(categories)
            if spec.used_in_ml:
                category_values[spec.output_column] = categories_with_unknown
            feature_frame[spec.output_column] = _xgb_native_series(normalized, categories)
            record_materialized_output(spec, "c")
            for category in categories_with_unknown:
                mapping_rows.append(_category_mapping_row(spec, category, None))
            continue
        raise ValueError(f"Unsupported encoding: {spec.encoding!r}")

    validate_no_forbidden_feature_columns(feature_columns)
    duplicated_features = _duplicated_names(feature_columns)
    if duplicated_features:
        raise ValueError(f"Encoded feature columns are duplicated: {duplicated_features}")
    duplicated_materialized = _duplicated_names(materialized_columns)
    if duplicated_materialized:
        raise ValueError(f"Materialized output columns are duplicated: {duplicated_materialized}")

    return MaterializedEncodingPayload(
        feature_frame=feature_frame,
        feature_columns=feature_columns,
        materialized_columns=materialized_columns,
        feature_types=feature_types,
        materialized_feature_types=materialized_feature_types,
        feature_spec_metadata=feature_spec_metadata,
        category_values=category_values,
        mapping_frame=pd.DataFrame(mapping_rows, columns=list(CATEGORY_MAPPING_CSV_COLUMNS)),
        unknown_frame=pd.DataFrame(unknown_rows, columns=list(CATEGORY_UNKNOWN_SUMMARY_CSV_COLUMNS)),
    )


def _write_encoding_outputs(
    prepared: PreparedEncodingInputs,
    payload: MaterializedEncodingPayload,
    *,
    artifact_prefix: str,
    input_label: str | None,
    contract_table: pd.DataFrame | None,
) -> EncodingResult:
    """Write encoded artifacts and validate the saved parquet contract."""

    feature_frame = payload.feature_frame
    paths = prepared.paths
    export_columns = list(dict.fromkeys([*META_COLUMNS, *payload.materialized_columns]))
    missing_export_columns = sorted(set(export_columns) - set(feature_frame.columns))
    if missing_export_columns:
        raise ValueError(f"encoded frame is missing export columns: {missing_export_columns[:30]}")
    all_df = feature_frame.loc[:, export_columns].reset_index(drop=True)
    split_frames = _split_encoded_frames(all_df)
    feature_columns_table = _make_output_contract_table(
        contract_table=contract_table,
        materialized_specs=prepared.materialized_specs,
        feature_spec_metadata=payload.feature_spec_metadata,
        materialized_feature_types=payload.materialized_feature_types,
        feature_types=payload.feature_types,
        materialized_columns=payload.materialized_columns,
        feature_frame=feature_frame,
    )
    categorical_columns = _validate_category_manifest_values(
        feature_columns=payload.feature_columns,
        feature_types=payload.feature_types,
        category_values=payload.category_values,
    )
    split_summary = make_split_summary(feature_frame.loc[:, list(META_COLUMNS)])
    row_counts = {
        "all": int(len(feature_frame)),
        "train": int(len(split_frames["train"])),
        "val": int(len(split_frames["val"])),
        "test": int(len(split_frames["test"])),
    }
    manifest: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "artifact_prefix": artifact_prefix,
        "input": input_label,
        "output_dir": str(paths.output_dir),
        "feature_columns": payload.feature_columns,
        "materialized_columns": payload.materialized_columns,
        "feature_types": payload.feature_types,
        "categorical_columns": categorical_columns,
        "category_values": payload.category_values,
        "unknown_category_policy": {"sentinel": UNKNOWN_CATEGORY, "applies_to": "xgb_native", "fit_split": "train"},
        "encoding_specs": [spec.__dict__ for spec in prepared.materialized_specs],
        "row_counts": row_counts,
        "outputs": {field: str(getattr(paths, field)) for _name, field in OUTPUT_PATH_FIELDS},
    }

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe_parquet(all_df, paths.all_path)
    for split_name, split_field in SPLIT_PATH_FIELDS:
        save_dataframe_parquet(split_frames[split_name], getattr(paths, split_field))
    save_dataframe_csv(feature_columns_table, paths.feature_contract_path)
    save_dataframe_csv(payload.mapping_frame, paths.category_mapping_path)
    save_dataframe_csv(payload.unknown_frame, paths.category_unknown_summary_path)
    save_dataframe_csv(split_summary, paths.split_summary_path)
    save_json({"feature_types": payload.feature_types}, paths.feature_types_path)
    save_json(manifest, paths.encoding_manifest_path)
    validate_encoding_outputs(
        paths,
        feature_columns=payload.feature_columns,
        materialized_columns=payload.materialized_columns,
        feature_types=payload.feature_types,
    )
    return EncodingResult(
        output_paths=paths,
        feature_columns=payload.feature_columns,
        materialized_columns=payload.materialized_columns,
        feature_types=payload.feature_types,
        row_counts=row_counts,
        encoding_manifest=manifest,
    )


def encode_split_frame(
    split_df: pd.DataFrame,
    specs: Iterable[EncodingSpec],
    *,
    output_dir: Union[str, Path],
    artifact_prefix: str,
    overwrite: bool = False,
    input_label: str | None = None,
    source_data_path: Union[str, Path] | None = None,
    contract_table: pd.DataFrame | None = None,
    input_category_values: Mapping[str, Any] | None = None,
) -> EncodingResult:
    """Materialize and export a split-aware feature frame.

    The public entry point is intentionally thin: prepare/validate inputs,
    materialize columns, then write files and validate saved parquet outputs.
    """

    prepared = _prepare_encoding_inputs(
        split_df,
        specs,
        output_dir=output_dir,
        artifact_prefix=artifact_prefix,
        overwrite=overwrite,
        input_category_values=input_category_values,
        source_data_path=source_data_path,
    )
    payload = _materialize_encoded_columns(prepared)
    return _write_encoding_outputs(
        prepared,
        payload,
        artifact_prefix=artifact_prefix,
        input_label=input_label,
        contract_table=contract_table,
    )
