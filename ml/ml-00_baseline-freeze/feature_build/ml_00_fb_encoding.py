"""FB encoding/export 단계.

이 모듈은 feature 연산이 끝난 DataFrame 또는 기존 split 포함 parquet를 받아
train split 기준으로 encoding을 fit하고 fb_outputs 검토용 산출물을 저장한다.
최종 ML 입력 검증은 train_val_test의 ML loader가 담당한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import pandas as pd

from ml_00_fb_build import _existing_split_metadata_frame
from ml_00_fb_catalog import make_split_summary
from ml_00_fb_io import (
    load_parquet_columns,
    parquet_columns,
    resolve_path,
    save_dataframe_csv,
    save_dataframe_parquet,
    save_json,
    utc_now_iso,
)
from ml_00_fb_schema import (
    normalize_category_strict,
    validate_no_forbidden_feature_columns,
    validate_no_forbidden_input_columns,
)


SUPPORTED_ENCODINGS = {"passthrough", "label_code", "xgb_native", "one_hot"}
SUPPORTED_BUILD_ACTIONS = {"carry_forward", "build", "encode", "drop"}
META_COLUMNS = ("tx_id", "timestamp", "split", "label")


@dataclass(frozen=True)
class EncodingSpec:
    """한 source column을 ML 입력 feature로 변환하는 encoding 선언."""

    source_column: str
    output_column: str
    encoding: str
    used_in_ml: bool = True


@dataclass(frozen=True)
class EncodingOutputPaths:
    """encoding 단계가 생성하는 산출물 경로."""

    output_dir: Path
    train_path: Path
    val_path: Path
    test_path: Path
    feature_contract_path: Path
    encoding_manifest_path: Path
    feature_types_path: Path
    category_mapping_path: Path
    category_unknown_summary_path: Path
    split_summary_path: Path


@dataclass(frozen=True)
class EncodingResult:
    """encoding 실행 결과."""

    output_paths: EncodingOutputPaths
    feature_columns: list[str]
    feature_types: dict[str, str]
    row_counts: dict[str, int]
    encoding_manifest: Mapping[str, Any]


def parse_used_in_ml_value(value: Any) -> bool:
    """CSV의 used_in_ml 값을 엄격한 boolean으로 해석한다."""

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unsupported used_in_ml value: {value!r}")


def load_encoding_specs(path: Union[str, Path]) -> list[EncodingSpec]:
    """encoding spec CSV 또는 ml_feature_columns.csv를 읽어 EncodingSpec 목록으로 반환한다."""

    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"encoding spec file not found: {spec_path}")

    table = pd.read_csv(spec_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})
    output_column_field = "output_column" if "output_column" in table.columns else "column_name"
    required = {output_column_field, "encoding", "used_in_ml"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"encoding spec CSV is missing columns: {sorted(missing)}")

    specs: list[EncodingSpec] = []
    for row_number, row in table.iterrows():
        used_in_ml = parse_used_in_ml_value(row["used_in_ml"])
        output_column = "" if pd.isna(row[output_column_field]) else str(row[output_column_field]).strip()
        raw_source_column = row["source_column"] if "source_column" in table.columns else ""
        source_column = "" if pd.isna(raw_source_column) else str(raw_source_column).strip()
        encoding = "" if pd.isna(row["encoding"]) else str(row["encoding"]).strip().lower()
        raw_build_action = row["build_action"] if "build_action" in table.columns else ""
        build_action = "" if pd.isna(raw_build_action) else str(raw_build_action).strip().lower()
        if not build_action:
            build_action = "encode" if encoding and encoding != "passthrough" else "carry_forward"
        if build_action not in SUPPORTED_BUILD_ACTIONS:
            raise ValueError(
                "Unsupported build_action. "
                f"csv_row={row_number + 2}, build_action={build_action!r}, "
                f"supported={sorted(SUPPORTED_BUILD_ACTIONS)}"
            )

        if "build_in_fb" in table.columns:
            build_in_fb = parse_used_in_ml_value(row["build_in_fb"])
            if used_in_ml and build_action == "encode" and not build_in_fb:
                raise ValueError(f"Selected encode row must have build_in_fb=TRUE. csv_row={row_number + 2}")

        if used_in_ml and build_action == "build":
            raise ValueError(
                "build_action='build' is not supported by encoding export. "
                f"Run feature build before encoding. csv_row={row_number + 2}"
            )
        if used_in_ml and build_action == "drop":
            raise ValueError(f"Selected row cannot use build_action='drop'. csv_row={row_number + 2}")
        if used_in_ml and build_action == "carry_forward":
            source_column = output_column
            encoding = "passthrough"

        if used_in_ml and (not source_column or not output_column):
            raise ValueError(f"Selected encoding spec has blank source/output column. csv_row={row_number + 2}")
        if used_in_ml and encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(
                "Unsupported encoding. "
                f"csv_row={row_number + 2}, encoding={encoding!r}, supported={sorted(SUPPORTED_ENCODINGS)}"
            )
        specs.append(
            EncodingSpec(
                source_column=source_column,
                output_column=output_column,
                encoding=encoding,
                used_in_ml=used_in_ml,
            )
        )
    return specs


def load_encoding_contract_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read the full encoding contract table, including used_in_ml=FALSE inventory rows."""

    contract_path = Path(path).expanduser().resolve()
    if not contract_path.exists():
        raise FileNotFoundError(f"encoding contract file not found: {contract_path}")
    table = pd.read_csv(contract_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})
    required = {"column_name", "used_in_ml", "source_column", "encoding"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"encoding contract CSV is missing columns: {sorted(missing)}")
    return table


def make_encoding_output_paths(output_dir: Union[str, Path], artifact_prefix: str) -> EncodingOutputPaths:
    """encoding 산출물 경로를 만든다."""

    prefix = str(artifact_prefix).strip()
    if not prefix:
        raise ValueError("artifact_prefix must not be empty.")
    base = resolve_path(output_dir)
    return EncodingOutputPaths(
        output_dir=base,
        train_path=base / f"{prefix}_Xy_train.parquet",
        val_path=base / f"{prefix}_Xy_val.parquet",
        test_path=base / f"{prefix}_Xy_test.parquet",
        feature_contract_path=base / f"{prefix}_feature_contract.csv",
        encoding_manifest_path=base / f"{prefix}_encoding_manifest.json",
        feature_types_path=base / f"{prefix}_feature_types.json",
        category_mapping_path=base / f"{prefix}_category_mapping_train_only.csv",
        category_unknown_summary_path=base / f"{prefix}_category_unknown_summary.csv",
        split_summary_path=base / f"{prefix}_split_summary.csv",
    )


def require_no_existing_encoding_outputs(paths: EncodingOutputPaths, overwrite: bool) -> None:
    """encoding 산출물이 이미 있으면 overwrite=False일 때 중단한다."""

    protected_outputs = [
        paths.train_path,
        paths.val_path,
        paths.test_path,
        paths.feature_contract_path,
        paths.encoding_manifest_path,
        paths.feature_types_path,
        paths.category_mapping_path,
        paths.category_unknown_summary_path,
        paths.split_summary_path,
    ]
    existing = [str(path) for path in protected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Existing encoding artifacts found. Set overwrite=True or change RUN_ID. "
            f"existing={existing}"
        )


def validate_encoding_outputs(
    paths: EncodingOutputPaths,
    *,
    feature_columns: list[str],
    feature_types: Mapping[str, str],
) -> None:
    """Validate only FB-owned encoding outputs immediately after export."""

    required_files = {
        "train": paths.train_path,
        "val": paths.val_path,
        "test": paths.test_path,
        "feature_contract": paths.feature_contract_path,
        "encoding_manifest": paths.encoding_manifest_path,
        "feature_types": paths.feature_types_path,
        "category_mapping": paths.category_mapping_path,
        "category_unknown_summary": paths.category_unknown_summary_path,
        "split_summary": paths.split_summary_path,
    }
    missing_files = {name: str(path) for name, path in required_files.items() if not path.is_file()}
    if missing_files:
        raise FileNotFoundError(f"encoding export did not create required files: {missing_files}")

    if not feature_columns:
        raise ValueError("encoding export produced no feature columns.")
    duplicated = sorted({column for column in feature_columns if feature_columns.count(column) > 1})
    if duplicated:
        raise ValueError(f"encoding export produced duplicated feature columns: {duplicated}")
    validate_no_forbidden_feature_columns(feature_columns)

    missing_feature_types = [column for column in feature_columns if column not in feature_types]
    if missing_feature_types:
        raise ValueError(f"feature_types is missing exported features: {missing_feature_types[:30]}")

    expected_columns = set(META_COLUMNS) | set(feature_columns)
    for split_name, split_path in {
        "train": paths.train_path,
        "val": paths.val_path,
        "test": paths.test_path,
    }.items():
        split_columns = set(parquet_columns(split_path))
        missing_columns = sorted(expected_columns - split_columns)
        if missing_columns:
            raise ValueError(
                "encoded split parquet is missing required columns. "
                f"split={split_name}, path={split_path}, missing={missing_columns[:30]}"
            )

    contract = pd.read_csv(paths.feature_contract_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})
    required_contract_columns = {"column_name", "used_in_ml", "source_column", "encoding", "xgb_feature_type"}
    missing_contract_columns = required_contract_columns - set(contract.columns)
    if missing_contract_columns:
        raise ValueError(f"feature contract is missing columns: {sorted(missing_contract_columns)}")

    selected_columns: list[str] = []
    for row_number, row in contract.iterrows():
        if parse_used_in_ml_value(row["used_in_ml"]):
            column = "" if pd.isna(row["column_name"]) else str(row["column_name"]).strip()
            if not column:
                raise ValueError(f"selected feature contract row has blank column_name. csv_row={row_number + 2}")
            selected_columns.append(column)

    if selected_columns != feature_columns:
        raise ValueError(
            "feature contract selected columns do not match exported feature columns. "
            f"expected_count={len(feature_columns)}, actual_count={len(selected_columns)}, "
            f"expected_head={feature_columns[:30]}, actual_head={selected_columns[:30]}"
        )


def _normalize_specs(specs: Iterable[EncodingSpec]) -> list[EncodingSpec]:
    selected = [spec for spec in specs if spec.used_in_ml]
    if not selected:
        raise ValueError("No selected encoding specs. At least one used_in_ml=True row is required.")

    for spec in selected:
        if spec.encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(f"Unsupported encoding: {spec.encoding!r}")

    output_names = [spec.output_column for spec in selected if spec.encoding != "one_hot"]
    duplicated = sorted({name for name in output_names if output_names.count(name) > 1})
    if duplicated:
        raise ValueError(f"Duplicated output columns in encoding specs: {duplicated}")

    validate_no_forbidden_input_columns(spec.source_column for spec in selected)
    validate_no_forbidden_feature_columns(output_names)
    return selected


def _slug_category(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value.strip()).strip("_").lower()
    return slug or "blank"


def _unknown_rows(
    *,
    output_column: str,
    source_column: str,
    encoding: str,
    normalized: pd.Series,
    categories: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    known = set(categories)
    for split_name in ("train", "val", "test"):
        split_mask = normalized.index.get_level_values("split") == split_name
        split_values = normalized.loc[split_mask]
        unknown_values = sorted(set(split_values.tolist()) - known)
        rows.append(
            {
                "feature_column": output_column,
                "source_column": source_column,
                "encoding": encoding,
                "split": split_name,
                "unknown_count": int((~split_values.isin(known)).sum()),
                "unknown_unique_count": int(len(unknown_values)),
                "unknown_examples": ";".join(str(value) for value in unknown_values[:5]),
            }
        )
    return rows


def _split_indexed_category(series: pd.Series, split_values: pd.Series, source_column: str) -> pd.Series:
    normalized = normalize_category_strict(series, source_col=source_column)
    return pd.Series(
        normalized.to_numpy(),
        index=pd.MultiIndex.from_arrays([split_values.to_numpy(), normalized.index], names=["split", "row_index"]),
        dtype="string",
    )


def encode_split_frame(
    split_df: pd.DataFrame,
    specs: Iterable[EncodingSpec],
    *,
    output_dir: Union[str, Path],
    artifact_prefix: str,
    overwrite: bool = False,
    input_label: str | None = None,
    contract_table: pd.DataFrame | None = None,
) -> EncodingResult:
    """split 컬럼이 확정된 DataFrame을 encoding하여 ML 입력 파일을 저장한다."""

    selected_specs = _normalize_specs(specs)
    required_columns = set(META_COLUMNS) | {spec.source_column for spec in selected_specs}
    missing = required_columns - set(split_df.columns)
    if missing:
        raise ValueError(f"encoding input is missing required columns: {sorted(missing)}")

    paths = make_encoding_output_paths(output_dir, artifact_prefix)
    require_no_existing_encoding_outputs(paths, overwrite=overwrite)

    base = split_df.loc[:, list(dict.fromkeys(META_COLUMNS))].copy()
    feature_frame = base.copy()
    feature_columns: list[str] = []
    feature_types: dict[str, str] = {}
    feature_spec_metadata: dict[str, dict[str, str]] = {}
    category_values: dict[str, list[str]] = {}
    mapping_rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []

    train_mask = split_df["split"] == "train"
    if not train_mask.any():
        raise ValueError("encoding input has no train rows.")

    for spec in selected_specs:
        source = split_df[spec.source_column]

        if spec.encoding == "passthrough":
            feature_frame[spec.output_column] = source
            feature_columns.append(spec.output_column)
            feature_types[spec.output_column] = "q"
            feature_spec_metadata[spec.output_column] = {
                "source_column": spec.source_column,
                "encoding": spec.encoding,
            }
            continue

        normalized = _split_indexed_category(source, split_df["split"], spec.source_column)
        train_values = normalized.loc["train"]
        categories = sorted(train_values.unique().tolist())
        if not categories:
            raise ValueError(f"train split has no category values. source_column={spec.source_column!r}")
        category_values[spec.output_column] = categories
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
            feature_frame[spec.output_column] = normalized.reset_index(level="split", drop=True).map(mapping).fillna(-1).astype("int32")
            feature_columns.append(spec.output_column)
            feature_types[spec.output_column] = "q"
            feature_spec_metadata[spec.output_column] = {
                "source_column": spec.source_column,
                "encoding": spec.encoding,
            }
            for category, code in mapping.items():
                mapping_rows.append(
                    {
                        "feature_column": spec.output_column,
                        "source_column": spec.source_column,
                        "category_value": category,
                        "encoded_value": int(code),
                        "encoding": spec.encoding,
                        "fit_split": "train",
                    }
                )
            continue

        if spec.encoding == "xgb_native":
            values = normalized.reset_index(level="split", drop=True)
            feature_frame[spec.output_column] = pd.Categorical(values.where(values.isin(categories)), categories=categories)
            feature_columns.append(spec.output_column)
            feature_types[spec.output_column] = "c"
            feature_spec_metadata[spec.output_column] = {
                "source_column": spec.source_column,
                "encoding": spec.encoding,
            }
            for category in categories:
                mapping_rows.append(
                    {
                        "feature_column": spec.output_column,
                        "source_column": spec.source_column,
                        "category_value": category,
                        "encoded_value": None,
                        "encoding": spec.encoding,
                        "fit_split": "train",
                    }
                )
            continue

        if spec.encoding == "one_hot":
            slugs = [_slug_category(category) for category in categories]
            if len(slugs) != len(set(slugs)):
                raise ValueError(f"one_hot category names collide after slugging. output_column={spec.output_column!r}")
            values = normalized.reset_index(level="split", drop=True)
            for category, slug in zip(categories, slugs):
                output_column = f"{spec.output_column}__{slug}"
                feature_frame[output_column] = (values == category).astype("int8")
                feature_columns.append(output_column)
                feature_types[output_column] = "q"
                feature_spec_metadata[output_column] = {
                    "source_column": spec.source_column,
                    "encoding": spec.encoding,
                }
                mapping_rows.append(
                    {
                        "feature_column": output_column,
                        "source_column": spec.source_column,
                        "category_value": category,
                        "encoded_value": 1,
                        "encoding": spec.encoding,
                        "fit_split": "train",
                    }
                )

    validate_no_forbidden_feature_columns(feature_columns)
    duplicated_features = sorted({column for column in feature_columns if feature_columns.count(column) > 1})
    if duplicated_features:
        raise ValueError(f"Encoded feature columns are duplicated: {duplicated_features}")

    train_df = feature_frame.loc[feature_frame["split"] == "train", list(META_COLUMNS) + feature_columns].reset_index(drop=True)
    val_df = feature_frame.loc[feature_frame["split"] == "val", list(META_COLUMNS) + feature_columns].reset_index(drop=True)
    test_df = feature_frame.loc[feature_frame["split"] == "test", list(META_COLUMNS) + feature_columns].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "encoded split output must not be empty. "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    if contract_table is None:
        feature_columns_table = pd.DataFrame(
            [
                {
                    "column_name": column,
                    "used_in_ml": "TRUE",
                    "source_column": feature_spec_metadata[column]["source_column"],
                    "encoding": feature_spec_metadata[column]["encoding"],
                    "feature_group": "encoded",
                    "dtype": str(feature_frame[column].dtype),
                    "xgb_feature_type": feature_types[column],
                }
                for column in feature_columns
            ]
        )
    else:
        feature_columns_table = contract_table.copy()
        feature_columns_table["used_in_ml"] = feature_columns_table["used_in_ml"].map(
            lambda value: "TRUE" if parse_used_in_ml_value(value) else "FALSE"
        )
        if "dtype" not in feature_columns_table.columns:
            feature_columns_table["dtype"] = ""
        if "xgb_feature_type" not in feature_columns_table.columns:
            feature_columns_table["xgb_feature_type"] = ""

        for column in feature_columns:
            matched = feature_columns_table["column_name"].astype(str).str.strip() == column
            if not matched.any():
                raise ValueError(f"selected feature is missing from output contract table: {column}")
            feature_columns_table.loc[matched, "source_column"] = feature_spec_metadata[column]["source_column"]
            feature_columns_table.loc[matched, "encoding"] = feature_spec_metadata[column]["encoding"]
            feature_columns_table.loc[matched, "dtype"] = str(feature_frame[column].dtype)
            feature_columns_table.loc[matched, "xgb_feature_type"] = feature_types[column]

        non_selected = feature_columns_table["used_in_ml"] != "TRUE"
        feature_columns_table.loc[non_selected & feature_columns_table["xgb_feature_type"].isna(), "xgb_feature_type"] = ""
    mapping_frame = pd.DataFrame(mapping_rows)
    unknown_frame = pd.DataFrame(unknown_rows)
    split_summary = make_split_summary(feature_frame.loc[:, list(META_COLUMNS)])

    row_counts = {
        "all": int(len(feature_frame)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }
    manifest: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "artifact_prefix": artifact_prefix,
        "input": input_label,
        "output_dir": str(paths.output_dir),
        "feature_columns": feature_columns,
        "feature_types": feature_types,
        "categorical_columns": [column for column in feature_columns if feature_types[column] == "c"],
        "category_values": category_values,
        "encoding_specs": [spec.__dict__ for spec in selected_specs],
        "row_counts": row_counts,
        "outputs": {
            "train_path": str(paths.train_path),
            "val_path": str(paths.val_path),
            "test_path": str(paths.test_path),
            "feature_contract_path": str(paths.feature_contract_path),
            "encoding_manifest_path": str(paths.encoding_manifest_path),
            "feature_types_path": str(paths.feature_types_path),
            "category_mapping_path": str(paths.category_mapping_path),
            "category_unknown_summary_path": str(paths.category_unknown_summary_path),
            "split_summary_path": str(paths.split_summary_path),
        },
    }

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe_parquet(train_df, paths.train_path)
    save_dataframe_parquet(val_df, paths.val_path)
    save_dataframe_parquet(test_df, paths.test_path)
    save_dataframe_csv(feature_columns_table, paths.feature_contract_path)
    save_dataframe_csv(mapping_frame, paths.category_mapping_path)
    save_dataframe_csv(unknown_frame, paths.category_unknown_summary_path)
    save_dataframe_csv(split_summary, paths.split_summary_path)
    save_json({"feature_types": feature_types}, paths.feature_types_path)
    save_json(manifest, paths.encoding_manifest_path)
    validate_encoding_outputs(paths, feature_columns=feature_columns, feature_types=feature_types)

    return EncodingResult(
        output_paths=paths,
        feature_columns=feature_columns,
        feature_types=feature_types,
        row_counts=row_counts,
        encoding_manifest=manifest,
    )


def encode_existing_split_for_ml(
    input_path: Union[str, Path],
    encoding_specs: Optional[Iterable[EncodingSpec]] = None,
    *,
    encoding_spec_path: Union[str, Path, None] = None,
    output_dir: Union[str, Path],
    artifact_prefix: str,
    overwrite: bool = False,
    tx_id_col: str = "tx_id",
    timestamp_col: str = "timestamp",
    label_col: str = "label",
    split_col: str = "split",
) -> EncodingResult:
    """기존 split 컬럼이 있는 단일 parquet에서 fb_outputs 검토용 encoded split 파일을 생성한다."""

    if encoding_specs is None and encoding_spec_path is None:
        raise ValueError("encoding_specs or encoding_spec_path is required.")
    if encoding_specs is not None and encoding_spec_path is not None:
        raise ValueError("Pass only one of encoding_specs or encoding_spec_path.")

    contract_table = load_encoding_contract_table(encoding_spec_path) if encoding_spec_path is not None else None
    specs = load_encoding_specs(encoding_spec_path) if encoding_spec_path is not None else list(encoding_specs or [])
    selected_specs = _normalize_specs(specs)

    resolved_input = resolve_path(input_path)
    available_columns = set(parquet_columns(resolved_input))
    required_columns = {tx_id_col, timestamp_col, label_col, split_col} | {spec.source_column for spec in selected_specs}
    missing = required_columns - available_columns
    if missing:
        raise ValueError(f"input parquet is missing encoding columns. path={resolved_input}, missing={sorted(missing)}")

    input_df = load_parquet_columns(resolved_input, required_columns)
    metadata = _existing_split_metadata_frame(
        input_df,
        source_path=resolved_input,
        tx_id_col=tx_id_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
        split_col=split_col,
    )
    split_df = input_df.copy().reset_index(drop=True)
    split_df["tx_id"] = metadata["tx_id"]
    split_df["timestamp"] = metadata["timestamp"]
    split_df["label"] = metadata["label"]
    split_df["split"] = metadata["split"].astype("string")

    return encode_split_frame(
        split_df,
        specs,
        output_dir=output_dir,
        artifact_prefix=artifact_prefix,
        overwrite=overwrite,
        input_label=str(resolved_input),
        contract_table=contract_table,
    )
