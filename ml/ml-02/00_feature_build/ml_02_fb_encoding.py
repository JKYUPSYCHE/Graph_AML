"""FB encoding/export 단계.

이 모듈은 feature 연산이 끝난 DataFrame 또는 기존 split 포함 parquet를 받아
train split 기준으로 encoding을 fit하고 fb_outputs 검토용 산출물을 저장한다.
최종 ML 입력 검증은 train_val_test의 ML loader가 담당한다.

핵심 정책
---------
- category mapping은 train split에서만 fit한다. val/test에 새로 등장한 값은 unknown으로 기록한다.
- ``build_action``은 contract row가 feature build 단계에서 어떤 방식으로 materialize되는지 나타낸다.
- ``feature_columns``는 used_in_ml=True인 모델 입력 후보이고, ``materialized_columns``는 parquet에 실제 저장된 전체 컬럼이다.
- 이 단계의 산출물은 fb_outputs 후보 산출물이다. 사람이 승인한 ml_inputs와 동일하다고 가정하지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

import pandas as pd

from ml_02_fb_catalog import make_split_summary
from ml_02_fb_io import (
    parquet_columns,
    parquet_row_count,
    parquet_schema_types,
    resolve_path,
    save_dataframe_csv,
    save_dataframe_parquet,
    save_json,
    utc_now_iso,
)
from ml_02_fb_schema import (
    normalize_category_strict,
    parse_numeric_strict,
    validate_no_forbidden_feature_columns,
    validate_no_forbidden_input_columns,
)


SUPPORTED_ENCODINGS = {"passthrough", "label_code", "xgb_native"}
SUPPORTED_BUILD_ACTIONS = {"carry_forward", "build", "encode"}
META_COLUMNS = ("tx_id", "timestamp", "split", "label")
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
CATEGORY_MAPPING_CSV_COLUMNS = (
    "feature_column",
    "source_column",
    "category_value",
    "encoded_value",
    "encoding",
    "fit_split",
)
CATEGORY_UNKNOWN_SUMMARY_CSV_COLUMNS = (
    "feature_column",
    "source_column",
    "encoding",
    "split",
    "unknown_count",
    "unknown_unique_count",
    "unknown_examples",
)


@dataclass(frozen=True)
class EncodingSpec:
    """한 source column을 output parquet column으로 materialize하는 encoding 선언.

    ``source_column``은 현재 split frame에서 읽을 컬럼이고, ``output_column``은 저장할 컬럼명이다.
    build/carry_forward row는 보통 source와 output이 같고, encode row는 원본 category를 별도 output으로 변환한다.
    """

    source_column: str
    output_column: str
    encoding: str
    used_in_ml: bool = True
    build_action: str = "carry_forward"
    xgb_feature_type: str = ""


@dataclass(frozen=True)
class EncodingOutputPaths:
    """encoding 단계가 생성하는 산출물 경로."""

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


@dataclass(frozen=True)
class EncodingResult:
    """encoding 실행 결과."""

    output_paths: EncodingOutputPaths
    feature_columns: list[str]
    materialized_columns: list[str]
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


def _numeric_passthrough_series(split_df: pd.DataFrame, spec: EncodingSpec) -> pd.Series:
    """passthrough(q) feature가 downstream numeric 입력 계약을 만족하는지 검증한다."""

    try:
        return parse_numeric_strict(split_df, spec.source_column, spec.output_column)
    except ValueError as exc:
        raise ValueError(
            "Encoding failed: passthrough feature must be numeric because xgb_feature_type='q'. "
            f"source_column={spec.source_column!r}, output_column={spec.output_column!r}, "
            f"encoding={spec.encoding!r}"
        ) from exc


def _normalize_xgb_feature_type(
    value: Any,
    *,
    encoding: str,
    used_in_ml: bool,
    csv_row: int | None = None,
) -> str:
    """contract의 xgb_feature_type을 q/c/blank 중 하나로 정규화한다."""

    observed = "" if pd.isna(value) else str(value).strip().lower()
    if observed:
        if observed not in {"q", "c"}:
            row_context = "" if csv_row is None else f" csv_row={csv_row},"
            raise ValueError(
                "Unsupported xgb_feature_type. "
                f"{row_context} xgb_feature_type={observed!r}, supported=['c', 'q']"
            )
        return observed
    if not used_in_ml:
        return ""
    if encoding == "xgb_native":
        return "c"
    if encoding in {"label_code", "passthrough"}:
        return "q"
    return ""


def _passthrough_series(split_df: pd.DataFrame, spec: EncodingSpec) -> pd.Series:
    """passthrough 컬럼을 contract의 xgb_feature_type 정책에 맞춰 materialize한다."""

    if spec.xgb_feature_type == "q":
        return _numeric_passthrough_series(split_df, spec)
    return split_df[spec.source_column]


def _output_paths_by_name(paths: EncodingOutputPaths) -> dict[str, Path]:
    return {name: getattr(paths, field_name) for name, field_name in OUTPUT_PATH_FIELDS}


def _duplicated_names(names: list[str]) -> list[str]:
    return sorted({name for name in names if names.count(name) > 1})


def _category_mapping_row(spec: EncodingSpec, category: str, encoded_value: int | None) -> dict[str, Any]:
    return {
        "feature_column": spec.output_column,
        "source_column": spec.source_column,
        "category_value": category,
        "encoded_value": encoded_value,
        "encoding": spec.encoding,
        "fit_split": "train",
    }


def load_encoding_specs(path: Union[str, Path]) -> list[EncodingSpec]:
    """encoding spec CSV 또는 fb input contract를 읽어 EncodingSpec 목록으로 반환한다.

    contract 작성 단계와 feature build 실행 단계의 CSV schema가 조금 달라도 이 함수에서
    ``column_name``/``output_column`` 차이를 흡수한다. 단, build_action/encoding 값은
    허용 목록에 없으면 조용히 보정하지 않고 즉시 실패시킨다.
    """

    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"encoding spec file not found: {spec_path}")

    table = pd.read_csv(spec_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})
    # author contract에서는 column_name, 단독 encoding spec에서는 output_column을 쓸 수 있다.
    # 둘 중 실제 존재하는 컬럼명을 output column 기준으로 통일한다.
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
        # 구버전/간이 spec에는 build_action이 비어 있을 수 있다.
        # encoding이 명시되어 있으면 encode, 아니면 기존 컬럼 전달(carry_forward)로 해석한다.
        if not build_action:
            build_action = "encode" if encoding and encoding != "passthrough" else "carry_forward"
        if build_action not in SUPPORTED_BUILD_ACTIONS:
            raise ValueError(
                "Unsupported build_action. "
                f"csv_row={row_number + 2}, build_action={build_action!r}, "
                f"supported={sorted(SUPPORTED_BUILD_ACTIONS)}"
            )

        # build/encode는 FB 단계가 직접 materialize하는 행이므로 build_in_fb=TRUE여야 한다.
        # carry_forward는 이미 source parquet에 존재하는 컬럼을 넘겨받는 경우가 있어 별도로 허용한다.
        if "build_in_fb" in table.columns:
            build_in_fb = parse_used_in_ml_value(row["build_in_fb"])
            if build_action in {"build", "encode"} and not build_in_fb:
                raise ValueError(
                    "Rows with build_action='build' or 'encode' must have build_in_fb=TRUE. "
                    f"csv_row={row_number + 2}"
                )

        # build/carry_forward는 이미 output_column 이름으로 split frame에 존재해야 한다.
        # 따라서 source_column을 output_column으로 맞춰 이후 공통 검증에서 같은 방식으로 처리한다.
        if build_action == "build":
            source_column = output_column
            encoding = "passthrough"
        if build_action == "carry_forward":
            source_column = output_column
            encoding = "passthrough"

        if not output_column:
            raise ValueError(f"Encoding spec has blank output column. csv_row={row_number + 2}")
        if not source_column:
            raise ValueError(f"Encoding spec has blank source column. csv_row={row_number + 2}")
        if encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(
                "Unsupported encoding. "
                f"csv_row={row_number + 2}, encoding={encoding!r}, supported={sorted(SUPPORTED_ENCODINGS)}"
            )
        raw_xgb_feature_type = row["xgb_feature_type"] if "xgb_feature_type" in table.columns else ""
        xgb_feature_type = _normalize_xgb_feature_type(
            raw_xgb_feature_type,
            encoding=encoding,
            used_in_ml=used_in_ml,
            csv_row=row_number + 2,
        )
        if build_action == "encode" and encoding == "passthrough":
            raise ValueError(f"Encode row must not use passthrough encoding. csv_row={row_number + 2}")
        specs.append(
            EncodingSpec(
                source_column=source_column,
                output_column=output_column,
                encoding=encoding,
                used_in_ml=used_in_ml,
                build_action=build_action,
                xgb_feature_type=xgb_feature_type,
            )
        )
    return specs


def make_encoding_output_paths(output_dir: Union[str, Path], artifact_prefix: str) -> EncodingOutputPaths:
    """encoding 산출물 경로를 만든다."""

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


def require_no_existing_encoding_outputs(paths: EncodingOutputPaths, overwrite: bool) -> None:
    """encoding 산출물이 이미 있으면 overwrite=False일 때 중단한다."""

    existing = [str(path) for path in _output_paths_by_name(paths).values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Existing encoding artifacts found. Set overwrite=True or change RUN_ID. "
            f"existing={existing}"
        )


def validate_encoding_outputs(
    paths: EncodingOutputPaths,
    *,
    feature_columns: list[str],
    materialized_columns: list[str],
    feature_types: Mapping[str, str],
) -> None:
    """FB encoding/export 직후 산출물 일관성을 검증한다.

    검증 범위는 이 함수가 방금 만든 fb_outputs 파일로 제한한다.
    학습 단계의 승인 CSV, feature hash, loader 검증은 별도 ML loader가 담당한다.
    """

    # 저장 완료 여부를 먼저 확인한다. 파일이 누락된 상태에서 schema 검증을 계속하면 원인 추적이 어려워진다.
    missing_files = {name: str(path) for name, path in _output_paths_by_name(paths).items() if not path.is_file()}
    if missing_files:
        raise FileNotFoundError(f"encoding export did not create required files: {missing_files}")

    # 모델 입력 후보와 실제 저장 컬럼을 분리해서 검증한다.
    # used_in_ml=False인 컬럼은 parquet에는 있을 수 있지만 feature_columns에는 없어야 한다.
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

    # all/train/val/test parquet는 같은 컬럼 집합과 같은 dtype을 가져야 한다.
    # split별 dtype 차이는 downstream XGBoost/loader에서 재현하기 어려운 오류를 만든다.
    expected_columns = set(META_COLUMNS) | set(materialized_columns)
    all_columns = parquet_columns(paths.all_path)
    missing_all_columns = sorted(expected_columns - set(all_columns))
    if missing_all_columns:
        raise ValueError(
            "encoded all parquet is missing required columns. "
            f"path={paths.all_path}, missing={missing_all_columns[:30]}"
        )
    all_types = parquet_schema_types(paths.all_path)
    split_row_total = 0
    for split_name, split_field in SPLIT_PATH_FIELDS:
        split_path = getattr(paths, split_field)
        split_columns_ordered = parquet_columns(split_path)
        missing_columns = sorted(expected_columns - set(split_columns_ordered))
        if missing_columns:
            raise ValueError(
                "encoded split parquet is missing required columns. "
                f"split={split_name}, path={split_path}, missing={missing_columns[:30]}"
            )
        if split_columns_ordered != all_columns:
            raise ValueError(
                "encoded split parquet columns do not match Xy_all columns. "
                f"split={split_name}, path={split_path}"
            )
        split_types = parquet_schema_types(split_path)
        mismatched_types = {
            column: {"all": all_types.get(column), "split": split_types.get(column)}
            for column in all_columns
            if all_types.get(column) != split_types.get(column)
        }
        if mismatched_types:
            raise ValueError(
                "encoded split parquet schema types do not match Xy_all schema types. "
                f"split={split_name}, mismatched={dict(list(mismatched_types.items())[:30])}"
            )
        split_row_total += parquet_row_count(split_path)
    all_row_count = parquet_row_count(paths.all_path)
    if all_row_count != split_row_total:
        raise ValueError(
            "encoded Xy_all row count does not equal train+val+test row count. "
            f"all={all_row_count}, split_total={split_row_total}, path={paths.all_path}"
        )

    # output contract는 parquet에 실제 materialize된 컬럼 순서를 설명해야 한다.
    # 여기서 순서까지 검증해 학습 feature list와 parquet 컬럼 순서가 어긋나는 문제를 조기에 막는다.
    contract = pd.read_csv(paths.feature_contract_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})
    required_contract_columns = {"column_name", "used_in_ml", "source_column", "encoding", "xgb_feature_type"}
    missing_contract_columns = required_contract_columns - set(contract.columns)
    if missing_contract_columns:
        raise ValueError(f"feature contract is missing columns: {sorted(missing_contract_columns)}")

    contract_columns: list[str] = []
    selected_columns: list[str] = []
    for row_number, row in contract.iterrows():
        column = "" if pd.isna(row["column_name"]) else str(row["column_name"]).strip()
        if not column:
            raise ValueError(f"feature contract row has blank column_name. csv_row={row_number + 2}")
        contract_columns.append(column)
        if parse_used_in_ml_value(row["used_in_ml"]):
            selected_columns.append(column)

    if contract_columns != materialized_columns:
        raise ValueError(
            "feature contract columns do not match materialized columns. "
            f"expected_count={len(materialized_columns)}, actual_count={len(contract_columns)}, "
            f"expected_head={materialized_columns[:30]}, actual_head={contract_columns[:30]}"
        )

    if selected_columns != feature_columns:
        raise ValueError(
            "feature contract selected columns do not match exported feature columns. "
            f"expected_count={len(feature_columns)}, actual_count={len(selected_columns)}, "
            f"expected_head={feature_columns[:30]}, actual_head={selected_columns[:30]}"
        )


def _normalize_specs(specs: Iterable[EncodingSpec]) -> list[EncodingSpec]:
    """encoding spec 목록을 list로 고정하고 기본 무결성을 확인한다."""

    normalized: list[EncodingSpec] = []
    for spec in specs:
        normalized_xgb_feature_type = _normalize_xgb_feature_type(
            spec.xgb_feature_type,
            encoding=spec.encoding,
            used_in_ml=spec.used_in_ml,
        )
        normalized.append(
            EncodingSpec(
                source_column=str(spec.source_column).strip(),
                output_column=str(spec.output_column).strip(),
                encoding=str(spec.encoding).strip().lower(),
                used_in_ml=spec.used_in_ml,
                build_action=str(spec.build_action).strip().lower(),
                xgb_feature_type=normalized_xgb_feature_type,
            )
        )
    if not normalized:
        raise ValueError("No encoding specs. At least one contract row is required.")

    selected = [spec for spec in normalized if spec.used_in_ml]
    if not selected:
        raise ValueError("No selected encoding specs. At least one used_in_ml=True row is required.")

    for spec in normalized:
        if spec.build_action not in SUPPORTED_BUILD_ACTIONS:
            raise ValueError(f"Unsupported build_action: {spec.build_action!r}")
        if spec.encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(f"Unsupported encoding: {spec.encoding!r}")
        if not str(spec.source_column).strip() or not str(spec.output_column).strip():
            raise ValueError(
                "Encoding specs must have non-empty source/output columns. "
                f"source_column={spec.source_column!r}, output_column={spec.output_column!r}"
            )

    materialized_names = [spec.output_column for spec in normalized]
    duplicated = _duplicated_names(materialized_names)
    if duplicated:
        raise ValueError(f"Duplicated output columns in encoding specs: {duplicated}")

    validate_no_forbidden_input_columns(spec.source_column for spec in selected)
    validate_no_forbidden_feature_columns(spec.output_column for spec in selected)
    return normalized


def _unknown_rows(
    *,
    output_column: str,
    source_column: str,
    encoding: str,
    normalized: pd.Series,
    categories: list[str],
) -> list[dict[str, Any]]:
    """train에서 fit한 category 목록 기준으로 split별 unknown category 요약 row를 만든다."""

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
    """category 값을 정규화하고 split 정보를 MultiIndex에 붙인다.

    MultiIndex를 쓰면 train split에서 fit한 category와 val/test unknown category를 같은 Series에서
    안정적으로 비교할 수 있다.
    """

    normalized = normalize_category_strict(series, source_col=source_column)
    return pd.Series(
        normalized.to_numpy(),
        index=pd.MultiIndex.from_arrays([split_values.to_numpy(), normalized.index], names=["split", "row_index"]),
        dtype="string",
    )


def _fit_train_categories(normalized: pd.Series, source_column: str) -> list[str]:
    """train split 기준 category 목록을 만든다."""

    train_values = normalized.loc["train"]
    categories = sorted(train_values.unique().tolist())
    if not categories:
        raise ValueError(f"train split has no category values. source_column={source_column!r}")
    return categories


def _label_code_series(normalized: pd.Series, categories: list[str]) -> pd.Series:
    """train category mapping으로 label_code numeric feature를 만든다."""

    mapping = {category: code for code, category in enumerate(categories)}
    return normalized.reset_index(level="split", drop=True).map(mapping).fillna(-1).astype("int32")


def _xgb_native_series(normalized: pd.Series, categories: list[str]) -> pd.Categorical:
    """train category만 허용하는 XGBoost native categorical series를 만든다."""

    values = normalized.reset_index(level="split", drop=True)
    return pd.Categorical(values.where(values.isin(categories)), categories=categories)


def _normalize_input_category_values(input_category_values: Mapping[str, Any] | None) -> dict[str, list[str]]:
    """수동 categorical carry-forward에 사용할 category_values payload를 정규화한다."""

    if input_category_values is None:
        return {}

    raw_values = input_category_values.get("category_values", input_category_values)
    if not isinstance(raw_values, MappingABC):
        raise ValueError("input_category_values must be a mapping or a payload with category_values mapping.")

    normalized: dict[str, list[str]] = {}
    for column, values in raw_values.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, IterableABC):
            raise ValueError(f"category values must be a non-string iterable. column={column!r}")
        categories = [str(value) for value in values]
        if not categories:
            raise ValueError(f"category values must not be empty. column={column!r}")
        normalized[str(column)] = categories
    return normalized


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
    """encoding 결과를 설명하는 output contract table을 만든다."""

    if contract_table is None:
        return pd.DataFrame(
            [
                {
                    "column_name": column,
                    "used_in_ml": "TRUE" if feature_spec.used_in_ml else "FALSE",
                    "source_column": feature_spec_metadata[column]["source_column"],
                    "encoding": feature_spec_metadata[column]["encoding"],
                    "feature_group": "encoded",
                    "dtype": str(feature_frame[column].dtype),
                    "xgb_feature_type": materialized_feature_types[column] if feature_spec.used_in_ml else "",
                    "materialized": "TRUE",
                }
                for feature_spec in materialized_specs
                for column in [feature_spec.output_column]
            ]
        )

    feature_columns_table = contract_table.copy()
    feature_columns_table["used_in_ml"] = feature_columns_table["used_in_ml"].map(
        lambda value: "TRUE" if parse_used_in_ml_value(value) else "FALSE"
    )
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

    non_selected = feature_columns_table["used_in_ml"] != "TRUE"
    feature_columns_table.loc[non_selected & feature_columns_table["xgb_feature_type"].isna(), "xgb_feature_type"] = ""
    return feature_columns_table


def _validate_category_manifest_values(
    *,
    feature_columns: list[str],
    feature_types: Mapping[str, str],
    category_values: Mapping[str, list[str]],
) -> list[str]:
    """manifest의 category_values가 XGBoost categorical feature만 설명하는지 확인한다."""

    categorical_columns = [column for column in feature_columns if feature_types[column] == "c"]
    missing_category_values = [column for column in categorical_columns if column not in category_values]
    if missing_category_values:
        raise ValueError(
            "encoding manifest category_values is missing categorical features. "
            f"missing={missing_category_values[:30]}, missing_count={len(missing_category_values)}"
        )
    extra_category_values = sorted(set(category_values) - set(categorical_columns))
    if extra_category_values:
        raise ValueError(
            "encoding manifest category_values contains non-categorical features. "
            f"extra={extra_category_values[:30]}, extra_count={len(extra_category_values)}"
        )
    return categorical_columns


def _split_encoded_frames(all_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Xy_all frame을 train/val/test frame으로 나누고 dtype/컬럼 순서를 확인한다."""

    split_frames = {split: all_df.loc[all_df["split"] == split].reset_index(drop=True) for split, _field in SPLIT_PATH_FIELDS}
    train_df, val_df, test_df = split_frames["train"], split_frames["val"], split_frames["test"]
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "encoded split output must not be empty. "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

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
            raise ValueError(
                f"{split_name} dtypes do not match Xy_all dtypes: "
                f"{dict(list(dtype_mismatch.items())[:30])}"
            )
    return split_frames


def encode_split_frame(
    split_df: pd.DataFrame,
    specs: Iterable[EncodingSpec],
    *,
    output_dir: Union[str, Path],
    artifact_prefix: str,
    overwrite: bool = False,
    input_label: str | None = None,
    contract_table: pd.DataFrame | None = None,
    input_category_values: Mapping[str, Any] | None = None,
) -> EncodingResult:
    """split 컬럼이 확정된 DataFrame을 encoding하여 FB 후보 입력 파일을 저장한다.

    이 함수는 split을 새로 만들지 않는다. 입력 DataFrame에 있는 train/val/test 값을 그대로 사용하고,
    category encoding은 train split에서만 fit한다.
    """

    split_df = split_df.reset_index(drop=True).copy()
    materialized_specs = _normalize_specs(specs)
    carry_forward_category_values = _normalize_input_category_values(input_category_values)
    required_columns = set(META_COLUMNS) | {spec.source_column for spec in materialized_specs}
    missing = required_columns - set(split_df.columns)
    if missing:
        raise ValueError(f"encoding input is missing required columns: {sorted(missing)}")

    paths = make_encoding_output_paths(output_dir, artifact_prefix)
    require_no_existing_encoding_outputs(paths, overwrite=overwrite)

    base = split_df.reset_index(drop=True).copy()
    feature_frame = base.copy()
    feature_columns: list[str] = []
    materialized_columns: list[str] = []
    feature_types: dict[str, str] = {}
    materialized_feature_types: dict[str, str] = {}
    feature_spec_metadata: dict[str, dict[str, str]] = {}
    category_values: dict[str, list[str]] = {}
    mapping_rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []

    # category encoder는 train split만 fit source로 사용한다.
    # val/test category를 fit에 포함하면 temporal leakage와 split contamination이 생긴다.
    train_mask = split_df["split"] == "train"
    if not train_mask.any():
        raise ValueError("encoding input has no train rows.")

    def record_materialized_output(spec: EncodingSpec, xgb_feature_type: str) -> None:
        # materialized_columns는 parquet 저장 대상 전체이고, feature_columns는 used_in_ml=True인 모델 입력 후보만 담는다.
        materialized_columns.append(spec.output_column)
        materialized_feature_types[spec.output_column] = xgb_feature_type
        feature_spec_metadata[spec.output_column] = {
            "source_column": spec.source_column,
            "encoding": spec.encoding,
        }
        if spec.used_in_ml:
            feature_columns.append(spec.output_column)
            feature_types[spec.output_column] = xgb_feature_type

    for spec in materialized_specs:
        source = split_df[spec.source_column]

        if spec.encoding == "passthrough":
            # passthrough는 이미 materialized된 컬럼을 그대로 내보낸다.
            # ML-01에서 승인된 native categorical carry-forward는 contract의 xgb_feature_type='c'를 보존한다.
            if spec.used_in_ml and spec.xgb_feature_type == "c" and spec.build_action != "encode":
                if spec.output_column not in carry_forward_category_values:
                    raise ValueError(
                        "manual categorical carry-forward is missing category values. "
                        f"column={spec.output_column!r}"
                    )
                category_values[spec.output_column] = carry_forward_category_values[spec.output_column]
            feature_frame[spec.output_column] = _passthrough_series(split_df, spec)
            record_materialized_output(spec, spec.xgb_feature_type)
            continue

        # label_code/xgb_native는 모두 train category 목록으로 fit한다.
        # train에 없는 val/test 값은 label_code=-1 또는 xgb_native missing category로 남기고 summary에 기록한다.
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
            # label_code는 XGBoost에는 numeric(q) feature로 전달된다. unknown category는 -1 sentinel로 둔다.
            mapping = {category: code for code, category in enumerate(categories)}
            feature_frame[spec.output_column] = _label_code_series(normalized, categories)
            record_materialized_output(spec, "q")
            for category, code in mapping.items():
                mapping_rows.append(_category_mapping_row(spec, category, int(code)))
            continue

        if spec.encoding == "xgb_native":
            # xgb_native는 pandas Categorical dtype으로 저장하고 XGBoost feature type을 categorical(c)로 기록한다.
            category_values[spec.output_column] = categories
            feature_frame[spec.output_column] = _xgb_native_series(normalized, categories)
            record_materialized_output(spec, "c")
            for category in categories:
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

    # split별 parquet는 Xy_all과 컬럼 순서/dtype이 같아야 한다. 저장 전에 메모리에서 먼저 확인한다.
    all_df = feature_frame.reset_index(drop=True)
    split_frames = _split_encoded_frames(all_df)
    train_df, val_df, test_df = split_frames["train"], split_frames["val"], split_frames["test"]

    feature_columns_table = _make_output_contract_table(
        contract_table=contract_table,
        materialized_specs=materialized_specs,
        feature_spec_metadata=feature_spec_metadata,
        materialized_feature_types=materialized_feature_types,
        feature_types=feature_types,
        materialized_columns=materialized_columns,
        feature_frame=feature_frame,
    )
    categorical_columns = _validate_category_manifest_values(
        feature_columns=feature_columns,
        feature_types=feature_types,
        category_values=category_values,
    )

    mapping_frame = pd.DataFrame(mapping_rows, columns=list(CATEGORY_MAPPING_CSV_COLUMNS))
    unknown_frame = pd.DataFrame(unknown_rows, columns=list(CATEGORY_UNKNOWN_SUMMARY_CSV_COLUMNS))
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
        "materialized_columns": materialized_columns,
        "feature_types": feature_types,
        "categorical_columns": categorical_columns,
        "category_values": category_values,
        "encoding_specs": [spec.__dict__ for spec in materialized_specs],
        "row_counts": row_counts,
        "outputs": {field: str(getattr(paths, field)) for _name, field in OUTPUT_PATH_FIELDS},
    }

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe_parquet(all_df, paths.all_path)
    for split_name, split_field in SPLIT_PATH_FIELDS:
        save_dataframe_parquet(split_frames[split_name], getattr(paths, split_field))
    save_dataframe_csv(feature_columns_table, paths.feature_contract_path)
    save_dataframe_csv(mapping_frame, paths.category_mapping_path)
    save_dataframe_csv(unknown_frame, paths.category_unknown_summary_path)
    save_dataframe_csv(split_summary, paths.split_summary_path)
    save_json({"feature_types": feature_types}, paths.feature_types_path)
    save_json(manifest, paths.encoding_manifest_path)
    validate_encoding_outputs(
        paths,
        feature_columns=feature_columns,
        materialized_columns=materialized_columns,
        feature_types=feature_types,
    )

    return EncodingResult(
        output_paths=paths,
        feature_columns=feature_columns,
        materialized_columns=materialized_columns,
        feature_types=feature_types,
        row_counts=row_counts,
        encoding_manifest=manifest,
    )
