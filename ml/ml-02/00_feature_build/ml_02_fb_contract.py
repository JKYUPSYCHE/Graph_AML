"""Feature contract helpers for FB input files.

contract는 노트북에서 사람이 검토한 feature build/encoding 계획표다.
이 모듈은 contract를 수정하지 않고, 실행 전에 문법과 source parquet 호환성만 검증한다.

검증 원칙
---------
- ``used_in_ml``/``build_in_fb`` 같은 boolean flag는 TRUE/FALSE 대문자만 허용한다.
- ``build_action``은 carry_forward/build/encode 중 하나여야 한다.
- carry_forward/encode row는 source parquet에 source_column이 실제 존재해야 한다.
- build row는 아직 source parquet에 없는 feature를 현재 FB 단계에서 만들 예정이므로 source 존재 검사를 건너뛴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml_02_fb_io import parquet_columns
from ml_02_fb_schema import validate_no_forbidden_feature_columns


ALLOWED_BOOL_VALUES = {"TRUE", "FALSE"}
ALLOWED_BUILD_ACTIONS = {"carry_forward", "build", "encode"}
ALLOWED_ENCODINGS = {"passthrough", "label_code", "xgb_native"}
CONTRACT_VERSION = 1

# contract CSV의 표준 컬럼 순서다.
# 작성 노트북과 build 노트북이 같은 schema를 공유해야 사람이 diff/검토하기 쉽다.
CONTRACT_COLUMNS = [
    "contract_version",
    "artifact_prefix",
    "column_name",
    "used_in_ml",
    "source_column",
    "column_origin",
    "encoding",
    "build_action",
    "build_in_fb",
    "xgb_feature_type",
    "feature_group",
    "dtype",
    "leakage_risk",
    "review_status",
    "excluded_reason",
    "selection_note",
    "contract_row_origin",
    "parent_artifact_prefix",
    "source_contract_path",
    "source_data_path",
    "feature_spec_name",
    "encoding_params",
    "materialized",
    "observed_dtype",
    "missing_count",
    "missing_rate",
    "unknown_category_count",
    "fit_split",
    "note",
]


@dataclass(frozen=True)
class ContractValidationResult:
    """contract 검증 결과 중 노트북에서 바로 확인할 요약값."""

    path: Path
    total_rows: int
    selected_count: int
    selected_columns: list[str]


def _require_file(path: str | Path, label: str) -> Path:
    """입력 경로가 실제 파일인지 확인하고 절대경로 Path로 반환한다."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found or not a file: {resolved}")
    return resolved


def _require_contract_columns(table: pd.DataFrame) -> None:
    """contract 검증에 필요한 최소 컬럼 존재 여부를 확인한다."""

    missing = {"column_name", "used_in_ml"} - set(table.columns)
    if missing:
        raise ValueError(f"feature contract is missing columns: {sorted(missing)}")


def _selected_columns(table: pd.DataFrame) -> list[str]:
    """used_in_ml=TRUE인 feature column 목록을 순서대로 추출하고 누수 위험 이름을 차단한다."""

    selected = table.loc[table["used_in_ml"] == "TRUE", "column_name"]
    columns = selected.astype(str).str.strip().tolist()
    blank_rows = [int(index) + 2 for index, column in zip(selected.index, columns) if not column]
    if blank_rows:
        raise ValueError(f"selected feature rows contain blank column_name. csv_rows={blank_rows[:30]}")

    duplicated = sorted({column for column in columns if columns.count(column) > 1})
    if duplicated:
        raise ValueError(f"duplicated selected feature columns: {duplicated}")
    validate_no_forbidden_feature_columns(columns)
    if not columns:
        raise ValueError("no selected feature columns found. At least one used_in_ml=TRUE row is required.")
    return columns


def validate_fb_input_contract(
    contract_path: str | Path,
    *,
    source_data_path: str | Path,
    artifact_prefix: str | None = None,
) -> ContractValidationResult:
    """FB input contract 문법과 source parquet 호환성을 검증한다.

    이 함수는 feature를 만들거나 contract 값을 보정하지 않는다. 잘못된 row를 발견하면
    CSV row 번호를 포함한 ValueError로 중단해 작성 노트북에서 수정하도록 한다.
    """

    path = _require_file(contract_path, "feature contract")
    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"used_in_ml": "string", "build_in_fb": "string", "materialized": "string"},
    )
    _require_contract_columns(table)
    # selected feature 판정 기준인 used_in_ml부터 엄격하게 정규화한다.
    # "true" 같은 소문자를 허용하지 않는 이유는 CSV diff에서 값 흔들림을 줄이기 위해서다.
    table["used_in_ml"] = _require_strict_bool(table["used_in_ml"], "used_in_ml")

    # 현재 RUN_ID/ARTIFACT_PREFIX와 다른 contract를 실수로 읽는 상황을 조기에 차단한다.
    if artifact_prefix is not None:
        if "artifact_prefix" not in table.columns:
            raise ValueError("feature contract is missing artifact_prefix column")
        mismatched = table[table["artifact_prefix"].astype(str) != artifact_prefix]
        if not mismatched.empty:
            rows = (mismatched.index + 2).tolist()
            raise ValueError(f"artifact_prefix mismatch. expected={artifact_prefix!r}, csv_rows={rows[:30]}")

    if "build_in_fb" in table.columns:
        _require_strict_bool(table["build_in_fb"], "build_in_fb")
    if "materialized" in table.columns:
        _require_strict_bool(table["materialized"], "materialized")
    # build_action/encoding은 뒤 단계에서 실행 분기를 결정하므로 알 수 없는 값을 허용하지 않는다.
    if "build_action" in table.columns:
        invalid_action = table[~table["build_action"].astype(str).isin(ALLOWED_BUILD_ACTIONS)]
        if not invalid_action.empty:
            rows = (invalid_action.index + 2).tolist()
            raise ValueError(f"unsupported build_action values. csv_rows={rows[:30]}")
    if "encoding" in table.columns:
        invalid_encoding = table[~table["encoding"].astype(str).isin(ALLOWED_ENCODINGS)]
        if not invalid_encoding.empty:
            rows = (invalid_encoding.index + 2).tolist()
            raise ValueError(f"unsupported encoding values. csv_rows={rows[:30]}")

    # 전체 contract에서 column_name이 중복되면 build/encode 결과가 어느 row에 대응되는지 모호해진다.
    all_columns = table["column_name"].astype(str).str.strip().tolist()
    duplicated = sorted({column for column in all_columns if all_columns.count(column) > 1})
    if duplicated:
        raise ValueError(f"duplicated contract column_name values: {duplicated}")

    selected = _selected_columns(table)
    # source parquet schema만 읽어 carry_forward/encode source 존재 여부를 확인한다.
    # build row는 현재 단계에서 새로 생성될 컬럼이므로 source parquet 존재 검사를 하지 않는다.
    source_columns = set(parquet_columns(_require_file(source_data_path, "source data parquet")))
    missing_sources: list[dict[str, str | int]] = []
    for index, row in table.iterrows():
        if row["used_in_ml"] != "TRUE":
            continue
        build_action = str(row["build_action"]).strip().lower() if "build_action" in table.columns else "carry_forward"
        if build_action not in {"carry_forward", "encode"}:
            continue
        raw_source = row["source_column"] if "source_column" in table.columns else row["column_name"]
        source_column = "" if pd.isna(raw_source) else str(raw_source).strip()
        if not source_column or source_column not in source_columns:
            missing_sources.append({"csv_row": int(index) + 2, "source_column": source_column})
    if missing_sources:
        raise ValueError(
            "selected carry_forward/encode source columns are missing from source parquet. "
            f"missing={missing_sources[:30]}, missing_count={len(missing_sources)}"
        )

    return ContractValidationResult(
        path=path,
        total_rows=int(len(table)),
        selected_count=int(len(selected)),
        selected_columns=selected,
    )


def _require_strict_bool(series: pd.Series, column_name: str) -> pd.Series:
    """TRUE/FALSE 문자열만 허용하는 contract flag 검증 함수."""

    if series.isna().any():
        rows = (series[series.isna()].index + 2).tolist()
        raise ValueError(f"{column_name} contains missing values. csv_rows={rows[:30]}")
    text = series.astype(str)
    invalid = text[~text.isin(ALLOWED_BOOL_VALUES)]
    if not invalid.empty:
        rows = (invalid.index + 2).tolist()
        raise ValueError(
            f"{column_name} contains unsupported values. "
            f"allowed_values={sorted(ALLOWED_BOOL_VALUES)}, "
            f"invalid_values={sorted(invalid.unique().tolist())[:30]}, "
            f"csv_rows={rows[:30]}"
        )
    return text.astype("string")
