"""ML-05 FB 입력 파일용 피처 계약 헬퍼.

코드 맵:
- 입력           : FB 입력 계약 CSV와 원본 parquet 스키마.
- 출력           : 선택된 ML 컬럼을 포함한 ContractValidationResult.
- 공개 객체/함수 : validate_fb_input_contract, ContractValidationResult, CONTRACT_COLUMNS.
- 누수 방지      : 피처 생성/export 전에 선택된 컬럼명을 검사한다.
- 참고           : action/encoding 정규화는 ml_05_fb_encoding_specs에 위임한다.
"""

from __future__ import annotations # 타입 힌트 지연 평가를 활성화한다.

from dataclasses import dataclass  # 검증 결과를 불변 dataclass로 정의하기 위해 사용한다.
from pathlib import Path           # CSV/parquet 경로 검증과 정규화에 사용한다.

import pandas as pd  

from ml_05_fb_encoding_specs import (  # build_action/encoding 정책과 허용값을 가져온다.
    ENCODE_ENCODINGS,                  # encode action에서 허용되는 실제 encoding 방식 목록.
    SUPPORTED_BUILD_ACTIONS,           # contract에서 허용되는 build_action 목록.
    SUPPORTED_ENCODINGS,               # contract에서 허용되는 encoding 목록.
    normalize_build_action_encoding,   # build_action과 encoding 값을 공통 규칙으로 정규화하는 함수.
)
from ml_05_fb_io import parquet_columns  # source parquet의 컬럼 목록을 읽는 함수.
from ml_05_fb_schema import validate_no_forbidden_feature_columns  # label/typology 누수 위험 피처명을 차단하는 함수.




ALLOWED_BOOL_VALUES = {"TRUE", "FALSE"}          # contract CSV에서 bool 값을 문자열 TRUE/FALSE로만 허용한다.
ALLOWED_BUILD_ACTIONS = SUPPORTED_BUILD_ACTIONS  # 허용 build_action 목록을 encoding spec 모듈 기준으로 맞춘다.
ALLOWED_ENCODINGS = SUPPORTED_ENCODINGS          # 허용 encoding 목록을 encoding spec 모듈 기준으로 맞춘다.
SOURCE_BACKED_ACTIONS = {"carry_forward", "encode"}  # source parquet 컬럼 존재 확인이 필요한 action 목록.
CONTRACT_VERSION = 1                             # 현재 FB input contract 형식 버전.

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
    """Summary returned after contract validation."""

    path: Path                   # 검증한 contract CSV 경로.
    total_rows: int              # contract 전체 row 수.
    selected_count: int          # used_in_ml=TRUE로 선택된 컬럼 수.
    selected_columns: list[str]  # ML 입력으로 선택된 컬럼명 목록.


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()  # 사용자 경로를 절대 경로로 정규화한다.
    if not resolved.is_file():                    # 경로가 실제 파일이 아니면 이후 로드를 진행할 수 없다.
        raise FileNotFoundError(f"{label} not found or not a file: {resolved}")
    return resolved  # 검증된 파일 경로를 반환한다.


def _require_contract_columns(table: pd.DataFrame) -> None:
    missing = {"column_name", "used_in_ml"} - set(table.columns)  # contract 검증에 필요한 최소 필수 컬럼 누락 여부를 확인한다.
    if missing:                                                   # 필수 컬럼이 없으면 선택 컬럼을 판단할 수 없다.
        raise ValueError(f"feature contract is missing columns: {sorted(missing)}")

def _require_strict_bool(series: pd.Series, column_name: str) -> pd.Series:
    if series.isna().any():                                # TRUE/FALSE 컬럼에 결측이 있으면 contract가 모호하다.
        rows = (series[series.isna()].index + 2).tolist()  # CSV 헤더를 고려해 실제 row 번호로 변환한다.
        raise ValueError(f"{column_name} contains missing values. csv_rows={rows[:30]}")
    text = series.astype(str)                        # TRUE/FALSE 비교를 위해 문자열로 변환한다.
    invalid = text[~text.isin(ALLOWED_BOOL_VALUES)]  # 허용값이 아닌 항목을 찾는다.
    if not invalid.empty:                            # TRUE/FALSE 외 값이 있으면 엄격히 실패시킨다.
        rows = (invalid.index + 2).tolist()          # CSV에서 문제 row를 찾기 쉽게 row 번호를 만든다.
        raise ValueError(
            f"{column_name} contains unsupported values. "
            f"allowed_values={sorted(ALLOWED_BOOL_VALUES)}, "
            f"invalid_values={sorted(invalid.unique().tolist())[:30]}, csv_rows={rows[:30]}"
        )
    return text.astype("string")  # 검증된 bool 문자열 Series를 반환한다.

def _cell_text(row: pd.Series, column: str) -> str:
    if column not in row.index or pd.isna(row[column]):  # 컬럼이 없거나 값이 비어 있으면 빈 문자열로 처리한다.
        return ""
    return str(row[column]).strip()  # 셀 값을 문자열로 바꾸고 앞뒤 공백을 제거한다.


def _normalize_action_encoding_columns(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return action/encoding values using the same policy as the encoder."""

    actions: list[str] = []              # 정규화된 build_action 값을 누적한다.
    encodings: list[str] = []            # 정규화된 encoding 값을 누적한다.
    for index, row in table.iterrows():  # contract row를 하나씩 순회한다.
        build_action = _cell_text(row, "build_action").lower()  # build_action을 소문자 문자열로 정규화한다.
        encoding = _cell_text(row, "encoding").lower()          # encoding을 소문자 문자열로 정규화한다.
        if not build_action:                                    # build_action이 비어 있으면 encoding 값으로 기본 action을 추론한다.
            build_action = "encode" if encoding and encoding != "passthrough" else "carry_forward"
        build_action, encoding = normalize_build_action_encoding(build_action, encoding, csv_row=int(index) + 2)  # encoder와 같은 규칙으로 action/encoding을 정규화한다.
        actions.append(build_action)  # 정규화된 action을 저장한다.
        encodings.append(encoding)    # 정규화된 encoding을 저장한다.
    return (
        pd.Series(actions, index=table.index, dtype="string"),    # 원래 index에 맞춘 build_action Series.
        pd.Series(encodings, index=table.index, dtype="string"),  # 원래 index에 맞춘 encoding Series.
    )


def _validate_action_encoding_pair(build_action: str, encoding: str, *, csv_row: int) -> None:
    """Validate action/encoding combinations before export."""

    if build_action != "encode":          # encode가 아닌 action은 여기서 encoding 조합 검증 대상이 아니다.
        return
    if encoding not in ENCODE_ENCODINGS:  # encode action은 passthrough가 아닌 실제 category encoding만 허용한다.
        raise ValueError(
            "encode rows must use a real category encoding. "
            f"allowed_encodings={sorted(ENCODE_ENCODINGS)}, encoding={encoding!r}, csv_row={csv_row}"
        )



def _materialized_source_column(row: pd.Series, build_action: str, table: pd.DataFrame) -> str:
    """Return the source column that encode_split_frame() will actually read."""

    if build_action == "carry_forward":  # carry_forward는 column_name 자체를 source parquet에서 읽는다.
        return "" if pd.isna(row["column_name"]) else str(row["column_name"]).strip()
    if build_action == "encode" and "source_column" in table.columns:  # encode는 source_column을 읽어 변환한다.
        return "" if pd.isna(row["source_column"]) else str(row["source_column"]).strip()
    return ""  # source parquet에서 직접 읽을 컬럼이 없는 action은 빈 문자열로 반환한다.



def _selected_columns(table: pd.DataFrame) -> list[str]:
    selected = table.loc[table["used_in_ml"] == "TRUE", "column_name"]  # ML 입력으로 선택된 row의 column_name만 추출한다.
    columns = selected.astype(str).str.strip().tolist()                 # 컬럼명을 문자열로 정리한다.
    blank_rows = [int(index) + 2 for index, column in zip(selected.index, columns) if not column]  # 선택 row 중 빈 column_name 위치를 찾는다.
    if blank_rows:  # 선택된 피처의 이름이 비어 있으면 학습 입력을 만들 수 없다.
        raise ValueError(f"selected feature rows contain blank column_name. csv_rows={blank_rows[:30]}")
    duplicated = sorted({column for column in columns if columns.count(column) > 1})  # 선택 컬럼 중 중복 이름을 찾는다.
    if duplicated:  # 동일 컬럼이 중복 선택되면 학습 입력 스키마가 모호해진다.
        raise ValueError(f"duplicated selected feature columns: {duplicated}")
    validate_no_forbidden_feature_columns(columns)  # 선택 피처명에 label/typology 누수 위험 이름이 있는지 확인한다.
    if not columns:  # 최소 하나 이상의 ML 입력 피처가 필요하다.
        raise ValueError("no selected feature columns found. At least one used_in_ml=TRUE row is required.")
    return columns  # 검증된 선택 피처 컬럼 목록을 반환한다.


def validate_fb_input_contract(
    contract_path: str | Path,
    *,
    source_data_path: str | Path,
    artifact_prefix: str | None = None,
) -> ContractValidationResult:
    """Validate contract CSV syntax and source parquet compatibility."""

    path = _require_file(contract_path, "feature contract")  # contract CSV 파일 경로를 검증한다.
    table = pd.read_csv(       # contract CSV를 읽는다.
        path,
        encoding="utf-8-sig",  # BOM이 있는 UTF-8 CSV도 처리한다.
        dtype={"used_in_ml": "string", "build_in_fb": "string", "materialized": "string"},  # bool-like 컬럼을 문자열로 읽어 엄격 검증한다.
    )
    _require_contract_columns(table)  # 최소 필수 컬럼 존재 여부를 확인한다.
    table["used_in_ml"] = _require_strict_bool(table["used_in_ml"], "used_in_ml")  # used_in_ml 값을 TRUE/FALSE로 엄격 검증한다.

    if artifact_prefix is not None:                 # 특정 artifact_prefix만 허용해야 하는 경우 검증한다.
        if "artifact_prefix" not in table.columns:  # prefix 검증을 하려면 artifact_prefix 컬럼이 필요하다.
            raise ValueError("feature contract is missing artifact_prefix column")
        mismatched = table[table["artifact_prefix"].astype(str) != artifact_prefix]  # 기대 prefix와 다른 row를 찾는다.
        if not mismatched.empty:                    # prefix가 섞여 있으면 잘못된 contract일 수 있다.
            rows = (mismatched.index + 2).tolist()  # 문제가 있는 CSV row 번호를 만든다.
            raise ValueError(f"artifact_prefix mismatch. expected={artifact_prefix!r}, csv_rows={rows[:30]}")

    if "build_in_fb" in table.columns:  # build_in_fb 컬럼이 있으면 TRUE/FALSE 형식을 검증한다.
        _require_strict_bool(table["build_in_fb"], "build_in_fb")
    if "materialized" in table.columns:  # materialized 컬럼이 있으면 TRUE/FALSE 형식을 검증한다.
        _require_strict_bool(table["materialized"], "materialized")
    table = table.copy()                 # 원본 로드 결과를 직접 수정하지 않도록 복사한다.
    table["build_action"], table["encoding"] = _normalize_action_encoding_columns(table)  # action/encoding을 encoder 정책과 동일하게 정규화한다.

    all_columns = table["column_name"].astype(str).str.strip().tolist()  # contract 전체 column_name 목록을 정리한다.
    duplicated = sorted({column for column in all_columns if all_columns.count(column) > 1})  # contract 전체에서 중복 column_name을 찾는다.
    if duplicated:  # column_name이 중복되면 contract row를 유일하게 식별하기 어렵다.
        raise ValueError(f"duplicated contract column_name values: {duplicated}")

    selected = _selected_columns(table)  # ML 입력으로 선택된 컬럼 목록을 검증해 가져온다.
    source_columns = set(parquet_columns(_require_file(source_data_path, "source data parquet")))  # source parquet의 실제 컬럼 목록을 set으로 만든다.
    missing_sources: list[dict[str, str | int]] = []  # source parquet에서 찾지 못한 materialized source 컬럼 정보를 모은다.
    for index, row in table.iterrows():               # contract row를 하나씩 source parquet와 대조한다.
        build_action = str(row["build_action"])       # 정규화된 build_action 값.
        encoding = str(row["encoding"])               # 정규화된 encoding 값.
        _validate_action_encoding_pair(build_action, encoding, csv_row=int(index) + 2)  # action과 encoding 조합이 유효한지 확인한다.
        if build_action not in SOURCE_BACKED_ACTIONS:  # source parquet 컬럼이 필요 없는 action은 존재 검사를 건너뛴다.
            continue
        source_column = _materialized_source_column(row, build_action, table)  # 실제 encode/export 단계에서 읽을 source 컬럼명을 구한다.
        if not source_column or source_column not in source_columns:  # 필요한 source 컬럼이 비어 있거나 parquet에 없으면 기록한다.
            missing_sources.append(
                {
                    "csv_row": int(index) + 2,       # 문제가 발생한 CSV row 번호.
                    "build_action": build_action,    # 해당 row의 build_action.
                    "source_column": source_column,  # 누락된 source 컬럼명.
                }
            )
    if missing_sources:  # source parquet에 없는 컬럼을 참조하면 export가 실패하므로 사전에 중단한다.
        raise ValueError(
            "materialized carry_forward/encode source columns are missing from source parquet. "
            f"missing={missing_sources[:30]}, missing_count={len(missing_sources)}"
        )

    return ContractValidationResult(  # contract 검증 요약 결과를 반환한다.
        path=path,                          # 검증한 contract 경로.
        total_rows=int(len(table)),         # contract 전체 row 수.
        selected_count=int(len(selected)),  # ML 입력으로 선택된 컬럼 수.
        selected_columns=selected,          # ML 입력으로 선택된 컬럼명 목록.
    )