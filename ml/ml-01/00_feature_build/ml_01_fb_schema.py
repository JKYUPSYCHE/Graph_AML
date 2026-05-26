"""
Feature build 입력 schema 처리 모듈

이 파일의 역할
----------------
1. 사용자가 FeatureSpec에 적은 입력 컬럼명을 실제 parquet 컬럼명으로 resolve한다.
2. timestamp, numeric, category 컬럼을 엄격하게 파싱한다.
3. feature build 내부에서 쓰는 표준 메타 컬럼(tx_id, timestamp, label)을 만든다.
4. label/pattern 계열 컬럼이 feature 입력/출력에 들어가는 것을 차단한다.

중요한 전제
-----------
- clean_base.parquet는 전처리 버전에 따라 컬럼명이 조금 다를 수 있다.
- 따라서 `amount` 같은 논리 이름을 `Amount Paid` 같은 실제 컬럼명으로 매핑할 수 있게 한다.
- 단, 파싱 실패/결측/공백은 조용히 넘기지 않고 즉시 ValueError로 중단한다.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd


# -----------------------------------------------------------------------------
# 1. 입력 컬럼 후보 목록
# -----------------------------------------------------------------------------
# key는 사용자가 FeatureSpec에서 쓰는 권장 컬럼명이다.
# value는 실제 parquet에 존재할 수 있는 후보 컬럼명이다.
# 예: FeatureSpec에는 "amount"라고 쓰되, 실제 파일에는 "Amount Paid"가 있어도 연결한다.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "tx_id": [
        "tx_id",
        "transaction_id",
        "Transaction ID",
        "Transaction_ID",
        "TransactionID",
        "transaction_id_raw",
    ],
    "timestamp": [
        "timestamp",
        "Timestamp",
        "transaction_time",
        "Transaction Time",
        "Transaction_Time",
        "time",
        "Time",
        "datetime",
        "DateTime",
    ],
    "sender_bank_id": [
        "sender_bank_id",
        "from_bank",
        "From Bank",
        "From_Bank",
        "sender_bank",
        "source_bank",
    ],
    "receiver_bank_id": [
        "receiver_bank_id",
        "to_bank",
        "To Bank",
        "To_Bank",
        "receiver_bank",
        "destination_bank",
    ],
    "sender_account_id": [
        "sender_account_id",
        "sender_account",
        "from_account_id",
        "from_account",
        "From Account",
        "From_Account",
        "Account",
        "orig_account",
        "source_account",
    ],
    "receiver_account_id": [
        "receiver_account_id",
        "receiver_account",
        "to_account_id",
        "to_account",
        "To Account",
        "To_Account",
        "Account.1",
        "Account_1",
        "beneficiary_account",
        "dest_account",
        "destination_account",
    ],
    "amount": [
        "amount",
        "Amount",
        "amount_paid",
        "Amount Paid",
        "Amount_Paid",
        "paid_amount",
    ],
    "amount_received": [
        "amount_received",
        "Amount Received",
        "Amount_Received",
        "received_amount",
        "recv_amount",
    ],
    "label": ["is_laundering", "label", "Is Laundering", "Is_Laundering", "target", "y"],
    "payment_currency": [
        "payment_currency",
        "Payment Currency",
        "Payment_Currency",
        "currency",
        "Currency",
        "amount_currency",
    ],
    "receiving_currency": [
        "receiving_currency",
        "Receiving Currency",
        "Receiving_Currency",
        "received_currency",
    ],
    "payment_format": [
        "payment_format",
        "Payment Format",
        "Payment_Format",
        "payment_type",
        "Payment Type",
        "format",
    ],
}

# feature 이름이나 input column에 들어오면 누수 위험이 큰 이름들이다.
# label 자체나 laundering/pattern 계열 컬럼이 모델 입력으로 들어가면 성능이 허위로 좋아질 수 있다.
# exact name은 완전 일치만 차단하고, substring은 패턴/유형명처럼 컬럼명 일부에 숨어 있는 경우까지 차단한다.
FORBIDDEN_EXACT_NAMES = {"label", "target", "y", "is_laundering", "is laundering"}
FORBIDDEN_SUBSTRINGS = {"laundering", "pattern", "typology", "attempt"}


def _example_values(series: pd.Series, mask: pd.Series, limit: int = 5) -> list[str]:
    """에러 메시지에 표시할 문제 값 예시를 만든다."""

    return series.loc[mask].astype(str).head(limit).tolist()


# -----------------------------------------------------------------------------
# 2. 컬럼명 resolve
# -----------------------------------------------------------------------------
def resolve_requested_column(
    available_columns: set[str],
    requested_col: str,
    column_map: Optional[Mapping[str, str]] = None,
) -> str:
    """
    사용자가 요청한 컬럼명을 실제 입력 파일의 컬럼명으로 변환한다.

    동작 순서
    ---------
    1. column_map에 명시된 실제 컬럼명이 있으면 그 값을 최우선 사용한다.
    2. requested_col이 실제 parquet 컬럼에 있으면 그대로 사용한다.
    3. 없으면 COLUMN_CANDIDATES에서 후보 컬럼명을 찾아본다.
    4. 후보도 없으면 어떤 컬럼이 필요한지 명시한 ValueError로 중단한다.
    """

    requested_col = str(requested_col).strip()
    if column_map is not None and requested_col in column_map:
        mapped_col = str(column_map[requested_col]).strip()
        if not mapped_col:
            raise ValueError(f"column_map has blank source column. requested_col={requested_col!r}")
        if mapped_col not in available_columns:
            raise ValueError(
                "Feature build failed: column_map points to a missing source column. "
                f"requested_col={requested_col!r}, mapped_col={mapped_col!r}, "
                f"available_columns={sorted(available_columns)[:80]}"
            )
        return mapped_col

    if requested_col in available_columns:
        return requested_col

    candidates = COLUMN_CANDIDATES.get(requested_col, [])
    for candidate in candidates:
        if candidate in available_columns:
            return candidate

    raise ValueError(
        "Feature build failed: required input column is missing. "
        f"requested_col={requested_col!r}, candidates={candidates}, "
        f"available_count={len(available_columns)}, "
        "fix=Check input schema or edit FeatureSpec.input_cols."
    )


def resolve_requested_columns(
    columns: Iterable[str],
    requested_columns: Iterable[str],
    column_map: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """여러 requested column을 한 번에 resolve하고 requested -> source mapping을 반환한다.

    반환 dict의 key는 FeatureSpec이 사용하는 logical column이고, value는 실제 source parquet 컬럼명이다.
    이 mapping은 build_summary.json에도 저장되어 이후 실행 재현과 schema diff 확인에 사용된다.
    """

    available_columns = set(str(column) for column in columns)
    requested = list(dict.fromkeys(str(column).strip() for column in requested_columns))
    if not requested:
        raise ValueError("requested_columns must not be empty.")
    return {column: resolve_requested_column(available_columns, column, column_map=column_map) for column in requested}


# -----------------------------------------------------------------------------
# 3. strict parsing 함수
# -----------------------------------------------------------------------------
def parse_datetime_strict(df: pd.DataFrame, source_col: str, logical_name: str) -> pd.Series:
    """
    timestamp 컬럼을 datetime으로 변환한다.

    결측이나 파싱 실패가 있으면 이후 feature가 잘못 계산되므로 즉시 중단한다.
    """

    raw = df[source_col]
    missing_mask = raw.isna()
    if missing_mask.any():
        raise ValueError(
            "Feature build failed: timestamp column has missing values. "
            f"logical_name={logical_name!r}, source_column={source_col!r}, "
            f"missing_count={int(missing_mask.sum())}"
        )

    parsed = pd.to_datetime(raw, errors="coerce")
    failed_mask = parsed.isna()
    if failed_mask.any():
        raise ValueError(
            "Feature build failed: timestamp parsing failed. "
            f"logical_name={logical_name!r}, source_column={source_col!r}, "
            f"failed_count={int(failed_mask.sum())}, "
            f"example_values={_example_values(raw, failed_mask)}"
        )
    return parsed


def parse_numeric_strict(df: pd.DataFrame, source_col: str, logical_name: str) -> pd.Series:
    """
    숫자 컬럼을 numeric으로 변환한다.

    금액/라벨 같은 컬럼에 문자열 쓰레기 값이 섞여 있으면 조용히 NaN으로 두지 않고 중단한다.
    """

    raw = df[source_col]
    missing_mask = raw.isna()
    if missing_mask.any():
        raise ValueError(
            "Feature build failed: numeric column has missing values. "
            f"logical_name={logical_name!r}, source_column={source_col!r}, "
            f"missing_count={int(missing_mask.sum())}"
        )

    parsed = pd.to_numeric(raw, errors="coerce")
    failed_mask = parsed.isna()
    if failed_mask.any():
        raise ValueError(
            "Feature build failed: numeric parsing failed. "
            f"logical_name={logical_name!r}, source_column={source_col!r}, "
            f"failed_count={int(failed_mask.sum())}, "
            f"example_values={_example_values(raw, failed_mask)}"
        )
    return parsed


def normalize_category_strict(series: pd.Series, source_col: str) -> pd.Series:
    """
    범주형 값을 문자열로 정규화한다.

    결측/공백 category를 임의 토큰으로 바꾸면 원인 파악이 어려우므로 즉시 중단한다.
    """

    missing_mask = series.isna()
    if missing_mask.any():
        raise ValueError(
            "Feature build failed: category column has missing values. "
            f"source_column={source_col!r}, missing_count={int(missing_mask.sum())}"
        )

    normalized = series.astype("string").str.strip()
    blank_mask = normalized == ""
    if blank_mask.any():
        raise ValueError(
            "Feature build failed: category column has blank values. "
            f"source_column={source_col!r}, blank_count={int(blank_mask.sum())}, "
            f"example_values={_example_values(series, blank_mask)}"
        )
    return normalized


# -----------------------------------------------------------------------------
# 4. 입력 DataFrame 표준화
# -----------------------------------------------------------------------------
def standardize_input_frame(
    df: pd.DataFrame,
    column_map: Mapping[str, str],
    *,
    tx_id_col: str,
    timestamp_col: str,
    label_col: str,
) -> pd.DataFrame:
    """
    원본 입력 DataFrame을 feature build용 DataFrame으로 표준화한다.

    처리 내용
    ---------
    1. column_map에 포함된 실제 source column만 복사한다.
    2. tx_id, timestamp, label 메타 컬럼을 표준 이름으로 만든다.
    3. timestamp와 label을 strict parsing한다.
    4. label이 0/1 이진값인지 확인한다.
    """

    standardized = pd.DataFrame(index=df.index)
    for requested_col, source_col in column_map.items():
        # requested_col 이름으로 통일해 이후 operation이 실제 원본 컬럼명을 몰라도 되게 한다.
        standardized[requested_col] = df[source_col]

    required_meta = {tx_id_col, timestamp_col, label_col}
    missing_meta = required_meta - set(standardized.columns)
    if missing_meta:
        raise ValueError(
            "Feature build failed: canonical metadata source columns were not loaded. "
            f"missing_meta={sorted(missing_meta)}"
        )

    tx_id = standardized[tx_id_col]
    if tx_id.isna().any():
        raise ValueError(
            "Feature build failed: tx_id column has missing values. "
            f"source_column={tx_id_col!r}, missing_count={int(tx_id.isna().sum())}"
        )

    duplicated_tx_id = tx_id.astype("string").duplicated(keep=False)
    if duplicated_tx_id.any():
        examples = tx_id.loc[duplicated_tx_id].astype(str).head(10).tolist()
        raise ValueError(
            "Feature build failed: tx_id column has duplicated values. "
            f"source_column={tx_id_col!r}, duplicated_count={int(duplicated_tx_id.sum())}, "
            f"examples={examples}"
        )
    standardized["tx_id"] = tx_id
    standardized["timestamp"] = parse_datetime_strict(standardized, timestamp_col, "timestamp")

    label = parse_numeric_strict(standardized, label_col, "label")
    label_values = sorted(label.dropna().unique().tolist())
    if not set(label_values).issubset({0, 1}):
        raise ValueError(
            "Feature build failed: label must be binary 0/1. "
            f"observed_values={label_values[:20]}"
        )
    standardized["label"] = label.astype("int8")
    if standardized.index.equals(pd.RangeIndex(len(standardized))):
        return standardized
    return standardized.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 5. 누수 위험 컬럼명 차단
# -----------------------------------------------------------------------------
def validate_no_forbidden_feature_columns(feature_columns: Iterable[str]) -> None:
    """생성 feature 이름에 label/pattern 계열 누수 위험 이름이 없는지 검사한다.

    output feature 이름만 검사한다. source column 누수 여부는 validate_no_forbidden_input_columns()에서 별도로 확인한다.
    """

    forbidden_exact = {name.lower() for name in FORBIDDEN_EXACT_NAMES}
    violations: list[str] = []
    for column in feature_columns:
        normalized = str(column).strip().lower()
        if normalized in forbidden_exact:
            violations.append(str(column))
        elif any(pattern in normalized for pattern in FORBIDDEN_SUBSTRINGS):
            violations.append(str(column))
    if violations:
        raise ValueError(
            "Data leakage risk: forbidden feature column names were selected. "
            f"violations={violations[:30]}, violation_count={len(violations)}"
        )


def validate_no_forbidden_input_columns(input_columns: Iterable[str]) -> None:
    """FeatureSpec.input_cols가 label/pattern 계열 컬럼을 직접 사용하지 않는지 검사한다.

    logical column과 resolved source column 양쪽에서 호출해, ``amount`` 같은 logical 이름이
    실수로 ``Is Laundering`` source column에 매핑되는 경우까지 차단한다.
    """

    forbidden_exact = {name.lower() for name in FORBIDDEN_EXACT_NAMES}
    violations: list[str] = []
    for column in input_columns:
        normalized = str(column).strip().lower()
        if normalized in forbidden_exact:
            violations.append(str(column))
        elif any(pattern in normalized for pattern in FORBIDDEN_SUBSTRINGS):
            violations.append(str(column))
    if violations:
        raise ValueError(
            "Data leakage risk: forbidden input columns were used by feature_specs. "
            f"violations={violations[:30]}, violation_count={len(violations)}"
        )
