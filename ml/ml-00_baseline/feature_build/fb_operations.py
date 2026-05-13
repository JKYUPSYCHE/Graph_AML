"""
Feature operation 실행 모듈

이 파일의 역할
----------------
1. FeatureSpec에 선언된 operation 이름을 실제 계산 함수로 연결한다.
2. current value, log1p, datetime part, category code, rolling, recency/first, fan-in/out feature를 계산한다.
3. operation 결과가 학습 입력으로 안전한지 검증한다.
4. 생성 feature의 분포/품질 정보를 feature_info 형태로 만든다.
5. category mapping 같은 operation별 부가 산출물을 artifacts로 반환한다.

중요한 설계 원칙
----------------
- 모든 operation은 `df`와 `FeatureSpec`을 입력으로 받는다.
- 모든 operation은 `FeatureOpResult`를 반환한다.
- 잘못된 입력 컬럼, 잘못된 파라미터, NaN/inf/non-numeric 출력은 즉시 에러로 중단한다.
- rolling/fan 계열은 현재 row와 미래 row를 보지 않도록 past-only 정책을 강제한다.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from fb_schema import normalize_category_strict, parse_datetime_strict, parse_numeric_strict
from fb_specs import FeatureOpResult, FeatureSpec, META_COLUMNS, feature_columns, validate_feature_specs


# -----------------------------------------------------------------------------
# 1. 공통 상수와 타입
# -----------------------------------------------------------------------------
# train에서 보지 못한 val/test category는 -1로 인코딩한다.
# 단, 이 값을 조용히 넘기지 않고 category_unknown_summary.csv에 기록한다.
UNKNOWN_CATEGORY_CODE = -1

# category feature가 하나도 선택되지 않아도 header가 있는 빈 CSV를 저장하기 위한 고정 schema다.
CATEGORY_MAPPING_COLUMNS: Tuple[str, ...] = (
    "feature_column",
    "source_column",
    "category_value",
    "encoded_value",
    "fit_split",
)
CATEGORY_UNKNOWN_COLUMNS: Tuple[str, ...] = (
    "feature_column",
    "source_column",
    "split",
    "unknown_count",
    "unknown_unique_count",
    "unknown_examples",
    "policy",
)

# operation 함수의 표준 타입이다. 모든 operation은 같은 입력/출력 계약을 따른다.
OperationRunner = Callable[[pd.DataFrame, FeatureSpec], FeatureOpResult]


# -----------------------------------------------------------------------------
# 2. 공통 검증/보조 함수
# -----------------------------------------------------------------------------
def _json_dumps(payload: Mapping[str, Any]) -> str:
    """feature_info.csv에 input_columns/params를 저장하기 위해 JSON 문자열로 변환한다."""

    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)


def _empty_category_mapping() -> pd.DataFrame:
    """category feature가 없을 때도 header가 있는 빈 mapping DataFrame을 반환한다."""

    return pd.DataFrame(columns=list(CATEGORY_MAPPING_COLUMNS))


def _empty_category_unknown_summary() -> pd.DataFrame:
    """category feature가 없을 때도 header가 있는 빈 unknown summary DataFrame을 반환한다."""

    return pd.DataFrame(columns=list(CATEGORY_UNKNOWN_COLUMNS))


def _require_columns(df: pd.DataFrame, columns: Tuple[str, ...], operation: str) -> None:
    """operation 실행에 필요한 입력 컬럼이 모두 있는지 확인한다."""

    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            "Feature operation failed: input DataFrame is missing required columns. "
            f"operation={operation!r}, missing_columns={sorted(missing)}"
        )


def _require_roles(spec: FeatureSpec, roles: Tuple[str, ...]) -> dict[str, str]:
    """
    FeatureSpec.input_cols에 operation이 요구하는 role이 있는지 확인한다.

    예: log1p는 input_col role이 필요하고, rolling_agg는 entity_col/timestamp_col/value_col이 필요하다.
    """

    missing_roles = set(roles) - set(spec.input_cols)
    if missing_roles:
        raise ValueError(
            "Feature operation failed: FeatureSpec.input_cols is missing roles. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, missing_roles={sorted(missing_roles)}"
        )
    return {role: str(spec.input_cols[role]) for role in roles}


def _require_allowed_params(spec: FeatureSpec, allowed: Tuple[str, ...]) -> None:
    """
    operation별로 허용된 params만 받는다.

    초보자가 잘못된 파라미터명을 넣었을 때 조용히 무시하지 않고 즉시 알려주기 위한 검증이다.
    """

    unknown = sorted(set(spec.params) - set(allowed))
    if unknown:
        raise ValueError(
            "Feature operation failed: unsupported params were provided. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, unknown_params={unknown}, "
            f"allowed_params={list(allowed)}"
        )


def _feature_info(
    features: pd.DataFrame,
    spec: FeatureSpec,
    input_columns: Mapping[str, str],
    params: Mapping[str, Any],
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """
    생성된 feature 컬럼의 분포/품질 정보를 만든다.

    여기서 확인하는 것
    ------------------
    - output_col이 실제로 생성됐는가
    - row 수가 0은 아닌가
    - missing 값이 없는가
    - numeric 변환이 가능한가
    - inf/-inf가 없는가
    - min/median/mean/max 같은 기본 분포가 어떤가
    """

    column = spec.output_col
    if column not in features.columns:
        raise ValueError(
            "Feature operation failed: output column was not created. "
            f"operation={spec.operation!r}, output_col={column!r}"
        )
    if len(features) == 0:
        raise ValueError(f"Feature operation failed: output feature is empty. output_col={column!r}")

    series = features[column]
    numeric = pd.to_numeric(series, errors="coerce")
    missing_count = int(series.isna().sum())
    if missing_count and not allow_missing:
        raise ValueError(
            "Feature operation failed: output feature contains missing values. "
            f"operation={spec.operation!r}, output_col={column!r}, missing_count={missing_count}"
        )
    failed_mask = numeric.isna()
    if failed_mask.any() and not allow_missing:
        examples = series.loc[failed_mask].astype(str).head(5).tolist()
        raise ValueError(
            "Feature operation failed: output feature must be numeric. "
            f"operation={spec.operation!r}, output_col={column!r}, examples={examples}"
        )

    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    inf_count = int(np.isinf(values).sum())
    if inf_count:
        raise ValueError(
            "Feature operation failed: output feature contains inf values. "
            f"operation={spec.operation!r}, output_col={column!r}, inf_count={inf_count}"
        )

    quantiles = numeric.quantile([0.25, 0.5, 0.75])
    # feature_info는 사람이 catalog와 함께 검토하는 품질 정보다.
    # input_columns/params를 함께 남겨 나중에 어떤 설정으로 만든 컬럼인지 추적 가능하게 한다.
    row = {
        "column_name": column,
        "operation": spec.operation,
        "input_columns": _json_dumps(input_columns),
        "params": _json_dumps(params),
        "dtype": str(features[column].dtype),
        "rows": int(len(features)),
        "missing_count": missing_count,
        "missing_rate": float(missing_count / len(features)),
        "inf_count": inf_count,
        "zero_count": int((numeric == 0).sum()),
        "zero_rate": float((numeric == 0).sum() / len(features)),
        "unique_count": int(numeric.nunique(dropna=True)),
        "min": float(numeric.min()),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "mean": float(numeric.mean()),
        "p75": float(quantiles.loc[0.75]),
        "max": float(numeric.max()),
        "near_zero_variance": bool(numeric.nunique(dropna=True) <= 1),
        "leakage_policy": spec.leakage_policy,
        "computational_cost": spec.computational_cost,
    }
    return pd.DataFrame([row])


def _finalize_result(
    series: pd.Series,
    spec: FeatureSpec,
    *,
    row_count: int,
    input_columns: Mapping[str, str],
    params: Mapping[str, Any],
    dtype: str,
    artifacts: Optional[Mapping[str, Any]] = None,
    allow_missing: bool = False,
) -> FeatureOpResult:
    """
    operation 결과 Series를 표준 FeatureOpResult로 마무리한다.

    모든 operation이 같은 검증과 같은 반환 형식을 갖도록 공통 후처리를 담당한다.
    """

    if len(series) != row_count:
        raise ValueError(
            "Feature operation failed: output row count mismatch. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, "
            f"expected_rows={row_count}, observed_rows={len(series)}"
        )
    features = pd.DataFrame({spec.output_col: series.reset_index(drop=True)})
    numeric = pd.to_numeric(features[spec.output_col], errors="coerce")
    if numeric.isna().any() and not allow_missing:
        bad_examples = features.loc[numeric.isna(), spec.output_col].astype(str).head(5).tolist()
        raise ValueError(
            "Feature operation failed: output cannot be converted to numeric dtype. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, examples={bad_examples}"
        )
    features[spec.output_col] = numeric.astype(dtype)
    info = _feature_info(
        features,
        spec,
        input_columns=input_columns,
        params=params,
        allow_missing=allow_missing,
    )
    artifact_payload: Mapping[str, Any] = {} if artifacts is None else artifacts
    return FeatureOpResult(features=features, feature_info=info, artifacts=artifact_payload)


def _param_value(spec: FeatureSpec, name: str, default: Any) -> Any:
    """FeatureSpec.params에서 값을 꺼내고, 없으면 명시한 기본값을 사용한다."""

    if name in spec.params:
        return spec.params[name]
    return default


# -----------------------------------------------------------------------------
# 3. current-row operation
# -----------------------------------------------------------------------------
def op_current_value(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """현재 row의 숫자 컬럼을 그대로 feature로 만든다."""

    _require_allowed_params(spec, ("dtype",))
    roles = _require_roles(spec, ("input_col",))
    input_col = roles["input_col"]
    _require_columns(df, (input_col,), spec.operation)
    values = parse_numeric_strict(df, input_col, spec.output_col)
    dtype = str(_param_value(spec, "dtype", "float32"))
    return _finalize_result(
        values,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype=dtype,
    )


def op_log1p(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """
    현재 row의 숫자 컬럼에 log1p 변환을 적용한다.

    금액처럼 right-skewed된 값의 스케일을 완화할 때 사용한다.
    음수 입력은 log1p 정의상 위험하므로 즉시 중단한다.
    """

    _require_allowed_params(spec, ("dtype",))
    roles = _require_roles(spec, ("input_col",))
    input_col = roles["input_col"]
    _require_columns(df, (input_col,), spec.operation)
    values = parse_numeric_strict(df, input_col, spec.output_col)
    negative_count = int((values < 0).sum())
    if negative_count:
        raise ValueError(
            "Feature operation failed: log1p input contains negative values. "
            f"input_col={input_col!r}, output_col={spec.output_col!r}, negative_count={negative_count}"
        )
    transformed = pd.Series(np.log1p(values), index=df.index)
    dtype = str(_param_value(spec, "dtype", "float32"))
    return _finalize_result(
        transformed,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype=dtype,
    )


def op_datetime_part(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """
    timestamp에서 hour/dayofweek/is_weekend/day/month를 추출한다.

    현재 row의 timestamp만 사용하므로 미래 정보 누수 위험은 낮다.
    """

    _require_allowed_params(spec, ("part", "dtype"))
    roles = _require_roles(spec, ("input_col",))
    input_col = roles["input_col"]
    _require_columns(df, (input_col,), spec.operation)
    part = str(_param_value(spec, "part", "")).strip()
    timestamp = parse_datetime_strict(df, input_col, spec.output_col)

    if part == "hour":
        values = timestamp.dt.hour
        default_dtype = "int16"
    elif part == "dayofweek":
        values = timestamp.dt.dayofweek
        default_dtype = "int16"
    elif part == "is_weekend":
        values = (timestamp.dt.dayofweek >= 5).astype("int8")
        default_dtype = "int8"
    elif part == "day":
        values = timestamp.dt.day
        default_dtype = "int16"
    elif part == "month":
        values = timestamp.dt.month
        default_dtype = "int16"
    else:
        raise ValueError(
            "Feature operation failed: unsupported datetime part. "
            f"part={part!r}, supported=['hour', 'dayofweek', 'is_weekend', 'day', 'month']"
        )

    dtype = str(_param_value(spec, "dtype", default_dtype))
    return _finalize_result(
        pd.Series(values, index=df.index),
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype=dtype,
    )


# -----------------------------------------------------------------------------
# 4. train-only category encoding
# -----------------------------------------------------------------------------
def op_category_code(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """
    범주형 컬럼을 train split에서만 fit한 integer code로 변환한다.

    누수 방지 정책
    ---------------
    - train split category만 mapping에 사용한다.
    - val/test에서 처음 등장한 category는 -1로 인코딩한다.
    - unknown category 수는 category_unknown_summary artifact로 기록한다.
    """

    _require_allowed_params(spec, ("unknown_policy",))
    roles = _require_roles(spec, ("input_col",))
    input_col = roles["input_col"]
    _require_columns(df, (input_col, "split"), spec.operation)
    unknown_policy = str(_param_value(spec, "unknown_policy", "encode_-1_and_report"))
    if unknown_policy != "encode_-1_and_report":
        raise ValueError(
            "Feature operation failed: unsupported unknown category policy. "
            f"unknown_policy={unknown_policy!r}, supported=['encode_-1_and_report']"
        )

    train_mask = df["split"] == "train"
    if not train_mask.any():
        raise ValueError("Feature operation failed: train split is empty; cannot fit category encoding.")

    normalized = normalize_category_strict(df[input_col], source_col=input_col)
    train_values = normalized.loc[train_mask]
    categories = sorted(train_values.unique().tolist())
    if not categories:
        raise ValueError(
            "Feature operation failed: train split has no category values. "
            f"input_col={input_col!r}, output_col={spec.output_col!r}"
        )
    mapping = {category: code for code, category in enumerate(categories)}
    encoded = normalized.map(mapping).fillna(UNKNOWN_CATEGORY_CODE).astype("int32")

    # mapping_frame은 category_mapping_train_only.csv로 저장된다.
    mapping_frame = pd.DataFrame(
        [
            {
                "feature_column": spec.output_col,
                "source_column": input_col,
                "category_value": category,
                "encoded_value": int(code),
                "fit_split": "train",
            }
            for category, code in mapping.items()
        ]
    )

    # split별 unknown category를 기록한다. train unknown은 정상적으로 0이어야 한다.
    unknown_rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        split_mask = df["split"] == split_name
        unknown_mask = split_mask & ~normalized.isin(mapping)
        unknown_values = sorted(normalized.loc[unknown_mask].unique().tolist())
        unknown_rows.append(
            {
                "feature_column": spec.output_col,
                "source_column": input_col,
                "split": split_name,
                "unknown_count": int(unknown_mask.sum()),
                "unknown_unique_count": int(len(unknown_values)),
                "unknown_examples": ";".join(str(value) for value in unknown_values[:5]),
                "policy": unknown_policy,
            }
        )
    unknown_frame = pd.DataFrame(unknown_rows)
    artifacts = {"category_mapping": mapping_frame, "category_unknown_summary": unknown_frame}
    return _finalize_result(
        encoded,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype="int32",
        artifacts=artifacts,
    )


# -----------------------------------------------------------------------------
# 5. temporal / graph helper
# -----------------------------------------------------------------------------
def _parse_window(window: Any, operation: str, output_col: str) -> pd.Timedelta:
    """'7D', '30D' 같은 window 문자열을 pandas Timedelta로 변환하고 양수인지 검사한다."""

    if str(window).strip() == "":
        raise ValueError(
            "Feature operation failed: window parameter must not be empty. "
            f"operation={operation!r}, output_col={output_col!r}"
        )
    parsed = pd.Timedelta(window)
    if parsed <= pd.Timedelta(0):
        raise ValueError(
            "Feature operation failed: window must be positive. "
            f"operation={operation!r}, output_col={output_col!r}, window={window!r}"
        )
    return parsed


# -----------------------------------------------------------------------------
# 6. rolling aggregation operation
# -----------------------------------------------------------------------------
def op_rolling_agg(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """
    entity별 과거 window 집계 feature를 만든다.

    예시
    ----
    sender_account_id 기준 현재 거래 이전 7일 amount 합계:
    `rolling_agg_spec('sender_account_id', 'timestamp', 'amount', ..., window='7D', agg='sum')`

    누수 방지 정책
    ---------------
    - `closed='left'`만 허용한다.
    - 현재 row의 value와 미래 row의 value는 rolling window에 포함하지 않는다.
    """

    _require_allowed_params(spec, ("window", "agg", "closed", "fill_value", "dtype"))
    roles = _require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    entity_col = roles["entity_col"]
    timestamp_col = roles["timestamp_col"]
    value_col = roles["value_col"]
    _require_columns(df, (entity_col, timestamp_col, value_col), spec.operation)

    window = _parse_window(_param_value(spec, "window", ""), spec.operation, spec.output_col)
    agg = str(_param_value(spec, "agg", "")).strip().lower()
    closed = str(_param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: rolling_agg only supports closed='left' to avoid current/future leakage. "
            f"observed_closed={closed!r}"
        )
    supported_aggs = {"sum", "mean", "std", "min", "max", "count"}
    if agg not in supported_aggs:
        raise ValueError(f"Feature operation failed: unsupported rolling agg. agg={agg!r}, supported={sorted(supported_aggs)}")

    # entity는 계좌 ID처럼 범주형 key 역할을 하므로 결측/공백을 엄격히 차단한다.
    entity = normalize_category_strict(df[entity_col], source_col=entity_col)
    timestamps = parse_datetime_strict(df, timestamp_col, spec.output_col)
    values = parse_numeric_strict(df, value_col, spec.output_col)

    # 원본 순서를 보존하기 위해 _row_order를 만든 뒤, entity/timestamp 기준으로 정렬해 rolling을 계산한다.
    work = pd.DataFrame(
        {
            "_entity": entity,
            "_timestamp": timestamps,
            "_value": values.astype("float64"),
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")

    parts: list[pd.DataFrame] = []
    for _entity_value, group in work.groupby("_entity", sort=False):
        # entity별로 timestamp index를 만들고 pandas rolling window를 적용한다.
        # closed='left' 때문에 현재 timestamp row는 집계에서 제외된다.
        indexed = group.set_index("_timestamp")
        rolling = indexed["_value"].rolling(window=window, closed="left")
        if agg == "sum":
            rolled = rolling.sum()
        elif agg == "mean":
            rolled = rolling.mean()
        elif agg == "std":
            rolled = rolling.std()
        elif agg == "min":
            rolled = rolling.min()
        elif agg == "max":
            rolled = rolling.max()
        else:
            rolled = rolling.count()
        parts.append(pd.DataFrame({"_row_order": group["_row_order"].to_numpy(), spec.output_col: rolled.to_numpy()}))

    # entity별로 나뉘어 계산된 결과를 원본 row 순서로 되돌린다.
    result = pd.concat(parts, ignore_index=True).sort_values("_row_order", kind="mergesort")
    values_out = pd.Series(result[spec.output_col].to_numpy(), index=df.index)
    fill_value = _param_value(spec, "fill_value", 0.0)
    values_out = values_out.fillna(fill_value)
    dtype = str(_param_value(spec, "dtype", "float32"))
    params = {"window": str(window), "agg": agg, "closed": closed, "fill_value": fill_value}
    return _finalize_result(
        values_out,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=params,
        dtype=dtype,
    )


# -----------------------------------------------------------------------------
# 7. recency / first transaction operation
# -----------------------------------------------------------------------------
def _entity_recency_parts(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    """
    entity별 직전 과거 timestamp와 첫 거래 flag를 함께 계산한다.

    같은 timestamp에 있는 거래끼리는 서로 과거로 보지 않는다.
    """

    roles = _require_roles(spec, ("entity_col", "timestamp_col"))
    entity_col = roles["entity_col"]
    timestamp_col = roles["timestamp_col"]
    _require_columns(df, (entity_col, timestamp_col), spec.operation)

    entity = normalize_category_strict(df[entity_col], source_col=entity_col)
    timestamps = parse_datetime_strict(df, timestamp_col, spec.output_col)
    work = pd.DataFrame(
        {
            "_entity": entity,
            "_timestamp": timestamps,
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")

    recency = pd.Series(np.nan, index=np.arange(len(df)), dtype="float64")
    is_first = pd.Series(0, index=np.arange(len(df)), dtype="int8")
    for _entity_value, entity_group in work.groupby("_entity", sort=False):
        last_timestamp: Optional[pd.Timestamp] = None
        for timestamp_value, time_group in entity_group.groupby("_timestamp", sort=False):
            row_orders = time_group["_row_order"].to_numpy()
            if last_timestamp is None:
                is_first.iloc[row_orders] = 1
            else:
                delta = timestamp_value - last_timestamp
                recency.iloc[row_orders] = float(delta.total_seconds())
            last_timestamp = timestamp_value

    return recency, is_first, roles


def op_recency_seconds_since_last(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """entity별 직전 과거 거래와의 시간 차이를 초 단위로 계산한다."""

    _require_allowed_params(spec, ("dtype", "fill_value"))
    recency, _is_first, roles = _entity_recency_parts(df, spec)
    fill_value = _param_value(spec, "fill_value", -1.0)
    recency = recency.fillna(fill_value)
    dtype = str(_param_value(spec, "dtype", "float64"))
    params = {"fill_value": fill_value}
    return _finalize_result(
        recency,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=params,
        dtype=dtype,
    )


def op_is_first_by_entity(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """entity 기준 과거 거래가 없으면 1, 있으면 0인 flag를 만든다."""

    _require_allowed_params(spec, ("dtype",))
    _recency, is_first, roles = _entity_recency_parts(df, spec)
    dtype = str(_param_value(spec, "dtype", "int8"))
    return _finalize_result(
        is_first,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype=dtype,
    )


# -----------------------------------------------------------------------------
# 8. fan-in / fan-out 공통 operation
# -----------------------------------------------------------------------------
def _distinct_counterparty_window(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """
    entity별 과거 window 내 고유 counterparty 수를 계산한다.

    fan-in과 fan-out은 방향만 다르고 계산 구조는 같다.
    - fan-in: receiver를 entity로 보고, sender 고유 수를 센다.
    - fan-out: sender를 entity로 보고, receiver 고유 수를 센다.

    같은 timestamp 정책
    --------------------
    같은 timestamp 그룹은 먼저 현재까지의 과거 counts를 출력한 뒤 queue에 넣는다.
    따라서 같은 timestamp 안의 row끼리 서로를 과거로 보지 않는다.
    """

    _require_allowed_params(spec, ("window", "dtype"))
    roles = _require_roles(spec, ("entity_col", "counterparty_col", "timestamp_col"))
    entity_col = roles["entity_col"]
    counterparty_col = roles["counterparty_col"]
    timestamp_col = roles["timestamp_col"]
    _require_columns(df, (entity_col, counterparty_col, timestamp_col), spec.operation)

    window = _parse_window(_param_value(spec, "window", ""), spec.operation, spec.output_col)
    entity = normalize_category_strict(df[entity_col], source_col=entity_col)
    counterparty = normalize_category_strict(df[counterparty_col], source_col=counterparty_col)
    timestamps = parse_datetime_strict(df, timestamp_col, spec.output_col)
    work = pd.DataFrame(
        {
            "_entity": entity,
            "_counterparty": counterparty,
            "_timestamp": timestamps,
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")

    # deque에는 현재 entity의 window 안에 남아 있는 과거 거래 시간과 counterparty를 저장한다.
    # Counter는 window 안 counterparty별 등장 횟수다. len(counts)가 고유 counterparty 수다.
    output = pd.Series(0, index=np.arange(len(df)), dtype="int32")
    for _entity_value, entity_group in work.groupby("_entity", sort=False):
        time_queue: deque[pd.Timestamp] = deque()
        value_queue: deque[str] = deque()
        counts: Counter[str] = Counter()
        for timestamp_value, time_group in entity_group.groupby("_timestamp", sort=False):
            lower_bound = timestamp_value - window
            # window 밖으로 밀려난 과거 거래를 queue와 Counter에서 제거한다.
            while time_queue and time_queue[0] < lower_bound:
                time_queue.popleft()
                old_value = value_queue.popleft()
                counts[old_value] -= 1
                if counts[old_value] <= 0:
                    del counts[old_value]

            # 현재 timestamp 그룹은 아직 queue에 넣기 전이므로, 현재/동시점 거래가 포함되지 않는다.
            output.iloc[time_group["_row_order"].to_numpy()] = len(counts)

            # 현재 timestamp 그룹을 다음 timestamp의 과거 거래로 사용하기 위해 queue에 넣는다.
            for event_time, counterparty_value in zip(time_group["_timestamp"], time_group["_counterparty"]):
                time_queue.append(event_time)
                value_queue.append(counterparty_value)
                counts[counterparty_value] += 1

    dtype = str(_param_value(spec, "dtype", "int32"))
    params = {"window": str(window)}
    return _finalize_result(
        output,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=params,
        dtype=dtype,
    )


def op_fan_in(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """receiver 기준 과거 window 내 고유 sender 수를 계산한다."""

    return _distinct_counterparty_window(df, spec)


def op_fan_out(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """sender 기준 과거 window 내 고유 receiver 수를 계산한다."""

    return _distinct_counterparty_window(df, spec)


# -----------------------------------------------------------------------------
# 9. operation registry와 실행기
# -----------------------------------------------------------------------------
# FeatureSpec.operation 문자열을 실제 함수로 연결하는 테이블이다.
# 새 operation을 추가할 때는 함수 작성 후 이 registry에 등록해야 한다.
OPERATION_REGISTRY: dict[str, OperationRunner] = {
    "current_value": op_current_value,
    "log1p": op_log1p,
    "datetime_part": op_datetime_part,
    "category_code": op_category_code,
    "rolling_agg": op_rolling_agg,
    "recency_seconds_since_last": op_recency_seconds_since_last,
    "is_first_by_entity": op_is_first_by_entity,
    "fan_in": op_fan_in,
    "fan_out": op_fan_out,
}


def run_operation(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """FeatureSpec.operation 이름을 registry에서 찾아 실제 operation을 실행한다."""

    if spec.operation not in OPERATION_REGISTRY:
        raise ValueError(
            "Feature build failed: unknown operation. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, "
            f"supported_operations={sorted(OPERATION_REGISTRY)}"
        )
    return OPERATION_REGISTRY[spec.operation](df, spec)


def execute_feature_specs(
    df: pd.DataFrame,
    feature_specs: Tuple[FeatureSpec, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    선택된 FeatureSpec 목록을 순서대로 실행하고 최종 feature frame을 만든다.

    반환값
    ------
    feature_frame:
        tx_id, split, label과 생성 feature 컬럼을 합친 학습 입력 DataFrame이다.
    feature_info:
        생성 feature별 분포/품질/파라미터 정보다.
    artifacts:
        category mapping, unknown category summary 같은 부가 산출물이다.
    """

    validate_feature_specs(feature_specs)
    missing_meta = set(META_COLUMNS) - set(df.columns)
    if missing_meta:
        raise ValueError(f"Feature execution input is missing metadata columns: {sorted(missing_meta)}")

    feature_parts: list[pd.DataFrame] = []
    feature_info_parts: list[pd.DataFrame] = []
    category_mapping_parts: list[pd.DataFrame] = []
    category_unknown_parts: list[pd.DataFrame] = []
    for spec in feature_specs:
        # FeatureSpec 하나당 operation 하나를 실행한다.
        result = run_operation(df, spec)
        expected = [spec.output_col]

        # operation이 선언한 output_col만 정확히 만들었는지 확인한다.
        if list(result.features.columns) != expected:
            raise ValueError(
                "Feature operation failed: result columns do not match spec output_col exactly. "
                f"operation={spec.operation!r}, expected={expected}, observed={list(result.features.columns)}"
            )
        if len(result.features) != len(df):
            raise ValueError(
                "Feature operation failed: result row count differs from input. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}, "
                f"input_rows={len(df)}, output_rows={len(result.features)}"
            )
        feature_parts.append(result.features.reset_index(drop=True))
        feature_info_parts.append(result.feature_info.reset_index(drop=True))

        # category_code 같은 operation은 추가 artifact를 반환한다.
        if "category_mapping" in result.artifacts:
            category_mapping_parts.append(result.artifacts["category_mapping"])
        if "category_unknown_summary" in result.artifacts:
            category_unknown_parts.append(result.artifacts["category_unknown_summary"])

    selected_columns = feature_columns(feature_specs)

    # 기존 학습 모듈과 연결하기 위해 meta columns를 앞에 두고 feature columns를 뒤에 붙인다.
    feature_frame = pd.concat([df.loc[:, list(META_COLUMNS)].reset_index(drop=True), *feature_parts], axis=1)
    feature_frame["label"] = feature_frame["label"].astype("int8")
    feature_frame["split"] = feature_frame["split"].astype("string")
    feature_info = pd.concat(feature_info_parts, ignore_index=True)
    artifacts = {
        "category_mapping": pd.concat(category_mapping_parts, ignore_index=True) if category_mapping_parts else _empty_category_mapping(),
        "category_unknown_summary": pd.concat(category_unknown_parts, ignore_index=True)
        if category_unknown_parts
        else _empty_category_unknown_summary(),
    }
    if list(feature_frame.columns) != [*META_COLUMNS, *selected_columns]:
        raise ValueError(
            "Feature build failed: final feature frame column order mismatch. "
            f"observed={list(feature_frame.columns)}, expected={[*META_COLUMNS, *selected_columns]}"
        )
    return feature_frame, feature_info, artifacts
