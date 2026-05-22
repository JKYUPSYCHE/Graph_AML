"""
Feature operation 실행 모듈

이 파일의 역할
----------------
1. FeatureSpec에 선언된 operation 이름을 실제 계산 함수로 연결한다.
2. ML-01 Stage 0에 필요한 rolling, recency/first, ratio, cumulative count feature를 계산한다.
3. operation 결과가 학습 입력으로 안전한지 검증한다.
4. 생성 feature의 분포/품질 정보를 feature_info 형태로 만든다.
5. 생성 feature 순서와 feature_info 계약을 검증한다.

중요한 설계 원칙
----------------
- 모든 operation은 `df`와 `FeatureSpec`을 입력으로 받는다.
- 모든 operation은 `FeatureOpResult`를 반환한다.
- 잘못된 입력 컬럼, 잘못된 파라미터, NaN/inf/non-numeric 출력은 즉시 에러로 중단한다.
- rolling/time-history 계열은 현재 row와 미래 row를 보지 않도록 past-only 정책을 강제한다.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ml_01_fb_schema import normalize_category_strict, parse_datetime_strict, parse_numeric_strict
from ml_01_fb_specs import FeatureOpResult, FeatureSpec, META_COLUMNS, feature_columns, validate_feature_specs
from ml_01_fb_rolling import execute_rolling_agg_specs_batched, op_rolling_agg, parse_window


# -----------------------------------------------------------------------------
# 1. 공통 상수와 타입
# -----------------------------------------------------------------------------
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

    예: rolling_agg는 entity_col/timestamp_col/value_col role이 필요하다.
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
# 3. recency / first transaction operation
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

    # 같은 entity/timestamp에 여러 거래가 있어도 서로를 직전 거래로 보지 않는다.
    # timestamp 단위로 먼저 중복 제거한 뒤 이전 timestamp를 계산한다.
    timestamp_frame = work[["_entity", "_timestamp"]].drop_duplicates(["_entity", "_timestamp"], keep="first")
    previous_timestamp = timestamp_frame.groupby("_entity", sort=False)["_timestamp"].shift(1)
    timestamp_frame["_recency"] = (timestamp_frame["_timestamp"] - previous_timestamp).dt.total_seconds()
    timestamp_frame["_is_first"] = previous_timestamp.isna().astype("int8")

    work = work.merge(timestamp_frame, on=["_entity", "_timestamp"], how="left", sort=False)
    row_orders = work["_row_order"].to_numpy()
    recency = pd.Series(np.nan, index=np.arange(len(df)), dtype="float64")
    recency.iloc[row_orders] = pd.to_numeric(work["_recency"], errors="coerce").to_numpy(dtype="float64")
    is_first = pd.Series(0, index=np.arange(len(df)), dtype="int8")
    is_first.iloc[row_orders] = work["_is_first"].to_numpy(dtype="int8")

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


def _recency_group_key(spec: FeatureSpec) -> tuple[str, str]:
    roles = _require_roles(spec, ("entity_col", "timestamp_col"))
    return roles["entity_col"], roles["timestamp_col"]


def execute_recency_specs_batched(
    df: pd.DataFrame,
    specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """같은 entity/timestamp를 쓰는 recency 계열 spec을 한 번의 정렬로 함께 실행한다.

    seconds_since_last와 is_first는 같은 중간 결과에서 파생된다. 따로 실행하면 같은 정렬과 merge를
    반복하므로, batch 실행으로 계산 비용을 줄이되 output_col별 FeatureOpResult 계약은 유지한다.
    """

    if not specs:
        return {}
    validate_feature_specs(specs)

    grouped_specs: dict[tuple[str, str], list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation not in {"recency_seconds_since_last", "is_first_by_entity"}:
            raise ValueError(
                "Feature build failed: execute_recency_specs_batched only accepts recency specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_recency_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        recency, is_first, roles = _entity_recency_parts(df, group_specs[0])
        for spec in group_specs:
            if spec.operation == "recency_seconds_since_last":
                _require_allowed_params(spec, ("dtype", "fill_value"))
                fill_value = _param_value(spec, "fill_value", -1.0)
                dtype = str(_param_value(spec, "dtype", "float64"))
                result = _finalize_result(
                    recency.fillna(fill_value),
                    spec,
                    row_count=len(df),
                    input_columns=roles,
                    params={"fill_value": fill_value},
                    dtype=dtype,
                )
            else:
                _require_allowed_params(spec, ("dtype",))
                dtype = str(_param_value(spec, "dtype", "int8"))
                result = _finalize_result(
                    is_first,
                    spec,
                    row_count=len(df),
                    input_columns=roles,
                    params=spec.params,
                    dtype=dtype,
                )
            if spec.output_col in results:
                raise ValueError(f"Feature build failed: duplicate recency batch result. output_col={spec.output_col!r}")
            results[spec.output_col] = result
    return results


# -----------------------------------------------------------------------------
# 4. ML-01 Stage 0 time-history operation
# -----------------------------------------------------------------------------
def op_cur_vs_mean_ratio(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """현재 금액을 entity별 과거 window 평균 금액으로 나눈 비율을 계산한다.

    단일 operation 실행(run_operation)용 구현이다. 공식 feature build 경로에서는
    rolling mean을 batch로 먼저 계산한 뒤 _execute_cur_vs_mean_ratio_from_mean()을 사용한다.
    """

    _require_allowed_params(spec, ("window", "closed", "fill_value", "zero_division_value", "dtype"))
    roles = _require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    entity_col = roles["entity_col"]
    timestamp_col = roles["timestamp_col"]
    value_col = roles["value_col"]
    _require_columns(df, (entity_col, timestamp_col, value_col), spec.operation)

    window = _param_value(spec, "window", "")
    closed = str(_param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: cur_vs_mean_ratio only supports closed='left' to avoid current/future leakage. "
            f"observed_closed={closed!r}"
        )
    zero_division_value = float(_param_value(spec, "zero_division_value", 0.0))
    fill_value = float(_param_value(spec, "fill_value", 0.0))
    dtype = str(_param_value(spec, "dtype", "float32"))

    rolling_mean_spec = FeatureSpec(
        operation="rolling_agg",
        output_col=f"__rolling_mean_for__{spec.output_col}",
        input_cols=roles,
        params={"window": window, "agg": "mean", "closed": closed, "fill_value": fill_value, "dtype": "float64"},
        leakage_policy=spec.leakage_policy,
    )
    rolling_mean = op_rolling_agg(df, rolling_mean_spec).features[rolling_mean_spec.output_col].astype("float64")
    current_value = parse_numeric_strict(df, value_col, spec.output_col).reset_index(drop=True).astype("float64")

    ratio = pd.Series(zero_division_value, index=np.arange(len(df)), dtype="float64")
    valid_denominator = rolling_mean.notna() & (rolling_mean != 0)
    ratio.loc[valid_denominator] = current_value.loc[valid_denominator] / rolling_mean.loc[valid_denominator]
    ratio = ratio.replace([np.inf, -np.inf], zero_division_value)
    params = {
        "window": str(parse_window(window, spec.operation, spec.output_col)),
        "closed": closed,
        "fill_value": fill_value,
        "zero_division_value": zero_division_value,
    }
    return _finalize_result(
        ratio,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=params,
        dtype=dtype,
    )


def _rolling_mean_spec_for_ratio(spec: FeatureSpec) -> FeatureSpec:
    """cur_vs_mean_ratio 계산에 필요한 내부 rolling mean spec을 만든다.

    사용자가 선택한 feature로 저장되지는 않지만, rolling_agg batch 계산을 재사용하기 위해
    임시 FeatureSpec 형태로 만든다.
    """

    _require_allowed_params(spec, ("window", "closed", "fill_value", "zero_division_value", "dtype"))
    roles = _require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    window = _param_value(spec, "window", "")
    closed = str(_param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: cur_vs_mean_ratio only supports closed='left' to avoid current/future leakage. "
            f"observed_closed={closed!r}"
        )
    fill_value = float(_param_value(spec, "fill_value", 0.0))
    return FeatureSpec(
        operation="rolling_agg",
        output_col=f"__rolling_mean_for__{spec.output_col}",
        input_cols=roles,
        params={"window": window, "agg": "mean", "closed": closed, "fill_value": fill_value, "dtype": "float64"},
        leakage_policy=spec.leakage_policy,
        used_in_ml=False,
    )


def _execute_cur_vs_mean_ratio_from_mean(
    df: pd.DataFrame,
    spec: FeatureSpec,
    rolling_mean: pd.Series,
) -> FeatureOpResult:
    """캐시된 rolling mean 결과를 사용해 cur_vs_mean_ratio를 계산한다.

    hidden rolling mean을 별도로 저장하지 않고 ratio output만 FeatureOpResult로 반환한다.
    denominator가 0이거나 비어 있으면 spec의 zero_division_value 정책을 따른다.
    """

    _require_allowed_params(spec, ("window", "closed", "fill_value", "zero_division_value", "dtype"))
    roles = _require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    value_col = roles["value_col"]
    _require_columns(df, (roles["entity_col"], roles["timestamp_col"], value_col), spec.operation)

    window = _param_value(spec, "window", "")
    closed = str(_param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: cur_vs_mean_ratio only supports closed='left' to avoid current/future leakage. "
            f"observed_closed={closed!r}"
        )
    zero_division_value = float(_param_value(spec, "zero_division_value", 0.0))
    fill_value = float(_param_value(spec, "fill_value", 0.0))
    dtype = str(_param_value(spec, "dtype", "float32"))

    if len(rolling_mean) != len(df):
        raise ValueError(
            "Feature operation failed: cached rolling mean row count mismatch. "
            f"output_col={spec.output_col!r}, expected_rows={len(df)}, observed_rows={len(rolling_mean)}"
        )
    rolling_mean = pd.to_numeric(rolling_mean.reset_index(drop=True), errors="coerce").astype("float64")
    current_value = parse_numeric_strict(df, value_col, spec.output_col).reset_index(drop=True).astype("float64")

    ratio = pd.Series(zero_division_value, index=np.arange(len(df)), dtype="float64")
    valid_denominator = rolling_mean.notna() & (rolling_mean != 0)
    ratio.loc[valid_denominator] = current_value.loc[valid_denominator] / rolling_mean.loc[valid_denominator]
    ratio = ratio.replace([np.inf, -np.inf], zero_division_value)
    params = {
        "window": str(parse_window(window, spec.operation, spec.output_col)),
        "closed": closed,
        "fill_value": fill_value,
        "zero_division_value": zero_division_value,
    }
    return _finalize_result(
        ratio,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=params,
        dtype=dtype,
    )


def op_cumulative_count(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """entity별 현재 timestamp 이전 누적 거래 건수를 계산한다."""

    _require_allowed_params(spec, ("dtype",))
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

    # 같은 timestamp group은 현재 시점 거래로 보고 history에서 제외한다.
    # group size를 먼저 구한 뒤 cumsum에서 현재 group size를 빼면 past_timestamp < current_timestamp 정책이 유지된다.
    timestamp_counts = work.groupby(["_entity", "_timestamp"], sort=False).size().rename("_group_size").reset_index()
    timestamp_counts["_history_count"] = (
        timestamp_counts.groupby("_entity", sort=False)["_group_size"].cumsum() - timestamp_counts["_group_size"]
    )
    work = work.merge(timestamp_counts[["_entity", "_timestamp", "_history_count"]], on=["_entity", "_timestamp"], how="left", sort=False)

    output = pd.Series(0, index=np.arange(len(df)), dtype="int64")
    output.iloc[work["_row_order"].to_numpy()] = work["_history_count"].to_numpy(dtype="int64")

    dtype = str(_param_value(spec, "dtype", "int32"))
    return _finalize_result(
        output,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype=dtype,
    )


# -----------------------------------------------------------------------------
# 6. operation registry와 실행기
# -----------------------------------------------------------------------------
# FeatureSpec.operation 문자열을 실제 함수로 연결하는 테이블이다.
# 새 operation을 추가할 때는 함수 작성 후 이 registry에 등록해야 한다.
OPERATION_REGISTRY: dict[str, OperationRunner] = {
    "rolling_agg": op_rolling_agg,
    "recency_seconds_since_last": op_recency_seconds_since_last,
    "is_first_by_entity": op_is_first_by_entity,
    "cur_vs_mean_ratio": op_cur_vs_mean_ratio,
    "cumulative_count": op_cumulative_count,
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
        operation별 부가 산출물이다. 현재 ML-01 Stage 0에서는 빈 DataFrame만 반환한다.
    """

    validate_feature_specs(feature_specs)
    missing_meta = set(META_COLUMNS) - set(df.columns)
    if missing_meta:
        raise ValueError(f"Feature execution input is missing metadata columns: {sorted(missing_meta)}")

    feature_parts: list[pd.DataFrame] = []
    feature_info_parts: list[pd.DataFrame] = []
    category_mapping_parts: list[pd.DataFrame] = []
    category_unknown_parts: list[pd.DataFrame] = []
    # rolling_agg와 cur_vs_mean_ratio는 같은 rolling 계산을 공유할 수 있다.
    # ratio는 hidden rolling mean spec을 만들어 batch에 함께 넣고, 최종 loop에서는 ratio 컬럼만 반환한다.
    rolling_specs = tuple(spec for spec in feature_specs if spec.operation == "rolling_agg")
    ratio_specs = tuple(spec for spec in feature_specs if spec.operation == "cur_vs_mean_ratio")
    ratio_mean_specs = {spec.output_col: _rolling_mean_spec_for_ratio(spec) for spec in ratio_specs}
    rolling_results = execute_rolling_agg_specs_batched(df, (*rolling_specs, *ratio_mean_specs.values()))

    # recency seconds와 is_first flag도 같은 entity/timestamp 정렬 결과를 공유한다.
    recency_specs = tuple(
        spec for spec in feature_specs if spec.operation in {"recency_seconds_since_last", "is_first_by_entity"}
    )
    recency_results = execute_recency_specs_batched(df, recency_specs)
    for spec in feature_specs:
        # rolling_agg는 같은 history 계산을 공유하기 위해 batch 결과를 사용한다.
        if spec.operation == "rolling_agg":
            result = rolling_results[spec.output_col]
        elif spec.operation == "cur_vs_mean_ratio":
            # hidden rolling mean은 사용자가 보는 feature_frame에 포함하지 않고 ratio 계산에만 사용한다.
            mean_spec = ratio_mean_specs[spec.output_col]
            result = _execute_cur_vs_mean_ratio_from_mean(
                df,
                spec,
                rolling_results[mean_spec.output_col].features[mean_spec.output_col],
            )
        elif spec.operation in {"recency_seconds_since_last", "is_first_by_entity"}:
            result = recency_results[spec.output_col]
        else:
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

        # 현재 ML-01 Stage 0 operation은 category artifact를 만들지 않지만, 반환 계약은 유지한다.
        # 이후 category operation이 추가되면 같은 artifacts dict에 붙여 저장할 수 있다.
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
