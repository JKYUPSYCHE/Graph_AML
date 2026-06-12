"""
Feature operation 실행 모듈

이 파일의 역할
----------------
1. FeatureSpec에 선언된 operation 이름을 실제 계산 함수로 연결한다.
2. ML-02 Stage 1에 필요한 계좌별 통계 feature를 계산한다.
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

from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd

from ml_02_fb_schema import normalize_category_strict, parse_datetime_strict, parse_numeric_strict
from ml_02_fb_specs import FeatureOpResult, FeatureSpec, META_COLUMNS, feature_columns, validate_feature_specs
from ml_02_fb_rolling import execute_rolling_agg_specs_batched, op_rolling_agg, parse_window
from ml_02_fb_operation_result_validation import finalize_result, param_value, require_allowed_params, require_columns, require_roles


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
RollingEntityAggGroupKey = tuple[str, str, str, str, str]


# -----------------------------------------------------------------------------
# 2. 공통 검증/보조 함수
# -----------------------------------------------------------------------------
def _empty_category_mapping() -> pd.DataFrame:
    """category feature가 없을 때도 header가 있는 빈 mapping DataFrame을 반환한다."""

    return pd.DataFrame(columns=list(CATEGORY_MAPPING_COLUMNS))


def _empty_category_unknown_summary() -> pd.DataFrame:
    """category feature가 없을 때도 header가 있는 빈 unknown summary DataFrame을 반환한다."""

    return pd.DataFrame(columns=list(CATEGORY_UNKNOWN_COLUMNS))


# -----------------------------------------------------------------------------
# 3. recency / first transaction operation
# -----------------------------------------------------------------------------
def _entity_recency_parts(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    """
    entity별 직전 과거 timestamp와 첫 거래 flag를 함께 계산한다.

    같은 timestamp에 있는 거래끼리는 서로 과거로 보지 않는다.
    """

    roles = require_roles(spec, ("entity_col", "timestamp_col"))
    entity_col = roles["entity_col"]
    timestamp_col = roles["timestamp_col"]
    require_columns(df, (entity_col, timestamp_col), spec.operation)

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

    require_allowed_params(spec, ("dtype", "fill_value"))
    recency, _is_first, roles = _entity_recency_parts(df, spec)
    fill_value = param_value(spec, "fill_value", -1.0)
    recency = recency.fillna(fill_value)
    dtype = str(param_value(spec, "dtype", "float64"))
    params = {"fill_value": fill_value}
    return finalize_result(
        recency,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=params,
        dtype=dtype,
    )


def op_is_first_by_entity(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """entity 기준 과거 거래가 없으면 1, 있으면 0인 flag를 만든다."""

    require_allowed_params(spec, ("dtype",))
    _recency, is_first, roles = _entity_recency_parts(df, spec)
    dtype = str(param_value(spec, "dtype", "int8"))
    return finalize_result(
        is_first,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype=dtype,
    )


def _recency_group_key(spec: FeatureSpec) -> tuple[str, str]:
    roles = require_roles(spec, ("entity_col", "timestamp_col"))
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
                require_allowed_params(spec, ("dtype", "fill_value"))
                fill_value = param_value(spec, "fill_value", -1.0)
                dtype = str(param_value(spec, "dtype", "float64"))
                result = finalize_result(
                    recency.fillna(fill_value),
                    spec,
                    row_count=len(df),
                    input_columns=roles,
                    params={"fill_value": fill_value},
                    dtype=dtype,
                )
            else:
                require_allowed_params(spec, ("dtype",))
                dtype = str(param_value(spec, "dtype", "int8"))
                result = finalize_result(
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
# 4. time-history operation
# -----------------------------------------------------------------------------
def op_cur_vs_mean_ratio(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """현재 금액을 entity별 과거 window 평균 금액으로 나눈 비율을 계산한다."""

    require_allowed_params(spec, ("window", "closed", "fill_value", "zero_division_value", "dtype"))
    roles = require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    entity_col = roles["entity_col"]
    timestamp_col = roles["timestamp_col"]
    value_col = roles["value_col"]
    require_columns(df, (entity_col, timestamp_col, value_col), spec.operation)

    window = param_value(spec, "window", "")
    closed = str(param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: cur_vs_mean_ratio only supports closed='left' to avoid current/future leakage. "
            f"observed_closed={closed!r}"
        )
    zero_division_value = float(param_value(spec, "zero_division_value", 0.0))
    fill_value = float(param_value(spec, "fill_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))

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
    return finalize_result(
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

    require_allowed_params(spec, ("window", "closed", "fill_value", "zero_division_value", "dtype"))
    roles = require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    window = param_value(spec, "window", "")
    closed = str(param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: cur_vs_mean_ratio only supports closed='left' to avoid current/future leakage. "
            f"observed_closed={closed!r}"
        )
    fill_value = float(param_value(spec, "fill_value", 0.0))
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

    require_allowed_params(spec, ("window", "closed", "fill_value", "zero_division_value", "dtype"))
    roles = require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    value_col = roles["value_col"]
    require_columns(df, (roles["entity_col"], roles["timestamp_col"], value_col), spec.operation)

    window = param_value(spec, "window", "")
    closed = str(param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: cur_vs_mean_ratio only supports closed='left' to avoid current/future leakage. "
            f"observed_closed={closed!r}"
        )
    zero_division_value = float(param_value(spec, "zero_division_value", 0.0))
    fill_value = float(param_value(spec, "fill_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))

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
    return finalize_result(
        ratio,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=params,
        dtype=dtype,
    )


def op_cumulative_count(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """entity별 현재 timestamp 이전 누적 거래 건수를 계산한다."""

    require_allowed_params(spec, ("dtype",))
    roles = require_roles(spec, ("entity_col", "timestamp_col"))
    entity_col = roles["entity_col"]
    timestamp_col = roles["timestamp_col"]
    require_columns(df, (entity_col, timestamp_col), spec.operation)

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

    dtype = str(param_value(spec, "dtype", "int32"))
    return finalize_result(
        output,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype=dtype,
    )


# -----------------------------------------------------------------------------
# 6. ML-02 Stage 1 account statistics operation
# -----------------------------------------------------------------------------
def _rolling_entity_agg_spec_parts(spec: FeatureSpec) -> tuple[dict[str, str], pd.Timedelta, str, str, float, str]:
    """rolling_entity_agg FeatureSpec에서 실행에 필요한 role/param을 추출하고 검증한다."""

    require_allowed_params(spec, ("window", "agg", "closed", "fill_value", "dtype"))
    roles = require_roles(spec, ("current_entity_col", "history_entity_col", "timestamp_col", "value_col"))
    window = parse_window(param_value(spec, "window", ""), spec.operation, spec.output_col)
    agg = str(param_value(spec, "agg", "sum")).strip().lower()
    closed = str(param_value(spec, "closed", "left")).strip().lower()
    if agg != "sum":
        raise ValueError(
            "Feature operation failed: rolling_entity_agg only supports agg='sum'. "
            f"output_col={spec.output_col!r}, observed_agg={agg!r}"
        )
    if closed != "left":
        raise ValueError(
            "Feature operation failed: rolling_entity_agg only supports closed='left' to avoid current/future leakage. "
            f"output_col={spec.output_col!r}, observed_closed={closed!r}"
        )

    fill_value = float(param_value(spec, "fill_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, window, agg, closed, fill_value, dtype


def _rolling_entity_agg_group_key(spec: FeatureSpec) -> RollingEntityAggGroupKey:
    """한 번의 정렬/groupby를 공유할 수 있는 rolling_entity_agg 묶음 key를 만든다."""

    roles, _window, _agg, closed, _fill_value, _dtype = _rolling_entity_agg_spec_parts(spec)
    return (
        roles["current_entity_col"],
        roles["history_entity_col"],
        roles["timestamp_col"],
        roles["value_col"],
        closed,
    )


def _execute_rolling_entity_agg_group(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    """같은 current/history/value 조합의 rolling_entity_agg spec을 묶어 실행한다.

    pandas groupby 객체를 entity마다 만들지 않고, 정렬된 numpy 배열의 entity 경계만
    순회한다. window별 rolling sum은 같은 entity scan 안에서 함께 계산한다.
    """

    if not specs:
        return {}

    group_key = _rolling_entity_agg_group_key(specs[0])
    first_roles, _first_window, _first_agg, _first_closed, _first_fill_value, _first_dtype = _rolling_entity_agg_spec_parts(specs[0])
    spec_parts: list[tuple[FeatureSpec, dict[str, str], pd.Timedelta, str, str, float, str]] = []
    for spec in specs:
        if spec.operation != "rolling_entity_agg":
            raise ValueError(
                "Feature build failed: rolling entity batch received a non-rolling_entity_agg spec. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        if _rolling_entity_agg_group_key(spec) != group_key:
            raise ValueError(
                "Feature build failed: rolling entity batch group contains incompatible specs. "
                f"first_output_col={specs[0].output_col!r}, observed_output_col={spec.output_col!r}"
            )
        spec_parts.append((spec, *_rolling_entity_agg_spec_parts(spec)))

    current_entity_col = first_roles["current_entity_col"]
    history_entity_col = first_roles["history_entity_col"]
    timestamp_col = first_roles["timestamp_col"]
    value_col = first_roles["value_col"]
    require_columns(df, (current_entity_col, history_entity_col, timestamp_col, value_col), "rolling_entity_agg")
    timestamps = parse_datetime_strict(df, timestamp_col, specs[0].output_col)
    values = parse_numeric_strict(df, value_col, specs[0].output_col).astype("float64")
    events = pd.DataFrame(
        {
            "_entity": normalize_category_strict(df[history_entity_col], source_col=history_entity_col),
            "_timestamp": timestamps,
            "_value": values,
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")
    queries = pd.DataFrame(
        {
            "_entity": normalize_category_strict(df[current_entity_col], source_col=current_entity_col),
            "_timestamp": timestamps,
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")

    event_count = len(events)
    query_count = len(queries)
    if event_count != len(df) or query_count != len(df):
        raise ValueError(
            "Feature build failed: rolling_entity_agg internal frame row count mismatch. "
            f"input_rows={len(df)}, event_rows={event_count}, query_rows={query_count}"
        )

    event_entities = events["_entity"].to_numpy()
    event_timestamps = events["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    event_values = events["_value"].to_numpy(dtype="float64", copy=False)

    query_entities = queries["_entity"].to_numpy()
    query_timestamps = queries["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    query_rows = queries["_row_order"].to_numpy(dtype="int64", copy=False)

    event_breaks = np.flatnonzero(event_entities[1:] != event_entities[:-1]) + 1
    event_starts = np.concatenate(([0], event_breaks))
    event_ends = np.concatenate((event_breaks, [event_count]))

    query_breaks = np.flatnonzero(query_entities[1:] != query_entities[:-1]) + 1
    query_starts = np.concatenate(([0], query_breaks))
    query_ends = np.concatenate((query_breaks, [query_count]))

    window_ns_values = tuple(
        dict.fromkeys(
            int(window.value)
            for _spec, _roles, window, _agg, _closed, _fill_value, _dtype in spec_parts
        )
    )
    raw_outputs = {window_ns: np.full(len(df), np.nan, dtype="float64") for window_ns in window_ns_values}

    event_group_idx = 0
    query_group_idx = 0
    while event_group_idx < len(event_starts) and query_group_idx < len(query_starts):
        event_entity = event_entities[event_starts[event_group_idx]]
        query_entity = query_entities[query_starts[query_group_idx]]

        if event_entity < query_entity:
            event_group_idx += 1
            continue
        if event_entity > query_entity:
            query_group_idx += 1
            continue

        event_start = int(event_starts[event_group_idx])
        event_end_bound = int(event_ends[event_group_idx])
        query_start = int(query_starts[query_group_idx])
        query_end = int(query_ends[query_group_idx])

        local_event_timestamps = event_timestamps[event_start:event_end_bound]
        local_event_values = event_values[event_start:event_end_bound]
        local_query_timestamps = query_timestamps[query_start:query_end]
        local_query_rows = query_rows[query_start:query_end]

        running_sums = {window_ns: 0.0 for window_ns in window_ns_values}
        window_starts = {window_ns: 0 for window_ns in window_ns_values}
        event_end = 0

        for query_timestamp_ns, row_order in zip(local_query_timestamps, local_query_rows):
            current_ts = int(query_timestamp_ns)

            # history 조건은 history_timestamp < current_timestamp다. 같은 timestamp는 아직 더하지 않는다.
            while event_end < len(local_event_timestamps) and int(local_event_timestamps[event_end]) < current_ts:
                value = float(local_event_values[event_end])
                for window_ns in window_ns_values:
                    running_sums[window_ns] += value
                event_end += 1

            for window_ns in window_ns_values:
                lower_bound_ns = current_ts - window_ns
                window_start = window_starts[window_ns]
                while window_start < event_end and int(local_event_timestamps[window_start]) < lower_bound_ns:
                    running_sums[window_ns] -= float(local_event_values[window_start])
                    window_start += 1
                window_starts[window_ns] = window_start
                raw_outputs[window_ns][row_order] = running_sums[window_ns]

        event_group_idx += 1
        query_group_idx += 1

    results: dict[str, FeatureOpResult] = {}
    for spec, roles, window, agg, closed, fill_value, dtype in spec_parts:
        output = pd.Series(raw_outputs[int(window.value)].copy(), index=df.index).fillna(fill_value)

        params = {"window": str(window), "agg": agg, "closed": closed, "fill_value": fill_value}
        results[spec.output_col] = finalize_result(
            output,
            spec,
            row_count=len(df),
            input_columns=roles,
            params=params,
            dtype=dtype,
        )
    return results


def execute_rolling_entity_agg_specs_batched(
    df: pd.DataFrame,
    specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """rolling_entity_agg spec 목록을 group key별로 묶어 실행하고 output_col별 결과를 반환한다."""

    if not specs:
        return {}
    validate_feature_specs(specs)

    grouped_specs: dict[RollingEntityAggGroupKey, list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation != "rolling_entity_agg":
            raise ValueError(
                "Feature build failed: execute_rolling_entity_agg_specs_batched only accepts rolling_entity_agg specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_rolling_entity_agg_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        group_results = _execute_rolling_entity_agg_group(df, tuple(group_specs))
        overlap = set(results) & set(group_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate rolling entity batch results. output_cols={sorted(overlap)}")
        results.update(group_results)
    return results


def op_rolling_entity_agg(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """현재 row 계좌와 과거 row의 방향별 계좌를 매칭해 past-only rolling sum을 계산한다."""

    return execute_rolling_entity_agg_specs_batched(df, (spec,))[spec.output_col]


# -----------------------------------------------------------------------------
# 7. operation registry와 실행기
# -----------------------------------------------------------------------------
# FeatureSpec.operation 문자열을 실제 함수로 연결하는 테이블이다.
# 새 operation을 추가할 때는 함수 작성 후 이 registry에 등록해야 한다.
# recency/ratio/cumulative operations are retained for external FeatureSpec
# compatibility and future ablation runs; ML-02 Stage 1 accountstats specs do
# not call them by default.
OPERATION_REGISTRY: dict[str, OperationRunner] = {
    "rolling_agg": op_rolling_agg,
    "rolling_entity_agg": op_rolling_entity_agg,
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


def _precompute_feature_results(
    df: pd.DataFrame,
    feature_specs: Tuple[FeatureSpec, ...],
) -> tuple[
    dict[str, FeatureOpResult],
    dict[str, FeatureOpResult],
    dict[str, FeatureOpResult],
    dict[str, FeatureSpec],
]:
    """batch 실행이 가능한 operation 결과를 미리 계산한다."""

    rolling_specs = tuple(spec for spec in feature_specs if spec.operation == "rolling_agg")
    rolling_entity_specs = tuple(spec for spec in feature_specs if spec.operation == "rolling_entity_agg")
    ratio_specs = tuple(spec for spec in feature_specs if spec.operation == "cur_vs_mean_ratio")
    ratio_mean_specs = {spec.output_col: _rolling_mean_spec_for_ratio(spec) for spec in ratio_specs}
    rolling_results = execute_rolling_agg_specs_batched(df, (*rolling_specs, *ratio_mean_specs.values()))
    rolling_entity_results = execute_rolling_entity_agg_specs_batched(df, rolling_entity_specs)

    recency_specs = tuple(
        spec for spec in feature_specs if spec.operation in {"recency_seconds_since_last", "is_first_by_entity"}
    )
    recency_results = execute_recency_specs_batched(df, recency_specs)
    return rolling_results, rolling_entity_results, recency_results, ratio_mean_specs


def _result_for_spec(
    df: pd.DataFrame,
    spec: FeatureSpec,
    *,
    rolling_results: dict[str, FeatureOpResult],
    rolling_entity_results: dict[str, FeatureOpResult],
    recency_results: dict[str, FeatureOpResult],
    ratio_mean_specs: dict[str, FeatureSpec],
) -> FeatureOpResult:
    """FeatureSpec 1개의 결과를 batch cache 또는 registry에서 가져온다."""

    if spec.operation == "rolling_agg":
        return rolling_results[spec.output_col]
    if spec.operation == "rolling_entity_agg":
        return rolling_entity_results[spec.output_col]
    if spec.operation == "cur_vs_mean_ratio":
        mean_spec = ratio_mean_specs[spec.output_col]
        return _execute_cur_vs_mean_ratio_from_mean(
            df,
            spec,
            rolling_results[mean_spec.output_col].features[mean_spec.output_col],
        )
    if spec.operation in {"recency_seconds_since_last", "is_first_by_entity"}:
        return recency_results[spec.output_col]
    return run_operation(df, spec)


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
        operation별 부가 산출물이다. 현재 Stage 1 accountstats operation은 category artifact를 만들지 않는다.
    """

    validate_feature_specs(feature_specs)
    missing_meta = set(META_COLUMNS) - set(df.columns)
    if missing_meta:
        raise ValueError(f"Feature execution input is missing metadata columns: {sorted(missing_meta)}")

    feature_parts: list[pd.DataFrame] = []
    feature_info_parts: list[pd.DataFrame] = []
    category_mapping_parts: list[pd.DataFrame] = []
    category_unknown_parts: list[pd.DataFrame] = []
    rolling_results, rolling_entity_results, recency_results, ratio_mean_specs = _precompute_feature_results(
        df,
        feature_specs,
    )
    for spec in feature_specs:
        result = _result_for_spec(
            df,
            spec,
            rolling_results=rolling_results,
            rolling_entity_results=rolling_entity_results,
            recency_results=recency_results,
            ratio_mean_specs=ratio_mean_specs,
        )
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

        # 현재 Stage 1 accountstats operation은 category artifact를 만들지 않지만, 반환 계약은 유지한다.
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
