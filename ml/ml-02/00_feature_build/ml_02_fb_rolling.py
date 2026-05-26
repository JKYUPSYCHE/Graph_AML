"""
ML-02 feature build rolling aggregation 모듈.

정책
----
- `closed="left"`만 허용한다.
- 현재 row와 같은 timestamp row는 history에 넣지 않는다.
- history 조건은 `current_timestamp - window <= past_timestamp < current_timestamp`이다.
- pandas `groupby().rolling()`은 사용하지 않는다. 환경 차이보다 명시적인 deque 계산을 우선한다.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Tuple

import numpy as np
import pandas as pd

from ml_02_fb_schema import normalize_category_strict, parse_datetime_strict, parse_numeric_strict
from ml_02_fb_specs import FeatureOpResult, FeatureSpec, validate_feature_specs
from ml_02_fb_operation_result_validation import finalize_result, param_value, require_allowed_params, require_columns, require_roles


SUPPORTED_ROLLING_AGGS: Tuple[str, ...] = ("sum", "mean", "std", "min", "max", "count")
RollingAggGroupKey = Tuple[str, str, str, str]


def parse_window(window: Any, operation: str, output_col: str) -> pd.Timedelta:
    """'1h', '7d' 같은 window 값을 pandas Timedelta로 변환하고 양수인지 확인한다."""

    if str(window).strip() == "":
        raise ValueError(
            "Feature operation failed: window parameter must not be empty. "
            f"operation={operation!r}, output_col={output_col!r}, window={window!r}"
        )
    parsed = pd.Timedelta(window)
    if parsed <= pd.Timedelta(0):
        raise ValueError(
            "Feature operation failed: window must be positive. "
            f"operation={operation!r}, output_col={output_col!r}, window={window!r}"
        )
    return parsed


def _rolling_agg_spec_parts(spec: FeatureSpec) -> tuple[dict[str, str], pd.Timedelta, str, str, Any, str]:
    """rolling_agg FeatureSpec에서 실행에 필요한 role/param을 추출하고 검증한다."""

    require_allowed_params(spec, ("window", "agg", "closed", "fill_value", "dtype"))
    roles = require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    window = parse_window(param_value(spec, "window", ""), spec.operation, spec.output_col)
    agg = str(param_value(spec, "agg", "")).strip().lower()
    closed = str(param_value(spec, "closed", "left")).strip().lower()
    entity_col = roles["entity_col"]
    timestamp_col = roles["timestamp_col"]
    value_col = roles["value_col"]

    if closed != "left":
        raise ValueError(
            "Feature operation failed: rolling_agg only supports closed='left' to avoid current/future leakage. "
            f"output_col={spec.output_col!r}, window={window}, agg={agg!r}, entity_col={entity_col!r}, "
            f"timestamp_col={timestamp_col!r}, value_col={value_col!r}, observed_closed={closed!r}"
        )
    if agg not in SUPPORTED_ROLLING_AGGS:
        raise ValueError(
            "Feature operation failed: unsupported rolling agg. "
            f"output_col={spec.output_col!r}, window={window}, agg={agg!r}, entity_col={entity_col!r}, "
            f"timestamp_col={timestamp_col!r}, value_col={value_col!r}, supported={list(SUPPORTED_ROLLING_AGGS)}"
        )

    fill_value = param_value(spec, "fill_value", 0.0)
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, window, agg, closed, fill_value, dtype


def _rolling_agg_group_key(spec: FeatureSpec) -> RollingAggGroupKey:
    """한 번의 정렬/스캔을 공유할 수 있는 rolling spec 묶음 key를 만든다."""

    roles, _window, _agg, closed, _fill_value, _dtype = _rolling_agg_spec_parts(spec)
    return (roles["entity_col"], roles["timestamp_col"], roles["value_col"], closed)


def _aggregate(event_count: int, running_sum: float, running_sum_sq: float, min_queue: deque, max_queue: deque, agg: str) -> float:
    """현재 deque 상태에서 요청한 집계값을 계산한다."""

    if event_count == 0:
        return np.nan
    if agg == "sum":
        return running_sum
    if agg == "mean":
        return running_sum / event_count
    if agg == "std":
        if event_count <= 1:
            return np.nan
        variance_numerator = running_sum_sq - (running_sum * running_sum / event_count)
        return float(np.sqrt(max(variance_numerator, 0.0) / (event_count - 1)))
    if agg == "min":
        return float(min_queue[0][0])
    if agg == "max":
        return float(max_queue[0][0])
    return float(event_count)


def _compute_past_window_multi_agg_values(
    work: pd.DataFrame,
    *,
    window: pd.Timedelta,
    aggs: Tuple[str, ...],
) -> dict[str, np.ndarray]:
    """정렬된 work frame을 window별 1회 스캔하며 여러 rolling agg 값을 함께 계산한다.

    ``work``는 entity/timestamp/원래 row 순서로 이미 정렬되어 있어야 한다.
    같은 entity/window 안에서는 deque 하나를 시간순으로 밀어가며 sum/count/std/min/max를 갱신한다.
    """

    unique_aggs = tuple(dict.fromkeys(aggs))
    if not unique_aggs:
        raise ValueError("Feature operation failed: rolling aggregation list must not be empty.")
    unsupported_aggs = sorted(set(unique_aggs) - set(SUPPORTED_ROLLING_AGGS))
    if unsupported_aggs:
        raise ValueError(
            "Feature operation failed: unsupported rolling aggs. "
            f"aggs={unsupported_aggs}, supported={list(SUPPORTED_ROLLING_AGGS)}"
        )

    row_count = len(work)
    result_values = {agg: np.full(row_count, np.nan, dtype="float64") for agg in unique_aggs}
    if row_count == 0:
        return result_values

    entities = work["_entity"].to_numpy()
    timestamp_ns = work["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    values = work["_value"].to_numpy(dtype="float64", copy=False)
    work_positions = work["_work_pos"].to_numpy(dtype="int64", copy=False)
    window_ns = int(window.value)

    entity_breaks = np.flatnonzero(entities[1:] != entities[:-1]) + 1
    entity_starts = np.concatenate(([0], entity_breaks))
    entity_ends = np.concatenate((entity_breaks, [row_count]))

    for entity_start, entity_end in zip(entity_starts, entity_ends):
        # 단일 거래 entity는 과거 이력이 없으므로 결과를 NaN으로 두고 후단 fill_value 처리에 맡긴다.
        if entity_end - entity_start <= 1:
            continue

        # event_queue는 현재 window 안에 남아 있는 과거 이벤트 전체를 보관한다.
        # min/max queue는 monotonic queue로 유지해 window 최소/최대를 O(1)에 가깝게 꺼낸다.
        event_queue: deque[tuple[int, float, int]] = deque()
        min_queue: deque[tuple[float, int]] = deque()
        max_queue: deque[tuple[float, int]] = deque()
        running_sum = 0.0
        running_sum_sq = 0.0
        event_count = 0
        sequence_id = 0

        local_timestamps = timestamp_ns[entity_start:entity_end]
        timestamp_breaks = np.flatnonzero(local_timestamps[1:] != local_timestamps[:-1]) + 1
        timestamp_starts = np.concatenate(([entity_start], entity_start + timestamp_breaks))
        timestamp_ends = np.concatenate((entity_start + timestamp_breaks, [entity_end]))

        for timestamp_start, timestamp_end in zip(timestamp_starts, timestamp_ends):
            lower_bound_ns = int(timestamp_ns[timestamp_start]) - window_ns
            # window 정의는 current_timestamp - window <= past_timestamp < current_timestamp다.
            # lower_bound보다 오래된 이벤트만 제거하고, 같은 timestamp 이벤트는 아직 queue에 넣지 않는다.
            while event_queue and event_queue[0][0] < lower_bound_ns:
                _old_timestamp_ns, old_value, old_sequence_id = event_queue.popleft()
                running_sum -= old_value
                running_sum_sq -= old_value * old_value
                event_count -= 1
                if min_queue and min_queue[0][1] == old_sequence_id:
                    min_queue.popleft()
                if max_queue and max_queue[0][1] == old_sequence_id:
                    max_queue.popleft()

            # 같은 timestamp group은 먼저 출력하고, 그 뒤 history에 추가한다.
            # 따라서 현재 row와 동시점 row가 history에 들어가는 누수를 막는다.
            positions = work_positions[timestamp_start:timestamp_end]
            for agg in unique_aggs:
                aggregate_value = _aggregate(event_count, running_sum, running_sum_sq, min_queue, max_queue, agg)
                result_values[agg][positions] = aggregate_value

            for row_idx in range(timestamp_start, timestamp_end):
                value_float = float(values[row_idx])
                event_queue.append((int(timestamp_ns[row_idx]), value_float, sequence_id))
                running_sum += value_float
                running_sum_sq += value_float * value_float
                event_count += 1
                while min_queue and min_queue[-1][0] > value_float:
                    min_queue.pop()
                min_queue.append((value_float, sequence_id))
                while max_queue and max_queue[-1][0] < value_float:
                    max_queue.pop()
                max_queue.append((value_float, sequence_id))
                sequence_id += 1

    return result_values


def _execute_rolling_agg_group(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    """같은 entity/timestamp/value/closed 조합의 rolling_agg spec을 묶어 실행한다.

    window와 agg가 다른 spec도 같은 입력 컬럼을 쓰면 정렬 결과를 공유할 수 있다.
    이 함수는 window별로 한 번씩만 스캔하고, 각 spec의 output_col에 맞게 결과를 재조립한다.
    """

    if not specs:
        return {}

    group_key = _rolling_agg_group_key(specs[0])
    first_roles, _first_window, _first_agg, _closed, _first_fill_value, _first_dtype = _rolling_agg_spec_parts(specs[0])
    spec_parts: list[tuple[FeatureSpec, dict[str, str], pd.Timedelta, str, str, Any, str]] = []
    for spec in specs:
        if spec.operation != "rolling_agg":
            raise ValueError(
                "Feature build failed: rolling batch received a non-rolling spec. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        if _rolling_agg_group_key(spec) != group_key:
            raise ValueError(
                "Feature build failed: rolling batch group contains incompatible specs. "
                f"first_output_col={specs[0].output_col!r}, observed_output_col={spec.output_col!r}"
            )
        spec_parts.append((spec, *_rolling_agg_spec_parts(spec)))

    entity_col = first_roles["entity_col"]
    timestamp_col = first_roles["timestamp_col"]
    value_col = first_roles["value_col"]
    require_columns(df, (entity_col, timestamp_col, value_col), "rolling_agg")

    entity = normalize_category_strict(df[entity_col], source_col=entity_col)
    timestamps = parse_datetime_strict(df, timestamp_col, specs[0].output_col)
    values = parse_numeric_strict(df, value_col, specs[0].output_col)
    # work는 계산 편의를 위한 내부 frame이다. _row_order를 보존해 마지막에 원래 입력 순서로 되돌린다.
    work = pd.DataFrame(
        {
            "_entity": entity,
            "_timestamp": timestamps,
            "_value": values.astype("float64"),
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")
    work = work.reset_index(drop=True)
    work["_work_pos"] = np.arange(len(work))

    # 같은 window 안에서는 여러 agg(count/sum/mean/std/max 등)를 한 번의 deque scan으로 함께 계산한다.
    aggs_by_window_ns: dict[int, list[str]] = {}
    windows_by_ns: dict[int, pd.Timedelta] = {}
    for _spec, _roles, window, agg, _closed, _fill_value, _dtype in spec_parts:
        window_ns = int(window.value)
        windows_by_ns.setdefault(window_ns, window)
        aggs_by_window_ns.setdefault(window_ns, [])
        if agg not in aggs_by_window_ns[window_ns]:
            aggs_by_window_ns[window_ns].append(agg)

    # raw_results key는 (window_ns, agg)다. sorted order 결과를 원래 입력 row order로 되돌려 저장한다.
    raw_results: dict[tuple[int, str], np.ndarray] = {}
    row_order = work["_row_order"].to_numpy()
    for window_ns, aggs in aggs_by_window_ns.items():
        sorted_results = _compute_past_window_multi_agg_values(
            work,
            window=windows_by_ns[window_ns],
            aggs=tuple(aggs),
        )
        for agg, sorted_values in sorted_results.items():
            key = (window_ns, agg)
            original_order_values = np.full(len(df), np.nan, dtype="float64")
            original_order_values[row_order] = sorted_values
            raw_results[key] = original_order_values

    results: dict[str, FeatureOpResult] = {}
    for spec, roles, window, agg, closed, fill_value, dtype in spec_parts:
        values_out = pd.Series(raw_results[(int(window.value), agg)].copy(), index=df.index).fillna(fill_value)
        params = {"window": str(window), "agg": agg, "closed": closed, "fill_value": fill_value}
        results[spec.output_col] = finalize_result(
            values_out,
            spec,
            row_count=len(df),
            input_columns=roles,
            params=params,
            dtype=dtype,
        )
    return results


def op_rolling_agg(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """FeatureSpec 1개에 대한 entity별 past-only rolling aggregation을 계산한다."""

    return execute_rolling_agg_specs_batched(df, (spec,))[spec.output_col]


def execute_rolling_agg_specs_batched(
    df: pd.DataFrame,
    specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """rolling_agg spec 목록을 group key별로 묶어 실행하고 output_col별 결과를 반환한다.

    group key가 같으면 정렬과 entity별 순회 비용을 공유한다. rolling feature가 많은 경우
    spec별 독립 실행보다 같은 semantics를 유지하면서 중복 계산을 줄일 수 있다.
    """

    if not specs:
        return {}
    validate_feature_specs(specs)

    grouped_specs: dict[RollingAggGroupKey, list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation != "rolling_agg":
            raise ValueError(
                "Feature build failed: execute_rolling_agg_specs_batched only accepts rolling_agg specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_rolling_agg_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        group_results = _execute_rolling_agg_group(df, tuple(group_specs))
        overlap = set(results) & set(group_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate rolling batch results. output_cols={sorted(overlap)}")
        results.update(group_results)
    return results
