"""Past-only rolling aggregation for ML-03 degree/count features.

Code map:
- Input: rolling_agg FeatureSpec rows and split_df columns.
- Output: FeatureOpResult objects for count/sum/mean/std/min/max windows.
- Public: execute_rolling_agg_specs_batched, parse_window, op_rolling_agg.
- Leakage guard: current timestamp group is emitted before it enters history.
- Notes: only closed='left' is accepted for ML-03 rolling features.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Tuple

import numpy as np
import pandas as pd

from ml_03_fb_operation_result_validation import finalize_result, param_value, require_allowed_params, require_columns, require_roles
from ml_03_fb_schema import normalize_category_strict, parse_datetime_strict, parse_numeric_strict
from ml_03_fb_specs import FeatureOpResult, FeatureSpec, validate_feature_specs


SUPPORTED_ROLLING_AGGS: Tuple[str, ...] = ("sum", "mean", "std", "min", "max", "count")
RollingAggGroupKey = Tuple[str, str, str, str]


def parse_window(window: Any, operation: str, output_col: str) -> pd.Timedelta:
    """Parse a positive pandas Timedelta window."""

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
    require_allowed_params(spec, ("window", "agg", "closed", "fill_value", "dtype"))
    roles = require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    window = parse_window(param_value(spec, "window", ""), spec.operation, spec.output_col)
    agg = str(param_value(spec, "agg", "")).strip().lower()
    closed = str(param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: rolling_agg only supports closed='left' to avoid current/future leakage. "
            f"output_col={spec.output_col!r}, observed_closed={closed!r}"
        )
    if agg not in SUPPORTED_ROLLING_AGGS:
        raise ValueError(
            "Feature operation failed: unsupported rolling agg. "
            f"output_col={spec.output_col!r}, agg={agg!r}, supported={list(SUPPORTED_ROLLING_AGGS)}"
        )
    fill_value = param_value(spec, "fill_value", 0.0)
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, window, agg, closed, fill_value, dtype


def _rolling_agg_group_key(spec: FeatureSpec) -> RollingAggGroupKey:
    roles, _window, _agg, closed, _fill_value, _dtype = _rolling_agg_spec_parts(spec)
    return (roles["entity_col"], roles["timestamp_col"], roles["value_col"], closed)


def _aggregate(event_count: int, running_sum: float, running_sum_sq: float, min_queue: deque, max_queue: deque, agg: str) -> float:
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
    """Compute rolling values with history condition lower_bound <= past_ts < current_ts."""

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
        if entity_end - entity_start <= 1:
            continue

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
            # Remove events older than the inclusive lower window bound.
            while event_queue and event_queue[0][0] < lower_bound_ns:
                _old_timestamp_ns, old_value, old_sequence_id = event_queue.popleft()
                running_sum -= old_value
                running_sum_sq -= old_value * old_value
                event_count -= 1
                if min_queue and min_queue[0][1] == old_sequence_id:
                    min_queue.popleft()
                if max_queue and max_queue[0][1] == old_sequence_id:
                    max_queue.popleft()

            positions = work_positions[timestamp_start:timestamp_end]
            for agg in unique_aggs:
                result_values[agg][positions] = _aggregate(event_count, running_sum, running_sum_sq, min_queue, max_queue, agg)

            # Leakage guard: same-timestamp rows enter history only after current outputs are written.
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

    work = pd.DataFrame(
        {
            "_entity": normalize_category_strict(df[entity_col], source_col=entity_col),
            "_timestamp": parse_datetime_strict(df, timestamp_col, specs[0].output_col),
            "_value": parse_numeric_strict(df, value_col, specs[0].output_col).astype("float64"),
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")
    work = work.reset_index(drop=True)
    work["_work_pos"] = np.arange(len(work))

    aggs_by_window_ns: dict[int, list[str]] = {}
    windows_by_ns: dict[int, pd.Timedelta] = {}
    for _spec, _roles, window, agg, _closed, _fill_value, _dtype in spec_parts:
        window_ns = int(window.value)
        windows_by_ns.setdefault(window_ns, window)
        aggs_by_window_ns.setdefault(window_ns, [])
        if agg not in aggs_by_window_ns[window_ns]:
            aggs_by_window_ns[window_ns].append(agg)

    raw_results: dict[tuple[int, str], np.ndarray] = {}
    row_order = work["_row_order"].to_numpy()
    for window_ns, aggs in aggs_by_window_ns.items():
        sorted_results = _compute_past_window_multi_agg_values(
            work,
            window=windows_by_ns[window_ns],
            aggs=tuple(aggs),
        )
        for agg, sorted_values in sorted_results.items():
            original_order_values = np.full(len(df), np.nan, dtype="float64")
            original_order_values[row_order] = sorted_values
            raw_results[(window_ns, agg)] = original_order_values

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
    """Execute one rolling aggregation spec."""

    return execute_rolling_agg_specs_batched(df, (spec,))[spec.output_col]


def execute_rolling_agg_specs_batched(
    df: pd.DataFrame,
    specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """Execute rolling_agg specs by shared input group."""

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
