"""ML-03 Stage 2 feature operation execution.

All history features enforce ``history_timestamp < current_timestamp`` and
exclude the current row plus every same-timestamp row.

Code map:
- Input: validated split_df and ML-03 FeatureSpec tuple.
- Output: generated feature frame, feature_info, and operation artifacts.
- Public: execute_feature_specs plus batched rolling counterparty executors.
- Leakage guard: counterparty scans add only event_timestamp < query_timestamp.
- Notes: execute_feature_specs is the main path; single-spec registry wrappers were removed.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd

from ml_03_fb_operation_result_validation import finalize_result, param_value, require_allowed_params, require_columns, require_roles
from ml_03_fb_rolling import execute_rolling_agg_specs_batched, parse_window
from ml_03_fb_schema import META_COLUMNS, normalize_category_strict, parse_datetime_strict, parse_numeric_strict
from ml_03_fb_specs import FeatureOpResult, FeatureSpec, feature_columns, validate_feature_specs


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

CounterpartyNuniqueGroupKey = tuple[str, str, str, str, str]
CounterpartyAmountGroupKey = tuple[str, str, str, str, str, str]
SUPPORTED_BATCH_OPERATIONS = {
    "rolling_agg",
    "rolling_counterparty_nunique",
    "rolling_counterparty_effective_n",
    "rolling_counterparty_top1_share",
}
_EPSILON = 1e-12
_FLOAT_REL_TOL = 1e-12
_FLOAT_ABS_TOL = 1e-9
_EXPIRE_REL_TOL = 1e-9
_EXPIRE_ABS_TOL = 1e-4


def _scaled_float_tolerance(*values: float) -> float:
    """Return a small tolerance scaled to the amount magnitude."""

    scale = max([1.0, *(abs(float(value)) for value in values)])
    return max(_FLOAT_ABS_TOL, scale * _FLOAT_REL_TOL)


def _expire_float_tolerance(*values: float) -> float:
    """Return tolerance for subtracting the final active amount event."""

    scale = max([1.0, *(abs(float(value)) for value in values)])
    return max(_EXPIRE_ABS_TOL, scale * _EXPIRE_REL_TOL)


@dataclass
class CounterpartyNuniqueState:
    """Rolling distinct counterparty state for one window."""

    queue: deque[tuple[int, str]] = field(default_factory=deque)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, timestamp_ns: int, counterparty: str) -> None:
        self.queue.append((timestamp_ns, counterparty))
        self.counts[counterparty] = self.counts.get(counterparty, 0) + 1

    def expire_before(self, lower_bound_ns: int) -> None:
        while self.queue and self.queue[0][0] < lower_bound_ns:
            _old_ts, old_counterparty = self.queue.popleft()
            remaining = self.counts[old_counterparty] - 1
            if remaining <= 0:
                del self.counts[old_counterparty]
            else:
                self.counts[old_counterparty] = remaining

    def value(self) -> float:
        return float(len(self.counts))


@dataclass
class CounterpartyAmountState:
    """Rolling amount concentration state for one window."""

    queue: deque[tuple[int, str, float]] = field(default_factory=deque)
    amounts: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    max_heap: list[tuple[float, str, int]] = field(default_factory=list)
    versions: dict[str, int] = field(default_factory=dict)
    total: float = 0.0
    sum_sq: float = 0.0

    def _next_version(self, counterparty: str) -> int:
        version = self.versions.get(counterparty, 0) + 1
        self.versions[counterparty] = version
        return version

    def _push_amount(self, counterparty: str, amount: float) -> None:
        version = self._next_version(counterparty)
        if amount > 0.0:
            heapq.heappush(self.max_heap, (-amount, counterparty, version))

    def _invalidate_amount(self, counterparty: str) -> None:
        self._next_version(counterparty)
        self.amounts.pop(counterparty, None)

    def _cleanup_empty_state(self) -> None:
        if self.queue:
            return
        self.amounts.clear()
        self.counts.clear()
        self.max_heap.clear()
        self.versions.clear()
        self.total = 0.0
        self.sum_sq = 0.0

    def _clamp_running_totals(self) -> None:
        total_tol = _scaled_float_tolerance(self.total)
        sum_sq_tol = _scaled_float_tolerance(self.sum_sq)
        if abs(self.total) <= total_tol:
            self.total = 0.0
        if abs(self.sum_sq) <= sum_sq_tol:
            self.sum_sq = 0.0

    def add(self, timestamp_ns: int, counterparty: str, amount: float) -> None:
        self.queue.append((timestamp_ns, counterparty, amount))
        old = float(self.amounts.get(counterparty, 0.0))
        new = old + amount
        self.amounts[counterparty] = new
        self.counts[counterparty] = self.counts.get(counterparty, 0) + 1
        self.total += amount
        self.sum_sq += new * new - old * old
        self._push_amount(counterparty, new)

    def expire_before(self, lower_bound_ns: int) -> None:
        while self.queue and self.queue[0][0] < lower_bound_ns:
            _timestamp_ns, counterparty, amount = self.queue.popleft()
            if counterparty not in self.amounts or counterparty not in self.counts:
                raise ValueError(
                    "Feature operation failed: counterparty amount state lost an active event. "
                    f"counterparty={counterparty!r}"
                )
            old = float(self.amounts[counterparty])
            remaining_count = self.counts[counterparty] - 1
            new = old - amount
            self.total -= amount
            if remaining_count <= 0:
                tolerance = _expire_float_tolerance(old, amount, new)
                if abs(new) > tolerance:
                    raise ValueError(
                        "Feature operation failed: last counterparty amount did not expire cleanly. "
                        f"counterparty={counterparty!r}, old_amount={old}, expired_amount={amount}, residual={new}"
                    )
                del self.counts[counterparty]
                self._invalidate_amount(counterparty)
                new = 0.0
            else:
                tolerance = _scaled_float_tolerance(old, amount, new)
                if new < 0.0 and abs(new) <= tolerance:
                    new = 0.0
                elif new < 0.0:
                    raise ValueError(
                        "Feature operation failed: counterparty amount state became negative. "
                        f"counterparty={counterparty!r}, old_amount={old}, expired_amount={amount}, new_amount={new}"
                    )
                self.counts[counterparty] = remaining_count
                self.amounts[counterparty] = new
                self._push_amount(counterparty, new)
            self.sum_sq += new * new - old * old
            self._clamp_running_totals()
        self._cleanup_empty_state()

    def _positive_amount_total(self) -> float:
        return float(math.fsum(amount for amount in self.amounts.values() if amount > 0.0))

    def _positive_amount_sum_sq(self) -> float:
        return float(math.fsum(amount * amount for amount in self.amounts.values() if amount > 0.0))

    def _current_max_amount(self) -> float:
        while self.max_heap:
            negative_amount, counterparty, version = self.max_heap[0]
            current = self.amounts.get(counterparty)
            if current is None or self.versions.get(counterparty) != version or current <= 0.0:
                heapq.heappop(self.max_heap)
                continue
            heap_amount = -negative_amount
            if not math.isclose(heap_amount, current, rel_tol=_FLOAT_REL_TOL, abs_tol=_FLOAT_ABS_TOL):
                heapq.heappop(self.max_heap)
                continue
            return float(current)
        return 0.0

    def effective_n(self) -> float:
        if not self.amounts:
            return 0.0
        total = self.total
        sum_sq = self.sum_sq
        if total <= 0.0 or sum_sq <= 0.0:
            total = self._positive_amount_total()
            sum_sq = self._positive_amount_sum_sq()
        if total <= 0.0 or sum_sq <= 0.0:
            return 0.0
        value = float((total * total) / sum_sq)
        if not math.isfinite(value) or value < 0.0:
            total = self._positive_amount_total()
            sum_sq = self._positive_amount_sum_sq()
            if total <= 0.0 or sum_sq <= 0.0:
                return 0.0
            value = float((total * total) / sum_sq)
        return max(value, 0.0)

    def top1_share(self) -> float:
        if not self.amounts:
            return 0.0
        max_amount = self._current_max_amount()
        if max_amount <= 0.0:
            return 0.0
        total = self.total
        tolerance = _scaled_float_tolerance(total, max_amount)
        if total <= 0.0 or max_amount > total + tolerance:
            total = self._positive_amount_total()
        if total <= 0.0:
            return 0.0
        value = float(max_amount / total)
        if not math.isfinite(value):
            total = self._positive_amount_total()
            value = 0.0 if total <= 0.0 else float(max_amount / total)
        if value < 0.0:
            if abs(value) <= _scaled_float_tolerance(value):
                return 0.0
            raise ValueError(f"Feature operation failed: top1_share became negative. value={value}")
        if value > 1.0:
            if value <= 1.0 + _scaled_float_tolerance(value):
                return 1.0
            total = self._positive_amount_total()
            value = 0.0 if total <= 0.0 else float(max_amount / total)
            if value > 1.0 + _scaled_float_tolerance(value):
                raise ValueError(
                    "Feature operation failed: top1_share exceeded 1 after state recomputation. "
                    f"max_amount={max_amount}, total={total}, value={value}"
                )
            if value < 0.0:
                raise ValueError(
                    "Feature operation failed: top1_share became negative after state recomputation. "
                    f"max_amount={max_amount}, total={total}, value={value}"
                )
        return min(max(value, 0.0), 1.0)


def _empty_category_mapping() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CATEGORY_MAPPING_COLUMNS))


def _empty_category_unknown_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CATEGORY_UNKNOWN_COLUMNS))


def _validate_closed_left(spec: FeatureSpec, operation: str) -> str:
    closed = str(param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            f"Feature operation failed: {operation} only supports closed='left' to avoid leakage. "
            f"output_col={spec.output_col!r}, observed_closed={closed!r}"
        )
    return closed


def _counterparty_nunique_spec_parts(spec: FeatureSpec) -> tuple[dict[str, str], pd.Timedelta, str, float, str]:
    require_allowed_params(spec, ("window", "closed", "fill_value", "dtype"))
    roles = require_roles(spec, ("current_entity_col", "history_entity_col", "counterparty_col", "timestamp_col"))
    window = parse_window(param_value(spec, "window", ""), spec.operation, spec.output_col)
    closed = _validate_closed_left(spec, spec.operation)
    fill_value = float(param_value(spec, "fill_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, window, closed, fill_value, dtype


def _counterparty_nunique_group_key(spec: FeatureSpec) -> CounterpartyNuniqueGroupKey:
    roles, _window, closed, _fill_value, _dtype = _counterparty_nunique_spec_parts(spec)
    return (
        roles["current_entity_col"],
        roles["history_entity_col"],
        roles["counterparty_col"],
        roles["timestamp_col"],
        closed,
    )


def _counterparty_amount_spec_parts(spec: FeatureSpec) -> tuple[dict[str, str], pd.Timedelta, str, float, str]:
    require_allowed_params(spec, ("window", "closed", "fill_value", "dtype"))
    roles = require_roles(
        spec,
        ("current_entity_col", "history_entity_col", "counterparty_col", "timestamp_col", "value_col"),
    )
    window = parse_window(param_value(spec, "window", ""), spec.operation, spec.output_col)
    closed = _validate_closed_left(spec, spec.operation)
    fill_value = float(param_value(spec, "fill_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, window, closed, fill_value, dtype


def _counterparty_amount_group_key(spec: FeatureSpec) -> CounterpartyAmountGroupKey:
    roles, _window, closed, _fill_value, _dtype = _counterparty_amount_spec_parts(spec)
    return (
        roles["current_entity_col"],
        roles["history_entity_col"],
        roles["counterparty_col"],
        roles["timestamp_col"],
        roles["value_col"],
        closed,
    )


def _entity_group_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        empty = np.array([], dtype="int64")
        return empty, empty
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    return np.concatenate(([0], breaks)), np.concatenate((breaks, [len(values)]))


def _timestamp_group_bounds(timestamp_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(timestamp_ns) == 0:
        empty = np.array([], dtype="int64")
        return empty, empty
    breaks = np.flatnonzero(timestamp_ns[1:] != timestamp_ns[:-1]) + 1
    return np.concatenate(([0], breaks)), np.concatenate((breaks, [len(timestamp_ns)]))


def _require_non_negative_values(values: pd.Series, *, value_col: str, operation: str) -> None:
    negative_mask = values < 0
    if bool(negative_mask.any()):
        examples = values.loc[negative_mask].head(5).tolist()
        raise ValueError(
            "Feature operation failed: amount concentration requires non-negative values. "
            f"operation={operation!r}, value_col={value_col!r}, negative_count={int(negative_mask.sum())}, "
            f"examples={examples}"
        )


def _event_query_frames(
    df: pd.DataFrame,
    *,
    current_entity_col: str,
    history_entity_col: str,
    counterparty_col: str,
    timestamp_col: str,
    output_col: str,
    value_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = [current_entity_col, history_entity_col, counterparty_col, timestamp_col]
    if value_col is not None:
        required_columns.append(value_col)
    require_columns(df, tuple(required_columns), "rolling_counterparty")

    timestamps = parse_datetime_strict(df, timestamp_col, output_col)
    events_payload: dict[str, Any] = {
        "_entity": normalize_category_strict(df[history_entity_col], source_col=history_entity_col),
        "_timestamp": timestamps,
        "_counterparty": normalize_category_strict(df[counterparty_col], source_col=counterparty_col),
        "_row_order": np.arange(len(df)),
    }
    if value_col is not None:
        values = parse_numeric_strict(df, value_col, output_col).astype("float64")
        _require_non_negative_values(values, value_col=value_col, operation="rolling_counterparty_amount")
        events_payload["_value"] = values

    events = pd.DataFrame(events_payload).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")
    queries = pd.DataFrame(
        {
            "_entity": normalize_category_strict(df[current_entity_col], source_col=current_entity_col),
            "_timestamp": timestamps,
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")

    return events.reset_index(drop=True), queries.reset_index(drop=True)


def _scan_counterparty_windows(
    *,
    event_entities: np.ndarray,
    event_timestamps: np.ndarray,
    query_entities: np.ndarray,
    query_timestamps: np.ndarray,
    query_rows: np.ndarray,
    window_ns_values: tuple[int, ...],
    make_state: Callable[[], Any],
    add_event: Callable[[Any, int], None],
    record_state: Callable[[int, Any, np.ndarray], None],
) -> None:
    """Run the shared entity/timestamp two-pointer scan for counterparty features."""

    event_starts, event_ends = _entity_group_bounds(event_entities)
    query_starts, query_ends = _entity_group_bounds(query_entities)

    event_group_idx = 0
    query_group_idx = 0
    while query_group_idx < len(query_starts):
        query_entity = query_entities[query_starts[query_group_idx]]
        while event_group_idx < len(event_starts) and event_entities[event_starts[event_group_idx]] < query_entity:
            event_group_idx += 1

        if event_group_idx >= len(event_starts) or event_entities[event_starts[event_group_idx]] != query_entity:
            query_group_idx += 1
            continue

        event_start = int(event_starts[event_group_idx])
        event_end_bound = int(event_ends[event_group_idx])
        query_start = int(query_starts[query_group_idx])
        query_end = int(query_ends[query_group_idx])

        local_query_timestamps = query_timestamps[query_start:query_end]
        timestamp_starts, timestamp_ends = _timestamp_group_bounds(local_query_timestamps)
        states = {window_ns: make_state() for window_ns in window_ns_values}
        event_end = event_start

        for timestamp_start, timestamp_end in zip(timestamp_starts, timestamp_ends):
            current_ts = int(local_query_timestamps[timestamp_start])
            # Leakage guard: only events strictly before the query timestamp enter history.
            while event_end < event_end_bound and int(event_timestamps[event_end]) < current_ts:
                for state in states.values():
                    add_event(state, event_end)
                event_end += 1

            row_slice = query_rows[query_start + timestamp_start : query_start + timestamp_end]
            for window_ns, state in states.items():
                # Window lower bound is inclusive; same-timestamp events were not added above.
                state.expire_before(current_ts - window_ns)
                record_state(window_ns, state, row_slice)

        query_group_idx += 1


def _execute_counterparty_nunique_group(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    if not specs:
        return {}

    group_key = _counterparty_nunique_group_key(specs[0])
    first_roles, _first_window, _first_closed, _first_fill_value, _first_dtype = _counterparty_nunique_spec_parts(specs[0])
    spec_parts: list[tuple[FeatureSpec, dict[str, str], pd.Timedelta, str, float, str]] = []
    for spec in specs:
        if spec.operation != "rolling_counterparty_nunique":
            raise ValueError(
                "Feature build failed: counterparty nunique batch received an incompatible spec. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        if _counterparty_nunique_group_key(spec) != group_key:
            raise ValueError(
                "Feature build failed: counterparty nunique batch group contains incompatible specs. "
                f"first_output_col={specs[0].output_col!r}, observed_output_col={spec.output_col!r}"
            )
        spec_parts.append((spec, *_counterparty_nunique_spec_parts(spec)))

    events, queries = _event_query_frames(
        df,
        current_entity_col=first_roles["current_entity_col"],
        history_entity_col=first_roles["history_entity_col"],
        counterparty_col=first_roles["counterparty_col"],
        timestamp_col=first_roles["timestamp_col"],
        output_col=specs[0].output_col,
    )

    event_entities = events["_entity"].to_numpy()
    event_timestamps = events["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    event_counterparties = events["_counterparty"].to_numpy()
    query_entities = queries["_entity"].to_numpy()
    query_timestamps = queries["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    query_rows = queries["_row_order"].to_numpy(dtype="int64", copy=False)

    window_ns_values = tuple(dict.fromkeys(int(window.value) for _spec, _roles, window, _closed, _fill_value, _dtype in spec_parts))
    raw_outputs = {window_ns: np.full(len(df), np.nan, dtype="float64") for window_ns in window_ns_values}

    def add_event(state: CounterpartyNuniqueState, event_index: int) -> None:
        state.add(int(event_timestamps[event_index]), str(event_counterparties[event_index]))

    def record_state(window_ns: int, state: CounterpartyNuniqueState, row_slice: np.ndarray) -> None:
        raw_outputs[window_ns][row_slice] = state.value()

    _scan_counterparty_windows(
        event_entities=event_entities,
        event_timestamps=event_timestamps,
        query_entities=query_entities,
        query_timestamps=query_timestamps,
        query_rows=query_rows,
        window_ns_values=window_ns_values,
        make_state=CounterpartyNuniqueState,
        add_event=add_event,
        record_state=record_state,
    )

    results: dict[str, FeatureOpResult] = {}
    for spec, roles, window, closed, fill_value, dtype in spec_parts:
        output = pd.Series(raw_outputs[int(window.value)].copy(), index=df.index).fillna(fill_value)
        params = {"window": str(window), "closed": closed, "fill_value": fill_value}
        results[spec.output_col] = finalize_result(
            output,
            spec,
            row_count=len(df),
            input_columns=roles,
            params=params,
            dtype=dtype,
        )
    return results


def _execute_counterparty_amount_group(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    if not specs:
        return {}

    valid_operations = {"rolling_counterparty_effective_n", "rolling_counterparty_top1_share"}
    group_key = _counterparty_amount_group_key(specs[0])
    first_roles, _first_window, _first_closed, _first_fill_value, _first_dtype = _counterparty_amount_spec_parts(specs[0])
    spec_parts: list[tuple[FeatureSpec, dict[str, str], pd.Timedelta, str, float, str]] = []
    for spec in specs:
        if spec.operation not in valid_operations:
            raise ValueError(
                "Feature build failed: counterparty amount batch received an incompatible spec. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        if _counterparty_amount_group_key(spec) != group_key:
            raise ValueError(
                "Feature build failed: counterparty amount batch group contains incompatible specs. "
                f"first_output_col={specs[0].output_col!r}, observed_output_col={spec.output_col!r}"
            )
        spec_parts.append((spec, *_counterparty_amount_spec_parts(spec)))

    events, queries = _event_query_frames(
        df,
        current_entity_col=first_roles["current_entity_col"],
        history_entity_col=first_roles["history_entity_col"],
        counterparty_col=first_roles["counterparty_col"],
        timestamp_col=first_roles["timestamp_col"],
        value_col=first_roles["value_col"],
        output_col=specs[0].output_col,
    )

    event_entities = events["_entity"].to_numpy()
    event_timestamps = events["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    event_counterparties = events["_counterparty"].to_numpy()
    event_values = events["_value"].to_numpy(dtype="float64", copy=False)
    query_entities = queries["_entity"].to_numpy()
    query_timestamps = queries["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    query_rows = queries["_row_order"].to_numpy(dtype="int64", copy=False)

    window_ns_values = tuple(dict.fromkeys(int(window.value) for _spec, _roles, window, _closed, _fill_value, _dtype in spec_parts))
    metric_by_spec = {spec.output_col: "top1_share" if spec.operation == "rolling_counterparty_top1_share" else "effective_n" for spec in specs}
    raw_outputs = {
        (window_ns, metric): np.full(len(df), np.nan, dtype="float64")
        for window_ns in window_ns_values
        for metric in {"effective_n", "top1_share"}
        if any(int(window.value) == window_ns and metric_by_spec[spec.output_col] == metric for spec, _roles, window, _closed, _fill_value, _dtype in spec_parts)
    }

    def add_event(state: CounterpartyAmountState, event_index: int) -> None:
        state.add(
            int(event_timestamps[event_index]),
            str(event_counterparties[event_index]),
            float(event_values[event_index]),
        )

    def record_state(window_ns: int, state: CounterpartyAmountState, row_slice: np.ndarray) -> None:
        if (window_ns, "effective_n") in raw_outputs:
            raw_outputs[(window_ns, "effective_n")][row_slice] = state.effective_n()
        if (window_ns, "top1_share") in raw_outputs:
            raw_outputs[(window_ns, "top1_share")][row_slice] = state.top1_share()

    _scan_counterparty_windows(
        event_entities=event_entities,
        event_timestamps=event_timestamps,
        query_entities=query_entities,
        query_timestamps=query_timestamps,
        query_rows=query_rows,
        window_ns_values=window_ns_values,
        make_state=CounterpartyAmountState,
        add_event=add_event,
        record_state=record_state,
    )

    results: dict[str, FeatureOpResult] = {}
    for spec, roles, window, closed, fill_value, dtype in spec_parts:
        metric = metric_by_spec[spec.output_col]
        output = pd.Series(raw_outputs[(int(window.value), metric)].copy(), index=df.index).fillna(fill_value)
        if metric == "top1_share" and bool(((output < 0.0) | (output > 1.0)).any()):
            raise ValueError(
                "Feature operation failed: top1_share must be within [0, 1]. "
                f"output_col={spec.output_col!r}, min={float(output.min())}, max={float(output.max())}"
            )
        params = {"window": str(window), "closed": closed, "fill_value": fill_value}
        results[spec.output_col] = finalize_result(
            output,
            spec,
            row_count=len(df),
            input_columns=roles,
            params=params,
            dtype=dtype,
        )
    return results


def execute_rolling_counterparty_nunique_specs_batched(
    df: pd.DataFrame,
    specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """Execute unique counterparty specs grouped by shared input columns."""

    if not specs:
        return {}
    validate_feature_specs(specs)
    grouped_specs: dict[CounterpartyNuniqueGroupKey, list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation != "rolling_counterparty_nunique":
            raise ValueError(
                "Feature build failed: execute_rolling_counterparty_nunique_specs_batched only accepts matching specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_counterparty_nunique_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        group_results = _execute_counterparty_nunique_group(df, tuple(group_specs))
        overlap = set(results) & set(group_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate counterparty nunique results. output_cols={sorted(overlap)}")
        results.update(group_results)
    return results


def execute_rolling_counterparty_amount_specs_batched(
    df: pd.DataFrame,
    specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """Execute effective_n/top1_share specs grouped by shared input columns."""

    if not specs:
        return {}
    validate_feature_specs(specs)
    valid_operations = {"rolling_counterparty_effective_n", "rolling_counterparty_top1_share"}
    grouped_specs: dict[CounterpartyAmountGroupKey, list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation not in valid_operations:
            raise ValueError(
                "Feature build failed: execute_rolling_counterparty_amount_specs_batched only accepts amount specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_counterparty_amount_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        group_results = _execute_counterparty_amount_group(df, tuple(group_specs))
        overlap = set(results) & set(group_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate counterparty amount results. output_cols={sorted(overlap)}")
        results.update(group_results)
    return results


def _precompute_feature_results(
    df: pd.DataFrame,
    feature_specs: Tuple[FeatureSpec, ...],
) -> tuple[dict[str, FeatureOpResult], dict[str, FeatureOpResult], dict[str, FeatureOpResult]]:
    """Run batchable operations once per compatible group."""

    rolling_specs = tuple(spec for spec in feature_specs if spec.operation == "rolling_agg")
    counterparty_nunique_specs = tuple(spec for spec in feature_specs if spec.operation == "rolling_counterparty_nunique")
    counterparty_amount_specs = tuple(
        spec
        for spec in feature_specs
        if spec.operation in {"rolling_counterparty_effective_n", "rolling_counterparty_top1_share"}
    )
    rolling_results = execute_rolling_agg_specs_batched(df, rolling_specs)
    nunique_results = execute_rolling_counterparty_nunique_specs_batched(df, counterparty_nunique_specs)
    amount_results = execute_rolling_counterparty_amount_specs_batched(df, counterparty_amount_specs)
    return rolling_results, nunique_results, amount_results


def _result_for_spec(
    spec: FeatureSpec,
    *,
    rolling_results: dict[str, FeatureOpResult],
    nunique_results: dict[str, FeatureOpResult],
    amount_results: dict[str, FeatureOpResult],
) -> FeatureOpResult:
    """Return a result from the batch caches in FeatureSpec order."""

    if spec.operation == "rolling_agg":
        return rolling_results[spec.output_col]
    if spec.operation == "rolling_counterparty_nunique":
        return nunique_results[spec.output_col]
    if spec.operation in {"rolling_counterparty_effective_n", "rolling_counterparty_top1_share"}:
        return amount_results[spec.output_col]
    raise ValueError(
        "Feature build failed: unknown operation. "
        f"operation={spec.operation!r}, output_col={spec.output_col!r}, "
        f"supported_operations={sorted(SUPPORTED_BATCH_OPERATIONS)}"
    )


def execute_feature_specs(
    df: pd.DataFrame,
    feature_specs: Tuple[FeatureSpec, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Execute selected specs and return feature frame, feature_info, and artifacts."""

    validate_feature_specs(feature_specs)
    missing_meta = set(META_COLUMNS) - set(df.columns)
    if missing_meta:
        raise ValueError(f"Feature execution input is missing metadata columns: {sorted(missing_meta)}")

    feature_parts: list[pd.DataFrame] = []
    feature_info_parts: list[pd.DataFrame] = []
    category_mapping_parts: list[pd.DataFrame] = []
    category_unknown_parts: list[pd.DataFrame] = []

    rolling_results, nunique_results, amount_results = _precompute_feature_results(df, feature_specs)
    for spec in feature_specs:
        result = _result_for_spec(
            spec,
            rolling_results=rolling_results,
            nunique_results=nunique_results,
            amount_results=amount_results,
        )
        expected = [spec.output_col]
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
        if "category_mapping" in result.artifacts:
            category_mapping_parts.append(result.artifacts["category_mapping"])
        if "category_unknown_summary" in result.artifacts:
            category_unknown_parts.append(result.artifacts["category_unknown_summary"])

    selected_columns = feature_columns(feature_specs)
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
