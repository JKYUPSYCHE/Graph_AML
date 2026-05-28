"""ML-04 Stage 3 sender-receiver relationship feature execution.

All history features enforce ``history_timestamp < current_timestamp`` and
exclude the current row plus every same-timestamp row.

Code map:
- Input: validated split_df and ML-04 FeatureSpec tuple.
- Output: generated feature frame, feature_info, and empty encoding artifacts.
- Public: execute_feature_specs plus operation-specific batched executors.
- Leakage guard: timestamp groups are queried before they enter history.
- Notes: ratio features use 0 when the historical denominator is missing or zero.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from ml_04_fb_operation_result_validation import finalize_result, param_value, require_allowed_params, require_columns, require_roles
from ml_04_fb_rolling import execute_rolling_agg_specs_batched, parse_window
from ml_04_fb_schema import META_COLUMNS, normalize_category_strict, parse_datetime_strict, parse_numeric_strict
from ml_04_fb_specs import FeatureOpResult, FeatureSpec, feature_columns, validate_feature_specs


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

SUPPORTED_BATCH_OPERATIONS = {
    "pair_window",
    "pair_history",
    "current_self_loop",
    "bank_pair_history",
    "rolling_agg",
}

PAIR_RATIO_STABILITY_LOG_METRIC = "current_vs_forward_mean_ratio_log1p_clip_p999_hist2"
PAIR_RATIO_STABILITY_FLAG_METRIC = "current_vs_forward_mean_ratio_hist_lt2_flag"
PAIR_RATIO_RAW_METRIC = "current_vs_forward_mean_ratio"
PAIR_RATIO_STABILITY_CLIP_QUANTILE = 0.999
PAIR_RATIO_STABILITY_MIN_HISTORY_COUNT = 2.0
PAIR_WINDOW_PROGRESS_ROWS = 500_000
PAIR_WINDOW_PROGRESS_FRACTION = 0.10

PAIR_WINDOW_METRICS = {
    "forward_tx_count",
    "forward_amount_sum",
    "forward_amount_mean",
    "forward_amount_std",
    "forward_amount_max",
    "reverse_exists",
    "reverse_tx_count",
    "net_amount_sumdiff",
    PAIR_RATIO_RAW_METRIC,
    PAIR_RATIO_STABILITY_LOG_METRIC,
    PAIR_RATIO_STABILITY_FLAG_METRIC,
    "bidirectional_amount_ratio",
}
PAIR_HISTORY_METRICS = {
    "seconds_since_last_tx",
    "is_new_pair",
    "age_hours_since_first",
}

PairWindowGroupKey = tuple[str, str, str, str, str]
PairHistoryGroupKey = tuple[str, str, str, str]
BankPairHistoryGroupKey = tuple[str, str, str]


@dataclass
class PairWindowState:
    """Rolling amount/count state for one directed pair and one window."""

    queue: deque[tuple[int, float, int]] = field(default_factory=deque)
    max_queue: deque[tuple[float, int]] = field(default_factory=deque)
    active_sequence_ids: set[int] = field(default_factory=set)
    total: float = 0.0
    sum_sq: float = 0.0
    count: int = 0
    next_sequence_id: int = 0

    def add(self, timestamp_ns: int, amount: float) -> None:
        sequence_id = self.next_sequence_id
        self.next_sequence_id += 1
        self.queue.append((timestamp_ns, amount, sequence_id))
        while self.max_queue and self.max_queue[-1][0] <= amount:
            self.max_queue.pop()
        self.max_queue.append((amount, sequence_id))
        self.active_sequence_ids.add(sequence_id)
        self.total += amount
        self.sum_sq += amount * amount
        self.count += 1

    def expire_before(self, lower_bound_ns: int) -> None:
        while self.queue and self.queue[0][0] < lower_bound_ns:
            _old_timestamp_ns, old_amount, old_sequence_id = self.queue.popleft()
            self.active_sequence_ids.discard(old_sequence_id)
            self.total -= old_amount
            self.sum_sq -= old_amount * old_amount
            self.count -= 1
        if self.count <= 0:
            self.total = 0.0
            self.sum_sq = 0.0
            self.count = 0
            self.queue.clear()
            self.max_queue.clear()
            self.active_sequence_ids.clear()

    def tx_count(self) -> float:
        return float(self.count)

    def amount_sum(self) -> float:
        return float(self.total)

    def amount_mean(self) -> float:
        return 0.0 if self.count == 0 else float(self.total / self.count)

    def amount_std(self) -> float:
        if self.count <= 1:
            return 0.0
        variance_numerator = self.sum_sq - (self.total * self.total / self.count)
        return float(np.sqrt(max(variance_numerator, 0.0) / (self.count - 1)))

    def amount_max(self) -> float:
        while self.max_queue and self.max_queue[0][1] not in self.active_sequence_ids:
            self.max_queue.popleft()
        if not self.max_queue:
            return 0.0
        return float(self.max_queue[0][0])


@dataclass
class PairWindowStore:
    """All directed-pair states for one rolling window."""

    states: dict[tuple[str, str], PairWindowState] = field(default_factory=dict)
    event_keys: deque[tuple[int, tuple[str, str]]] = field(default_factory=deque)

    def expire_before(self, lower_bound_ns: int) -> None:
        while self.event_keys and self.event_keys[0][0] < lower_bound_ns:
            _old_timestamp_ns, old_key = self.event_keys.popleft()
            state = self.states.get(old_key)
            if state is None:
                continue
            state.expire_before(lower_bound_ns)
            if state.count == 0:
                del self.states[old_key]

    def get(self, key: tuple[str, str]) -> Optional[PairWindowState]:
        return self.states.get(key)

    def add(self, timestamp_ns: int, key: tuple[str, str], amount: float) -> None:
        self.states.setdefault(key, PairWindowState()).add(timestamp_ns, amount)
        self.event_keys.append((timestamp_ns, key))


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


def _pair_window_spec_parts(spec: FeatureSpec) -> tuple[dict[str, str], pd.Timedelta, str, str, float, str]:
    require_allowed_params(spec, ("window", "metric", "closed", "fill_value", "dtype"))
    roles = require_roles(spec, ("sender_col", "receiver_col", "timestamp_col", "amount_col"))
    window = parse_window(param_value(spec, "window", ""), spec.operation, spec.output_col)
    metric = str(param_value(spec, "metric", "")).strip()
    if metric not in PAIR_WINDOW_METRICS:
        raise ValueError(
            "Feature operation failed: unsupported pair_window metric. "
            f"output_col={spec.output_col!r}, metric={metric!r}, supported={sorted(PAIR_WINDOW_METRICS)}"
        )
    closed = _validate_closed_left(spec, spec.operation)
    fill_value = float(param_value(spec, "fill_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, window, metric, closed, fill_value, dtype


def _pair_window_group_key(spec: FeatureSpec) -> PairWindowGroupKey:
    roles, _window, _metric, closed, _fill_value, _dtype = _pair_window_spec_parts(spec)
    return (
        roles["sender_col"],
        roles["receiver_col"],
        roles["timestamp_col"],
        roles["amount_col"],
        closed,
    )


def _pair_history_spec_parts(spec: FeatureSpec) -> tuple[dict[str, str], str, float, str]:
    require_allowed_params(spec, ("metric", "window", "fill_value", "dtype"))
    roles = require_roles(spec, ("sender_col", "receiver_col", "timestamp_col", "amount_col"))
    metric = str(param_value(spec, "metric", "")).strip()
    if metric not in PAIR_HISTORY_METRICS:
        raise ValueError(
            "Feature operation failed: unsupported pair_history metric. "
            f"output_col={spec.output_col!r}, metric={metric!r}, supported={sorted(PAIR_HISTORY_METRICS)}"
        )
    fill_value = float(param_value(spec, "fill_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, metric, fill_value, dtype


def _pair_history_group_key(spec: FeatureSpec) -> PairHistoryGroupKey:
    roles, _metric, _fill_value, _dtype = _pair_history_spec_parts(spec)
    return roles["sender_col"], roles["receiver_col"], roles["timestamp_col"], roles["amount_col"]


def _bank_pair_history_spec_parts(spec: FeatureSpec) -> tuple[dict[str, str], str, float, str]:
    require_allowed_params(spec, ("metric", "window", "fill_value", "dtype"))
    roles = require_roles(spec, ("sender_bank_col", "receiver_bank_col", "timestamp_col"))
    metric = str(param_value(spec, "metric", "")).strip()
    if metric != "cumulative_count":
        raise ValueError(
            "Feature operation failed: unsupported bank_pair_history metric. "
            f"output_col={spec.output_col!r}, metric={metric!r}, supported=['cumulative_count']"
        )
    fill_value = float(param_value(spec, "fill_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, metric, fill_value, dtype


def _bank_pair_history_group_key(spec: FeatureSpec) -> BankPairHistoryGroupKey:
    roles, _metric, _fill_value, _dtype = _bank_pair_history_spec_parts(spec)
    return roles["sender_bank_col"], roles["receiver_bank_col"], roles["timestamp_col"]


def _require_non_negative_amounts(values: pd.Series, *, value_col: str, operation: str) -> None:
    negative_mask = values < 0
    if bool(negative_mask.any()):
        examples = values.loc[negative_mask].head(5).tolist()
        raise ValueError(
            "Feature operation failed: pair relationship amounts must be non-negative. "
            f"operation={operation!r}, value_col={value_col!r}, negative_count={int(negative_mask.sum())}, "
            f"examples={examples}"
        )


def _pair_work_frame(df: pd.DataFrame, roles: dict[str, str], output_col: str) -> pd.DataFrame:
    sender_col = roles["sender_col"]
    receiver_col = roles["receiver_col"]
    timestamp_col = roles["timestamp_col"]
    amount_col = roles["amount_col"]
    require_columns(df, (sender_col, receiver_col, timestamp_col, amount_col), "pair")

    amount = parse_numeric_strict(df, amount_col, output_col).astype("float64")
    _require_non_negative_amounts(amount, value_col=amount_col, operation="pair")
    work = pd.DataFrame(
        {
            "_sender": normalize_category_strict(df[sender_col], source_col=sender_col),
            "_receiver": normalize_category_strict(df[receiver_col], source_col=receiver_col),
            "_timestamp": parse_datetime_strict(df, timestamp_col, output_col),
            "_amount": amount,
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_timestamp", "_row_order"], kind="mergesort")
    return work.reset_index(drop=True)


def _timestamp_group_bounds(timestamp_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(timestamp_ns) == 0:
        empty = np.array([], dtype="int64")
        return empty, empty
    breaks = np.flatnonzero(timestamp_ns[1:] != timestamp_ns[:-1]) + 1
    return np.concatenate(([0], breaks)), np.concatenate((breaks, [len(timestamp_ns)]))


def _pair_window_progress_interval(row_count: int) -> int:
    """Return the next row interval for pair-window progress logs."""

    if row_count <= 0:
        return 1
    ten_percent = max(1, int(row_count * PAIR_WINDOW_PROGRESS_FRACTION))
    return min(PAIR_WINDOW_PROGRESS_ROWS, ten_percent)


def _count(state: Optional[PairWindowState]) -> float:
    return 0.0 if state is None else state.tx_count()


def _amount_sum(state: Optional[PairWindowState]) -> float:
    return 0.0 if state is None else state.amount_sum()


def _amount_mean(state: Optional[PairWindowState]) -> float:
    return 0.0 if state is None else state.amount_mean()


def _amount_std(state: Optional[PairWindowState]) -> float:
    return 0.0 if state is None else state.amount_std()


def _amount_max(state: Optional[PairWindowState]) -> float:
    return 0.0 if state is None else state.amount_max()


def _state_value(
    metric: str,
    current_amount: float,
    forward: Optional[PairWindowState],
    reverse: Optional[PairWindowState],
) -> float:
    if metric == "forward_tx_count":
        return _count(forward)
    if metric == "forward_amount_sum":
        return _amount_sum(forward)
    if metric == "forward_amount_mean":
        return _amount_mean(forward)
    if metric == "forward_amount_std":
        return _amount_std(forward)
    if metric == "forward_amount_max":
        return _amount_max(forward)
    if metric == "reverse_exists":
        return 1.0 if _count(reverse) > 0.0 else 0.0
    if metric == "reverse_tx_count":
        return _count(reverse)
    if metric == "net_amount_sumdiff":
        return _amount_sum(forward) - _amount_sum(reverse)
    if metric in {PAIR_RATIO_RAW_METRIC, PAIR_RATIO_STABILITY_LOG_METRIC}:
        denominator = _amount_mean(forward)
        return 0.0 if denominator <= 0.0 else float(current_amount / denominator)
    if metric == PAIR_RATIO_STABILITY_FLAG_METRIC:
        return 1.0 if _count(forward) < PAIR_RATIO_STABILITY_MIN_HISTORY_COUNT else 0.0
    if metric == "bidirectional_amount_ratio":
        denominator = _amount_sum(reverse)
        return 0.0 if denominator <= 0.0 else float(_amount_sum(forward) / denominator)
    raise ValueError(f"Unsupported pair_window metric during execution: {metric!r}")


def _finite_non_negative_values(values: np.ndarray, *, output_col: str, value_name: str) -> np.ndarray:
    """Validate numeric values used by pair-ratio stability transforms."""

    checked = np.asarray(values, dtype="float64")
    invalid_mask = ~np.isfinite(checked)
    if bool(invalid_mask.any()):
        examples = checked[invalid_mask][:5].tolist()
        raise ValueError(
            "Feature operation failed: pair ratio stability input contains NaN/inf values. "
            f"output_col={output_col!r}, value_name={value_name!r}, invalid_count={int(invalid_mask.sum())}, "
            f"examples={examples}"
        )
    negative_mask = checked < 0.0
    if bool(negative_mask.any()):
        examples = checked[negative_mask][:5].tolist()
        raise ValueError(
            "Feature operation failed: pair ratio stability input contains negative values. "
            f"output_col={output_col!r}, value_name={value_name!r}, negative_count={int(negative_mask.sum())}, "
            f"examples={examples}"
        )
    return checked


def _split_values_for_stability(df: pd.DataFrame, *, output_col: str) -> np.ndarray:
    if "split" not in df.columns:
        raise ValueError(
            "Feature operation failed: pair ratio stability requires an existing split column "
            "to fit clipping cap on train only. "
            f"output_col={output_col!r}"
        )
    split_values = df["split"].astype("string").str.strip().str.lower().to_numpy()
    observed = set(str(value) for value in split_values)
    required = {"train", "val", "test"}
    if not required.issubset(observed):
        raise ValueError(
            "Feature operation failed: pair ratio stability requires train/val/test split values. "
            f"output_col={output_col!r}, observed={sorted(observed)}"
        )
    return split_values


def _stabilized_pair_ratio_values(
    raw_ratio: np.ndarray,
    history_count: np.ndarray,
    split_values: np.ndarray,
    *,
    output_col: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return log1p(train-p999-clipped ratio), gated to zero when history count is below 2."""

    ratio = _finite_non_negative_values(raw_ratio, output_col=output_col, value_name="raw_ratio")
    count = _finite_non_negative_values(history_count, output_col=output_col, value_name="history_count")
    history_lt_min = count < PAIR_RATIO_STABILITY_MIN_HISTORY_COUNT
    train_mask = split_values == "train"
    fit_mask = train_mask & ~history_lt_min
    fit_count = int(fit_mask.sum())
    if fit_count <= 0:
        raise ValueError(
            "Feature operation failed: pair ratio stability has no train rows with sufficient history. "
            f"output_col={output_col!r}, min_history_count={PAIR_RATIO_STABILITY_MIN_HISTORY_COUNT:g}"
        )

    cap = float(np.quantile(ratio[fit_mask], PAIR_RATIO_STABILITY_CLIP_QUANTILE))
    if not np.isfinite(cap) or cap < 0.0:
        raise ValueError(
            "Feature operation failed: pair ratio stability fitted an invalid clipping cap. "
            f"output_col={output_col!r}, cap={cap!r}"
        )

    capped = np.minimum(ratio, cap)
    transformed = np.log1p(capped)
    transformed[history_lt_min] = 0.0
    metadata: dict[str, Any] = {
        "clip_fit_split": "train",
        "clip_quantile": PAIR_RATIO_STABILITY_CLIP_QUANTILE,
        "clip_value": cap,
        "min_history_count": int(PAIR_RATIO_STABILITY_MIN_HISTORY_COUNT),
        "train_fit_rows": fit_count,
        "history_lt_min_rows": int(history_lt_min.sum()),
    }
    return transformed, metadata


def _history_lt_min_flag_values(
    history_count: np.ndarray,
    *,
    output_col: str,
) -> np.ndarray:
    count = _finite_non_negative_values(history_count, output_col=output_col, value_name="history_count")
    return (count < PAIR_RATIO_STABILITY_MIN_HISTORY_COUNT).astype("float64")


def _execute_pair_window_group(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    if not specs:
        return {}

    started_at = perf_counter()
    group_key = _pair_window_group_key(specs[0])
    first_roles, _first_window, _first_metric, _first_closed, _first_fill_value, _first_dtype = _pair_window_spec_parts(specs[0])
    spec_parts: list[tuple[FeatureSpec, dict[str, str], pd.Timedelta, str, str, float, str]] = []
    for spec in specs:
        if spec.operation != "pair_window":
            raise ValueError(f"Feature build failed: pair_window batch received operation={spec.operation!r}")
        if _pair_window_group_key(spec) != group_key:
            raise ValueError(
                "Feature build failed: pair_window batch group contains incompatible specs. "
                f"first_output_col={specs[0].output_col!r}, observed_output_col={spec.output_col!r}"
            )
        spec_parts.append((spec, *_pair_window_spec_parts(spec)))

    work = _pair_work_frame(df, first_roles, specs[0].output_col)
    timestamp_ns = work["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    senders = work["_sender"].to_numpy()
    receivers = work["_receiver"].to_numpy()
    amounts = work["_amount"].to_numpy(dtype="float64", copy=False)
    row_orders = work["_row_order"].to_numpy(dtype="int64", copy=False)

    window_ns_values = tuple(dict.fromkeys(int(window.value) for _spec, _roles, window, _metric, _closed, _fill, _dtype in spec_parts))
    stability_window_ns_values = tuple(
        dict.fromkeys(
            int(window.value)
            for _spec, _roles, window, metric, _closed, _fill, _dtype in spec_parts
            if metric in {PAIR_RATIO_STABILITY_LOG_METRIC, PAIR_RATIO_STABILITY_FLAG_METRIC}
        )
    )
    outputs = {spec.output_col: np.full(len(df), np.nan, dtype="float64") for spec, *_rest in spec_parts}
    history_count_outputs_by_window = {
        window_ns: np.full(len(df), np.nan, dtype="float64") for window_ns in stability_window_ns_values
    }
    stores = {window_ns: PairWindowStore() for window_ns in window_ns_values}
    timestamp_starts, timestamp_ends = _timestamp_group_bounds(timestamp_ns)
    row_count = len(work)
    progress_interval = _pair_window_progress_interval(row_count)
    next_progress = progress_interval
    print(
        "[ML-04 pair_window] start "
        f"rows={row_count} specs={len(spec_parts)} windows={len(window_ns_values)} "
        f"timestamp_groups={len(timestamp_starts)}",
        flush=True,
    )

    for timestamp_start, timestamp_end in zip(timestamp_starts, timestamp_ends):
        current_ts = int(timestamp_ns[timestamp_start])
        for window_ns, store in stores.items():
            store.expire_before(current_ts - window_ns)
        for work_index in range(timestamp_start, timestamp_end):
            row_order = int(row_orders[work_index])
            forward_key = (str(senders[work_index]), str(receivers[work_index]))
            reverse_key = (forward_key[1], forward_key[0])
            current_amount = float(amounts[work_index])
            for window_ns, history_count_output in history_count_outputs_by_window.items():
                history_count_output[row_order] = _count(stores[window_ns].get(forward_key))
            for spec, _roles, window, metric, _closed, _fill_value, _dtype in spec_parts:
                window_ns = int(window.value)
                store = stores[window_ns]
                forward_state = store.get(forward_key)
                reverse_state = store.get(reverse_key)
                outputs[spec.output_col][row_order] = _state_value(metric, current_amount, forward_state, reverse_state)

        for work_index in range(timestamp_start, timestamp_end):
            event_key = (str(senders[work_index]), str(receivers[work_index]))
            event_ts = int(timestamp_ns[work_index])
            event_amount = float(amounts[work_index])
            for store in stores.values():
                store.add(event_ts, event_key, event_amount)

        processed_rows = int(timestamp_end)
        if processed_rows >= next_progress or processed_rows == row_count:
            elapsed = perf_counter() - started_at
            pct = 100.0 if row_count == 0 else processed_rows / row_count * 100.0
            print(
                "[ML-04 pair_window] progress "
                f"rows={processed_rows}/{row_count} pct={pct:.1f} elapsed_sec={elapsed:.1f}",
                flush=True,
            )
            while next_progress <= processed_rows:
                next_progress += progress_interval

    needs_stability = any(
        metric in {PAIR_RATIO_STABILITY_LOG_METRIC, PAIR_RATIO_STABILITY_FLAG_METRIC}
        for _spec, _roles, _window, metric, _closed, _fill_value, _dtype in spec_parts
    )
    split_values = _split_values_for_stability(df, output_col=specs[0].output_col) if needs_stability else np.array([], dtype=object)

    results: dict[str, FeatureOpResult] = {}
    for spec, roles, window, metric, closed, fill_value, dtype in spec_parts:
        values = outputs[spec.output_col].copy()
        params = {"window": str(window), "metric": metric, "closed": closed, "fill_value": fill_value}
        if metric in {PAIR_RATIO_STABILITY_LOG_METRIC, PAIR_RATIO_STABILITY_FLAG_METRIC}:
            window_ns = int(window.value)
            history_count = history_count_outputs_by_window.get(window_ns)
            if history_count is None:
                raise ValueError(
                    "Feature operation failed: pair ratio stability did not compute internal history counts. "
                    f"output_col={spec.output_col!r}, window={window}"
                )
            if metric == PAIR_RATIO_STABILITY_LOG_METRIC:
                values, stability_params = _stabilized_pair_ratio_values(
                    values,
                    history_count,
                    split_values,
                    output_col=spec.output_col,
                )
                params.update(stability_params)
            else:
                values = _history_lt_min_flag_values(history_count, output_col=spec.output_col)
                params.update(
                    {
                        "min_history_count": int(PAIR_RATIO_STABILITY_MIN_HISTORY_COUNT),
                        "flag_condition": "history_count < min_history_count",
                    }
                )
        output = pd.Series(values, index=df.index).fillna(fill_value)
        results[spec.output_col] = finalize_result(
            output,
            spec,
            row_count=len(df),
            input_columns=roles,
            params=params,
            dtype=dtype,
        )
    elapsed = perf_counter() - started_at
    print(
        "[ML-04 pair_window] end "
        f"rows={row_count} specs={len(spec_parts)} windows={len(window_ns_values)} elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return results


def _execute_pair_history_group(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    if not specs:
        return {}

    group_key = _pair_history_group_key(specs[0])
    first_roles, _first_metric, _first_fill_value, _first_dtype = _pair_history_spec_parts(specs[0])
    spec_parts: list[tuple[FeatureSpec, dict[str, str], str, float, str]] = []
    for spec in specs:
        if spec.operation != "pair_history":
            raise ValueError(f"Feature build failed: pair_history batch received operation={spec.operation!r}")
        if _pair_history_group_key(spec) != group_key:
            raise ValueError(
                "Feature build failed: pair_history batch group contains incompatible specs. "
                f"first_output_col={specs[0].output_col!r}, observed_output_col={spec.output_col!r}"
            )
        spec_parts.append((spec, *_pair_history_spec_parts(spec)))

    work = _pair_work_frame(df, first_roles, specs[0].output_col)
    timestamp_ns = work["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    senders = work["_sender"].to_numpy()
    receivers = work["_receiver"].to_numpy()
    row_orders = work["_row_order"].to_numpy(dtype="int64", copy=False)

    outputs = {spec.output_col: np.full(len(df), np.nan, dtype="float64") for spec, *_rest in spec_parts}
    seen_count: dict[tuple[str, str], int] = {}
    first_seen_ts: dict[tuple[str, str], int] = {}
    last_seen_ts: dict[tuple[str, str], int] = {}
    timestamp_starts, timestamp_ends = _timestamp_group_bounds(timestamp_ns)

    for timestamp_start, timestamp_end in zip(timestamp_starts, timestamp_ends):
        current_ts = int(timestamp_ns[timestamp_start])
        for work_index in range(timestamp_start, timestamp_end):
            key = (str(senders[work_index]), str(receivers[work_index]))
            row_order = int(row_orders[work_index])
            count = seen_count.get(key, 0)
            for spec, _roles, metric, _fill_value, _dtype in spec_parts:
                if metric == "is_new_pair":
                    outputs[spec.output_col][row_order] = 1.0 if count == 0 else 0.0
                elif metric == "seconds_since_last_tx":
                    last_ts = last_seen_ts.get(key)
                    outputs[spec.output_col][row_order] = 0.0 if last_ts is None else float((current_ts - last_ts) / 1_000_000_000)
                elif metric == "age_hours_since_first":
                    first_ts = first_seen_ts.get(key)
                    outputs[spec.output_col][row_order] = 0.0 if first_ts is None else float((current_ts - first_ts) / 3_600_000_000_000)
                else:
                    raise ValueError(f"Unsupported pair_history metric during execution: {metric!r}")

        for work_index in range(timestamp_start, timestamp_end):
            key = (str(senders[work_index]), str(receivers[work_index]))
            seen_count[key] = seen_count.get(key, 0) + 1
            first_seen_ts.setdefault(key, int(timestamp_ns[work_index]))
            last_seen_ts[key] = int(timestamp_ns[work_index])

    results: dict[str, FeatureOpResult] = {}
    for spec, roles, metric, fill_value, dtype in spec_parts:
        output = pd.Series(outputs[spec.output_col].copy(), index=df.index).fillna(fill_value)
        params = {"metric": metric, "window": param_value(spec, "window", "whist"), "fill_value": fill_value}
        results[spec.output_col] = finalize_result(
            output,
            spec,
            row_count=len(df),
            input_columns=roles,
            params=params,
            dtype=dtype,
        )
    return results


def op_current_self_loop(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """Execute the current-row sender==receiver flag."""

    require_allowed_params(spec, ("window", "true_value", "false_value", "dtype"))
    roles = require_roles(spec, ("sender_col", "receiver_col"))
    sender_col = roles["sender_col"]
    receiver_col = roles["receiver_col"]
    require_columns(df, (sender_col, receiver_col), "current_self_loop")
    true_value = float(param_value(spec, "true_value", 1.0))
    false_value = float(param_value(spec, "false_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))
    sender = normalize_category_strict(df[sender_col], source_col=sender_col)
    receiver = normalize_category_strict(df[receiver_col], source_col=receiver_col)
    output = pd.Series(np.where(sender.to_numpy() == receiver.to_numpy(), true_value, false_value), index=df.index)
    params = {"window": param_value(spec, "window", "wcur"), "true_value": true_value, "false_value": false_value}
    return finalize_result(output, spec, row_count=len(df), input_columns=roles, params=params, dtype=dtype)


def _execute_bank_pair_history_group(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    if not specs:
        return {}

    group_key = _bank_pair_history_group_key(specs[0])
    first_roles, _first_metric, _first_fill_value, _first_dtype = _bank_pair_history_spec_parts(specs[0])
    spec_parts: list[tuple[FeatureSpec, dict[str, str], str, float, str]] = []
    for spec in specs:
        if spec.operation != "bank_pair_history":
            raise ValueError(f"Feature build failed: bank_pair_history batch received operation={spec.operation!r}")
        if _bank_pair_history_group_key(spec) != group_key:
            raise ValueError(
                "Feature build failed: bank_pair_history batch group contains incompatible specs. "
                f"first_output_col={specs[0].output_col!r}, observed_output_col={spec.output_col!r}"
            )
        spec_parts.append((spec, *_bank_pair_history_spec_parts(spec)))

    sender_bank_col = first_roles["sender_bank_col"]
    receiver_bank_col = first_roles["receiver_bank_col"]
    timestamp_col = first_roles["timestamp_col"]
    require_columns(df, (sender_bank_col, receiver_bank_col, timestamp_col), "bank_pair_history")
    work = pd.DataFrame(
        {
            "_sender_bank": normalize_category_strict(df[sender_bank_col], source_col=sender_bank_col),
            "_receiver_bank": normalize_category_strict(df[receiver_bank_col], source_col=receiver_bank_col),
            "_timestamp": parse_datetime_strict(df, timestamp_col, specs[0].output_col),
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_timestamp", "_row_order"], kind="mergesort")
    work = work.reset_index(drop=True)

    timestamp_ns = work["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    sender_banks = work["_sender_bank"].to_numpy()
    receiver_banks = work["_receiver_bank"].to_numpy()
    row_orders = work["_row_order"].to_numpy(dtype="int64", copy=False)
    outputs = {spec.output_col: np.full(len(df), np.nan, dtype="float64") for spec, *_rest in spec_parts}
    seen_count: dict[tuple[str, str], int] = {}
    timestamp_starts, timestamp_ends = _timestamp_group_bounds(timestamp_ns)

    for timestamp_start, timestamp_end in zip(timestamp_starts, timestamp_ends):
        for work_index in range(timestamp_start, timestamp_end):
            key = (str(sender_banks[work_index]), str(receiver_banks[work_index]))
            row_order = int(row_orders[work_index])
            for spec, _roles, metric, _fill_value, _dtype in spec_parts:
                if metric != "cumulative_count":
                    raise ValueError(f"Unsupported bank_pair_history metric during execution: {metric!r}")
                outputs[spec.output_col][row_order] = float(seen_count.get(key, 0))
        for work_index in range(timestamp_start, timestamp_end):
            key = (str(sender_banks[work_index]), str(receiver_banks[work_index]))
            seen_count[key] = seen_count.get(key, 0) + 1

    results: dict[str, FeatureOpResult] = {}
    for spec, roles, metric, fill_value, dtype in spec_parts:
        output = pd.Series(outputs[spec.output_col].copy(), index=df.index).fillna(fill_value)
        params = {"metric": metric, "window": param_value(spec, "window", "whist"), "fill_value": fill_value}
        results[spec.output_col] = finalize_result(
            output,
            spec,
            row_count=len(df),
            input_columns=roles,
            params=params,
            dtype=dtype,
        )
    return results


def execute_pair_window_specs_batched(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    """Execute pair window specs grouped by shared input columns."""

    if not specs:
        return {}
    validate_feature_specs(specs)
    grouped_specs: dict[PairWindowGroupKey, list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation != "pair_window":
            raise ValueError(
                "Feature build failed: execute_pair_window_specs_batched only accepts pair_window specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_pair_window_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        group_results = _execute_pair_window_group(df, tuple(group_specs))
        overlap = set(results) & set(group_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate pair_window results. output_cols={sorted(overlap)}")
        results.update(group_results)
    return results


def execute_pair_history_specs_batched(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    """Execute all-history pair specs grouped by shared input columns."""

    if not specs:
        return {}
    validate_feature_specs(specs)
    grouped_specs: dict[PairHistoryGroupKey, list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation != "pair_history":
            raise ValueError(
                "Feature build failed: execute_pair_history_specs_batched only accepts pair_history specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_pair_history_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        group_results = _execute_pair_history_group(df, tuple(group_specs))
        overlap = set(results) & set(group_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate pair_history results. output_cols={sorted(overlap)}")
        results.update(group_results)
    return results


def execute_bank_pair_history_specs_batched(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    """Execute bank-pair cumulative history specs grouped by shared input columns."""

    if not specs:
        return {}
    validate_feature_specs(specs)
    grouped_specs: dict[BankPairHistoryGroupKey, list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation != "bank_pair_history":
            raise ValueError(
                "Feature build failed: execute_bank_pair_history_specs_batched only accepts bank_pair_history specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_bank_pair_history_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        group_results = _execute_bank_pair_history_group(df, tuple(group_specs))
        overlap = set(results) & set(group_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate bank_pair_history results. output_cols={sorted(overlap)}")
        results.update(group_results)
    return results


def _precompute_feature_results(
    df: pd.DataFrame,
    feature_specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """Run each batchable operation once per compatible group."""

    pair_window_specs = tuple(spec for spec in feature_specs if spec.operation == "pair_window")
    pair_history_specs = tuple(spec for spec in feature_specs if spec.operation == "pair_history")
    self_loop_specs = tuple(spec for spec in feature_specs if spec.operation == "current_self_loop")
    bank_pair_specs = tuple(spec for spec in feature_specs if spec.operation == "bank_pair_history")
    rolling_specs = tuple(spec for spec in feature_specs if spec.operation == "rolling_agg")

    results: dict[str, FeatureOpResult] = {}
    for batch_results in (
        execute_pair_window_specs_batched(df, pair_window_specs),
        execute_pair_history_specs_batched(df, pair_history_specs),
        {spec.output_col: op_current_self_loop(df, spec) for spec in self_loop_specs},
        execute_bank_pair_history_specs_batched(df, bank_pair_specs),
        execute_rolling_agg_specs_batched(df, rolling_specs),
    ):
        overlap = set(results) & set(batch_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate operation results. output_cols={sorted(overlap)}")
        results.update(batch_results)
    return results


def _result_for_spec(spec: FeatureSpec, *, results: dict[str, FeatureOpResult]) -> FeatureOpResult:
    """Return a result from the batch cache in FeatureSpec order."""

    if spec.operation not in SUPPORTED_BATCH_OPERATIONS:
        raise ValueError(
            "Feature build failed: unknown operation. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, "
            f"supported_operations={sorted(SUPPORTED_BATCH_OPERATIONS)}"
        )
    if spec.output_col not in results:
        raise ValueError(f"Feature build failed: missing operation result. output_col={spec.output_col!r}")
    return results[spec.output_col]


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

    results = _precompute_feature_results(df, feature_specs)
    for spec in feature_specs:
        result = _result_for_spec(spec, results=results)
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

    selected_columns = feature_columns(feature_specs)
    feature_frame = pd.concat([df.loc[:, list(META_COLUMNS)].reset_index(drop=True), *feature_parts], axis=1)
    feature_frame["label"] = feature_frame["label"].astype("int8")
    feature_frame["split"] = feature_frame["split"].astype("string")
    feature_info = pd.concat(feature_info_parts, ignore_index=True)
    artifacts = {
        "category_mapping": _empty_category_mapping(),
        "category_unknown_summary": _empty_category_unknown_summary(),
    }
    if list(feature_frame.columns) != [*META_COLUMNS, *selected_columns]:
        raise ValueError(
            "Feature build failed: final feature frame column order mismatch. "
            f"observed={list(feature_frame.columns)}, expected={[*META_COLUMNS, *selected_columns]}"
        )
    return feature_frame, feature_info, artifacts
