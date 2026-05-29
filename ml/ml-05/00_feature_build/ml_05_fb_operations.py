"""ML-05 Stage 4 flow-balance/pass-through feature execution.

All history features enforce ``history_timestamp < current_timestamp`` and
exclude the current row plus every same-timestamp row.

Code map:
- Input: validated split_df and ML-05 FeatureSpec tuple.
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

from ml_05_fb_operation_result_validation import finalize_result, param_value, require_allowed_params, require_columns, require_roles
from ml_05_fb_rolling import execute_rolling_agg_specs_batched, parse_window
from ml_05_fb_schema import META_COLUMNS, normalize_category_strict, parse_datetime_strict, parse_numeric_strict
from ml_05_fb_specs import FeatureOpResult, FeatureSpec, feature_columns, validate_feature_specs


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
    "flowbalance_window",
    "passflow_window",
    "rolling_agg",
}

FLOWBALANCE_METRICS = {
    "in_out_ratio",
    "sumdiff",
    "residual_abs",
    "balanced_state_flag",
}
PASSFLOW_METRICS = {
    "seconds_since_last_in",
    "current_to_last_in_amount_ratio",
    "recent_flag",
    "sequence_count",
    "historical_sequence_count",
}
ACCOUNT_ROLES = {"sender", "receiver"}
ACCOUNT_PROGRESS_ROWS = 500_000
ACCOUNT_PROGRESS_FRACTION = 0.10

FlowbalanceGroupKey = tuple[str, str, str, str, str]
PassflowGroupKey = tuple[str, str, str, str, str]


@dataclass
class AccountFlowState:
    """Rolling inbound/outbound amount state for one account and one window."""

    queue: deque[tuple[int, str, float]] = field(default_factory=deque)
    inbound_sum: float = 0.0
    outbound_sum: float = 0.0

    def add(self, timestamp_ns: int, direction: str, amount: float) -> None:
        if direction not in {"in", "out"}:
            raise ValueError(f"Unsupported account flow direction: {direction!r}")
        self.queue.append((timestamp_ns, direction, amount))
        if direction == "in":
            self.inbound_sum += amount
        else:
            self.outbound_sum += amount

    def expire_before(self, lower_bound_ns: int) -> None:
        while self.queue and self.queue[0][0] < lower_bound_ns:
            _timestamp_ns, direction, amount = self.queue.popleft()
            if direction == "in":
                self.inbound_sum -= amount
            else:
                self.outbound_sum -= amount
        if not self.queue:
            self.inbound_sum = 0.0
            self.outbound_sum = 0.0

    def is_empty(self) -> bool:
        return not self.queue


@dataclass
class AccountFlowStore:
    """All account flow states for one rolling window."""

    states: dict[str, AccountFlowState] = field(default_factory=dict)
    event_keys: deque[tuple[int, str]] = field(default_factory=deque)

    def expire_before(self, lower_bound_ns: int) -> None:
        while self.event_keys and self.event_keys[0][0] < lower_bound_ns:
            _old_timestamp_ns, account = self.event_keys.popleft()
            state = self.states.get(account)
            if state is None:
                continue
            state.expire_before(lower_bound_ns)
            if state.is_empty():
                del self.states[account]

    def get(self, account: str) -> Optional[AccountFlowState]:
        return self.states.get(account)

    def add(self, timestamp_ns: int, account: str, direction: str, amount: float) -> None:
        self.states.setdefault(account, AccountFlowState()).add(timestamp_ns, direction, amount)
        self.event_keys.append((timestamp_ns, account))


@dataclass
class PassFlowState:
    """Rolling recent inbound and inbound-then-outbound sequence state for one account."""

    inbound_events: deque[tuple[int, float]] = field(default_factory=deque)
    sequence_events: deque[int] = field(default_factory=deque)

    def expire_before(self, lower_bound_ns: int) -> None:
        while self.inbound_events and self.inbound_events[0][0] < lower_bound_ns:
            self.inbound_events.popleft()
        while self.sequence_events and self.sequence_events[0] < lower_bound_ns:
            self.sequence_events.popleft()

    def last_inbound(self) -> Optional[tuple[int, float]]:
        if not self.inbound_events:
            return None
        return self.inbound_events[-1]

    def add_inbound(self, timestamp_ns: int, amount: float) -> None:
        self.inbound_events.append((timestamp_ns, amount))

    def add_sequence(self, timestamp_ns: int) -> None:
        self.sequence_events.append(timestamp_ns)

    def sequence_count(self) -> float:
        return float(len(self.sequence_events))

    def is_empty(self) -> bool:
        return not self.inbound_events and not self.sequence_events


@dataclass
class PassFlowStore:
    """All pass-through states for one rolling window."""

    states: dict[str, PassFlowState] = field(default_factory=dict)
    event_keys: deque[tuple[int, str]] = field(default_factory=deque)

    def expire_before(self, lower_bound_ns: int) -> None:
        while self.event_keys and self.event_keys[0][0] < lower_bound_ns:
            _old_timestamp_ns, account = self.event_keys.popleft()
            state = self.states.get(account)
            if state is None:
                continue
            state.expire_before(lower_bound_ns)
            if state.is_empty():
                del self.states[account]

    def get(self, account: str) -> Optional[PassFlowState]:
        return self.states.get(account)

    def ensure(self, account: str) -> PassFlowState:
        return self.states.setdefault(account, PassFlowState())

    def add_inbound(self, timestamp_ns: int, account: str, amount: float) -> None:
        self.ensure(account).add_inbound(timestamp_ns, amount)
        self.event_keys.append((timestamp_ns, account))

    def add_sequence(self, timestamp_ns: int, account: str) -> None:
        self.ensure(account).add_sequence(timestamp_ns)
        self.event_keys.append((timestamp_ns, account))


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


def _validate_account_role(value: Any, *, operation: str, output_col: str) -> str:
    role = str(value).strip().lower()
    if role not in ACCOUNT_ROLES:
        raise ValueError(
            "Feature operation failed: unsupported account role. "
            f"operation={operation!r}, output_col={output_col!r}, role={role!r}, supported={sorted(ACCOUNT_ROLES)}"
        )
    return role


def _flowbalance_spec_parts(spec: FeatureSpec) -> tuple[dict[str, str], pd.Timedelta, str, str, str, float, float, str]:
    require_allowed_params(spec, ("window", "metric", "role", "closed", "fill_value", "balance_threshold_quantile", "dtype"))
    roles = require_roles(spec, ("sender_col", "receiver_col", "timestamp_col", "amount_col"))
    window = parse_window(param_value(spec, "window", ""), spec.operation, spec.output_col)
    metric = str(param_value(spec, "metric", "")).strip().lower()
    if metric not in FLOWBALANCE_METRICS:
        raise ValueError(
            "Feature operation failed: unsupported flowbalance metric. "
            f"output_col={spec.output_col!r}, metric={metric!r}, supported={sorted(FLOWBALANCE_METRICS)}"
        )
    role = _validate_account_role(param_value(spec, "role", ""), operation=spec.operation, output_col=spec.output_col)
    closed = _validate_closed_left(spec, spec.operation)
    fill_value = float(param_value(spec, "fill_value", 0.0))
    balance_quantile = float(param_value(spec, "balance_threshold_quantile", 0.25))
    if not 0.0 <= balance_quantile <= 1.0:
        raise ValueError(
            "Feature operation failed: balance_threshold_quantile must be between 0 and 1. "
            f"output_col={spec.output_col!r}, balance_threshold_quantile={balance_quantile!r}"
        )
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, window, metric, role, closed, fill_value, balance_quantile, dtype


def _flowbalance_group_key(spec: FeatureSpec) -> FlowbalanceGroupKey:
    roles, _window, _metric, _role, closed, _fill_value, _quantile, _dtype = _flowbalance_spec_parts(spec)
    return (
        roles["sender_col"],
        roles["receiver_col"],
        roles["timestamp_col"],
        roles["amount_col"],
        closed,
    )


def _passflow_spec_parts(spec: FeatureSpec) -> tuple[dict[str, str], pd.Timedelta, str, str, str, float, str]:
    require_allowed_params(spec, ("window", "metric", "role", "closed", "fill_value", "dtype"))
    roles = require_roles(spec, ("sender_col", "receiver_col", "timestamp_col", "amount_col"))
    window = parse_window(param_value(spec, "window", ""), spec.operation, spec.output_col)
    metric = str(param_value(spec, "metric", "")).strip().lower()
    if metric not in PASSFLOW_METRICS:
        raise ValueError(
            "Feature operation failed: unsupported passflow metric. "
            f"output_col={spec.output_col!r}, metric={metric!r}, supported={sorted(PASSFLOW_METRICS)}"
        )
    role = _validate_account_role(param_value(spec, "role", ""), operation=spec.operation, output_col=spec.output_col)
    if role == "receiver" and metric != "historical_sequence_count":
        raise ValueError(
            "Feature operation failed: receiver passflow only supports historical_sequence_count. "
            f"output_col={spec.output_col!r}, metric={metric!r}"
        )
    if role == "sender" and metric == "historical_sequence_count":
        raise ValueError(
            "Feature operation failed: sender passflow uses sequence_count, not historical_sequence_count. "
            f"output_col={spec.output_col!r}"
        )
    closed = _validate_closed_left(spec, spec.operation)
    fill_value = float(param_value(spec, "fill_value", 0.0))
    dtype = str(param_value(spec, "dtype", "float32"))
    return roles, window, metric, role, closed, fill_value, dtype


def _passflow_group_key(spec: FeatureSpec) -> PassflowGroupKey:
    roles, _window, _metric, _role, closed, _fill_value, _dtype = _passflow_spec_parts(spec)
    return (
        roles["sender_col"],
        roles["receiver_col"],
        roles["timestamp_col"],
        roles["amount_col"],
        closed,
    )


def _require_non_negative_amounts(values: pd.Series, *, value_col: str, operation: str) -> None:
    negative_mask = values < 0
    if bool(negative_mask.any()):
        examples = values.loc[negative_mask].head(5).tolist()
        raise ValueError(
            "Feature operation failed: account flow amounts must be non-negative. "
            f"operation={operation!r}, value_col={value_col!r}, negative_count={int(negative_mask.sum())}, "
            f"examples={examples}"
        )


def _account_work_frame(df: pd.DataFrame, roles: dict[str, str], output_col: str) -> pd.DataFrame:
    sender_col = roles["sender_col"]
    receiver_col = roles["receiver_col"]
    timestamp_col = roles["timestamp_col"]
    amount_col = roles["amount_col"]
    require_columns(df, (sender_col, receiver_col, timestamp_col, amount_col), "account_flow")

    amount = parse_numeric_strict(df, amount_col, output_col).astype("float64")
    _require_non_negative_amounts(amount, value_col=amount_col, operation="account_flow")
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


def _progress_interval(row_count: int) -> int:
    if row_count <= 0:
        return 1
    ten_percent = max(1, int(row_count * ACCOUNT_PROGRESS_FRACTION))
    return min(ACCOUNT_PROGRESS_ROWS, ten_percent)


def _account_for_role(role: str, sender: str, receiver: str) -> str:
    return sender if role == "sender" else receiver


def _flow_values(state: Optional[AccountFlowState]) -> tuple[float, float]:
    if state is None:
        return 0.0, 0.0
    return max(float(state.inbound_sum), 0.0), max(float(state.outbound_sum), 0.0)


def _balance_residual_ratio(inbound_sum: float, outbound_sum: float) -> float:
    total = inbound_sum + outbound_sum
    if total <= 0.0:
        return np.nan
    return abs(inbound_sum - outbound_sum) / total


def _flowbalance_value(metric: str, inbound_sum: float, outbound_sum: float) -> float:
    if metric == "in_out_ratio":
        return 0.0 if outbound_sum <= 0.0 else float(inbound_sum / outbound_sum)
    if metric == "sumdiff":
        return float(inbound_sum - outbound_sum)
    if metric == "residual_abs":
        return float(abs(inbound_sum - outbound_sum))
    if metric == "balanced_state_flag":
        return _balance_residual_ratio(inbound_sum, outbound_sum)
    raise ValueError(f"Unsupported flowbalance metric during execution: {metric!r}")


def _split_values_for_train_fit(df: pd.DataFrame, *, output_col: str) -> np.ndarray:
    if "split" not in df.columns:
        raise ValueError(
            "Feature operation failed: train-fitted flow-balance threshold requires an existing split column. "
            f"output_col={output_col!r}"
        )
    split_values = df["split"].astype("string").str.strip().str.lower().to_numpy()
    observed = set(str(value) for value in split_values)
    required = {"train", "val", "test"}
    if not required.issubset(observed):
        raise ValueError(
            "Feature operation failed: train-fitted flow-balance threshold requires train/val/test split values. "
            f"output_col={output_col!r}, observed={sorted(observed)}"
        )
    return split_values


def _balanced_state_values(
    residual_ratio: np.ndarray,
    split_values: np.ndarray,
    *,
    quantile: float,
    output_col: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    ratio = np.asarray(residual_ratio, dtype="float64")
    finite = np.isfinite(ratio)
    fit_mask = (split_values == "train") & finite
    fit_count = int(fit_mask.sum())
    if fit_count <= 0:
        raise ValueError(
            "Feature operation failed: balanced_state_flag has no train rows with prior inbound/outbound history. "
            f"output_col={output_col!r}, fix=Use a larger split-aware sample or full input."
        )
    threshold = float(np.quantile(ratio[fit_mask], quantile))
    if not np.isfinite(threshold):
        raise ValueError(
            "Feature operation failed: balanced_state_flag fitted an invalid threshold. "
            f"output_col={output_col!r}, threshold={threshold!r}"
        )
    values = np.zeros(len(ratio), dtype="float64")
    values[finite & (ratio <= threshold)] = 1.0
    metadata = {
        "balance_threshold_fit_split": "train",
        "balance_threshold_quantile": quantile,
        "balance_threshold_value": threshold,
        "train_fit_rows": fit_count,
        "no_history_rows": int((~finite).sum()),
    }
    return values, metadata


def _execute_flowbalance_group(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    if not specs:
        return {}

    started_at = perf_counter()
    group_key = _flowbalance_group_key(specs[0])
    first_roles, _first_window, _first_metric, _first_role, _closed, _fill_value, _quantile, _dtype = _flowbalance_spec_parts(specs[0])
    spec_parts: list[tuple[FeatureSpec, dict[str, str], pd.Timedelta, str, str, str, float, float, str]] = []
    for spec in specs:
        if spec.operation != "flowbalance_window":
            raise ValueError(f"Feature build failed: flowbalance batch received operation={spec.operation!r}")
        if _flowbalance_group_key(spec) != group_key:
            raise ValueError(
                "Feature build failed: flowbalance batch group contains incompatible specs. "
                f"first_output_col={specs[0].output_col!r}, observed_output_col={spec.output_col!r}"
            )
        spec_parts.append((spec, *_flowbalance_spec_parts(spec)))

    work = _account_work_frame(df, first_roles, specs[0].output_col)
    timestamp_ns = work["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    senders = work["_sender"].to_numpy()
    receivers = work["_receiver"].to_numpy()
    amounts = work["_amount"].to_numpy(dtype="float64", copy=False)
    row_orders = work["_row_order"].to_numpy(dtype="int64", copy=False)

    window_ns_values = tuple(dict.fromkeys(int(window.value) for _spec, _roles, window, *_rest in spec_parts))
    outputs = {spec.output_col: np.full(len(df), np.nan, dtype="float64") for spec, *_rest in spec_parts}
    stores = {window_ns: AccountFlowStore() for window_ns in window_ns_values}
    timestamp_starts, timestamp_ends = _timestamp_group_bounds(timestamp_ns)
    row_count = len(work)
    progress_interval = _progress_interval(row_count)
    next_progress = progress_interval
    print(
        "[ML-05 flowbalance_window] start "
        f"rows={row_count} specs={len(spec_parts)} windows={len(window_ns_values)} "
        f"timestamp_groups={len(timestamp_starts)}",
        flush=True,
    )

    for timestamp_start, timestamp_end in zip(timestamp_starts, timestamp_ends):
        current_ts = int(timestamp_ns[timestamp_start])
        for window_ns, store in stores.items():
            store.expire_before(current_ts - window_ns)

        for work_index in range(timestamp_start, timestamp_end):
            sender = str(senders[work_index])
            receiver = str(receivers[work_index])
            row_order = int(row_orders[work_index])
            for spec, _roles, window, metric, role, _closed, _fill_value, _quantile, _dtype in spec_parts:
                account = _account_for_role(role, sender, receiver)
                state = stores[int(window.value)].get(account)
                inbound_sum, outbound_sum = _flow_values(state)
                outputs[spec.output_col][row_order] = _flowbalance_value(metric, inbound_sum, outbound_sum)

        for work_index in range(timestamp_start, timestamp_end):
            event_ts = int(timestamp_ns[work_index])
            amount = float(amounts[work_index])
            sender = str(senders[work_index])
            receiver = str(receivers[work_index])
            for store in stores.values():
                store.add(event_ts, sender, "out", amount)
                store.add(event_ts, receiver, "in", amount)

        processed_rows = int(timestamp_end)
        if processed_rows >= next_progress or processed_rows == row_count:
            elapsed = perf_counter() - started_at
            pct = 100.0 if row_count == 0 else processed_rows / row_count * 100.0
            print(
                "[ML-05 flowbalance_window] progress "
                f"rows={processed_rows}/{row_count} pct={pct:.1f} elapsed_sec={elapsed:.1f}",
                flush=True,
            )
            while next_progress <= processed_rows:
                next_progress += progress_interval

    needs_train_fit = any(metric == "balanced_state_flag" for _spec, _roles, _window, metric, *_rest in spec_parts)
    split_values = _split_values_for_train_fit(df, output_col=specs[0].output_col) if needs_train_fit else np.array([], dtype=object)

    results: dict[str, FeatureOpResult] = {}
    for spec, roles, window, metric, role, closed, fill_value, quantile, dtype in spec_parts:
        values = outputs[spec.output_col].copy()
        params: dict[str, Any] = {
            "window": str(window),
            "metric": metric,
            "role": role,
            "closed": closed,
            "fill_value": fill_value,
        }
        if metric == "balanced_state_flag":
            values, threshold_params = _balanced_state_values(
                values,
                split_values,
                quantile=quantile,
                output_col=spec.output_col,
            )
            params.update(threshold_params)
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
        "[ML-05 flowbalance_window] end "
        f"rows={row_count} specs={len(spec_parts)} windows={len(window_ns_values)} elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return results


def _passflow_sender_value(metric: str, state: Optional[PassFlowState], current_ts: int, current_amount: float) -> float:
    last_inbound = None if state is None else state.last_inbound()
    if metric == "sequence_count":
        return 0.0 if state is None else state.sequence_count()
    if last_inbound is None:
        return 0.0
    last_ts, last_amount = last_inbound
    if metric == "seconds_since_last_in":
        return float((current_ts - last_ts) / 1_000_000_000)
    if metric == "current_to_last_in_amount_ratio":
        return 0.0 if last_amount <= 0.0 else float(current_amount / last_amount)
    if metric == "recent_flag":
        return 1.0
    raise ValueError(f"Unsupported sender passflow metric during execution: {metric!r}")


def _passflow_receiver_value(metric: str, state: Optional[PassFlowState]) -> float:
    if metric != "historical_sequence_count":
        raise ValueError(f"Unsupported receiver passflow metric during execution: {metric!r}")
    return 0.0 if state is None else state.sequence_count()


def _execute_passflow_group(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    if not specs:
        return {}

    started_at = perf_counter()
    group_key = _passflow_group_key(specs[0])
    first_roles, _first_window, _first_metric, _first_role, _closed, _fill_value, _dtype = _passflow_spec_parts(specs[0])
    spec_parts: list[tuple[FeatureSpec, dict[str, str], pd.Timedelta, str, str, str, float, str]] = []
    for spec in specs:
        if spec.operation != "passflow_window":
            raise ValueError(f"Feature build failed: passflow batch received operation={spec.operation!r}")
        if _passflow_group_key(spec) != group_key:
            raise ValueError(
                "Feature build failed: passflow batch group contains incompatible specs. "
                f"first_output_col={specs[0].output_col!r}, observed_output_col={spec.output_col!r}"
            )
        spec_parts.append((spec, *_passflow_spec_parts(spec)))

    work = _account_work_frame(df, first_roles, specs[0].output_col)
    timestamp_ns = work["_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    senders = work["_sender"].to_numpy()
    receivers = work["_receiver"].to_numpy()
    amounts = work["_amount"].to_numpy(dtype="float64", copy=False)
    row_orders = work["_row_order"].to_numpy(dtype="int64", copy=False)

    window_ns_values = tuple(dict.fromkeys(int(window.value) for _spec, _roles, window, *_rest in spec_parts))
    outputs = {spec.output_col: np.full(len(df), np.nan, dtype="float64") for spec, *_rest in spec_parts}
    stores = {window_ns: PassFlowStore() for window_ns in window_ns_values}
    timestamp_starts, timestamp_ends = _timestamp_group_bounds(timestamp_ns)
    row_count = len(work)
    progress_interval = _progress_interval(row_count)
    next_progress = progress_interval
    print(
        "[ML-05 passflow_window] start "
        f"rows={row_count} specs={len(spec_parts)} windows={len(window_ns_values)} "
        f"timestamp_groups={len(timestamp_starts)}",
        flush=True,
    )

    for timestamp_start, timestamp_end in zip(timestamp_starts, timestamp_ends):
        current_ts = int(timestamp_ns[timestamp_start])
        for window_ns, store in stores.items():
            store.expire_before(current_ts - window_ns)

        for work_index in range(timestamp_start, timestamp_end):
            sender = str(senders[work_index])
            receiver = str(receivers[work_index])
            current_amount = float(amounts[work_index])
            row_order = int(row_orders[work_index])
            for spec, _roles, window, metric, role, _closed, _fill_value, _dtype in spec_parts:
                state = stores[int(window.value)].get(_account_for_role(role, sender, receiver))
                if role == "sender":
                    outputs[spec.output_col][row_order] = _passflow_sender_value(metric, state, current_ts, current_amount)
                else:
                    outputs[spec.output_col][row_order] = _passflow_receiver_value(metric, state)

        for _window_ns, store in stores.items():
            for work_index in range(timestamp_start, timestamp_end):
                sender = str(senders[work_index])
                sender_state = store.get(sender)
                if sender_state is not None and sender_state.last_inbound() is not None:
                    store.add_sequence(int(timestamp_ns[work_index]), sender)
            for work_index in range(timestamp_start, timestamp_end):
                receiver = str(receivers[work_index])
                store.add_inbound(int(timestamp_ns[work_index]), receiver, float(amounts[work_index]))

        processed_rows = int(timestamp_end)
        if processed_rows >= next_progress or processed_rows == row_count:
            elapsed = perf_counter() - started_at
            pct = 100.0 if row_count == 0 else processed_rows / row_count * 100.0
            print(
                "[ML-05 passflow_window] progress "
                f"rows={processed_rows}/{row_count} pct={pct:.1f} elapsed_sec={elapsed:.1f}",
                flush=True,
            )
            while next_progress <= processed_rows:
                next_progress += progress_interval

    results: dict[str, FeatureOpResult] = {}
    for spec, roles, window, metric, role, closed, fill_value, dtype in spec_parts:
        output = pd.Series(outputs[spec.output_col].copy(), index=df.index).fillna(fill_value)
        params = {"window": str(window), "metric": metric, "role": role, "closed": closed, "fill_value": fill_value}
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
        "[ML-05 passflow_window] end "
        f"rows={row_count} specs={len(spec_parts)} windows={len(window_ns_values)} elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return results


def execute_flowbalance_specs_batched(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    """Execute flow-balance specs grouped by shared input columns."""

    if not specs:
        return {}
    validate_feature_specs(specs)
    grouped_specs: dict[FlowbalanceGroupKey, list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation != "flowbalance_window":
            raise ValueError(
                "Feature build failed: execute_flowbalance_specs_batched only accepts flowbalance_window specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_flowbalance_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        group_results = _execute_flowbalance_group(df, tuple(group_specs))
        overlap = set(results) & set(group_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate flowbalance results. output_cols={sorted(overlap)}")
        results.update(group_results)
    return results


def execute_passflow_specs_batched(df: pd.DataFrame, specs: Tuple[FeatureSpec, ...]) -> dict[str, FeatureOpResult]:
    """Execute pass-through sequence specs grouped by shared input columns."""

    if not specs:
        return {}
    validate_feature_specs(specs)
    grouped_specs: dict[PassflowGroupKey, list[FeatureSpec]] = {}
    for spec in specs:
        if spec.operation != "passflow_window":
            raise ValueError(
                "Feature build failed: execute_passflow_specs_batched only accepts passflow_window specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        grouped_specs.setdefault(_passflow_group_key(spec), []).append(spec)

    results: dict[str, FeatureOpResult] = {}
    for group_specs in grouped_specs.values():
        group_results = _execute_passflow_group(df, tuple(group_specs))
        overlap = set(results) & set(group_results)
        if overlap:
            raise ValueError(f"Feature build failed: duplicate passflow results. output_cols={sorted(overlap)}")
        results.update(group_results)
    return results


def _precompute_feature_results(
    df: pd.DataFrame,
    feature_specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """Run each batchable operation once per compatible group."""

    flowbalance_specs = tuple(spec for spec in feature_specs if spec.operation == "flowbalance_window")
    passflow_specs = tuple(spec for spec in feature_specs if spec.operation == "passflow_window")
    rolling_specs = tuple(spec for spec in feature_specs if spec.operation == "rolling_agg")

    results: dict[str, FeatureOpResult] = {}
    for batch_results in (
        execute_flowbalance_specs_batched(df, flowbalance_specs),
        execute_passflow_specs_batched(df, passflow_specs),
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
