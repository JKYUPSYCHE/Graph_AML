"""ML-05 Stage 4 flow-balance/pass-through FeatureSpec declarations.

This module declares feature contracts only. Actual computation is in
``ml_05_fb_operations.py``.

Code map:
- Input: none; specs are declared from fixed ML-05 Stage 4 catalog rules.
- Output: FeatureSpec tuples and generated feature column names.
- Public: FeatureSpec, FeatureOpResult, ml05_stage4_feature_specs, feature_columns.
- Leakage guard: every history spec uses timestamp < current_timestamp.
- Notes: full48 = flow-balance 32 + sender pass-flow 12 + receiver historical pass-flow 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Tuple

import pandas as pd


ML05_STAGE4_FEATURE_COUNT = 48
BALANCED_STATE_THRESHOLD_QUANTILE = 0.25

DEFAULT_FLOWBALANCE_WINDOWS: Tuple[Tuple[str, str], ...] = (
    ("w6h", "6h"),
    ("w1d", "1d"),
    ("w3d", "3d"),
    ("w7d", "7d"),
)

DEFAULT_SENDER_PASSFLOW_WINDOWS: Tuple[Tuple[str, str], ...] = (
    ("w1h", "1h"),
    ("w6h", "6h"),
    ("w1d", "1d"),
)

DEFAULT_RECEIVER_PASSFLOW_WINDOWS: Tuple[Tuple[str, str], ...] = (
    ("w6h", "6h"),
    ("w1d", "1d"),
    ("w3d", "3d"),
    ("w7d", "7d"),
)


@dataclass(frozen=True)
class FeatureSpec:
    """Single generated feature contract."""

    operation: str
    output_col: str
    input_cols: Mapping[str, str]
    params: Mapping[str, Any] = field(default_factory=dict)
    family: str = "unspecified"
    description: str = ""
    aml_typology: str = "unspecified"
    entity_scope: str = "transaction_row"
    direction: str = "current"
    leakage_policy: str = "unspecified"
    computational_cost: str = "low"
    used_in_ml: bool = True

    def required_columns(self) -> list[str]:
        """Return unique input column names required by this spec."""

        return list(dict.fromkeys(str(column) for column in self.input_cols.values()))


@dataclass(frozen=True)
class FeatureOpResult:
    """Standard operation result payload."""

    features: pd.DataFrame
    feature_info: pd.DataFrame
    artifacts: Mapping[str, Any] = field(default_factory=dict)


def _clean_name(value: str, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"FeatureSpec {field_name} must not be empty.")
    return cleaned


def validate_feature_specs(feature_specs: Tuple[FeatureSpec, ...]) -> None:
    """Validate basic FeatureSpec consistency before execution."""

    if not feature_specs:
        raise ValueError("feature_specs must not be empty. Add at least one FeatureSpec.")

    output_cols = [_clean_name(spec.output_col, "output_col") for spec in feature_specs]
    duplicated = sorted({column for column in output_cols if output_cols.count(column) > 1})
    if duplicated:
        raise ValueError(
            "Feature build failed: duplicate output columns in feature_specs. "
            f"duplicated={duplicated}, fix=Use unique output_col values."
        )

    for index, spec in enumerate(feature_specs):
        _clean_name(spec.operation, f"operation at index {index}")
        if not spec.input_cols:
            raise ValueError(
                "Feature build failed: FeatureSpec.input_cols must not be empty. "
                f"index={index}, output_col={spec.output_col!r}"
            )
        for role, column in spec.input_cols.items():
            _clean_name(str(role), f"input role at index {index}")
            _clean_name(str(column), f"input column for role {role!r} at index {index}")


def required_input_columns(
    feature_specs: Tuple[FeatureSpec, ...],
    extra_columns: Optional[Iterable[str]] = None,
) -> list[str]:
    """Return unique input columns required by selected specs."""

    validate_feature_specs(feature_specs)
    required: list[str] = []
    if extra_columns is not None:
        required.extend(str(column) for column in extra_columns)
    for spec in feature_specs:
        required.extend(spec.required_columns())
    return list(dict.fromkeys(required))


def feature_columns(feature_specs: Tuple[FeatureSpec, ...]) -> list[str]:
    """Return generated feature columns in execution order."""

    validate_feature_specs(feature_specs)
    return [spec.output_col for spec in feature_specs]


def _account_flow_input_cols() -> dict[str, str]:
    return {
        "sender_col": "sender_account_id",
        "receiver_col": "receiver_account_id",
        "timestamp_col": "timestamp",
        "amount_col": "amount",
    }


def flowbalance_window_spec(
    output_col: str,
    *,
    window: str,
    metric: str,
    role: str,
    description: str,
    aml_typology: str,
    computational_cost: str = "low",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """Declare an account-level inbound/outbound amount-balance feature."""

    params: dict[str, Any] = {"window": window, "metric": metric, "role": role, "closed": "left", "fill_value": 0.0}
    if metric == "balanced_state_flag":
        params["balance_threshold_quantile"] = BALANCED_STATE_THRESHOLD_QUANTILE
    return FeatureSpec(
        operation="flowbalance_window",
        output_col=output_col,
        input_cols=_account_flow_input_cols(),
        params=params,
        family="flowbalance",
        description=description,
        aml_typology=aml_typology,
        entity_scope=role,
        direction="net",
        leakage_policy="past-only; history_timestamp < current_timestamp; same-timestamp rows excluded",
        computational_cost=computational_cost,
        used_in_ml=used_in_ml,
    )


def passflow_window_spec(
    output_col: str,
    *,
    window: str,
    metric: str,
    role: str,
    description: str,
    aml_typology: str,
    computational_cost: str = "low",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """Declare an account-level inbound-then-outbound sequence feature."""

    return FeatureSpec(
        operation="passflow_window",
        output_col=output_col,
        input_cols=_account_flow_input_cols(),
        params={"window": window, "metric": metric, "role": role, "closed": "left", "fill_value": 0.0},
        family="passflow",
        description=description,
        aml_typology=aml_typology,
        entity_scope=role,
        direction="in_then_out",
        leakage_policy="past-only; history_timestamp < current_timestamp; same-timestamp rows excluded",
        computational_cost=computational_cost,
        used_in_ml=used_in_ml,
    )


def _add_flowbalance_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for role in ("sender", "receiver"):
        role_label = "current sender" if role == "sender" else "current receiver"
        for suffix, window in DEFAULT_FLOWBALANCE_WINDOWS:
            specs.append(
                flowbalance_window_spec(
                    f"flowbalance__{role}__net__amount__in_out_ratio__{suffix}",
                    window=window,
                    metric="in_out_ratio",
                    role=role,
                    description=(
                        f"Past inbound amount divided by past outbound amount for the {role_label} over {window}; "
                        "zero when outbound denominator is missing or zero."
                    ),
                    aml_typology="pass_through,flow_balance",
                    used_in_ml=used_in_ml,
                )
            )
            specs.append(
                flowbalance_window_spec(
                    f"flowbalance__{role}__net__amount__sumdiff__{suffix}",
                    window=window,
                    metric="sumdiff",
                    role=role,
                    description=f"Past inbound amount minus outbound amount for the {role_label} over {window}.",
                    aml_typology="pass_through,residual",
                    used_in_ml=used_in_ml,
                )
            )
            specs.append(
                flowbalance_window_spec(
                    f"flowbalance__{role}__net__amount__residual_abs__{suffix}",
                    window=window,
                    metric="residual_abs",
                    role=role,
                    description=f"Absolute past inbound/outbound amount difference for the {role_label} over {window}.",
                    aml_typology="residual,flow_balance",
                    used_in_ml=used_in_ml,
                )
            )
            specs.append(
                flowbalance_window_spec(
                    f"flowbalance__{role}__net__amount__balanced_state_flag__{suffix}",
                    window=window,
                    metric="balanced_state_flag",
                    role=role,
                    description=(
                        f"Whether the {role_label} has a train-fitted low residual ratio between "
                        f"past inbound and outbound amount over {window}."
                    ),
                    aml_typology="balanced_state,pass_through",
                    used_in_ml=used_in_ml,
                )
            )


def _add_sender_passflow_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for suffix, window in DEFAULT_SENDER_PASSFLOW_WINDOWS:
        specs.append(
            passflow_window_spec(
                f"passflow__sender__in_then_out__seconds_since_last_in__{suffix}",
                window=window,
                metric="seconds_since_last_in",
                role="sender",
                description=f"Seconds since the current sender's most recent inbound transaction within {window}.",
                aml_typology="pass_through,recency",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            passflow_window_spec(
                f"passflow__sender__in_then_out__current_to_last_in_amount_ratio__{suffix}",
                window=window,
                metric="current_to_last_in_amount_ratio",
                role="sender",
                description=(
                    f"Current outbound amount divided by the current sender's most recent inbound amount within {window}; "
                    "zero when the recent inbound amount is missing or zero."
                ),
                aml_typology="pass_through,amount_conservation",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            passflow_window_spec(
                f"passflow__sender__in_then_out__recent_flag__{suffix}",
                window=window,
                metric="recent_flag",
                role="sender",
                description=f"Whether the current sender has an earlier inbound transaction within {window}.",
                aml_typology="pass_through",
                computational_cost="low",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            passflow_window_spec(
                f"passflow__sender__in_then_out__sequence_count__{suffix}",
                window=window,
                metric="sequence_count",
                role="sender",
                description=f"Past inbound-then-outbound sequence count for the current sender over {window}.",
                aml_typology="pass_through",
                computational_cost="medium",
                used_in_ml=used_in_ml,
            )
        )


def _add_receiver_passflow_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for suffix, window in DEFAULT_RECEIVER_PASSFLOW_WINDOWS:
        specs.append(
            passflow_window_spec(
                f"passflow__receiver__in_then_out__historical_sequence_count__{suffix}",
                window=window,
                metric="historical_sequence_count",
                role="receiver",
                description=f"Past inbound-then-outbound sequence count for the current receiver over {window}.",
                aml_typology="pass_through",
                computational_cost="medium",
                used_in_ml=used_in_ml,
            )
        )


def ml05_stage4_feature_specs(*, used_in_ml: bool = True) -> Tuple[FeatureSpec, ...]:
    """Return the fixed ML-05 Stage 4 full48 feature specs."""

    specs: list[FeatureSpec] = []
    _add_flowbalance_specs(specs, used_in_ml=used_in_ml)
    _add_sender_passflow_specs(specs, used_in_ml=used_in_ml)
    _add_receiver_passflow_specs(specs, used_in_ml=used_in_ml)

    result = tuple(specs)
    validate_feature_specs(result)
    if len(result) != ML05_STAGE4_FEATURE_COUNT:
        raise ValueError(
            f"ML-05 full48 feature set must contain {ML05_STAGE4_FEATURE_COUNT} specs. observed={len(result)}"
        )
    return result
