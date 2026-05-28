"""ML-03 Stage 2 fan-in/fan-out FeatureSpec declarations.

This module declares feature build contracts only. Actual computation is in
``ml_03_fb_operations.py`` and ``ml_03_fb_rolling.py``.

Code map:
- Input: none; specs are declared from fixed ML-03 Stage 2 catalog rules.
- Output: FeatureSpec tuples and generated feature column names.
- Public: FeatureSpec, FeatureOpResult, ml03_stage2_feature_specs, feature_columns.
- Leakage guard: every spec declares closed='left' and past-only policy text.
- Notes: full46 = fanin 15 + fanout 15 + top1 10 + bankfan 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Tuple

import pandas as pd

from ml_03_fb_schema import META_COLUMNS


# Full46 breakdown: fanin 15, fanout 15, top1_share 10, bankfan 6.
ML03_STAGE2_FEATURE_COUNT = 46


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


DEFAULT_FAN_WINDOWS: Tuple[Tuple[str, str], ...] = (
    ("w1h", "1h"),
    ("w6h", "6h"),
    ("w1d", "1d"),
    ("w3d", "3d"),
    ("w7d", "7d"),
)

DEFAULT_BANK_FAN_WINDOWS: Tuple[Tuple[str, str], ...] = (
    ("w1d", "1d"),
    ("w3d", "3d"),
    ("w7d", "7d"),
)


def rolling_agg_spec(
    entity_col: str,
    timestamp_col: str,
    value_col: str,
    output_col: str,
    *,
    window: str,
    agg: str,
    fill_value: float = 0.0,
    family: str = "fanin_fanout",
    description: str = "Past-only rolling aggregation by entity.",
    aml_typology: str = "fan-in/fan-out",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """Declare a past-only rolling aggregation feature."""

    return FeatureSpec(
        operation="rolling_agg",
        output_col=output_col,
        input_cols={"entity_col": entity_col, "timestamp_col": timestamp_col, "value_col": value_col},
        params={"window": window, "agg": agg, "closed": "left", "fill_value": fill_value},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="account",
        direction="past_window",
        leakage_policy="past-only; history_timestamp < current_timestamp; same-timestamp rows excluded",
        computational_cost="medium",
        used_in_ml=used_in_ml,
    )


def rolling_counterparty_nunique_spec(
    current_entity_col: str,
    history_entity_col: str,
    counterparty_col: str,
    timestamp_col: str,
    output_col: str,
    *,
    window: str,
    fill_value: float = 0.0,
    family: str,
    description: str,
    aml_typology: str,
    used_in_ml: bool = True,
) -> FeatureSpec:
    """Declare a unique counterparty count feature."""

    return FeatureSpec(
        operation="rolling_counterparty_nunique",
        output_col=output_col,
        input_cols={
            "current_entity_col": current_entity_col,
            "history_entity_col": history_entity_col,
            "counterparty_col": counterparty_col,
            "timestamp_col": timestamp_col,
        },
        params={"window": window, "closed": "left", "fill_value": fill_value},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="account",
        direction="past_window",
        leakage_policy="past-only; history_timestamp < current_timestamp; same-timestamp rows excluded",
        computational_cost="medium",
        used_in_ml=used_in_ml,
    )


def rolling_counterparty_effective_n_spec(
    current_entity_col: str,
    history_entity_col: str,
    counterparty_col: str,
    timestamp_col: str,
    value_col: str,
    output_col: str,
    *,
    window: str,
    fill_value: float = 0.0,
    family: str,
    description: str,
    aml_typology: str,
    used_in_ml: bool = True,
) -> FeatureSpec:
    """Declare an amount-weighted effective counterparty count feature."""

    return FeatureSpec(
        operation="rolling_counterparty_effective_n",
        output_col=output_col,
        input_cols={
            "current_entity_col": current_entity_col,
            "history_entity_col": history_entity_col,
            "counterparty_col": counterparty_col,
            "timestamp_col": timestamp_col,
            "value_col": value_col,
        },
        params={"window": window, "closed": "left", "fill_value": fill_value},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="account",
        direction="past_window",
        leakage_policy="past-only; effective_n uses only history_timestamp < current_timestamp",
        computational_cost="medium",
        used_in_ml=used_in_ml,
    )


def rolling_counterparty_top1_share_spec(
    current_entity_col: str,
    history_entity_col: str,
    counterparty_col: str,
    timestamp_col: str,
    value_col: str,
    output_col: str,
    *,
    window: str,
    fill_value: float = 0.0,
    family: str,
    description: str,
    aml_typology: str,
    used_in_ml: bool = True,
) -> FeatureSpec:
    """Declare a top-1 counterparty amount share feature."""

    return FeatureSpec(
        operation="rolling_counterparty_top1_share",
        output_col=output_col,
        input_cols={
            "current_entity_col": current_entity_col,
            "history_entity_col": history_entity_col,
            "counterparty_col": counterparty_col,
            "timestamp_col": timestamp_col,
            "value_col": value_col,
        },
        params={"window": window, "closed": "left", "fill_value": fill_value},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="account",
        direction="past_window",
        leakage_policy="past-only; top1_share uses only history_timestamp < current_timestamp",
        computational_cost="medium",
        used_in_ml=used_in_ml,
    )


def _add_fanin_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for suffix, window in DEFAULT_FAN_WINDOWS:
        suffix = _clean_name(suffix, "window suffix")
        window = _clean_name(window, "window")
        specs.append(
            rolling_counterparty_nunique_spec(
                "receiver_account_id",
                "receiver_account_id",
                "sender_account_id",
                "timestamp",
                f"fanin__receiver__in__counterparty__nunique__{suffix}",
                window=window,
                family="fanin",
                description=f"Receiver past inbound unique sender count over {window}.",
                aml_typology="fan-in,gather",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            rolling_agg_spec(
                "receiver_account_id",
                "timestamp",
                "amount_received",
                f"fanin__receiver__in__tx_count__degree__{suffix}",
                window=window,
                agg="count",
                family="fanin",
                description=f"Receiver past inbound transaction count over {window}.",
                aml_typology="fan-in,gather",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            rolling_counterparty_effective_n_spec(
                "receiver_account_id",
                "receiver_account_id",
                "sender_account_id",
                "timestamp",
                "amount_received",
                f"fanin__receiver__in__counterparty_amount__effective_n__{suffix}",
                window=window,
                family="fanin",
                description=f"Receiver past inbound amount-weighted effective sender count over {window}.",
                aml_typology="fan-in,gather,concentration",
                used_in_ml=used_in_ml,
            )
        )


def _add_fanout_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for suffix, window in DEFAULT_FAN_WINDOWS:
        suffix = _clean_name(suffix, "window suffix")
        window = _clean_name(window, "window")
        specs.append(
            rolling_counterparty_nunique_spec(
                "sender_account_id",
                "sender_account_id",
                "receiver_account_id",
                "timestamp",
                f"fanout__sender__out__counterparty__nunique__{suffix}",
                window=window,
                family="fanout",
                description=f"Sender past outbound unique receiver count over {window}.",
                aml_typology="fan-out,scatter",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            rolling_agg_spec(
                "sender_account_id",
                "timestamp",
                "amount",
                f"fanout__sender__out__tx_count__degree__{suffix}",
                window=window,
                agg="count",
                family="fanout",
                description=f"Sender past outbound transaction count over {window}.",
                aml_typology="fan-out,scatter",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            rolling_counterparty_effective_n_spec(
                "sender_account_id",
                "sender_account_id",
                "receiver_account_id",
                "timestamp",
                "amount",
                f"fanout__sender__out__counterparty_amount__effective_n__{suffix}",
                window=window,
                family="fanout",
                description=f"Sender past outbound amount-weighted effective receiver count over {window}.",
                aml_typology="fan-out,scatter,concentration",
                used_in_ml=used_in_ml,
            )
        )


def _add_top1_share_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for suffix, window in DEFAULT_FAN_WINDOWS:
        specs.append(
            rolling_counterparty_top1_share_spec(
                "receiver_account_id",
                "receiver_account_id",
                "sender_account_id",
                "timestamp",
                "amount_received",
                f"fanin__receiver__in__counterparty_amount__top1_share__{suffix}",
                window=window,
                family="fanin",
                description=f"Receiver past inbound largest sender amount share over {window}.",
                aml_typology="fan-in,concentration",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            rolling_counterparty_top1_share_spec(
                "sender_account_id",
                "sender_account_id",
                "receiver_account_id",
                "timestamp",
                "amount",
                f"fanout__sender__out__counterparty_amount__top1_share__{suffix}",
                window=window,
                family="fanout",
                description=f"Sender past outbound largest receiver amount share over {window}.",
                aml_typology="fan-out,concentration",
                used_in_ml=used_in_ml,
            )
        )


def _add_bankfan_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for suffix, window in DEFAULT_BANK_FAN_WINDOWS:
        specs.append(
            rolling_counterparty_nunique_spec(
                "sender_account_id",
                "sender_account_id",
                "receiver_bank_id",
                "timestamp",
                f"bankfan__sender__out__to_bank__nunique__{suffix}",
                window=window,
                family="bankfan",
                description=f"Sender past outbound unique receiving bank count over {window}.",
                aml_typology="fan-out,corridor",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            rolling_counterparty_nunique_spec(
                "receiver_account_id",
                "receiver_account_id",
                "sender_bank_id",
                "timestamp",
                f"bankfan__receiver__in__from_bank__nunique__{suffix}",
                window=window,
                family="bankfan",
                description=f"Receiver past inbound unique sending bank count over {window}.",
                aml_typology="fan-in,corridor",
                used_in_ml=used_in_ml,
            )
        )


def ml03_stage2_feature_specs(*, used_in_ml: bool = True) -> Tuple[FeatureSpec, ...]:
    """Return the fixed ML-03 Stage 2 full46 feature specs."""

    specs: list[FeatureSpec] = []
    _add_fanin_specs(specs, used_in_ml=used_in_ml)
    _add_fanout_specs(specs, used_in_ml=used_in_ml)
    _add_top1_share_specs(specs, used_in_ml=used_in_ml)
    _add_bankfan_specs(specs, used_in_ml=used_in_ml)
    result = tuple(specs)
    validate_feature_specs(result)
    if len(result) != ML03_STAGE2_FEATURE_COUNT:
        raise ValueError(
            f"ML-03 full46 feature set must contain {ML03_STAGE2_FEATURE_COUNT} specs. observed={len(result)}"
        )
    return result