"""ML-04 Stage 3 sender-receiver relationship FeatureSpec declarations.

This module declares feature contracts only. Actual computation is in
``ml_04_fb_operations.py``.

Code map:
- Input: none; specs are declared from fixed ML-04 Stage 3 catalog rules.
- Output: FeatureSpec tuples and generated feature column names.
- Public: FeatureSpec, FeatureOpResult, ml04_stage3_feature_specs, feature_columns.
- Leakage guard: every history spec uses timestamp < current_timestamp.
- Notes: full45 = pair relationship 44 + bank-pair corridor cumulative count 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Tuple

import pandas as pd


ML04_STAGE3_FEATURE_COUNT = 45

DEFAULT_PAIR_WINDOWS: Tuple[Tuple[str, str], ...] = (
    ("w1h", "1h"),
    ("w6h", "6h"),
    ("w1d", "1d"),
    ("w3d", "3d"),
    ("w7d", "7d"),
)

DEFAULT_STABLE_PAIR_WINDOWS: Tuple[Tuple[str, str], ...] = (
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


def _pair_input_cols() -> dict[str, str]:
    return {
        "sender_col": "sender_account_id",
        "receiver_col": "receiver_account_id",
        "timestamp_col": "timestamp",
        "amount_col": "amount",
    }


def pair_window_spec(
    output_col: str,
    *,
    window: str,
    metric: str,
    direction: str,
    family: str = "pair",
    description: str,
    aml_typology: str,
    computational_cost: str = "medium",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """Declare a pair window feature with strict past-only history."""

    return FeatureSpec(
        operation="pair_window",
        output_col=output_col,
        input_cols=_pair_input_cols(),
        params={"window": window, "metric": metric, "closed": "left", "fill_value": 0.0},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="directed_pair",
        direction=direction,
        leakage_policy="past-only; history_timestamp < current_timestamp; same-timestamp rows excluded",
        computational_cost=computational_cost,
        used_in_ml=used_in_ml,
    )


def pair_history_spec(
    output_col: str,
    *,
    metric: str,
    description: str,
    aml_typology: str,
    computational_cost: str = "medium",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """Declare an all-history pair recency/newness feature."""

    return FeatureSpec(
        operation="pair_history",
        output_col=output_col,
        input_cols=_pair_input_cols(),
        params={"metric": metric, "window": "whist", "fill_value": 0.0},
        family="pair",
        description=description,
        aml_typology=aml_typology,
        entity_scope="directed_pair",
        direction="forward",
        leakage_policy="past-only all-history; history_timestamp < current_timestamp; same-timestamp rows excluded",
        computational_cost=computational_cost,
        used_in_ml=used_in_ml,
    )


def current_self_loop_spec(*, used_in_ml: bool = True) -> FeatureSpec:
    """Declare the current-row self-loop flag."""

    return FeatureSpec(
        operation="current_self_loop",
        output_col="pair__sender_receiver__current__is_self_loop",
        input_cols={"sender_col": "sender_account_id", "receiver_col": "receiver_account_id"},
        params={"window": "wcur", "true_value": 1.0, "false_value": 0.0},
        family="pair",
        description="Current row sender and receiver are the same account.",
        aml_typology="self-loop",
        entity_scope="transaction_row",
        direction="current",
        leakage_policy="current-row-only; no history or future rows used",
        computational_cost="low",
        used_in_ml=used_in_ml,
    )


def bank_pair_cumulative_spec(*, used_in_ml: bool = True) -> FeatureSpec:
    """Declare the bank-pair corridor cumulative transaction count."""

    return FeatureSpec(
        operation="bank_pair_history",
        output_col="pairhist__bank_pair__all__tx_count__cum__whist",
        input_cols={
            "sender_bank_col": "sender_bank_id",
            "receiver_bank_col": "receiver_bank_id",
            "timestamp_col": "timestamp",
        },
        params={"metric": "cumulative_count", "window": "whist", "fill_value": 0.0},
        family="pairhist",
        description="Past cumulative count for the sender-bank -> receiver-bank corridor.",
        aml_typology="corridor,repeated_bank_pair",
        entity_scope="bank_pair",
        direction="forward",
        leakage_policy="past-only all-history; history_timestamp < current_timestamp; same-timestamp rows excluded",
        computational_cost="low",
        used_in_ml=used_in_ml,
    )


def _add_forward_pair_window_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for suffix, window in DEFAULT_PAIR_WINDOWS:
        specs.append(
            pair_window_spec(
                f"pair__sender_receiver__forward__tx_count__count__{suffix}",
                window=window,
                metric="forward_tx_count",
                direction="forward",
                description=f"Past same-direction sender->receiver transaction count over {window}.",
                aml_typology="repeated_pair,layering",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            pair_window_spec(
                f"pair__sender_receiver__forward__amount__sum__{suffix}",
                window=window,
                metric="forward_amount_sum",
                direction="forward",
                description=f"Past same-direction sender->receiver amount sum over {window}.",
                aml_typology="repeated_pair,layering",
                used_in_ml=used_in_ml,
            )
        )


def _add_forward_pair_amount_distribution_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for suffix, window in DEFAULT_STABLE_PAIR_WINDOWS:
        for metric, label in (
            ("forward_amount_mean", "mean"),
            ("forward_amount_std", "std"),
            ("forward_amount_max", "max"),
        ):
            specs.append(
                pair_window_spec(
                    f"pair__sender_receiver__forward__amount__{label}__{suffix}",
                    window=window,
                    metric=metric,
                    direction="forward",
                    description=f"Past same-direction sender->receiver amount {label} over {window}.",
                    aml_typology="repeated_pair,amount_anomaly",
                    used_in_ml=used_in_ml,
                )
            )


def _add_reverse_and_bidirectional_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    for suffix, window in DEFAULT_STABLE_PAIR_WINDOWS:
        specs.append(
            pair_window_spec(
                f"pair__sender_receiver__reverse__exists__{suffix}",
                window=window,
                metric="reverse_exists",
                direction="reverse",
                description=f"Whether receiver->sender reverse history exists over {window}.",
                aml_typology="reverse_pair,cycle",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            pair_window_spec(
                f"pair__sender_receiver__reverse__tx_count__count__{suffix}",
                window=window,
                metric="reverse_tx_count",
                direction="reverse",
                description=f"Past receiver->sender reverse transaction count over {window}.",
                aml_typology="reverse_pair,cycle",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            pair_window_spec(
                f"pair__sender_receiver__net__amount__sumdiff__{suffix}",
                window=window,
                metric="net_amount_sumdiff",
                direction="net",
                description=f"Forward amount sum minus reverse amount sum over {window}.",
                aml_typology="reverse_pair,net_flow",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            pair_window_spec(
                f"pair__sender_receiver__forward__amount__cur_vs_mean_ratio__{suffix}",
                window=window,
                metric="current_vs_forward_mean_ratio",
                direction="forward",
                description=f"Current amount divided by past same-direction pair mean over {window}; zero when denominator is missing or zero.",
                aml_typology="amount_anomaly,repeated_pair",
                used_in_ml=False,
            )
        )
        specs.append(
            pair_window_spec(
                f"pair__sender_receiver__forward__amount__cur_vs_mean_ratio__log1p_clip_p999_hist2__{suffix}",
                window=window,
                metric="current_vs_forward_mean_ratio_log1p_clip_p999_hist2",
                direction="forward",
                description=(
                    f"Log1p of current-vs-past pair mean ratio over {window}, "
                    "train p99.9 clipped and set to zero when history count is below 2."
                ),
                aml_typology="amount_anomaly,repeated_pair,stability_transform",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            pair_window_spec(
                f"pair__sender_receiver__forward__amount__cur_vs_mean_ratio__hist_lt2_flag__{suffix}",
                window=window,
                metric="current_vs_forward_mean_ratio_hist_lt2_flag",
                direction="forward",
                description=f"Whether past same-direction pair transaction count over {window} is below 2.",
                aml_typology="amount_anomaly,repeated_pair,history_sufficiency",
                computational_cost="low",
                used_in_ml=used_in_ml,
            )
        )
        specs.append(
            pair_window_spec(
                f"pair__sender_receiver__bidirectional__amount_ratio__{suffix}",
                window=window,
                metric="bidirectional_amount_ratio",
                direction="bidirectional",
                description=f"Forward amount sum divided by reverse amount sum over {window}; zero when denominator is missing or zero.",
                aml_typology="reverse_pair,cycle",
                used_in_ml=used_in_ml,
            )
        )


def _add_relationship_history_specs(specs: list[FeatureSpec], *, used_in_ml: bool) -> None:
    specs.append(
        pair_history_spec(
            "pair__sender_receiver__forward__seconds_since_last_tx",
            metric="seconds_since_last_tx",
            description="Seconds since the previous same-direction pair transaction; zero for new pairs.",
            aml_typology="repeated_pair,burst",
            used_in_ml=used_in_ml,
        )
    )
    specs.append(
        pair_history_spec(
            "pair__sender_receiver__forward__is_new_pair",
            metric="is_new_pair",
            description="Current sender->receiver pair has no earlier same-direction transaction.",
            aml_typology="new_relationship",
            computational_cost="low",
            used_in_ml=used_in_ml,
        )
    )
    specs.append(current_self_loop_spec(used_in_ml=used_in_ml))
    specs.append(
        pair_history_spec(
            "pair__sender_receiver__forward__age_hours_since_first",
            metric="age_hours_since_first",
            description="Hours since the first earlier same-direction pair transaction; zero for new pairs.",
            aml_typology="repeated_pair,relationship_age",
            used_in_ml=used_in_ml,
        )
    )


def ml04_stage3_feature_specs(*, used_in_ml: bool = True) -> Tuple[FeatureSpec, ...]:
    """Return the fixed ML-04 Stage 3 full45 feature specs."""

    specs: list[FeatureSpec] = []
    _add_forward_pair_window_specs(specs, used_in_ml=used_in_ml)
    _add_forward_pair_amount_distribution_specs(specs, used_in_ml=used_in_ml)
    _add_reverse_and_bidirectional_specs(specs, used_in_ml=used_in_ml)
    _add_relationship_history_specs(specs, used_in_ml=used_in_ml)
    specs.append(bank_pair_cumulative_spec(used_in_ml=used_in_ml))

    result = tuple(specs)
    validate_feature_specs(result)
    if len(result) != ML04_STAGE3_FEATURE_COUNT:
        raise ValueError(
            f"ML-04 full45 feature set must contain {ML04_STAGE3_FEATURE_COUNT} specs. observed={len(result)}"
        )
    return result
