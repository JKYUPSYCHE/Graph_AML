"""Feature catalog and split summary builders for ML-04.

Code map:
- Input: FeatureSpec tuple or split-aware feature frame.
- Output: feature catalog, ML feature column table, and split summary DataFrames.
- Public: make_feature_catalog, make_feature_columns_table, make_split_summary.
- Leakage guard: catalog records past-only rule and same-timestamp exclusion check.
- Notes: empty train/val/test summaries fail instead of silently reporting zero rows.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Tuple

import pandas as pd

from ml_04_fb_schema import validate_no_forbidden_feature_columns
from ml_04_fb_specs import FeatureSpec, feature_columns, validate_feature_specs


FEATURE_SET_LABEL = "full45_pair_ratio_stability"


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)


def _window_from_spec(spec: FeatureSpec) -> str:
    return str(spec.params.get("window", "")).strip()


def _selection_status(spec: FeatureSpec) -> str:
    if "amount__cur_vs_mean_ratio__log1p_clip_p999_hist2" in spec.output_col:
        return f"candidate_{FEATURE_SET_LABEL};train_p999_clipped;history_lt2_set_to_0"
    if "amount__cur_vs_mean_ratio__hist_lt2_flag" in spec.output_col:
        return f"candidate_{FEATURE_SET_LABEL};history_sufficiency_flag"
    if "forward__amount__cur_vs_mean_ratio__w" in spec.output_col:
        return f"candidate_{FEATURE_SET_LABEL};raw_ratio_materialized_for_audit;excluded_from_ml"
    if "bidirectional__amount_ratio" in spec.output_col:
        return f"candidate_{FEATURE_SET_LABEL};denominator_zero_handled_as_0"
    return f"candidate_{FEATURE_SET_LABEL}"


def make_feature_columns_table(feature_specs: Tuple[FeatureSpec, ...]) -> pd.DataFrame:
    """Create the ML feature list table consumed by training code."""

    validate_feature_specs(feature_specs)
    columns = feature_columns(feature_specs)
    validate_no_forbidden_feature_columns(columns)
    return pd.DataFrame(
        {
            "column_name": columns,
            "used_in_ml": ["TRUE" if spec.used_in_ml else "FALSE" for spec in feature_specs],
        }
    )


def make_feature_catalog(
    feature_specs: Tuple[FeatureSpec, ...],
    *,
    experiment_id: str,
    run_name: str,
) -> pd.DataFrame:
    """Convert selected FeatureSpecs to a reviewable ML-04 feature catalog."""

    validate_feature_specs(feature_specs)
    columns = feature_columns(feature_specs)
    validate_no_forbidden_feature_columns(columns)
    rows: list[dict[str, Any]] = []
    for spec in feature_specs:
        rows.append(
            {
                "feature_name": spec.output_col,
                "column_name": spec.output_col,
                "experiment_id": experiment_id,
                "run_name": run_name,
                "stage": "Stage 3",
                "operation": spec.operation,
                "feature_family": spec.family,
                "entity_scope": spec.entity_scope,
                "direction": spec.direction,
                "window": _window_from_spec(spec),
                "input_columns": _json_dumps(spec.input_cols),
                "params": _json_dumps(spec.params),
                "description": spec.description,
                "aml_typology": spec.aml_typology,
                "leakage_policy": spec.leakage_policy,
                "leakage_rule": "Use only rows with history_timestamp < current_timestamp; lower window bound inclusive for window features.",
                "leakage_check": "Pair operations emit each timestamp group before adding same-timestamp rows to history.",
                "computational_cost": spec.computational_cost,
                "selection_status": _selection_status(spec),
                "used_in_ml": "TRUE" if spec.used_in_ml else "FALSE",
            }
        )
    catalog = pd.DataFrame(rows)
    observed = catalog["column_name"].tolist()
    if observed != columns:
        raise ValueError(f"Feature catalog column order mismatch. observed={observed}, expected={columns}")

    required_non_empty = ["stage", "window", "leakage_rule", "leakage_check", "aml_typology", "selection_status"]
    for column in required_non_empty:
        blank = catalog[column].astype("string").str.strip().fillna("") == ""
        if bool(blank.any()):
            raise ValueError(
                "Feature catalog failed: required metadata column contains blank values. "
                f"column={column!r}, blank_rows={(catalog.index[blank] + 2).tolist()[:30]}"
            )
    return catalog


def make_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize row counts, periods, and labels by split."""

    required = {"split", "timestamp", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split summary input is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        if split_df.empty:
            raise ValueError(f"split summary input has no rows for required split: {split_name}")
        label_counts = split_df["label"].astype(int).value_counts().to_dict()
        rows.append(
            {
                "split": split_name,
                "rows": int(len(split_df)),
                "timestamp_min": split_df["timestamp"].min(),
                "timestamp_max": split_df["timestamp"].max(),
                "label_0_count": int(label_counts.get(0, 0)),
                "label_1_count": int(label_counts.get(1, 0)),
                "positive_rate": float(split_df["label"].mean()) if len(split_df) else 0.0,
            }
        )
    return pd.DataFrame(rows)
