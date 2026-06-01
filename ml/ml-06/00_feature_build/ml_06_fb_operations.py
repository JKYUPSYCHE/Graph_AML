"""ML-06 recency and ratio preprocessing operations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_06_fb_schema import numeric_values, validate_required_columns
from ml_06_fb_specs import (
    AUDIT_RECENCY_COLUMNS,
    FIRST_FLAG_COLUMNS,
    RATIO_QUANTILE,
    TARGET_RECENCY_COLUMNS,
    RatioTransformSpec,
    discover_base_ratio_columns,
    ratio_transform_spec,
)


RATIO_COMPARE_RTOL = 1e-6
RATIO_COMPARE_ATOL = 1e-7


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    finite_values = values[finite]
    if len(finite_values) == 0:
        return {
            "finite_count": 0,
            "nan_or_inf_count": int((~finite).sum()),
            "min": np.nan,
            "max": np.nan,
            "negative_count": 0,
            "minus1_count": 0,
            "zero_count": 0,
        }
    return {
        "finite_count": int(finite.sum()),
        "nan_or_inf_count": int((~finite).sum()),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "negative_count": int((finite_values < 0).sum()),
        "minus1_count": int((finite_values == -1.0).sum()),
        "zero_count": int((finite_values == 0.0).sum()),
    }


def _value_examples(
    *,
    indexes: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in indexes[:limit]:
        row: dict[str, Any] = {
            "row_index": int(index),
            "actual": float(actual[index]),
        }
        if expected is not None:
            row["expected"] = float(expected[index])
        rows.append(row)
    return rows


def _validate_first_flag_values(df: pd.DataFrame, column: str) -> np.ndarray:
    values = numeric_values(df, column, context="first flag validation")
    invalid = ~np.isin(values, [0.0, 1.0])
    if bool(invalid.any()):
        indexes = np.flatnonzero(invalid)
        raise ValueError(
            "first flag column must contain only 0/1 values. "
            f"flag_column={column!r}, invalid_count={int(invalid.sum())}, "
            f"examples={_value_examples(indexes=indexes, actual=values)}"
        )
    return values


def _validate_recency_first_flag_alignment(
    df: pd.DataFrame,
    *,
    recency_column: str,
    flag_column: str,
) -> None:
    recency_values = numeric_values(df, recency_column, context="recency first flag validation")
    invalid_recency = recency_values < -1.0
    if bool(invalid_recency.any()):
        indexes = np.flatnonzero(invalid_recency)
        raise ValueError(
            "target recency has negative values other than -1 sentinel. "
            f"recency_column={recency_column!r}, invalid_count={int(invalid_recency.sum())}, "
            f"examples={_value_examples(indexes=indexes, actual=recency_values)}"
        )

    flag_values = _validate_first_flag_values(df, flag_column)
    sentinel_mask = recency_values == -1.0
    first_flag_mask = flag_values == 1.0
    mismatch = sentinel_mask != first_flag_mask
    if bool(mismatch.any()):
        indexes = np.flatnonzero(mismatch)
        examples = [
            {
                "row_index": int(index),
                "recency": float(recency_values[index]),
                "first_flag": float(flag_values[index]),
            }
            for index in indexes[:5]
        ]
        raise ValueError(
            "recency -1 sentinel mask must match first flag 1 mask before correction. "
            f"recency_column={recency_column!r}, flag_column={flag_column!r}, "
            f"mismatch_count={int(mismatch.sum())}, examples={examples}"
        )


def _validate_target_recency_inputs(df: pd.DataFrame) -> None:
    for recency_column, flag_column in zip(TARGET_RECENCY_COLUMNS, FIRST_FLAG_COLUMNS):
        _validate_recency_first_flag_alignment(
            df,
            recency_column=recency_column,
            flag_column=flag_column,
        )


def apply_recency_sentinel_policy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert only the two target recency ``-1`` sentinels to ``0``."""

    validate_required_columns(df, list(TARGET_RECENCY_COLUMNS), context="recency sentinel policy")
    missing_flags = [column for column in FIRST_FLAG_COLUMNS if column not in df.columns]
    if missing_flags:
        raise ValueError(f"first flag columns are required to preserve first-transaction meaning: {missing_flags}")

    output = df.copy()
    _validate_target_recency_inputs(output)

    rows: list[dict[str, Any]] = []
    for column in AUDIT_RECENCY_COLUMNS:
        if column not in output.columns:
            rows.append(
                {
                    "column_name": column,
                    "policy": "missing_audit_column",
                    "changed": "FALSE",
                    "before_minus1_count": np.nan,
                    "after_minus1_count": np.nan,
                    "changed_count": 0,
                    "note": "audit column not present in input",
                }
            )
            continue

        before = numeric_values(output, column, context="recency audit")
        before_summary = _numeric_summary(before)
        changed_mask = np.zeros(len(output), dtype=bool)
        policy = "audit_only_no_value_change"
        note = ""

        if column in TARGET_RECENCY_COLUMNS:
            changed_mask = before == -1.0
            corrected = before.copy()
            corrected[changed_mask] = 0.0
            output[column] = corrected.astype(output[column].dtype, copy=False)
            policy = "minus1_to_zero"
            note = "first transaction meaning preserved by existing first flag columns"
        elif before_summary["negative_count"]:
            note = "negative values observed in audit-only recency-like column; value was not changed"

        after = numeric_values(output, column, context="recency audit after")
        after_summary = _numeric_summary(after)
        rows.append(
            {
                "column_name": column,
                "policy": policy,
                "changed": "TRUE" if bool(changed_mask.any()) else "FALSE",
                "row_count": int(len(output)),
                "changed_count": int(changed_mask.sum()),
                "before_min": before_summary["min"],
                "before_max": before_summary["max"],
                "before_negative_count": before_summary["negative_count"],
                "before_minus1_count": before_summary["minus1_count"],
                "before_zero_count": before_summary["zero_count"],
                "after_min": after_summary["min"],
                "after_max": after_summary["max"],
                "after_negative_count": after_summary["negative_count"],
                "after_minus1_count": after_summary["minus1_count"],
                "after_zero_count": after_summary["zero_count"],
                "note": note,
            }
        )
    return output, pd.DataFrame(rows)


def _split_count_dict(split: pd.Series, mask: np.ndarray) -> dict[str, int]:
    values: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        values[split_name] = int((mask & (split.to_numpy() == split_name)).sum())
    return values


def _validate_ratio_values(values: np.ndarray, *, column: str) -> None:
    negative = values < 0.0
    if bool(negative.any()):
        examples = values[negative][:5].tolist()
        raise ValueError(
            "ratio transform requires non-negative finite values. "
            f"column={column!r}, negative_count={int(negative.sum())}, examples={examples}"
        )


def _fit_train_quantile(values: np.ndarray, split: pd.Series, *, column: str, quantile: float) -> tuple[float, int]:
    train_mask = split.to_numpy() == "train"
    train_values = values[train_mask]
    if len(train_values) == 0:
        raise ValueError(f"ratio clipping cannot fit without train rows: column={column!r}")
    cap = float(np.quantile(train_values, quantile))
    if not np.isfinite(cap):
        raise ValueError(f"ratio clipping fitted an invalid cap: column={column!r}, cap={cap!r}")
    return cap, int(len(train_values))


def _validate_existing_ratio_output(
    *,
    actual: np.ndarray,
    expected: np.ndarray,
    base_column: str,
    output_column: str,
    transform: str,
) -> None:
    matches = np.isclose(
        actual,
        expected.astype("float64", copy=False),
        rtol=RATIO_COMPARE_RTOL,
        atol=RATIO_COMPARE_ATOL,
        equal_nan=False,
    )
    mismatch = ~matches
    if bool(mismatch.any()):
        indexes = np.flatnonzero(mismatch)
        raise ValueError(
            "existing ratio transform column does not match expected values. "
            f"column={output_column!r}, base_column={base_column!r}, transform={transform!r}, "
            f"mismatch_count={int(mismatch.sum())}, "
            f"examples={_value_examples(indexes=indexes, actual=actual, expected=expected)}"
        )


def apply_ratio_transforms(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    base_ratio_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, list[RatioTransformSpec]]:
    """Create pure log1p and train-p99.99 clipping columns for base ratios."""

    validate_required_columns(df, ["split"], context="ratio transform")
    split = df["split"].astype("string").str.strip().str.lower()
    output = df.copy()
    base_columns = (
        list(dict.fromkeys(base_ratio_columns))
        if base_ratio_columns is not None
        else discover_base_ratio_columns(feature_columns, output.columns)
    )
    validate_required_columns(output, base_columns, context="ratio transform base columns")
    specs = [ratio_transform_spec(column) for column in base_columns]

    transform_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    for spec in specs:
        values = numeric_values(output, spec.base_column, context="ratio transform")
        _validate_ratio_values(values, column=spec.base_column)

        expected_log_values = np.log1p(values).astype("float32")
        log_action = "reused_existing" if spec.log1p_column in output.columns else "created"
        if log_action == "created":
            output[spec.log1p_column] = expected_log_values
        log_values = numeric_values(output, spec.log1p_column, context="ratio log1p output")
        if log_action == "reused_existing":
            _validate_existing_ratio_output(
                actual=log_values,
                expected=expected_log_values,
                base_column=spec.base_column,
                output_column=spec.log1p_column,
                transform="log1p",
            )
        reuse_rows.append(
            {
                "base_column": spec.base_column,
                "transform": "log1p",
                "output_column": spec.log1p_column,
                "action": log_action,
                "reason": (
                    "existing output column passed value equality validation"
                    if log_action == "reused_existing"
                    else "created pure log1p output"
                ),
            }
        )
        transform_rows.append(
            {
                "base_column": spec.base_column,
                "transform": "log1p",
                "output_column": spec.log1p_column,
                "action": log_action,
                "input_max": float(values.max()),
                "output_max": float(log_values.max()),
                "train_fit_rows": None,
                "train_cap": None,
                "clipped_count": None,
                "clipped_count_by_split": None,
            }
        )

        clip_action = "reused_existing" if spec.clip_column in output.columns else "created"
        cap, train_fit_rows = _fit_train_quantile(values, split, column=spec.base_column, quantile=RATIO_QUANTILE)
        clipped_mask = values > cap
        expected_clip_values = np.minimum(values, cap).astype("float32")
        if clip_action == "created":
            output[spec.clip_column] = expected_clip_values
        clip_values = numeric_values(output, spec.clip_column, context="ratio clip output")
        if clip_action == "reused_existing":
            _validate_existing_ratio_output(
                actual=clip_values,
                expected=expected_clip_values,
                base_column=spec.base_column,
                output_column=spec.clip_column,
                transform="clip_train_p9999",
            )
        reuse_rows.append(
            {
                "base_column": spec.base_column,
                "transform": "clip_train_p9999",
                "output_column": spec.clip_column,
                "action": clip_action,
                "reason": (
                    "existing output column passed value equality validation"
                    if clip_action == "reused_existing"
                    else "created train p99.99 clipped output"
                ),
            }
        )
        transform_rows.append(
            {
                "base_column": spec.base_column,
                "transform": "clip_train_p9999",
                "output_column": spec.clip_column,
                "action": clip_action,
                "input_max": float(values.max()),
                "output_max": float(clip_values.max()),
                "train_fit_rows": train_fit_rows,
                "train_cap": cap,
                "clipped_count": int(clipped_mask.sum()),
                "clipped_count_by_split": _split_count_dict(split, clipped_mask),
            }
        )

    manifest = {
        "ratio_quantile": RATIO_QUANTILE,
        "base_ratio_columns": base_columns,
        "base_ratio_count": len(base_columns),
        "transforms": transform_rows,
        "skip_policy": (
            "reuse existing pure transform outputs only after value equality validation; "
            "existing composite transforms are not treated as pure log1p or pure clip"
        ),
    }
    return output, manifest, pd.DataFrame(reuse_rows), specs
