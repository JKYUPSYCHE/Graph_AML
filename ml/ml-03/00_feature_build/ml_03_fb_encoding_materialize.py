"""Category and passthrough materialization helpers for ML-03 FB export.

Code map:
- Input: split_df columns and normalized EncodingSpec rows.
- Output: encoded Series plus category mapping/unknown summary rows.
- Public: passthrough_series, fit_train_categories, label_code_series, xgb_native_series.
- Leakage guard: category vocab is fitted from train split only.
- Notes: accountstats sum clamp only handles tiny ML-02 floating residuals.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from typing import Any, Mapping

import pandas as pd

from ml_03_fb_encoding_specs import EncodingSpec, UNKNOWN_CATEGORY
from ml_03_fb_schema import normalize_category_strict, parse_numeric_strict


CATEGORY_MAPPING_CSV_COLUMNS = (
    "feature_column",
    "source_column",
    "category_value",
    "encoded_value",
    "encoding",
    "fit_split",
)
CATEGORY_UNKNOWN_SUMMARY_CSV_COLUMNS = (
    "feature_column",
    "source_column",
    "encoding",
    "split",
    "unknown_count",
    "unknown_unique_count",
    "unknown_examples",
)
ACCOUNTSTATS_SUM_NEGATIVE_EPSILON = 1e-3
ACCOUNTSTATS_SUM_PREFIX = "accountstats__"
ACCOUNTSTATS_SUM_TOKEN = "__amount__sum__"


def categories_with_unknown(categories: list[str]) -> list[str]:
    """Append the unknown sentinel when it is not already part of the categories."""

    return categories if UNKNOWN_CATEGORY in categories else [*categories, UNKNOWN_CATEGORY]


def category_mapping_row(spec: EncodingSpec, category: str, encoded_value: int | None) -> dict[str, Any]:
    """Build one category mapping artifact row."""

    return {
        "feature_column": spec.output_column,
        "source_column": spec.source_column,
        "category_value": category,
        "encoded_value": encoded_value,
        "encoding": spec.encoding,
        "fit_split": "train",
    }


def is_accountstats_sum_feature(column: str) -> bool:
    """Return True for ML-02 accountstats amount sum carry-forward features."""

    return str(column).startswith(ACCOUNTSTATS_SUM_PREFIX) and ACCOUNTSTATS_SUM_TOKEN in str(column)


def clamp_accountstats_sum_near_zero_negative(series: pd.Series, spec: EncodingSpec) -> pd.Series:
    """Clamp tiny negative floating residuals in ML-02 accountstats sum features to zero."""

    if not is_accountstats_sum_feature(spec.output_column):
        return series

    negative_mask = series < 0
    if not bool(negative_mask.any()):
        return series

    too_negative_mask = series < -ACCOUNTSTATS_SUM_NEGATIVE_EPSILON
    if bool(too_negative_mask.any()):
        examples = series.loc[too_negative_mask].head(5).tolist()
        raise ValueError(
            "Encoding failed: accountstats sum feature contains negative values beyond floating residual tolerance. "
            f"output_column={spec.output_column!r}, "
            f"epsilon={ACCOUNTSTATS_SUM_NEGATIVE_EPSILON}, "
            f"negative_count={int(too_negative_mask.sum())}, "
            f"examples={examples}"
        )

    output = series.copy()
    output.loc[negative_mask] = 0.0
    return output


def numeric_passthrough_series(split_df: pd.DataFrame, spec: EncodingSpec) -> pd.Series:
    """Parse a passthrough column as numeric and surface encoding context on failure."""

    try:
        series = parse_numeric_strict(split_df, spec.source_column, spec.output_column)
    except ValueError as exc:
        raise ValueError(
            "Encoding failed: passthrough feature must be numeric because xgb_feature_type='q'. "
            f"source_column={spec.source_column!r}, output_column={spec.output_column!r}, encoding={spec.encoding!r}"
        ) from exc
    return clamp_accountstats_sum_near_zero_negative(series, spec)


def passthrough_series(split_df: pd.DataFrame, spec: EncodingSpec) -> pd.Series:
    """Materialize a passthrough feature with the configured XGBoost feature type."""

    if spec.xgb_feature_type == "q":
        return numeric_passthrough_series(split_df, spec)
    return split_df[spec.source_column]


def split_indexed_category(series: pd.Series, split_values: pd.Series, source_column: str) -> pd.Series:
    """Normalize categorical values and index them by split plus source row index."""

    normalized = normalize_category_strict(series, source_col=source_column)
    return pd.Series(
        normalized.to_numpy(),
        index=pd.MultiIndex.from_arrays([split_values.to_numpy(), normalized.index], names=["split", "row_index"]),
        dtype="string",
    )


def fit_train_categories(normalized: pd.Series, source_column: str) -> list[str]:
    """Fit category vocabulary on train split only."""

    train_values = normalized.loc["train"]
    categories = sorted(train_values.unique().tolist())
    if not categories:
        raise ValueError(f"train split has no category values. source_column={source_column!r}")
    return categories


def label_code_series(normalized: pd.Series, categories: list[str]) -> pd.Series:
    """Encode known categories as train-fitted integer codes and unknowns as -1."""

    mapping = {category: code for code, category in enumerate(categories)}
    return normalized.reset_index(level="split", drop=True).map(mapping).fillna(-1).astype("int32")


def xgb_native_series(normalized: pd.Series, categories: list[str]) -> pd.Categorical:
    """Encode categories as pandas Categorical for XGBoost native categorical input."""

    values = normalized.reset_index(level="split", drop=True)
    categories_with_unknown_value = categories_with_unknown(categories)
    mapped_values = values.where(values.isna() | values.isin(categories), UNKNOWN_CATEGORY)
    return pd.Categorical(mapped_values, categories=categories_with_unknown_value)


def unknown_rows(
    *,
    output_column: str,
    source_column: str,
    encoding: str,
    normalized: pd.Series,
    categories: list[str],
) -> list[dict[str, Any]]:
    """Summarize validation/test category values that were not fitted on train."""

    rows: list[dict[str, Any]] = []
    known = set(categories)
    for split_name in ("train", "val", "test"):
        split_mask = normalized.index.get_level_values("split") == split_name
        split_values = normalized.loc[split_mask]
        unknown_values = sorted(set(split_values.tolist()) - known)
        rows.append(
            {
                "feature_column": output_column,
                "source_column": source_column,
                "encoding": encoding,
                "split": split_name,
                "unknown_count": int((~split_values.isin(known)).sum()),
                "unknown_unique_count": int(len(unknown_values)),
                "unknown_examples": ";".join(str(value) for value in unknown_values[:5]),
            }
        )
    return rows


def normalize_input_category_values(input_category_values: Mapping[str, Any] | None) -> dict[str, list[str]]:
    """Normalize manual category values used by categorical passthrough features."""

    if input_category_values is None:
        return {}
    raw_values = input_category_values.get("category_values", input_category_values)
    if not isinstance(raw_values, MappingABC):
        raise ValueError("input_category_values must be a mapping or a payload with category_values mapping.")
    normalized: dict[str, list[str]] = {}
    for column, values in raw_values.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, IterableABC):
            raise ValueError(f"category values must be a non-string iterable. column={column!r}")
        categories = [str(value) for value in values]
        if not categories:
            raise ValueError(f"category values must not be empty. column={column!r}")
        normalized[str(column)] = categories
    return normalized
