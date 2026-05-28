"""Schema resolution and strict parsing for ML-04 feature build.

Code map:
- Input: raw/source DataFrame columns and user-provided column maps.
- Output: canonical metadata columns, parsed Series, and leakage-name validation.
- Public: resolve_requested_columns, standardize_input_frame, strict parsers, forbidden-name validators.
- Leakage guard: blocks label/typology-like names from feature inputs and outputs.
- Notes: META_COLUMNS is the shared build/export metadata contract.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional

import pandas as pd


META_COLUMNS = ("tx_id", "timestamp", "split", "label")

COLUMN_CANDIDATES: dict[str, list[str]] = {
    "tx_id": ["tx_id", "transaction_id", "Transaction ID", "Transaction_ID", "TransactionID", "transaction_id_raw"],
    "timestamp": ["timestamp", "Timestamp", "transaction_time", "Transaction Time", "Transaction_Time", "time", "Time", "datetime", "DateTime"],
    "sender_bank_id": ["sender_bank_id", "from_bank", "From Bank", "From_Bank", "sender_bank", "source_bank"],
    "receiver_bank_id": ["receiver_bank_id", "to_bank", "To Bank", "To_Bank", "receiver_bank", "destination_bank"],
    "sender_account_id": ["sender_account_id", "sender_account", "from_account_id", "from_account", "From Account", "From_Account", "Account", "orig_account", "source_account"],
    "receiver_account_id": ["receiver_account_id", "receiver_account", "to_account_id", "to_account", "To Account", "To_Account", "Account.1", "Account_1", "beneficiary_account", "dest_account", "destination_account"],
    "amount": ["amount", "Amount", "amount_paid", "Amount Paid", "Amount_Paid", "paid_amount"],
    "amount_received": ["amount_received", "Amount Received", "Amount_Received", "received_amount", "recv_amount"],
    "label": ["is_laundering", "label", "Is Laundering", "Is_Laundering", "target", "y"],
    "payment_currency": ["payment_currency", "Payment Currency", "Payment_Currency", "currency", "Currency", "amount_currency"],
    "receiving_currency": ["receiving_currency", "Receiving Currency", "Receiving_Currency", "received_currency"],
    "payment_format": ["payment_format", "Payment Format", "Payment_Format", "payment_type", "Payment Type", "format"],
}

FORBIDDEN_EXACT_NAMES = {"label", "target", "y", "is_laundering", "is laundering"}
FORBIDDEN_SUBSTRINGS = {"laundering", "pattern", "typology", "attempt"}


def _example_values(series: pd.Series, mask: pd.Series, limit: int = 5) -> list[str]:
    return series.loc[mask].astype(str).head(limit).tolist()


def parse_datetime_series_strict(
    raw: pd.Series,
    *,
    missing_message: Callable[[int], str],
    failed_message: Callable[[int, list[str]], str],
) -> pd.Series:
    """Parse datetimes with caller-owned error text."""

    missing_mask = raw.isna()
    if missing_mask.any():
        raise ValueError(missing_message(int(missing_mask.sum())))
    parsed = pd.to_datetime(raw, errors="coerce")
    failed_mask = parsed.isna()
    if failed_mask.any():
        raise ValueError(failed_message(int(failed_mask.sum()), _example_values(raw, failed_mask)))
    return parsed


def parse_numeric_series_strict(
    raw: pd.Series,
    *,
    missing_message: Callable[[int], str],
    failed_message: Callable[[int, list[str]], str],
) -> pd.Series:
    """Parse numeric values with caller-owned error text."""

    missing_mask = raw.isna()
    if missing_mask.any():
        raise ValueError(missing_message(int(missing_mask.sum())))
    parsed = pd.to_numeric(raw, errors="coerce")
    failed_mask = parsed.isna()
    if failed_mask.any():
        raise ValueError(failed_message(int(failed_mask.sum()), _example_values(raw, failed_mask)))
    return parsed


def resolve_requested_column(
    available_columns: set[str],
    requested_col: str,
    column_map: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve a logical column name to an input source column."""

    requested_col = str(requested_col).strip()
    if column_map is not None and requested_col in column_map:
        mapped_col = str(column_map[requested_col]).strip()
        if not mapped_col:
            raise ValueError(f"column_map has blank source column. requested_col={requested_col!r}")
        if mapped_col not in available_columns:
            raise ValueError(
                "Feature build failed: column_map points to a missing source column. "
                f"requested_col={requested_col!r}, mapped_col={mapped_col!r}, "
                f"available_columns={sorted(available_columns)[:80]}"
            )
        return mapped_col

    if requested_col in available_columns:
        return requested_col

    candidates = COLUMN_CANDIDATES.get(requested_col, [])
    for candidate in candidates:
        if candidate in available_columns:
            return candidate

    raise ValueError(
        "Feature build failed: required input column is missing. "
        f"requested_col={requested_col!r}, candidates={candidates}, available_count={len(available_columns)}, "
        "fix=Check input schema or edit FeatureSpec.input_cols."
    )


def resolve_requested_columns(
    columns: Iterable[str],
    requested_columns: Iterable[str],
    column_map: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Resolve multiple logical names while preserving requested order."""

    available_columns = set(str(column) for column in columns)
    requested = list(dict.fromkeys(str(column).strip() for column in requested_columns))
    if not requested:
        raise ValueError("requested_columns must not be empty.")
    return {column: resolve_requested_column(available_columns, column, column_map=column_map) for column in requested}


def parse_datetime_strict(df: pd.DataFrame, source_col: str, logical_name: str) -> pd.Series:
    """Parse datetimes and fail on missing or invalid values."""

    raw = df[source_col]
    return parse_datetime_series_strict(
        raw,
        missing_message=lambda missing_count: (
            "Feature build failed: timestamp column has missing values. "
            f"logical_name={logical_name!r}, source_column={source_col!r}, missing_count={missing_count}"
        ),
        failed_message=lambda failed_count, examples: (
            "Feature build failed: timestamp parsing failed. "
            f"logical_name={logical_name!r}, source_column={source_col!r}, failed_count={failed_count}, "
            f"example_values={examples}"
        ),
    )


def parse_numeric_strict(df: pd.DataFrame, source_col: str, logical_name: str) -> pd.Series:
    """Parse numeric values and fail on missing or invalid values."""

    raw = df[source_col]
    return parse_numeric_series_strict(
        raw,
        missing_message=lambda missing_count: (
            "Feature build failed: numeric column has missing values. "
            f"logical_name={logical_name!r}, source_column={source_col!r}, missing_count={missing_count}"
        ),
        failed_message=lambda failed_count, examples: (
            "Feature build failed: numeric parsing failed. "
            f"logical_name={logical_name!r}, source_column={source_col!r}, failed_count={failed_count}, "
            f"example_values={examples}"
        ),
    )


def normalize_category_strict(series: pd.Series, source_col: str) -> pd.Series:
    """Normalize categorical values as non-empty strings."""

    missing_mask = series.isna()
    if missing_mask.any():
        raise ValueError(
            "Feature build failed: category column has missing values. "
            f"source_column={source_col!r}, missing_count={int(missing_mask.sum())}"
        )
    normalized = series.astype("string").str.strip()
    blank_mask = normalized == ""
    if blank_mask.any():
        raise ValueError(
            "Feature build failed: category column has blank values. "
            f"source_column={source_col!r}, blank_count={int(blank_mask.sum())}, "
            f"example_values={_example_values(series, blank_mask)}"
        )
    return normalized


def standardize_input_frame(
    df: pd.DataFrame,
    column_map: Mapping[str, str],
    *,
    tx_id_col: str,
    timestamp_col: str,
    label_col: str,
) -> pd.DataFrame:
    """Create canonical feature-build columns from a raw input frame."""

    standardized = pd.DataFrame(index=df.index)
    for requested_col, source_col in column_map.items():
        standardized[requested_col] = df[source_col]

    required_meta = {tx_id_col, timestamp_col, label_col}
    missing_meta = required_meta - set(standardized.columns)
    if missing_meta:
        raise ValueError(
            "Feature build failed: canonical metadata source columns were not loaded. "
            f"missing_meta={sorted(missing_meta)}"
        )

    tx_id = standardized[tx_id_col]
    if tx_id.isna().any():
        raise ValueError(
            "Feature build failed: tx_id column has missing values. "
            f"source_column={tx_id_col!r}, missing_count={int(tx_id.isna().sum())}"
        )
    duplicated_tx_id = tx_id.astype("string").duplicated(keep=False)
    if duplicated_tx_id.any():
        examples = tx_id.loc[duplicated_tx_id].astype(str).head(10).tolist()
        raise ValueError(
            "Feature build failed: tx_id column has duplicated values. "
            f"source_column={tx_id_col!r}, duplicated_count={int(duplicated_tx_id.sum())}, examples={examples}"
        )

    standardized["tx_id"] = tx_id
    standardized["timestamp"] = parse_datetime_strict(standardized, timestamp_col, "timestamp")
    label = parse_numeric_strict(standardized, label_col, "label")
    label_values = sorted(label.dropna().unique().tolist())
    if not set(label_values).issubset({0, 1}):
        raise ValueError("Feature build failed: label must be binary 0/1. " f"observed_values={label_values[:20]}")
    standardized["label"] = label.astype("int8")
    if standardized.index.equals(pd.RangeIndex(len(standardized))):
        return standardized
    return standardized.reset_index(drop=True)


def validate_no_forbidden_feature_columns(feature_columns: Iterable[str]) -> None:
    """Reject generated feature names that look like labels or typologies."""

    forbidden_exact = {name.lower() for name in FORBIDDEN_EXACT_NAMES}
    violations: list[str] = []
    for column in feature_columns:
        normalized = str(column).strip().lower()
        if normalized in forbidden_exact:
            violations.append(str(column))
        elif any(pattern in normalized for pattern in FORBIDDEN_SUBSTRINGS):
            violations.append(str(column))
    if violations:
        raise ValueError(
            "Data leakage risk: forbidden feature column names were selected. "
            f"violations={violations[:30]}, violation_count={len(violations)}"
        )


def validate_no_forbidden_input_columns(input_columns: Iterable[str]) -> None:
    """Reject FeatureSpec input columns that look like label-derived fields."""

    forbidden_exact = {name.lower() for name in FORBIDDEN_EXACT_NAMES}
    violations: list[str] = []
    for column in input_columns:
        normalized = str(column).strip().lower()
        if normalized in forbidden_exact:
            violations.append(str(column))
        elif any(pattern in normalized for pattern in FORBIDDEN_SUBSTRINGS):
            violations.append(str(column))
    if violations:
        raise ValueError(
            "Data leakage risk: forbidden input columns were used by feature_specs. "
            f"violations={violations[:30]}, violation_count={len(violations)}"
        )
