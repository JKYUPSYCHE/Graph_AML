"""Feature contract helpers for ML-05 FB input files.

Code map:
- Input: FB input contract CSV and source parquet schema.
- Output: ContractValidationResult with selected ML columns.
- Public: validate_fb_input_contract, ContractValidationResult, CONTRACT_COLUMNS.
- Leakage guard: selected column names are checked before feature build/export.
- Notes: action/encoding normalization delegates to ml_05_fb_encoding_specs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml_05_fb_encoding_specs import (
    ENCODE_ENCODINGS,
    SUPPORTED_BUILD_ACTIONS,
    SUPPORTED_ENCODINGS,
    normalize_build_action_encoding,
)
from ml_05_fb_io import parquet_columns
from ml_05_fb_schema import validate_no_forbidden_feature_columns


ALLOWED_BOOL_VALUES = {"TRUE", "FALSE"}
ALLOWED_BUILD_ACTIONS = SUPPORTED_BUILD_ACTIONS
ALLOWED_ENCODINGS = SUPPORTED_ENCODINGS
SOURCE_BACKED_ACTIONS = {"carry_forward", "encode"}
CONTRACT_VERSION = 1

CONTRACT_COLUMNS = [
    "contract_version",
    "artifact_prefix",
    "column_name",
    "used_in_ml",
    "source_column",
    "column_origin",
    "encoding",
    "build_action",
    "build_in_fb",
    "xgb_feature_type",
    "feature_group",
    "dtype",
    "leakage_risk",
    "review_status",
    "excluded_reason",
    "selection_note",
    "contract_row_origin",
    "parent_artifact_prefix",
    "source_contract_path",
    "source_data_path",
    "feature_spec_name",
    "encoding_params",
    "materialized",
    "observed_dtype",
    "missing_count",
    "missing_rate",
    "unknown_category_count",
    "fit_split",
    "note",
]


@dataclass(frozen=True)
class ContractValidationResult:
    """Summary returned after contract validation."""

    path: Path
    total_rows: int
    selected_count: int
    selected_columns: list[str]


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found or not a file: {resolved}")
    return resolved


def _require_contract_columns(table: pd.DataFrame) -> None:
    missing = {"column_name", "used_in_ml"} - set(table.columns)
    if missing:
        raise ValueError(f"feature contract is missing columns: {sorted(missing)}")


def _require_strict_bool(series: pd.Series, column_name: str) -> pd.Series:
    if series.isna().any():
        rows = (series[series.isna()].index + 2).tolist()
        raise ValueError(f"{column_name} contains missing values. csv_rows={rows[:30]}")
    text = series.astype(str)
    invalid = text[~text.isin(ALLOWED_BOOL_VALUES)]
    if not invalid.empty:
        rows = (invalid.index + 2).tolist()
        raise ValueError(
            f"{column_name} contains unsupported values. "
            f"allowed_values={sorted(ALLOWED_BOOL_VALUES)}, "
            f"invalid_values={sorted(invalid.unique().tolist())[:30]}, csv_rows={rows[:30]}"
        )
    return text.astype("string")


def _cell_text(row: pd.Series, column: str) -> str:
    if column not in row.index or pd.isna(row[column]):
        return ""
    return str(row[column]).strip()


def _normalize_action_encoding_columns(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return action/encoding values using the same policy as the encoder."""

    actions: list[str] = []
    encodings: list[str] = []
    for index, row in table.iterrows():
        build_action = _cell_text(row, "build_action").lower()
        encoding = _cell_text(row, "encoding").lower()
        if not build_action:
            build_action = "encode" if encoding and encoding != "passthrough" else "carry_forward"
        build_action, encoding = normalize_build_action_encoding(build_action, encoding, csv_row=int(index) + 2)
        actions.append(build_action)
        encodings.append(encoding)
    return (
        pd.Series(actions, index=table.index, dtype="string"),
        pd.Series(encodings, index=table.index, dtype="string"),
    )


def _validate_action_encoding_pair(build_action: str, encoding: str, *, csv_row: int) -> None:
    """Validate action/encoding combinations before export."""

    if build_action != "encode":
        return
    if encoding not in ENCODE_ENCODINGS:
        raise ValueError(
            "encode rows must use a real category encoding. "
            f"allowed_encodings={sorted(ENCODE_ENCODINGS)}, encoding={encoding!r}, csv_row={csv_row}"
        )


def _materialized_source_column(row: pd.Series, build_action: str, table: pd.DataFrame) -> str:
    """Return the source column that encode_split_frame() will actually read."""

    if build_action == "carry_forward":
        return "" if pd.isna(row["column_name"]) else str(row["column_name"]).strip()
    if build_action == "encode" and "source_column" in table.columns:
        return "" if pd.isna(row["source_column"]) else str(row["source_column"]).strip()
    return ""


def _selected_columns(table: pd.DataFrame) -> list[str]:
    selected = table.loc[table["used_in_ml"] == "TRUE", "column_name"]
    columns = selected.astype(str).str.strip().tolist()
    blank_rows = [int(index) + 2 for index, column in zip(selected.index, columns) if not column]
    if blank_rows:
        raise ValueError(f"selected feature rows contain blank column_name. csv_rows={blank_rows[:30]}")
    duplicated = sorted({column for column in columns if columns.count(column) > 1})
    if duplicated:
        raise ValueError(f"duplicated selected feature columns: {duplicated}")
    validate_no_forbidden_feature_columns(columns)
    if not columns:
        raise ValueError("no selected feature columns found. At least one used_in_ml=TRUE row is required.")
    return columns


def validate_fb_input_contract(
    contract_path: str | Path,
    *,
    source_data_path: str | Path,
    artifact_prefix: str | None = None,
) -> ContractValidationResult:
    """Validate contract CSV syntax and source parquet compatibility."""

    path = _require_file(contract_path, "feature contract")
    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"used_in_ml": "string", "build_in_fb": "string", "materialized": "string"},
    )
    _require_contract_columns(table)
    table["used_in_ml"] = _require_strict_bool(table["used_in_ml"], "used_in_ml")

    if artifact_prefix is not None:
        if "artifact_prefix" not in table.columns:
            raise ValueError("feature contract is missing artifact_prefix column")
        mismatched = table[table["artifact_prefix"].astype(str) != artifact_prefix]
        if not mismatched.empty:
            rows = (mismatched.index + 2).tolist()
            raise ValueError(f"artifact_prefix mismatch. expected={artifact_prefix!r}, csv_rows={rows[:30]}")

    if "build_in_fb" in table.columns:
        _require_strict_bool(table["build_in_fb"], "build_in_fb")
    if "materialized" in table.columns:
        _require_strict_bool(table["materialized"], "materialized")
    table = table.copy()
    table["build_action"], table["encoding"] = _normalize_action_encoding_columns(table)

    all_columns = table["column_name"].astype(str).str.strip().tolist()
    duplicated = sorted({column for column in all_columns if all_columns.count(column) > 1})
    if duplicated:
        raise ValueError(f"duplicated contract column_name values: {duplicated}")

    selected = _selected_columns(table)
    source_columns = set(parquet_columns(_require_file(source_data_path, "source data parquet")))
    missing_sources: list[dict[str, str | int]] = []
    for index, row in table.iterrows():
        build_action = str(row["build_action"])
        encoding = str(row["encoding"])
        _validate_action_encoding_pair(build_action, encoding, csv_row=int(index) + 2)
        if build_action not in SOURCE_BACKED_ACTIONS:
            continue
        source_column = _materialized_source_column(row, build_action, table)
        if not source_column or source_column not in source_columns:
            missing_sources.append(
                {
                    "csv_row": int(index) + 2,
                    "build_action": build_action,
                    "source_column": source_column,
                }
            )
    if missing_sources:
        raise ValueError(
            "materialized carry_forward/encode source columns are missing from source parquet. "
            f"missing={missing_sources[:30]}, missing_count={len(missing_sources)}"
        )

    return ContractValidationResult(
        path=path,
        total_rows=int(len(table)),
        selected_count=int(len(selected)),
        selected_columns=selected,
    )
