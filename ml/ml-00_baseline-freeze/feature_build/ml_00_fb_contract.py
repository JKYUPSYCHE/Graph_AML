"""Feature contract helpers for FB input files."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from ml_00_fb_io import parquet_columns
from ml_00_fb_schema import validate_no_forbidden_feature_columns


ALLOWED_BOOL_VALUES = {"TRUE", "FALSE"}
ALLOWED_BUILD_ACTIONS = {"carry_forward", "build", "encode", "drop"}
ALLOWED_ENCODINGS = {"passthrough", "label_code", "xgb_native", "one_hot"}
CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ContractValidationResult:
    path: Path
    total_rows: int
    selected_count: int
    selected_columns: list[str]


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found or not a file: {resolved}")
    return resolved


def _guard_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")


def _normalize_bool_series(series: pd.Series, column_name: str) -> pd.Series:
    if series.isna().any():
        rows = (series[series.isna()].index + 2).tolist()
        raise ValueError(f"{column_name} contains missing values. csv_rows={rows[:30]}")

    mapping = {"TRUE": "TRUE", "FALSE": "FALSE", "True": "TRUE", "False": "FALSE", True: "TRUE", False: "FALSE"}
    normalized = series.map(mapping)
    invalid = series[normalized.isna()]
    if not invalid.empty:
        rows = (invalid.index + 2).tolist()
        raise ValueError(
            f"{column_name} contains unsupported values. "
            f"allowed_values={sorted(ALLOWED_BOOL_VALUES)}, "
            f"invalid_values={sorted(invalid.astype(str).unique().tolist())[:30]}, "
            f"csv_rows={rows[:30]}"
        )
    return normalized.astype("string")


def _require_contract_columns(table: pd.DataFrame) -> None:
    missing = {"column_name", "used_in_ml"} - set(table.columns)
    if missing:
        raise ValueError(f"feature contract is missing columns: {sorted(missing)}")


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


def copy_feature_columns_to_fb_inputs(
    source_path: str | Path,
    fb_input_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy source ml_feature_columns.csv to fb_inputs without changing the source file."""

    source = _require_file(source_path, "source feature columns")
    target_dir = Path(fb_input_dir).expanduser().resolve()
    target = target_dir / "ml_feature_columns.csv"
    _guard_output(target, overwrite=overwrite)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def prepare_fb_input_contract(
    feature_columns_path: str | Path,
    output_path: str | Path,
    *,
    artifact_prefix: str,
    source_data_path: str | Path,
    overwrite: bool = False,
) -> ContractValidationResult:
    """Create a normalized fb_inputs feature contract from a legacy feature CSV."""

    source = _require_file(feature_columns_path, "feature columns")
    output = Path(output_path).expanduser().resolve()
    _guard_output(output, overwrite=overwrite)

    table = pd.read_csv(source, encoding="utf-8-sig", dtype={"used_in_ml": "string"})
    _require_contract_columns(table)
    table = table.copy()
    table["used_in_ml"] = _normalize_bool_series(table["used_in_ml"], "used_in_ml")
    table = _with_contract_defaults(table, artifact_prefix=artifact_prefix, source_data_path=source_data_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    return validate_fb_input_contract(output, source_data_path=source_data_path, artifact_prefix=artifact_prefix)


def validate_fb_input_contract(
    contract_path: str | Path,
    *,
    source_data_path: str | Path,
    artifact_prefix: str | None = None,
) -> ContractValidationResult:
    """Validate FB input contract syntax and source parquet compatibility."""

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
    if "build_action" in table.columns:
        invalid_action = table[~table["build_action"].astype(str).isin(ALLOWED_BUILD_ACTIONS)]
        if not invalid_action.empty:
            rows = (invalid_action.index + 2).tolist()
            raise ValueError(f"unsupported build_action values. csv_rows={rows[:30]}")
    if "encoding" in table.columns:
        invalid_encoding = table[~table["encoding"].astype(str).isin(ALLOWED_ENCODINGS)]
        if not invalid_encoding.empty:
            rows = (invalid_encoding.index + 2).tolist()
            raise ValueError(f"unsupported encoding values. csv_rows={rows[:30]}")

    all_columns = table["column_name"].astype(str).str.strip().tolist()
    duplicated = sorted({column for column in all_columns if all_columns.count(column) > 1})
    if duplicated:
        raise ValueError(f"duplicated contract column_name values: {duplicated}")

    selected = _selected_columns(table)
    source_columns = set(parquet_columns(_require_file(source_data_path, "source data parquet")))
    missing_sources: list[dict[str, str | int]] = []
    for index, row in table.iterrows():
        if row["used_in_ml"] != "TRUE":
            continue
        build_action = str(row["build_action"]).strip().lower() if "build_action" in table.columns else "carry_forward"
        if build_action not in {"carry_forward", "encode"}:
            continue
        raw_source = row["source_column"] if "source_column" in table.columns else row["column_name"]
        source_column = "" if pd.isna(raw_source) else str(raw_source).strip()
        if not source_column or source_column not in source_columns:
            missing_sources.append({"csv_row": int(index) + 2, "source_column": source_column})
    if missing_sources:
        raise ValueError(
            "selected carry_forward/encode source columns are missing from source parquet. "
            f"missing={missing_sources[:30]}, missing_count={len(missing_sources)}"
        )

    return ContractValidationResult(
        path=path,
        total_rows=int(len(table)),
        selected_count=int(len(selected)),
        selected_columns=selected,
    )


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
            f"invalid_values={sorted(invalid.unique().tolist())[:30]}, "
            f"csv_rows={rows[:30]}"
        )
    return text.astype("string")


def _with_contract_defaults(table: pd.DataFrame, *, artifact_prefix: str, source_data_path: str | Path) -> pd.DataFrame:
    if not str(artifact_prefix).strip():
        raise ValueError("artifact_prefix must not be empty.")

    source_path = _require_file(source_data_path, "source data parquet")
    source_columns = set(parquet_columns(source_path))
    output = table.copy()
    output["contract_version"] = CONTRACT_VERSION
    output["artifact_prefix"] = str(artifact_prefix).strip()
    if "source_column" not in output.columns:
        output["source_column"] = output["column_name"]
    else:
        output["source_column"] = output["source_column"].fillna("").astype(str).str.strip()
        blank_source = output["source_column"] == ""
        output.loc[blank_source, "source_column"] = output.loc[blank_source, "column_name"].astype(str).str.strip()
    if "encoding" not in output.columns:
        output["encoding"] = "passthrough"
    output["encoding"] = output["encoding"].fillna("passthrough").astype(str).str.strip().str.lower()
    if "build_action" not in output.columns:
        output["build_action"] = "carry_forward"
    output["build_action"] = output["build_action"].fillna("carry_forward").astype(str).str.strip().str.lower()
    if "build_in_fb" not in output.columns:
        output["build_in_fb"] = "FALSE"
    else:
        output["build_in_fb"] = _normalize_bool_series(output["build_in_fb"], "build_in_fb")
    output["source_data_path"] = str(source_path)
    output["materialized"] = output["column_name"].astype(str).isin(source_columns).map({True: "TRUE", False: "FALSE"})
    if "contract_row_origin" not in output.columns:
        output["contract_row_origin"] = "new_in_current_run"
    if "review_status" not in output.columns:
        output["review_status"] = "pending"
    return output
