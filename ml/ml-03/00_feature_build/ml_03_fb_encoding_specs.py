"""Encoding spec parsing and normalization for ML-03 FB export.

Code map:
- Input: FB input contract CSV rows or in-memory EncodingSpec objects.
- Output: normalized EncodingSpec rows used by encode_split_frame().
- Public: EncodingSpec, load_encoding_specs, normalize_encoding_specs, policy constants.
- Leakage guard: validates selected and materialized non-meta feature names.
- Notes: this module is the source of truth for build_action/encoding policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union

import pandas as pd

from ml_03_fb_schema import META_COLUMNS, validate_no_forbidden_feature_columns, validate_no_forbidden_input_columns


SUPPORTED_ENCODINGS = {"passthrough", "label_code", "xgb_native"}
SUPPORTED_BUILD_ACTIONS = {"carry_forward", "build", "encode"}
ENCODE_ENCODINGS = {"label_code", "xgb_native"}
PASSTHROUGH_BUILD_ACTIONS = {"build", "carry_forward"}
UNKNOWN_CATEGORY = "__UNKNOWN__"


@dataclass(frozen=True)
class EncodingSpec:
    """One materialized output column contract."""

    source_column: str
    output_column: str
    encoding: str
    used_in_ml: bool = True
    build_action: str = "carry_forward"
    xgb_feature_type: str = ""


def parse_used_in_ml_value(value: Any) -> bool:
    """Parse common boolean spellings from CSV cells."""

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unsupported used_in_ml value: {value!r}")


def normalize_xgb_feature_type(
    value: Any,
    *,
    encoding: str,
    used_in_ml: bool,
    csv_row: int | None = None,
) -> str:
    """Normalize XGBoost feature type and infer defaults from encoding."""

    observed = "" if pd.isna(value) else str(value).strip().lower()
    if observed:
        if observed not in {"q", "c"}:
            row_context = "" if csv_row is None else f" csv_row={csv_row},"
            raise ValueError(f"Unsupported xgb_feature_type.{row_context} xgb_feature_type={observed!r}, supported=['c', 'q']")
        return observed
    if not used_in_ml:
        return ""
    if encoding == "xgb_native":
        return "c"
    if encoding in {"label_code", "passthrough"}:
        return "q"
    return ""


def _csv_row_context(csv_row: int | None) -> str:
    return "" if csv_row is None else f" csv_row={csv_row},"


def normalize_build_action_encoding(
    build_action: str,
    encoding: str,
    *,
    csv_row: int | None = None,
) -> tuple[str, str]:
    """Normalize build/carry rows and reject contradictory encode rows."""

    if build_action not in SUPPORTED_BUILD_ACTIONS:
        row_context = _csv_row_context(csv_row)
        raise ValueError(
            f"Unsupported build_action.{row_context} "
            f"build_action={build_action!r}, supported={sorted(SUPPORTED_BUILD_ACTIONS)}"
        )
    if build_action in PASSTHROUGH_BUILD_ACTIONS:
        return build_action, "passthrough"
    if encoding not in ENCODE_ENCODINGS:
        row_context = _csv_row_context(csv_row)
        raise ValueError(
            "Encode row must use a real category encoding. "
            f"{row_context}encoding={encoding!r}, supported={sorted(ENCODE_ENCODINGS)}"
        )
    return build_action, encoding


def _duplicated_names(names: list[str]) -> list[str]:
    return sorted({name for name in names if names.count(name) > 1})


def load_encoding_specs(path: Union[str, Path]) -> list[EncodingSpec]:
    """Load EncodingSpec rows from an encoding CSV or FB contract CSV."""

    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"encoding spec file not found: {spec_path}")
    table = pd.read_csv(spec_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})
    output_column_field = "output_column" if "output_column" in table.columns else "column_name"
    required = {output_column_field, "encoding", "used_in_ml"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"encoding spec CSV is missing columns: {sorted(missing)}")

    specs: list[EncodingSpec] = []
    for row_number, row in table.iterrows():
        used_in_ml = parse_used_in_ml_value(row["used_in_ml"])
        output_column = "" if pd.isna(row[output_column_field]) else str(row[output_column_field]).strip()
        raw_source_column = row["source_column"] if "source_column" in table.columns else ""
        source_column = "" if pd.isna(raw_source_column) else str(raw_source_column).strip()
        encoding = "" if pd.isna(row["encoding"]) else str(row["encoding"]).strip().lower()
        raw_build_action = row["build_action"] if "build_action" in table.columns else ""
        build_action = "" if pd.isna(raw_build_action) else str(raw_build_action).strip().lower()
        if not build_action:
            build_action = "encode" if encoding and encoding != "passthrough" else "carry_forward"
        if build_action not in SUPPORTED_BUILD_ACTIONS:
            raise ValueError(f"Unsupported build_action. csv_row={row_number + 2}, build_action={build_action!r}")
        if "build_in_fb" in table.columns:
            build_in_fb = parse_used_in_ml_value(row["build_in_fb"])
            if build_action in {"build", "encode"} and not build_in_fb:
                raise ValueError(f"Rows with build_action='build' or 'encode' must have build_in_fb=TRUE. csv_row={row_number + 2}")
        build_action, encoding = normalize_build_action_encoding(
            build_action,
            encoding,
            csv_row=row_number + 2,
        )
        if build_action in PASSTHROUGH_BUILD_ACTIONS:
            source_column = output_column
        if not output_column:
            raise ValueError(f"Encoding spec has blank output column. csv_row={row_number + 2}")
        if not source_column:
            raise ValueError(f"Encoding spec has blank source column. csv_row={row_number + 2}")
        raw_xgb_feature_type = row["xgb_feature_type"] if "xgb_feature_type" in table.columns else ""
        specs.append(
            EncodingSpec(
                source_column=source_column,
                output_column=output_column,
                encoding=encoding,
                used_in_ml=used_in_ml,
                build_action=build_action,
                xgb_feature_type=normalize_xgb_feature_type(
                    raw_xgb_feature_type,
                    encoding=encoding,
                    used_in_ml=used_in_ml,
                    csv_row=row_number + 2,
                ),
            )
        )
    return specs


def normalize_encoding_specs(specs: Iterable[EncodingSpec]) -> list[EncodingSpec]:
    """Normalize in-memory specs with the same rules as CSV-loaded specs."""

    normalized: list[EncodingSpec] = []
    for spec in specs:
        source_column = str(spec.source_column).strip()
        output_column = str(spec.output_column).strip()
        encoding = str(spec.encoding).strip().lower()
        build_action = str(spec.build_action).strip().lower()
        if not build_action:
            build_action = "encode" if encoding and encoding != "passthrough" else "carry_forward"
        build_action, encoding = normalize_build_action_encoding(build_action, encoding)
        if build_action in PASSTHROUGH_BUILD_ACTIONS:
            source_column = output_column
        normalized_spec = EncodingSpec(
            source_column=source_column,
            output_column=output_column,
            encoding=encoding,
            used_in_ml=spec.used_in_ml,
            build_action=build_action,
            xgb_feature_type=normalize_xgb_feature_type(
                spec.xgb_feature_type,
                encoding=encoding,
                used_in_ml=spec.used_in_ml,
            ),
        )
        normalized.append(normalized_spec)
    if not normalized:
        raise ValueError("No encoding specs. At least one contract row is required.")
    if not [spec for spec in normalized if spec.used_in_ml]:
        raise ValueError("No selected encoding specs. At least one used_in_ml=True row is required.")
    for spec in normalized:
        if spec.build_action not in SUPPORTED_BUILD_ACTIONS:
            raise ValueError(f"Unsupported build_action: {spec.build_action!r}")
        if spec.encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(f"Unsupported encoding: {spec.encoding!r}")
        if not spec.source_column or not spec.output_column:
            raise ValueError(
                "Encoding specs must have non-empty source/output columns. "
                f"source_column={spec.source_column!r}, output_column={spec.output_column!r}"
            )
    duplicated = _duplicated_names([spec.output_column for spec in normalized])
    if duplicated:
        raise ValueError(f"Duplicated output columns in encoding specs: {duplicated}")
    selected_specs = [spec for spec in normalized if spec.used_in_ml]

    validate_no_forbidden_input_columns(spec.source_column for spec in selected_specs)
    validate_no_forbidden_feature_columns(spec.output_column for spec in selected_specs)
    return normalized
