"""Public facade for ML train/validation/test I/O helpers.

The public import path stays `ml_io`; implementation details live in the
private `_ml_io_*` modules next to this file.
"""

from __future__ import annotations

from _ml_io_artifacts import file_sha256, load_json, save_json
from _ml_io_features import (
    FORBIDDEN_EXACT_NAMES,
    FORBIDDEN_SUBSTRINGS,
    UNKNOWN_CATEGORY,
    FeatureColumnsCheckResult,
    apply_encoding_manifest,
    categorical_columns_from_manifest,
    check_feature_columns_file,
    feature_columns_hash,
    load_encoding_manifest,
    load_feature_columns,
    load_saved_feature_columns,
    normalize_feature_columns_file,
    normalize_used_in_ml_values,
    parse_used_in_ml,
    save_feature_columns,
    used_in_ml_mask,
    validate_no_forbidden_features,
)
from _ml_io_inputs import (
    InputPaths,
    get_parquet_columns,
    label_summary,
    load_split,
    preflight_ml_inputs,
    print_input_paths,
    read_parquet_columns,
    require_input_files,
    resolve_project_path,
    validate_features,
    validate_labels,
    validate_parquet_split_values,
    validate_split_column,
)

__all__ = [
    "FORBIDDEN_EXACT_NAMES",
    "FORBIDDEN_SUBSTRINGS",
    "UNKNOWN_CATEGORY",
    "FeatureColumnsCheckResult",
    "InputPaths",
    "apply_encoding_manifest",
    "categorical_columns_from_manifest",
    "check_feature_columns_file",
    "feature_columns_hash",
    "file_sha256",
    "get_parquet_columns",
    "label_summary",
    "load_encoding_manifest",
    "load_feature_columns",
    "load_json",
    "load_saved_feature_columns",
    "load_split",
    "normalize_feature_columns_file",
    "normalize_used_in_ml_values",
    "parse_used_in_ml",
    "preflight_ml_inputs",
    "print_input_paths",
    "read_parquet_columns",
    "require_input_files",
    "resolve_project_path",
    "save_feature_columns",
    "save_json",
    "used_in_ml_mask",
    "validate_features",
    "validate_labels",
    "validate_no_forbidden_features",
    "validate_parquet_split_values",
    "validate_split_column",
]
