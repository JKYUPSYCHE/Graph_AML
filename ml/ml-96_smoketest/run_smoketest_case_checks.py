"""Run ML smoketest fixture contract and bad-case checks.

This script validates only synthetic local fixtures under ml/ml-96_smoketest.
It does not train models and must not be used for performance claims.

Examples:
    python ml/ml-96_smoketest/run_smoketest_case_checks.py
    python ml/ml-96_smoketest/run_smoketest_case_checks.py --io-module ml-00
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parents[1]


def load_ml_io_module(name: str) -> ModuleType:
    """Load the requested ml_io module by file path because experiment dirs use hyphens."""

    module_paths = {
        "ml-00": PROJECT_ROOT / "ml" / "ml-00_baseline-freeze" / "train_val_test" / "ml_00_ml_io.py",
        "ml-01": PROJECT_ROOT / "ml" / "ml-01" / "01_train_val_test" / "ml_01_ml_io.py",
    }
    module_path = module_paths[name]
    if not module_path.is_file():
        raise FileNotFoundError(f"ml_io module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"{name.replace('-', '_')}_ml_io", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ml_io module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expect_error(name: str, fn: Callable[[], object], expected_message: str) -> None:
    """Assert that a bad case fails with an error message containing expected_message."""

    try:
        fn()
    except Exception as exc:
        message = str(exc)
        if expected_message not in message:
            raise AssertionError(
                f"{name}: expected message containing {expected_message!r}, got {message!r}"
            ) from exc
        first_line = message.splitlines()[0] if message else ""
        print(f"[EXPECTED ERROR] {name}: {type(exc).__name__}: {first_line}")
        return
    raise AssertionError(f"{name}: expected failure but succeeded.")


def run_checks(io_module: ModuleType, smoke_dir: Path, label_col: str) -> None:
    """Run all fixture checks against the selected ml_io module."""

    feature_dir = smoke_dir / "ml_features"
    contract_dir = smoke_dir / "contract_cases"
    bad_dir = smoke_dir / "bad_cases"
    feature_columns_path = feature_dir / "ml_feature_columns_smoketest.csv"
    feature_columns = io_module.load_feature_columns(feature_columns_path, label_col=label_col)

    contract_cases = {
        "shuffled_columns": contract_dir / "val_shuffled_columns_smoketest_contract_cases.parquet",
        "extra_column": contract_dir / "val_extra_column_smoketest_contract_cases.parquet",
        "project_feature_names": contract_dir / "val_project_feature_names_smoketest_contract_cases.parquet",
    }
    for name, path in contract_cases.items():
        x, y = io_module.load_split(path, feature_columns, label_col=label_col, expected_split="val")
        if list(x.columns) != feature_columns:
            raise AssertionError(f"{name}: feature order mismatch.")
        print(f"[CONTRACT PASS] {name}: x_shape={x.shape}, y_rows={len(y)}")

    load_split_cases = [
        ("missing_feature", bad_dir / "val_missing_feature_smoketest_bad_cases.parquet", "missing required columns"),
        (
            "missing_multiple_features",
            bad_dir / "val_missing_multiple_features_smoketest_bad_cases.parquet",
            "missing required columns",
        ),
        ("missing_label", bad_dir / "val_missing_label_smoketest_bad_cases.parquet", "missing required columns"),
        ("nan_label", bad_dir / "val_nan_label_smoketest_bad_cases.parquet", "Labels contain NaN"),
        (
            "non_binary_label",
            bad_dir / "val_non_binary_label_smoketest_bad_cases.parquet",
            "Labels must be binary 0/1",
        ),
        ("string_label", bad_dir / "val_string_label_smoketest_bad_cases.parquet", "Unable to parse"),
        ("wrong_split", bad_dir / "val_wrong_split_smoketest_bad_cases.parquet", "Unexpected split values"),
        (
            "missing_split",
            bad_dir / "val_missing_split_smoketest_bad_cases.parquet",
            "missing required split column",
        ),
        ("null_split", bad_dir / "val_null_split_smoketest_bad_cases.parquet", "Split column contains missing"),
        ("blank_split", bad_dir / "val_blank_split_smoketest_bad_cases.parquet", "Split column contains blank"),
        ("mixed_split", bad_dir / "val_mixed_split_smoketest_bad_cases.parquet", "Unexpected split values"),
        ("nan_feature", bad_dir / "val_nan_feature_smoketest_bad_cases.parquet", "NaN values"),
        ("wrong_dtype", bad_dir / "val_wrong_dtype_smoketest_bad_cases.parquet", "All features must be numeric"),
        ("null_explosion", bad_dir / "val_null_explosion_smoketest_bad_cases.parquet", "NaN values"),
        ("inf_feature", bad_dir / "val_inf_feature_smoketest_bad_cases.parquet", "infinite values"),
        (
            "negative_inf_feature",
            bad_dir / "val_negative_inf_feature_smoketest_bad_cases.parquet",
            "infinite values",
        ),
        ("empty_rows", bad_dir / "val_empty_rows_smoketest_bad_cases.parquet", "Unexpected split values"),
        ("single_class", bad_dir / "val_single_class_smoketest_bad_cases.parquet", "Both classes are required"),
    ]

    for name, path, expected_message in load_split_cases:
        expect_error(
            name,
            lambda path=path: io_module.load_split(
                path,
                feature_columns=feature_columns,
                label_col=label_col,
                expected_split="val",
            ),
            expected_message,
        )

    catalog_cases = [
        ("label_leak_catalog", bad_dir / "val_label_leak_catalog_smoketest_bad_cases.csv", "Data leakage risk"),
        (
            "duplicate_feature_catalog",
            bad_dir / "val_duplicate_feature_catalog_smoketest_bad_cases.csv",
            "Duplicated selected feature columns",
        ),
        ("empty_feature_catalog", bad_dir / "val_empty_feature_catalog_smoketest_bad_cases.csv", "No usable feature columns"),
        (
            "invalid_used_in_ml_catalog",
            bad_dir / "val_invalid_used_in_ml_catalog_smoketest_bad_cases.csv",
            "unsupported values",
        ),
        ("forbidden_name_catalog", bad_dir / "val_forbidden_name_catalog_smoketest_bad_cases.csv", "Data leakage risk"),
        ("target_name_catalog", bad_dir / "val_target_name_catalog_smoketest_bad_cases.csv", "Data leakage risk"),
        ("y_name_catalog", bad_dir / "val_y_name_catalog_smoketest_bad_cases.csv", "Data leakage risk"),
        ("pattern_name_catalog", bad_dir / "val_pattern_name_catalog_smoketest_bad_cases.csv", "Data leakage risk"),
        (
            "missing_column_name_catalog",
            bad_dir / "val_missing_column_name_catalog_smoketest_bad_cases.csv",
            "missing columns",
        ),
        (
            "missing_used_in_ml_catalog",
            bad_dir / "val_missing_used_in_ml_catalog_smoketest_bad_cases.csv",
            "missing columns",
        ),
        (
            "blank_selected_feature_catalog",
            bad_dir / "val_blank_selected_feature_catalog_smoketest_bad_cases.csv",
            "blank column_name",
        ),
        (
            "missing_selected_feature_name_catalog",
            bad_dir / "val_missing_selected_feature_name_catalog_smoketest_bad_cases.csv",
            "missing column_name",
        ),
    ]

    for name, path, expected_message in catalog_cases:
        expect_error(
            name,
            lambda path=path: io_module.load_feature_columns(path, label_col=label_col),
            expected_message,
        )

    print("[SMOKETEST CASE CHECKS PASS]")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run ML smoketest fixture contract/bad-case checks.")
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=ROOT_DIR,
        help="Directory containing ml_features/, contract_cases/, bad_cases/.",
    )
    parser.add_argument(
        "--io-module",
        choices=("ml-00", "ml-01"),
        default="ml-01",
        help="ml_io implementation to validate against.",
    )
    parser.add_argument("--label-col", default="label", help="Target label column name.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    smoke_dir = args.smoke_dir.expanduser().resolve()
    io_module = load_ml_io_module(args.io_module)
    print("smoke_dir:", smoke_dir)
    print("io_module:", args.io_module)
    run_checks(io_module=io_module, smoke_dir=smoke_dir, label_col=args.label_col)


if __name__ == "__main__":
    main()
