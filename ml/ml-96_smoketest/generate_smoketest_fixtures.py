"""Generate local smoketest fixtures for the ML-00 pipeline.

The generated files are synthetic and are intended only for schema/contract
checks. They are not suitable for model performance claims.

Examples:
    python ml/ml-96_smoketest/generate_smoketest_fixtures.py --dry-run
    python ml/ml-96_smoketest/generate_smoketest_fixtures.py --out-dir /tmp/ml-smoke --overwrite
    python ml/ml-96_smoketest/generate_smoketest_fixtures.py --overwrite
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 42
FEATURE_COUNT = 12
TRAIN_ROWS = 1_200
VAL_ROWS = 400
TEST_ROWS = 400
POSITIVE_RATIO = 0.05

ROOT_DIR = Path(__file__).resolve().parent
FEATURE_DIR_NAME = "ml_features"
CONTRACT_DIR_NAME = "contract_cases"
BAD_DIR_NAME = "bad_cases"


@dataclass(frozen=True)
class GeneratedFile:
    """A fixture file planned or written by this script."""

    path: Path
    rows: int | None
    description: str


def resolve_output_dir(path: str | Path) -> Path:
    """Resolve output dir independently of the current working directory."""

    out_dir = Path(path).expanduser()
    if out_dir.is_absolute():
        return out_dir.resolve()
    return (Path.cwd() / out_dir).resolve()


def feature_columns() -> list[str]:
    """Return the synthetic feature names used by ML-00 smoketest fixtures."""

    return [f"feature_{idx:02d}" for idx in range(FEATURE_COUNT)]


def make_labels(row_count: int, rng: np.random.Generator) -> np.ndarray:
    """Create binary labels with both classes present and deterministic shuffle."""

    positive_count = max(1, int(round(row_count * POSITIVE_RATIO)))
    if positive_count >= row_count:
        raise ValueError(f"row_count must exceed positive_count. row_count={row_count}")

    labels = np.zeros(row_count, dtype=np.int8)
    labels[:positive_count] = 1
    rng.shuffle(labels)
    return labels


def make_split(split: str, row_count: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build one synthetic train/val/test split with numeric features and labels."""

    labels = make_labels(row_count, rng)
    columns = feature_columns()

    values = rng.normal(loc=0.0, scale=1.0, size=(row_count, len(columns))).astype("float32")
    label_signal = labels.astype("float32")
    values[:, 0] += label_signal * 2.0
    values[:, 1] += label_signal * 1.5
    values[:, 2] -= label_signal * 1.0

    frame = pd.DataFrame(values, columns=columns)
    frame["amount"] = np.round(
        rng.lognormal(mean=6.0 + label_signal * 0.3, sigma=0.5, size=row_count), 2
    )
    frame["tx_id"] = [f"{split.upper()}_{idx:06d}" for idx in range(row_count)]
    frame["split"] = split
    frame["label"] = pd.Series(labels, dtype="int8")
    return frame


def base_catalog_rows() -> list[dict[str, object]]:
    """Return common feature catalog rows before bad-case mutation."""

    rows: list[dict[str, object]] = []
    for column in feature_columns():
        rows.append(
            {
                "column_name": column,
                "used_in_ml": True,
                "stage": "SMOKETEST",
                "feature_group": "synthetic_numeric",
                "description": "numeric smoke-test feature",
            }
        )

    rows.extend(
        [
            {
                "column_name": "amount",
                "used_in_ml": False,
                "stage": "SMOKETEST",
                "feature_group": "metadata",
                "description": "extra numeric column not used by model",
            },
            {
                "column_name": "tx_id",
                "used_in_ml": False,
                "stage": "SMOKETEST",
                "feature_group": "metadata",
                "description": "transaction id not used by model",
            },
            {
                "column_name": "split",
                "used_in_ml": False,
                "stage": "SMOKETEST",
                "feature_group": "metadata",
                "description": "split marker for validation",
            },
            {
                "column_name": "label",
                "used_in_ml": False,
                "stage": "SMOKETEST",
                "feature_group": "target",
                "description": "target label; true only in bad leak catalog",
            },
        ]
    )
    return rows


def make_catalog(
    *,
    label_leak: bool = False,
    duplicate_feature: bool = False,
    empty_features: bool = False,
    invalid_used_in_ml: bool = False,
    forbidden_name: bool = False,
) -> pd.DataFrame:
    """Create valid or intentionally invalid feature catalog variants."""

    rows = base_catalog_rows()

    if empty_features:
        for row in rows:
            if str(row["column_name"]).startswith("feature_"):
                row["used_in_ml"] = False

    if invalid_used_in_ml:
        rows[0]["used_in_ml"] = "maybe"

    if label_leak:
        for row in rows:
            if row["column_name"] == "label":
                row["used_in_ml"] = True
                break

    if duplicate_feature:
        rows.insert(
            FEATURE_COUNT,
            {
                "column_name": "feature_01",
                "used_in_ml": True,
                "stage": "SMOKETEST",
                "feature_group": "synthetic_numeric",
                "description": "intentional duplicate feature for bad-case validation",
            },
        )

    if forbidden_name:
        rows.insert(
            FEATURE_COUNT,
            {
                "column_name": "future_laundering_score",
                "used_in_ml": True,
                "stage": "SMOKETEST",
                "feature_group": "bad_case",
                "description": "intentional forbidden feature name for leakage guard validation",
            },
        )

    return pd.DataFrame(rows)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a parquet fixture file after creating the parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def write_catalog(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV catalog with UTF-8 BOM for spreadsheet compatibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def planned_files(out_dir: Path) -> list[GeneratedFile]:
    """Return all files this script writes."""

    ml_features = out_dir / FEATURE_DIR_NAME
    contract_cases = out_dir / CONTRACT_DIR_NAME
    bad_cases = out_dir / BAD_DIR_NAME

    return [
        GeneratedFile(ml_features / "ml_exp00_Xy_train_smoketest.parquet", TRAIN_ROWS, "base train split"),
        GeneratedFile(ml_features / "ml_exp00_Xy_val_smoketest.parquet", VAL_ROWS, "base validation split"),
        GeneratedFile(ml_features / "ml_exp00_Xy_test_smoketest.parquet", TEST_ROWS, "base test split"),
        GeneratedFile(ml_features / "ml_feature_columns_smoketest.csv", None, "base feature catalog"),
        GeneratedFile(
            contract_cases / "val_shuffled_columns_smoketest_contract_cases.parquet",
            VAL_ROWS,
            "contract case: shuffled physical columns",
        ),
        GeneratedFile(
            contract_cases / "val_extra_column_smoketest_contract_cases.parquet",
            VAL_ROWS,
            "contract case: extra unused column",
        ),
        GeneratedFile(
            bad_cases / "val_missing_feature_smoketest_bad_cases.parquet",
            VAL_ROWS,
            "bad case: required feature missing",
        ),
        GeneratedFile(
            bad_cases / "val_missing_label_smoketest_bad_cases.parquet",
            VAL_ROWS,
            "bad case: label missing",
        ),
        GeneratedFile(
            bad_cases / "val_wrong_split_smoketest_bad_cases.parquet",
            VAL_ROWS,
            "bad case: split value mismatch",
        ),
        GeneratedFile(
            bad_cases / "val_nan_feature_smoketest_bad_cases.parquet",
            VAL_ROWS,
            "bad case: NaN feature values",
        ),
        GeneratedFile(
            bad_cases / "val_wrong_dtype_smoketest_bad_cases.parquet",
            VAL_ROWS,
            "bad case: non-numeric feature dtype",
        ),
        GeneratedFile(
            bad_cases / "val_null_explosion_smoketest_bad_cases.parquet",
            VAL_ROWS,
            "bad case: many null feature values",
        ),
        GeneratedFile(
            bad_cases / "val_inf_feature_smoketest_bad_cases.parquet",
            VAL_ROWS,
            "bad case: infinite feature values",
        ),
        GeneratedFile(
            bad_cases / "val_single_class_smoketest_bad_cases.parquet",
            VAL_ROWS,
            "bad case: single-class labels",
        ),
        GeneratedFile(
            bad_cases / "val_label_leak_catalog_smoketest_bad_cases.csv",
            None,
            "bad case catalog: label selected as feature",
        ),
        GeneratedFile(
            bad_cases / "val_duplicate_feature_catalog_smoketest_bad_cases.csv",
            None,
            "bad case catalog: duplicated selected feature",
        ),
        GeneratedFile(
            bad_cases / "val_empty_feature_catalog_smoketest_bad_cases.csv",
            None,
            "bad case catalog: no selected features",
        ),
        GeneratedFile(
            bad_cases / "val_invalid_used_in_ml_catalog_smoketest_bad_cases.csv",
            None,
            "bad case catalog: invalid used_in_ml value",
        ),
        GeneratedFile(
            bad_cases / "val_forbidden_name_catalog_smoketest_bad_cases.csv",
            None,
            "bad case catalog: forbidden leakage-like feature name",
        ),
    ]


def ensure_can_write(files: list[GeneratedFile], overwrite: bool) -> None:
    """Fail before writing anything if existing files would be overwritten."""

    if overwrite:
        return

    existing = [item.path for item in files if item.path.exists()]
    if existing:
        preview = "\n".join(f"  - {path}" for path in existing[:30])
        raise FileExistsError(
            "Refusing to overwrite existing smoketest fixture files. "
            "Pass --overwrite only when intentional. Existing files:\n"
            f"{preview}"
        )


def generate_fixtures(out_dir: Path, seed: int, overwrite: bool) -> list[GeneratedFile]:
    """Generate all smoketest fixture files and return the written file list."""

    files = planned_files(out_dir)
    ensure_can_write(files, overwrite=overwrite)

    rng = np.random.default_rng(seed)
    train = make_split("train", TRAIN_ROWS, rng)
    val = make_split("val", VAL_ROWS, rng)
    test = make_split("test", TEST_ROWS, rng)

    ml_features = out_dir / FEATURE_DIR_NAME
    contract_cases = out_dir / CONTRACT_DIR_NAME
    bad_cases = out_dir / BAD_DIR_NAME

    write_parquet(train, ml_features / "ml_exp00_Xy_train_smoketest.parquet")
    write_parquet(val, ml_features / "ml_exp00_Xy_val_smoketest.parquet")
    write_parquet(test, ml_features / "ml_exp00_Xy_test_smoketest.parquet")
    write_catalog(make_catalog(), ml_features / "ml_feature_columns_smoketest.csv")

    shuffled_columns = ["label", "tx_id", "amount", "split", *reversed(feature_columns())]
    write_parquet(
        val.loc[:, shuffled_columns],
        contract_cases / "val_shuffled_columns_smoketest_contract_cases.parquet",
    )

    extra_column = val.copy()
    extra_column["unused_runtime_note"] = "extra_column_should_be_ignored"
    write_parquet(extra_column, contract_cases / "val_extra_column_smoketest_contract_cases.parquet")

    write_parquet(
        val.drop(columns=["feature_03"]),
        bad_cases / "val_missing_feature_smoketest_bad_cases.parquet",
    )
    write_parquet(
        val.drop(columns=["label"]),
        bad_cases / "val_missing_label_smoketest_bad_cases.parquet",
    )

    wrong_split = val.copy()
    wrong_split["split"] = "train"
    write_parquet(wrong_split, bad_cases / "val_wrong_split_smoketest_bad_cases.parquet")

    nan_feature = val.copy()
    nan_feature["feature_04"] = np.nan
    write_parquet(nan_feature, bad_cases / "val_nan_feature_smoketest_bad_cases.parquet")

    wrong_dtype = val.copy()
    wrong_dtype["feature_05"] = wrong_dtype["feature_05"].map(lambda value: f"bad_{value:.4f}")
    write_parquet(wrong_dtype, bad_cases / "val_wrong_dtype_smoketest_bad_cases.parquet")

    null_explosion = val.copy()
    null_explosion["feature_06"] = np.nan
    write_parquet(null_explosion, bad_cases / "val_null_explosion_smoketest_bad_cases.parquet")

    inf_feature = val.copy()
    inf_feature.loc[inf_feature.index[:2], "feature_07"] = np.inf
    write_parquet(inf_feature, bad_cases / "val_inf_feature_smoketest_bad_cases.parquet")

    single_class = val.copy()
    single_class["label"] = 0
    write_parquet(single_class, bad_cases / "val_single_class_smoketest_bad_cases.parquet")

    write_catalog(
        make_catalog(label_leak=True),
        bad_cases / "val_label_leak_catalog_smoketest_bad_cases.csv",
    )
    write_catalog(
        make_catalog(duplicate_feature=True),
        bad_cases / "val_duplicate_feature_catalog_smoketest_bad_cases.csv",
    )
    write_catalog(
        make_catalog(empty_features=True),
        bad_cases / "val_empty_feature_catalog_smoketest_bad_cases.csv",
    )
    write_catalog(
        make_catalog(invalid_used_in_ml=True),
        bad_cases / "val_invalid_used_in_ml_catalog_smoketest_bad_cases.csv",
    )
    write_catalog(
        make_catalog(forbidden_name=True),
        bad_cases / "val_forbidden_name_catalog_smoketest_bad_cases.csv",
    )

    return files


def print_plan(out_dir: Path, seed: int, overwrite: bool, dry_run: bool) -> None:
    """Print generation settings and planned files."""

    files = planned_files(out_dir)
    existing = [item.path for item in files if item.path.exists()]

    print("out_dir:", out_dir)
    print("seed:", seed)
    print("dry_run:", dry_run)
    print("overwrite:", overwrite)
    print("planned_file_count:", len(files))
    if existing:
        print("existing_file_count:", len(existing))
        if not overwrite:
            print("existing files would block generation without --overwrite")

    for item in files:
        row_text = "csv" if item.rows is None else f"{item.rows} rows"
        marker = "exists" if item.path.exists() else "new"
        print(f"- [{marker}] {item.path} ({row_text}; {item.description})")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate ML-00 smoketest fixture files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR,
        help="Output directory containing ml_features/, contract_cases/, bad_cases/. Defaults to this script directory.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic synthetic data.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing generated fixture files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned writes without creating files.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    out_dir = resolve_output_dir(args.out_dir)
    print_plan(out_dir=out_dir, seed=args.seed, overwrite=args.overwrite, dry_run=args.dry_run)

    if args.dry_run:
        return

    files = generate_fixtures(out_dir=out_dir, seed=args.seed, overwrite=args.overwrite)
    print("generated_file_count:", len(files))


if __name__ == "__main__":
    main()
