"""I/O helpers for ML-03 feature build.

Code map:
- Input: project-relative paths, parquet schema requests, and DataFrames to save.
- Output: resolved paths, loaded parquet samples, saved CSV/parquet/JSON artifacts.
- Public: resolve_path, parquet_columns, load_parquet_columns, load_parquet_split_sample.
- Leakage guard: split sampling tries to include train/val/test instead of first train-only rows.
- Notes: DEFAULT_INPUT_PATH remains the ML-02 r01 all parquet contract.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import pandas as pd
import pyarrow.parquet as pq

from ml_03_fb_utils import BASE_DIR


DEFAULT_INPUT_PATH = BASE_DIR / "ml" / "ml-02" / "fb_outputs" / "r01" / "ml_02__r01_Xy_all.parquet"


def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve absolute or project-relative paths."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    base = BASE_DIR if base_dir is None else Path(base_dir).expanduser().resolve()
    return (base / candidate).resolve()


def parquet_columns(path: Union[str, Path]) -> list[str]:
    """Read parquet schema column names without loading all rows."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    return list(pq.ParquetFile(path).schema_arrow.names)


def parquet_row_count(path: Union[str, Path]) -> int:
    """Read parquet row count from metadata."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    return int(pq.ParquetFile(path).metadata.num_rows)


def parquet_schema_types(path: Union[str, Path]) -> dict[str, str]:
    """Read parquet schema types from metadata."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    schema = pq.ParquetFile(path).schema_arrow
    return {field.name: str(field.type) for field in schema}


def load_parquet_columns(
    path: Union[str, Path],
    columns: Iterable[str],
    sample_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load selected parquet columns, optionally only the first batch."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    if sample_rows is not None and sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer or None.")

    selected_columns = list(dict.fromkeys(str(column) for column in columns))
    if not selected_columns:
        raise ValueError("columns must not be empty.")

    if sample_rows is None:
        return pd.read_parquet(path, columns=selected_columns)

    parquet_file = pq.ParquetFile(path)
    batches = parquet_file.iter_batches(batch_size=sample_rows, columns=selected_columns)
    try:
        first_batch = next(batches)
    except StopIteration as exc:
        raise ValueError(f"parquet file has no rows: {path}") from exc
    return first_batch.to_pandas()


def load_parquet_split_sample(
    path: Union[str, Path],
    columns: Iterable[str],
    *,
    sample_rows: int,
    split_col: str = "split",
    split_values: tuple[str, ...] = ("train", "val", "test"),
    scan_summary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a small sample that contains rows from every split when available.

    A plain first-batch sample usually contains only train rows because ML-ready
    parquets are time sorted. This helper scans parquet row groups in batches and
    keeps at most ``sample_rows // len(split_values)`` rows per split.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    if sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer.")

    selected_columns = list(dict.fromkeys(str(column) for column in columns))
    if split_col not in selected_columns:
        selected_columns.append(split_col)
    if not selected_columns:
        raise ValueError("columns must not be empty.")

    parquet_file = pq.ParquetFile(path)
    schema_names = list(parquet_file.schema_arrow.names)
    if split_col not in schema_names:
        raise ValueError(f"split column is missing from parquet schema: {split_col!r}")
    split_column_index = schema_names.index(split_col)
    per_split = max(1, sample_rows // len(split_values))
    remaining = {split: per_split for split in split_values}
    parts: list[pd.DataFrame] = []
    stats = {
        "path": str(path),
        "sample_rows": int(sample_rows),
        "split_col": split_col,
        "split_values": list(split_values),
        "row_groups_total": int(parquet_file.num_row_groups),
        "row_groups_seen": 0,
        "row_groups_read": 0,
        "row_groups_skipped_by_stats": 0,
        "batches_read": 0,
        "rows_read": 0,
        "per_split_target": int(per_split),
        "collected_by_split": {split: 0 for split in split_values},
    }

    for row_group_index in range(parquet_file.num_row_groups):
        if all(count <= 0 for count in remaining.values()):
            break
        stats["row_groups_seen"] += 1
        row_group = parquet_file.metadata.row_group(row_group_index)
        split_stats = row_group.column(split_column_index).statistics
        needed_splits = {split for split, count in remaining.items() if count > 0}
        if split_stats is not None and split_stats.min is not None and split_stats.max is not None:
            min_value = str(split_stats.min)
            max_value = str(split_stats.max)
            if not any(min_value <= split <= max_value for split in needed_splits):
                stats["row_groups_skipped_by_stats"] += 1
                continue

        row_group_was_read = False
        for batch in parquet_file.iter_batches(
            batch_size=max(sample_rows, 1024),
            columns=selected_columns,
            row_groups=[row_group_index],
        ):
            if all(count <= 0 for count in remaining.values()):
                break
            row_group_was_read = True
            stats["batches_read"] += 1
            stats["rows_read"] += int(batch.num_rows)
            batch_df = batch.to_pandas()
            normalized_split = batch_df[split_col].astype("string").str.strip().str.lower()
            for split in split_values:
                take_count = remaining[split]
                if take_count <= 0:
                    continue
                split_rows = batch_df.loc[normalized_split == split]
                if split_rows.empty:
                    continue
                selected = split_rows.head(take_count)
                parts.append(selected)
                remaining[split] -= len(selected)
                stats["collected_by_split"][split] += int(len(selected))
        if row_group_was_read:
            stats["row_groups_read"] += 1

    if not parts:
        raise ValueError(f"parquet split sample produced no rows: {path}")
    sampled = pd.concat(parts, ignore_index=True)
    stats["remaining_by_split"] = {split: int(count) for split, count in remaining.items()}
    stats["output_rows"] = int(len(sampled))
    if scan_summary is not None:
        scan_summary.clear()
        scan_summary.update(stats)
    return sampled.loc[:, selected_columns]


def save_json(payload: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Save mapping payload as UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, ensure_ascii=False, indent=2)


def save_dataframe_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save a DataFrame as UTF-8-SIG CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_dataframe_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save a DataFrame as parquet."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def utc_now_iso() -> str:
    """Return current UTC timestamp for run metadata."""

    return datetime.now(timezone.utc).isoformat()
