"""ML-06 feature-build I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_PATH = BASE_DIR / "ml/ml-05/ml_inputs/r00/ml_05__r00_Xy_all.parquet"
DEFAULT_SOURCE_CONTRACT_PATH = BASE_DIR / "ml/ml-05/ml_inputs/r00/ml_05__r00_fb_output_feature_contract_approve.csv"
DEFAULT_FEATURE_COLUMNS_PATH = BASE_DIR / "ml/ml-05/ml_outputs/r00/ml_05__r00__d00-optuna_t25_feature_columns.json"
DEFAULT_SOURCE_FEATURE_TYPES_PATH = BASE_DIR / "ml/ml-05/ml_inputs/r00/ml_05__r00_feature_types.json"
DEFAULT_SOURCE_ENCODING_MANIFEST_PATH = BASE_DIR / "ml/ml-05/ml_inputs/r00/ml_05__r00_encoding_manifest.json"
DEFAULT_FB_INPUT_DIR = BASE_DIR / "ml/ml-06/fb_inputs/r00"
DEFAULT_INPUT_CONTRACT_PATH = DEFAULT_FB_INPUT_DIR / "ml_06__r00_fb_input_feature_contract.csv"
DEFAULT_INPUT_CATEGORY_VALUES_PATH = DEFAULT_FB_INPUT_DIR / "ml_06__r00_category_values.json"
DEFAULT_FB_OUTPUT_DIR = BASE_DIR / "ml/ml-06/fb_outputs/r00"
DEFAULT_ML_INPUT_DIR = BASE_DIR / "ml/ml-06/ml_inputs/r00"


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to the repository root by default."""

    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    root = BASE_DIR if base_dir is None else Path(base_dir).expanduser().resolve()
    return (root / raw_path).resolve()


def utc_now_iso() -> str:
    """Return an ISO timestamp without adding a project dependency."""

    return pd.Timestamp.utcnow().isoformat()


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON."""

    json_path = resolve_path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> Path:
    """Write UTF-8 JSON after checking overwrite policy."""

    output_path = resolve_path(path)
    require_parent_dir(output_path)
    require_can_write(output_path, overwrite=overwrite)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return output_path


def require_parent_dir(path: str | Path) -> None:
    """Create the parent directory for an output path."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def require_can_write(path: str | Path, *, overwrite: bool) -> None:
    """Refuse accidental overwrite unless explicitly allowed."""

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists and overwrite=False: {output_path}")


def require_no_existing_outputs(paths: Iterable[str | Path], *, overwrite: bool) -> None:
    """Validate overwrite policy for a group of output paths."""

    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "ML-06 output already exists and overwrite=False. "
            f"existing={existing[:20]}, existing_count={len(existing)}"
        )


def load_feature_columns(path: str | Path) -> list[str]:
    """Load feature columns from a list or a dict-style artifact."""

    payload = read_json(path)
    if isinstance(payload, list):
        columns = payload
    elif isinstance(payload, dict):
        columns = payload.get("feature_columns", payload.get("columns", []))
    else:
        raise ValueError(f"unsupported feature column JSON payload type: {type(payload).__name__}")
    if not isinstance(columns, list) or not all(str(column).strip() for column in columns):
        raise ValueError(f"feature column artifact must contain a non-empty list of names: {path}")
    return [str(column).strip() for column in columns]


def load_parquet_frame(path: str | Path, *, sample_rows: int | None = None) -> pd.DataFrame:
    """Load a parquet frame, optionally taking up to N rows per split."""

    input_path = resolve_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"input parquet not found: {input_path}")
    if sample_rows is None:
        return pd.read_parquet(input_path)
    if sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer or None")
    return load_parquet_split_sample(input_path, sample_rows=sample_rows)


def load_parquet_split_sample(path: str | Path, *, sample_rows: int) -> pd.DataFrame:
    """Read at most ``sample_rows`` rows from each train/val/test split."""

    input_path = resolve_path(path)
    parquet_file = pq.ParquetFile(input_path)
    selected_batches: list[pa.RecordBatch] = []
    remaining = {"train": sample_rows, "val": sample_rows, "test": sample_rows}
    required = set(remaining)

    for batch in parquet_file.iter_batches(batch_size=65_536):
        frame = batch.to_pandas()
        if "split" not in frame.columns:
            raise ValueError(f"input parquet is missing split column: {input_path}")
        frame["_split_norm"] = frame["split"].astype("string").str.strip().str.lower()
        parts: list[pd.DataFrame] = []
        for split_name in ("train", "val", "test"):
            need = remaining[split_name]
            if need <= 0:
                continue
            split_part = frame.loc[frame["_split_norm"] == split_name].drop(columns=["_split_norm"])
            if split_part.empty:
                continue
            take = split_part.head(need)
            parts.append(take)
            remaining[split_name] -= len(take)
        if parts:
            selected_batches.append(pa.RecordBatch.from_pandas(pd.concat(parts, ignore_index=True)))
        if all(count <= 0 for count in remaining.values()):
            break

    missing = sorted(split for split in required if remaining[split] == sample_rows)
    if missing:
        raise ValueError(f"sample input did not contain required split values: {missing}")
    if not selected_batches:
        raise ValueError(f"sample input produced no rows: {input_path}")
    return pa.Table.from_batches(selected_batches).to_pandas()


def save_dataframe_csv(df: pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    """Save a DataFrame as UTF-8-SIG CSV."""

    output_path = resolve_path(path)
    require_parent_dir(output_path)
    require_can_write(output_path, overwrite=overwrite)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def save_dataframe_parquet(df: pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    """Save a DataFrame as parquet."""

    output_path = resolve_path(path)
    require_parent_dir(output_path)
    require_can_write(output_path, overwrite=overwrite)
    df.to_parquet(output_path, index=False)
    return output_path
