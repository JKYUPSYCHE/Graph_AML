"""ML-01 feature build 검증 helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd


def validate_unique_tx_ids(df: pd.DataFrame) -> None:
    """train/val/test 전체 합본에서 tx_id가 중복되지 않는지 확인한다."""

    duplicated = df["tx_id"].astype("string").duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "tx_id"].astype(str).head(10).tolist()
        raise ValueError(
            "Feature build failed: tx_id values are duplicated across split files. "
            f"duplicated_count={int(duplicated.sum())}, examples={examples}"
        )


def validate_time_split(df: pd.DataFrame) -> None:
    """split 결과가 train < val < test 시간 순서를 만족하는지 검사한다."""

    counts = df["split"].value_counts().to_dict()
    missing = {"train", "val", "test"} - set(counts)
    if missing:
        raise ValueError(f"Missing required split values in existing split column: {sorted(missing)}")

    train_max = df.loc[df["split"] == "train", "timestamp"].max()
    val_min = df.loc[df["split"] == "val", "timestamp"].min()
    val_max = df.loc[df["split"] == "val", "timestamp"].max()
    test_min = df.loc[df["split"] == "test", "timestamp"].min()
    if train_max >= val_min:
        raise ValueError(f"Time split boundary violation: train_max={train_max}, val_min={val_min}")
    if val_max >= test_min:
        raise ValueError(f"Time split boundary violation: val_max={val_max}, test_min={test_min}")


def normalize_existing_split_values(series: pd.Series, *, source_path: Path, split_col: str) -> pd.Series:
    """기존 split 컬럼 값을 train/val/test canonical 값으로 정규화하고 검증한다."""

    if series.isna().any():
        raise ValueError(
            "Existing split column has missing values. "
            f"path={source_path}, split_col={split_col!r}, missing_count={int(series.isna().sum())}"
        )

    normalized = series.astype("string").str.strip().str.lower()
    blank_mask = normalized == ""
    if blank_mask.any():
        raise ValueError(
            "Existing split column has blank values. "
            f"path={source_path}, split_col={split_col!r}, blank_count={int(blank_mask.sum())}"
        )

    allowed = {"train", "val", "test"}
    invalid = normalized[~normalized.isin(allowed)]
    if not invalid.empty:
        raise ValueError(
            "Existing split column has unsupported values. "
            f"path={source_path}, split_col={split_col!r}, allowed={sorted(allowed)}, "
            f"observed_examples={sorted(invalid.unique().tolist())[:20]}"
        )
    return normalized


def existing_split_metadata_frame(
    df: pd.DataFrame,
    *,
    source_path: Path,
    tx_id_col: str,
    timestamp_col: str,
    label_col: str,
    split_col: str,
) -> pd.DataFrame:
    """split-only 검증에 필요한 canonical metadata frame을 만든다."""

    required = {tx_id_col, timestamp_col, label_col, split_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Single parquet input is missing columns required for existing split export. "
            f"path={source_path}, missing={sorted(missing)}"
        )

    tx_id = df[tx_id_col]
    if tx_id.isna().any():
        raise ValueError(
            "tx_id column has missing values. "
            f"path={source_path}, tx_id_col={tx_id_col!r}, missing_count={int(tx_id.isna().sum())}"
        )

    raw_timestamp = df[timestamp_col]
    if raw_timestamp.isna().any():
        raise ValueError(
            "timestamp column has missing values. "
            f"path={source_path}, timestamp_col={timestamp_col!r}, missing_count={int(raw_timestamp.isna().sum())}"
        )
    timestamp = pd.to_datetime(raw_timestamp, errors="coerce")
    if timestamp.isna().any():
        failed = raw_timestamp.loc[timestamp.isna()].astype(str).head(5).tolist()
        raise ValueError(
            "timestamp parsing failed for existing split export. "
            f"path={source_path}, timestamp_col={timestamp_col!r}, "
            f"failed_count={int(timestamp.isna().sum())}, example_values={failed}"
        )

    raw_label = df[label_col]
    if raw_label.isna().any():
        raise ValueError(
            "label column has missing values. "
            f"path={source_path}, label_col={label_col!r}, missing_count={int(raw_label.isna().sum())}"
        )
    label = pd.to_numeric(raw_label, errors="coerce")
    if label.isna().any():
        failed = raw_label.loc[label.isna()].astype(str).head(5).tolist()
        raise ValueError(
            "label parsing failed for existing split export. "
            f"path={source_path}, label_col={label_col!r}, "
            f"failed_count={int(label.isna().sum())}, example_values={failed}"
        )
    label_values = sorted(label.dropna().unique().tolist())
    if not set(label_values).issubset({0, 1}):
        raise ValueError(
            "label must be binary 0/1 for existing split export. "
            f"path={source_path}, label_col={label_col!r}, observed_values={label_values[:20]}"
        )

    metadata = pd.DataFrame(
        {
            "tx_id": tx_id.reset_index(drop=True),
            "timestamp": timestamp.reset_index(drop=True),
            "label": label.astype("int8").reset_index(drop=True),
            "split": normalize_existing_split_values(df[split_col], source_path=source_path, split_col=split_col).reset_index(drop=True),
        }
    )
    validate_unique_tx_ids(metadata)
    validate_time_split(metadata)
    return metadata


def _numeric_validation_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """semantic validation에서 사용할 feature column을 finite numeric series로 변환한다."""

    if column not in frame.columns:
        raise ValueError(f"Stage 0 rolling validation failed: required column is missing. column={column!r}")
    numeric = pd.to_numeric(frame[column], errors="coerce").reset_index(drop=True)
    missing_mask = numeric.isna()
    if missing_mask.any():
        examples = frame.loc[missing_mask.to_numpy(), column].astype(str).head(5).tolist()
        raise ValueError(
            "Stage 0 rolling validation failed: feature column contains non-numeric or missing values. "
            f"column={column!r}, bad_count={int(missing_mask.sum())}, examples={examples}"
        )
    inf_mask = numeric.isin([float("inf"), float("-inf")])
    if inf_mask.any():
        raise ValueError(
            "Stage 0 rolling validation failed: feature column contains inf values. "
            f"column={column!r}, inf_count={int(inf_mask.sum())}"
        )
    return numeric.astype("float64")


def _stage0_monotonic_tolerance(
    *,
    short_col: str,
    long_col: str,
    short_values: pd.Series,
    long_values: pd.Series,
    base_tolerance: float,
) -> Union[float, pd.Series]:
    """count는 엄격히, amount sum은 float32/누적합 반올림 오차만 허용한다."""

    is_amount_sum = "__amount__sum__" in short_col and "__amount__sum__" in long_col
    if not is_amount_sum:
        return base_tolerance

    magnitude = np.maximum(
        short_values.abs().to_numpy(dtype="float64", copy=False),
        long_values.abs().to_numpy(dtype="float64", copy=False),
    )
    magnitude = np.maximum(magnitude, 1.0)
    roundoff_tolerance = 2.0 * float(np.finfo(np.float32).eps) * magnitude
    allowed_tolerance = np.maximum(roundoff_tolerance, max(base_tolerance, 1e-5))
    return pd.Series(allowed_tolerance, index=short_values.index, dtype="float64")


def _validation_tolerance_at(tolerance: Union[float, pd.Series], index: int) -> float:
    if isinstance(tolerance, pd.Series):
        return float(tolerance.iloc[index])
    return float(tolerance)


def _stage0_monotonic_rules() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Stage 0 count/sum window monotonic 검증 대상 window와 prefix를 반환한다."""

    return (
        ("w1h", "w6h", "w1d", "w3d", "w7d"),
        (
            "timehist__sender__out__tx_count__count__",
            "timehist__receiver__in__tx_count__count__",
            "timehist__sender__out__amount__sum__",
            "timehist__receiver__in__amount__sum__",
        ),
    )


def _stage0_diversity_rules() -> tuple[tuple[str, str], ...]:
    """Stage 0 short/long window 복제 검증 대상 feature pair를 반환한다."""

    return (
        ("timehist__sender__out__tx_count__count__w1h", "timehist__sender__out__tx_count__count__w7d"),
        ("timehist__receiver__in__tx_count__count__w1h", "timehist__receiver__in__tx_count__count__w7d"),
        ("timehist__sender__out__amount__sum__w1h", "timehist__sender__out__amount__sum__w7d"),
        ("timehist__receiver__in__amount__sum__w1h", "timehist__receiver__in__amount__sum__w7d"),
        ("timehist__sender__out__amount__cur_vs_mean_ratio__w1d", "timehist__sender__out__amount__cur_vs_mean_ratio__w7d"),
        ("timehist__receiver__in__amount__cur_vs_mean_ratio__w1d", "timehist__receiver__in__amount__cur_vs_mean_ratio__w7d"),
    )


def _stage0_duplicate_timestamp_rules() -> tuple[tuple[str, str, str, str], ...]:
    """Stage 0 same-timestamp leakage 검증 대상 count feature를 반환한다."""

    return (
        ("sender_account_id", "timehist__sender__out__tx_count__count__", "w1h", "1h"),
        ("sender_account_id", "timehist__sender__out__tx_count__count__", "w7d", "7d"),
        ("receiver_account_id", "timehist__receiver__in__tx_count__count__", "w1h", "1h"),
        ("receiver_account_id", "timehist__receiver__in__tx_count__count__", "w7d", "7d"),
    )


def validate_stage0_rolling_outputs(
    feature_frame: pd.DataFrame,
    *,
    min_rows_for_diversity_check: int = 1_000,
    duplicate_group_sample_size: int = 3,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """ML-01 Stage 0 rolling 산출물이 저장되기 전에 semantic 오류를 차단한다."""

    if feature_frame.empty:
        raise ValueError("Stage 0 rolling validation failed: feature_frame is empty.")
    if min_rows_for_diversity_check <= 0:
        raise ValueError("min_rows_for_diversity_check must be a positive integer.")
    if duplicate_group_sample_size < 0:
        raise ValueError("duplicate_group_sample_size must be zero or a positive integer.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    windows, monotonic_prefixes = _stage0_monotonic_rules()
    diversity_pairs = _stage0_diversity_rules()
    duplicate_count_checks = _stage0_duplicate_timestamp_rules()

    series_cache: dict[str, pd.Series] = {}

    def get_series(column: str) -> pd.Series:
        if column not in series_cache:
            series_cache[column] = _numeric_validation_series(feature_frame, column)
        return series_cache[column]

    monotonic_failures: list[dict[str, Any]] = []
    monotonic_checks = 0
    for prefix in monotonic_prefixes:
        existing_windows = [window for window in windows if f"{prefix}{window}" in feature_frame.columns]
        for short_window, long_window in zip(existing_windows, existing_windows[1:]):
            short_col = f"{prefix}{short_window}"
            long_col = f"{prefix}{long_window}"
            short_values = get_series(short_col)
            long_values = get_series(long_col)
            monotonic_tolerance = _stage0_monotonic_tolerance(
                short_col=short_col,
                long_col=long_col,
                short_values=short_values,
                long_values=long_values,
                base_tolerance=tolerance,
            )
            excess = short_values - long_values - monotonic_tolerance
            bad_mask = excess > 0
            monotonic_checks += 1
            if bad_mask.any():
                first_bad_index = int(bad_mask[bad_mask].index[0])
                monotonic_failures.append(
                    {
                        "short_col": short_col,
                        "long_col": long_col,
                        "bad_count": int(bad_mask.sum()),
                        "first_bad_index": first_bad_index,
                        "short_value": float(short_values.iloc[first_bad_index]),
                        "long_value": float(long_values.iloc[first_bad_index]),
                        "allowed_tolerance": _validation_tolerance_at(monotonic_tolerance, first_bad_index),
                        "max_excess": float(excess.loc[bad_mask].max()),
                    }
                )
    if monotonic_failures:
        raise ValueError(
            "Stage 0 rolling validation failed: shorter count/sum window is greater than longer window. "
            f"failures={monotonic_failures[:10]}"
        )

    diversity_failures: list[dict[str, Any]] = []
    diversity_checks = 0
    if len(feature_frame) >= min_rows_for_diversity_check:
        for short_col, long_col in diversity_pairs:
            if short_col not in feature_frame.columns or long_col not in feature_frame.columns:
                continue
            short_values = get_series(short_col)
            long_values = get_series(long_col)
            diversity_checks += 1
            max_abs_diff = float((short_values - long_values).abs().max())
            has_signal = bool((short_values.abs().max() > tolerance) or (long_values.abs().max() > tolerance))
            has_variation = bool(short_values.nunique(dropna=True) > 1 or long_values.nunique(dropna=True) > 1)
            if max_abs_diff <= tolerance and has_signal and has_variation:
                diversity_failures.append(
                    {
                        "short_col": short_col,
                        "long_col": long_col,
                        "max_abs_diff": max_abs_diff,
                        "short_unique_count": int(short_values.nunique(dropna=True)),
                        "long_unique_count": int(long_values.nunique(dropna=True)),
                    }
                )
    if diversity_failures:
        raise ValueError(
            "Stage 0 rolling validation failed: short and long rolling windows are exact duplicates. "
            "This usually indicates a stale notebook import or old rolling implementation. "
            f"failures={diversity_failures[:10]}"
        )

    duplicate_timestamp_checks = 0
    if duplicate_group_sample_size > 0 and "timestamp" in feature_frame.columns:
        timestamps = pd.to_datetime(feature_frame["timestamp"], errors="coerce").reset_index(drop=True)
        if timestamps.isna().any():
            raise ValueError(
                "Stage 0 rolling validation failed: timestamp column cannot be parsed. "
                f"bad_count={int(timestamps.isna().sum())}"
            )
        for entity_col, prefix, window_suffix, window_value in duplicate_count_checks:
            count_col = f"{prefix}{window_suffix}"
            if entity_col not in feature_frame.columns or count_col not in feature_frame.columns:
                continue
            entity = feature_frame[entity_col].astype("string").str.strip().reset_index(drop=True)
            if entity.isna().any() or (entity == "").any():
                raise ValueError(
                    "Stage 0 rolling validation failed: entity column has missing or blank values. "
                    f"entity_col={entity_col!r}"
                )
            key_frame = pd.DataFrame({"_entity": entity, "_timestamp": timestamps})
            duplicate_mask = key_frame.duplicated(["_entity", "_timestamp"], keep=False)
            duplicate_keys = key_frame.loc[duplicate_mask, ["_entity", "_timestamp"]].drop_duplicates().head(
                duplicate_group_sample_size
            )
            if duplicate_keys.empty:
                continue
            count_values = get_series(count_col)
            window = pd.Timedelta(window_value)
            for _, duplicate_key in duplicate_keys.iterrows():
                entity_value = duplicate_key["_entity"]
                timestamp_value = duplicate_key["_timestamp"]
                same_entity = entity == entity_value
                same_timestamp = timestamps == timestamp_value
                group_mask = same_entity & same_timestamp
                expected_count = int((same_entity & (timestamps >= timestamp_value - window) & (timestamps < timestamp_value)).sum())
                observed_values = count_values.loc[group_mask].unique()
                duplicate_timestamp_checks += 1
                if any(abs(float(observed) - expected_count) > tolerance for observed in observed_values):
                    raise ValueError(
                        "Stage 0 rolling validation failed: duplicate timestamp rows were included as history. "
                        f"entity_col={entity_col!r}, count_col={count_col!r}, entity={entity_value!r}, "
                        f"timestamp={timestamp_value}, expected_count={expected_count}, "
                        f"observed_values={[float(value) for value in observed_values[:10]]}"
                    )

    total_checks = monotonic_checks + diversity_checks + duplicate_timestamp_checks
    if total_checks == 0:
        raise ValueError(
            "Stage 0 rolling validation could not run any check. "
            "Confirm that Stage 0 rolling columns are present before saving artifacts."
        )

    return {
        "rows": int(len(feature_frame)),
        "monotonic_checks": monotonic_checks,
        "diversity_checks": diversity_checks,
        "duplicate_timestamp_checks": duplicate_timestamp_checks,
        "validated_columns": sorted(series_cache),
    }
