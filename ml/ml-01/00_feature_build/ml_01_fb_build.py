"""
Feature build 전체 실행 모듈

이 파일의 역할
----------------
1. 기존 split 컬럼이 있는 단일 parquet를 train/val/test parquet로 분리한다.
2. split 컬럼이 있는 단일 parquet 또는 DataFrame 입력에서 feature build 전체 흐름을 실행한다.
3. 입력 컬럼 resolve, strict parsing, 기존 split 검증/보존, operation 실행을 순서대로 연결한다.
4. train/val/test parquet와 catalog/summary CSV를 저장한다.
5. 사용자가 선택한 FeatureSpec 목록을 feature 생성의 유일한 기준으로 사용한다.

중요한 설계 원칙
----------------
- Stage 이름은 feature 생성을 결정하지 않는다.
- `feature_specs` 목록만 어떤 feature가 생성될지 결정한다.
- 컬럼명이 바뀌면 노트북의 `column_map` dict를 우선 수정한다.
- full data 실행 전 sample_rows로 smoke build를 먼저 수행하는 것을 권장한다.
- overwrite=False가 기본이며 기존 산출물이 있으면 즉시 중단한다.

처음 읽는 순서
--------------
1. `FeatureBuildConfig`와 `FeatureBuildResult`로 입력/출력 계약을 확인한다.
2. 공개 진입점인 `build_features()`와 `build_features_from_frame()`을 먼저 읽는다.
3. 두 진입점이 공통으로 도달하는 `_build_from_raw_frame()`과 `_build_from_split_frame()`을 읽는다.
4. operation 세부 계산은 이 파일이 아니라 `ml_01_fb_operations.py`와 `ml_01_fb_rolling.py`에서 확인한다.
5. `validate_stage0_rolling_outputs()`는 build 후 저장 전 semantic guard로 따로 읽는다.

섹션 지도
---------
1. 실행 설정과 결과 객체
2. split/frame 조립 helper
3. Stage 0 rolling semantic validation
4. 기존 split metadata 검증 helper
5. 공개 feature build 진입점
6. 내부 FeatureSpec/build 검증 helper
7. 공통 build 본체
8. artifact summary helper
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ml_01_fb_catalog import make_feature_catalog, make_feature_columns_table, make_split_summary
from ml_01_fb_io import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    FeatureBuildOutputPaths,
    load_parquet_columns,
    make_output_paths,
    parquet_columns,
    require_no_existing_outputs,
    resolve_path,
    save_dataframe_csv,
    save_dataframe_parquet,
    save_json,
    utc_now_iso,
)
from ml_01_fb_operations import execute_feature_specs
from ml_01_fb_schema import (
    resolve_requested_columns,
    standardize_input_frame,
    validate_no_forbidden_input_columns,
)
from ml_01_fb_specs import (
    FeatureSpec,
    feature_columns,
    required_input_columns,
    validate_feature_specs,
)


# -----------------------------------------------------------------------------
# 1. 실행 설정과 결과 객체
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureBuildConfig:
    """
    feature build 실행 설정이다.

    주요 필드
    ---------
    input_path:
        입력 parquet 경로. None이면 build_features()에서는 사용할 수 없고
        build_features_from_frame()을 써야 한다.
    output_dir:
        산출물 저장 폴더. None이면 파일 저장 없이 메모리 결과만 반환한다.
        공식 노트북 경로는 output_dir=None으로 build 후 encode_split_frame()에서 저장한다.
        output_dir을 지정하는 direct-save 모드는 보조 실행 경로다.
    feature_specs:
        생성할 ML-01 Stage 0 feature 선언 목록. 명시하지 않으면 실행을 중단한다.
    column_map:
        노트북에서 직접 지정하는 logical column -> source column 매핑이다.
        예: {"amount": "Amount Paid"}. 지정된 값은 기본 후보보다 우선한다.
    sample_rows:
        sample smoke build용 row 수. None이면 전체 parquet를 읽는다.
    overwrite:
        기존 산출물 덮어쓰기 허용 여부. 기본값 False.
    """
    # --- 입력 / 출력 경로 (None 허용 = 호출 방식에 따라 다름) ---
    # 상대경로로 들어와도 __post_init__에서 절대경로로 정규
    input_path: Optional[Union[str, Path]] = DEFAULT_INPUT_PATH
    output_dir: Optional[Union[str, Path]] = DEFAULT_OUTPUT_DIR
    # base_dir: input_path/output_dir이 상대경로일 때 해석 기준이 되는 루트.
    # None이면 ml_01_fb_io.resolve_path() 내부에서 ml_01_fb_utils.BASE_DIR(=Git 루트)를 사용한다.
    base_dir: Optional[Union[str, Path]] = None

    # --- 실행 식별자 (재현성 메타데이터 + 산출물 파일명 prefix) ---
    # experiment_id는 {experiment_id}_Xy_train.parquet 같은 파일명에 들어간다.
    # run_name은 feature_catalog.csv에 기록되어 사람이 어떤 실행이었는지 추적하는 용도.
    experiment_id: str = "feature_build"
    run_name: str = "user_selected_operations"

    # --- feature 생성 선언 ---
    # ML-01은 contract가 확정한 BUILD_FEATURE_SPECS를 명시적으로 넘겨야 한다.
    feature_specs: Optional[Tuple[FeatureSpec, ...]] = None

    # --- 컬럼명 매핑 ---
    # 노트북의 COLUMN_MAP이 그대로 들어온다. 예: {"amount": "Amount Paid"}.
    # 여기 없는 logical key는 ml_01_fb_schema.COLUMN_CANDIDATES로 자동 fallback된다.
    # 명시성을 위해 모든 key를 직접 넘기는 것을 권장.
    column_map: Optional[Mapping[str, str]] = None

    sample_rows: Optional[int] = None    # smoke build 옵션
    overwrite: bool = False              # overwrite 옵션

    tx_id_col: str = "tx_id"
    timestamp_col: str = "timestamp"
    label_col: str = "label"

    def __post_init__(self) -> None:
        """
        설정 객체 생성 직후 경로와 숫자 파라미터를 검증한다.

        주의: 이 클래스는 frozen=True라 일반 대입(self.x = ...)이 금지
        그래서 경로 정규화나 column_map 클렌징처럼 "값을 교체"해야 할 때는
        object.__setattr__(self, "필드명", 값)으로 frozen 제약을 한 번만 우회
        이 패턴은 dataclass 공식 문서에서 권장하는 frozen + __post_init__ 관용구
        """
        # [1] 경로 정규화: 상대경로로 들어와도 절대경로로 바꿔 둔다.
        if self.input_path is not None:
            object.__setattr__(self, "input_path", resolve_path(self.input_path, self.base_dir))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", resolve_path(self.output_dir, self.base_dir))

        # [2] sample_rows 검증: None은 허용(=전체 로드), 0이나 음수는 의미가 없으므로 차단.
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer or None.")
        if not str(self.experiment_id).strip():
            raise ValueError("experiment_id must not be empty.")

        # [3] 식별자 검증: 빈 문자열/공백만 있으면 파일명 생성 시 사고가 나므로 차단.
        if not str(self.run_name).strip():
            raise ValueError("run_name must not be empty.")

        # [4] column_map 클렌징:
        # 사용자가 노트북에서 손으로 만든 dict라 다음과 같은 사소한 오류에 대한 보정 필요
        #   - key/value 앞뒤 공백  -> strip()으로 정리
        #   - key 빈 문자열         -> 즉시 에러 (resolve 단계까지 가면 원인 파악이 어려움)
        #   - strip  key 중복       -> 즉시 에러 (어느 매핑이 이길지 모호해짐)
        # 클렌징된 dict로 교체 (frozen이므로 object.__setattr__ 사용).
        if self.column_map is not None:
            cleaned_column_map: dict[str, str] = {}
            for logical_name, source_column in self.column_map.items():
                logical = str(logical_name).strip()
                source = str(source_column).strip()
                if not logical or not source:
                    raise ValueError(
                        "column_map keys and values must not be blank. "
                        f"logical_name={logical_name!r}, source_column={source_column!r}"
                    )
                if logical in cleaned_column_map:
                    raise ValueError(f"column_map has duplicated logical name after stripping: {logical!r}")
                cleaned_column_map[logical] = source
            object.__setattr__(self, "column_map", cleaned_column_map)


@dataclass(frozen=True)
class FeatureBuildResult:
    """feature build 실행 후 사용자에게 반환되는 결과 객체다."""

    output_paths: Optional[FeatureBuildOutputPaths]
    feature_columns: list[str]
    row_counts: dict[str, int]
    build_summary: Mapping[str, Any]
    feature_frame: pd.DataFrame
    feature_info: pd.DataFrame


# -----------------------------------------------------------------------------
# 2. split/frame 조립 helper
# -----------------------------------------------------------------------------
# ML-01 feature build 기본 흐름은 source parquet의 기존 split 컬럼만 사용한다.
# 이 섹션은 split 경계 검증과 원본 frame + 생성 feature 조립에 필요한 작은 helper를 모은다.
# -----------------------------------------------------------------------------
def _validate_time_split(df: pd.DataFrame) -> None:
    """
    split 결과가 train < val < test 시간 순서를 만족하는지 검사

    검사 항목
    --------
    1. train/val/test 세 split이 모두 존재 (어느 하나도 비어서는 안 됨)
    2. max(train.timestamp) < min(val.timestamp)
    3. max(val.timestamp)   < min(test.timestamp)
    """

    # [1] 세 split이 모두 존재하는지 확인. 누락 시 이후 학습 단계에서 KeyError가 나므로 여기서 잡는다.
    counts = df["split"].value_counts().to_dict()
    missing = {"train", "val", "test"} - set(counts)
    if missing:
        raise ValueError(f"Missing required split values in existing split column: {sorted(missing)}")

    # [2] 인접 split 경계에서 시간 역전 또는 같은 timestamp 공유가 없는지 검사.
    # temporal/history feature의 과거 기준은 past_timestamp < current_timestamp로 고정한다.
    # 따라서 같은 timestamp가 split 경계 양쪽에 있으면 보수적으로 실패시킨다.
    train_max = df.loc[df["split"] == "train", "timestamp"].max()
    val_min = df.loc[df["split"] == "val", "timestamp"].min()
    val_max = df.loc[df["split"] == "val", "timestamp"].max()
    test_min = df.loc[df["split"] == "test", "timestamp"].min()
    if train_max >= val_min:
        raise ValueError(f"Time split boundary violation: train_max={train_max}, val_min={val_min}")
    if val_max >= test_min:
        raise ValueError(f"Time split boundary violation: val_max={val_max}, test_min={test_min}")



def _split_feature_frame(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    최종 feature_frame(=meta + 생성 feature)을 train/val/test 3개로 분리

    호출 시점
    --------
    execute_feature_specs()가 끝난 뒤, 파일 저장 직전에 호출한다.
    feature 계산까지 마친 한 덩어리 frame을 split별 parquet로 저장하기 위한 마지막 단계.

    빈 split 방어
    ------------
    _validate_time_split이 이미 비어 있지 않음을 보장하지만, reset_index까지 거친 뒤 한 번 더 확인
    """

    train_df = feature_frame[feature_frame["split"] == "train"].reset_index(drop=True)
    val_df = feature_frame[feature_frame["split"] == "val"].reset_index(drop=True)
    test_df = feature_frame[feature_frame["split"] == "test"].reset_index(drop=True)

    # 하나라도 비면 학습 단계에서 의미 없는 산출물이 만들어지므로 명시적으로 중단.
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Feature split output must not be empty. "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    return train_df, val_df, test_df


def _preserve_source_columns(raw_df: pd.DataFrame, standardized_df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼 전체를 보존하고 feature build용 표준 컬럼을 덧붙인다."""

    output = raw_df.reset_index(drop=True).copy(deep=False)
    for column in standardized_df.columns:
        output[column] = standardized_df[column].reset_index(drop=True)
    return output


def _append_generated_features(
    source_frame: pd.DataFrame,
    built_feature_frame: pd.DataFrame,
    generated_columns: list[str],
) -> pd.DataFrame:
    """원본 보존 frame에 새로 계산한 feature 컬럼만 추가한다."""

    source = source_frame.reset_index(drop=True).copy(deep=False)
    built = built_feature_frame.reset_index(drop=True)
    if len(source) != len(built):
        raise ValueError(
            "Feature build failed: source and generated feature row counts differ. "
            f"source_rows={len(source)}, generated_rows={len(built)}"
        )

    for meta_col in ["tx_id", "split", "label"]:
        if meta_col not in source.columns or meta_col not in built.columns:
            raise ValueError(f"Feature build failed: metadata column is missing before append. column={meta_col!r}")
        left = source[meta_col].astype("string").reset_index(drop=True)
        right = built[meta_col].astype("string").reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"Feature build failed: metadata order mismatch before append. column={meta_col!r}")

    collisions = [column for column in generated_columns if column in source.columns]
    if collisions:
        raise ValueError(
            "Feature build refused to overwrite existing columns. "
            f"generated_columns_already_exist={collisions[:30]}, collision_count={len(collisions)}"
        )

    for column in generated_columns:
        if column not in built.columns:
            raise ValueError(f"Feature build failed: generated column is missing. column={column!r}")
        source[column] = built[column]
    source["label"] = built["label"].astype("int8")
    source["split"] = built["split"].astype("string")
    return source


# =============================================================================
# 3. Stage 0 rolling semantic validation
# =============================================================================
# build 후 저장 전에 실행하는 공개 검증 함수와 내부 helper다.
# 누락/inf 검증으로 잡히지 않는 rolling window 의미 오류를 차단한다.
# -----------------------------------------------------------------------------
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


def validate_stage0_rolling_outputs(
    feature_frame: pd.DataFrame,
    *,
    min_rows_for_diversity_check: int = 1_000,
    duplicate_group_sample_size: int = 3,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """
    ML-01 Stage 0 rolling 산출물이 저장되기 전에 semantic 오류를 차단한다.

    이 검증은 기존 missing/inf 검증으로 잡히지 않는 stale import/old rolling logic 문제를 겨냥한다.
    - count/sum은 더 긴 window 값이 더 짧은 window 값보다 작아지면 실패한다.
      단 amount sum은 float32/누적합 반올림 오차 범위만 허용한다.
    - 충분히 큰 데이터에서 w1h와 w7d가 완전히 같으면 window별 결과 복제로 보고 실패한다.
    - duplicate timestamp sample에서 count feature가 동일 timestamp row를 과거로 포함하면 실패한다.
    """

    if feature_frame.empty:
        raise ValueError("Stage 0 rolling validation failed: feature_frame is empty.")
    if min_rows_for_diversity_check <= 0:
        raise ValueError("min_rows_for_diversity_check must be a positive integer.")
    if duplicate_group_sample_size < 0:
        raise ValueError("duplicate_group_sample_size must be zero or a positive integer.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    windows = ("w1h", "w6h", "w1d", "w3d", "w7d")
    monotonic_prefixes = (
        "timehist__sender__out__tx_count__count__",
        "timehist__receiver__in__tx_count__count__",
        "timehist__sender__out__amount__sum__",
        "timehist__receiver__in__amount__sum__",
    )
    diversity_pairs = (
        (
            "timehist__sender__out__tx_count__count__w1h",
            "timehist__sender__out__tx_count__count__w7d",
        ),
        (
            "timehist__receiver__in__tx_count__count__w1h",
            "timehist__receiver__in__tx_count__count__w7d",
        ),
        (
            "timehist__sender__out__amount__sum__w1h",
            "timehist__sender__out__amount__sum__w7d",
        ),
        (
            "timehist__receiver__in__amount__sum__w1h",
            "timehist__receiver__in__amount__sum__w7d",
        ),
        (
            "timehist__sender__out__amount__cur_vs_mean_ratio__w1d",
            "timehist__sender__out__amount__cur_vs_mean_ratio__w7d",
        ),
        (
            "timehist__receiver__in__amount__cur_vs_mean_ratio__w1d",
            "timehist__receiver__in__amount__cur_vs_mean_ratio__w7d",
        ),
    )
    duplicate_count_checks = (
        ("sender_account_id", "timehist__sender__out__tx_count__count__", "w1h", "1h"),
        ("sender_account_id", "timehist__sender__out__tx_count__count__", "w7d", "7d"),
        ("receiver_account_id", "timehist__receiver__in__tx_count__count__", "w1h", "1h"),
        ("receiver_account_id", "timehist__receiver__in__tx_count__count__", "w7d", "7d"),
    )

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



# =============================================================================
# 4. 기존 split metadata 검증 helper
# =============================================================================
# ML-01 build는 기존 split 컬럼을 검증하고 보존한다.
# split을 새로 만들지 않으며, train/val/test 시간 경계가 깨지면 즉시 실패한다.
# -----------------------------------------------------------------------------
def _normalize_existing_split_values(series: pd.Series, *, source_path: Path, split_col: str) -> pd.Series:
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


def _existing_split_metadata_frame(
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
            "split": _normalize_existing_split_values(df[split_col], source_path=source_path, split_col=split_col).reset_index(drop=True),
        }
    )
    _validate_unique_tx_ids(metadata)
    _validate_time_split(metadata)
    return metadata


# =============================================================================
# 5. 공개 feature build 진입점 (사용자 코드/노트북에서 직접 호출)
# =============================================================================
# 기본 권장 흐름은 단일 parquet 입력이다. DataFrame 입력은 smoke/debug용 보조 진입점으로 유지한다.
# 두 진입점은 모두 최종적으로 _build_from_split_frame()으로 수렴한다.
#
#   build_features(config)
#       └─ split 컬럼이 있는 parquet 1개에서 시작. split을 다시 만들지 않는다.
#       └─ 기존 split 포함 ML parquet에서 feature를 새로 만들 때 사용하는 기본 패턴.
#
#   build_features_from_frame(df, ...)
#       └─ split 컬럼이 있는 메모리 DataFrame에서 시작. split을 다시 만들지 않는다.
#       └─ toy 데이터로 operation 동작을 빠르게 확인할 때만 사용하는 보조 경로.
#
# -----------------------------------------------------------------------------

def build_features(config: Optional[FeatureBuildConfig] = None) -> FeatureBuildResult:
    """
    parquet 입력 파일에서 feature build를 실행

    실행 순서
    ---------
    1. 입력 parquet 존재 확인
    2. 명시적으로 전달된 FeatureSpec 목록 검증
    3. FeatureSpec과 메타 컬럼이 요구하는 logical column 목록 산출
    4. parquet schema와 매칭하여 실제 source column으로 resolve
    5. 필요한 컬럼만 parquet에서 로드 (메모리 절약 + sample_rows 지원)
    6. `_build_from_raw_frame()`으로 기존 split 검증/보존 후 공통 build 흐름 실행
    """
    # config=None이면 기본값 객체를 만들지만, feature_specs는 명시적으로 전달해야 한다.
    config = FeatureBuildConfig() if config is None else config

    # input_path 없으면 이 함수로는 진행 불가. 사용자에게 올바른 진입점 안내
    if config.input_path is None:
        raise ValueError("input_path is required for build_features(). Use build_features_from_frame() for DataFrame input.")
    # __post_init__에서 절대경로로 정규화돼 있지만, Path 객체로 한 번 더  확인
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input parquet not found: {input_path}")

    specs = _require_feature_specs(config.feature_specs)
    _validate_specs_for_build(specs)                                                           # 중복/빈 spec + 누수 위험 컬럼 차단 동시 수행.

    # meta column(tx_id/timestamp/label)과 feature operation이 요구하는 컬럼만 read해서 로드한다. 컬럼 매핑도 여기서 한 번에 수행한다.
    requested_columns = required_input_columns(
        specs,
        extra_columns=[config.tx_id_col, config.timestamp_col, config.label_col],
    )
    # parquet schema에서 실제 컬럼명 목록을 얻고, logical→source 매핑을 확정
    # column_map(사용자 명시) → COLUMN_CANDIDATES(자동 fallback) 순으로 resolve.
    source_columns = parquet_columns(input_path)
    column_map = resolve_requested_columns(source_columns, requested_columns, column_map=config.column_map)

    # ML-01 이후 산출 parquet는 원본 메타데이터/원본 컬럼을 모두 보존해야 하므로 전체 컬럼을 읽는다.
    # sample_rows가 지정되면 첫 batch만 읽어 smoke build 속도 확보
    raw_df = load_parquet_columns(input_path, source_columns, sample_rows=config.sample_rows)

    # 이후 공통 build 흐름으로 위임.
    # input_label/input_mode는 build_summary.json에 기록될 추적 정보로, 파일 경로 또는 "dataframe" 같은 식별자 문자열 포함
    return _build_from_raw_frame(
        raw_df,
        column_map=column_map,
        config=config,
        input_label=str(input_path),
        input_mode="single_parquet",
    )


def build_features_from_frame(
    df: pd.DataFrame,
    *,
    feature_specs: Optional[Tuple[FeatureSpec, ...]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    experiment_id: str = "feature_build_frame",
    run_name: str = "dataframe_input",
    column_map: Optional[Mapping[str, str]] = None,
    overwrite: bool = False,
    tx_id_col: str = "tx_id",
    timestamp_col: str = "timestamp",
    label_col: str = "label",
) -> FeatureBuildResult:
    """
    메모리 DataFrame에서 feature build를 실행하는 보조 진입점

    용도
    ----
    - 작은 toy DataFrame으로 operation 동작을 빠르게 검증할 때.
    - output_dir=None이면 파일 저장 없이 FeatureBuildResult만 반환한다. (단위 테스트, 노트북 디버깅용)

    build_features()와의 차이
    -------------------------
    - 입력이 parquet 경로가 아니라 메모리상의 DataFrame
    - parquet 컬럼 추출/로드 단계가 없고 df.columns에서 바로 resolve
    - 그 외 흐름(spec 검증 → resolve → _build_from_raw_frame)은 동일
    """
    # FeatureBuildConfig를 명시적으로 생성. input_path=None을 박아 build_features()와 구분.
    # 모든 build 파라미터를 config 객체 하나로 통일해 _build_from_raw_frame에 전달
    config = FeatureBuildConfig(
        input_path=None,
        output_dir=output_dir,
        base_dir=base_dir,
        experiment_id=experiment_id,
        run_name=run_name,
        feature_specs=feature_specs,
        column_map=column_map,
        overwrite=overwrite,
        tx_id_col=tx_id_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
    )
    specs = _require_feature_specs(config.feature_specs)
    _validate_specs_for_build(specs)
    requested_columns = required_input_columns(
        specs,
        extra_columns=[config.tx_id_col, config.timestamp_col, config.label_col],
    )
    resolved_columns = resolve_requested_columns(df.columns, requested_columns, column_map=config.column_map)
    return _build_from_raw_frame(
        df,
        column_map=resolved_columns,
        config=config,
        input_label="dataframe",
        input_mode="dataframe",
    )


# =============================================================================
# 6. 내부 FeatureSpec/build 검증 helper
# =============================================================================
# 공개 진입점이 공유하는 검증 로직을 모은 섹션이다.
# 모든 검증은 "조용히 통과시키지 않고 즉시 ValueError로 중단" 원칙
# -----------------------------------------------------------------------------
def _validate_specs_for_build(specs: Tuple[FeatureSpec, ...]) -> None:
    """
    FeatureSpec 기본 검증 + 누수 위험 input column 차단을 함께 수행한다.
    두 검증의 차이
    --------------
    - validate_feature_specs: spec 자체의 정합성 (output_col 중복, 빈 input_cols 등)
    - validate_no_forbidden_input_columns: input column 이름에 label/laundering/pattern
      같은 누수 위험 단어가 들어가면 차단 (예: is_laundering을 feature 입력으로 쓰는 실수)
    """

    validate_feature_specs(specs)
    # 모든 spec의 required_columns를 평탄화(flatten)해 한 번에 검사.
    validate_no_forbidden_input_columns(
        column for spec in specs for column in spec.required_columns()
    )


def _require_feature_specs(feature_specs: Optional[Tuple[FeatureSpec, ...]]) -> Tuple[FeatureSpec, ...]:
    """ML-01 build는 contract가 확정한 FeatureSpec 목록을 명시적으로 받아야 한다."""

    if feature_specs is None:
        raise ValueError(
            "ML-01 feature build requires explicit feature_specs. "
            "Build BUILD_FEATURE_SPECS from the fb input contract and pass it to FeatureBuildConfig."
        )
    validate_feature_specs(feature_specs)
    return feature_specs


def _validate_resolved_feature_source_columns(
    specs: Tuple[FeatureSpec, ...],
    resolved_columns: Mapping[str, str],
) -> None:
    """
    FeatureSpec 입력이 실제 source column으로 resolve된 뒤에도 누수 위험 이름을 차단한다.

    `label` 같은 metadata logical column은 feature 입력 목록에 포함하지 않는다. 따라서
    label source가 `is_laundering`인 정상 target 매핑은 허용하되, feature 입력이
    `column_map`을 통해 `Is Laundering` 같은 source column을 가리키는 경우는 차단한다.
    """

    feature_input_columns = list(
        dict.fromkeys(column for spec in specs for column in spec.required_columns())
    )
    missing = [column for column in feature_input_columns if column not in resolved_columns]
    if missing:
        raise ValueError(
            "Feature build failed: resolved source columns are missing feature inputs. "
            f"missing={missing[:30]}, missing_count={len(missing)}"
        )
    validate_no_forbidden_input_columns(resolved_columns[column] for column in feature_input_columns)


def _validate_unique_tx_ids(df: pd.DataFrame) -> None:
    """
    train/val/test 전체 합본에서 tx_id가 중복되지 않는지 확인
    """
    # 문자열로 정규화한 뒤 중복 검사 (정수/문자열 혼합 데이터로 인한 매칭 실패 방지).
    duplicated = df["tx_id"].astype("string").duplicated(keep=False)
    if duplicated.any():  # 디버깅을 돕기 위해 상위 10개 예시를 함께 에러 메시지에 포함.
        examples = df.loc[duplicated, "tx_id"].astype(str).head(10).tolist()
        raise ValueError(
            "Feature build failed: tx_id values are duplicated across split files. "
            f"duplicated_count={int(duplicated.sum())}, examples={examples}"
        )

# =============================================================================
# 7. 공통 build 본체 (공개 진입점이 최종적으로 도달하는 곳)
# =============================================================================
# _build_from_raw_frame:  split 포함 입력 → 표준화 + 기존 split 검증/보존 → _build_from_split_frame 호출
# _build_from_split_frame: split 확정 입력 → operation 실행 + catalog 생성 + 저장
# -----------------------------------------------------------------------------

def _build_from_raw_frame(
    raw_df: pd.DataFrame,
    *,
    column_map: Mapping[str, str],
    config: FeatureBuildConfig,
    input_label: str,
    input_mode: str,
) -> FeatureBuildResult:
    """
    parquet 입력과 DataFrame 입력이 공통으로 사용하는 build 전반부.

    이 함수의 역할
    -------------
    raw DataFrame을 받아:
      1) 메타 컬럼을 표준화하고,
      2) 기존 split 컬럼을 검증/보존한 뒤,
      3) split이 확정된 후속 흐름(`_build_from_split_frame`)으로 넘긴다.

    build_features() / build_features_from_frame()가 호출
    모든 공개 진입점은 기존 split 컬럼을 보존한 뒤 _build_from_split_frame으로 들어간다.
    """
    # [1] 입력 표준화: column_map을 기반으로 logical 컬럼명으로 통일 -> timestamp/label/tx_id를 strict parsing
    #  NaN, 잘못된 dtype, label 비0/1, tx_id 중복 등 확인
    clean_df = standardize_input_frame(
        raw_df,
        column_map,
        tx_id_col=config.tx_id_col,
        timestamp_col=config.timestamp_col,
        label_col=config.label_col,
    )

    source_with_meta = _preserve_source_columns(raw_df, clean_df)

    # [2] 기존 split 확정.
    # SOURCE_PARQUET_PATH의 split 컬럼만 truth source로 사용한다. split이 없으면 재분할하지 않고 중단한다.
    if "split" not in source_with_meta.columns:
        raise ValueError(
            "Feature build requires an existing split column in the input parquet/DataFrame. "
            "This ML-01 path does not create a new train/val/test split. "
            f"input={input_label}"
        )

    metadata = _existing_split_metadata_frame(
        source_with_meta,
        source_path=Path(input_label),
        tx_id_col="tx_id",
        timestamp_col="timestamp",
        label_col="label",
        split_col="split",
    )
    split_df = source_with_meta.copy(deep=False)
    split_df["tx_id"] = metadata["tx_id"]
    split_df["timestamp"] = metadata["timestamp"]
    split_df["label"] = metadata["label"]
    split_df["split"] = metadata["split"].astype("string")
    effective_input_mode = f"{input_mode}_existing_split"

    # [3] split이 확정된 후속 흐름으로 위임.
    return _build_from_split_frame(
        split_df,
        column_map=column_map,
        config=config,
        input_label=input_label,
        input_mode=effective_input_mode,
    )


def _build_from_split_frame(
    split_df: pd.DataFrame,
    *,
    column_map: Mapping[str, str],
    config: FeatureBuildConfig,
    input_label: Any,
    input_mode: str,
) -> FeatureBuildResult:
    """
    split 컬럼이 확정된 DataFrame에서 feature 계산과 저장을 수행

    공식 노트북 경로는 output_dir=None으로 이 함수의 파일 저장을 건너뛰고,
    validate_stage0_rolling_outputs() 통과 후 encode_split_frame()에서 최종 parquet와
    output contract를 저장한다. output_dir이 지정된 direct-save 모드는 보조 실행 경로다.

    호출 흐름의 종착점
    ------------------
    build_features                  ┐
    build_features_from_frame       ├─► _build_from_raw_frame ─┐
                                    │                          ├─► 여기
                                    └────────────────────────────┘

    처리 단계
    ---------
    1. FeatureSpec 확정 + 출력 경로 객체 생성
    2. (있다면) 기존 산출물 overwrite 정책 확인
    3. split이 확정된 frame을 다시 한 번 정렬 + invariant 재검증
    4. FeatureSpec operation 실행 → feature_frame / feature_info / artifacts
    5. train/val/test로 분리
    6. catalog/요약 테이블 생성
    7. build_summary.json 메타데이터 조립
    8. output_dir이 있으면 모든 산출물을 디스크에 저장
    """
    # [1] spec 확정 + 출력 경로 객체 생성.
    # output_dir이 None이면 메모리 결과만 반환하는 모드 → output_paths도 None.
    specs = _require_feature_specs(config.feature_specs)
    _validate_resolved_feature_source_columns(specs, column_map)
    output_paths = make_output_paths(config.output_dir, config.experiment_id) if config.output_dir is not None else None

    # [2] overwrite 보호: 기존 산출물이 하나라도 있으면 overwrite=False일 때 즉시 중단.
    # output_paths가 None이면 저장하지 않으므로 이 단계도 건너뛴다.
    if output_paths is not None:
        require_no_existing_outputs(output_paths, overwrite=config.overwrite)

    # [3] 정렬 + invariant 재검증.
    # 기존 split 합본에 동일한 정렬/검증 로직을 적용해 시간 순서가 train ≤ val ≤ test인지 확인
    split_df = split_df.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    _validate_time_split(split_df)

    # [4] feature 계산 실행.
    # FeatureSpec 목록에 있는 operation만 실행
    # run_name/experiment_id 같은 라벨은 어떤 feature가 만들어질지 결정하지 않는다 (그냥 추적용).
    # 반환값:
    #   feature_frame: meta(tx_id/split/label) + 생성 feature 컬럼이 합쳐진 최종 DataFrame
    #   feature_info:  컬럼별 missing/inf/분포 통계 (사람이 검토)
    #   artifacts:     category_mapping, category_unknown_summary 등 부가 산출물 dict
    generated_feature_frame, feature_info, artifacts = execute_feature_specs(split_df, specs)
    selected_feature_columns = feature_columns(specs)
    feature_frame = _append_generated_features(split_df, generated_feature_frame, selected_feature_columns)

    # [5] split 분리. _split_feature_frame은 빈 split 검사도 함께 수행.
    train_df, val_df, test_df = _split_feature_frame(feature_frame)

    # [6] catalog / 요약 테이블 생성.
    # selected_feature_columns:    모델 입력 truth source용 컬럼 목록
    # feature_columns_table:       feature_contract.csv (column_name + used_in_ml)
    # feature_catalog:             feature_catalog.csv    (사람이 검토하는 설명서)
    # split_summary:               split_summary.csv      (기간, row 수, label 분포)
    feature_columns_table = make_feature_columns_table(specs)
    feature_catalog = make_feature_catalog(specs, experiment_id=config.experiment_id, run_name=config.run_name)
    split_summary = make_split_summary(split_df)

    # [7] build_summary.json 메타데이터 조립.
    # 이 dict는 "다음 작업자(또는 미래의 자신)가 이 실행을 재현할 때 필요한 모든 것"을 담는다.
    # 입력 경로, 출력 경로, sample 여부, ratio, column_map, resolve된 컬럼, operation 목록, row count 등.
    row_counts = {
        "all": int(len(feature_frame)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }
    # category_code operation이 없거나 모든 unknown이 0이어도 안전하게 0을 반환한다.
    unknown_category_total = _unknown_category_total(artifacts.get("category_unknown_summary", pd.DataFrame()))

    build_summary: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "experiment_id": config.experiment_id,
        "run_name": config.run_name,
        "input_mode": input_mode,     # "single_parquet_existing_split" / "dataframe_existing_split" / "split_parquet"
        "input": input_label,         # str 또는 dict (split_paths 모드일 때 dict)
        "output_dir": str(output_paths.output_dir) if output_paths is not None else None,
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,
        "overwrite": config.overwrite,

        # ML-01은 기존 split만 사용하므로 split 생성 옵션은 항상 None으로 기록.
        # JSON에 None을 남겨두면 "여기서는 적용 안 됐다"는 사실이 명시적으로 보존된다.
        "train_ratio": None,
        "val_ratio": None,

        "preserve_timestamp_groups": None,

        # 사용자가 노트북에서 넘긴 매핑(configured)과 실제 resolve된 매핑(resolved)을 모두 기록.
        # 둘이 다를 수 있다 (configured에 없던 logical은 COLUMN_CANDIDATES로 채워짐).
        "configured_column_map": dict(config.column_map) if config.column_map is not None else None,
        "resolved_columns": dict(column_map),
        "feature_columns": selected_feature_columns,
        "operations": [spec.operation for spec in specs],
        "unknown_category_total": unknown_category_total,
        "row_counts": row_counts,
    }

    # [8] 디스크 저장.
    # 저장은 모든 검증/계산이 끝난 뒤 마지막에 한 번에 수행
    if output_paths is not None:
        output_paths.output_dir.mkdir(parents=True, exist_ok=True)

        # 학습 입력용 train/val/test parquet.
        save_dataframe_parquet(train_df, output_paths.train_path)
        save_dataframe_parquet(val_df, output_paths.val_path)
        save_dataframe_parquet(test_df, output_paths.test_path)

        # 사람이 검토하는 CSV 산출물 6종.
        save_dataframe_csv(feature_columns_table, output_paths.feature_columns_path)
        save_dataframe_csv(feature_catalog, output_paths.feature_catalog_path)
        save_dataframe_csv(split_summary, output_paths.split_summary_path)
        save_dataframe_csv(feature_info, output_paths.feature_info_path)
        save_dataframe_csv(artifacts["category_mapping"], output_paths.category_mapping_path)
        save_dataframe_csv(artifacts["category_unknown_summary"], output_paths.category_unknown_summary_path)

        # 저장이 끝난 뒤에야 outputs 섹션을 build_summary에 추가한
        # → JSON 안의 outputs 키 존재 여부 = "디스크 저장까지 성공했는가" 판별기준
        build_summary["outputs"] = {
            "train_path": str(output_paths.train_path),
            "val_path": str(output_paths.val_path),
            "test_path": str(output_paths.test_path),
            "feature_columns_path": str(output_paths.feature_columns_path),
            "feature_catalog_path": str(output_paths.feature_catalog_path),
            "split_summary_path": str(output_paths.split_summary_path),
            "feature_info_path": str(output_paths.feature_info_path),
            "category_mapping_path": str(output_paths.category_mapping_path),
            "category_unknown_summary_path": str(output_paths.category_unknown_summary_path),
        }
        save_json(build_summary, output_paths.build_summary_path)

    # [9] 메모리 결과 반환. output_dir=None인 경우에도 사용자는 이 객체로 결과를 받는다.
    # feature_frame/feature_info는 항상 메모리에 함께 반환되므로 노트북에서 즉시 display 가능.
    return FeatureBuildResult(
        output_paths=output_paths,
        feature_columns=selected_feature_columns,
        row_counts=row_counts,
        build_summary=build_summary,
        feature_frame=feature_frame,
        feature_info=feature_info,
    )


# =============================================================================
# 8. artifact summary helper
# =============================================================================
def _unknown_category_total(unknown_summary: pd.DataFrame) -> int:
    """
    category_unknown_summary artifact에서 전체 unknown category 건수를 합산

    용도
    ----
    build_summary.json의 "unknown_category_total" 필드에 들어가는 단일 정수.
    val/test에 처음 등장한 category(=train에서 보지 못한 값)가 얼마나 많은지를 보여주는 지표.
    0이면 안전, 크면 category 분포 shift를 의심할 신호.

    빈 DataFrame 처리
    ----------------
    category_code spec이 하나도 없으면 unknown_summary는 빈 DataFrame이다 (header만).
    이 경우 0을 반환하고, 컬럼명 검사 단계로 넘어가지 않는다.
    """

    if unknown_summary.empty:
        return 0
    # 빈 frame이 아닌데 컬럼이 없다면 ml_01_fb_operations.py의 schema가 깨진 것 → 즉시 중단.
    if "unknown_count" not in unknown_summary.columns:
        raise ValueError("category_unknown_summary artifact is missing unknown_count column.")
    return int(unknown_summary["unknown_count"].sum())
