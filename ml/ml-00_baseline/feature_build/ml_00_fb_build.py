"""
Feature build 전체 실행 모듈

이 파일의 역할
----------------
1. 기존 split 컬럼이 있는 단일 parquet를 train/val/test parquet로 분리한다.
2. parquet, split parquet 3개, 또는 DataFrame 입력에서 feature build 전체 흐름을 실행한다.
3. 입력 컬럼 resolve, strict parsing, split 확정, operation 실행을 순서대로 연결한다.
4. train/val/test parquet와 catalog/summary CSV를 저장한다.
5. 사용자가 선택한 FeatureSpec 목록을 feature 생성의 유일한 기준으로 사용한다.

중요한 설계 원칙
----------------
- Stage 이름은 feature 생성을 결정하지 않는다.
- `feature_specs` 목록만 어떤 feature가 생성될지 결정한다.
- 컬럼명이 바뀌면 노트북의 `column_map` dict를 우선 수정한다.
- full data 실행 전 sample_rows로 smoke build를 먼저 수행하는 것을 권장한다.
- overwrite=False가 기본이며 기존 산출물이 있으면 즉시 중단한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import pandas as pd

from ml_00_fb_catalog import make_feature_catalog, make_feature_columns_table, make_split_summary
from ml_00_fb_io import (
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
from ml_00_fb_operations import execute_feature_specs
from ml_00_fb_schema import (
    resolve_requested_columns,
    standardize_input_frame,
    validate_no_forbidden_input_columns,
)
from ml_00_fb_specs import (
    FeatureSpec,
    default_feature_specs,
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
    feature_specs:
        생성할 feature 선언 목록. None이면 default_feature_specs()를 사용한다.
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
    # None이면 ml_00_fb_io.resolve_path() 내부에서 ml_00_fb_utils.BASE_DIR(=Git 루트)를 사용한다.
    base_dir: Optional[Union[str, Path]] = None
    
    # --- 실행 식별자 (재현성 메타데이터 + 산출물 파일명 prefix) ---
    # experiment_id는 {experiment_id}_Xy_train.parquet 같은 파일명에 들어간다.
    # run_name은 feature_catalog.csv에 기록되어 사람이 어떤 실행이었는지 추적하는 용도.
    experiment_id: str = "feature_build"
    run_name: str = "user_selected_operations"

    # --- feature 생성 선언 ---
    # None이면 default_feature_specs()의 ML-00 기본 10개가 사용된다.
    # 노트북에서 ML00_FEATURE_SPECS / ML01_FEATURE_SPECS 등을 넘긴다.
    feature_specs: Optional[Tuple[FeatureSpec, ...]] = None
    
    # --- 컬럼명 매핑 ---
    # 노트북의 COLUMN_MAP이 그대로 들어온다. 예: {"amount": "Amount Paid"}.
    # 여기 없는 logical key는 ml_00_fb_schema.COLUMN_CANDIDATES로 자동 fallback된다.
    # 명시성을 위해 모든 key를 직접 넘기는 것을 권장.
    column_map: Optional[Mapping[str, str]] = None
    
    sample_rows: Optional[int] = None    # smoke build 옵션 
    overwrite: bool = False              # overwrite 옵션 
    
    # --- 시간순 split 비율 (랜덤 split 아님, 미래 누수 방지) ---
    # 기본 0.6/0.2/0.2.
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    # 같은 timestamp 그룹이 train/val 또는 val/test 경계에 걸쳐 있을 때, True면 그룹을 쪼개지 않고 경계를 그룹 뒤로 밀어낸다.
    # 같은 시점 거래를 서로 다른 split에 넣지 않으려는 옵션.
    preserve_timestamp_groups: bool = True
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
            
        # [5] split 비율 검증:
        # 시간순 분할이므로 train_ratio + val_ratio < 1 이어야 test 구간이 남는다.
        # 0이나 1을 허용하면 split 중 하나가 비게 되어 이후 검증이 모두 실패한다.
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1. train_ratio={self.train_ratio}")
        if not 0 < self.val_ratio < 1:
            raise ValueError(f"val_ratio must be between 0 and 1. val_ratio={self.val_ratio}")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError(
                "train_ratio + val_ratio must be less than 1. "
                f"train_ratio={self.train_ratio}, val_ratio={self.val_ratio}"
            )


@dataclass(frozen=True)
class FeatureBuildResult:
    """feature build 실행 후 사용자에게 반환되는 결과 객체다."""

    output_paths: Optional[FeatureBuildOutputPaths]
    feature_columns: list[str]
    row_counts: dict[str, int]
    build_summary: Mapping[str, Any]
    feature_frame: pd.DataFrame
    feature_info: pd.DataFrame


@dataclass(frozen=True)
class ExistingSplitOutputPaths:
    """기존 split 컬럼 기준 분할 저장 함수가 생성하는 산출물 경로다."""

    output_dir: Path
    train_path: Path
    val_path: Path
    test_path: Path
    split_summary_path: Path
    build_summary_path: Path


@dataclass(frozen=True)
class ExistingSplitResult:
    """기존 split 컬럼 기준 분할 저장 후 반환되는 결과 객체다."""

    output_paths: ExistingSplitOutputPaths
    row_counts: dict[str, int]
    split_summary: pd.DataFrame
    build_summary: Mapping[str, Any]


# -----------------------------------------------------------------------------
# 2. 시간순 train/val/test split
# -----------------------------------------------------------------------------
# 이 섹션의 함수들은 build_features() 같은 진입점이 단일 parquet 또는 DataFrame을 받았을 때만 사용됨.
#  build_features_from_split_paths() 이미 split이 결정돼 있으므로 _assign_time_split()를 호출 하지 않음.
# -----------------------------------------------------------------------------
def _boundary_after_timestamp(timestamps: pd.Series, boundary: int) -> int:
    """
    split boundary가 같은 timestamp 그룹의 중간을 자르면, 그 그룹 뒤로 boundary를 밀어준다.
        예: timestamps=[1, 2, 2, 3], boundary=2 -> boundary=3 (같은 timestamp 2 그룹 뒤로 이동)     
    """

    # preserve_timestamp_groups=True일 때만 _assign_time_split()에서 호출
    row_count = len(timestamps)

    # 경계값이 데이터 범위를 벗어나면 그대로 클램프해서 반환
    if boundary <= 0:
        return 0
    if boundary >= row_count:
        return row_count

    # boundary는 "이 인덱스부터 다음 split이 시작"을 의미
    boundary_timestamp = timestamps.iloc[boundary - 1]

    # boundary 위치의 row가 기준 timestamp와 같다면(=같은 그룹의 일부라면) 그룹이 끝날 때까지 boundary를 한 칸씩 뒤로 민다
    while boundary < row_count and timestamps.iloc[boundary] == boundary_timestamp:
        boundary += 1
    return boundary


def _assign_time_split(df: pd.DataFrame, config: FeatureBuildConfig) -> pd.DataFrame:
    """
    timestamp 기준으로 정렬한 뒤 train/val/test split 컬럼을 부여

    주의
    ----
    - 호출 전에 df는 표준화(standardize_input_frame)를 거쳐야 함. 즉 'tx_id', 'timestamp', 'label' 컬럼이 존재해야 함
    - 결과 DataFrame은 timestamp + tx_id 기준 정렬된 새 frame이다(원본 mutate 아님, 새데이터 셋 복사본).
    """

    # [1] 필수 메타 컬럼 존재 확인. standardize_input_frame()에서 보장하지만 한 번 더 체크.
    required = {config.tx_id_col, config.timestamp_col, config.label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split input is missing columns: {sorted(missing)}")

    # [2] 시간순 정렬.
    # - 1차 키: timestamp (시간순 split의 기본)
    # - 2차 키: tx_id (같은 timestamp의 row 사이에서 결정론적 순서를 보장 → 재현성)
    # - kind='mergesort': 안정 정렬. 같은 키일 때 원래 순서를 유지해 비결정성 제거.
    sorted_df = df.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    row_count = len(sorted_df)
    
    # train/val/test 각 1행은 있어야 split이 의미가 있음. (테스트 런 할때 필요 )
    if row_count < 3:
        raise ValueError(f"Need at least 3 rows to create train/val/test split. row_count={row_count}")

    # [3] 비율 → 행 인덱스 boundary 계산.
    # int()는 내림(floor)이라 약간의 비율 오차가 생기지만 test에 흡수되므로 무방.
    train_end = int(row_count * config.train_ratio)
    val_end = int(row_count * (config.train_ratio + config.val_ratio))
    
    # [4] (선택) 같은 timestamp 그룹이 경계를 가로지르면 boundary를 뒤로 민다.
    if config.preserve_timestamp_groups:
        # 같은 timestamp를 train/val/test 경계에서 쪼개지 않으려면 boundary를 뒤로 이동한다.
        train_end = _boundary_after_timestamp(sorted_df["timestamp"], train_end)
        val_end = _boundary_after_timestamp(sorted_df["timestamp"], val_end)

    # [5] boundary 정합성 검사.
    # 가능한 실패 시나리오:
    #   - train_ratio가 너무 작아 train_end=0
    #   - preserve_timestamp_groups가 boundary를 너무 많이 밀어 val이 사라짐
    #   - val_end가 데이터 끝까지 도달해 test가 비게 됨
    if train_end <= 0 or val_end <= train_end or val_end >= row_count:
        raise ValueError(
            "Invalid split boundaries after timestamp-group adjustment. "
            f"row_count={row_count}, train_end={train_end}, val_end={val_end}"
        )
        
    # [6] split 컬럼 생성.
    # 기본값 "test"로 채운 뒤 앞쪽을 train/val로 덮어쓰는 방식. dtype="string"
    split_values = pd.Series("test", index=sorted_df.index, dtype="string")
    split_values.iloc[:train_end] = "train"
    split_values.iloc[train_end:val_end] = "val"
    
    # 원본을 mutate하지 않기 위해 copy 후 컬럼 추가.
    split_df = sorted_df.copy()
    split_df["split"] = split_values
    
    # [7] 시간 순서가 진짜로 train ≤ val ≤ test인지 사후 검증. boundary 계산 로직이 바뀌어도 invariant가 깨지면 오류나옴
    _validate_time_split(split_df)
    return split_df


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
        raise ValueError(f"Missing split values after split assignment: {sorted(missing)}")

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



# =============================================================================
# 3. 기존 split 컬럼 기준 단일 parquet 분할
# =============================================================================
# 03_ml_feature_process_v2.ipynb처럼 feature가 이미 들어 있는 단일 parquet를
# train/val/test 파일로만 나눌 때 쓰는 경량 진입점이다.
# 이 경로는 feature를 재계산하지 않고, ml_feature_columns.csv도 생성하지 않는다.
# -----------------------------------------------------------------------------
def _make_existing_split_output_paths(output_dir: Union[str, Path], experiment_id: str) -> ExistingSplitOutputPaths:
    """기존 split 컬럼 기준 분할 저장 산출물 경로를 만든다."""

    base = resolve_path(output_dir)
    return ExistingSplitOutputPaths(
        output_dir=base,
        train_path=base / f"{experiment_id}_Xy_train.parquet",
        val_path=base / f"{experiment_id}_Xy_val.parquet",
        test_path=base / f"{experiment_id}_Xy_test.parquet",
        split_summary_path=base / f"{experiment_id}_split_summary.csv",
        build_summary_path=base / f"{experiment_id}_split_existing_summary.json",
    )


def _require_no_existing_split_outputs(paths: ExistingSplitOutputPaths, overwrite: bool) -> None:
    """split-only 산출물이 이미 있을 때 overwrite=False이면 중단한다."""

    protected_outputs = [
        paths.train_path,
        paths.val_path,
        paths.test_path,
        paths.split_summary_path,
        paths.build_summary_path,
    ]
    existing = [str(path) for path in protected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Existing split artifacts found. Set overwrite=True to replace them. "
            f"existing={existing}"
        )


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


def split_single_parquet_by_existing_split(
    input_path: Union[str, Path],
    *,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    base_dir: Optional[Union[str, Path]] = None,
    experiment_id: str = "feature_build",
    overwrite: bool = False,
    tx_id_col: str = "tx_id",
    timestamp_col: str = "timestamp",
    label_col: str = "label",
    split_col: str = "split",
) -> ExistingSplitResult:
    """
    기존 split 컬럼이 있는 단일 parquet를 train/val/test parquet 3개로 저장한다.

    사용 목적
    ---------
    - `03_ml_feature_process_v2.ipynb`가 만든 단일 ML-ready parquet를 `ml_*.py` 입력 계약에 맞게 나눈다.
    - feature를 새로 만들지 않고, 파일 분리와 split 검증만 수행한다.
    - `ml_feature_columns.csv`는 사용자가 실험별로 직접 관리하므로 이 함수는 생성하지 않는다.

    검증 항목
    ---------
    1. `tx_id`, `timestamp`, `label`, `split` 컬럼 존재
    2. split 값이 train/val/test 중 하나이며 세 split이 모두 존재
    3. tx_id 중복 없음
    4. train <= val <= test 시간 순서 유지
    """

    if not str(experiment_id).strip():
        raise ValueError("experiment_id must not be empty.")

    resolved_input_path = resolve_path(input_path, base_dir)
    if not resolved_input_path.exists():
        raise FileNotFoundError(f"input parquet not found: {resolved_input_path}")

    output_paths = _make_existing_split_output_paths(
        resolve_path(output_dir, base_dir),
        str(experiment_id).strip(),
    )
    _require_no_existing_split_outputs(output_paths, overwrite=overwrite)

    available_columns = set(parquet_columns(resolved_input_path))
    required = {tx_id_col, timestamp_col, label_col, split_col}
    missing = required - available_columns
    if missing:
        raise ValueError(
            "Single parquet input is missing columns required for existing split export. "
            f"path={resolved_input_path}, missing={sorted(missing)}"
        )

    # split-only export는 원본 feature 컬럼을 모두 보존해야 하므로 전체 parquet를 읽는다.
    full_df = pd.read_parquet(resolved_input_path).reset_index(drop=True)
    metadata = _existing_split_metadata_frame(
        full_df,
        source_path=resolved_input_path,
        tx_id_col=tx_id_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
        split_col=split_col,
    )

    output_df = full_df.copy()
    # 검증은 canonical metadata로 수행했으므로 저장 parquet도 같은 값을 사용한다.
    # 원본 컬럼명이 이미 표준이어도 dtype/표현 불일치를 남기지 않도록 항상 덮어쓴다.
    output_df["tx_id"] = metadata["tx_id"]
    output_df["timestamp"] = metadata["timestamp"]
    output_df["label"] = metadata["label"]
    output_df["split"] = metadata["split"].astype("string")

    split_frames = {
        "train": output_df.loc[metadata["split"] == "train"].reset_index(drop=True),
        "val": output_df.loc[metadata["split"] == "val"].reset_index(drop=True),
        "test": output_df.loc[metadata["split"] == "test"].reset_index(drop=True),
    }

    row_counts = {"all": int(len(output_df))}
    row_counts.update({split_name: int(len(split_df)) for split_name, split_df in split_frames.items()})
    if any(count == 0 for split_name, count in row_counts.items() if split_name != "all"):
        raise ValueError(f"Existing split export produced an empty split. row_counts={row_counts}")

    split_summary = make_split_summary(metadata)
    build_summary: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "experiment_id": str(experiment_id).strip(),
        "input_mode": "single_parquet_existing_split",
        "input": str(resolved_input_path),
        "output_dir": str(output_paths.output_dir),
        "feature_build_skipped": True,
        "feature_columns_created": False,
        "overwrite": overwrite,
        "source_columns": {
            "tx_id_col": tx_id_col,
            "timestamp_col": timestamp_col,
            "label_col": label_col,
            "split_col": split_col,
        },
        "row_counts": row_counts,
        "outputs": {
            "train_path": str(output_paths.train_path),
            "val_path": str(output_paths.val_path),
            "test_path": str(output_paths.test_path),
            "split_summary_path": str(output_paths.split_summary_path),
        },
    }

    output_paths.output_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe_parquet(split_frames["train"], output_paths.train_path)
    save_dataframe_parquet(split_frames["val"], output_paths.val_path)
    save_dataframe_parquet(split_frames["test"], output_paths.test_path)
    save_dataframe_csv(split_summary, output_paths.split_summary_path)
    save_json(build_summary, output_paths.build_summary_path)

    return ExistingSplitResult(
        output_paths=output_paths,
        row_counts=row_counts,
        split_summary=split_summary,
        build_summary=build_summary,
    )



# =============================================================================
# 4. 공개 feature build 진입점 (사용자 코드/노트북에서 직접 호출)
# =============================================================================
# 기본 권장 흐름은 단일 parquet 입력이다. split parquet/DataFrame 입력은 보조 진입점으로 유지한다.
# 세 가지 진입점은 모두 최종적으로 _build_from_split_frame()으로 수렴한다.
#
#   build_features(config)
#       └─ parquet 1개에서 시작. 시간순 split을 내부에서 생성.
#       └─ raw/clean_base에서 feature를 새로 만들 때 사용하는 기본 패턴.
#
#   build_features_from_frame(df, ...)
#       └─ 메모리 DataFrame에서 시작. 시간순 split을 내부에서 생성.
#       └─ toy 데이터로 operation 동작을 빠르게 확인할 때만 사용하는 보조 경로.
#
#   build_features_from_split_paths(train, val, test, ...)
#       └─ 이미 split된 parquet 3개에서 시작. split을 다시 만들지 않는다.
#       └─ 외부 파이프라인이 split 파일 3개를 이미 확정한 경우의 보조 경로.
# -----------------------------------------------------------------------------

def build_features(config: Optional[FeatureBuildConfig] = None) -> FeatureBuildResult:
    """
    parquet 입력 파일에서 feature build를 실행

    실행 순서
    ---------
    1. 입력 parquet 존재 확인
    2. FeatureSpec 목록 결정 (None이면 default 사용)
    3. FeatureSpec과 메타 컬럼이 요구하는 logical column 목록 산출
    4. parquet schema와 매칭하여 실제 source column으로 resolve
    5. 필요한 컬럼만 parquet에서 로드 (메모리 절약 + sample_rows 지원)
    6. `_build_from_raw_frame()`으로 공통 build 흐름 실행
    """
     # config=None이면 기본값 객체를 사용. 노트북에서 별도 설정 없이 한 줄 실행을 가능하게 함.
    config = FeatureBuildConfig() if config is None else config
       
    # input_path 없으면 이 함수로는 진행 불가. 사용자에게 올바른 진입점 안내
    if config.input_path is None:
        raise ValueError("input_path is required for build_features(). Use build_features_from_frame() for DataFrame input.")
    # __post_init__에서 절대경로로 정규화돼 있지만, Path 객체로 한 번 더  확인
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input parquet not found: {input_path}")

    specs = default_feature_specs() if config.feature_specs is None else config.feature_specs  # FeatureSpec 미지정 시 default(ML-00의 10개) 사용.
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
    
    # column_map.values()는 source column 이름들. 이 컬럼들만 parquet에서 로드
    # sample_rows가 지정되면 첫 batch만 읽어 smoke build 속도 확보
    raw_df = load_parquet_columns(input_path, column_map.values(), sample_rows=config.sample_rows)
    
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
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    preserve_timestamp_groups: bool = False,
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
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        preserve_timestamp_groups=preserve_timestamp_groups,
        tx_id_col=tx_id_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
    )
    specs = default_feature_specs() if config.feature_specs is None else config.feature_specs
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


def build_features_from_split_paths(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    test_path: Union[str, Path],
    *,
    feature_specs: Optional[Tuple[FeatureSpec, ...]] = None,
    output_dir: Optional[Union[str, Path]] = DEFAULT_OUTPUT_DIR,
    base_dir: Optional[Union[str, Path]] = None,
    experiment_id: str = "feature_build_split",
    run_name: str = "split_parquet_input",
    column_map: Optional[Mapping[str, str]] = None,
    sample_rows: Optional[int] = None,
    overwrite: bool = False,
    tx_id_col: str = "tx_id",
    timestamp_col: str = "timestamp",
    label_col: str = "label",
) -> FeatureBuildResult:
    """
    이미 train/val/test로 나뉜 parquet 3개에서 feature build를 실행하는 보조 진입점이다.

    핵심 차이
    ---------
    - 이 함수는 split을 **다시 만들지 않는다**. 파일별 역할(train/val/test)을 그대로 read 하고, 시간 순서만 사후 검증
    - 컬럼 resolve는 **train schema 기준 1회**만 수행, val/test는 같은 source column이 존재하는지만 확인 → 세 파일 스키마 일관성 강제

    언제 쓰나
    ---------
    - 전처리 파이프라인이 이미 split을 결정한 경우 (예: 다른 실험과 공정 비교를 위해 고정된 split을 재사용).
    - 단일 parquet → 시간순 split 흐름을 우회하고 외부 split을 신뢰할 때.

    주의
    ----
    - parquet 내부에 "split" 컬럼이 있으면 파일 역할과 일치하는지 검사 (예: train_path 파일의 모든 row가 split="train"이어야 함)
    - 합친 후 tx_id 중복이 있으면 즉시 중단. 같은 거래가 두 split에 들어가는  account-level leakage를 차단하는 1차 방어선.
    """
    # 다른 진입점과 동일하게 모든 옵션을 FeatureBuildConfig 하나로 통일.
    # input_path=None: 이 진입점은 단일 parquet가 아닌 3개 파일을 직접 받으므로 unused.
    config = FeatureBuildConfig(
        input_path=None,
        output_dir=output_dir,
        base_dir=base_dir,
        experiment_id=experiment_id,
        run_name=run_name,
        feature_specs=feature_specs,
        column_map=column_map,
        sample_rows=sample_rows,
        overwrite=overwrite,
        tx_id_col=tx_id_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
    )
    
    # [1] 세 경로를 절대경로로 정규화하고 존재 확인.
    paths = {
        "train": resolve_path(train_path, base_dir),
        "val": resolve_path(val_path, base_dir),
        "test": resolve_path(test_path, base_dir),
    }
    missing_files = {split_name: str(path) for split_name, path in paths.items() if not path.exists()}
    if missing_files:
        raise FileNotFoundError(f"split parquet files not found: {missing_files}")
    
    # [2] FeatureSpec 결정 및 검증 (다른 진입점과 동일).
    specs = default_feature_specs() if config.feature_specs is None else config.feature_specs
    _validate_specs_for_build(specs)
    requested_columns = required_input_columns(
        specs,
        extra_columns=[config.tx_id_col, config.timestamp_col, config.label_col],
    )

    # [3] train schema 기준으로 column resolve 1회 수행
    # val/test는 별도 resolve를 하지 않고, 동일 source column 존재만 확인
    resolved_columns = resolve_requested_columns(
        parquet_columns(paths["train"]),
        requested_columns,
        column_map=config.column_map,
    )
    # logical → source 매핑에서 source 값만 중복 제거하며 추출 (읽을 컬럼 목록).
    # dict.fromkeys로 입력 순서를 유지하면서 중복 제거
    source_columns = list(dict.fromkeys(resolved_columns.values()))

    # [4] 세 split 파일을 순서대로 로드 → 표준화 → 합치기.
    split_frames: list[pd.DataFrame] = []
    for split_name, path in paths.items():
        # [4-a] 스키마 일치 확인: train에서 resolve한 source column이 val/test에도 모두 있어야 함.
        available_columns = set(parquet_columns(path))
        missing_sources = [column for column in source_columns if column not in available_columns]
        if missing_sources:
            raise ValueError(
                "Split parquet is missing source columns resolved from train schema. "
                f"split={split_name!r}, path={path}, missing_sources={missing_sources}, "
                "fix=Use one consistent column_map for all split files or fix preprocessing output."
            )
        # [4-b] 읽을 컬럼 목록 구성. 파일에 'split' 컬럼이 이미 있으면 같이 읽어 검증에 사용.
        # (없으면 새로 부여하므로 굳이 로드하지 않는다 → 메모리 절약)
        read_columns = list(source_columns)
        if "split" in available_columns and "split" not in read_columns:
            read_columns.append("split")
        
        # [4-c] 컬럼 선택 로드 (+ sample_rows 지원).
        raw_df = load_parquet_columns(path, read_columns, sample_rows=config.sample_rows)
        if raw_df.empty:
            raise ValueError(f"split parquet has no rows. split={split_name!r}, path={path}")
        
        # [4-d] 파일 내 split 컬럼이 있다면 파일 역할과 일치하는지 검사. 예: train_path 파일의 split 컬럼이 전부 "train"이어야 함. 섞여 있으면 즉시 중단.
        if "split" in raw_df.columns:
            _validate_existing_split_column(raw_df["split"], expected_split=split_name, source_path=path)
            
        # [4-e] schema 표준화: tx_id/timestamp/label 메타 컬럼을 표준 이름과 표준 타입으로 정렬.
        # 이 과정에서 timestamp 파싱 실패, label 비0/1, tx_id 중복 등이 확인됨
        clean_df = standardize_input_frame(
            raw_df,
            resolved_columns,
            tx_id_col=config.tx_id_col,
            timestamp_col=config.timestamp_col,
            label_col=config.label_col,
        )
        # 파일 역할을 split 컬럼으로 새로 부여 (원본에 있던 split 컬럼은 위에서 검증용으로만 사용).
        clean_df["split"] = split_name
        split_frames.append(clean_df)

    # [5] 세 split을 하나의 DataFrame으로 CONCAT. 이후 검증과 build 흐름은 단일 parquet에서 시작한 경우와 동일하게 진행
    split_df = pd.concat(split_frames, ignore_index=True)
    
    # [6] 전체 합본에서 tx_id 중복 검사.
    _validate_unique_tx_ids(split_df)
    
    # [7] 시간순 정렬 + train≤val≤test invariant 검사.
    split_df = split_df.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    
    # [8] split이 확정된 상태에서 feature 계산 본체로 진입.
    # input_label은 dict 형태로 세 파일 경로를 모두 build_summary.json에 남긴다.
    _validate_time_split(split_df)
    return _build_from_split_frame(
        split_df,
        column_map=resolved_columns,
        config=config,
        input_label={split_name: str(path) for split_name, path in paths.items()},
        input_mode="split_parquet",
    )



# =============================================================================
# 4. 내부 검증 헬퍼
# =============================================================================
# 세 진입점이 공유하는 검증 로직을 모은 섹션
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


def _validate_existing_split_column(series: pd.Series, *, expected_split: str, source_path: Path) -> None:
    """
    split parquet 내부에 이미 들어 있는 split 컬럼이 파일 역할과 일치하는지 확인

    검사 항목
    --------
    1. split 컬럼에 결측이 없을 것
    2. 파일 내 모든 split 값이 expected_split과 동일할 것 (예: train_path 파일이라면 모든 row의 split="train")

    대소문자/공백은 정규화 후 비교한다 → " Train " 같은 표기도 허용.
    """
    # [1] 결측 차단. NaN이 섞여 있으면 어느 split에 속하는지 불명확하므로 즉시 중단.
    if series.isna().any():
        raise ValueError(
            "Split parquet has missing split values. "
            f"expected_split={expected_split!r}, path={source_path}, missing_count={int(series.isna().sum())}"
        )
        
    # [2] 정규화 후 unique 값을 expected와 비교." train "이나 "Train"도 허용하기 위해 strip + lower로 통일.
    values = set(series.astype("string").str.strip().str.lower().unique().tolist())
    expected = expected_split.lower()
    if values != {expected}:
        raise ValueError(
            "Split parquet has unexpected split values. "
            f"expected_split={expected!r}, observed_values={sorted(values)}, path={source_path}"
        )


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
# 5. 공통 build 본체 (세 진입점이 최종적으로 도달하는 곳)
# =============================================================================
# _build_from_raw_frame:  split 미생성 입력 → 표준화 + split 생성 → _build_from_split_frame 호출
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
    아직 split이 결정되지 않은 raw DataFrame을 받아:
      1) 메타 컬럼을 표준화하고,
      2) 시간순 split을 생성한 뒤,
      3) split이 확정된 후속 흐름(`_build_from_split_frame`)으로 넘긴다.

    build_features() / build_features_from_frame()가 호출
    build_features_from_split_paths()는 이 함수를 건너뛰고 곧장 _build_from_split_frame 호출
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

    # [2] 시간순 split 부여.
    # feature operation이 train/val/test 역할을 알 수 있도록 split 컬럼을 먼저 생성
    # 특히 category_code 같은 operation은 train split만 보고 인코딩을 fit하므로 필수 단계
    split_df = _assign_time_split(clean_df, config)
    
    # [3] split이 확정된 후속 흐름으로 위임.
    return _build_from_split_frame(
        split_df,
        column_map=column_map,
        config=config,
        input_label=input_label,
        input_mode=input_mode,
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
    호출 흐름의 종착점
    ------------------
    build_features                  ┐
    build_features_from_frame       ├─► _build_from_raw_frame ─┐
                                    │                          ├─► 여기
    build_features_from_split_paths ┘                          │
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
    specs = default_feature_specs() if config.feature_specs is None else config.feature_specs    
    _validate_resolved_feature_source_columns(specs, column_map)
    output_paths = make_output_paths(config.output_dir, config.experiment_id) if config.output_dir is not None else None
    
    # [2] overwrite 보호: 기존 산출물이 하나라도 있으면 overwrite=False일 때 즉시 중단. 
    # output_paths가 None이면 저장하지 않으므로 이 단계도 건너뛴다.    
    if output_paths is not None:
        require_no_existing_outputs(output_paths, overwrite=config.overwrite)
    
    # [3] 정렬 + invariant 재검증.
    # _assign_time_split을 이미 거친 데이터든, 외부 split 합본이든 동일한 정렬/검증 로직을 거쳐 시간 순서가 train ≤ val ≤ test인지 확인
    split_df = split_df.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    _validate_time_split(split_df)

    # [4] feature 계산 실행.
    # FeatureSpec 목록에 있는 operation만 실행
    # run_name/experiment_id 같은 라벨은 어떤 feature가 만들어질지 결정하지 않는다 (그냥 추적용).
    # 반환값:
    #   feature_frame: meta(tx_id/split/label) + 생성 feature 컬럼이 합쳐진 최종 DataFrame
    #   feature_info:  컬럼별 missing/inf/분포 통계 (사람이 검토)
    #   artifacts:     category_mapping, category_unknown_summary 등 부가 산출물 dict
    feature_frame, feature_info, artifacts = execute_feature_specs(split_df, specs)
    
    # [5] split 분리. _split_feature_frame은 빈 split 검사도 함께 수행.
    train_df, val_df, test_df = _split_feature_frame(feature_frame)

    # [6] catalog / 요약 테이블 생성.
    # selected_feature_columns:    모델 입력 truth source용 컬럼 목록
    # feature_columns_table:       feature_contract.csv (column_name + used_in_ml)
    # feature_catalog:             feature_catalog.csv    (사람이 검토하는 설명서)
    # split_summary:               split_summary.csv      (기간, row 수, label 분포)
    selected_feature_columns = feature_columns(specs)
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
        "input_mode": input_mode,     # "single_parquet" / "dataframe" / "split_parquet"
        "input": input_label,         # str 또는 dict (split_paths 모드일 때 dict)
        "output_dir": str(output_paths.output_dir) if output_paths is not None else None,
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,
        "overwrite": config.overwrite,
        
        # split_parquet 모드에서는 ratio가 무의미하므로 None으로 기록.
        # JSON에 None을 남겨두면 "여기서는 적용 안 됐다"는 사실이 명시적으로 보존된다.
        "train_ratio": None if input_mode == "split_parquet" else config.train_ratio,
        "val_ratio": None if input_mode == "split_parquet" else config.val_ratio,
        
        "preserve_timestamp_groups": None if input_mode == "split_parquet" else config.preserve_timestamp_groups,
        
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
    # 빈 frame이 아닌데 컬럼이 없다면 ml_00_fb_operations.py의 schema가 깨진 것 → 즉시 중단.
    if "unknown_count" not in unknown_summary.columns:
        raise ValueError("category_unknown_summary artifact is missing unknown_count column.")
    return int(unknown_summary["unknown_count"].sum())
