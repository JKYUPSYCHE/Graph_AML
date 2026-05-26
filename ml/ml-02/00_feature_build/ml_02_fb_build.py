"""
Feature build 전체 실행 모듈

이 파일의 역할
----------------
1. 기존 split 컬럼이 있는 단일 parquet 또는 DataFrame을 검증한다.
2. split 컬럼이 있는 단일 parquet 또는 DataFrame 입력에서 feature build 전체 흐름을 실행한다.
3. 입력 컬럼 resolve, strict parsing, 기존 split 검증/보존, operation 실행을 순서대로 연결한다.
4. 파일 저장 없이 메모리 결과를 반환한다. 최종 parquet/csv/json 저장은 encode_split_frame()이 담당한다.
5. 사용자가 선택한 FeatureSpec 목록을 feature 생성의 유일한 기준으로 사용한다.

중요한 설계 원칙
----------------
- Stage 이름은 feature 생성을 결정하지 않는다.
- `feature_specs` 목록만 어떤 feature가 생성될지 결정한다.
- 컬럼명이 바뀌면 노트북의 `column_map` dict를 우선 수정한다.
- full data 실행 전 sample_rows로 smoke build를 먼저 수행하는 것을 권장한다.
- build 단계는 산출물을 저장하지 않는다. overwrite 보호는 encoding/export 단계에서 처리한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import pandas as pd

import ml_02_fb_build_validation as build_validation
from ml_02_fb_build_artifacts import assemble_build_artifacts, preserve_source_columns
from ml_02_fb_io import (
    DEFAULT_INPUT_PATH,
    load_parquet_columns,
    parquet_columns,
    resolve_path,
)
from ml_02_fb_schema import (
    resolve_requested_columns,
    standardize_input_frame,
    validate_no_forbidden_input_columns,
)
from ml_02_fb_specs import (
    FeatureSpec,
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
        더 이상 build 단계에서 사용하지 않는다. 최종 저장은 encode_split_frame()이 담당한다.
    feature_specs:
        생성할 ML-02 Stage 1 feature 선언 목록. 명시하지 않으면 실행을 중단한다.
    column_map:
        노트북에서 직접 지정하는 logical column -> source column 매핑이다.
        예: {"amount": "Amount Paid"}. 지정된 값은 기본 후보보다 우선한다.
    sample_rows:
        sample smoke build용 row 수. None이면 전체 parquet를 읽는다.
    overwrite:
        build 단계에서는 사용하지 않는다. encoding/export 단계 인자와 맞추기 위해 보존한다.
    """
    # --- 입력 경로 / build 단계 미사용 출력 경로 ---
    # 상대경로로 들어와도 __post_init__에서 절대경로로 정규
    input_path: Optional[Union[str, Path]] = DEFAULT_INPUT_PATH
    output_dir: Optional[Union[str, Path]] = None
    # base_dir: input_path가 상대경로일 때 해석 기준이 되는 루트.
    # None이면 ml_02_fb_io.resolve_path() 내부에서 ml_02_fb_utils.BASE_DIR(=Git 루트)를 사용한다.
    base_dir: Optional[Union[str, Path]] = None

    # --- 실행 식별자 (재현성 메타데이터 + 산출물 파일명 prefix) ---
    # experiment_id/run_name은 build_summary에 기록되어 어떤 실행이었는지 추적하는 용도.
    experiment_id: str = "feature_build"
    run_name: str = "user_selected_operations"

    # --- feature 생성 선언 ---
    # ML-02는 contract가 확정한 BUILD_FEATURE_SPECS를 명시적으로 넘겨야 한다.
    feature_specs: Optional[Tuple[FeatureSpec, ...]] = None

    # --- 컬럼명 매핑 ---
    # 노트북의 COLUMN_MAP이 그대로 들어온다. 예: {"amount": "Amount Paid"}.
    # 여기 없는 logical key는 ml_02_fb_schema.COLUMN_CANDIDATES로 자동 fallback된다.
    # 명시성을 위해 모든 key를 직접 넘기는 것을 권장.
    column_map: Optional[Mapping[str, str]] = None

    sample_rows: Optional[int] = None    # smoke build 옵션
    overwrite: bool = False              # build 단계에서는 파일 저장을 하지 않으므로 사용하지 않음

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
            raise ValueError(
                "FeatureBuildConfig.output_dir is no longer used. "
                "Run build_features() for in-memory feature creation, then save final artifacts with encode_split_frame()."
            )

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

    output_paths: None
    feature_columns: list[str]
    row_counts: dict[str, int]
    build_summary: Mapping[str, Any]
    feature_frame: pd.DataFrame
    feature_info: pd.DataFrame


# =============================================================================
# 4. 공개 feature build 진입점 (사용자 코드/노트북에서 직접 호출)
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

    # meta column(tx_id/timestamp/label)과 feature operation이 요구하는 logical 컬럼 목록을 확정한다.
    requested_columns = required_input_columns(
        specs,
        extra_columns=[config.tx_id_col, config.timestamp_col, config.label_col],
    )
    # parquet schema에서 실제 컬럼명 목록을 얻고, logical→source 매핑을 확정
    # column_map(사용자 명시) → COLUMN_CANDIDATES(자동 fallback) 순으로 resolve.
    source_columns = parquet_columns(input_path)
    column_map = resolve_requested_columns(source_columns, requested_columns, column_map=config.column_map)

    # ML-02 이후 산출 parquet는 ML-01 승인본의 기존 컬럼과 feature를 보존해야 하므로 전체 컬럼을 읽는다.
    # 대용량 입력에서는 sample_rows smoke build를 먼저 실행하고, full build 메모리 사용량을 별도로 확인한다.
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
    - build 단계는 항상 파일 저장 없이 FeatureBuildResult만 반환한다. (단위 테스트, 노트북 디버깅용)

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
# 4. 내부 검증 헬퍼
# =============================================================================
# 공개 진입점이 공유하는 검증 로직을 모은 섹션
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
    """ML-02 build는 contract가 확정한 FeatureSpec 목록을 명시적으로 받아야 한다."""

    if feature_specs is None:
        raise ValueError(
            "ML-02 feature build requires explicit feature_specs. "
            "Build BUILD_FEATURE_SPECS from the fb input contract and pass it to FeatureBuildConfig."
        )
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


# =============================================================================
# 5. 공통 build 본체 (공개 진입점이 최종적으로 도달하는 곳)
# =============================================================================
# _build_from_raw_frame:  split 포함 입력 → 표준화 + 기존 split 검증/보존 → _build_from_split_frame 호출
# _build_from_split_frame: split 확정 입력 → operation 실행 + 메모리 결과 반환
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

    source_with_meta = preserve_source_columns(raw_df, clean_df)

    # [2] 기존 split 확정.
    # SOURCE_PARQUET_PATH의 split 컬럼만 truth source로 사용한다. split이 없으면 재분할하지 않고 중단한다.
    if "split" not in source_with_meta.columns:
        raise ValueError(
            "Feature build requires an existing split column in the input parquet/DataFrame. "
            "This ML-02 path does not create a new train/val/test split. "
            f"input={input_label}"
        )

    metadata = build_validation.existing_split_metadata_frame(
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
    split 컬럼이 확정된 DataFrame에서 feature 계산을 수행한다.

    build 단계는 파일 저장 없이 메모리 결과만 반환한다.
    최종 parquet/csv/json 저장은 encode_split_frame()이 담당한다.
    """
    specs = _require_feature_specs(config.feature_specs)
    _validate_resolved_feature_source_columns(specs, column_map)

    split_df = split_df.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    build_validation.validate_time_split(split_df)

    build_artifacts = assemble_build_artifacts(
        split_df,
        specs=specs,
        config=config,
        column_map=column_map,
        input_label=input_label,
        input_mode=input_mode,
    )

    return FeatureBuildResult(
        output_paths=None,
        feature_columns=build_artifacts.selected_feature_columns,
        row_counts=build_artifacts.row_counts,
        build_summary=build_artifacts.build_summary,
        feature_frame=build_artifacts.feature_frame,
        feature_info=build_artifacts.feature_info,
    )
