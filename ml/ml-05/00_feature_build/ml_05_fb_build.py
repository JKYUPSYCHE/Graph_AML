"""ML-05 피처 생성 진입점.

이 모듈은 split 정보가 포함된 ML-04 피처 parquet 또는 DataFrame을 읽고,
기존 split을 검증한 뒤 ML-05 Stage 4 flow-balance/pass-through 피처만 추가한다.
파일을 직접 저장하지 않으며, export는 ``ml_05_fb_encoding``에서 처리한다.

코드 맵:
- 입력           : split 정보가 포함된 ML-04 parquet 또는 메모리상의 DataFrame.
- 출력           : 메모리상의 feature_frame과 메타데이터를 포함한 FeatureBuildResult.
- 공개 객체/함수 : FeatureBuildConfig, FeatureBuildResult, build_features, build_features_from_frame.
- 누수 방지      : 기존 시간 기준 split을 검증하고, build 전에 timestamp/tx_id 기준으로 정렬한다.
- 참고           : 여기서는 output_dir을 허용하지 않으며, encode_split_frame()만 저장 단계를 담당한다.
"""

from __future__ import annotations  # 타입 힌트 지연 평가를 활성화한다.

from dataclasses import dataclass                        # 설정/결과 객체를 dataclass로 정의하기 위해 사용한다.
from pathlib import Path                                 # 입력 parquet 경로 처리에 사용한다.
from typing import Any, Mapping, Optional, Tuple, Union  # 설정값과 타입 힌트에 사용한다.

import pandas as pd  

import ml_05_fb_build_validation as build_validation                                   # 기존 split, tx_id, 시간 순서 검증 함수 모듈.
from ml_05_fb_build_artifacts import assemble_build_artifacts, preserve_source_columns # build 산출물 조립과 원본 컬럼 보존 함수.
# parquet 입력 로드와 경로 처리 유틸.
from ml_05_fb_io import DEFAULT_INPUT_PATH, load_parquet_columns, load_parquet_split_sample, parquet_columns, resolve_path 
from ml_05_fb_operations import SUPPORTED_BATCH_OPERATIONS  # ML-05에서 batch 실행 가능한 operation 목록.
from ml_05_fb_schema import resolve_requested_columns, standardize_input_frame, validate_no_forbidden_input_columns  # 컬럼 매핑, 표준화, 금지 컬럼 검증 함수.
from ml_05_fb_specs import FeatureSpec, ml05_stage4_feature_specs, required_input_columns, validate_feature_specs    # 피처 스펙 정의와 검증 함수.


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Feature build execution config."""

    input_path: Optional[Union[str, Path]] = DEFAULT_INPUT_PATH  # 기본 입력 parquet 경로.
    output_dir: Optional[Union[str, Path]] = None                # 현재 build 단계에서는 저장 경로를 사용하지 않는다.
    base_dir: Optional[Union[str, Path]] = None                  # 상대 경로 해석 기준 디렉터리.
    experiment_id: str = "ML-05"                             # 실험 ID.
    run_name: str = "stage4_flowbalance_passflow_full48"     # 실행 run 이름.
    feature_specs: Optional[Tuple[FeatureSpec, ...]] = None  # 직접 지정할 FeatureSpec 목록.
    column_map: Optional[Mapping[str, str]] = None           # 논리 컬럼명과 실제 입력 컬럼명의 매핑.
    sample_rows: Optional[int] = None                        # split별 샘플 row 수, None이면 전체 로드.
    overwrite: bool = False                                  # downstream summary에 기록되는 overwrite 설정.
    preserve_source_columns: bool = True                     # 원본 컬럼을 최종 feature frame에 보존할지 여부.
    tx_id_col: str = "tx_id"          # 거래 ID 컬럼명.
    timestamp_col: str = "timestamp"  # 거래 시각 컬럼명.
    label_col: str = "label"          # 라벨 컬럼명.

    def __post_init__(self) -> None:
        if self.input_path is not None:  # parquet 입력 경로가 있으면 실제 Path로 정규화한다.
            object.__setattr__(self, "input_path", resolve_path(self.input_path, self.base_dir))
        if self.output_dir is not None:  # 이 단계에서 직접 저장하는 동작을 막는다.
            raise ValueError(
                "FeatureBuildConfig.output_dir is no longer used. "
                "Run build_features() for in-memory feature creation, then save final artifacts with encode_split_frame()."
            )
        if self.sample_rows is not None and self.sample_rows <= 0:  # 샘플 row 수는 양수만 허용한다.
            raise ValueError("sample_rows must be a positive integer or None.")
        if not str(self.experiment_id).strip():  # 빈 experiment_id를 막는다.
            raise ValueError("experiment_id must not be empty.")
        if not str(self.run_name).strip():  # 빈 run_name을 막는다.
            raise ValueError("run_name must not be empty.")

        if self.column_map is not None:              # 사용자 지정 컬럼 매핑이 있으면 정리/검증한다.
            cleaned_column_map: dict[str, str] = {}  # 공백 제거 후 저장할 컬럼 매핑.
            for logical_name, source_column in self.column_map.items():  # 논리명과 실제 원본 컬럼명을 하나씩 확인한다.
                logical = str(logical_name).strip()  # 논리 컬럼명의 앞뒤 공백을 제거한다.
                source = str(source_column).strip()  # 실제 원본 컬럼명의 앞뒤 공백을 제거한다.
                if not logical or not source:        # key/value 중 하나라도 비어 있으면 매핑이 불가능하다.
                    raise ValueError(
                        "column_map keys and values must not be blank. "
                        f"logical_name={logical_name!r}, source_column={source_column!r}"
                    )
                if logical in cleaned_column_map:     # 공백 제거 후 논리명이 중복되면 모호하므로 실패시킨다.
                    raise ValueError(f"column_map has duplicated logical name after stripping: {logical!r}")
                cleaned_column_map[logical] = source  # 정리된 매핑을 저장한다.
            object.__setattr__(self, "column_map", cleaned_column_map)  # frozen dataclass 내부 값을 검증된 매핑으로 교체한다.


@dataclass(frozen=True)
class FeatureBuildResult:
    """In-memory feature build result."""

    output_paths: None                # 이 단계는 파일을 저장하지 않으므로 항상 None이다.
    feature_columns: list[str]        # 이번 build에서 생성한 ML-05 피처 컬럼 목록.
    row_counts: dict[str, int]        # 전체/train/val/test row 수 요약.
    build_summary: Mapping[str, Any]  # 재현성 확인용 build 설정과 메타데이터.
    feature_frame: pd.DataFrame       # 생성 피처가 붙은 전체 feature frame.
    feature_info: pd.DataFrame        # 생성 피처별 설명/품질 메타정보.

def build_features(config: Optional[FeatureBuildConfig] = None) -> FeatureBuildResult:
    """Build ML-05 Stage 4 features from a split-aware parquet input without saving files."""

    config = FeatureBuildConfig() if config is None else config  # 설정이 없으면 기본 설정을 사용한다.
    if config.input_path is None:                                # parquet 입력 함수이므로 input_path가 반드시 필요하다.
        raise ValueError("input_path is required for build_features(). Use build_features_from_frame() for DataFrame input.")
    input_path = Path(config.input_path)  # 입력 경로를 Path 객체로 변환한다.
    if not input_path.exists():           # 입력 parquet 파일 존재 여부를 확인한다.
        raise FileNotFoundError(f"input parquet not found: {input_path}")

    specs = _require_feature_specs(config.feature_specs)  # 명시된 스펙이 없으면 ML-05 기본 Stage 4 스펙을 사용한다.
    _validate_specs_for_build(specs)                      # 스펙 구조와 지원 operation 여부를 검증한다.
    requested_columns = required_input_columns(           # 피처 생성과 메타 검증에 필요한 입력 컬럼 목록을 만든다.
        specs,
        extra_columns=[config.tx_id_col, config.timestamp_col, config.label_col, "split"],
    )
    source_columns = parquet_columns(input_path)  # parquet 파일의 실제 컬럼 목록을 읽는다.
    column_map = resolve_requested_columns(source_columns, requested_columns, column_map=config.column_map)  # 필요한 컬럼을 실제 원본 컬럼명으로 resolve한다.
    load_columns = _parquet_columns_for_build(  # 원본 보존 여부에 따라 parquet에서 읽을 컬럼을 결정한다.
        source_columns,
        requested_columns=requested_columns,
        column_map=column_map,
        preserve_source_columns=config.preserve_source_columns,
    )

    sample_scan_summary: dict[str, Any] | None = None  # 샘플 로드 시 scan 요약을 저장할 변수.
    if config.sample_rows is None:                     # sample_rows가 없으면 전체 데이터를 읽는다.
        raw_df = load_parquet_columns(input_path, load_columns, sample_rows=None)
    else:  # sample_rows가 있으면 split별 샘플만 읽는다.
        sample_scan_summary = {}  # split별 샘플 scan 결과를 채울 dict.
        raw_df = load_parquet_split_sample(
            input_path,
            load_columns,
            sample_rows=config.sample_rows,
            split_col="split",
            scan_summary=sample_scan_summary,
        )
    return _build_from_raw_frame(  # 로드된 원본 frame을 표준화하고 실제 피처 생성을 수행한다.
        raw_df,
        column_map=column_map,
        config=config,
        input_label=str(input_path),
        input_mode="single_parquet" if config.preserve_source_columns else "single_parquet_minimal_columns",
        sample_scan_summary=sample_scan_summary,
    )


def build_features_from_frame(
    df: pd.DataFrame,
    *,
    feature_specs: Optional[Tuple[FeatureSpec, ...]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    experiment_id: str = "ML-05",
    run_name: str = "stage4_flowbalance_passflow_full48",
    column_map: Optional[Mapping[str, str]] = None,
    overwrite: bool = False,
    tx_id_col: str = "tx_id",
    timestamp_col: str = "timestamp",
    label_col: str = "label",
) -> FeatureBuildResult:
    """Build ML-05 Stage 4 features from an in-memory split-aware DataFrame without saving files."""

    config = FeatureBuildConfig(  # DataFrame 입력용 build 설정 객체를 만든다.
        input_path=None,          # 파일 경로 입력이 아니라 메모리 DataFrame 입력임을 표시한다.
        output_dir=output_dir,    # 현재 build 단계에서는 저장하지 않으며, 값이 있으면 config에서 예외가 난다.
        base_dir=base_dir,        # 상대 경로 해석 기준 디렉터리.
        experiment_id=experiment_id,  
        run_name=run_name,  
        feature_specs=feature_specs,  # 사용자가 지정한 피처 스펙, 없으면 기본 ML-05 스펙을 쓴다.
        column_map=column_map,        # 논리 컬럼명과 실제 입력 컬럼명의 매핑.
        overwrite=overwrite,          # overwrite 설정을 summary에 기록하기 위한 값.
        tx_id_col=tx_id_col,          # 거래 ID 컬럼명.
        timestamp_col=timestamp_col,  # 거래 시각 컬럼명.
        label_col=label_col,          # 라벨 컬럼명.
    )
    specs = _require_feature_specs(config.feature_specs)  # 명시된 스펙이 없으면 ML-05 기본 Stage 4 스펙을 가져온다.
    _validate_specs_for_build(specs)                      # 스펙 구조, 지원 operation, 금지 입력 컬럼을 검증한다.
    requested_columns = required_input_columns(           # 피처 생성과 메타 검증에 필요한 입력 컬럼 목록을 만든다.
        specs,
        extra_columns=[config.tx_id_col, config.timestamp_col, config.label_col, "split"],
    )
    resolved_columns = resolve_requested_columns(df.columns, requested_columns, column_map=config.column_map)
    return _build_from_raw_frame(  # 표준화, split 검증, 피처 생성을 공통 내부 함수에 맡긴다.
        df,
        column_map=resolved_columns,
        config=config,
        input_label="dataframe",  # summary에 기록할 입력 식별자.
        input_mode="dataframe",   # summary에 기록할 입력 방식.
        sample_scan_summary=None, # DataFrame 입력은 parquet 샘플 scan 요약이 없다.
    )

def _validate_specs_for_build(specs: Tuple[FeatureSpec, ...]) -> None:
    """Validate specs and block feature inputs that look label-derived."""

    validate_feature_specs(specs)  # FeatureSpec 자체의 필수 값과 구조를 검증한다.
    unsupported_specs = [          # ML-05 batch build에서 지원하지 않는 operation을 가진 스펙을 모은다.
        {"output_col": spec.output_col, "operation": spec.operation}
        for spec in specs
        if spec.operation not in SUPPORTED_BATCH_OPERATIONS
    ]
    if unsupported_specs:  # 지원하지 않는 operation이 있으면 build를 중단한다.
        raise ValueError(
            "Feature build failed: unsupported ML-05 feature operation. "
            f"unsupported_specs={unsupported_specs[:30]}, unsupported_count={len(unsupported_specs)}, "
            f"supported_operations={sorted(SUPPORTED_BATCH_OPERATIONS)}"
        )
    validate_no_forbidden_input_columns(column for spec in specs for column in spec.required_columns())  # label/typology처럼 누수 위험이 있는 입력 컬럼명을 차단한다.


def _require_feature_specs(
    feature_specs: Optional[Tuple[FeatureSpec, ...]],
) -> Tuple[FeatureSpec, ...]:
    """Return explicit specs or the fixed ML-05 Stage 4 specs."""

    if feature_specs is None:   # 별도 스펙이 없으면 고정된 ML-05 Stage 4 기본 스펙을 사용한다.
        return ml05_stage4_feature_specs()
    return feature_specs        # 사용자가 넘긴 명시적 스펙을 그대로 사용한다.


def _validate_resolved_feature_source_columns(
    specs: Tuple[FeatureSpec, ...],
    resolved_columns: Mapping[str, str],
) -> None:
    """Validate resolved source columns for feature inputs only."""

    feature_input_columns = list(dict.fromkeys(column for spec in specs for column in spec.required_columns()))  # 스펙들이 요구하는 입력 컬럼을 중복 제거해 모은다.
    missing = [column for column in feature_input_columns if column not in resolved_columns]  # resolve 결과에 없는 필수 피처 입력 컬럼을 찾는다.
    if missing:  # 필요한 입력 컬럼 매핑이 없으면 피처를 만들 수 없다.
        raise ValueError(
            "Feature build failed: resolved source columns are missing feature inputs. "
            f"missing={missing[:30]}, missing_count={len(missing)}"
        )
    validate_no_forbidden_input_columns(resolved_columns[column] for column in feature_input_columns)  # 실제 원본 컬럼명 기준으로도 누수 위험 컬럼명을 차단한다.


def _parquet_columns_for_build(
    source_columns: list[str],
    *,
    requested_columns: list[str],
    column_map: Mapping[str, str],
    preserve_source_columns: bool,
) -> list[str]:
    """Return parquet columns needed by the current build mode."""

    if preserve_source_columns:  # 원본 컬럼 보존 모드면 parquet의 모든 컬럼을 읽는다.
        return source_columns
    return list(dict.fromkeys(column_map[column] for column in requested_columns))  # 최소 로드 모드면 필요한 컬럼만 중복 제거해 읽는다.


def _build_from_raw_frame(
    raw_df: pd.DataFrame,
    *,
    column_map: Mapping[str, str],
    config: FeatureBuildConfig,
    input_label: str,
    input_mode: str,
    sample_scan_summary: Mapping[str, Any] | None,
) -> FeatureBuildResult:
    """Standardize metadata and validate the existing split before computing features."""

    clean_df = standardize_input_frame(  # 입력 컬럼을 tx_id/timestamp/label 등 표준 컬럼명과 타입으로 정리한다.
        raw_df,
        column_map,
        tx_id_col=config.tx_id_col,
        timestamp_col=config.timestamp_col,
        label_col=config.label_col,
    )
    if config.preserve_source_columns:  # 원본 컬럼 보존 설정이면 raw_df에 표준 메타 컬럼을 추가한다.
        source_with_meta = preserve_source_columns(raw_df, clean_df)
    else:  # 원본 컬럼 보존이 아니면 표준화된 최소 컬럼만 사용한다.
        source_with_meta = clean_df.reset_index(drop=True).copy(deep=False)
    if "split" not in source_with_meta.columns:  # ML-05는 기존 split을 사용하므로 split 컬럼이 반드시 필요하다.
        raise ValueError(
            "Feature build requires an existing split column in the input parquet/DataFrame. "
            "This ML-05 path does not create a new train/val/test split. "
            f"input={input_label}"
        )

    metadata = build_validation.existing_split_metadata_frame(  # tx_id/timestamp/label/split 메타데이터를 검증하고 표준화한다.
        source_with_meta,
        source_path=Path(input_label),
        tx_id_col="tx_id",
        timestamp_col="timestamp",
        label_col="label",
        split_col="split",
    )
    split_df = source_with_meta.copy(deep=False)   # 표준 메타데이터를 반영할 작업용 DataFrame을 만든다.
    split_df["tx_id"] = metadata["tx_id"]          # 검증된 표준 tx_id로 교체한다.
    split_df["timestamp"] = metadata["timestamp"]  # 검증/파싱된 표준 timestamp로 교체한다.
    split_df["label"] = metadata["label"]          # 검증된 이진 label로 교체한다.
    split_df["split"] = metadata["split"].astype("string")  # 정규화된 split 값을 문자열 타입으로 고정한다.
    effective_input_mode = f"{input_mode}_existing_split"   # 기존 split을 사용했다는 정보를 input_mode에 반영한다.
    return _build_from_split_frame(                         # 검증된 split frame으로 실제 ML-05 피처를 생성한다.
        split_df,
        column_map=column_map,
        config=config,
        input_label=input_label,
        input_mode=effective_input_mode,
        sample_scan_summary=sample_scan_summary,
    )

def _build_from_split_frame(
    split_df: pd.DataFrame,
    *,
    column_map: Mapping[str, str],
    config: FeatureBuildConfig,
    input_label: Any,
    input_mode: str,
    sample_scan_summary: Mapping[str, Any] | None,
) -> FeatureBuildResult:
    """Compute ML-05 Stage 4 features from a validated split frame."""

    specs = _require_feature_specs(config.feature_specs)          # 사용할 FeatureSpec 목록을 확정한다.
    _validate_resolved_feature_source_columns(specs, column_map)  # 피처 입력 컬럼들이 실제 원본 컬럼으로 resolve됐는지 검증한다.
    split_df = split_df.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)  # 시간순 처리와 동일 timestamp 안정 정렬을 위해 정렬한다.
    build_validation.validate_time_split(split_df)  # 정렬 후에도 train < val < test 시간 경계가 유지되는지 재검증한다.
    build_artifacts = assemble_build_artifacts(     # 피처 생성, split 분리, summary 생성을 한 번에 조립한다.
        split_df,
        specs=specs,
        config=config,
        column_map=column_map,
        input_label=input_label,
        input_mode=input_mode,
        sample_scan_summary=sample_scan_summary,
    )
    return FeatureBuildResult(  # 외부 호출자에게 필요한 in-memory build 결과만 반환한다.
        output_paths=None,      # 이 단계는 파일을 저장하지 않으므로 None이다.
        feature_columns=build_artifacts.selected_feature_columns,  # 새로 생성된 ML-05 피처 컬럼 목록.
        row_counts=build_artifacts.row_counts,        # 전체/train/val/test row 수 요약.
        build_summary=build_artifacts.build_summary,  # 재현성 확인용 build summary.
        feature_frame=build_artifacts.feature_frame,  # 원본 컬럼과 생성 피처가 결합된 전체 frame.
        feature_info=build_artifacts.feature_info,    # 생성 피처별 설명과 품질 메타정보.
    )
