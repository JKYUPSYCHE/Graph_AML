"""ML-05 피처 생성 산출물 조립 헬퍼.

ML-05의 입력은 ML-04 피처 parquet 파일이라고 가정한다.
이 모듈은 모든 입력 컬럼을 보존하고, 새로 생성된 Stage 4 피처만 추가한다.

코드 맵:
- 입력      : 검증된 split_df, FeatureSpec 튜플, build 설정 메타데이터.
- 출력      : 추가 피처가 붙은 feature_frame, split별 frame, summary를 포함한 BuildArtifacts.
- 공개 함수 : assemble_build_artifacts, append_generated_features, split_feature_frame.
- 누수 방지 : 기존 시간 기준 split을 보존하고, 생성 피처가 기존 컬럼을 덮어쓰지 못하게 한다.
- 참고      : build 산출물은 메모리에 유지되며, encode_split_frame()이 최종 파일을 저장한다.
"""

from __future__ import annotations  # 타입 힌트 지연 평가로 순환 참조와 런타임 비용을 줄인다.

from dataclasses import dataclass       # 산출물 묶음을 불변 데이터 클래스로 정의하기 위해 사용한다.
from typing import Any, Mapping, Tuple  # 설정 객체와 읽기 전용 매핑 타입 힌트에 사용한다.

import pandas as pd  

from ml_05_fb_io import utc_now_iso                      # build summary 생성 시 UTC 생성 시각을 기록한다.
from ml_05_fb_operations import execute_feature_specs    # FeatureSpec 목록을 실제 피처 생성 로직으로 실행한다.
from ml_05_fb_specs import FeatureSpec, feature_columns  # 피처 스펙 타입과 생성될 컬럼명 추출 함수.

@dataclass(frozen=True)
class BuildArtifacts:
    """Internal artifact bundle returned by build assembly."""

    feature_frame: pd.DataFrame      # 원본 split_df에 ML-05 생성 피처를 붙인 전체 데이터.
    train_df: pd.DataFrame           # feature_frame 중 train split만 분리한 데이터.
    val_df: pd.DataFrame             # feature_frame 중 validation split만 분리한 데이터.
    test_df: pd.DataFrame            # feature_frame 중 test split만 분리한 데이터.
    feature_info: pd.DataFrame       # 생성된 피처별 메타정보.
    operation_artifacts: dict[str, pd.DataFrame]  # 연산별 부가 산출물 테이블 묶음.
    selected_feature_columns: list[str]           # 이번 build에서 새로 생성한 피처 컬럼명 목록.
    row_counts: dict[str, int]                    # all/train/val/test row count 메타데이터.
    build_summary: dict[str, Any]                 # 재현성 확인용 build 설정과 요약 정보.



def split_feature_frame(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the final feature frame into train/val/test frames."""

    train_df = feature_frame[feature_frame["split"] == "train"].reset_index(drop=True)  # train split만 추출하고 index를 재정렬한다.
    val_df = feature_frame[feature_frame["split"] == "val"].reset_index(drop=True)      # val split만 추출하고 index를 재정렬한다.
    test_df = feature_frame[feature_frame["split"] == "test"].reset_index(drop=True)    # test split만 추출하고 index를 재정렬한다.
    if train_df.empty or val_df.empty or test_df.empty:                                 # 세 split 중 하나라도 비면 실패시킨다.
        raise ValueError(
            "Feature split output must not be empty. "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    return train_df, val_df, test_df                                                    # 분리된 train/val/test DataFrame을 반환한다.


def preserve_source_columns(raw_df: pd.DataFrame, standardized_df: pd.DataFrame) -> pd.DataFrame:
    """Preserve all source columns and append canonical build columns."""

    output = raw_df.reset_index(drop=True).copy(deep=False)              # 원본 컬럼을 유지한 복사본을 만든다.
    for column in standardized_df.columns:                               # 표준화된 build 컬럼을 하나씩 순회한다.
        output[column] = standardized_df[column].reset_index(drop=True)  # row 순서를 맞춘 뒤 표준화 컬럼을 덮어쓴다.
    return output                                                        # 원본 컬럼과 표준화 컬럼을 모두 가진 DataFrame을 반환한다.


def append_generated_features(
    source_frame: pd.DataFrame,
    built_feature_frame: pd.DataFrame,
    generated_columns: list[str],
) -> pd.DataFrame:
    """Append generated ML-05 feature columns without overwriting inputs."""

    source = source_frame.reset_index(drop=True).copy(deep=False)  # 입력 frame의 row 순서를 고정하고 복사본을 만든다.
    built = built_feature_frame.reset_index(drop=True)             # 생성 피처 frame도 같은 기준으로 index를 재정렬한다.
    if len(source) != len(built):                                  # 원본과 생성 결과 row 수가 다르면 중단한다.
        raise ValueError(
            "Feature build failed: source and generated feature row counts differ. "
            f"source_rows={len(source)}, generated_rows={len(built)}"
        )

    for meta_col in ["tx_id", "timestamp", "split", "label"]:                # row 정합성 검증에 필요한 핵심 메타 컬럼.
        if meta_col not in source.columns or meta_col not in built.columns:  # 어느 한쪽에 메타 컬럼이 없으면 비교할 수 없다.
            raise ValueError(f"Feature build failed: metadata column is missing before append. column={meta_col!r}")
        left = source[meta_col].astype("string").reset_index(drop=True)     # 원본 메타 컬럼을 문자열 기준으로 정규화한다.
        right = built[meta_col].astype("string").reset_index(drop=True)     # 생성 결과 메타 컬럼도 같은 기준으로 정규화한다.
        if not left.equals(right):                                          # 값 또는 순서가 다르면 중단한다.
            raise ValueError(f"Feature build failed: metadata order mismatch before append. column={meta_col!r}")

    collisions = [column for column in generated_columns if column in source.columns]  # 새 피처명이 기존 컬럼과 충돌하는지 확인한다.
    if collisions:  # 기존 입력 컬럼 overwrite를 방지한다.
        raise ValueError(
            "Feature build refused to overwrite existing columns. "
            f"generated_columns_already_exist={collisions[:30]}, collision_count={len(collisions)}"
        )

    for column in generated_columns:                     # 검증된 생성 피처 컬럼만 원본 frame에 추가한다.
        if column not in built.columns:                  # 스펙상 생성되어야 할 컬럼이 실제 결과에 없으면 실패시킨다.
            raise ValueError(f"Feature build failed: generated column is missing. column={column!r}")
        source[column] = built[column]                # 원본 row 순서와 검증된 생성 피처를 결합한다.
    source["label"] = built["label"].astype("int8")   # label dtype을 ML 입력에 맞게 int8로 고정한다.
    source["split"] = built["split"].astype("string") # split dtype을 문자열 타입으로 고정한다.
    return source                                     # 원본 컬럼과 신규 피처를 함께 가진 최종 feature frame을 반환한다.


def assemble_build_artifacts(
    split_df: pd.DataFrame,
    *,
    specs: Tuple[FeatureSpec, ...],
    config: Any,
    column_map: Mapping[str, str],
    input_label: Any,
    input_mode: str,
    sample_scan_summary: Mapping[str, Any] | None = None,
) -> BuildArtifacts:
    """Assemble generated feature frame, summaries, and operation artifacts."""

    generated_feature_frame, feature_info, operation_artifacts = execute_feature_specs(split_df, specs)    # 스펙 기반으로 피처와 부가 산출물을 생성한다.
    selected_feature_columns = feature_columns(specs)                                                      # 이번 스펙에서 생성될 피처 컬럼명만 추출한다.
    feature_frame = append_generated_features(split_df, generated_feature_frame, selected_feature_columns) # 원본 split_df에 신규 피처를 안전하게 붙인다.
    train_df, val_df, test_df = split_feature_frame(feature_frame)               # 최종 feature frame을 train/val/test로 다시 분리한다.
    row_counts = feature_row_counts(feature_frame, train_df, val_df, test_df)    # split별 row 수를 재현성 메타데이터로 만든다.
    build_summary = make_build_summary(                                          # 실행 설정과 생성 결과 요약을 하나의 dict로 만든다.
        config=config,
        column_map=column_map,
        input_label=input_label,
        input_mode=input_mode,
        sample_scan_summary=sample_scan_summary,
        selected_feature_columns=selected_feature_columns,
        specs=specs,
        operation_artifacts=operation_artifacts,
        row_counts=row_counts,
    )
    return BuildArtifacts(                  # 이후 저장/인코딩 단계에서 사용할 산출물 묶음을 반환한다.
        feature_frame=feature_frame,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_info=feature_info,
        operation_artifacts=operation_artifacts,
        selected_feature_columns=selected_feature_columns,
        row_counts=row_counts,
        build_summary=build_summary,
    )


def feature_row_counts(
    feature_frame: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    """Create shared row-count metadata."""

    return {
        "all": int(len(feature_frame)), # 전체 feature frame의 row 수를 기록한다.
        "train": int(len(train_df)),    # train split row 수를 기록한다.
        "val": int(len(val_df)),        # validation split row 수를 기록한다.
        "test": int(len(test_df)),      # test split row 수를 기록한다.
    }


def make_build_summary(
    *,
    config: Any,
    column_map: Mapping[str, str],
    input_label: Any,
    input_mode: str,
    sample_scan_summary: Mapping[str, Any] | None,
    selected_feature_columns: list[str],
    specs: Tuple[FeatureSpec, ...],
    operation_artifacts: Mapping[str, pd.DataFrame],
    row_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Create reproducibility metadata for the build result."""

    unknown_category_total = unknown_category_total_from_summary(operation_artifacts.get("category_unknown_summary", pd.DataFrame()))  # category encoding 과정에서 발생한 unknown 값 총합을 계산한다.
    return {
        "created_at_utc": utc_now_iso(),            # build summary가 생성된 UTC 시각.
        "experiment_id": config.experiment_id,      # 실행된 실험 ID.
        "run_name": config.run_name,                # 실행 run 이름.
        "input_mode": input_mode,                   # 입력이 parquet인지 DataFrame인지 등 입력 방식.
        "input": input_label,                       # 입력 데이터 경로 또는 입력 식별자.
        "output_dir": None,                         # 실제 저장은 이 단계가 아니라 encode 단계에서 처리하므로 None으로 둔다.
        "sample_rows": config.sample_rows,          # 샘플 build를 했다면 split별 샘플 row 수.
        "sampled": config.sample_rows is not None,  # 샘플 실행 여부.
        "sample_scan_summary": dict(sample_scan_summary) if sample_scan_summary is not None else None,  # 샘플 스캔 결과 요약.
        "overwrite": config.overwrite,                              # 기존 산출물 overwrite 허용 여부.
        "preserve_source_columns": config.preserve_source_columns,  # 원본 컬럼 보존 여부.
        "train_ratio": None,                                        # ML-05는 기존 split을 사용하므로 새 split 비율은 기록하지 않는다.
        "val_ratio": None,                                          # ML-05는 기존 split을 사용하므로 새 validation 비율은 기록하지 않는다.
        "preserve_timestamp_groups": None,                          # ML-05는 기존 split을 검증만 하므로 새 timestamp group 옵션은 없다.
        "configured_column_map": dict(config.column_map) if config.column_map is not None else None,  # 사용자가 지정한 컬럼 매핑.
        "resolved_columns": dict(column_map),                             # 실제 입력 컬럼에서 resolve된 최종 컬럼 매핑.
        "feature_columns": selected_feature_columns,                      # 이번 build에서 생성된 피처 컬럼 목록.
        "feature_count": len(selected_feature_columns),                   # 생성된 피처 컬럼 개수.
        "operations": [spec.operation for spec in specs],                 # FeatureSpec 순서대로 실행된 operation 목록.
        "unique_operations": sorted({spec.operation for spec in specs}),  # 중복 제거 후 정렬한 operation 목록.
        "unknown_category_total": unknown_category_total,                 # category unknown 값 총합.
        "row_counts": dict(row_counts),                                   # all/train/val/test row 수 요약.
    }


def unknown_category_total_from_summary(unknown_summary: pd.DataFrame) -> int:
    """Sum unknown category counts from operation artifacts."""

    if unknown_summary.empty:                           # unknown category 요약 artifact가 없거나 비어 있으면 unknown은 0으로 본다.
        return 0
    if "unknown_count" not in unknown_summary.columns:  # unknown 수를 합산하려면 unknown_count 컬럼이 반드시 필요하다.
        raise ValueError("category_unknown_summary artifact is missing unknown_count column.")
    return int(unknown_summary["unknown_count"].sum())  # split/컬럼별 unknown_count를 모두 합산해 정수로 반환한다.
