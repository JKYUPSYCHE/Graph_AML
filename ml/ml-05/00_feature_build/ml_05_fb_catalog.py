"""ML-05 피처 카탈로그와 split 요약 생성기.

코드 맵:
- 입력      : FeatureSpec 튜플 또는 split 정보가 포함된 feature frame.
- 출력      : 피처 카탈로그, ML 피처 컬럼 테이블, split 요약 DataFrame.
- 공개 함수 : make_feature_catalog, make_feature_columns_table, make_split_summary.
- 누수 방지 : 카탈로그에 과거 데이터만 사용한다는 규칙과 동일 timestamp 제외 검사를 기록한다.
- 참고      : train/val/test 요약이 비어 있으면 row 수를 0으로 조용히 보고하지 않고 실패시킨다.
"""

from __future__ import annotations  # 타입 힌트 지연 평가를 활성화한다.

import json                             # dict/list 메타데이터를 JSON 문자열로 저장하기 위해 사용한다.
from typing import Any, Mapping, Tuple  # payload와 FeatureSpec 튜플 타입 힌트에 사용한다.

import pandas as pd  # catalog와 split summary를 DataFrame으로 만들기 위해 사용한다.

from ml_05_fb_schema import validate_no_forbidden_feature_columns                # label/typology 누수 위험이 있는 피처명을 차단한다.
from ml_05_fb_specs import FeatureSpec, feature_columns, validate_feature_specs  # 피처 스펙 타입, 컬럼 추출, 스펙 검증 함수.


FEATURE_SET_LABEL = "full48_flowbalance_passflow"  # ML-05 Stage 4 피처셋을 식별하는 공통 라벨.


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)  # dict 메타데이터를 정렬된 JSON 문자열로 변환한다.


def _window_from_spec(spec: FeatureSpec) -> str:
    return str(spec.params.get("window", "")).strip()  # FeatureSpec params에서 window 값을 문자열로 꺼낸다.


def _selection_status(spec: FeatureSpec) -> str:
    if "balanced_state_flag" in spec.output_col:  # balance 상태 flag 피처는 train 기준 quantile threshold를 사용한다.
        return f"candidate_{FEATURE_SET_LABEL};train_quantile_threshold;no_history_set_to_0"
    if "amount__in_out_ratio" in spec.output_col or "current_to_last_in_amount_ratio" in spec.output_col:  # ratio 피처는 분모 0 처리 규칙을 기록한다.
        return f"candidate_{FEATURE_SET_LABEL};denominator_zero_handled_as_0"
    if "sequence_count" in spec.output_col:  # 과거 sequence 개수 기반 피처임을 표시한다.
        return f"candidate_{FEATURE_SET_LABEL};past_sequence_count"
    return f"candidate_{FEATURE_SET_LABEL}"  # 별도 특수 처리 없는 기본 후보 피처 상태값.


def make_feature_columns_table(feature_specs: Tuple[FeatureSpec, ...]) -> pd.DataFrame:
    """Create the ML feature list table consumed by training code."""

    validate_feature_specs(feature_specs)           # FeatureSpec 목록의 구조와 필수 값을 검증한다.
    columns = feature_columns(feature_specs)        # 학습에 전달할 피처 컬럼명을 스펙 순서대로 추출한다.
    validate_no_forbidden_feature_columns(columns)  # 생성 피처명에 label/typology 등 누수 위험 이름이 있는지 확인한다.
    return pd.DataFrame(  # 학습 코드가 사용할 피처 컬럼 테이블을 만든다.
        {
            "column_name": columns,  # 피처 컬럼명.
            "used_in_ml": ["TRUE" if spec.used_in_ml else "FALSE" for spec in feature_specs],  # ML 입력 사용 여부를 문자열로 기록한다.
        }
    )


def make_feature_catalog(
    feature_specs: Tuple[FeatureSpec, ...],
    *,
    experiment_id: str,
    run_name: str,
) -> pd.DataFrame:
    """Convert selected FeatureSpecs to a reviewable ML-05 feature catalog."""

    validate_feature_specs(feature_specs)           # FeatureSpec 목록의 구조와 필수 값을 먼저 검증한다.
    columns = feature_columns(feature_specs)        # 스펙 순서 기준 피처 컬럼명 목록을 만든다.
    validate_no_forbidden_feature_columns(columns)  # 카탈로그에 기록될 피처명이 누수 위험 이름인지 확인한다.
    rows: list[dict[str, Any]] = []                 # catalog DataFrame으로 바꿀 row dict 목록.
    for spec in feature_specs:                      # 각 FeatureSpec을 카탈로그 row로 변환한다.
        rows.append(
            {
                "feature_name": spec.output_col,
                "column_name": spec.output_col,
                "experiment_id": experiment_id,
                "run_name": run_name,
                "stage": "Stage 4",
                "operation": spec.operation,
                "feature_family": spec.family,
                "entity_scope": spec.entity_scope,
                "direction": spec.direction,
                "window": _window_from_spec(spec),
                "input_columns": _json_dumps(spec.input_cols),
                "params": _json_dumps(spec.params),
                "description": spec.description,
                "aml_typology": spec.aml_typology,
                "leakage_policy": spec.leakage_policy,
                "leakage_rule": "Use only rows with history_timestamp < current_timestamp; lower window bound inclusive for window features.",
                "leakage_check": "Account flow operations emit each timestamp group before adding same-timestamp rows to history.",
                "computational_cost": spec.computational_cost,
                "selection_status": _selection_status(spec),
                "used_in_ml": "TRUE" if spec.used_in_ml else "FALSE",
            }
        )
    catalog = pd.DataFrame(rows)                # row 목록을 feature catalog DataFrame으로 변환한다.
    observed = catalog["column_name"].tolist()  # 실제 catalog에 들어간 컬럼 순서를 확인한다.
    if observed != columns:                     # FeatureSpec 순서와 catalog 컬럼 순서가 다르면 재현성이 깨질 수 있다.
        raise ValueError(f"Feature catalog column order mismatch. observed={observed}, expected={columns}")
    
    # 빈 값이면 안 되는 필수 메타데이터 컬럼.
    required_non_empty = ["stage", "window", "leakage_rule", "leakage_check", "aml_typology", "selection_status"] 
    for column in required_non_empty:                                          # 필수 메타데이터 컬럼을 하나씩 검사한다.
        blank = catalog[column].astype("string").str.strip().fillna("") == ""  # 공백 또는 결측 값을 blank로 판단한다.
        if bool(blank.any()):      # 필수 메타데이터에 빈 값이 있으면 catalog 품질 문제로 실패시킨다.
            raise ValueError(
                "Feature catalog failed: required metadata column contains blank values. "
                f"column={column!r}, blank_rows={(catalog.index[blank] + 2).tolist()[:30]}"
            )
    return catalog  # 검증이 끝난 feature catalog를 반환한다.


def make_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize row counts, periods, and labels by split."""

    required = {"split", "timestamp", "label"}  # split 요약에 필요한 필수 컬럼.
    missing = required - set(df.columns)        # 입력 DataFrame에서 누락된 필수 컬럼을 찾는다.
    if missing:  # 필수 컬럼이 없으면 split별 기간/라벨 요약을 만들 수 없다.
        raise ValueError(f"split summary input is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []              # split별 요약 row를 담을 목록.
    for split_name in ["train", "val", "test"]:  # train, val, test 순서로 요약을 만든다.
        split_df = df[df["split"] == split_name] # 현재 split에 해당하는 row만 추출한다.
        if split_df.empty:  # 필수 split이 비어 있으면 조용히 0건으로 보고하지 않고 실패시킨다.
            raise ValueError(f"split summary input has no rows for required split: {split_name}")
        label_counts = split_df["label"].astype(int).value_counts().to_dict()  # 현재 split의 label 0/1 개수를 집계한다.
        rows.append(
            {
                "split": split_name,         # split 이름.
                "rows": int(len(split_df)),  # 현재 split의 전체 row 수.
                "timestamp_min": split_df["timestamp"].min(),  # 현재 split의 시작 timestamp.
                "timestamp_max": split_df["timestamp"].max(),  # 현재 split의 마지막 timestamp.
                "label_0_count": int(label_counts.get(0, 0)),  # 정상 label 개수.
                "label_1_count": int(label_counts.get(1, 0)),  # laundering label 개수.
                "positive_rate": float(split_df["label"].mean()) if len(split_df) else 0.0,  # 현재 split의 positive 비율.
            }
        )
    return pd.DataFrame(rows)  # split별 요약 row를 DataFrame으로 반환한다.
