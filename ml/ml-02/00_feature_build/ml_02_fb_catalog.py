"""
Feature catalog와 요약 테이블 생성 모듈

이 파일의 역할
----------------
1. 기존 학습 모듈이 읽는 `ml_feature_columns.csv`를 만든다.
2. 사람이 feature 의미를 검토하는 `feature_catalog.csv`를 만든다.
3. train/val/test split별 row 수, 기간, label 분포를 요약한다.

중요한 구분
-----------
- `ml_feature_columns.csv`: 모델 입력 truth source. column_name과 used_in_ml만 중요하다.
  used_in_ml은 문자열 "TRUE" / "FALSE"로 저장한다.
- `feature_catalog.csv`: 설명/관리용. operation, params, leakage_policy 등을 사람이 검토한다.
- `feature_info.csv`: 실제 생성된 컬럼의 분포/품질 정보. 이 파일은 ml_02_fb_operations.py에서 만든다.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Tuple

import pandas as pd

from ml_02_fb_schema import validate_no_forbidden_feature_columns
from ml_02_fb_specs import FeatureSpec, feature_columns, validate_feature_specs


def _json_dumps(payload: Mapping[str, Any]) -> str:
    """dict 형태 metadata를 CSV 한 칸에 넣기 위해 JSON 문자열로 변환한다."""

    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)


def make_feature_columns_table(feature_specs: Tuple[FeatureSpec, ...]) -> pd.DataFrame:
    """
    기존 ML 학습 모듈이 사용할 feature column 목록을 만든다.

    반환 컬럼
    ---------
    - column_name: feature 컬럼명
    - used_in_ml: 모델 입력으로 사용할지 여부. 문자열 "TRUE" / "FALSE"로 저장
    """

    validate_feature_specs(feature_specs)
    columns = feature_columns(feature_specs)
    validate_no_forbidden_feature_columns(columns)
    return pd.DataFrame(
        {
            "column_name": columns,
            "used_in_ml": ["TRUE" if spec.used_in_ml else "FALSE" for spec in feature_specs],
        }
    )


def make_feature_catalog(
    feature_specs: Tuple[FeatureSpec, ...],
    *,
    experiment_id: str,
    run_name: str,
) -> pd.DataFrame:
    """
    선택된 FeatureSpec 목록을 사람이 검토하기 쉬운 catalog 형태로 변환한다.

    catalog는 feature 생성의 결과 설명서 역할을 한다.
    실제 모델이 어떤 컬럼을 쓸지는 `ml_feature_columns.csv`가 기준이다.
    """

    validate_feature_specs(feature_specs)
    columns = feature_columns(feature_specs)
    validate_no_forbidden_feature_columns(columns)
    rows: list[dict[str, Any]] = []
    for spec in feature_specs:
        # FeatureSpec 하나가 catalog row 하나가 된다.
        # input_columns와 params는 구조가 있는 값이므로 JSON 문자열로 저장한다.
        rows.append(
            {
                "feature_name": spec.output_col,
                "column_name": spec.output_col,
                "experiment_id": experiment_id,
                "run_name": run_name,
                "operation": spec.operation,
                "feature_family": spec.family,
                "entity_scope": spec.entity_scope,
                "direction": spec.direction,
                "input_columns": _json_dumps(spec.input_cols),
                "params": _json_dumps(spec.params),
                "description": spec.description,
                "aml_typology": spec.aml_typology,
                "leakage_policy": spec.leakage_policy,
                "computational_cost": spec.computational_cost,
                "used_in_ml": "TRUE" if spec.used_in_ml else "FALSE",
            }
        )
    catalog = pd.DataFrame(rows)
    observed = catalog["column_name"].tolist()
    if observed != columns:
        raise ValueError(f"Feature catalog column order mismatch. observed={observed}, expected={columns}")
    return catalog


def make_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """train/val/test split별 기간, row 수, label 분포를 요약한다.

    이 요약은 split을 새로 만들지 않는다. 입력 DataFrame에 이미 있는 split 값을 기준으로
    feature build/encoding 산출물이 어떤 기간과 label 분포를 가졌는지 확인하는 검토용 테이블이다.
    """

    required = {"split", "timestamp", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split summary input is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        label_counts = split_df["label"].astype(int).value_counts().to_dict()
        rows.append(
            {
                "split": split_name,
                "rows": int(len(split_df)),
                "timestamp_min": split_df["timestamp"].min(),
                "timestamp_max": split_df["timestamp"].max(),
                "label_0_count": int(label_counts.get(0, 0)),
                "label_1_count": int(label_counts.get(1, 0)),
                "positive_rate": float(split_df["label"].mean()) if len(split_df) else 0.0,
            }
        )
    return pd.DataFrame(rows)
