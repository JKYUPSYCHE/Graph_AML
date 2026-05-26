"""Feature operation 공통 검증/후처리 helper.

이 모듈은 rolling/non-rolling operation이 함께 쓰는 작은 도구만 둔다.
실제 feature 계산 알고리즘은 각 operation 모듈에 남긴다.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ml_02_fb_specs import FeatureOpResult, FeatureSpec


def _json_dumps(payload: Mapping[str, Any]) -> str:
    """feature_info.csv에 input_columns/params를 저장하기 위해 JSON 문자열로 변환한다."""

    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)


def require_columns(df: pd.DataFrame, columns: Tuple[str, ...], operation: str) -> None:
    """operation 실행에 필요한 입력 컬럼이 모두 있는지 확인한다."""

    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            "Feature operation failed: input DataFrame is missing required columns. "
            f"operation={operation!r}, missing_columns={sorted(missing)}"
        )


def require_roles(spec: FeatureSpec, roles: Tuple[str, ...]) -> dict[str, str]:
    """FeatureSpec.input_cols에 operation이 요구하는 role이 있는지 확인한다."""

    missing_roles = set(roles) - set(spec.input_cols)
    if missing_roles:
        raise ValueError(
            "Feature operation failed: FeatureSpec.input_cols is missing roles. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, missing_roles={sorted(missing_roles)}"
        )
    return {role: str(spec.input_cols[role]) for role in roles}


def require_allowed_params(spec: FeatureSpec, allowed: Tuple[str, ...]) -> None:
    """operation별로 허용된 params만 받는다."""

    unknown = sorted(set(spec.params) - set(allowed))
    if unknown:
        raise ValueError(
            "Feature operation failed: unsupported params were provided. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, unknown_params={unknown}, "
            f"allowed_params={list(allowed)}"
        )


def param_value(spec: FeatureSpec, name: str, default: Any) -> Any:
    """FeatureSpec.params에서 값을 꺼내고, 없으면 명시한 기본값을 사용한다."""

    if name in spec.params:
        return spec.params[name]
    return default


def feature_info(
    features: pd.DataFrame,
    spec: FeatureSpec,
    input_columns: Mapping[str, str],
    params: Mapping[str, Any],
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """생성된 feature 컬럼의 분포/품질 정보를 만든다."""

    column = spec.output_col
    if column not in features.columns:
        raise ValueError(
            "Feature operation failed: output column was not created. "
            f"operation={spec.operation!r}, output_col={column!r}"
        )
    if len(features) == 0:
        raise ValueError(f"Feature operation failed: output feature is empty. output_col={column!r}")

    series = features[column]
    numeric = pd.to_numeric(series, errors="coerce")
    missing_count = int(series.isna().sum())
    if missing_count and not allow_missing:
        raise ValueError(
            "Feature operation failed: output feature contains missing values. "
            f"operation={spec.operation!r}, output_col={column!r}, missing_count={missing_count}"
        )
    failed_mask = numeric.isna()
    if failed_mask.any() and not allow_missing:
        examples = series.loc[failed_mask].astype(str).head(5).tolist()
        raise ValueError(
            "Feature operation failed: output feature must be numeric. "
            f"operation={spec.operation!r}, output_col={column!r}, examples={examples}"
        )

    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    inf_count = int(np.isinf(values).sum())
    if inf_count:
        raise ValueError(
            "Feature operation failed: output feature contains inf values. "
            f"operation={spec.operation!r}, output_col={column!r}, inf_count={inf_count}"
        )

    quantiles = numeric.quantile([0.25, 0.5, 0.75])
    return pd.DataFrame(
        [
            {
                "column_name": column,
                "operation": spec.operation,
                "input_columns": _json_dumps(input_columns),
                "params": _json_dumps(params),
                "dtype": str(features[column].dtype),
                "rows": int(len(features)),
                "missing_count": missing_count,
                "missing_rate": float(missing_count / len(features)),
                "inf_count": inf_count,
                "zero_count": int((numeric == 0).sum()),
                "zero_rate": float((numeric == 0).sum() / len(features)),
                "unique_count": int(numeric.nunique(dropna=True)),
                "min": float(numeric.min()),
                "p25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "mean": float(numeric.mean()),
                "p75": float(quantiles.loc[0.75]),
                "max": float(numeric.max()),
                "near_zero_variance": bool(numeric.nunique(dropna=True) <= 1),
                "leakage_policy": spec.leakage_policy,
                "computational_cost": spec.computational_cost,
            }
        ]
    )


def finalize_result(
    series: pd.Series,
    spec: FeatureSpec,
    *,
    row_count: int,
    input_columns: Mapping[str, str],
    params: Mapping[str, Any],
    dtype: str,
    artifacts: Optional[Mapping[str, Any]] = None,
    allow_missing: bool = False,
) -> FeatureOpResult:
    """operation 결과 Series를 표준 FeatureOpResult로 마무리한다."""

    if len(series) != row_count:
        raise ValueError(
            "Feature operation failed: output row count mismatch. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, "
            f"expected_rows={row_count}, observed_rows={len(series)}"
        )
    features = pd.DataFrame({spec.output_col: series.reset_index(drop=True)})
    numeric = pd.to_numeric(features[spec.output_col], errors="coerce")
    if numeric.isna().any() and not allow_missing:
        bad_examples = features.loc[numeric.isna(), spec.output_col].astype(str).head(5).tolist()
        raise ValueError(
            "Feature operation failed: output cannot be converted to numeric dtype. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, examples={bad_examples}"
        )
    features[spec.output_col] = numeric.astype(dtype)
    info = feature_info(
        features,
        spec,
        input_columns=input_columns,
        params=params,
        allow_missing=allow_missing,
    )
    artifact_payload: Mapping[str, Any] = {} if artifacts is None else artifacts
    return FeatureOpResult(features=features, feature_info=info, artifacts=artifact_payload)
