"""
Feature operation 실행 모듈

이 파일의 역할
----------------
1. FeatureSpec에 선언된 operation 이름을 실제 계산 함수로 연결한다.
2. operation 결과가 학습 입력으로 안전한지 검증한다.
3. rolling/recency처럼 같은 중간 계산을 공유할 수 있는 operation은 batch로 실행한다.
4. 생성 feature의 분포/품질 정보를 feature_info 형태로 만든다.
5. 생성 feature 순서와 feature_info 계약을 검증한다.

설계 원칙
----------------
- 모든 operation은 `df`와 `FeatureSpec`을 입력으로 받는다.
- 모든 operation은 `FeatureOpResult`를 반환한다.
- 잘못된 입력 컬럼, 잘못된 파라미터, NaN/inf/non-numeric 출력은 즉시 에러로 중단한다.
- rolling/time-history 계열은 현재 row와 미래 row를 보지 않도록 past-only 정책을 강제한다.

처음 읽는 순서
--------------
1. 파일 하단의 `execute_feature_specs()`를 먼저 읽어 전체 dispatch 흐름을 확인한다.
2. `OPERATION_REGISTRY`에서 operation 문자열과 실제 함수 연결을 확인한다.
3. 관심 operation 함수만 골라 세부 계산을 읽는다.
4. rolling 세부 계산은 이 파일이 아니라 `ml_01_fb_rolling.py`에서 확인한다.

실행 흐름 요약
--------------
`execute_feature_specs()`는 아래 순서로 동작한다.
1. spec 목록과 meta column을 검증한다.
2. rolling/ratio/recency처럼 중간 계산을 공유하는 spec을 먼저 batch 실행한다.
3. 사용자가 선언한 feature_specs 순서대로 결과를 다시 조립한다.
4. feature_frame, feature_info, artifacts 반환 계약을 검증한다.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd

from ml_01_fb_schema import normalize_category_strict, parse_datetime_strict, parse_numeric_strict
from ml_01_fb_specs import FeatureOpResult, FeatureSpec, META_COLUMNS, feature_columns, validate_feature_specs
from ml_01_fb_rolling import execute_rolling_agg_specs_batched, op_rolling_agg, parse_window
from ml_01_fb_operation_result_validation import (
    finalize_result as _finalize_result,
    param_value as _param_value,
    require_allowed_params as _require_allowed_params,
    require_columns as _require_columns,
    require_roles as _require_roles,
)


# -----------------------------------------------------------------------------
# 1. 공통 반환 artifact schema
# -----------------------------------------------------------------------------
# category feature가 하나도 선택되지 않아도 header가 있는 빈 CSV를 저장하기 위한 고정 schema다.
CATEGORY_MAPPING_COLUMNS: Tuple[str, ...] = (
    "feature_column",
    "source_column",
    "category_value",
    "encoded_value",
    "fit_split",
)

CATEGORY_UNKNOWN_COLUMNS: Tuple[str, ...] = (
    "feature_column",
    "source_column",
    "split",
    "unknown_count",
    "unknown_unique_count",
    "unknown_examples",
    "policy",
)

# operation 함수의 표준 타입이다. 모든 operation은 같은 입력/출력 계약을 따른다.
OperationRunner = Callable[[pd.DataFrame, FeatureSpec], FeatureOpResult]

def _empty_category_mapping() -> pd.DataFrame:
    """category feature가 없을 때도 header가 있는 빈 mapping DataFrame을 반환한다."""
    return pd.DataFrame(columns=list(CATEGORY_MAPPING_COLUMNS))

def _empty_category_unknown_summary() -> pd.DataFrame:
    """category feature가 없을 때도 header가 있는 빈 unknown summary DataFrame을 반환한다."""
    return pd.DataFrame(columns=list(CATEGORY_UNKNOWN_COLUMNS))


def _entity_timestamp_work(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, dict[str, str]]:
    """entity/timestamp 기반 past-only operation이 공유하는 정렬 work frame을 만든다."""

    roles = _require_roles(spec, ("entity_col", "timestamp_col"))
    entity_col = roles["entity_col"]
    timestamp_col = roles["timestamp_col"]
    _require_columns(df, (entity_col, timestamp_col), spec.operation)

    entity = normalize_category_strict(df[entity_col], source_col=entity_col)
    timestamps = parse_datetime_strict(df, timestamp_col, spec.output_col)
    work = pd.DataFrame(
        {
            "_entity": entity,
            "_timestamp": timestamps,
            "_row_order": np.arange(len(df)),
        }
    ).sort_values(["_entity", "_timestamp", "_row_order"], kind="mergesort")
    return work, roles


# -----------------------------------------------------------------------------
# 2. Recency / first transaction operation
# -----------------------------------------------------------------------------
def _entity_recency_parts(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    """
    entity별 직전 과거 timestamp와 첫 거래 flag를 함께 계산한다.
    같은 entity 기준으로 현재 거래 이전의 가장 최근 거래 시각을 찾고,
    그 차이를 초 단위 recency로 계산한다.
    동시에 해당 entity의 첫 거래인지 여부도 계산한다.
    반환값:
        recency:
            현재 거래 timestamp - 직전 과거 timestamp, 단위는 초.
            해당 entity의 첫 거래이면 직전 timestamp가 없으므로 NaN으로 둔다.
            이후 op_recency_seconds_since_last()에서 fill_value로 채운다.
        is_first:
            해당 entity의 첫 거래이면 1, 아니면 0.
        roles:
            spec.input_cols에서 꺼낸 역할 매핑.
            예: {"entity_col": "sender_account_id", "timestamp_col": "timestamp"}
    """
    work, roles = _entity_timestamp_work(df, spec)

    # 같은 entity/timestamp에 여러 거래가 있어도 서로를 직전 거래로 보지 않는다.
    # 그래서 row 단위가 아니라 entity + timestamp 단위로 먼저 중복 제거한다.
    timestamp_frame = work[["_entity", "_timestamp"]].drop_duplicates(["_entity", "_timestamp"], keep="first")

    # entity별로 이전 timestamp를 구한다.
    # shift(1)은 같은 entity 안에서 바로 이전 timestamp를 가져온다.
    previous_timestamp = timestamp_frame.groupby("_entity", sort=False)["_timestamp"].shift(1)

    # 현재 timestamp와 이전 timestamp의 차이를 초 단위로 계산한다.
    # 첫 거래는 previous_timestamp가 없으므로 _recency는 NaN이 된다.
    timestamp_frame["_recency"] = (timestamp_frame["_timestamp"] - previous_timestamp).dt.total_seconds()

    # previous_timestamp가 없으면 해당 entity의 첫 거래 timestamp다. 첫 거래이면 1, 아니면 0으로 표시한다.
    timestamp_frame["_is_first"] = previous_timestamp.isna().astype("int8")

    # timestamp 단위로 계산한 recency/is_first 값을 원래 row 단위 work에 다시 붙인다.
    work = work.merge(timestamp_frame, on=["_entity", "_timestamp"], how="left", sort=False)

    # 정렬된 work에서 원래 df row 위치를 가져온다. 이 위치를 사용해 최종 결과 Series를 원본 df 순서에 맞게 복원한다.
    row_orders = work["_row_order"].to_numpy()

    # recency 결과 Series를 원본 df 길이로 만든다. 먼저 NaN으로 초기화한 뒤, row_orders 위치에 계산값을 채운다.
    recency = pd.Series(np.nan, index=np.arange(len(df)), dtype="float64")
    recency.iloc[row_orders] = pd.to_numeric(work["_recency"], errors="coerce").to_numpy(dtype="float64")

    # is_first 결과 Series를 원본 df 길이로 만든다. 기본값은 0이고, row_orders 위치에 계산된 first flag를 채운다.
    is_first = pd.Series(0, index=np.arange(len(df)), dtype="int8")
    is_first.iloc[row_orders] = work["_is_first"].to_numpy(dtype="int8")
    
    # recency와 is_first는 같은 중간 계산에서 나오므로 함께 반환한다.
    # roles는 후속 _finalize_result()에서 input column metadata로 사용된다.
    return recency, is_first, roles


def op_recency_seconds_since_last(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """
    entity별 직전 과거 거래와의 시간 차이를 초 단위로 계산한다.
    이 operation은 _entity_recency_parts()가 만든 recency 결과만 사용한다.
    첫 거래는 직전 과거 거래가 없어서 recency가 NaN으로 계산된다.
    기본 fill_value=-1.0은 실제 시간 차이가 아니라 "이전 거래 없음"을 나타내는 sentinel 값이다.
    """
    _require_allowed_params(spec, ("dtype", "fill_value"))      # 잘못된 파라미터가 들어오면 중단     
    recency, _is_first, roles = _entity_recency_parts(df, spec) # recency와 is_first는 같은 정렬/merge 결과에서 함께 계산된다.
    fill_value = _param_value(spec, "fill_value", -1.0)         # 첫 거래처럼 직전 과거 거래가 없어 recency가 NaN인 row에 채울 값이다.
    recency = recency.fillna(fill_value)                        # 모델 입력에는 NaN을 남기지 않기 위해 sentinel 값으로 채운다.
    dtype = str(_param_value(spec, "dtype", "float64"))         # 최종 feature dtype을 결정한다.

    # feature_info에 기록할 실제 적용 파라미터다.
    # dtype은 _finalize_result() 인자로 별도 전달되므로 여기에는 fill_value만 남긴다.
    params = {"fill_value": fill_value}

    # Series를 표준 FeatureOpResult로 변환한다.
    return _finalize_result(
        recency,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=params,
        dtype=dtype,
    )


def op_is_first_by_entity(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """
    entity 기준 과거 거래가 없으면 1, 있으면 0인 flag를 만든다.
    이 함수의 최종 출력은 "첫 거래 여부" flag 1개다.
    값의 의미:
        1 = 해당 entity의 첫 timestamp 거래
        0 = 같은 entity의 더 과거 timestamp 거래가 존재함
    """
    
    _require_allowed_params(spec, ("dtype",))                    # is_first_by_entity는 dtype만 파라미터로 허용한다.
    _recency, is_first, roles = _entity_recency_parts(df, spec)  # recency와 is_first를 함께 계산한다.   
    dtype = str(_param_value(spec, "dtype", "int8"))             # 기본 dtype은 int8이다.

    # is_first Series를 표준 FeatureOpResult로 포장한다.
    return _finalize_result(
        is_first,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype=dtype,
    )


def _recency_group_key(spec: FeatureSpec) -> tuple[str, str]:
    """
    recency 계열 feature를 batch 계산할 때 묶을 기준 key를 만든다.
    같은 entity_col과 timestamp_col을 쓰는 spec들은 동일한 정렬/merge 결과를 공유할 수 있다.
    예를 들어 sender 기준 recency와 sender 기준 is_first는 같은 group으로 묶을 수 있다.
    반대로 sender 기준 feature와 receiver 기준 feature는 entity_col이 다르므로 분리해야 한다.
    """
    
    roles = _require_roles(spec, ("entity_col", "timestamp_col"))  # spec.input_cols에서 batch grouping에 필요한 역할 컬럼을 꺼낸다.
    return roles["entity_col"], roles["timestamp_col"]             # 이 tuple이 execute_recency_specs_batched()의 grouping key가 된다.


def execute_recency_specs_batched(
    df: pd.DataFrame,
    specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """
    같은 entity/timestamp를 쓰는 recency 계열 spec을 한 번의 정렬로 함께 실행한다.
    seconds_since_last와 is_first는 같은 중간 결과에서 파생된다.
    따로 실행하면 같은 정렬, groupby, merge를 반복하므로 계산 비용이 커진다.
    이 함수는 같은 entity_col/timestamp_col 조합끼리 spec을 묶고,
    group마다 _entity_recency_parts()를 한 번만 호출한다.
    반환값은 output_col -> FeatureOpResult dict다.
    batch로 계산하더라도 각 feature의 반환 계약은 개별 operation과 동일하게 유지한다.
    """
    # 실행할 recency spec이 없으면 빈 결과를 반환한다.
    if not specs:
        return {}
    
    # FeatureSpec 자체의 중복 output_col, 필수 필드 등 기본 계약을 먼저 검증한다.
    validate_feature_specs(specs)

    # 같은 entity_col/timestamp_col을 쓰는 spec끼리 묶는다. 같은 group 안에서는 정렬/merge 결과를 재사용할 수 있다.
    grouped_specs: dict[tuple[str, str], list[FeatureSpec]] = {}
    for spec in specs:
        # 이 batch 실행기는 recency 계열 operation만 처리한다.
        # 다른 operation이 섞이면 잘못된 호출이므로 즉시 실패시킨다.
        if spec.operation not in {"recency_seconds_since_last", "is_first_by_entity"}:
            raise ValueError(
                "Feature build failed: execute_recency_specs_batched only accepts recency specs. "
                f"operation={spec.operation!r}, output_col={spec.output_col!r}"
            )
        # entity_col/timestamp_col 조합을 기준으로 spec을 누적한다.
        grouped_specs.setdefault(_recency_group_key(spec), []).append(spec)
    results: dict[str, FeatureOpResult] = {}

    # group마다 recency/is_first 중간 결과를 한 번만 계산한다.
    for group_specs in grouped_specs.values():
        # 같은 group의 spec들은 entity_col/timestamp_col이 같으므로 첫 spec 기준으로 계산해도 된다.
        recency, is_first, roles = _entity_recency_parts(df, group_specs[0])
        for spec in group_specs:
            if spec.operation == "recency_seconds_since_last":
                
                _require_allowed_params(spec, ("dtype", "fill_value")) # recency_seconds_since_last는 dtype과 fill_value만 허용한다.
                fill_value = _param_value(spec, "fill_value", -1.0)    # 첫 거래 NaN을 채울 sentinel 값이다.
                dtype = str(_param_value(spec, "dtype", "float64"))    # 최종 출력 dtype이다.
                
                # 공통 recency 결과를 재사용하되, spec별 fill_value와 dtype 정책을 적용한다.
                result = _finalize_result(
                    recency.fillna(fill_value),
                    spec,
                    row_count=len(df),
                    input_columns=roles,
                    params={"fill_value": fill_value},
                    dtype=dtype,
                )
            else:                
                _require_allowed_params(spec, ("dtype",)) # is_first_by_entity는 dtype만 허용한다.                
                dtype = str(_param_value(spec, "dtype", "int8")) # flag feature이므로 기본 dtype은 int8이다.

                # 공통 is_first 결과를 spec의 output_col로 포장한다.
                result = _finalize_result(
                    is_first,
                    spec,
                    row_count=len(df),
                    input_columns=roles,
                    params=spec.params,
                    dtype=dtype,
                )

            # output_col 중복은 최종 feature_frame 컬럼 충돌로 이어지므로 여기서 차단한다.
            if spec.output_col in results:
                raise ValueError(f"Feature build failed: duplicate recency batch result. output_col={spec.output_col!r}")
            # execute_feature_specs()에서 output_col로 바로 찾아 쓸 수 있도록 저장한다.
            results[spec.output_col] = result

    return results


# -----------------------------------------------------------------------------
# 3. ML-01 Stage 0 time-history operation
# -----------------------------------------------------------------------------
def op_cur_vs_mean_ratio(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """
    현재 금액을 entity별 과거 window 평균 금액으로 나눈 비율을 계산한다.
    계산식
    ------
    ratio = 현재 거래 금액 / 같은 entity의 과거 window 평균 거래 금액
    사용 경로
    --------
        단독 실행 경로 = 여기서 rolling mean까지 직접 계산
        전체 build 경로 = rolling mean은 batch에서 계산, 여기 함수는 우회
    """
    rolling_mean_spec = _rolling_mean_spec_for_ratio(spec)
    rolling_mean = op_rolling_agg(df, rolling_mean_spec).features[rolling_mean_spec.output_col]
    return _execute_cur_vs_mean_ratio_from_mean(df, spec, rolling_mean)

def _rolling_mean_spec_for_ratio(spec: FeatureSpec) -> FeatureSpec:
    """
    cur_vs_mean_ratio 계산에 필요한 내부 rolling mean spec을 만든다.
    cur_vs_mean_ratio는 다음 계산식으로 만들어진다.
        현재 거래 금액 / 같은 entity의 과거 window 평균 금액
    이때 분모인 "과거 window 평균 금액"은 rolling_agg operation으로 계산할 수 있다.
    그래서 ratio spec을 바로 계산하지 않고, 먼저 rolling mean용 임시 FeatureSpec을 만든다.
    이 임시 spec은 최종 feature로 저장되지 않는다.
    """
    # 원래 ratio spec에서 허용하는 파라미터만 받는다.
    _require_allowed_params(spec, ("window", "closed", "fill_value", "zero_division_value", "dtype"))

    # ratio 계산에 필요한 역할 컬럼을 가져온다. 같은 roles를 rolling mean 계산에도 그대로 사용한다.
    # entity_col    : sender 또는 receiver처럼 과거 평균을 묶을 기준
    # timestamp_col : 과거 window 판단 기준 시간
    # value_col     : 평균을 계산할 금액 컬럼
    roles = _require_roles(spec, ("entity_col", "timestamp_col", "value_col"))

    # rolling mean을 계산할 시간 window다.
    window = _param_value(spec, "window", "")

    # current/future leakage 방지를 위해 closed='left'만 허용한다.
    closed = str(_param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: cur_vs_mean_ratio only supports closed='left' to avoid current/future leakage. "
            f"observed_closed={closed!r}"
        )
    
    # 과거 window 안에 값이 없을 때 rolling mean 쪽에서 사용할 fill 값이다.
    fill_value = float(_param_value(spec, "fill_value", 0.0))
    
    # ratio 계산에 필요한 rolling mean용 임시 FeatureSpec을 만든다.
    # operation은 rolling_agg, agg는 mean으로 고정한다.
    # output_col 앞의 "__rolling_mean_for__"는 최종 feature가 아니라 내부 중간값임을 표시한다.
    return FeatureSpec(
        operation="rolling_agg",
        output_col=f"__rolling_mean_for__{spec.output_col}",
        input_cols=roles,
        params={"window": window, "agg": "mean", "closed": closed, "fill_value": fill_value, "dtype": "float64"},
        leakage_policy=spec.leakage_policy,
        used_in_ml=False,
    )


def _execute_cur_vs_mean_ratio_from_mean(
    df: pd.DataFrame,
    spec: FeatureSpec,
    rolling_mean: pd.Series,
) -> FeatureOpResult:
    """
    캐시된 rolling mean 결과를 사용해 cur_vs_mean_ratio를 계산한다.

    왜 별도 함수가 필요한가
    ---------------------
    cur_vs_mean_ratio feature가 여러 개 있으면 각 feature마다 rolling mean 계산이 필요하다.
    매번 op_rolling_agg()를 따로 실행하면 같은 정렬/rolling 계산이 반복된다.
    그래서 공식 feature build 경로에서는:
        1. _rolling_mean_spec_for_ratio()로 내부 rolling mean spec을 만든다.
        2. execute_rolling_agg_specs_batched()에서 rolling mean들을 한 번에 계산한다.
        3. 이 함수가 캐시된 rolling mean을 받아 ratio만 계산한다.
    hidden rolling mean
    ----------
    rolling_mean은 ratio 계산을 위한 내부 중간값이다.
    최종 feature_frame에는 rolling_mean 컬럼을 저장하지 않고,
    ratio output_col만 FeatureOpResult로 반환한다.
    """
    # 이 operation에서 허용하는 spec.params key를 제한한다.
    _require_allowed_params(spec, ("window", "closed", "fill_value", "zero_division_value", "dtype"))

    # spec.input_cols에서 계산에 필요한 역할 컬럼을 꺼낸다.
    roles = _require_roles(spec, ("entity_col", "timestamp_col", "value_col"))
    value_col = roles["value_col"]
    
    # 입력 DataFrame에 필요한 컬럼이 실제로 존재하는지 확인한다.
    _require_columns(df, (roles["entity_col"], roles["timestamp_col"], value_col), spec.operation)

    # ratio 계산에 사용된 rolling window를 가져온다.
    window = _param_value(spec, "window", "")

    # closed='left'만 허용한다.
    closed = str(_param_value(spec, "closed", "left")).strip().lower()
    if closed != "left":
        raise ValueError(
            "Feature operation failed: cur_vs_mean_ratio only supports closed='left' to avoid current/future leakage. "
            f"observed_closed={closed!r}"
        )
    
    # 분모가 없거나 0이라 ratio를 계산할 수 없을 때 넣을 값이다.
    zero_division_value = float(_param_value(spec, "zero_division_value", 0.0))

    # rolling_mean 생성 시 사용된 fill 정책이다.
    fill_value = float(_param_value(spec, "fill_value", 0.0))

    # 최종 ratio feature의 dtype이다.
    dtype = str(_param_value(spec, "dtype", "float32"))

    # 캐시된 rolling_mean은 입력 df와 1:1 row alignment가 맞아야 한다.
    # 길이가 다르면 현재 row 금액과 과거 평균이 잘못 매칭되므로 즉시 실패시킨다.
    if len(rolling_mean) != len(df):
        raise ValueError(
            "Feature operation failed: cached rolling mean row count mismatch. "
            f"output_col={spec.output_col!r}, expected_rows={len(df)}, observed_rows={len(rolling_mean)}"
        )
    
    # rolling_mean index를 0..N-1로 맞추고 숫자형으로 변환한다.
    rolling_mean = pd.to_numeric(rolling_mean.reset_index(drop=True), errors="coerce").astype("float64")

    # 현재 거래 금액을 strict하게 숫자로 파싱한다.
    current_value = parse_numeric_strict(df, value_col, spec.output_col).reset_index(drop=True).astype("float64")

    # 기본값은 zero_division_value로 채운다. 이후 분모가 유효한 row만 실제 ratio 값으로 덮어쓴다.
    ratio = pd.Series(zero_division_value, index=np.arange(len(df)), dtype="float64")

    # 분모가 NaN이 아니고 0도 아닌 row만 나눗셈 가능하다.
    valid_denominator = rolling_mean.notna() & (rolling_mean != 0)

    # 유효한 row에 대해서만 현재 금액 / 과거 평균 금액 비율을 계산한다.
    ratio.loc[valid_denominator] = current_value.loc[valid_denominator] / rolling_mean.loc[valid_denominator]

    # 예외적인 inf 값은 최종 검증 전에 zero_division_value로 치환한다. _finalize_result()는 inf를 허용하지 않는다.
    ratio = ratio.replace([np.inf, -np.inf], zero_division_value)

    # feature_info에 기록할 실제 적용 파라미터다.
    # window는 parse_window()를 거쳐 정규화된 문자열로 남긴다.
    params = {
        "window": str(parse_window(window, spec.operation, spec.output_col)),
        "closed": closed,
        "fill_value": fill_value,
        "zero_division_value": zero_division_value,
    }
    
    # ratio Series를 표준 FeatureOpResult로 포장한다.
    # _finalize_result()는 output_col 적용, dtype 변환, row 수 검증,
    # NaN/inf/non-numeric 검증, feature_info 생성을 담당한다.
    return _finalize_result(
        ratio,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=params,
        dtype=dtype,
    )


def op_cumulative_count(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """entity별 현재 timestamp 이전 누적 거래 건수를 계산한다."""

    _require_allowed_params(spec, ("dtype",))
    work, roles = _entity_timestamp_work(df, spec)

    # 같은 timestamp group은 현재 시점 거래로 보고 history에서 제외한다.
    # group size를 먼저 구한 뒤 cumsum에서 현재 group size를 빼면 past_timestamp < current_timestamp 정책이 유지된다.
    timestamp_counts = work.groupby(["_entity", "_timestamp"], sort=False).size().rename("_group_size").reset_index()
    timestamp_counts["_history_count"] = (
        timestamp_counts.groupby("_entity", sort=False)["_group_size"].cumsum() - timestamp_counts["_group_size"]
    )
    work = work.merge(timestamp_counts[["_entity", "_timestamp", "_history_count"]], on=["_entity", "_timestamp"], how="left", sort=False)

    output = pd.Series(0, index=np.arange(len(df)), dtype="int64")
    output.iloc[work["_row_order"].to_numpy()] = work["_history_count"].to_numpy(dtype="int64")

    dtype = str(_param_value(spec, "dtype", "int32"))
    return _finalize_result(
        output,
        spec,
        row_count=len(df),
        input_columns=roles,
        params=spec.params,
        dtype=dtype,
    )


# -----------------------------------------------------------------------------
# 4. operation registry와 실행기
# -----------------------------------------------------------------------------
# FeatureSpec.operation 문자열을 실제 함수로 연결하는 테이블이다.
# 새 operation을 추가할 때는 함수 작성 후 이 registry에 등록해야 한다.
OPERATION_REGISTRY: dict[str, OperationRunner] = {
    "rolling_agg": op_rolling_agg,
    "recency_seconds_since_last": op_recency_seconds_since_last,
    "is_first_by_entity": op_is_first_by_entity,
    "cur_vs_mean_ratio": op_cur_vs_mean_ratio,
    "cumulative_count": op_cumulative_count,
}


def run_operation(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOpResult:
    """FeatureSpec.operation 이름을 registry에서 찾아 실제 operation을 실행한다."""

    if spec.operation not in OPERATION_REGISTRY:
        raise ValueError(
            "Feature build failed: unknown operation. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, "
            f"supported_operations={sorted(OPERATION_REGISTRY)}"
        )
    return OPERATION_REGISTRY[spec.operation](df, spec)


def _validate_operation_result(result: FeatureOpResult, spec: FeatureSpec, row_count: int) -> None:
    """operation 결과가 FeatureSpec 계약과 입력 row 수를 정확히 만족하는지 확인한다."""

    expected = [spec.output_col]
    if list(result.features.columns) != expected:
        raise ValueError(
            "Feature operation failed: result columns do not match spec output_col exactly. "
            f"operation={spec.operation!r}, expected={expected}, observed={list(result.features.columns)}"
        )
    if len(result.features) != row_count:
        raise ValueError(
            "Feature operation failed: result row count differs from input. "
            f"operation={spec.operation!r}, output_col={spec.output_col!r}, "
            f"input_rows={row_count}, output_rows={len(result.features)}"
        )


def _append_category_artifacts(
    result: FeatureOpResult,
    category_mapping_parts: list[pd.DataFrame],
    category_unknown_parts: list[pd.DataFrame],
) -> None:
    """operation이 반환한 category artifact를 최종 artifacts 목록에 누적한다."""

    if "category_mapping" in result.artifacts:
        category_mapping_parts.append(result.artifacts["category_mapping"])
    if "category_unknown_summary" in result.artifacts:
        category_unknown_parts.append(result.artifacts["category_unknown_summary"])


def _precompute_batched_results(
    df: pd.DataFrame,
    feature_specs: Tuple[FeatureSpec, ...],
) -> dict[str, FeatureOpResult]:
    """공유 중간 계산이 필요한 operation 결과를 먼저 만든다."""

    precomputed: dict[str, FeatureOpResult] = {}

    rolling_specs = tuple(spec for spec in feature_specs if spec.operation == "rolling_agg")
    ratio_specs = tuple(spec for spec in feature_specs if spec.operation == "cur_vs_mean_ratio")
    ratio_mean_specs = {spec.output_col: _rolling_mean_spec_for_ratio(spec) for spec in ratio_specs}
    rolling_results = execute_rolling_agg_specs_batched(df, (*rolling_specs, *ratio_mean_specs.values()))
    precomputed.update({spec.output_col: rolling_results[spec.output_col] for spec in rolling_specs})
    for spec in ratio_specs:
        mean_spec = ratio_mean_specs[spec.output_col]
        precomputed[spec.output_col] = _execute_cur_vs_mean_ratio_from_mean(
            df,
            spec,
            rolling_results[mean_spec.output_col].features[mean_spec.output_col],
        )

    recency_specs = tuple(
        spec for spec in feature_specs if spec.operation in {"recency_seconds_since_last", "is_first_by_entity"}
    )
    precomputed.update(execute_recency_specs_batched(df, recency_specs))
    return precomputed


def execute_feature_specs(
    df: pd.DataFrame,
    feature_specs: Tuple[FeatureSpec, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    선택된 FeatureSpec 목록을 순서대로 실행하고 최종 feature frame을 만든다.

    반환값
    ------
    feature_frame:
        tx_id, split, label과 생성 feature 컬럼을 합친 학습 입력 DataFrame이다.
    feature_info:
        생성 feature별 분포/품질/파라미터 정보다.
    artifacts:
        operation별 부가 산출물이다. 현재 ML-01 Stage 0에서는 빈 DataFrame만 반환한다.
    """

    validate_feature_specs(feature_specs)
    missing_meta = set(META_COLUMNS) - set(df.columns)
    if missing_meta:
        raise ValueError(f"Feature execution input is missing metadata columns: {sorted(missing_meta)}")

    feature_parts: list[pd.DataFrame] = []
    feature_info_parts: list[pd.DataFrame] = []
    category_mapping_parts: list[pd.DataFrame] = []
    category_unknown_parts: list[pd.DataFrame] = []

    # [1] 공유 가능한 중간 계산을 먼저 만든다. rolling 계산 구현은 그대로 두고 조립부만 단순 조회 구조로 유지한다.
    precomputed_results = _precompute_batched_results(df, feature_specs)

    # [2] 사용자가 선언한 feature_specs 순서대로 결과를 조립한다.
    # batch로 먼저 계산한 feature도 여기서는 원래 spec 순서에 맞춰 feature_parts에 들어간다.
    for spec in feature_specs:
        result = precomputed_results.get(spec.output_col)
        if result is None:
            result = run_operation(df, spec)

        # operation이 선언한 output_col만 만들었고 입력과 같은 row 수를 반환했는지 확인한다.
        _validate_operation_result(result, spec, len(df))
        feature_parts.append(result.features.reset_index(drop=True))
        feature_info_parts.append(result.feature_info.reset_index(drop=True))

        # 현재 ML-01 Stage 0 operation은 category artifact를 만들지 않지만, 반환 계약은 유지한다.
        # 이후 category operation이 추가되면 같은 artifacts dict에 붙여 저장할 수 있다.
        _append_category_artifacts(result, category_mapping_parts, category_unknown_parts)

    selected_columns = feature_columns(feature_specs)

    # [3] meta columns + 생성 feature columns + artifact를 표준 반환 계약으로 조립한다.
    # 기존 학습 모듈과 연결하기 위해 meta columns를 앞에 두고 feature columns를 뒤에 붙인다.
    feature_frame = pd.concat([df.loc[:, list(META_COLUMNS)].reset_index(drop=True), *feature_parts], axis=1)
    feature_frame["label"] = feature_frame["label"].astype("int8")
    feature_frame["split"] = feature_frame["split"].astype("string")
    feature_info = pd.concat(feature_info_parts, ignore_index=True)
    artifacts = {
        "category_mapping": pd.concat(category_mapping_parts, ignore_index=True) if category_mapping_parts else _empty_category_mapping(),
        "category_unknown_summary": pd.concat(category_unknown_parts, ignore_index=True)
        if category_unknown_parts
        else _empty_category_unknown_summary(),
    }
    if list(feature_frame.columns) != [*META_COLUMNS, *selected_columns]:
        raise ValueError(
            "Feature build failed: final feature frame column order mismatch. "
            f"observed={list(feature_frame.columns)}, expected={[*META_COLUMNS, *selected_columns]}"
        )
    return feature_frame, feature_info, artifacts
