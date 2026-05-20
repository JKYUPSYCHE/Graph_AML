"""
FeatureSpec 선언 모듈
이 파일의 역할
----------------
이 파일은 feature를 직접 계산하지 않는다.
대신 "어떤 입력 컬럼으로 어떤 feature 컬럼을 만들지"를 선언한다.
실제 계산은 ml_00_fb_operations.py에서 수행한다.
따라서 이 파일은 feature build의 설정표에 가깝다.

주요 구성
----------------
1. FeatureSpec
   - feature 1개를 만들기 위한 선언 정보
   - 입력 컬럼, 출력 컬럼, operation 이름, params, catalog 메타데이터를 담는다.
2. FeatureOpResult
   - operation 실행 결과를 담는 자료구조
   - 실제 계산 결과는 ml_00_fb_operations.py에서 생성된다.
3. *_spec helper 함수
   - FeatureSpec(...)을 직접 만들지 않도록 돕는 편의 함수
   - 예: log1p_spec(), rolling_agg_spec(), fan_in_spec()
   - helper를 쓰면 operation 이름, params key, catalog 메타데이터 누락 위험이 줄어든다.
4. spec 검증/조회 함수
   - validate_feature_specs(): output_col 중복, 빈 값 같은 기본 오류를 확인한다.
   - required_input_columns(): 필요한 입력 컬럼 목록을 모은다.
   - feature_columns(): 생성될 feature 컬럼 목록을 반환한다.
   
설계 원칙
----------------
- FeatureSpec 1개는 feature column 1개를 의미한다.
- 이 파일은 선언만 담당하고 계산은 담당하지 않는다.
- Stage 이름이 feature를 결정하지 않는다.
- 실제로 어떤 feature가 생성되는지는 FeatureSpec 목록이 결정한다.
- 모델 입력 여부는 이후 ml_feature_columns.csv의 used_in_ml 값으로 제어한다.

설계 의도 
----------------
feature 계산 함수와 feature 선택 로직을 섞으면, 실험마다 feature 조합을 바꿀 때 코드 수정이 많아진다.
이 구조에서는 노트북에서 FeatureSpec 목록만 바꾸면 된다.
예:
- ML-00: 기본 feature spec 목록 사용
- ML-01: ML-00 목록에 rolling/time-history feature spec 추가
- 특정 feature 제외: spec 목록에서 빼거나 used_in_ml=False로 관리

장점
----------------
- feature 조합을 노트북에서 쉽게 바꿀 수 있다.
- 생성될 feature 목록을 실행 전에 확인할 수 있다.
- feature_catalog.csv에 spec 정보를 저장해 실험 재현성을 높일 수 있다.
- 같은 operation을 여러 feature에서 재사용할 수 있다.

주의할 점
----------------
1. operation 이름은 문자열이다.
   - 예: "rolling_agg", "log1p"
   - 오타가 나면 정적 분석으로는 잡기 어렵다.
   - 따라서 노트북에서는 FeatureSpec(...)을 직접 만들기보다 *_spec helper를 사용한다.
   - registry에 없는 operation은 실행 초기에 ValueError로 실패한다.
2. params는 자유 형식 dict다.
   - 잘못된 key나 타입은 이 파일만으로는 모두 잡을 수 없다.
   - 실제 operation 함수에서 허용 key와 값 타입을 다시 검증한다.
   - helper 함수를 쓰면 params 오타 위험이 줄어든다.
3. output_col 중복만 자동 차단한다.
   - 이름은 다르지만 의미가 같은 feature까지 자동으로 막지는 못한다.
   - 중복 의미 feature는 feature_info.csv, feature_catalog.csv, correlation 점검으로 확인한다.
4. dataclass(frozen=True)여도 내부 dict는 변경될 수 있다.
   - spec은 생성 후 수정하지 않는 것을 원칙으로 한다.
   - 변경이 필요하면 in-place 수정하지 말고 새 FeatureSpec을 만든다.
5. catalog 메타데이터는 helper 사용을 전제로 한다.
   - FeatureSpec(...)을 직접 호출하면 family, leakage_policy 같은 메타데이터가 비어 있을 수 있다.
   - catalog 검토 시 "unspecified" 항목은 우선 확인한다.
6. required_columns는 논리 컬럼명 기준이다.
   - 예: "amount", "sender_account_id"
   - 실제 parquet 컬럼명과의 매핑은 schema/COLUMN_MAP 단계에서 처리한다.
   - COLUMN_MAP은 가능한 한 명시적으로 작성한다.
7. computational_cost, aml_typology 같은 값은 참고용 메타데이터다.
   - 실제 속도나 메모리 사용량을 보장하지 않는다.
   - 최종 판단은 feature build 실행 결과와 build_summary, feature_info를 기준으로 한다.
8. used_in_ml 기본값은 True다.
   - 실험용으로 catalog에만 남기고 모델 입력에서 제외하려면 used_in_ml=False를 명시한다.
   - 학습 전 ml_feature_columns.csv를 확인해야 한다.
   
운영 권장 방식
----------------
- 노트북에서는 항상 *_spec helper를 사용한다.
- FeatureSpec(...) 직접 생성은 특별한 이유가 있을 때만 사용한다.
- feature build 전에 validate_feature_specs()로 spec 목록을 검증한다.
- full data 실행 전에는 생성될 feature column 목록과 used_in_ml 값을 확인한다.
- 새 feature를 추가하면 feature_catalog.csv와 feature_info.csv를 함께 검토한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Tuple

import pandas as pd


# =============================================================================
# 1. 공통 메타 컬럼
# =============================================================================
# 모든 feature frame은 학습 모듈 연결을 위해 tx_id, split, label을 함께 가진다.
# 실제 모델 입력 feature는 이 3개를 제외한 FeatureSpec.output_col 목록이다.

# 왜 상수로 박아두는가
# --------------------
# - ml_00_fb_operations.execute_feature_specs()가 최종 feature_frame을 조립할 때 [META_COLUMNS] + [생성 feature columns] 순서로 컬럼을 배치.
# - Tuple로 생성해서 수정 불가 하도록 설정 
META_COLUMNS: Tuple[str, ...] = ("tx_id", "split", "label")


# =============================================================================
# 2. FeatureSpec / FeatureOpResult — 핵심 자료구조
# =============================================================================

@dataclass(frozen=True)
class FeatureSpec:
    """
    사용자가 노트북에서 직접 검토/수정하는 feature 생성 선언
    이 객체는 "데이터" 임, 함수도 아니고 실행 가능한 무언가도 아니다.
    그래서 다음이 가능하다:
      - 노트북에서 list/tuple로 묶어 보관, 슬라이싱, 정렬
      - feature_catalog.csv로 그대로 직렬화 (재현성 메타데이터)
      - 실행 전에 spec 목록만 검사해서 오류를 조기에 차단
      - 같은 spec을 ML-00 / ML-01 실험에 재사용

    frozen=True 인 이유
    -------------------
    생성 후 mutate를 방지,  spec 목록을 만든 뒤 build 중간에 실수로 바뀌는 에러 없도록

    주요 필드
    ---------
    operation:  실행할 연산 이름. `ml_00_fb_operations.OPERATION_REGISTRY`에 등록되어 있어야 한다.
    
    output_col: 생성될 feature 컬럼명. 중복되면 feature build가 즉시 실패한다.
    input_cols: operation이 요구하는 역할명과 실제 입력 컬럼명의 매핑이다.
        예: {"input_col": "amount"}, {"entity_col": "sender_account_id", ...}
        
    params: window, agg, part 같은 operation별 추가 파라미터다.
    
    used_in_ml: `ml_feature_columns.csv`의 used_in_ml 값으로 저장된다.
    """
    
    # --- 실행 결정 필드 (operation/output/입력 매핑) ---

    # 어떤 operation을 실행할지 결정 예: "current_value", "log1p", "rolling_agg".
    # 이 문자열이 ml_00_fb_operations.OPERATION_REGISTRY의 key와 정확히 매치돼야 한다.
    # 등록되지 않은 이름이면 ml_00_fb_operations.run_operation()에서 즉시 에러.
    operation: str

    # 최종 생성될 feature 컬럼명
    # 학습 단계가 이 이름을 컬럼 key로 사용하므로, validate_feature_specs에서
    # 중복을 강제로 차단 (같은 컬럼명이 둘이면 어느 것이 들어갈지 모호함)
    output_col: str

    # operation이 사용할 입력 컬럼명
    #   key   = role(역할 이름. operation 함수가 "내가 어떤 입력을 원한다"고 정한 이름)
    #   value = 실제 입력 DataFrame 컬럼명 (logical column. 보통 노트북의 COLUMN_MAP key)
    # 예시:
    #   log1p          → {"input_col": "amount"}
    #   rolling_agg    → {"entity_col": "sender_account_id",
    #                     "timestamp_col": "timestamp",
    #                     "value_col": "amount"}
    # 같은 operation이라도 input_cols가 다르면 다른 feature가 된다.
    input_cols: Mapping[str, str]

    # operation별 선택 파라미터다. 예: {"window": "7D", "agg": "sum"}.
    # 각 operation 함수가 _require_allowed_params()로 허용 키를 검사하므로,
    # 잘못된 키를 넣으면 실행 시 즉시 거부된다.
    # field(default_factory=dict): 가변 객체(dict)를 기본값으로 쓸 때의 dataclass 관용구.
    #   default={} 를 쓰면 모든 인스턴스가 동일한 dict를 공유해 사고가 난다.
    params: Mapping[str, Any] = field(default_factory=dict)

    # --- 설명/관리용 메타데이터 (feature_catalog.csv로 저장) ---
    family: str = "unspecified"
    description: str = ""
    aml_typology: str = "unspecified"     # AML 도메인에서 어떤 유형의 행동 패턴을 노리는 feature인지.
    entity_scope: str = "transaction_row" # feature가 어떤 단위를 보는지. 보통 "transaction_row" 또는 "account".
    direction: str = "current"            # 시간 방향. "current"(현재 row), "past_window"(과거 window 집계),
    leakage_policy: str = "unspecified"   # 데이터 누수 위험. "current-row-only", "past-only", "encoder-fitted-on-train-only" 등.
    computational_cost: str = "low"
    
    # 모델 입력으로 쓸지 여부. ml_feature_columns.csv에 그대로 저장된다.
    # False로 두면 catalog에는 남지만 학습 모듈에서 제외 가능.
    # (실험적으로 만들어 보고는 싶은데 모델에 넣고 싶지 않은 경우 등)
    used_in_ml: bool = True

    def required_columns(self) -> list[str]:
        """
        이 FeatureSpec 하나가 요구하는 실제 입력 컬럼 목록을 반환

        input_cols dict의 value(= 실제 컬럼명)들만 모은다. key(역할명)는 제외.
        예: input_cols = {"entity_col": "sender_account_id", "timestamp_col": "timestamp", "value_col": "amount"}
            → ["sender_account_id", "timestamp", "amount"]

        dict.fromkeys(...)는 입력 순서를 유지하면서 중복을 제거
        (예: entity_col과 counterparty_col이 같은 컬럼을 가리키는 일은 거의 없지만, 혹시 그런 경우에도 한 번만 반환)
        """
        return list(dict.fromkeys(str(column) for column in self.input_cols.values()))


@dataclass(frozen=True)
class FeatureOpResult:
    """
    operation 하나가 반환하는 표준 결과 객체
    표준화 이유
    ---------------
    9개 operation(current_value, log1p, rolling_agg, ...)이 각자 다른 형태로 결과를 반환하면
    execute_feature_specs()에서 분기 처리가 폭발, 모든 operation이 이 객체 하나로 반환하도록 강제 
    호출부 코드는 result.features / result.feature_info / result.artifacts 세 가지만 다루면 되게 한다.

    features:
        생성 feature 컬럼만 담은 DataFrame이다. 보통 1개 컬럼
        (현재 구조는 "1 spec = 1 output column" 이지만, 여러 컬럼을 반환하도록 확장할 여지를 위해 DataFrame 타입으로 설계.)
    feature_info:
        생성 컬럼의 missing/inf/분포/파라미터 정보를 담은 DataFrame이다. feature_catalog.csv의 컬럼과 유사한 형태지만, operation 실행 결과에 따라 동적으로 채워짐
        ml_00_fb_operations._feature_info()가 모든 operation에서 같은 schema로 채워 넣는다.
    artifacts:
        category mapping, unknown category summary 같은 부가 산출물, category_code operation처럼 "train fit 결과를 별도 CSV로 남겨야 하는"
        경우에만 사용된다. 대부분의 operation은 빈 dict를 반환.
    """
    features: pd.DataFrame
    feature_info: pd.DataFrame
    artifacts: Mapping[str, Any] = field(default_factory=dict)


def _clean_name(value: str, field_name: str) -> str:
    """
    빈 문자열을 조용히 허용하지 않기 위한 내부 검증 함수

    왜 별도 함수로 빼는가
    --------------------
    validate_feature_specs()에서 operation 이름, output_col, input role,
    input column 등 여러 위치에서 같은 검사를 반복한다. 한 곳에 모아 두면
    "공백만 있는 이름을 어떻게 처리할지" 정책이 한 줄에서 결정된다.

    동작
    ----
    - str()로 변환 후 strip() → "  amount  " 같은 입력도 허용
    - 결과가 빈 문자열이면 어느 필드가 문제인지 명시한 ValueError로 중단
    - 정상이면 clean된 이름을 반환 (호출자가 그대로 써도 안전한 값)
    """


    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"FeatureSpec {field_name} must not be empty.")
    return cleaned


def validate_feature_specs(feature_specs: Tuple[FeatureSpec, ...]) -> None:
    """
    FeatureSpec 목록을 실행 전에 검증한다.
    이 함수의 목적
    ---------------
    feature build는 여러 FeatureSpec을 순서대로 실행해 feature frame을 만든다.
    spec 목록에 기본 오류가 있으면 계산 중간이나 저장 단계에서 애매하게 실패할 수 있다.
    그래서 실제 operation 실행 전에, spec 목록만 보고 확인 가능한 오류를 먼저 차단한다.
    여기서 막는 오류
    ---------------
    - FeatureSpec,output_col 목록이 비어 있음
    - output_col 중복
    - operation 이름이 비어 있음
    - input_cols, input role, input column 이 비어 있음
    여기서 막지 않는 오류
    --------------------
    - operation 이름이 registry에 실제 등록되어 있는지
    - params key/value가 해당 operation에 맞는지
    - input column이 실제 DataFrame에 존재하는지
    - output_col 이름은 다르지만 의미가 같은 feature 중복
    위 항목들은 `ml_00_fb_operations.py`의 operation 검증,
    schema/COLUMN_MAP 해석 단계, feature_info/catalog 사후 점검에서 확인한다.
    반환값
    ------
    없음. 문제가 있으면 ValueError를 발생시킨다.
    """
    # spec 목록이 비어 있으면 중단한다.
    if not feature_specs:
        raise ValueError("feature_specs must not be empty. Add at least one FeatureSpec.")
    # 모든 output_col을 먼저 정리/검증한다.
    # _clean_name()은 문자열 앞뒤 공백을 제거하고, 빈 문자열이면 ValueError를 발생시킨다.
    output_cols = [_clean_name(spec.output_col, "output_col") for spec in feature_specs]
    # 같은 output_col이 두 번 이상 나오면 최종 feature frame에서 컬럼 충돌이 발생한다.
    # set comprehension으로 중복 후보를 모으고 sorted()로 에러 메시지 순서를 안정화한다.
    duplicated = sorted({column for column in output_cols if output_cols.count(column) > 1})
    if duplicated:
        raise ValueError(
            "Feature build failed: duplicate output columns in feature_specs. "
            f"duplicated={duplicated}, fix=Use unique output_col values."
        )
    # 각 spec 내부의 필수 선언값을 확인한다.
    # index를 에러 메시지에 넣으면 노트북에서 긴 spec 목록 중 어느 항목이 문제인지 찾기 쉽다.
    for index, spec in enumerate(feature_specs):
        # operation은 OPERATION_REGISTRY와 연결되는 문자열 key다.
        # 여기서는 빈 값만 막고, 실제 registry 존재 여부는 run_operation()에서 확인한다.
        _clean_name(spec.operation, f"operation at index {index}")
        # input_cols가 없으면 operation이 어떤 입력 컬럼을 써야 하는지 알 수 없다.
        # output_col을 같이 보여주면 문제가 난 feature를 빠르게 찾을 수 있다.
        if not spec.input_cols:
            raise ValueError(
                "Feature build failed: FeatureSpec.input_cols must not be empty. "
                f"index={index}, output_col={spec.output_col!r}"
            )
        # input_cols는 role -> column 매핑이다.
        # role 예: "input_col", "entity_col", "timestamp_col"
        # column 예: "amount", "sender_account_id", "timestamp"
        for role, column in spec.input_cols.items():
            # role이 비어 있으면 operation 함수가 입력의 의미를 해석할 수 없다.
            _clean_name(str(role), f"input role at index {index}")
            # column이 비어 있으면 schema 매핑이나 DataFrame 컬럼 조회 단계에서 애매하게 실패한다.
            # 따라서 여기서 어느 role의 column이 비었는지 명시하고 중단한다.
            _clean_name(str(column), f"input column for role {role!r} at index {index}")


def required_input_columns(
    feature_specs: Tuple[FeatureSpec, ...],
    extra_columns: Optional[Iterable[str]] = None,
) -> list[str]:
    """선택된 FeatureSpec 전체가 요구하는 입력 컬럼 목록을 중복 없이 반환한다."""

    validate_feature_specs(feature_specs)
    required: list[str] = []
    if extra_columns is not None:
        required.extend(str(column) for column in extra_columns)
    for spec in feature_specs:
        required.extend(spec.required_columns())
    return list(dict.fromkeys(required))


def feature_columns(feature_specs: Tuple[FeatureSpec, ...]) -> list[str]:
    """FeatureSpec 실행 순서대로 생성될 feature 컬럼명을 반환한다."""

    validate_feature_specs(feature_specs)
    return [spec.output_col for spec in feature_specs]


DEFAULT_TIMEHIST_WINDOWS: Tuple[Tuple[str, str], ...] = (
    ("w1h", "1h"),
    ("w6h", "6h"),
    ("w1d", "1d"),
    ("w3d", "3d"),
    ("w7d", "7d"),
)


def current_value_spec(
    input_col: str,
    output_col: str,
    *,
    family: str = "raw_value",
    description: str = "Current row numeric value.",
    aml_typology: str = "current_row_context",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """현재 row의 숫자 컬럼을 그대로 feature로 쓰겠다는 선언을 만든다."""

    return FeatureSpec(
        operation="current_value",
        output_col=output_col,
        input_cols={"input_col": input_col},
        family=family,
        description=description,
        aml_typology=aml_typology,
        leakage_policy="current-row-only",
        used_in_ml=used_in_ml,
    )


def log1p_spec(
    input_col: str,
    output_col: str,
    *,
    family: str = "raw_amount",
    description: str = "Log1p transform of a non-negative numeric column.",
    aml_typology: str = "large_amount",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """음수가 아닌 숫자 컬럼에 log1p 변환을 적용하는 feature 선언을 만든다."""

    return FeatureSpec(
        operation="log1p",
        output_col=output_col,
        input_cols={"input_col": input_col},
        family=family,
        description=description,
        aml_typology=aml_typology,
        leakage_policy="current-row-only; input must be non-negative",
        used_in_ml=used_in_ml,
    )


def datetime_part_spec(
    input_col: str,
    output_col: str,
    *,
    part: str,
    family: str = "raw_time",
    description: str = "Datetime component extracted from current row timestamp.",
    aml_typology: str = "temporal_context",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """timestamp 컬럼에서 hour/dayofweek/weekend 같은 시간 파생 feature 선언을 만든다."""

    return FeatureSpec(
        operation="datetime_part",
        output_col=output_col,
        input_cols={"input_col": input_col},
        params={"part": part},
        family=family,
        description=description,
        aml_typology=aml_typology,
        leakage_policy="current-row-only",
        used_in_ml=used_in_ml,
    )


def category_code_spec(
    input_col: str,
    output_col: str,
    *,
    family: str = "raw_category",
    description: str = "Train-only integer encoding of a categorical column.",
    aml_typology: str = "categorical_context",
    used_in_ml: bool = True,
    unknown_policy: str = "encode_-1_and_report",
) -> FeatureSpec:
    """범주형 컬럼을 train-only integer code로 변환하는 feature 선언을 만든다."""

    return FeatureSpec(
        operation="category_code",
        output_col=output_col,
        input_cols={"input_col": input_col},
        params={"unknown_policy": unknown_policy},
        family=family,
        description=description,
        aml_typology=aml_typology,
        leakage_policy="encoder-fitted-on-train-only",
        used_in_ml=used_in_ml,
    )


def rolling_agg_spec(
    entity_col: str,
    timestamp_col: str,
    value_col: str,
    output_col: str,
    *,
    window: str,
    agg: str,
    fill_value: float = 0.0,
    family: str = "temporal_rolling",
    description: str = "Past-only rolling aggregation by entity.",
    aml_typology: str = "burst_or_structuring",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """entity별 과거 window rolling aggregation feature 선언을 만든다."""

    return FeatureSpec(
        operation="rolling_agg",
        output_col=output_col,
        input_cols={"entity_col": entity_col, "timestamp_col": timestamp_col, "value_col": value_col},
        params={"window": window, "agg": agg, "closed": "left", "fill_value": fill_value},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="account",
        direction="past_window",
        leakage_policy="past-only; rolling window uses closed=left",
        computational_cost="medium",
        used_in_ml=used_in_ml,
    )


def recency_seconds_spec(
    entity_col: str,
    timestamp_col: str,
    output_col: str,
    *,
    fill_value: float = -1.0,
    family: str = "recency",
    description: str = "Seconds since the entity's previous transaction; first transaction is filled with a sentinel.",
    aml_typology: str = "recency",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """entity 기준 직전 과거 거래와의 시간 차이 feature 선언을 만든다."""

    return FeatureSpec(
        operation="recency_seconds_since_last",
        output_col=output_col,
        input_cols={"entity_col": entity_col, "timestamp_col": timestamp_col},
        params={"fill_value": fill_value},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="account",
        direction="past_event",
        leakage_policy="past-only; same-timestamp rows excluded; first transaction uses sentinel fill_value; pair with is_first flag",
        computational_cost="medium",
        used_in_ml=used_in_ml,
    )


def is_first_entity_spec(
    entity_col: str,
    timestamp_col: str,
    output_col: str,
    *,
    family: str = "flag",
    description: str = "Flag indicating no previous transaction for the entity.",
    aml_typology: str = "cold_start",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """entity 기준 과거 거래가 없으면 1인 첫 거래 flag feature 선언을 만든다."""

    return FeatureSpec(
        operation="is_first_by_entity",
        output_col=output_col,
        input_cols={"entity_col": entity_col, "timestamp_col": timestamp_col},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="account",
        direction="past_event",
        leakage_policy="past-only; same-timestamp rows excluded",
        computational_cost="low",
        used_in_ml=used_in_ml,
    )


def fan_in_spec(
    receiver_col: str,
    sender_col: str,
    timestamp_col: str,
    output_col: str,
    *,
    window: str,
    family: str = "graph_fan",
    description: str = "Past-only unique inbound counterparties by receiver.",
    aml_typology: str = "gather",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """수취 entity 기준 과거 window 내 고유 송금자 수 feature 선언을 만든다."""

    return FeatureSpec(
        operation="fan_in",
        output_col=output_col,
        input_cols={"entity_col": receiver_col, "counterparty_col": sender_col, "timestamp_col": timestamp_col},
        params={"window": window},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="receiver_account",
        direction="incoming",
        leakage_policy="past-only; same-timestamp rows excluded",
        computational_cost="medium",
        used_in_ml=used_in_ml,
    )


def fan_out_spec(
    sender_col: str,
    receiver_col: str,
    timestamp_col: str,
    output_col: str,
    *,
    window: str,
    family: str = "graph_fan",
    description: str = "Past-only unique outbound counterparties by sender.",
    aml_typology: str = "scatter",
    used_in_ml: bool = True,
) -> FeatureSpec:
    """송금 entity 기준 과거 window 내 고유 수취자 수 feature 선언을 만든다."""

    return FeatureSpec(
        operation="fan_out",
        output_col=output_col,
        input_cols={"entity_col": sender_col, "counterparty_col": receiver_col, "timestamp_col": timestamp_col},
        params={"window": window},
        family=family,
        description=description,
        aml_typology=aml_typology,
        entity_scope="sender_account",
        direction="outgoing",
        leakage_policy="past-only; same-timestamp rows excluded",
        computational_cost="medium",
        used_in_ml=used_in_ml,
    )


def timehist_rolling_specs(
    entity_name: str,
    direction: str,
    entity_col: str,
    *,
    timestamp_col: str = "timestamp",
    value_col: str = "amount",
    windows: Tuple[Tuple[str, str], ...] = DEFAULT_TIMEHIST_WINDOWS,
    used_in_ml: bool = True,
) -> Tuple[FeatureSpec, ...]:
    """
    timehist 계열 rolling count/sum/mean feature 선언 묶음을 만든다.

    반환되는 각 원소는 여전히 FeatureSpec 1개 = output column 1개 규칙을 따른다.
    """

    entity_name = _clean_name(entity_name, "entity_name")
    direction = _clean_name(direction, "direction")
    entity_col = _clean_name(entity_col, "entity_col")
    timestamp_col = _clean_name(timestamp_col, "timestamp_col")
    value_col = _clean_name(value_col, "value_col")
    if not windows:
        raise ValueError("timehist_rolling_specs windows must not be empty.")

    specs: list[FeatureSpec] = []
    prefix = f"timehist__{entity_name}__{direction}"
    for suffix, window in windows:
        suffix = _clean_name(suffix, "window suffix")
        window = _clean_name(window, "window")
        specs.extend(
            [
                rolling_agg_spec(
                    entity_col,
                    timestamp_col,
                    value_col,
                    f"{prefix}__tx_count__count__{suffix}",
                    window=window,
                    agg="count",
                    family="timehist",
                    description="Past-only transaction count by entity and window.",
                    aml_typology="velocity",
                    used_in_ml=used_in_ml,
                ),
                rolling_agg_spec(
                    entity_col,
                    timestamp_col,
                    value_col,
                    f"{prefix}__amount__sum__{suffix}",
                    window=window,
                    agg="sum",
                    family="timehist",
                    description="Past-only amount sum by entity and window.",
                    aml_typology="amount_burst",
                    used_in_ml=used_in_ml,
                ),
                rolling_agg_spec(
                    entity_col,
                    timestamp_col,
                    value_col,
                    f"{prefix}__amount__mean__{suffix}",
                    window=window,
                    agg="mean",
                    family="timehist",
                    description="Past-only amount mean by entity and window.",
                    aml_typology="average_behavior",
                    used_in_ml=used_in_ml,
                ),
            ]
        )

    result = tuple(specs)
    validate_feature_specs(result)
    return result


def recency_feature_specs(
    entity_name: str,
    direction: str,
    entity_col: str,
    *,
    timestamp_col: str = "timestamp",
    used_in_ml: bool = True,
) -> Tuple[FeatureSpec, ...]:
    """
    recency seconds와 first-transaction flag feature 선언 묶음을 만든다.

    반환되는 각 원소는 여전히 FeatureSpec 1개 = output column 1개 규칙을 따른다.
    """

    entity_name = _clean_name(entity_name, "entity_name")
    direction = _clean_name(direction, "direction")
    entity_col = _clean_name(entity_col, "entity_col")
    timestamp_col = _clean_name(timestamp_col, "timestamp_col")
    result = (
        recency_seconds_spec(
            entity_col,
            timestamp_col,
            f"recency__{entity_name}__{direction}__seconds_since_last",
            used_in_ml=used_in_ml,
        ),
        is_first_entity_spec(
            entity_col,
            timestamp_col,
            f"flag__{entity_name}__{direction}__is_first_tx",
            used_in_ml=used_in_ml,
        ),
    )
    validate_feature_specs(result)
    return result


def default_feature_specs() -> Tuple[FeatureSpec, ...]:
    """
    기본 current-row feature 10개를 반환한다.

    이 기본값은 smoke build와 간단한 baseline 입력 생성을 위한 조합이다.
    graph/rolling feature를 추가하려면 노트북에서 별도 FeatureSpec 목록을 직접 만든다.
    """

    return (
        log1p_spec("amount", "amount__current__log1p"),
        current_value_spec("amount", "amount__current__value", family="raw_amount", aml_typology="large_amount"),
        category_code_spec(
            "sender_bank_id",
            "cat__from_bank__code",
            family="raw_bank",
            aml_typology="cross_institution_flow",
        ),
        category_code_spec(
            "payment_currency",
            "cat__payment_currency__code",
            family="raw_currency",
            aml_typology="currency_conversion",
        ),
        category_code_spec(
            "payment_format",
            "cat__payment_format__code",
            family="raw_payment_format",
            aml_typology="payment_channel",
        ),
        category_code_spec(
            "receiving_currency",
            "cat__receiving_currency__code",
            family="raw_currency",
            aml_typology="currency_conversion",
        ),
        category_code_spec(
            "receiver_bank_id",
            "cat__to_bank__code",
            family="raw_bank",
            aml_typology="cross_institution_flow",
        ),
        datetime_part_spec("timestamp", "time__row__dayofweek", part="dayofweek"),
        datetime_part_spec("timestamp", "time__row__hour", part="hour"),
        datetime_part_spec("timestamp", "time__row__is_weekend", part="is_weekend"),
    )
