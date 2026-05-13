"""
FeatureSpec 선언 모듈

이 파일의 역할
----------------
1. 사용자가 노트북에서 조합할 `FeatureSpec` 자료구조를 정의한다.
2. operation 실행 결과를 담는 `FeatureOpResult` 자료구조를 정의한다.
3. 자주 쓰는 feature 선언 함수(`log1p_spec`, `fan_in_spec` 등)를 제공한다.
4. 선택된 FeatureSpec 목록의 중복/빈 값 같은 기본 오류를 실행 전에 차단한다.

중요한 설계 원칙
----------------
- 이 파일은 feature를 계산하지 않는다.
- 이 파일은 어떤 입력 컬럼을 써서 어떤 출력 컬럼을 만들지 "선언"만 한다.
- 실제 계산은 `fb_operations.py`가 담당한다.
- Stage 이름이 아니라 FeatureSpec 목록이 생성 feature를 결정한다.

전체 구조 요약
----------------
이 모듈은 크게 세 층으로 나뉜다.

  [Layer 1] 자료구조        : FeatureSpec, FeatureOpResult (dataclass)
                              ↓ "어떤 데이터를 들고 다닐 것인가"

  [Layer 2] 검증 / 조회 함수: validate_feature_specs, required_input_columns, feature_columns
                              ↓ "spec 목록을 안전하게 다루기 위한 보조 함수"

  [Layer 3] 헬퍼 함수       : current_value_spec, log1p_spec, ... fan_out_spec
                              ↓ "FeatureSpec(...)을 직접 생성자로 부르지 않고 있는 이름의 함수로 만들기 위한 편의 layer"
                              ↓ default_feature_specs()는 이들의 조합 예시

선언 vs 계산의 분리 (설계 의도)
----------------------------------------
일반적인 feature engineering 코드는 "함수 = feature 1개"로 구현
이 경우 새 feature를 만들 때마다 새 함수가 생기고, 어떤 feature를 켜고 끄려면 호출하는 쪽 코드를 수정해야 함 
=> 실험마다 feature 조합을 갈아끼우기 불편

이 모듈은 다음과 같이 설계 하였다.
  - FeatureSpec은 "무엇을 만들지"만 적은 데이터 (= 직렬화/저장/재사용 가능)
  - fb_operations.py의 OPERATION_REGISTRY가 "어떻게 만들지"를 담는다
  - 둘을 연결하는 키는 spec.operation 문자열 하나

강점: 
  - 노트북에서 FeatureSpec 목록만 바꿔 ML-00 / ML-01 실험을 갈아끼울 수 있다.
  - 같은 spec을 feature_catalog.csv에 그대로 직렬화해 재현 메타데이터로 쓴다.
  - 실행 전에 spec 목록만 보고도 "어떤 feature가 만들어질지" 검증할 수 있다.
  
리스크 및 관리방법: 
  본 모듈의 리스크 대부분은 "정적으로 잡히지 않는 문자열/메타데이터 규약"에서 발생한다.
  핵심 방어선은 두 가지:
    (1) 노트북에서는 항상 *_spec 헬퍼 함수를 사용한다 (FeatureSpec 직접 생성 금지).
    (2) 데이터 로드 전에 validate_feature_specs + fb_operations의 params 검증으로 조용한 에러 통과를 차단
  실측이 필요한 항목(R8)이나 의미 충돌(R4)은 모듈 단독으로는 막을 수 없으므로 feature_info.csv / feature_catalog.csv 사후 검토 단계와 함께 운영 해야 함 
  
   [R1] 문자열 키 결합 (string-based coupling)
       위험: spec.operation은 "rolling_agg" 같은 평문 문자열로 fb_operations.OPERATION_REGISTRY와 연결된다. 
            operation 이름을 한쪽에서 오타로 적거나(예: "roling_agg"), 한쪽만 이름을 바꾸면
             정적 분석 도구가 잡지 못한다.
       관리: (a) fb_specs의 모든 헬퍼 함수(rolling_agg_spec 등)가 operation
                문자열을 내부에 박아두므로, 사용자가 직접 FeatureSpec(...)을
                호출하지 않는 한 오타 위험이 줄어든다 → 노트북에서는 항상
                *_spec 헬퍼만 쓸 것.
             (b) fb_operations.run_operation()이 registry에 없는 이름을
                즉시 ValueError로 차단하므로, 잘못된 키는 런타임 시작 직후 잡힌다.
             (c) 향후 enum/Literal 타입으로 강화하면 타입 체커가 잡을 수 있음
                (현재는 미적용 — 트레이드오프: 헬퍼 추가 시 두 곳을 동기화해야 함).

  [R2] 헬퍼 미사용 시 catalog 메타데이터 누락
       위험: 사용자가 노트북에서 헬퍼(log1p_spec 등) 대신 FeatureSpec(...)을
             직접 호출하면 leakage_policy / family / aml_typology 등 catalog
             필드가 기본값("unspecified" / "")으로 남는다. 실행은 정상이지만
             feature_catalog.csv에서 사람이 검토할 정보가 비어 사후 추적이 어렵다.
       관리: (a) 노트북 예시(default_feature_specs, ML00_FEATURE_SPECS,
                ML01_FEATURE_SPECS)에서 헬퍼 사용을 표준 패턴으로 보여준다.
             (b) catalog 검토 시 family="unspecified" / leakage_policy="unspecified"
                항목을 우선 확인하는 운영 체크를 둔다 (현재는 사람 검토).
             (c) 더 강하게 막으려면 validate_feature_specs에서 "unspecified" 값을
                금지하는 strict 모드를 추가할 수 있다 (현재는 미적용 — 트레이드오프:
                실험 단계에서 가벼운 spec을 만들기 어려워짐).

  [R3] frozen=True 우회 가능성
       위험: dataclass(frozen=True)는 일반 대입(self.x = ...)을 막지만,
             object.__setattr__()로 우회 가능하다. 또한 input_cols / params에
             담긴 dict 자체는 frozen이 아니므로 .update() / __setitem__으로
             내용을 바꿀 수 있다. spec이 build 도중에 mutate되면 재현성이 깨진다.
       관리: (a) 코드 컨벤션상 spec은 노트북에서 한 번 만든 뒤 읽기만 한다.
                fb_build / fb_operations 어디서도 spec을 수정하지 않는다.
             (b) input_cols / params 내용을 mutate해야 하는 일이 생기면,
                기존 spec을 dataclasses.replace(spec, params={...})로 새로 만든다
                (in-place 수정 금지).
             (c) 향후 input_cols / params를 MappingProxyType 또는 frozendict로
                감싸 mutate를 원천 차단할 수 있다 (현재는 미적용 — Mapping 타입의
                관용성을 유지하기 위한 의도적 선택).

  [R4] output_col 충돌은 막지만 의미 충돌은 못 막음
       위험: validate_feature_specs는 동일한 output_col 문자열 중복만 차단한다.
             서로 다른 이름이지만 의미가 같은 feature(예: "amount__current__value"와
             "amount__raw")가 둘 다 들어가도 통과한다. 모델에 정보가 중복되어
             특히 선형 모델의 계수 해석이 불안정해진다.
       관리: (a) feature 명명 규칙을 통일한다 (예: "<source>__<scope>__<transform>").
                노트북에서 prefix 패턴을 일관되게 사용.
             (b) feature_info.csv의 correlation / unique_count 점검을 통해
                중복성 높은 컬럼을 사후 탐지한다.
             (c) catalog의 family / aml_typology 같은 그룹 라벨을 활용해
                같은 family에 동일 의미가 둘 이상 있는지 검토한다.

  [R5] params 키/타입 오류는 런타임까지 미발견
       위험: params는 Mapping[str, Any]로 자유 형식이다. 잘못된 키
             (예: rolling_agg에 "windwo"를 넣음)나 잘못된 타입(예: window=7 정수)을
             넣으면 fb_specs 단계에서는 통과한다.
       관리: (a) fb_operations의 각 op 함수가 _require_allowed_params()로
                허용 키를 검사하고, _parse_window() / _require_roles() 등이
                값/타입을 strict 검증한다 → 데이터 로드 전에 실패시키는 정책.
             (b) 헬퍼 함수(rolling_agg_spec 등)는 키워드 인자를 받아 params를
                내부에서 구성하므로, 헬퍼를 쓰면 키 오타가 발생할 수 없다.
                → 노트북에서는 헬퍼 사용을 권장 (R1 / R2 관리방법과 같은 맥락).

  [R6] required_columns의 컬럼 이름 매칭은 logical level
       위험: spec.required_columns()가 반환하는 값은 노트북이 쓰는 "logical 이름"
             (예: "amount", "sender_account_id")이다. 실제 parquet 컬럼이
             "Amount Paid" / "From Account" 같이 다르면 이 모듈에서는 못 잡는다.
       관리: (a) 매칭은 fb_schema.resolve_requested_columns()가 담당한다.
                노트북의 COLUMN_MAP 최우선 → fb_schema.COLUMN_CANDIDATES fallback 순.
             (b) COLUMN_MAP을 가능한 한 명시적으로 채워 fallback에 의존하지 않는다
                (CLAUDE.md의 "추론보다 명시" 원칙).
             (c) COLUMN_CANDIDATES도 못 찾으면 fb_schema에서 ValueError로 중단되어
                데이터 로드 전에 실패한다.

  [R7] 헬퍼와 default_feature_specs의 사일런트 드리프트
       위험: 노트북의 ML00_FEATURE_SPECS와 default_feature_specs()가 "같다"고
             가정하는 주석이 있지만, 코드로 강제되지 않는다. 한쪽만 수정되면
             재현 결과가 어긋난다.
       관리: (a) 두 정의가 정말로 항상 같아야 한다면 노트북에서
                `ML00_FEATURE_SPECS = default_feature_specs()`로 직접 참조한다.
             (b) 의도적으로 분리하려면 (예: 노트북에서는 ML00 + α를 쓰고 싶다),
                둘이 다르다는 사실을 노트북/README에 명시한다.
             (c) feature_catalog.csv를 실험 간 diff로 비교해 의도치 않은 변경을 탐지.

  [R8] computational_cost / aml_typology 등은 작성자의 짐작
       위험: computational_cost("low"/"medium"/"high")는 실측이 아니다.
             실제 대용량(IBM AML Large 176M행)에서는 "medium"으로 표시된
             rolling/fan 계열이 병목이 될 수 있다.
       관리: (a) 실제 측정은 fb_operations 실행 후 feature_info.csv와
                build_summary.json의 row_counts / 실행 시간 메모로 별도 기록한다.
             (b) 본 모듈의 라벨은 사람이 catalog를 훑을 때의 "1차 신호"로만
                활용하고, 의사결정(예: 어떤 feature를 끌지)은 실측 기반으로 한다.
             (c) 향후 build_summary.json에 per-operation duration / peak memory를
                자동 기록하도록 확장 가능 (현재는 미적용).

  [R9] used_in_ml=True 기본값
       위험: 모든 헬퍼가 used_in_ml=True를 기본값으로 둔다. 실험 목적으로 추가한
             feature를 catalog에만 남기고 모델에는 넣지 않으려면 사용자가 명시적으로
             False를 줘야 한다. 잊으면 의도치 않게 모델 입력에 들어간다.
       관리: (a) ml_feature_columns.csv의 used_in_ml 컬럼을 학습 코드 진입 전에
                사람이 한 번 검토한다 (노트북 "10. full data 실행 전 체크리스트" 항목).
             (b) 실험용 spec을 만들 때는 used_in_ml=False를 명시하는 컨벤션을 둔다. 
  
  
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
# - fb_operations.execute_feature_specs()가 최종 feature_frame을 조립할 때 [META_COLUMNS] + [생성 feature columns] 순서로 컬럼을 배치.
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
    operation:  실행할 연산 이름. `fb_operations.OPERATION_REGISTRY`에 등록되어 있어야 한다.
    
    output_col: 생성될 feature 컬럼명. 중복되면 feature build가 즉시 실패한다.
    input_cols: operation이 요구하는 역할명과 실제 입력 컬럼명의 매핑이다.
        예: {"input_col": "amount"}, {"entity_col": "sender_account_id", ...}
        
    params: window, agg, part 같은 operation별 추가 파라미터다.
    
    used_in_ml: `ml_feature_columns.csv`의 used_in_ml 값으로 저장된다.
    """
    
    # --- 실행 결정 필드 (operation/output/입력 매핑) ---

    # 어떤 operation을 실행할지 결정 예: "current_value", "log1p", "rolling_agg".
    # 이 문자열이 fb_operations.OPERATION_REGISTRY의 key와 정확히 매치돼야 한다.
    # 등록되지 않은 이름이면 fb_operations.run_operation()에서 즉시 에러.
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
        fb_operations._feature_info()가 모든 operation에서 같은 schema로 채워 넣는다.
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
    FeatureSpec 목록을 데이터 로드 전에 검증한다.

    여기서 막는 오류
    ---------------
    - FeatureSpec 목록이 비어 있음
    - output_col 중복
    - operation 이름이 비어 있음
    - input_cols가 비어 있음
    - input role 또는 input column이 빈 문자열
    """

    if not feature_specs:
        raise ValueError("feature_specs must not be empty. Add at least one FeatureSpec.")

    output_cols = [_clean_name(spec.output_col, "output_col") for spec in feature_specs]
    duplicated = sorted({column for column in output_cols if output_cols.count(column) > 1})
    if duplicated:
        raise ValueError(
            "Feature build failed: duplicate output columns in feature_specs. "
            f"duplicated={duplicated}, fix=Use unique output_col values."
        )

    for index, spec in enumerate(feature_specs):
        _clean_name(spec.operation, f"operation at index {index}")
        if not spec.input_cols:
            raise ValueError(
                "Feature build failed: FeatureSpec.input_cols must not be empty. "
                f"index={index}, output_col={spec.output_col!r}"
            )
        for role, column in spec.input_cols.items():
            _clean_name(str(role), f"input role at index {index}")
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
