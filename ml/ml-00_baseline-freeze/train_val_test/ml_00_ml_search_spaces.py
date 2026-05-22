"""
ML-00 XGBoost 하이퍼파라미터 search space preset 정의 모듈

전체 흐름
1. tuning_step별 discrete hyperparameter 후보 목록을 XGB_SEARCH_SPACES에 정의
2. 사용 가능한 preset 이름을 list_search_spaces()로 확인
3. get_search_space(name)으로 특정 preset의 복사본을 가져옴

중요한 전제
- 이 모듈은 search space의 source of truth 역할만 수행
- 학습, validation, random sampling은 수행하지 않음
- 반환 시 복사본을 제공해 호출자가 원본 preset을 실수로 오염시키지 않도록 함
"""

from __future__ import annotations

from typing import Union

Number = Union[int, float]


# -----------------------------------------------------------------------------
# 1. XGBoost search space preset 정의
# -----------------------------------------------------------------------------
# 각 값은 random search에서 선택 가능한 discrete 후보 목록이다.
# 실제 trial sampling은 ml_00_ml_tune.py가 담당한다.
XGB_SEARCH_SPACES: dict[str, dict[str, list[Number]]] = {
    "model_select_minimal": {
        "n_estimators": [800, 1200],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1.0, 5.0, 10.0],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "reg_lambda": [1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.1],
        "gamma": [0.0, 0.1],
        "early_stopping_rounds": [50],
    },
    "feature_shortlist": {
        "n_estimators": [1200, 2000],
        "learning_rate": [0.02, 0.03, 0.05],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1.0, 3.0, 5.0, 10.0],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_lambda": [1.0, 3.0, 10.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "gamma": [0.0, 0.1, 1.0],
        "early_stopping_rounds": [50],
    },
    "final_tuning": {
        "n_estimators": [2000, 3000],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1.0, 3.0, 5.0, 10.0, 20.0],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [1.0, 3.0, 10.0, 30.0],
        "reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "gamma": [0.0, 0.1, 1.0, 5.0],
        "early_stopping_rounds": [50],
    },
}


# -----------------------------------------------------------------------------
# 2. preset 조회 함수
# -----------------------------------------------------------------------------
def list_search_spaces() -> list[str]:
    """
    사용 가능한 search space preset 이름을 정렬된 리스트로 반환
    """

    return sorted(XGB_SEARCH_SPACES)


def get_search_space(name: str) -> dict[str, list[Number]]:
    """
    지정한 search space preset의 복사본을 반환

    동작 의도
    - 없는 preset 이름이면 즉시 ValueError 발생
    - 내부 list를 복사해 호출자가 반환값을 수정해도 XGB_SEARCH_SPACES 원본은 유지
    """

    if name not in XGB_SEARCH_SPACES:
        raise ValueError(f"Unknown XGB search space: {name!r}. available={list_search_spaces()}")
    return {param: list(values) for param, values in XGB_SEARCH_SPACES[name].items()}
