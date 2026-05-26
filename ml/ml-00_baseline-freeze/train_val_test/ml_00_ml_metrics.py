"""
이진 AML 분류 평가 지표와 threshold 선택 공통 유틸리티 모듈

전체 흐름
1. y_true와 positive-class probability가 평가 가능한 형태인지 검증
2. validation split에서 F1 최대 threshold를 선택
3. 고정 threshold 기준 F1, Recall, Precision, Average Precision, confusion matrix를 계산
4. confusion matrix를 CSV 저장용 DataFrame으로 변환

중요한 전제
- 이 모듈은 XGBoost에 의존하지 않음
- 입력은 0/1 label과 positive-class probability만 필요
- Accuracy 단독 평가는 극단적 불균형 AML 문제에 부적절하므로 제공하지 않음
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


# -----------------------------------------------------------------------------
# 1. metric 계산 전 입력 검증
# -----------------------------------------------------------------------------
def validate_binary_scores(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    이진 label과 probability score가 metric 계산 가능한 형태인지 검증

    검사 항목
    1. y_true와 probabilities가 1차원인지 확인
    2. 두 배열 길이가 같은지 확인
    3. y_true가 0/1 label만 포함하는지 확인
    4. probability에 NaN/inf가 없는지 확인
    5. probability가 0~1 범위인지 확인
    6. metric 계산을 위해 양 class가 모두 존재하는지 확인
    """

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    if y.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional. shape={y.shape}")
    if p.ndim != 1:
        raise ValueError(f"probabilities must be 1-dimensional. shape={p.shape}")
    if len(y) != len(p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y)}, len(probabilities)={len(p)}")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError(f"y_true must contain only 0/1 labels. values={sorted(set(np.unique(y).tolist()))}")
    if not np.isfinite(p).all():
        raise ValueError("probabilities contain NaN or infinite values.")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be between 0 and 1.")
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes are required to compute binary metrics.")

    return y, p


# -----------------------------------------------------------------------------
# 2. validation threshold 선택
# -----------------------------------------------------------------------------
def select_threshold_by_f1(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """
    주어진 validation split에서 F1이 최대가 되는 threshold를 선택

    동작 의도
    - precision_recall_curve가 반환하는 후보 threshold를 순회
    - 각 threshold의 F1을 계산
    - F1이 가장 큰 threshold의 summary metric을 반환

    주의
    - 이 함수는 validation threshold 선택용이며 test threshold 재조정에 사용하면 안 됨
    """

    y, p = validate_binary_scores(y_true, probabilities)
    precision, recall, thresholds = precision_recall_curve(y, p)

    if thresholds.size == 0:
        threshold = 0.5
        return evaluate_at_threshold(y, p, threshold)["summary"]

    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds, dtype="float64"),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    threshold = float(thresholds[best_index])
    return evaluate_at_threshold(y, p, threshold)["summary"]


# -----------------------------------------------------------------------------
# 3. 고정 threshold 기준 metric 계산
# -----------------------------------------------------------------------------
def evaluate_at_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """
    하나의 고정 threshold에서 이진 분류 metric을 계산

    반환 항목
    - f1, recall, precision
    - average_precision: threshold-independent ranking metric
    - tn, fp, fn, tp: confusion matrix 구성요소
    """

    y, p = validate_binary_scores(y_true, probabilities)
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1. threshold={threshold}")

    predictions = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    summary = {
        "threshold": threshold,
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y, p)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    confusion = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return {
        "summary": summary,
        "confusion_matrix": confusion,
    }


# -----------------------------------------------------------------------------
# 4. confusion matrix CSV 저장 보조 함수
# -----------------------------------------------------------------------------
def confusion_matrix_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    """
    confusion matrix dict를 CSV 저장하기 쉬운 1행 DataFrame으로 변환
    """

    return pd.DataFrame([metrics["confusion_matrix"]])
