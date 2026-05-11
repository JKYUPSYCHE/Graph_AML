"""
Validation split에서 threshold를 선택하고 validation metric 산출물을 저장하는 모듈

전체 흐름
1. ml_train.train_xgb()가 저장한 model.pkl, feature_columns.json, train_summary.json을 읽음
2. validation parquet를 학습 때 저장한 feature 순서로 로드
3. 모델이 예측한 positive-class probability를 계산
4. threshold_strategy에 따라 threshold를 선택
   - max_f1: validation split에서 F1이 최대가 되는 threshold 자동 선택
   - manual: 사용자가 지정한 manual_threshold를 고정 threshold로 사용
5. threshold.json, metrics_val.json, confusion_matrix_val.csv를 저장

중요한 전제
- threshold 선택은 validation split에서만 수행
- 기본 threshold 기준은 max_f1이며, 수동 고정이 필요할 때만 manual을 사용
- test split은 이 모듈에서 절대 읽거나 평가하지 않음
- feature_columns_hash로 학습 시점 feature 순서와 validation 입력 feature 순서를 연결
- test 단계에서는 여기서 저장한 threshold.json을 그대로 사용해야 하며, test set에서 threshold를 다시 고르면 데이터 누수 위험이 있음
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_io import (
    feature_columns_hash,
    label_summary,
    load_json,
    load_saved_feature_columns,
    load_split,
    resolve_project_path,
    save_json,
)
from ml_metrics import confusion_matrix_frame, evaluate_at_threshold, select_threshold_by_f1


# -----------------------------------------------------------------------------
# 1. Validation 실행 설정과 반환 결과 자료구조
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationConfig:
    """
    validation threshold 선택에 필요한 입력 경로와 실행 옵션 묶음

    경로 처리 방식
    - val_path, output_dir는 str 또는 Path로 받을 수 있음
    - 상대경로를 사용할 경우 project_root 기준으로 절대경로 변환

    주요 옵션
    - sample_rows: smoke/debug 전용. None이면 validation split 전체 사용
    - allow_nan: feature NaN 허용 여부. XGBoost 계열은 NaN 처리가 가능하지만 기본은 엄격 검증
    - overwrite: 기존 validation 산출물 덮어쓰기 여부. 기본은 False로 보호
    - threshold_strategy: threshold 선택 방식. 기본값은 "max_f1"
      - "max_f1": validation F1이 최대가 되는 threshold를 자동 선택
      - "manual": manual_threshold 값을 그대로 사용
    - manual_threshold: threshold_strategy="manual"일 때 사용할 고정 threshold

    threshold 관련 주의
    - manual_threshold는 0 이상 1 이하만 허용
    - manual_threshold는 threshold_strategy="manual"일 때만 지정 가능
    - max_f1은 validation 기준 선택이며, final test에서 threshold를 다시 고르면 안 됨
    """

    val_path: Path | str
    output_dir: Path | str
    project_root: Path | str | None = None
    label_col: str = "label"
    sample_rows: int | None = None
    allow_nan: bool = False
    overwrite: bool = False

    model_file_name: str = "model.pkl"
    feature_columns_file_name: str = "feature_columns.json"
    train_summary_file_name: str = "train_summary.json"
    threshold_file_name: str = "threshold.json"
    metrics_file_name: str = "metrics_val.json"
    confusion_matrix_file_name: str = "confusion_matrix_val.csv"
    
    threshold_strategy: str = "max_f1"
    manual_threshold: float | None = None

    def __post_init__(self) -> None:
        """dataclass 생성 직후 경로를 정규화하고 sample_rows 값을 검증"""

        object.__setattr__(
            self,
            "val_path",
            resolve_project_path(self.val_path, self.project_root),
        )
        object.__setattr__(
            self,
            "output_dir",
            resolve_project_path(self.output_dir, self.project_root),
        )
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer.")
        
        allowed_threshold_strategies = {"max_f1", "manual"}
        if self.threshold_strategy not in allowed_threshold_strategies:
            raise ValueError(
                "Unsupported threshold_strategy. "
                f"threshold_strategy={self.threshold_strategy!r}, "
                f"allowed={sorted(allowed_threshold_strategies)}"
            )
        if self.threshold_strategy == "manual":
            if self.manual_threshold is None:
                raise ValueError("manual_threshold is required when threshold_strategy='manual'.")
            manual_threshold = float(self.manual_threshold)
            if not 0 <= manual_threshold <= 1:
                raise ValueError(f"manual_threshold must be between 0 and 1. manual_threshold={manual_threshold}")
        elif self.manual_threshold is not None:
            raise ValueError("manual_threshold is only allowed when threshold_strategy='manual'.")


@dataclass(frozen=True)
class ValidationResult:
    """
    validation 실행 후 생성된 산출물 경로와 메모리상 결과를 함께 반환하는 객체

    사용 목적
    - 노트북에서 저장 파일 경로를 바로 확인
    - 후속 final test 또는 tuning runner에서 threshold/metric 값을 재사용
    """

    output_dir: Path
    threshold_path: Path
    metrics_path: Path
    confusion_matrix_path: Path
    threshold_info: dict[str, Any]
    val_metrics: dict[str, Any]


# -----------------------------------------------------------------------------
# 2. validation 산출물 overwrite 보호
# -----------------------------------------------------------------------------
def prepare_validation_outputs(config: ValidationConfig) -> None:
    """
    validation 산출물이 이미 있을 때 조용히 덮어쓰지 않도록 차단

    보호 대상
    - threshold.json
    - metrics_val.json
    - confusion_matrix_val.csv
    """

    output_paths = [
        config.output_dir / config.threshold_file_name,
        config.output_dir / config.metrics_file_name,
        config.output_dir / config.confusion_matrix_file_name,
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError(
            "Existing validation artifacts found. Set overwrite=True to replace them. "
            f"existing={existing}"
        )


# -----------------------------------------------------------------------------
# 3. validation threshold 선택 및 metric 저장
# -----------------------------------------------------------------------------
def validate_xgb(config: ValidationConfig) -> ValidationResult:
    """
    학습된 XGBoost 모델로 validation probability를 만들고 threshold/metric 저장

    동작 순서
    1. 학습 산출물(model, feature_columns, train_summary) 존재 여부 확인
    2. validation parquet 존재 여부 확인
    3. 저장된 feature column 순서와 hash를 로드
    4. validation split을 X, y로 로드
    5. predict_proba()로 positive-class probability 계산
    6. threshold_strategy에 따라 threshold 선택
       - max_f1이면 validation F1 최대 threshold 자동 선택
       - manual이면 manual_threshold 값을 고정 threshold로 사용
    7. threshold, metric, confusion matrix 저장

    주의
    - 이 함수는 final test를 수행하지 않음
    - 선택된 threshold 또는 수동 threshold는 threshold.json에 저장됨
    - test 단계는 threshold.json의 값을 그대로 사용해야 하며 test에서 재조정하면 안 됨
    - threshold_payload에는 threshold_strategy와 manual_threshold를 함께 기록해 사후 해석이 가능하게 함
    """
    # train_xgb()가 저장한 학습 산출물은 모두 같은 output_dir 에 저장된다고 가정
    # validation 단계는 새 모델을 학습하지 않고, 이미 저장된 model.pkl을 그대로 불러와 사용
    model_path = config.output_dir / config.model_file_name
    feature_columns_path = config.output_dir / config.feature_columns_file_name
    train_summary_path = config.output_dir / config.train_summary_file_name

    # 학습 단계에서 만든 model.pkl를 못찾으면 validation을 진행할 수 없으므로 
    # 에러 메시지출력하고 실패하도록 함
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if not feature_columns_path.exists():
        raise FileNotFoundError(f"feature columns file not found: {feature_columns_path}")
    if not train_summary_path.exists():
        raise FileNotFoundError(f"train summary file not found. Run ml_train.train_xgb() first: {train_summary_path}")
    if not config.val_path.exists():
        raise FileNotFoundError(f"validation parquet not found: {config.val_path}")

    # threshold.json, metrics_val.json, confusion_matrix_val.csv가 이미 있을 때
    # overwrite=False이면 기존 validation 결과를 덮어쓰지 않도록 차단
    prepare_validation_outputs(config)

    # 저장된 XGBoost 모델을 로드하여 validation split에서 예측 확률 계산
    model = joblib.load(model_path)    
    feature_columns = load_saved_feature_columns(feature_columns_path) # 학습 때 저장한 feature column 목록 로드
    train_summary = load_json(train_summary_path)                      # 학습 때 저장한 train_summary.json 로드
    features_hash = feature_columns_hash(feature_columns)              # feature column 목록에서 hash 계산 (validation 입력과 학습 시점 feature 연결 고리)

    # validation parquet를 X, y로 로드하는데, 이때 feature_columns 순서대로 로드하여 모델 입력과 일치하도록 함
    x_val, y_val = load_split(
        config.val_path,
        feature_columns=feature_columns,
        label_col=config.label_col,
        sample_rows=config.sample_rows,
        allow_nan=config.allow_nan,
        expected_split="val", # expected_split="val"은 parquet 내부 split 컬럼이 val인지 확인
    )

    probabilities = model.predict_proba(x_val)[:, 1]                                       # positive-class probability 추출
    if config.threshold_strategy == "max_f1":
        threshold_info = select_threshold_by_f1(y_val, probabilities)
        threshold = threshold_info["threshold"]
    elif config.threshold_strategy == "manual":
        threshold = float(config.manual_threshold)
        threshold_info = evaluate_at_threshold(y_val, probabilities, threshold)["summary"]
    else:
        raise ValueError(f"Unsupported threshold_strategy: {config.threshold_strategy!r}")
    val_metrics = evaluate_at_threshold(y_val, probabilities, threshold)

    # threshold.json에 저장할 내용
    # 목적은 "어떤 validation 기준으로 threshold를 골랐는지"를 재현 가능하게 남기는 것
    threshold_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_split": "val",
        "selection_metric": config.threshold_strategy,
        "threshold": float(threshold_info["threshold"]),
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,
        "val_path": str(config.val_path),
        "feature_count": len(feature_columns),
        "feature_columns_hash": features_hash,
        "model_file_name": config.model_file_name,
        "feature_columns_file_name": config.feature_columns_file_name,
        "train_summary_file_name": config.train_summary_file_name,
        "model_path": str(model_path),
        "feature_columns_path": str(feature_columns_path),
        "train_summary_path": str(train_summary_path),
        "train_sampled": bool(train_summary.get("sampled")),
        "train_sample_rows": train_summary.get("sample_rows"),
        "metrics_at_threshold": threshold_info,
        "threshold_strategy": config.threshold_strategy,
        "manual_threshold": config.manual_threshold,
    }
    # metrics_val.json에 저장할 내용
    # 목적은 validation split에서 고정 threshold로 계산한 성능과 label 분포를 남기는 것
    metrics_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "val",
        "val_path": str(config.val_path),
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,
        "feature_count": len(feature_columns),
        "feature_columns_hash": features_hash,
        "label_summary": label_summary(y_val),
        "metrics": val_metrics["summary"],
        "confusion_matrix": val_metrics["confusion_matrix"],
    }
    # validation 단계에서 생성할 세 가지 산출물 경로를 확정하고 저장
    threshold_path = config.output_dir / config.threshold_file_name
    metrics_path = config.output_dir / config.metrics_file_name
    confusion_matrix_path = config.output_dir / config.confusion_matrix_file_name

    save_json(threshold_payload, threshold_path)
    save_json(metrics_payload, metrics_path)
    confusion_matrix_frame(val_metrics).to_csv(confusion_matrix_path, index=False)
    
    # 노트북/후속 코드에서 저장 경로와 metric 내용을 바로 확인할 수 있도록 결과 객체를 반환
    return ValidationResult(
        output_dir=config.output_dir,
        threshold_path=threshold_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        threshold_info=threshold_payload,
        val_metrics=val_metrics,
    )
