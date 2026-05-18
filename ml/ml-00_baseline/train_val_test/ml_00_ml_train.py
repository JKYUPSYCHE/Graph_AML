"""
XGBoost 학습 모듈: train/validation parquet 파일로 이진분류 모델을 학습

1. 코드 전체 요약
이 코드는 train.parquet와 val.parquet를 읽어 XGBoost 이진분류 모델을 학습하고, 
학습 산출물을 output_dir에 저장하는 모듈이다.
주요 산출물은 model.pkl, feature_columns.json, train_summary.json, feature_importance.csv이다. Threshold 선택, 
validation 성능 리포트 생성, final test 평가는 이 코드의 책임이 아니다.

2. 데이터 흐름 요약
XGBTrainConfig 입력 → 경로 정규화 → 입력 파일 확인 → feature 목록 로드 → train/val parquet 로드 
→ train label로 scale_pos_weight 계산 → XGBoost 학습 → 모델/feature/importance 저장 → train_summary.json 저장 
→ XGBTrainResult 반환

3. 변경 시 주의점
- feature_columns 순서가 바뀌면 모델 입력 순서와 feature_columns_hash가 바뀐다.
- label_col 변경 시 load_feature_columns(), load_split(), label summary 전체에 영향이 있다.
- sample_rows는 train과 validation 모두에 적용되므로 실험 결과 해석 시 반드시 표시해야 한다.
- overwrite=False 기본값은 기존 실험 산출물 보호 목적이다.
- validation은 early stopping용이며 threshold 선택이나 최종 평가로 쓰면 안 된다.
- scale_pos_weight는 train label만 기준으로 계산된다.

이 파일의 핵심 역할
------------------
1. 사용자가 노트북에서 입력한 train/validation parquet 경로를 확정
2. ml_feature_columns.csv에서 실제 ML에 사용할 feature 목록을 read 및 확정
3. train split은 모델 학습에 사용
4. validation split은 학습 중 early stopping 평가에만 사용
5. 학습이 끝나면 아래 산출물을 output_dir에 저장
   - model.pkl: 학습된 XGBoost 모델
   - feature_columns.json: 모델 입력 feature 순서
   - train_summary.json: 학습 재현성 확인용 메타데이터

중요한 설계 의도
----------------
- 이 모듈은 threshold를 고르지 않는다. 확률값을 0/1 예측으로 바꾸는 threshold 선택은 ml_00_ml_val.py에서 따로 수행
- 이 모듈은 final test 평가를 하지 않는다.
  test split은 모델/feature/threshold 선택이 끝난 뒤 ml_00_ml_test.py에서 한 번만 사용해야 함 
- 즉, 이 파일의 책임은 "모델 학습과 학습 산출물 저장"까지
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
from ml_00_ml_utils import set_seed

# ml_00_ml_io는 이전 단계에서 만든 입출력/검증 유틸리티 모듈이다. 이 학습 모듈은 parquet를 직접 읽지 않고, load_split()에 위임한다.
# 변경 영향:
# - load_feature_columns()의 feature 선택 규칙이 바뀌면 모델 입력 컬럼 전체가 바뀐다.
# - load_split()의 결측치/타입/expected_split 검증 규칙이 바뀌면 학습 가능 데이터가 달라진다.
# 확인 필요: 각 함수의 세부 검증 기준은 ml_00_ml_io 구현을 확인해야 한다.
from ml_00_ml_io import (
    feature_columns_hash,
    file_sha256,
    label_summary,
    load_feature_columns,
    load_split,
    resolve_project_path,
    save_feature_columns,
    save_json,
)

# ml_00_ml_resource는 학습 시간, 메모리, 데이터 프로파일, 환경 정보, XGBoost 진단 정보, feature importance 저장을 담당한다.
# 이 모듈의 결과물 관리와 재현성 기록은 아래 함수들에 크게 의존한다.
# 확인 필요: MemoryTracker가 측정하는 메모리 기준이 RSS인지 peak인지 등은 구현 확인 필요.
from ml_00_ml_resource import (
    MemoryTracker,
    RuntimeTracker,
    collect_environment,
    make_data_profile,
    make_run_metadata,
    make_xgboost_diagnostics,
    save_feature_importance,
)

class MissingOptionalDependencyError(RuntimeError):
    """
    선택 의존성인 xgboost가 설치되어 있지 않을 때 발생시키는 예외.
    RuntimeError  이유
    - 코드 문법 문제가 아니라 실행 환경 문제이기 때문.
    - 예: pip install xgboost가 되어 있지 않은 상태에서 학습 함수를 호출하면 이 예외를 발생
    """

@dataclass(frozen=True)
class XGBTrainConfig:
    """
    XGBoost 학습에 필요한 사용자 입력값과 하이퍼파라미터 묶음
    frozen=True
    ----------------
    - Config 객체를 만든 뒤 실수로 값을 바꾸지 못하게 한다.
    - 재현성이 중요한 학습 설정에서는 설정값이 중간에 바뀌는 것을 막는 편이 안전
    경로 처리 방식
    --------------
    - train_path, val_path, feature_columns_path, output_dir는 str 또는 Path로 받을 수 있다.
    - 상대경로를 사용할 경우 project_root가 반드시 필요
    - __post_init__에서 resolve_project_path()를 호출해 절대경로로 변환
    데이터 흐름에서의 역할
    --------------------
    - 이 객체가 train_xgb()의 단일 입력 계약이다.
    - 학습 데이터 위치, feature 목록 위치, 저장 위치, seed, 하이퍼파라미터가 모두 여기서 결정된다.
    - 이후 train_summary.json에는 이 설정값 중 핵심 항목이 저장되어 실험 재현성 확인에 쓰인다.
    """
    train_path: Path | str                  # 입력 데이터 경로: train split parquet.
    val_path: Path | str                    # 입력 데이터 경로: validation split parquet.
    feature_columns_path: Path | str        # ml_feature_columns.csv 경로.
    output_dir: Path | str                  # 학습 산출물 저장 디렉터리.
    project_root: Path | str | None = None  # 상대경로 해석 기준이 되는 프로젝트 루트. None이면 입력 경로들은 절대경로여야 함
    label_col: str = "label"                # 정답 label 컬럼명. 바꾸면 load_split(), label_summary(), scale_pos_weight 계산에 모두 영향.
    sample_rows: int | None = None          # 디버깅/샘플 실험용 행 수 제한. None이면 전체 사용.
    allow_nan: bool = False                 # feature NaN 허용 여부. False면 load_split() 단계에서 차단될 가능성이 높다. 확인 필요.
    overwrite: bool = False                 # 기존 산출물 보호 옵션. False면 model.pkl 등 기존 파일이 있을 때 실패.
    seed: int = 42                          # 난수 고정값. 모델 재현성과 샘플링 동작에 영향.
    n_estimators: int = 300                 # 최대 tree 개수. early stopping이 걸리면 실제 사용 tree는 더 적을 수 있다.
    max_depth: int = 4                      # tree 깊이. 복잡도와 과적합에 직접 영향.
    learning_rate: float = 0.05             # boosting step 크기. 작을수록 보통 더 많은 tree가 필요.
    subsample: float = 0.9                  # tree별 row sampling 비율.
    colsample_bytree: float = 0.9           # tree별 feature sampling 비율.
    min_child_weight: float = 1.0           # leaf 분기 최소 가중치. 클수록 보수적인 tree가 된다.
    reg_lambda: float = 1.0                 # L2 정규화 강도.
    reg_alpha: float = 0.0                  # L1 정규화 강도.
    gamma: float = 0.0                      # 추가 split을 만들기 위한 최소 손실 감소량.
    early_stopping_rounds: int = 30         # validation AUPRC 개선이 멈췄을 때 학습을 중단하는 기준.
    n_jobs: int = -1                        # -1은 사용 가능한 모든 코어 사용.
    

    def __post_init__(self) -> None:
        """dataclass 생성 직후 경로를 절대경로로 정규화하고 기본 유효성 검사를 수행
        역할
        ----
        1. 사용자가 입력한 경로를 프로젝트 기준 절대경로로 변환한다.
        2. 학습 전에 잘못된 설정값을 미리 차단한다.
        변경 영향
        --------
        - resolve_project_path() 정책이 바뀌면 모든 입력/출력 경로 해석이 바뀐다.
        - frozen=True dataclass이므로 내부 값 변경은 object.__setattr__로만 수행된다.
        """
        # train 데이터 파일 경로를 절대경로로 정규화한다.
        # - "data/train.parquet" 같은 상대경로가 들어오면 project_root 기준으로 해석
        # - 이미 절대경로라면 그대로 정리해서 Path 형태로 사용
        # frozen=True 때문에 self.train_path = ... 방식은 불가능하므로 object.__setattr__을 사용한다.
        object.__setattr__(
            self,
            "train_path",
            resolve_project_path(self.train_path, self.project_root),
        )
        
        # validation 데이터 파일 경로를 같은 정책으로 정규화한다.
        object.__setattr__(
            self,
            "val_path",
            resolve_project_path(self.val_path, self.project_root),
        )
        
        # 학습에 사용할 feature 목록 CSV 경로를 절대경로로 정규화한다. 이 CSV에서 선택된 컬럼 순서가 XGBoost 입력 feature 순서가 된다.
        object.__setattr__(
            self,
            "feature_columns_path",
            resolve_project_path(self.feature_columns_path, self.project_root),
        )
        
        # 모델, feature 목록, 학습 summary, feature importance를 저장할 폴더 경로를 정규화한다.
        # prepare_output_dir()에서 실제 생성 및 overwrite 보호 검사를 수행한다.
        object.__setattr__(
            self,
            "output_dir",
            resolve_project_path(self.output_dir, self.project_root),
        )
        
        # sample_rows는 일부 데이터만 읽는 디버깅 옵션이다. 0 이하이면 즉시 차단한다.
        # sample_rows가 설정되면 train뿐 아니라 validation에도 동일하게 적용된다.
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer.")


@dataclass(frozen=True)
class XGBTrainResult:
    """
    train_xgb() 실행 결과를 담는 반환 객체
    목적
    ----
    - 노트북에서 함수 실행 후 어디에 어떤 파일이 저장됐는지 확인한다.
    - 학습 데이터 규모와 불균형 보정값을 즉시 확인한다.
    - 후속 validation 단계에서 model_path, feature_columns_path, train_summary_path를 넘기기 쉽도록 한다.
    주의
    ----
    - model 필드에는 학습된 모델 객체가 그대로 들어간다.
    - 파일로 저장된 모델은 model_path의 model.pkl이다.
    """
    output_dir: Path
    model_path: Path
    feature_columns_path: Path
    train_summary_path: Path
    feature_columns: list[str]
    feature_columns_hash: str
    train_rows: int
    val_rows: int
    scale_pos_weight: float
    training_time_sec: float
    model: Any
    

def require_training_files(config: XGBTrainConfig) -> None:
    """
    학습에 필요한 입력 파일이 실제로 존재하는지 확인한다.
    입력 파일
    --------
    - train parquet,  validation parquet,  feature columns CSV
    변경 영향
    --------
    - 여기서는 파일 존재 여부만 본다.
    - 파일 내부 schema, label, feature 존재 여부는 load_feature_columns(), load_split()에서 확인한다.
    """
    missing = []
    for path in [config.train_path, config.val_path, config.feature_columns_path]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing training input files: {missing}")
    

def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """학습 산출물 저장 폴더를 준비하고 기존 산출물 덮어쓰기 여부를 검사"""
   
    output_dir.mkdir(parents=True, exist_ok=True)      # output_dir가 없으면 생성한다.

    protected_outputs = [
        output_dir / "model.pkl",
        output_dir / "feature_columns.json",
        output_dir / "train_summary.json",
        output_dir / "feature_importance.csv",
    ]
    
    # 기존 산출물이 하나라도 있고 overwrite=False이면 실행을 중단한다.
    existing = [str(path) for path in protected_outputs if path.exists()] 
    if existing and not overwrite:
        raise FileExistsError(
            "Existing training artifacts found. Set overwrite=True to replace them. "
            f"existing={existing}"
        )
        

def compute_scale_pos_weight(y_train) -> float:
    """
    이진분류 label 불균형 보정을 위한 scale_pos_weight 값을 계산한다.
    데이터 흐름
    ----------
    y_train
    -> label별 개수 계산 -> negative_count / positive_count 계산 -> XGBClassifier(scale_pos_weight=...)에 전달
    AML 관점
    --------
    - AML/이상탐지에서는 label=1이 매우 적은 경우가 많다.
    - scale_pos_weight는 positive class 손실 가중치를 키워 불균형을 완화한다.
    주의
    ----
    - validation/test label 분포는 이 값 계산에 쓰지 않는다.
    - label 값이 0/1이라고 가정한다. 다른 label encoding을 쓰면 수정 필요.
    """
    label_counts = y_train.value_counts().to_dict() # value_counts()로 label별 개수를 계산합니다.
    positive_count = int(label_counts.get(1, 0))    # label=1을 positive class로 간주하고 개수를 가져옴 label=1이 없으면 0으로 처리
    negative_count = int(label_counts.get(0, 0))    # label=0은 negative class
    if positive_count == 0:  # positive가 하나도 없으면 분모가 0이 되고, 이진분류 학습 자체도 의미가 없다.
        raise ValueError("Train split has no positive labels; scale_pos_weight cannot be computed.")
    return negative_count / positive_count        # XGBoost에 전달할 scale_pos_weight 값  예: 정상 1140개, 이상 60개이면 1140 / 60 = 19.0


def get_xgb_classifier_class() -> type[Any]:
    """
    xgboost.XGBClassifier 클래스를 지연 import한다.
    xgboost 의존성 에러는 train_xgb()가 build_xgb_model()에 도달했을 때 발생한다.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "xgboost is required. Install the project environment before training."
        ) from exc
    return XGBClassifier # 호출부에서는 이 클래스를 받아 XGBClassifier(...) 형태로 실제 모델 객체 생성


def build_xgb_model(config: XGBTrainConfig, scale_pos_weight: float) -> Any:
    """
    XGBClassifier 모델 객체를 생성하고 학습 설정을 적용한다.
    매개변수
    --------
    - config: XGBTrainConfig 객체. 학습 경로, seed, 하이퍼파라미터가 담긴다.
    - scale_pos_weight: train label 분포에서 계산한 클래스 불균형 보정값.
    모델 작동 의도
    --------------
    - objective="binary:logistic": 0/1 이진분류 확률 출력
    - eval_metric="aucpr": 불균형 데이터에서 ROC AUC보다 positive class 탐지 성능 변화를 더 민감하게 보기 위한 PR AUC 사용
    - tree_method="hist": 대용량 tabular 데이터에서 빠른 histogram 기반 학습 사용
    - early_stopping_rounds: validation 성능이 개선되지 않으면 불필요한 tree 추가 중단
    코드 주의점
    -----------
    - config.xxx는 XGBTrainConfig 객체 안에 저장된 설정값을 꺼내 쓰는 문법이다.
    - early_stopping_rounds는 XGBoost 1.6+ 스타일에 맞춰 생성자에 전달한다.
    - 구버전 XGBoost에서는 fit(..., early_stopping_rounds=...) 방식이 필요할 수 있다. 확인 필요.
    """
    # xgboost.XGBClassifier 클래스를 호출한다. get_xgb_classifier_class() 내부에서 xgboost 설치 여부도 함께 확인한다.
    XGBClassifier = get_xgb_classifier_class()
    
    # 아래 객체는 하이퍼파라미터와 학습 정책이 설정된 모델 인스턴스다.
    return XGBClassifier(
        objective="binary:logistic",              # 이진분류용 objective. predict_proba로 label=1 확률을 얻는 후속 흐름에 사용 가능.
        eval_metric="aucpr",                      # validation eval_set에 대해 AUPRC 계열 지표를 계산한다.
        tree_method="hist",                       # 대용량 tabular 데이터에서 빠른 histogram 기반 학습 사용.
        n_estimators=config.n_estimators,         # 만들 tree 개수의 최대값. early stopping이 있으면 실제 best_iteration은 더 작을 수 있다.
        max_depth=config.max_depth,               # 각 tree의 최대 깊이. 과적합과 표현력에 영향.
        learning_rate=config.learning_rate,       # 각 boosting step 반영 비율.
        subsample=config.subsample,               # 각 tree를 학습할 때 사용할 row 비율.
        colsample_bytree=config.colsample_bytree, # 각 tree를 학습할 때 사용할 feature 비율.
        min_child_weight=config.min_child_weight, # leaf node가 추가로 분기되기 위해 필요한 최소 가중치 합.
        reg_lambda=config.reg_lambda,             # L2 정규화 강도.
        reg_alpha=config.reg_alpha,               # L1 정규화 강도.
        gamma=config.gamma,                       # leaf node 추가 분기에 필요한 최소 손실 감소량.
        scale_pos_weight=scale_pos_weight,        # train label 분포 기반 클래스 불균형 보정값.
        random_state=config.seed,                 # XGBoost 내부 난수 고정.
        n_jobs=config.n_jobs,                     # 병렬 학습에 사용할 CPU worker 수.
        early_stopping_rounds=config.early_stopping_rounds,
    )



def train_xgb(config: XGBTrainConfig) -> XGBTrainResult:
    """
    XGBoost 모델을 학습하고 학습 산출물을 저장한다.
    전체 실행 흐름
    --------------
    1. 난수 시드 고정
    2. 입력 파일 존재 확인
    3. output_dir 준비 및 기존 산출물 보호
    4. ml_feature_columns.csv에서 사용할 feature 목록 로드
    5. train/validation parquet를 X, y로 로드
    6. label 불균형 보정값 계산
    7. XGBoost 모델 생성 및 학습
    8. model.pkl, feature_columns.json, feature_importance.csv 저장
    9. train_summary.json 저장
    10. XGBTrainResult 반환
    
    이 함수가 하지 않는 것
    ---------------------
    - classification threshold 선택을 하지 않는다.
    - final test 평가를 하지 않는다.
    - feature engineering을 하지 않는다.
    - AML rule 기반 탐지를 적용하지 않는다. 이 코드는 XGBoost ML 모델 학습만 수행한다.
    
    결과물 관리
    -----------
    - output_dir 아래 고정 파일명으로 산출물을 저장한다.
    - overwrite=False이면 기존 결과를 보호한다.
    - train_summary.json은 후속 validation/test 단계에서 provenance 확인 기준이 될 수 있다.
    """
    # 전체 train_xgb 실행 시간을 측정하기 위한 시작 시각이다.
    # runtime_tracker는 세부 단계별 시간을 기록하고, total_started는 전체 시간을 따로 계산한다.
    total_started = time.perf_counter()
    
    # 단계별 실행 시간을 측정한다. train_summary["runtime_sec"]에 저장되어 병목 구간 추적에 사용된다.
    runtime_tracker = RuntimeTracker()
    
    # 학습 전체 구간의 메모리 사용량을 추적한다. snapshot 이름은 train_summary의 memory_profile 분석에 쓰인다.
    memory_tracker = MemoryTracker(scope="train_full")
    memory_tracker.start()
    try:
        # 입력 준비 단계:
        # - seed 고정
        # - 입력 파일 존재 확인
        # - output_dir 생성 및 기존 산출물 보호
        with runtime_tracker.measure("prepare_inputs"):
            set_seed(config.seed)
            require_training_files(config)
            prepare_output_dir(config.output_dir, overwrite=config.overwrite)
            
        # ml_feature_columns.csv에서 실제 모델 입력에 사용할 feature 목록을 읽는다.
        # load_feature_columns() 내부에서 label/target/leakage 의심 컬럼 차단 가능성이 있다.
        # 확인 필요: 차단 규칙과 used_in_ml 해석 방식은 ml_00_ml_io 구현 확인 필요.
        with runtime_tracker.measure("load_feature_columns"):
            feature_columns = load_feature_columns(
                config.feature_columns_path,
                label_col=config.label_col,
            )
            
        # train split을 X, y로 로드한다.
        # expected_split="train"은 parquet 내부 split 컬럼이 있을 경우 train인지 확인하는 용도다.
        # 확인 필요: split 컬럼이 없을 때 load_split()이 허용하는지 여부.
        with runtime_tracker.measure("load_train_split"):
            x_train, y_train = load_split(
                config.train_path,
                feature_columns=feature_columns,
                label_col=config.label_col,
                sample_rows=config.sample_rows,
                allow_nan=config.allow_nan,
                expected_split="train",
            )
            
        # train 로드 직후 메모리 스냅샷을 남긴다.
        # 대용량 parquet에서 로드 비용과 메모리 증가량을 추적하는 지점이다.
        memory_tracker.snapshot("after_train_load")
        
        # validation split을 X, y로 로드한다.
        # validation은 모델 학습 중 eval_set으로만 사용된다.
        # threshold 선택 또는 최종 test 평가는 이 함수에서 수행하지 않는다.
        with runtime_tracker.measure("load_val_split"):
            x_val, y_val = load_split(
                config.val_path,
                feature_columns=feature_columns,
                label_col=config.label_col,
                sample_rows=config.sample_rows,
                allow_nan=config.allow_nan,
                expected_split="val",
            )
    
        memory_tracker.snapshot("after_val_load")    # validation 로드 직후 메모리 스냅샷을 남긴다.
        
        with runtime_tracker.measure("build_model"):              # 모델 생성 전 필요한 파생값을 만든다.
            features_hash = feature_columns_hash(feature_columns) # feature 순서까지 반영한 hash다.
            scale_pos_weight = compute_scale_pos_weight(y_train)
            model = build_xgb_model(config, scale_pos_weight=scale_pos_weight)
            
        # 실제 모델 학습 지점이다.
        # eval_set으로 validation 데이터를 넘겨 early stopping과 validation AUPRC 계산에 사용한다.
        # verbose=False이므로 학습 로그는 콘솔에 출력하지 않는다.
        with runtime_tracker.measure("fit"):
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=False,
            )
            
        # fit 단계 소요 시간만 별도 필드로 저장한다. 전체 실행 시간은 아래 runtime_sec["total_train_xgb"]에 따로 저장된다.
        training_time_sec = runtime_tracker.as_dict().get("fit", 0.0)
        
        memory_tracker.snapshot("after_fit")   # 학습 완료 직후 메모리 스냅샷이다.
        
    
        # 학습 결과와 환경 정보를 train_summary에 넣기 위해 메타데이터를 구성한다.
        with runtime_tracker.measure("build_metadata"):
            # early stopping 기준으로 선택된 best iteration.
            # XGBoost 버전/설정에 따라 속성이 없을 수 있어 getattr로 읽고 없으면 None으로 둔다.
            best_iteration = getattr(model, "best_iteration", None)
            if best_iteration is not None:
                best_iteration = int(best_iteration)
                
            # validation 기준 best score.
            # 현재 build_xgb_model()에서 eval_metric="aucpr"를 사용하므로 PR AUC 계열 점수로 해석된다.
            best_score = getattr(model, "best_score", None)
            if best_score is not None:
                best_score = float(best_score)
                
            # best_iteration이 n_estimators 끝까지 갔는지 확인한다.
            n_estimators_ceiling_hit = None if best_iteration is None else best_iteration >= config.n_estimators - 1
            
            # 데이터 프로파일을 만든다.
            # 경로, row 수, feature 수, label 분포, dtype/결측 정보 등이 포함될 가능성이 있다.
            # 확인 필요: make_data_profile()의 정확한 포함 항목은 구현 확인 필요.
            data_profile = make_data_profile(
                {
                    "train": (config.train_path, x_train, y_train),
                    "val": (config.val_path, x_val, y_val),
                },
                feature_columns,
            )
            
            # XGBoost 모델 내부 진단 정보를 수집한다.
            xgboost_diagnostics = make_xgboost_diagnostics(model)
            
            # 실행 환경 정보를 수집한다. 
            environment = collect_environment()
            
        # 저장할 산출물 경로를 고정 이름으로 생성한다.
        # 후속 validation/test 모듈은 train_summary의 파일명 필드를 통해 이 산출물을 찾을 수 있다.
        model_path = config.output_dir / "model.pkl"
        feature_columns_out = config.output_dir / "feature_columns.json"
        train_summary_path = config.output_dir / "train_summary.json"
        feature_importance_path = config.output_dir / "feature_importance.csv"
        
        # 학습된 모델과 해석용 중요도를 저장한다.
        # 보안 주의: pickle/joblib 파일은 신뢰할 수 있는 파일만 로드해야 한다.
        with runtime_tracker.measure("save_artifacts"):
            # 학습된 XGBoost 모델 객체를 저장한다.
            # 후속 validation/test에서 이 파일을 로드해 predict_proba를 수행할 수 있다.
            joblib.dump(model, model_path)
            
            # 모델 입력 feature 순서를 JSON으로 저장한다.
            # 모델 재사용 시 반드시 같은 순서로 feature matrix를 구성해야 한다.
            save_feature_columns(feature_columns, feature_columns_out)
            
            # feature importance를 CSV로 저장한다.
            # 모델 해석, feature 제거 검토, leakage 의심 feature 점검에 활용할 수 있다.
            feature_importance_frame = save_feature_importance(model, feature_columns, feature_importance_path)
            
            # 저장된 산출물의 sha256 hash를 계산한다.
            # train_summary에 기록해 후속 단계에서 파일 변경 여부를 확인할 수 있다.
            model_sha256 = file_sha256(model_path)
            feature_columns_file_sha256 = file_sha256(feature_columns_out)
            feature_importance_file_sha256 = file_sha256(feature_importance_path)
            
        # 전체 학습 종료 직전 메모리 스냅샷이다.
        memory_tracker.snapshot("end")
        
    finally:
        # 예외가 발생하더라도 memory_tracker를 종료해 가능한 범위의 메모리 프로파일을 확보한다.
        # 단, 예외 발생 시 아래 train_summary 저장까지 도달하지 못할 수 있다.
        memory_profile = memory_tracker.finish()
        
    # 단계별 runtime dict를 만든 뒤 전체 실행 시간을 추가한다.
    runtime_sec = runtime_tracker.as_dict()
    runtime_sec["total_train_xgb"] = float(time.perf_counter() - total_started)
    
    # 실행마다 고유한 run_id를 생성한다.
    # seed가 같아도 uuid는 매번 달라지므로 run_id 자체는 재현 가능한 값이 아니다.
    run_id = f"xgb_train_{uuid4().hex}"
    
    # train_summary 생성 시각을 UTC ISO format으로 저장한다.
    created_at = datetime.now(timezone.utc).isoformat()
    
    # 학습 재현성과 사후 검토를 위한 메타데이터다.
    # 후속 validation/test 모듈에서 provenance check에 활용될 수 있다.
    train_summary = {
        "created_at": created_at,                                    # UTC 기준 생성 시각.
        "run_id": run_id,                                            # 실행 단위 식별자. 매 실행마다 달라진다.
        "seed": config.seed,                                         # 난수 고정값.
        "run_metadata": make_run_metadata(config.output_dir, seed=config.seed),
        "label_col": config.label_col,                               # 정답 label 컬럼명.
        "train_path": str(config.train_path),                        # 학습에 사용한 train parquet 경로.
        "val_path": str(config.val_path),                            # 학습에 사용한 validation parquet 경로.
        "feature_columns_source": str(config.feature_columns_path),  # feature 목록 CSV 원본 경로.
        
        # 후속 모듈이 참조할 artifact 파일명이다.
        # output_dir와 조합해 실제 파일 경로를 만들 수 있다.
        "model_file_name": "model.pkl",
        "feature_columns_file_name": "feature_columns.json",
        "train_summary_file_name": "train_summary.json",
        "feature_importance_file_name": "feature_importance.csv",
        
        # 저장된 산출물의 파일 hash다.
        # 후속 validation/test에서 파일이 바뀌지 않았는지 확인하는 기준으로 쓸 수 있다.
        "model_sha256": model_sha256,
        "feature_columns_file_sha256": feature_columns_file_sha256,
        "feature_importance_file_sha256": feature_importance_file_sha256,
        
        # 사용한 feature 개수, 목록, 순서 hash다.
        # feature 목록과 순서는 모델 입력 재현성의 핵심이다.
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_columns_hash": features_hash,
        
        # 실제 로드된 train/validation row 수와 label 분포다.
        # sample_rows가 설정된 경우 전체 parquet row 수가 아니라 로드된 row 수로 기록된다.
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "train_positive_ratio": float(y_train.mean()),
        "train_label_summary": label_summary(y_train),
        "val_label_summary": label_summary(y_val),
        
        # train label 분포에서 계산한 클래스 불균형 보정값이다.
        "scale_pos_weight": float(scale_pos_weight), 
        
        # early stopping 관련 정보, best_score는 현재 eval_metric="aucpr" 기준으로 해석한다.
        "best_iteration": best_iteration,
        "best_score": best_score,
        "n_estimators_ceiling_hit": n_estimators_ceiling_hit,
        
        # sample 실행 여부와 설정값  실험 결과 비교 시 sample 실험과 full 실험을 혼동하지 않기 위한 필드다.
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,
        
        # feature NaN 허용 여부, XGBoost는 NaN을 처리할 수 있지만, 이 프로젝트에서 허용할지 여부는 config로 통제한다.
        "allow_nan": config.allow_nan,
        
        # 실제 XGBoost에 전달한 주요 하이퍼파라미터,  재학습이나 실험 비교 시 가장 먼저 확인해야 하는 설정 묶음이다.
        "xgboost_params": {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "min_child_weight": config.min_child_weight,
            "reg_lambda": config.reg_lambda,
            "reg_alpha": config.reg_alpha,
            "gamma": config.gamma,
            "early_stopping_rounds": config.early_stopping_rounds,
            "tree_method": "hist",
            "eval_metric": "aucpr",
        },
        
        # fit 단계 소요 시간이다. 기존 필드명을 유지한다.
        "training_time_sec": float(training_time_sec),
        
        # 메모리와 실행 시간 프로파일이다.  대용량 AML 데이터에서 병목 추적과 실험 비용 비교에 중요하다.
        "memory_mb": memory_profile["memory_mb"],
        "memory_mb_semantics": memory_profile["memory_mb_semantics"],
        "runtime_sec": runtime_sec,
        "memory_profile": memory_profile,
        
        # 데이터, 실행 환경, XGBoost 내부 상태 진단 정보다. 정확한 세부 필드는 ml_00_ml_resource 구현 확인 필요.
        "data_profile": data_profile,
        "environment": environment,
        "xgboost_diagnostics": xgboost_diagnostics,
        
        # feature_importance.csv에 저장된 row 수다.  일반적으로 feature 수와 일치해야 할 가능성이 높지만, 구현에 따라 다를 수 있다. 확인 필요.
        "feature_importance_rows": int(len(feature_importance_frame)),
    }
    # train_summary.json 저장 지점이다. 이 파일은 실험 재현성, 결과 비교, validation/test 검증의 기준 자료다.
    save_json(train_summary, train_summary_path)
    
    # 노트북 사용자가 저장 경로와 핵심 학습 정보를 바로 확인할 수 있도록 결과 객체를 반환한다.
    # 반환 객체에는 저장된 파일 경로와 학습된 model 객체가 함께 들어 있다.
    return XGBTrainResult(
        output_dir=config.output_dir,
        model_path=model_path,
        feature_columns_path=feature_columns_out,
        train_summary_path=train_summary_path,
        feature_columns=feature_columns,
        feature_columns_hash=features_hash,
        train_rows=len(y_train),
        val_rows=len(y_val),
        scale_pos_weight=float(scale_pos_weight),
        training_time_sec=float(training_time_sec),
        model=model,
    )
