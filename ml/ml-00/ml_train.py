"""XGBoost 학습 모듈: train/validation parquet 파일로 이진분류 모델을 학습

이 파일의 핵심 역할
------------------
1. 사용자가 노트북에서 입력한 train/validation parquet 경로를 확정
2. feature catalog CSV에서 실제 ML에 사용할 feature 목록을 read 및 확정
3. train split은 모델 학습에 사용
4. validation split은 학습 중 early stopping 평가에만 사용
5. 학습이 끝나면 아래 산출물을 output_dir에 저장
   - model.pkl: 학습된 XGBoost 모델
   - feature_columns.json: 모델 입력 feature 순서
   - train_summary.json: 학습 재현성 확인용 메타데이터

중요한 설계 의도
----------------
- 이 모듈은 threshold를 고르지 않는다. 확률값을 0/1 예측으로 바꾸는 threshold 선택은 ml_val.py에서 따로 수행
- 이 모듈은 final test 평가를 하지 않는다.
  test split은 모델/feature/threshold 선택이 끝난 뒤 ml_test.py에서 한 번만 사용해야 함 
- 즉, 이 파일의 책임은 "모델 학습과 학습 산출물 저장"까지
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from ml_utils import set_seed

# ml_io는 이전 단계에서 만든 입출력/검증 유틸리티 모듈
# 이 학습 모듈은 parquet를 직접 읽지 않고, ml_io.load_split()에 위임
from ml_io import (
    feature_columns_hash,
    label_summary,
    load_feature_columns,
    load_split,
    resolve_project_path,
    save_feature_columns,
    save_json,
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
    """


    train_path: Path | str                  # 입력 데이터 경로: train split parquet. 
    val_path: Path | str                    # 입력 데이터 경로: validation split parquet.
    feature_columns_path: Path | str        # feature catalog CSV 경로.
    output_dir: Path | str                  # 학습 산출물 저장 디렉터리.
    project_root: Path | str | None = None  # 상대경로 해석 기준이 되는 프로젝트 루트,None이면 입력 경로들은 절대경로여야함     

    label_col: str = "label"                 # 정답 label 컬럼명
    sample_rows: int | None = None           # 학습 데이터 샘플링 행 수, None이면 전체 사용
    allow_nan: bool = False                  # feature에 NaN 허용 여부, False면 NaN이 있으면 에러, True면 NaN 허용 (XGBoost가 자체적으로 처리)
    overwrite: bool = False                  # output_dir에 기존 학습 산출물이 있으면 덮어쓸지 여부, False면 에러, True면 덮어쓰기

    seed: int = 42
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    early_stopping_rounds: int = 30
    n_jobs: int = -1                          # -1은 사용 가능한 모든 코어

    def __post_init__(self) -> None:
        """dataclass 생성 직후 경로를 절대경로로 정규화하고 기본 유효성 검사를 수행
            역할
            1. 사용자가 입력한 경로를 프로젝트 기준 절대경로로 변환한다.
            2. 학습 전에 잘못된 설정값을 미리 차단한다.        
        """
        # train 데이터 파일 경로를 절대경로로 정규화한다.
        # 예:
        # - "data/train.parquet" 같은 상대경로가 들어오면 project_root 기준으로 해석
        # - 이미 절대경로라면 그대로 정리해서 Path 형태로 사용
        #
        # frozen=True 때문에 self.train_path = ... 방식은 불가능하므로
        # object.__setattr__을 사용
        object.__setattr__(
            self,
            "train_path",
            resolve_project_path(self.train_path, self.project_root),
        )
        object.__setattr__(
            self,
            "val_path",
            resolve_project_path(self.val_path, self.project_root),
        )
        object.__setattr__( # 학습에 사용할 feature 목록 CSV 경로를 절대경로로 정규화
            self,
            "feature_columns_path",
            resolve_project_path(self.feature_columns_path, self.project_root),
        )
        object.__setattr__( # 모델, 로그, 설정 파일 등 학습 산출물을 저장할 폴더 경로를 절대경로로 정규화
            self,
            "output_dir",
            resolve_project_path(self.output_dir, self.project_root),
        )
        # sample_rows는 일부 데이터만 읽는 디버깅 옵션
        # 0 이하이면 의미가 없으므로 즉시 차단
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer.")


@dataclass(frozen=True)
class XGBTrainResult:
    """
    train_xgb() 실행 결과를 담는 반환 객체
    노트북에서 함수 실행 후 어디에 어떤 파일이 저장됐는지,
    학습 데이터 규모와 불균형 보정값이 얼마였는지 등을 확인할 수 있도록 설계
    학습 함수가 끝난 뒤 결과를 정리해서 반환하기 위한 자료형 으로, 모델 객체 자체도 포함할 수 있다.
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
    학습에 필요한 입력 파일이 실제로 존재하는지 확인
    """
    missing = []
    for path in [config.train_path, config.val_path, config.feature_columns_path]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing training input files: {missing}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """학습 산출물 저장 폴더를 준비하고 기존 산출물 덮어쓰기 여부를 검사"""
    output_dir.mkdir(parents=True, exist_ok=True) # output_dir가 없으면 생성한다.
    protected_outputs = [                         # 이 파일들이 이미 있으면 이전 실험 결과를 덮어쓸 위험이 있다.
        output_dir / "model.pkl",
        output_dir / "feature_columns.json",
        output_dir / "train_summary.json",
    ]
    existing = [str(path) for path in protected_outputs if path.exists()]
    if existing and not overwrite:   # overwrite=False인 기본 상태에서는 기존 결과를 보호한다.
        raise FileExistsError(
            "Existing training artifacts found. Set overwrite=True to replace them. "
            f"existing={existing}"
        )

def compute_scale_pos_weight(y_train) -> float:
    """
    이진분류 label 불균형 보정을 위한 scale_pos_weight 값을 계산
    XGBoost의 scale_pos_weight는 일반적으로 negative_count / positive_count로 둔다.
    positive class가 희소한 AML/이상탐지 문제에서 positive class의 손실 가중치를 키우는 효과가 있다.
    """
    label_counts = y_train.value_counts().to_dict() # value_counts()로 label별 개수를 계산합니다.
    positive_count = int(label_counts.get(1, 0))   # label=1을 positive class로 간주하고 개수를 가져옴 label=1이 없으면 0으로 처리
    negative_count = int(label_counts.get(0, 0))   # label=0은 negative class
    if positive_count == 0:  # positive가 하나도 없으면 분모가 0이 되고, 이진분류 학습 자체도 의미가 없다.
        raise ValueError("Train split has no positive labels; scale_pos_weight cannot be computed.")
    return negative_count / positive_count        # XGBoost에 전달할 scale_pos_weight 값  예: 정상 1140개, 이상 60개이면 1140 / 60 = 19.0

def get_xgb_classifier_class() -> type[Any]:
    """
    xgboost.XGBClassifier 클래스를 지연 import한다.
    지연 import 쓰는 이유
    - 이 모듈을 읽거나 설정 객체를 만드는 단계에서는 xgboost가 꼭 필요하지 않다.
    - 실제 학습을 수행할 때만 xgboost 설치 여부를 확인하면 에러 메시지를 더 명확히 줄 수 있다.
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
    config.xxx = XGBTrainConfig에 저장된 학습 설정값을 꺼내서 XGBClassifier에 전달
    
    매개변수
    - config: XGBTrainConfig 객체. 학습에 필요한 하이퍼파라미터와 실행 설정이 담긴 객체
    - scale_pos_weight: 클래스 불균형 보정값
    모델 작동 의도
    - objective="binary:logistic": 0/1 이진분류 확률 출력
    - eval_metric="aucpr": 불균형 데이터에서 ROC AUC보다 더 민감할 수 있는 PR AUC 사용
    - tree_method="hist": 대용량 tabular 데이터에서 빠른 histogram 기반 학습 사용
    - early_stopping_rounds: validation 성능이 개선되지 않으면 불필요한 tree 추가 멈춤
    코드 주의점
    - config.xxx는 XGBTrainConfig 객체 안에 저장된 설정값을 꺼내 쓰는 문법
    - early_stopping_rounds는 XGBoost 1.6+ 스타일에 맞춰 생성자에 전달
    - 구버전 XGBoost에서는 fit(..., early_stopping_rounds=...) 방식이 필요할 수 있음
    """
    # xgboost.XGBClassifier 클래스호출, get_xgb_classifier_class() 내부에서 xgboost 설치 여부도 함께 확인
    XGBClassifier = get_xgb_classifier_class()

    return XGBClassifier(                         # config.xxx 값들은 XGBTrainConfig에 저장된 학습 설정값
        objective="binary:logistic",              # binary:logistic은 이진분류용 objective
        eval_metric="aucpr",                      # AUPRC 기준으로 validation 성능을 평가
        tree_method="hist",                       # 대용량 tabular 데이터에서 빠른 histogram 기반 학습 사용
        n_estimators=config.n_estimators,         # 만들 tree 개수의 최대값
        max_depth=config.max_depth,               # 각 tree의 최대 깊이
        learning_rate=config.learning_rate,        
        subsample=config.subsample,               # 각 tree를 학습할 때 사용할 row 비율
        colsample_bytree=config.colsample_bytree, # 각 tree를 학습할 때 사용할 feature 비율
        min_child_weight=config.min_child_weight, # leaf node가 추가로 분기되기 위해 필요한 최소 가중치 합
        reg_lambda=config.reg_lambda,             # L2 정규화 강도
        reg_alpha=config.reg_alpha,               # L1 정규화 강도
        gamma=config.gamma,                       # leaf node 추가 분기 필요 최소 손실 감소량 
        scale_pos_weight=scale_pos_weight,        # 클래스 불균형 보정값
        random_state=config.seed,
        n_jobs=config.n_jobs,
        early_stopping_rounds=config.early_stopping_rounds,
    )


def train_xgb(config: XGBTrainConfig) -> XGBTrainResult:
    """
    XGBoost 모델을 학습하고 학습 산출물을 저장한다.
    전체 실행 흐름
    1. 난수 시드 고정
    2. 입력 파일 존재 확인
    3. output_dir 준비 및 기존 산출물 보호
    4. feature catalog에서 사용할 feature 목록 로드
    5. train/validation parquet를 X, y로 로드
    6. label 불균형 보정값 계산
    7. XGBoost 모델 생성 및 학습
    8. model.pkl, feature_columns.json, train_summary.json 저장
    9. XGBTrainResult 반환

    이 함수가 하지 않는 것
    - classification threshold 선택을 하지 않는다.
    - final test 평가를 하지 않는다.
    - feature engineering을 하지 않는다.
    """

    set_seed(config.seed)
    require_training_files(config)
    prepare_output_dir(config.output_dir, overwrite=config.overwrite)
    
    # feature catalog CSV에서 실제 모델 입력에 사용할 feature 목록을 read
    # load_feature_columns() 내부에서 label/target/leakage 의심 컬럼 차단 
    feature_columns = load_feature_columns(
        config.feature_columns_path,
        label_col=config.label_col,
    )

    # train split을 X, y로 로드
    x_train, y_train = load_split(
        config.train_path,
        feature_columns=feature_columns,
        label_col=config.label_col,
        sample_rows=config.sample_rows,
        allow_nan=config.allow_nan,
        expected_split="train",     # parquet 안에 split 테그 컬럼 있으면 값이 train인지 확인
    )
    
    # # validation split을 X, y로 로드. 이 데이터는 모델 학습 중 early stopping 평가에만 사용
    # 이 validation은 early stopping용이며, threshold 선택은 별도 모듈에서 수행
    x_val, y_val = load_split(
        config.val_path,
        feature_columns=feature_columns,
        label_col=config.label_col,
        sample_rows=config.sample_rows,
        allow_nan=config.allow_nan,
        expected_split="val",
    )

    features_hash = feature_columns_hash(feature_columns) # feature 순서까지 반영한 hash. 모델 재현성 검증에 활용
    scale_pos_weight = compute_scale_pos_weight(y_train) # AML/이상탐지처럼 positive class가 적은 문제에서 class imbalance 보정을 위해 사용
    model = build_xgb_model(config, scale_pos_weight=scale_pos_weight) # 설정값과 불균형 보정값을 반영해 XGBClassifier 객체를 만든다.

    started = time.perf_counter() # perf_counter는 경과 시간 측정에 적합한 고해상도 타이머
    
    # 실제 모델 학습.
    # eval_set으로 validation 데이터를 넘겨 early stopping과 validation metric 계산에 사용
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    training_time_sec = time.perf_counter() - started   # 전체 학습 소요 시간
    
    # early stopping 기준으로 선택된 best iteration
    # XGBoost 버전/설정에 따라 속성이 없을 수 있어 getattr로 read, 없으면 None 반환
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        best_iteration = int(best_iteration)
    best_score = getattr(model, "best_score", None)
    
    # validation 기준 best score
    # 현재 build_xgb_model()에서 eval_metric="aucpr"를 사용하므로 PR AUC 계열 점수가 best_score로 선택
    if best_score is not None:
        best_score = float(best_score)
        # best_iteration이 n_estimators 끝까지 갔는지 확인,  True이면 early stopping 전에 tree 개수 상한에 도달했다는 뜻
        # 이 경우 n_estimators를 늘릴 필요가 있는지 검토해야 함 
    n_estimators_ceiling_hit = None if best_iteration is None else best_iteration >= config.n_estimators - 1

    # 저장할 산출물 경로를 고정 이름으로 생성
    model_path = config.output_dir / "model.pkl"
    feature_columns_out = config.output_dir / "feature_columns.json"
    train_summary_path = config.output_dir / "train_summary.json"

    # 학습된 모델 저장, 주의: pickle/joblib 파일은 신뢰할 수 있는 파일만 로드해야 한다.
    joblib.dump(model, model_path)
    # 모델이 기대하는 feature 순서를 별도 JSON으로 저장
    save_feature_columns(feature_columns, feature_columns_out)     

    # 학습 재현성과 사후 검토를 위한 메타데이터.
    # 후속 validation/test 모듈에서 provenance check에 활용될 수 있다.
    train_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),       # UTC 기준 생성 시각
        "label_col": config.label_col,                              # 정답 label 컬럼명
        "train_path": str(config.train_path),                       # 학습에 사용한 입력 파일 경로
        "val_path": str(config.val_path),                           # 학습에 사용한 입력 파일 경로
        "feature_columns_source": str(config.feature_columns_path), # feature catalog 원본 경로
        # 후속 모듈이 참조할 artifact 파일명
        "model_file_name": "model.pkl",                             
        "feature_columns_file_name": "feature_columns.json",
        "train_summary_file_name": "train_summary.json",
        # 사용한 feature 개수, 목록, 순서 hash
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_columns_hash": features_hash,
        # 실제 로드된 train/validation row 수, train label 분포, 불균형 보정값
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "train_positive_ratio": float(y_train.mean()),
        "train_label_summary": label_summary(y_train),
        "val_label_summary": label_summary(y_val),
        "scale_pos_weight": float(scale_pos_weight),
        # early stopping 관련 정보
        "best_iteration": best_iteration,
        "best_score": best_score,
        "n_estimators_ceiling_hit": n_estimators_ceiling_hit,
        # sample 실행 여부와 설정값, 학습 시간
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,
        # feature NaN 허용 여부
        "allow_nan": config.allow_nan,
        # 실제 XGBoost에 전달한 주요 하이퍼파라미터
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
        "training_time_sec": float(training_time_sec), # 학습 소요 시간
    }
    save_json(train_summary, train_summary_path)       # 이 파일은 실험 재현성, 결과 비교, validation/test 검증의 기준 자료

    # 노트북 사용자가 저장 경로와 핵심 학습 정보를 바로 확인할 수 있도록 결과 객체를 반환
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
