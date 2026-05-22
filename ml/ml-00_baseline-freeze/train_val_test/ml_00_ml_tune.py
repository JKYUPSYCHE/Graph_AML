"""
ML-00 XGBoost random search tuning runner 모듈

전체 흐름
1. ml_00_ml_search_spaces.py의 discrete search space preset을 읽음
2. random_search_seed 기준으로 중복 없는 hyperparameter 조합을 샘플링
3. trial별 하위 디렉터리를 만들고 trial_params.json을 저장
4. 각 trial에서 ml_00_ml_train.train_xgb()와 ml_00_ml_val.validate_xgb()를 순차 실행
5. trial별 validation metric을 모아 trials_summary.csv와 tuning_summary.json을 저장

중요한 전제
- 이 모듈은 train + validation 반복 실행까지만 담당
- final test는 절대 실행하지 않음
- sample_rows는 smoke/debug 전용이며 성능 판단용으로 사용하면 안 됨
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from ml_00_ml_io import load_json, resolve_project_path, save_json
from ml_00_ml_search_spaces import Number, get_search_space
from ml_00_ml_train import XGBTrainConfig, train_xgb
from ml_00_ml_val import ValidationConfig, validate_xgb


ALLOWED_SELECTION_METRICS = {"average_precision", "f1", "recall", "precision"}


# -----------------------------------------------------------------------------
# 1. Tuning 실행 설정과 반환 결과 자료구조
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class XGBTuneConfig:
    """
    random search tuning 실행에 필요한 입력 경로와 옵션 묶음

    주요 옵션
    - search_space_name: ml_00_ml_search_spaces.py에 정의된 preset 이름
    - trial_count: 샘플링할 unique hyperparameter 조합 수
    - random_search_seed: trial 조합 샘플링 재현성 제어
    - selection_metric: best trial 선택 기준 metric
    - overwrite: 기존 tuning 산출물 보호 여부

    주의
    - final test 관련 옵션은 의도적으로 제공하지 않음
    """

    train_path: Union[Path, str]
    val_path: Union[Path, str]
    feature_columns_path: Union[Path, str]
    output_dir: Union[Path, str]
    project_root: Optional[Union[Path, str]] = None

    label_col: str = "label"
    search_space_name: str = "model_select_minimal"
    trial_count: int = 8
    random_search_seed: int = 42
    seed: int = 42
    selection_metric: str = "average_precision"
    sample_rows: Optional[int] = None
    allow_nan: bool = False
    overwrite: bool = False
    n_jobs: int = -1

    def __post_init__(self) -> None:
        """dataclass 생성 직후 경로와 tuning 설정값을 검증"""

        object.__setattr__(self, "train_path", resolve_project_path(self.train_path, self.project_root))
        object.__setattr__(self, "val_path", resolve_project_path(self.val_path, self.project_root))
        object.__setattr__(
            self,
            "feature_columns_path",
            resolve_project_path(self.feature_columns_path, self.project_root),
        )
        object.__setattr__(self, "output_dir", resolve_project_path(self.output_dir, self.project_root))

        if self.trial_count <= 0:
            raise ValueError("trial_count must be a positive integer.")
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer.")
        if self.selection_metric not in ALLOWED_SELECTION_METRICS:
            raise ValueError(
                "Unsupported selection_metric. "
                f"selection_metric={self.selection_metric!r}, allowed={sorted(ALLOWED_SELECTION_METRICS)}"
            )
        get_search_space(self.search_space_name)


@dataclass(frozen=True)
class XGBTuneResult:
    """
    tuning 실행 후 생성된 요약 산출물과 best trial 정보를 반환하는 객체
    """

    output_dir: Path
    tuning_summary_path: Path
    trials_summary_path: Path
    best_trial_id: str
    best_trial_dir: Path
    best_metric_value: float
    trials: list[dict[str, Any]]


# -----------------------------------------------------------------------------
# 2. tuning output directory 보호
# -----------------------------------------------------------------------------
def prepare_tuning_output_dir(output_dir: Path, overwrite: bool) -> None:
    """
    tuning output directory를 만들고 기존 산출물을 기본적으로 보호

    이유
    - random search는 trial별 산출물이 많아 조용한 overwrite가 발생하면 재현성이 깨짐
    - overwrite=False일 때 기존 파일이나 디렉터리가 있으면 즉시 실패
    """

    if output_dir.exists():
        existing = [str(path) for path in output_dir.iterdir()]
        if existing and not overwrite:
            raise FileExistsError(
                "Existing tuning artifacts found. Set overwrite=True to replace them. "
                f"existing={existing[:30]}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 3. random search parameter sampling
# -----------------------------------------------------------------------------
def sample_search_params(
    search_space: dict[str, list[Number]],
    trial_count: int,
    random_search_seed: int,
) -> list[dict[str, Number]]:
    """
    discrete search space에서 중복 없는 hyperparameter 조합을 샘플링

    동작 의도
    - itertools.product로 가능한 모든 조합을 만든 뒤 random_search_seed로 샘플링
    - 같은 seed와 같은 search space이면 항상 같은 trial 조합을 반환
    - trial_count가 가능한 조합 수보다 크면 중복 trial을 만들지 않고 실패
    """

    if trial_count <= 0:
        raise ValueError("trial_count must be a positive integer.")
    for param_name, values in search_space.items():
        if not values:
            raise ValueError(f"Search space parameter has no candidate values: {param_name}")

    param_names = list(search_space)
    combinations = list(itertools.product(*(search_space[name] for name in param_names)))
    if trial_count > len(combinations):
        raise ValueError(
            "trial_count exceeds the number of unique parameter combinations. "
            f"trial_count={trial_count}, available={len(combinations)}"
        )

    rng = random.Random(random_search_seed)
    sampled = rng.sample(combinations, k=trial_count)
    return [dict(zip(param_names, values)) for values in sampled]


# -----------------------------------------------------------------------------
# 4. random search 실행: train + validation 반복
# -----------------------------------------------------------------------------
def run_xgb_random_search(config: XGBTuneConfig) -> XGBTuneResult:
    """
    random search trial을 순차 실행하고 tuning summary를 저장

    동작 순서
    1. output_dir overwrite 안전장치 확인
    2. search space preset 로드
    3. trial_count만큼 중복 없는 parameter 조합 샘플링
    4. trial별 디렉터리에 trial_params.json 저장
    5. trial별 ml_00_ml_train.train_xgb() 실행
    6. trial별 ml_00_ml_val.validate_xgb() 실행
    7. selection_metric 기준 best trial 선택
    8. trials_summary.csv와 tuning_summary.json 저장

    주의
    - 실패 trial을 건너뛰지 않고 즉시 중단
    - final test는 호출하지 않음
    """

    prepare_tuning_output_dir(config.output_dir, overwrite=config.overwrite)
    search_space = get_search_space(config.search_space_name)
    sampled_params = sample_search_params(search_space, config.trial_count, config.random_search_seed)

    trial_records: list[dict[str, Any]] = []
    for trial_index, params in enumerate(sampled_params, start=1):
        trial_id = f"trial_{trial_index:04d}"
        trial_dir = config.output_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_params_path = trial_dir / "trial_params.json"
        save_json(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "trial_id": trial_id,
                "trial_index": trial_index,
                "search_space_name": config.search_space_name,
                "random_search_seed": config.random_search_seed,
                "seed": config.seed,
                "params": params,
                "final_test_evaluated": False,
            },
            trial_params_path,
        )

        train_result = train_xgb(
            XGBTrainConfig(
                train_path=config.train_path,
                val_path=config.val_path,
                feature_columns_path=config.feature_columns_path,
                output_dir=trial_dir,
                label_col=config.label_col,
                sample_rows=config.sample_rows,
                allow_nan=config.allow_nan,
                overwrite=config.overwrite,
                seed=config.seed,
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=float(params["subsample"]),
                colsample_bytree=float(params["colsample_bytree"]),
                min_child_weight=float(params["min_child_weight"]),
                reg_lambda=float(params["reg_lambda"]),
                reg_alpha=float(params["reg_alpha"]),
                gamma=float(params["gamma"]),
                early_stopping_rounds=int(params["early_stopping_rounds"]),
                n_jobs=config.n_jobs,
            )
        )
        val_result = validate_xgb(
            ValidationConfig(
                val_path=config.val_path,
                output_dir=trial_dir,
                label_col=config.label_col,
                sample_rows=config.sample_rows,
                allow_nan=config.allow_nan,
                overwrite=config.overwrite,
            )
        )

        train_summary = load_json(train_result.train_summary_path)
        metrics = val_result.val_metrics["summary"]
        trial_record = {
            "trial_id": trial_id,
            "trial_index": trial_index,
            "trial_dir": str(trial_dir),
            "search_space_name": config.search_space_name,
            "random_search_seed": config.random_search_seed,
            "seed": config.seed,
            "selection_metric": config.selection_metric,
            "selection_metric_value": float(metrics[config.selection_metric]),
            "val_average_precision": float(metrics["average_precision"]),
            "val_f1": float(metrics["f1"]),
            "val_recall": float(metrics["recall"]),
            "val_precision": float(metrics["precision"]),
            "threshold": float(metrics["threshold"]),
            "best_iteration": train_summary.get("best_iteration"),
            "best_score": train_summary.get("best_score"),
            "n_estimators_ceiling_hit": train_summary.get("n_estimators_ceiling_hit"),
            "training_time_sec": train_summary.get("training_time_sec"),
            "sample_rows": config.sample_rows,
            "sampled": config.sample_rows is not None,
            "final_test_evaluated": False,
        }
        trial_record.update({f"param_{name}": value for name, value in params.items()})
        trial_records.append(trial_record)

    best_trial = max(trial_records, key=lambda record: record["selection_metric_value"])
    trials_summary_path = config.output_dir / "trials_summary.csv"
    tuning_summary_path = config.output_dir / "tuning_summary.json"
    pd.DataFrame(trial_records).to_csv(trials_summary_path, index=False)
    save_json(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "search_space_name": config.search_space_name,
            "trial_count": config.trial_count,
            "random_search_seed": config.random_search_seed,
            "seed": config.seed,
            "selection_metric": config.selection_metric,
            "sample_rows": config.sample_rows,
            "sampled": config.sample_rows is not None,
            "train_path": str(config.train_path),
            "val_path": str(config.val_path),
            "feature_columns_path": str(config.feature_columns_path),
            "output_dir": str(config.output_dir),
            "trials_summary_file_name": "trials_summary.csv",
            "best_trial_id": best_trial["trial_id"],
            "best_trial_dir": best_trial["trial_dir"],
            "best_metric_value": best_trial["selection_metric_value"],
            "final_test_evaluated": False,
            "trials": trial_records,
        },
        tuning_summary_path,
    )

    return XGBTuneResult(
        output_dir=config.output_dir,
        tuning_summary_path=tuning_summary_path,
        trials_summary_path=trials_summary_path,
        best_trial_id=str(best_trial["trial_id"]),
        best_trial_dir=Path(str(best_trial["trial_dir"])),
        best_metric_value=float(best_trial["selection_metric_value"]),
        trials=trial_records,
    )
