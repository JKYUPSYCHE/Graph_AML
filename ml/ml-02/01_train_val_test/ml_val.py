"""
1. 코드 전체 요약
이 모듈은 학습된 XGBoost 모델의 validation split 성능을 계산하고,
test 평가에 사용할 threshold를 validation 기준으로 확정해 저장하는 코드다.

핵심 역할은 다음과 같다.
- model.pkl, feature_columns.json, train_summary.json을 로드
- validation parquet를 학습 당시 feature 순서로 로드
- predict_proba()로 laundering 확률 계산
- max_f1 또는 manual 방식으로 threshold 선택
- threshold.json, metrics_val.json, confusion_matrix_val.csv, prediction_scores_val.parquet 저장
- test split은 읽지 않음

2. 데이터 흐름 요약
ValidationConfig
→ 경로 정규화 / 옵션 검증
→ model.pkl, feature_columns.json, train_summary.json 존재 확인
→ feature column 목록 로드
→ train_summary와 현재 validation 입력의 provenance 검증
→ validation parquet 로드
→ model.predict_proba()
→ threshold 선택
→ validation metric 계산
→ threshold.json 저장
→ confusion_matrix_val.csv 저장
→ prediction_scores_val.parquet 저장
→ metrics_val.json 저장
→ ValidationResult 반환

중요한 데이터 연결 고리는 다음이다.
- feature_columns.json: 학습 때 사용한 feature 순서
- feature_columns_hash: feature 목록/순서 정합성 검증
- train_summary.json: 학습 run과 validation run 연결
- threshold.json: test 평가에서 재사용해야 하는 threshold
- metrics_val.json: validation 성능과 실행 환경 기록
- prediction_scores_val.parquet: threshold sweep 재계산용 row별 score

3. 변경 시 주의점
- load_split(... expected_split="val")은 validation 전용 검증이므로 test 평가 코드에 그대로 복사하면 안 된다.
- threshold를 test set에서 다시 고르면 데이터 누수이다.
- feature_columns 순서를 바꾸면 모델 입력 순서가 달라져 예측 결과가 깨질 수 있다.
- train_summary.json의 val_path, feature_columns_hash, model_sha256 검증은 재현성 보호 장치이므로 임의 제거하면 안 된다.
- sample_rows를 쓰면 smoke/debug 결과이며 전체 validation 성능으로 해석하면 안 된다.
- overwrite=False 기본값은 기존 결과 보호 목적이다.
- manual_threshold는 validation 정책을 고정할 때만 사용해야 한다.


Validation split에서 threshold를 선택하고 validation metric 산출물을 저장하는 모듈

전체 흐름
1. ml_train.train_xgb()가 저장한 model.pkl, feature_columns.json, train_summary.json을 읽음
2. validation parquet를 학습 때 저장한 feature 순서로 로드
3. 모델이 예측한 positive-class probability를 계산
4. threshold_strategy에 따라 threshold를 선택
   - max_f1: validation split에서 F1이 최대가 되는 threshold 자동 선택
   - manual: 사용자가 지정한 manual_threshold를 고정 threshold로 사용
5. threshold.json, metrics_val.json, confusion_matrix_val.csv, prediction_scores_val.parquet를 저장

중요한 전제
- threshold 선택은 validation split에서만 수행
- 기본 threshold 기준은 max_f1이며, 수동 고정이 필요할 때만 manual을 사용
- test split은 이 모듈에서 절대 읽거나 평가하지 않음
- feature_columns_hash로 학습 시점 feature 순서와 validation 입력 feature 순서를 연결
- test 단계에서는 여기서 저장한 threshold.json을 그대로 사용해야 하며, test set에서 threshold를 다시 고르면 데이터 누수 위험이 있음
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from _ml_io_artifacts import _load_training_artifact_bundle
from ml_io import (
    categorical_columns_from_manifest,
    file_sha256,
    label_summary,
    load_json,
    load_split,
    read_parquet_columns,
    resolve_project_path,
    save_json,
)
from ml_metrics import confusion_matrix_frame, evaluate_at_threshold, select_threshold_by_f1
from ml_resource import (
    MemoryTracker,
    RuntimeTracker,
    collect_environment,
    derive_feature_assoc_file_name,
    make_data_profile,
    make_mixed_feature_association_payload,
    make_run_metadata,
    make_score_profile,
)


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
    # validation parquet 입력 경로.
    # 이 파일은 모델 학습에 쓰인 validation split과 동일해야 하며, 아래 validate_xgb()에서 train_summary.json의 val_path와 비교 검증된다.
    val_path: Path | str

    # 학습 산출물과 validation 산출물을 함께 관리하는 디렉터리.
    # model.pkl, feature_columns.json, train_summary.json을 여기서 읽고, threshold.json, metrics_val.json, confusion_matrix_val.csv도 여기에 저장한다.
    output_dir: Path | str

    # 상대경로 해석 기준. None이면 resolve_project_path()의 기본 기준을 따른다.
    project_root: Path | str | None = None
    label_col: str = "label"        # 정답 라벨 컬럼명.
    sample_rows: int | None = None  # validation 전체 대신 앞쪽 일부 row만 읽는 옵션. 빠른 smoke test 용
    allow_nan: bool = False         # feature 결측치 허용 여부.
    overwrite: bool = False         # 기존 validation 산출물 덮어쓰기 허용 여부.

    # 학습 단계 산출물 파일명. output_dir 아래에서 이 파일들을 읽어 validation을 수행한다.
    model_file_name: str = "model.pkl"
    feature_columns_file_name: str = "feature_columns.json"
    train_summary_file_name: str = "train_summary.json"
    scores_train_summary_file_name: str = "scores_train_summary.json"

    # validation 단계 산출물 파일명.  test 평가에서는 여기서 저장한 threshold.json을 재사용해야 한다.
    threshold_file_name: str = "threshold.json"
    metrics_file_name: str = "metrics_val.json"
    confusion_matrix_file_name: str = "confusion_matrix_val.csv"
    prediction_scores_file_name: str | None = None
    export_prediction_scores: bool = True

    threshold_strategy: str = "max_f1"    # threshold 선택 정책
    manual_threshold: float | None = None # manual 전략일 때만 쓰는 고정 threshold.
    encoding_manifest_path: Path | str | None = None
    export_feature_assoc: bool = True
    feature_assoc_val_sample_rows: int = 300_000

    def __post_init__(self) -> None:
        """dataclass 생성 직후 경로를 정규화하고 sample_rows 값을 검증"""
        # frozen=True dataclass이므로 일반 대입이 불가능, object.__setattr__으로 생성 직후에만 경로를 정규화한다.
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
        if self.encoding_manifest_path is not None:
            object.__setattr__(
                self,
                "encoding_manifest_path",
                resolve_project_path(self.encoding_manifest_path, self.project_root),
            )

        prediction_scores_file_name = self.prediction_scores_file_name
        if prediction_scores_file_name is None:
            prediction_scores_file_name = derive_prediction_scores_file_name(self.metrics_file_name)
            object.__setattr__(self, "prediction_scores_file_name", prediction_scores_file_name)
        elif not str(prediction_scores_file_name).strip():
            raise ValueError("prediction_scores_file_name must be a non-empty file name.")
        elif not str(prediction_scores_file_name).endswith(".parquet"):
            raise ValueError(
                "prediction_scores_file_name must be a parquet file. "
                f"prediction_scores_file_name={prediction_scores_file_name!r}"
            )

        # sample_rows는 validation 일부만 읽는 옵션이므로 1 이상이 아니면 중단시킨다.
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer.")
        if self.feature_assoc_val_sample_rows <= 0:
            raise ValueError("feature_assoc_val_sample_rows must be a positive integer.")

        # threshold 선택 정책은 명시적으로 허용된 값만 받는다.
        allowed_threshold_strategies = {"max_f1", "manual"}
        if self.threshold_strategy not in allowed_threshold_strategies:
            raise ValueError(
                "Unsupported threshold_strategy. "
                f"threshold_strategy={self.threshold_strategy!r}, "
                f"allowed={sorted(allowed_threshold_strategies)}"
            )

        # manual 전략에서는 반드시 manual_threshold가 필요하다. 이 값은 모델 score를 class label로 바꾸는 기준이므로 0~1 범위만 허용한다.
        if self.threshold_strategy == "manual":
            if self.manual_threshold is None:
                raise ValueError("manual_threshold is required when threshold_strategy='manual'.")
            manual_threshold = float(self.manual_threshold)
            if not 0 <= manual_threshold <= 1:
                raise ValueError(f"manual_threshold must be between 0 and 1. manual_threshold={manual_threshold}")

        # max_f1 전략에서는 manual_threshold를 받지 않는다. 두 정책이 섞이면 결과 해석이 모호해지므로 설정 단계에서 차단한다.
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
    output_dir: Path          # validation 산출물이 저장된 디렉터리.
    threshold_path: Path      # test 평가에서 재사용해야 하는 threshold.json 경로.
    metrics_path: Path        # validation 성능, score profile, runtime, memory 정보가 저장된 JSON 경로.
    metrics_train_path: Path            
    confusion_matrix_path: Path          # validation confusion matrix CSV 경로.
    prediction_scores_path: Path | None  # validation row별 prediction score parquet 경로.
    feature_assoc_path: Path | None
    threshold_info: dict[str, Any]   # threshold.json에 저장한 payload를 메모리에서도 바로 확인하기 위한 값
    val_metrics: dict[str, Any]      # evaluate_at_threshold() 결과.
    train_metrics: dict[str, Any]


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
    - prediction_scores_val.parquet
    """
    # validation이 생성할 파일만 overwrite 보호 대상으로 본다.
    # 학습 산출물(model.pkl 등)은 읽기 대상이므로 여기서 삭제하거나 갱신하지 않는다.
    output_paths = [
        config.output_dir / config.threshold_file_name,
        config.output_dir / config.metrics_file_name,
        config.output_dir / derive_metrics_train_file_name(config.metrics_file_name),
        config.output_dir / config.confusion_matrix_file_name,
    ]
    if config.export_prediction_scores:
        output_paths.append(config.output_dir / str(config.prediction_scores_file_name))
    if config.export_feature_assoc:
        output_paths.append(config.output_dir / derive_feature_assoc_file_name(config.metrics_file_name, "val"))

    # 기존 산출물이 하나라도 있으면 overwrite=True가 아닌 이상 실패시킨다.
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError(
            "Existing validation artifacts found. Set overwrite=True to replace them. "
            f"existing={existing}"
        )


def derive_prediction_scores_file_name(metrics_file_name: str) -> str:
    """metrics_val 파일명에서 같은 artifact prefix의 validation score parquet 파일명을 만든다."""

    name = str(metrics_file_name)
    if name == "metrics_val.json":
        return "prediction_scores_val.parquet"
    if name.endswith("_metrics_val.json"):
        return f"{name.removesuffix('_metrics_val.json')}_prediction_scores_val.parquet"
    if name.endswith(".json"):
        return f"{name.removesuffix('.json')}_prediction_scores_val.parquet"
    return f"{name}_prediction_scores_val.parquet"

def derive_metrics_train_file_name(metrics_file_name: str) -> str:
    """metrics_val 파일명에서 같은 artifact prefix의 train metrics 파일명을 만든다."""

    name = str(metrics_file_name)
    if name == "metrics_val.json":
        return "metrics_train.json"
    if name.endswith("_metrics_val.json"):
        return f"{name.removesuffix('_metrics_val.json')}_metrics_train.json"
    if name.endswith(".json"):
        return f"{name.removesuffix('.json')}_metrics_train.json"
    return f"{name}_metrics_train.json"


def save_validation_prediction_scores(
    *,
    val_path: Path,
    output_path: Path,
    y_val: pd.Series,
    probabilities,
    run_id: str,
    label_col: str,
    sample_rows: int | None,
) -> dict[str, Any]:
    """validation row별 tx_id, label, score를 parquet로 저장하고 저장 메타데이터를 반환한다."""

    metadata_columns = ["tx_id", "split", label_col]
    metadata = read_parquet_columns(val_path, metadata_columns, sample_rows)

    if len(metadata) != len(y_val) or len(metadata) != len(probabilities):
        raise ValueError(
            "Prediction score row count mismatch. "
            f"metadata_rows={len(metadata)}, y_rows={len(y_val)}, score_rows={len(probabilities)}, val_path={val_path}"
        )

    if metadata["tx_id"].isna().any():
        raise ValueError(f"tx_id contains null values in validation metadata. val_path={val_path}")

    split_values = metadata["split"].astype("string").str.strip().str.lower()
    if split_values.isna().any() or set(split_values.unique().tolist()) != {"val"}:
        raise ValueError(
            "Validation metadata has unexpected split values. "
            f"values={sorted(str(value) for value in split_values.dropna().unique().tolist())}, val_path={val_path}"
        )

    metadata_y = pd.to_numeric(metadata[label_col], errors="raise").astype("int8").reset_index(drop=True)
    loaded_y = pd.Series(y_val, name=label_col).astype("int8").reset_index(drop=True)
    if not metadata_y.equals(loaded_y):
        raise ValueError(
            "Validation label mismatch while saving prediction scores. "
            "The metadata rows may not match the model input rows."
        )

    scores = pd.Series(probabilities, name="score", dtype="float64").reset_index(drop=True)
    if scores.isna().any():
        raise ValueError("Prediction scores contain NaN values.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction scores must be between 0 and 1.")

    prediction_scores = pd.DataFrame(
        {
            "tx_id": metadata["tx_id"].astype("string").reset_index(drop=True),
            "split": split_values.reset_index(drop=True),
            "label": loaded_y,
            "score": scores,
            "run_id": str(run_id),
            "row_position": range(len(scores)),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_scores.to_parquet(output_path, index=False)

    return {
        "path": str(output_path),
        "file_name": output_path.name,
        "rows": int(len(prediction_scores)),
        "columns": prediction_scores.columns.tolist(),
        "score_column": "score",
        "label_column": "label",
        "id_column": "tx_id",
        "split": "val",
        "sample_rows": sample_rows,
        "sampled": sample_rows is not None,
        "sha256": file_sha256(output_path),
    }


def _prepare_validation(config: ValidationConfig) -> tuple[Path, Path, Path, Path]:
    """Resolve required training artifact paths and protect validation outputs."""

    model_path = config.output_dir / config.model_file_name
    feature_columns_path = config.output_dir / config.feature_columns_file_name
    train_summary_path = config.output_dir / config.train_summary_file_name
    scores_train_summary_path = config.output_dir / config.scores_train_summary_file_name

    if not scores_train_summary_path.exists():
        raise FileNotFoundError(
            "scores train summary file not found. Rerun ml_train.train_xgb() with logloss learning curve enabled: "
            f"{scores_train_summary_path}"
        )
    if not config.val_path.exists():
        raise FileNotFoundError(f"validation parquet not found: {config.val_path}")

    prepare_validation_outputs(config)
    return model_path, feature_columns_path, train_summary_path, scores_train_summary_path


def _load_validation_artifacts(
    *,
    config: ValidationConfig,
    model_path: Path,
    feature_columns_path: Path,
    train_summary_path: Path,
    scores_train_summary_path: Path,
) -> dict[str, Any]:
    """Load model artifacts, verify provenance, and read the training learning curve."""

    bundle = _load_training_artifact_bundle(
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        train_summary_path=train_summary_path,
        current_val_path=config.val_path,
        encoding_manifest_path=config.encoding_manifest_path,
    )
    scores_train_summary = load_json(scores_train_summary_path)
    scores_train_summary_sha256 = file_sha256(scores_train_summary_path)
    learning_curve = scores_train_summary.get("learning_curve")
    if not isinstance(learning_curve, dict) or learning_curve.get("loss_name") != "logloss":
        raise ValueError(
            "scores_train_summary.json is missing logloss learning_curve. "
            f"path={scores_train_summary_path}"
        )

    return {
        "model": bundle.model,
        "feature_columns": bundle.feature_columns,
        "features_hash": bundle.feature_columns_hash,
        "train_summary": bundle.train_summary,
        "train_run_id": bundle.train_summary["run_id"],
        "model_sha256": bundle.model_sha256,
        "feature_columns_file_sha256": bundle.feature_columns_file_sha256,
        "train_summary_sha256": bundle.train_summary_sha256,
        "encoding_manifest_path": bundle.encoding_manifest_path,
        "encoding_manifest": bundle.encoding_manifest,
        "scores_train_summary_sha256": scores_train_summary_sha256,
        "learning_curve": learning_curve,
    }


def _load_validation_split(
    *,
    config: ValidationConfig,
    feature_columns: list[str],
    encoding_manifest: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load validation X/y using the training feature order."""

    return load_split(
        config.val_path,
        feature_columns=feature_columns,
        label_col=config.label_col,
        sample_rows=config.sample_rows,
        allow_nan=config.allow_nan,
        expected_split="val",
        encoding_manifest=encoding_manifest,
    )


def _run_validation(
    *,
    config: ValidationConfig,
    model: Any,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    runtime_tracker: RuntimeTracker,
) -> tuple[Any, float, dict[str, Any], dict[str, Any]]:
    """Predict validation scores, select threshold, and compute metrics."""

    with runtime_tracker.measure("predict_proba"):
        probabilities = model.predict_proba(x_val)[:, 1]

    with runtime_tracker.measure("threshold_selection"):
        if config.threshold_strategy == "max_f1":
            threshold_info = select_threshold_by_f1(y_val, probabilities)
            threshold = threshold_info["threshold"]
        elif config.threshold_strategy == "manual":
            threshold = float(config.manual_threshold)
            threshold_info = evaluate_at_threshold(y_val, probabilities, threshold)["summary"]
        else:
            raise ValueError(f"Unsupported threshold_strategy: {config.threshold_strategy!r}")

    with runtime_tracker.measure("evaluate"):
        val_metrics = evaluate_at_threshold(y_val, probabilities, threshold)
    return probabilities, float(threshold), threshold_info, val_metrics


def _save_validation_outputs(
    *,
    config: ValidationConfig,
    threshold_payload: dict[str, Any],
    val_metrics: dict[str, Any],
    y_val: pd.Series,
    probabilities: Any,
    train_run_id: str,
    seed: Any,
    x_val: pd.DataFrame,
    feature_columns: list[str],
    encoding_manifest: dict[str, Any] | None,
    features_hash: str,
) -> dict[str, Any]:
    """Save validation threshold, metrics side artifacts, and optional row scores."""

    threshold_path = config.output_dir / config.threshold_file_name
    metrics_path = config.output_dir / config.metrics_file_name
    confusion_matrix_path = config.output_dir / config.confusion_matrix_file_name
    prediction_scores_path = (
        config.output_dir / str(config.prediction_scores_file_name)
        if config.export_prediction_scores
        else None
    )
    feature_assoc_path = config.output_dir / derive_feature_assoc_file_name(config.metrics_file_name, "val")

    prediction_scores_info = None
    if config.export_prediction_scores:
        if prediction_scores_path is None:
            raise ValueError("prediction_scores_path is required when export_prediction_scores=True.")
        prediction_scores_info = save_validation_prediction_scores(
            val_path=config.val_path,
            output_path=prediction_scores_path,
            y_val=y_val,
            probabilities=probabilities,
            run_id=train_run_id,
            label_col=config.label_col,
            sample_rows=config.sample_rows,
        )

    save_json(threshold_payload, threshold_path)
    confusion_matrix_frame(val_metrics).to_csv(confusion_matrix_path, index=False)

    if config.export_feature_assoc:
        feature_assoc_payload = make_mixed_feature_association_payload(
            x_val,
            feature_columns,
            categorical_feature_columns=categorical_columns_from_manifest(encoding_manifest, feature_columns),
            split="val",
            source_path=config.val_path,
            feature_columns_hash_value=features_hash,
            run_id=train_run_id,
            run_metadata=make_run_metadata(
                config.output_dir,
                seed=seed,
                artifact_file_name=feature_assoc_path.name,
            ),
            max_rows=config.feature_assoc_val_sample_rows,
            sample_strategy="deterministic_evenly_spaced",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        save_json(feature_assoc_payload, feature_assoc_path)

    return {
        "threshold_path": threshold_path,
        "metrics_path": metrics_path,
        "confusion_matrix_path": confusion_matrix_path,
        "prediction_scores_path": prediction_scores_path,
        "feature_assoc_path": feature_assoc_path,
        "prediction_scores_info": prediction_scores_info,
    }


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
    # 전체 validation 함수 실행 시간을 별도로 기록한다. RuntimeTracker의 구간별 시간과 함께 metrics_val.json에 저장된다.
    total_started = time.perf_counter()

    # 구간별 runtime 측정기. load_artifacts, load_val_split, predict_proba 등 병목 지점을 나눠 기록한다.
    runtime_tracker = RuntimeTracker()

    # validation 단계의 메모리 사용량 추적기.
    memory_tracker = MemoryTracker(scope="validate")
    memory_tracker.start()
    try:
        with runtime_tracker.measure("prepare_validation"):
            model_path, feature_columns_path, train_summary_path, scores_train_summary_path = _prepare_validation(config)

        with runtime_tracker.measure("load_artifacts"):
            artifacts = _load_validation_artifacts(
                config=config,
                model_path=model_path,
                feature_columns_path=feature_columns_path,
                train_summary_path=train_summary_path,
                scores_train_summary_path=scores_train_summary_path,
            )

        feature_columns = artifacts["feature_columns"]
        features_hash = artifacts["features_hash"]
        train_summary = artifacts["train_summary"]
        train_run_id = artifacts["train_run_id"]
        model = artifacts["model"]
        model_sha256 = artifacts["model_sha256"]
        feature_columns_file_sha256 = artifacts["feature_columns_file_sha256"]
        train_summary_sha256 = artifacts["train_summary_sha256"]
        encoding_manifest_path = artifacts["encoding_manifest_path"]
        encoding_manifest = artifacts["encoding_manifest"]
        scores_train_summary_sha256 = artifacts["scores_train_summary_sha256"]
        learning_curve = artifacts["learning_curve"]

        memory_tracker.snapshot("after_artifact_load")

        with runtime_tracker.measure("load_val_split"):
            x_val, y_val = _load_validation_split(
                config=config,
                feature_columns=feature_columns,
                encoding_manifest=encoding_manifest,
            )

        memory_tracker.snapshot("after_val_load")
        probabilities, threshold, threshold_info, val_metrics = _run_validation(
            config=config,
            model=model,
            x_val=x_val,
            y_val=y_val,
            runtime_tracker=runtime_tracker,
        )
        train_path_value = train_summary.get("train_path")
        if not train_path_value:
            raise ValueError("train_summary.json is missing train_path.")

        train_path = Path(str(train_path_value)).expanduser().resolve()
        if not train_path.exists():
            raise FileNotFoundError(f"train parquet not found: {train_path}")

        with runtime_tracker.measure("load_train_split_for_metrics"):
            x_train, y_train = load_split(
                train_path,
                feature_columns=feature_columns,
                label_col=config.label_col,
                sample_rows=config.sample_rows,
                allow_nan=config.allow_nan,
                expected_split="train",
                encoding_manifest=encoding_manifest,
            )

        memory_tracker.snapshot("after_train_metric_load")

        with runtime_tracker.measure("predict_proba_train"):
            train_probabilities = model.predict_proba(x_train)[:, 1]

        with runtime_tracker.measure("evaluate_train"):
            train_metrics = evaluate_at_threshold(y_train, train_probabilities, threshold)

        with runtime_tracker.measure("build_metadata"):
            # 학습 summary 안의 run_metadata에서 seed를 가져온다.
            # 구버전 train_summary 호환을 위해 top-level seed도 fallback으로 보는 구조로 보인다.
            train_run_metadata = train_summary.get("run_metadata")
            seed = train_run_metadata.get("seed") if isinstance(train_run_metadata, dict) else train_summary.get("seed")

            # validation 데이터 profile 생성.
            data_profile = make_data_profile({"val": (config.val_path, x_val, y_val)}, feature_columns)

            train_data_profile = make_data_profile({"train": (train_path, x_train, y_train)}, feature_columns)

            # score 분포 profile 생성.
            score_profile = make_score_profile(y_val, probabilities, threshold)

            train_score_profile = make_score_profile(y_train, train_probabilities, threshold)

            # 실행 환경 정보 수집.
            environment = collect_environment()

        # threshold.json에 저장할 내용
        # 목적은 "어떤 validation 기준으로 threshold를 골랐는지"를 재현 가능하게 남기는 것
        threshold_payload = {
            # UTC 생성 시각.
            "created_at": datetime.now(timezone.utc).isoformat(),

            # 학습 run ID. validation과 test 결과를 같은 학습 run에 연결하는 핵심 키다.
            "run_id": train_run_id,

            # validation 산출물 기준 metadata.
            # make_run_metadata()가 어떤 필드를 넣는지는 확인 필요.
            "run_metadata": make_run_metadata(
                config.output_dir,
                seed=seed,
                artifact_file_name=config.threshold_file_name,
            ),

            # threshold 선택에 사용한 split.
            # 이 값은 항상 val이어야 하며 test로 바뀌면 데이터 누수다.
            "selection_split": "val",

            # threshold 선택 방식.
            "selection_metric": config.threshold_strategy,

            # 최종 선택된 threshold. test 단계는 이 값을 그대로 사용해야 한다.
            "threshold": float(threshold_info["threshold"]),

            # sample_rows 사용 여부 기록. sample이 적용된 threshold는 최종 결과로 쓰기 전에 주의해야 한다.
            "sample_rows": config.sample_rows,
            "sampled": config.sample_rows is not None,

            # validation 입력 파일 경로와 feature 정보.
            "val_path": str(config.val_path),
            "feature_count": len(feature_columns),
            "feature_columns_hash": features_hash,
            "encoding_manifest_path": None if encoding_manifest_path is None else str(encoding_manifest_path),

            # 학습 산출물 무결성 추적용 hash.
            "model_sha256": model_sha256,
            "feature_columns_file_sha256": feature_columns_file_sha256,
            "train_summary_sha256": train_summary_sha256,

            # 파일명과 전체 경로를 모두 저장해 후속 추적을 쉽게 한다.
            "model_file_name": config.model_file_name,
            "feature_columns_file_name": config.feature_columns_file_name,
            "train_summary_file_name": config.train_summary_file_name,
            "scores_train_summary_file_name": config.scores_train_summary_file_name,
            "model_path": str(model_path),
            "feature_columns_path": str(feature_columns_path),
            "train_summary_path": str(train_summary_path),
            "scores_train_summary_path": str(scores_train_summary_path),

            # 학습이 sample 기반이었는지 기록. 학습 자체가 sample이면 validation metric 해석에도 제한이 생긴다.
            "train_sampled": bool(train_summary.get("sampled")),
            "train_sample_rows": train_summary.get("sample_rows"),

            # 선택 threshold에서의 metric 또는 threshold 탐색 결과.
            # max_f1일 때와 manual일 때 구조가 완전히 동일한지는 확인 필요.
            "metrics_at_threshold": threshold_info,

            # threshold 정책 재현을 위한 명시 기록.
            "threshold_strategy": config.threshold_strategy,
            "manual_threshold": config.manual_threshold,
        }
        with runtime_tracker.measure("save_outputs"):
            saved_outputs = _save_validation_outputs(
                config=config,
                threshold_payload=threshold_payload,
                val_metrics=val_metrics,
                y_val=y_val,
                probabilities=probabilities,
                train_run_id=train_run_id,
                seed=seed,
                x_val=x_val,
                feature_columns=feature_columns,
                encoding_manifest=encoding_manifest,
                features_hash=features_hash,
            )

        threshold_path = saved_outputs["threshold_path"]
        metrics_path = saved_outputs["metrics_path"]
        confusion_matrix_path = saved_outputs["confusion_matrix_path"]
        prediction_scores_path = saved_outputs["prediction_scores_path"]
        feature_assoc_path = saved_outputs["feature_assoc_path"]
        prediction_scores_info = saved_outputs["prediction_scores_info"]

        # validation 종료 시점 메모리 스냅샷.
        memory_tracker.snapshot("end")

    finally:
        # 예외 발생 여부와 관계없이 메모리 측정을 종료한다.
        # 단, 예외가 prepare 단계 이전에 발생하면 아래 try 블록 밖 metrics_payload 생성은 실행되지 않는다.
        memory_profile = memory_tracker.finish()

    # 구간별 runtime을 dict로 변환하고 전체 함수 실행 시간을 추가한다.
    runtime_sec = runtime_tracker.as_dict()
    runtime_sec["total_validate_xgb"] = float(time.perf_counter() - total_started)

    # metrics_val.json에 저장할 내용
    # 목적은 validation split에서 고정 threshold로 계산한 성능, score 분포, 실행 정보를 남기는 것
    metrics_payload = {
        # UTC 생성 시각.
        "created_at": datetime.now(timezone.utc).isoformat(),

        # 이 metric이 validation split 기준임을 명시한다.
        "split": "val",

        # 학습 run ID.
        "run_id": train_run_id,

        # validation 산출물 기준 metadata.
        "run_metadata": make_run_metadata(
            config.output_dir,
            seed=seed,
            artifact_file_name=config.metrics_file_name,
        ),

        # validation 데이터 경로와 sample 여부.
        "val_path": str(config.val_path),
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,

        # feature 정합성 추적 정보.
        "feature_count": len(feature_columns),
        "feature_columns_hash": features_hash,
        "encoding_manifest_path": None if encoding_manifest_path is None else str(encoding_manifest_path),

        # validation label 분포 요약.
        # 클래스 불균형 확인에 중요하며, Accuracy 단독 해석을 피하는 근거가 된다.
        "label_summary": label_summary(y_val),

        # threshold 적용 후 validation metric summary.
        # F1, precision, recall, AP 등이 포함되는지는 evaluate_at_threshold() 구현 확인 필요.
        "metrics": val_metrics["summary"],

        # threshold 적용 후 confusion matrix.
        "confusion_matrix": val_metrics["confusion_matrix"],

        # 메모리 사용량 요약과 의미 설명.
        "memory_mb": memory_profile["memory_mb"],
        "memory_mb_semantics": memory_profile["memory_mb_semantics"],

        # 구간별 runtime과 전체 runtime.
        "runtime_sec": runtime_sec,

        # 상세 메모리 profile.
        "memory_profile": memory_profile,

        # validation 입력 데이터 profile.
        "data_profile": data_profile,

        # 실행 환경 정보.
        "environment": environment,

        # probability score 분포와 threshold 기준 profile.
        "score_profile": score_profile,

        # row별 validation score parquet. 추후 threshold sweep 재계산에 사용한다.
        "prediction_scores": prediction_scores_info,

        "feature_association_artifacts": {
            "val": str(feature_assoc_path) if config.export_feature_assoc else None,
        },

        # 리더보드가 metrics_val.json만 읽어도 train/val logloss curve를 그릴 수 있게 동일 구조를 포함한다.
        "learning_curve": {
            **learning_curve,
            "source_file_name": config.scores_train_summary_file_name,
            "source_path": str(scores_train_summary_path),
            "source_sha256": scores_train_summary_sha256,
        },
    }

    # metrics_val.json 저장.
    # threshold.json, confusion_matrix_val.csv 저장 이후 마지막으로 저장된다.
    save_json(metrics_payload, metrics_path)

    metrics_train_path = config.output_dir / derive_metrics_train_file_name(config.metrics_file_name)

    metrics_train_payload = {
        **metrics_payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "run_metadata": make_run_metadata(
            config.output_dir,
            seed=seed,
            artifact_file_name=metrics_train_path.name,
        ),
        "train_path": str(train_path),
        "val_path": str(config.val_path),
        "label_summary": label_summary(y_train),
        "metrics": train_metrics["summary"],
        "confusion_matrix": train_metrics["confusion_matrix"],
        "data_profile": train_data_profile,
        "score_profile": train_score_profile,
        "prediction_scores": None,
        "feature_association_artifacts": {
            "train": train_summary.get("feature_association_artifacts", {}).get("train"),
        },
    }
    save_json(metrics_train_payload, metrics_train_path)

    # 노트북/후속 코드에서 저장 경로와 metric 내용을 바로 확인할 수 있도록 결과 객체를 반환
    return ValidationResult(
        output_dir=config.output_dir,
        threshold_path=threshold_path,
        metrics_path=metrics_path,
        metrics_train_path=metrics_train_path,
        confusion_matrix_path=confusion_matrix_path,
        prediction_scores_path=prediction_scores_path,
        feature_assoc_path=feature_assoc_path if config.export_feature_assoc else None,
        threshold_info=threshold_payload,
        val_metrics=val_metrics,
        train_metrics=train_metrics,
    )
