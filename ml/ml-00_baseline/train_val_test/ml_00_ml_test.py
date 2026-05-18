"""
최종 test split 평가 전용 모듈

전체 흐름
1. 사용자가 confirm_final_test=True를 명시했는지 확인
2. 학습 산출물과 validation threshold 산출물이 final test에 안전한 상태인지 검증
3. test parquet를 학습 때 저장한 feature 순서로 로드
4. 고정된 validation threshold로 test metric을 계산
5. metrics_test.json, confusion_matrix_test.csv를 저장

안전 규칙
- 모델/feature/threshold 선택 중에는 실행하지 않음
- final test는 최종 configuration 확정 후 1회만 실행
- sampled model, sampled threshold, sampled test는 차단
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ml_00_ml_io import (
    feature_columns_hash,
    file_sha256,
    label_summary,
    load_json,
    load_saved_feature_columns,
    load_split,
    resolve_project_path,
    save_json,
)
from ml_00_ml_metrics import confusion_matrix_frame, evaluate_at_threshold
from ml_00_ml_resource import (
    MemoryTracker,
    RuntimeTracker,
    collect_environment,
    make_data_profile,
    make_run_metadata,
    make_score_profile,
)


# -----------------------------------------------------------------------------
# 1. Final test 실행 설정과 반환 결과 자료구조
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TestConfig:
    """
    final test 실행에 필요한 입력 경로와 안전 옵션 묶음

    주요 안전장치
    - confirm_final_test=False가 기본값이므로 실수로 test를 실행하지 못함
    - sample_rows는 final test에서 사용할 수 없음
    - overwrite=False가 기본값이므로 기존 final test 산출물을 보호
    """
    test_path: Path | str
    output_dir: Path | str
    project_root: Path | str | None = None
    confirm_final_test: bool = False
    label_col: str = "label"
    sample_rows: int | None = None
    allow_nan: bool = False
    overwrite: bool = False

    model_file_name: str = "model.pkl"
    feature_columns_file_name: str = "feature_columns.json"
    train_summary_file_name: str = "train_summary.json"
    threshold_file_name: str = "threshold.json"
    metrics_file_name: str = "metrics_test.json"
    confusion_matrix_file_name: str = "confusion_matrix_test.csv"

    def __post_init__(self) -> None:
        """dataclass 생성 직후 경로를 정규화하고 sample_rows 값을 검증"""
        object.__setattr__(
            self,
            "test_path",
            resolve_project_path(self.test_path, self.project_root),
        )
        object.__setattr__(
            self,
            "output_dir",
            resolve_project_path(self.output_dir, self.project_root),
        )
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer.")


@dataclass(frozen=True)
class TestResult:
    """
    final test 실행 후 생성된 산출물 경로와 metric 결과를 반환하는 객체
    """
    output_dir: Path
    metrics_path: Path
    confusion_matrix_path: Path
    test_metrics: dict[str, Any]


# -----------------------------------------------------------------------------
# 2. final test 산출물 overwrite 보호
# -----------------------------------------------------------------------------
def prepare_test_outputs(config: TestConfig) -> None:
    """
    final test 산출물이 이미 있을 때 조용히 덮어쓰지 않도록 차단

    이유
    - final test는 최종 1회 사용 원칙이 있으므로 재실행 흔적을 명시적으로 관리해야 함
    - overwrite=True는 의도적인 재평가 상황에서만 사용해야 함
    """

    output_paths = [
        config.output_dir / config.metrics_file_name,
        config.output_dir / config.confusion_matrix_file_name,
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError(
            "Existing final test artifacts found. Set overwrite=True only if rerunning final evaluation intentionally. "
            f"existing={existing}"
        )


# -----------------------------------------------------------------------------
# 3. final test provenance 검증
# -----------------------------------------------------------------------------
def require_final_test_provenance(
    threshold_payload: dict[str, Any],
    train_summary: dict[str, Any],
    feature_columns_hash_value: str,
    model_sha256_value: str,
    feature_columns_file_sha256_value: str,
    train_summary_sha256_value: str,
    config: TestConfig,
) -> None:
    """
    모델과 threshold 산출물이 final test에 사용할 수 있는 상태인지 검증

    검증 항목
    - train_summary가 sampled 학습 결과가 아닌지 확인
    - threshold.json이 sampled validation 결과가 아닌지 확인
    - threshold가 validation split에서 선택되었는지 확인
    - feature_columns_hash와 주요 파일명이 서로 맞는지 확인
    """

    if bool(train_summary.get("sampled")) or train_summary.get("sample_rows") is not None:
        raise ValueError(
            "Final test is blocked because the model was trained with sampled data. "
            "Run ml_00_ml_train.train_xgb() with sample_rows=None before final test."
        )

    if bool(threshold_payload.get("sampled")) or threshold_payload.get("sample_rows") is not None:
        raise ValueError(
            "Final test is blocked because threshold.json was produced from sampled validation data. "
            "Run ml_00_ml_val.validate_xgb() with sample_rows=None before final test."
        )

    train_run_id = train_summary.get("run_id")
    if not train_run_id:
        raise ValueError("train_summary.json is missing run_id. Rerun ml_00_ml_train.train_xgb() with the updated module.")
    if threshold_payload.get("run_id") != train_run_id:
        raise ValueError(
            "Final test provenance check failed: run_id. "
            f"threshold={threshold_payload.get('run_id')!r}, train_summary={train_run_id!r}"
        )

    checks = {
        "train_feature_columns_hash": (train_summary.get("feature_columns_hash"), feature_columns_hash_value),
        "train_model_sha256": (train_summary.get("model_sha256"), model_sha256_value),
        "train_feature_columns_file_sha256": (
            train_summary.get("feature_columns_file_sha256"),
            feature_columns_file_sha256_value,
        ),
        "selection_split": (threshold_payload.get("selection_split"), "val"),
        "feature_columns_hash": (threshold_payload.get("feature_columns_hash"), feature_columns_hash_value),
        "model_sha256": (threshold_payload.get("model_sha256"), model_sha256_value),
        "feature_columns_file_sha256": (
            threshold_payload.get("feature_columns_file_sha256"),
            feature_columns_file_sha256_value,
        ),
        "train_summary_sha256": (threshold_payload.get("train_summary_sha256"), train_summary_sha256_value),
        "model_file_name": (threshold_payload.get("model_file_name"), config.model_file_name),
        "feature_columns_file_name": (
            threshold_payload.get("feature_columns_file_name"),
            config.feature_columns_file_name,
        ),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"Final test provenance check failed: {name}. actual={actual!r}, expected={expected!r}"
            )

    threshold_train_summary_name = threshold_payload.get("train_summary_file_name")
    if threshold_train_summary_name is not None and threshold_train_summary_name != config.train_summary_file_name:
        raise ValueError(
            "Final test provenance check failed: train_summary_file_name. "
            f"actual={threshold_train_summary_name!r}, expected={config.train_summary_file_name!r}"
        )


# -----------------------------------------------------------------------------
# 4. final test metric 계산 및 저장
# -----------------------------------------------------------------------------
def test_xgb(config: TestConfig) -> TestResult:
    """
    고정된 모델과 validation threshold로 test split을 1회 평가

    동작 순서
    1. confirm_final_test와 sample_rows 안전 조건 확인
    2. model, feature_columns, train_summary, threshold, test parquet 존재 여부 확인
    3. final test provenance 검증
    4. test split을 X, y로 로드
    5. predict_proba()와 고정 threshold로 metric 계산
    6. metrics_test.json과 confusion_matrix_test.csv 저장
    """

    total_started = time.perf_counter()
    runtime_tracker = RuntimeTracker()

    if not config.confirm_final_test:
        raise ValueError(
            "Test evaluation is locked by default. "
            "Set confirm_final_test=True only after model/feature/threshold selection is finished."
        )
    if config.sample_rows is not None:
        raise ValueError(
            "Final test does not allow sample_rows. Use sample_rows=None for full test evaluation; "
            "use ml-00_smoke_test.ipynb for fixture or sampled checks."
        )

    memory_tracker = MemoryTracker(scope="test")
    memory_tracker.start()
    try:
        with runtime_tracker.measure("prepare_test"):
            model_path = config.output_dir / config.model_file_name
            feature_columns_path = config.output_dir / config.feature_columns_file_name
            train_summary_path = config.output_dir / config.train_summary_file_name
            threshold_path = config.output_dir / config.threshold_file_name

            if not model_path.exists():
                raise FileNotFoundError(f"model file not found: {model_path}")
            if not feature_columns_path.exists():
                raise FileNotFoundError(f"feature columns file not found: {feature_columns_path}")
            if not train_summary_path.exists():
                raise FileNotFoundError(f"train summary file not found: {train_summary_path}")
            if not threshold_path.exists():
                raise FileNotFoundError(f"threshold file not found. Run ml_00_ml_val.validate_xgb() first: {threshold_path}")
            if not config.test_path.exists():
                raise FileNotFoundError(f"test parquet not found: {config.test_path}")

            prepare_test_outputs(config)

        with runtime_tracker.measure("load_artifacts"):
            feature_columns = load_saved_feature_columns(feature_columns_path)
            features_hash = feature_columns_hash(feature_columns)
            train_summary = load_json(train_summary_path)
            threshold_payload = load_json(threshold_path)
            model_sha256 = file_sha256(model_path)
            feature_columns_file_sha256 = file_sha256(feature_columns_path)
            train_summary_sha256 = file_sha256(train_summary_path)
            threshold_sha256 = file_sha256(threshold_path)
            require_final_test_provenance(
                threshold_payload,
                train_summary,
                features_hash,
                model_sha256,
                feature_columns_file_sha256,
                train_summary_sha256,
                config,
            )
            threshold = float(threshold_payload["threshold"])
            model = joblib.load(model_path)
        memory_tracker.snapshot("after_artifact_load")

        with runtime_tracker.measure("load_test_split"):
            x_test, y_test = load_split(
                config.test_path,
                feature_columns=feature_columns,
                label_col=config.label_col,
                sample_rows=config.sample_rows,
                allow_nan=config.allow_nan,
                expected_split="test",
            )
        memory_tracker.snapshot("after_test_load")

        with runtime_tracker.measure("predict_proba"):
            probabilities = model.predict_proba(x_test)[:, 1]

        with runtime_tracker.measure("evaluate"):
            test_metrics = evaluate_at_threshold(y_test, probabilities, threshold)

        with runtime_tracker.measure("build_metadata"):
            train_run_metadata = train_summary.get("run_metadata")
            seed = train_run_metadata.get("seed") if isinstance(train_run_metadata, dict) else train_summary.get("seed")
            data_profile = make_data_profile({"test": (config.test_path, x_test, y_test)}, feature_columns)
            score_profile = make_score_profile(y_test, probabilities, threshold)
            environment = collect_environment()

        metrics_path = config.output_dir / config.metrics_file_name
        confusion_matrix_path = config.output_dir / config.confusion_matrix_file_name

        with runtime_tracker.measure("save_outputs"):
            confusion_matrix_frame(test_metrics).to_csv(confusion_matrix_path, index=False)
        memory_tracker.snapshot("end")
    finally:
        memory_profile = memory_tracker.finish()

    runtime_sec = runtime_tracker.as_dict()
    runtime_sec["total_test_xgb"] = float(time.perf_counter() - total_started)

    metrics_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "run_metadata": make_run_metadata(config.output_dir, seed=seed),
        "test_path": str(config.test_path),
        "threshold_source": str(threshold_path),
        "threshold_sha256": threshold_sha256,
        "threshold_strategy": threshold_payload.get("threshold_strategy"),
        "threshold": threshold,
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,
        "feature_count": len(feature_columns),
        "feature_columns_hash": features_hash,
        "model_sha256": model_sha256,
        "feature_columns_file_sha256": feature_columns_file_sha256,
        "train_summary_sha256": train_summary_sha256,
        "run_id": train_summary.get("run_id"),
        "label_summary": label_summary(y_test),
        "model_file_name": config.model_file_name,
        "feature_columns_file_name": config.feature_columns_file_name,
        "train_summary_file_name": config.train_summary_file_name,
        "metrics": test_metrics["summary"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "memory_mb": memory_profile["memory_mb"],
        "memory_mb_semantics": memory_profile["memory_mb_semantics"],
        "runtime_sec": runtime_sec,
        "memory_profile": memory_profile,
        "data_profile": data_profile,
        "environment": environment,
        "score_profile": score_profile,
    }

    save_json(metrics_payload, metrics_path)

    return TestResult(
        output_dir=config.output_dir,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        test_metrics=test_metrics,
    )
