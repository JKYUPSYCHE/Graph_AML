"""
최종 test split 평가 전용 모듈

전체 흐름
1. 사용자가 confirm_final_test=True를 명시했는지 확인
2. 학습 산출물과 validation threshold 산출물이 final test에 안전한 상태인지 검증
3. test parquet를 학습 때 저장한 feature 순서로 로드
4. 고정된 validation threshold로 test metric을 계산
5. metrics_test.json, confusion_matrix_test.csv, prediction_scores_test.parquet,
   feature_assoc_mixed_test.json을 저장

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
from ml_metrics import confusion_matrix_frame, evaluate_at_threshold
from ml_resource import (
    MemoryTracker,
    RuntimeTracker,
    collect_environment,
    derive_feature_assoc_file_name,
    make_data_profile,
    make_mixed_feature_association_payload,
    make_posthoc_logloss_learning_curve,
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
    encoding_manifest_path: Path | str | None = None
    export_feature_assoc: bool = True  # Backward compatible; final test always exports this artifact.
    feature_assoc_test_sample_rows: int | None = None

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
        if self.encoding_manifest_path is not None:
            object.__setattr__(
                self,
                "encoding_manifest_path",
                resolve_project_path(self.encoding_manifest_path, self.project_root),
            )
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer.")
        if self.feature_assoc_test_sample_rows is not None and self.feature_assoc_test_sample_rows <= 0:
            raise ValueError("feature_assoc_test_sample_rows must be a positive integer or None.")


@dataclass(frozen=True)
class TestResult:
    """
    final test 실행 후 생성된 산출물 경로와 metric 결과를 반환하는 객체
    """
    output_dir: Path
    metrics_path: Path
    confusion_matrix_path: Path
    feature_assoc_path: Path | None
    test_metrics: dict[str, Any]
    prediction_scores_path: Path | None = None


# -----------------------------------------------------------------------------
# 2. final test 산출물 overwrite 보호
# -----------------------------------------------------------------------------
def derive_prediction_scores_test_file_name(metrics_file_name: str) -> str:
    """metrics_test 파일명에서 같은 artifact prefix의 test score parquet 파일명을 만든다."""

    name = str(metrics_file_name)
    if name == "metrics_test.json":
        return "prediction_scores_test.parquet"
    if name.endswith("_metrics_test.json"):
        return f"{name.removesuffix('_metrics_test.json')}_prediction_scores_test.parquet"
    if name.endswith(".json"):
        return f"{name.removesuffix('.json')}_prediction_scores_test.parquet"
    return f"{name}_prediction_scores_test.parquet"


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
        config.output_dir / derive_prediction_scores_test_file_name(config.metrics_file_name),
        config.output_dir / derive_feature_assoc_file_name(config.metrics_file_name, "test"),
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError(
            "Existing final test artifacts found. Set overwrite=True only if rerunning final evaluation intentionally. "
            f"existing={existing}"
        )


def save_test_prediction_scores(
    *,
    test_path: Path,
    output_path: Path,
    y_test: pd.Series,
    probabilities,
    threshold: float,
    run_id: str,
    label_col: str,
) -> dict[str, Any]:
    """test row별 tx_id, label, score, prediction, threshold를 parquet로 저장한다."""

    threshold_value = float(threshold)
    if pd.isna(threshold_value) or not 0 <= threshold_value <= 1:
        raise ValueError(f"threshold must be between 0 and 1. threshold={threshold_value}")

    metadata_columns = ["tx_id", "split", label_col]
    metadata = read_parquet_columns(test_path, metadata_columns, sample_rows=None)

    if len(metadata) != len(y_test) or len(metadata) != len(probabilities):
        raise ValueError(
            "Prediction score row count mismatch. "
            f"metadata_rows={len(metadata)}, y_rows={len(y_test)}, score_rows={len(probabilities)}, test_path={test_path}"
        )

    if metadata["tx_id"].isna().any():
        raise ValueError(f"tx_id contains null values in test metadata. test_path={test_path}")

    split_values = metadata["split"].astype("string").str.strip().str.lower()
    if split_values.isna().any() or set(split_values.unique().tolist()) != {"test"}:
        raise ValueError(
            "Test metadata has unexpected split values. "
            f"values={sorted(str(value) for value in split_values.dropna().unique().tolist())}, test_path={test_path}"
        )

    metadata_y = pd.to_numeric(metadata[label_col], errors="raise").astype("int8").reset_index(drop=True)
    loaded_y = pd.Series(y_test, name=label_col).astype("int8").reset_index(drop=True)
    if not metadata_y.equals(loaded_y):
        raise ValueError(
            "Test label mismatch while saving prediction scores. "
            "The metadata rows may not match the model input rows."
        )

    scores = pd.Series(probabilities, name="score", dtype="float64").reset_index(drop=True)
    if scores.isna().any():
        raise ValueError("Prediction scores contain NaN values.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction scores must be between 0 and 1.")

    predictions = (scores >= threshold_value).astype("int8")
    prediction_scores = pd.DataFrame(
        {
            "tx_id": metadata["tx_id"].astype("string").reset_index(drop=True),
            "split": split_values.reset_index(drop=True),
            "label": loaded_y,
            "score": scores,
            "prediction": predictions,
            "threshold": pd.Series(threshold_value, index=scores.index, dtype="float64"),
            "run_id": str(run_id),
            "row_position": pd.Series(range(len(scores)), dtype="int64"),
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
        "prediction_column": "prediction",
        "threshold_column": "threshold",
        "label_column": "label",
        "id_column": "tx_id",
        "split": "test",
        "sample_rows": None,
        "sampled": False,
        "sha256": file_sha256(output_path),
    }

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
            "Run ml_train.train_xgb() with sample_rows=None before final test."
        )

    if bool(threshold_payload.get("sampled")) or threshold_payload.get("sample_rows") is not None:
        raise ValueError(
            "Final test is blocked because threshold.json was produced from sampled validation data. "
            "Run ml_val.validate_xgb() with sample_rows=None before final test."
        )

    train_run_id = train_summary.get("run_id")
    if not train_run_id:
        raise ValueError("train_summary.json is missing run_id. Rerun ml_train.train_xgb() with the updated module.")
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


def _prepare_test(config: TestConfig) -> tuple[Path, Path, Path, Path]:
    """Validate final-test safety flags, required files, and output protection."""

    if not config.confirm_final_test:
        raise ValueError(
            "Test evaluation is locked by default. "
            "Set confirm_final_test=True only after model/feature/threshold selection is finished."
        )
    if config.sample_rows is not None:
        raise ValueError(
            "Final test does not allow sample_rows. Use sample_rows=None for full test evaluation; "
            "use smoke_test.ipynb for fixture or sampled checks."
        )

    model_path = config.output_dir / config.model_file_name
    feature_columns_path = config.output_dir / config.feature_columns_file_name
    train_summary_path = config.output_dir / config.train_summary_file_name
    threshold_path = config.output_dir / config.threshold_file_name

    if not threshold_path.exists():
        raise FileNotFoundError(f"threshold file not found. Run ml_val.validate_xgb() first: {threshold_path}")
    if not config.test_path.exists():
        raise FileNotFoundError(f"test parquet not found: {config.test_path}")

    prepare_test_outputs(config)
    return model_path, feature_columns_path, train_summary_path, threshold_path


def _load_test_artifacts(
    *,
    config: TestConfig,
    model_path: Path,
    feature_columns_path: Path,
    train_summary_path: Path,
    threshold_path: Path,
) -> dict[str, Any]:
    """Load model and threshold artifacts, then run final-test provenance checks."""

    bundle = _load_training_artifact_bundle(
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        train_summary_path=train_summary_path,
        encoding_manifest_path=config.encoding_manifest_path,
    )
    threshold_payload = load_json(threshold_path)
    threshold_sha256 = file_sha256(threshold_path)
    require_final_test_provenance(
        threshold_payload,
        bundle.train_summary,
        bundle.feature_columns_hash,
        bundle.model_sha256,
        bundle.feature_columns_file_sha256,
        bundle.train_summary_sha256,
        config,
    )

    return {
        "model": bundle.model,
        "feature_columns": bundle.feature_columns,
        "features_hash": bundle.feature_columns_hash,
        "train_summary": bundle.train_summary,
        "model_sha256": bundle.model_sha256,
        "feature_columns_file_sha256": bundle.feature_columns_file_sha256,
        "train_summary_sha256": bundle.train_summary_sha256,
        "encoding_manifest_path": bundle.encoding_manifest_path,
        "encoding_manifest": bundle.encoding_manifest,
        "threshold_payload": threshold_payload,
        "threshold_sha256": threshold_sha256,
        "threshold": float(threshold_payload["threshold"]),
    }


def _load_test_split(
    *,
    config: TestConfig,
    feature_columns: list[str],
    encoding_manifest: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load test X/y using the training feature order."""

    return load_split(
        config.test_path,
        feature_columns=feature_columns,
        label_col=config.label_col,
        sample_rows=config.sample_rows,
        allow_nan=config.allow_nan,
        expected_split="test",
        encoding_manifest=encoding_manifest,
    )


def _run_test(
    *,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    runtime_tracker: RuntimeTracker,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Predict test scores, evaluate metrics, and build diagnostic learning curve."""

    with runtime_tracker.measure("predict_proba"):
        probabilities = model.predict_proba(x_test)[:, 1]

    with runtime_tracker.measure("evaluate"):
        test_metrics = evaluate_at_threshold(y_test, probabilities, threshold)

    with runtime_tracker.measure("build_test_learning_curve"):
        learning_curve = make_posthoc_logloss_learning_curve(
            model,
            x_test,
            y_test,
            split_name="test",
        )
    return probabilities, test_metrics, learning_curve


def _save_test_outputs(
    *,
    config: TestConfig,
    y_test: pd.Series,
    probabilities: Any,
    threshold: float,
    train_summary: dict[str, Any],
    test_metrics: dict[str, Any],
    x_test: pd.DataFrame,
    feature_columns: list[str],
    encoding_manifest: dict[str, Any] | None,
    features_hash: str,
    seed: Any,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    """Save final-test row scores, confusion matrix, and feature association payload."""

    metrics_path = config.output_dir / config.metrics_file_name
    confusion_matrix_path = config.output_dir / config.confusion_matrix_file_name
    prediction_scores_path = config.output_dir / derive_prediction_scores_test_file_name(config.metrics_file_name)
    feature_assoc_path = config.output_dir / derive_feature_assoc_file_name(config.metrics_file_name, "test")

    prediction_scores_info = save_test_prediction_scores(
        test_path=config.test_path,
        output_path=prediction_scores_path,
        y_test=y_test,
        probabilities=probabilities,
        threshold=threshold,
        run_id=str(train_summary.get("run_id")),
        label_col=config.label_col,
    )
    confusion_matrix_frame(test_metrics).to_csv(confusion_matrix_path, index=False)

    feature_assoc_payload = make_mixed_feature_association_payload(
        x_test,
        feature_columns,
        categorical_feature_columns=categorical_columns_from_manifest(encoding_manifest, feature_columns),
        split="test",
        source_path=config.test_path,
        feature_columns_hash_value=features_hash,
        run_id=train_summary.get("run_id"),
        run_metadata=make_run_metadata(
            config.output_dir,
            seed=seed,
            artifact_file_name=feature_assoc_path.name,
        ),
        max_rows=config.feature_assoc_test_sample_rows,
        sample_strategy="full" if config.feature_assoc_test_sample_rows is None else "deterministic_evenly_spaced",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_json(feature_assoc_payload, feature_assoc_path)

    return metrics_path, confusion_matrix_path, prediction_scores_path, feature_assoc_path, prediction_scores_info


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
    6. metrics_test.json, confusion_matrix_test.csv, prediction_scores_test.parquet,
       feature_assoc_mixed_test.json 저장
    """

    total_started = time.perf_counter()
    runtime_tracker = RuntimeTracker()
    feature_assoc_path: Path | None = None

    memory_tracker = MemoryTracker(scope="test")
    memory_tracker.start()
    try:
        with runtime_tracker.measure("prepare_test"):
            model_path, feature_columns_path, train_summary_path, threshold_path = _prepare_test(config)

        with runtime_tracker.measure("load_artifacts"):
            artifacts = _load_test_artifacts(
                config=config,
                model_path=model_path,
                feature_columns_path=feature_columns_path,
                train_summary_path=train_summary_path,
                threshold_path=threshold_path,
            )

        feature_columns = artifacts["feature_columns"]
        features_hash = artifacts["features_hash"]
        train_summary = artifacts["train_summary"]
        model = artifacts["model"]
        model_sha256 = artifacts["model_sha256"]
        feature_columns_file_sha256 = artifacts["feature_columns_file_sha256"]
        train_summary_sha256 = artifacts["train_summary_sha256"]
        encoding_manifest_path = artifacts["encoding_manifest_path"]
        encoding_manifest = artifacts["encoding_manifest"]
        threshold_payload = artifacts["threshold_payload"]
        threshold_sha256 = artifacts["threshold_sha256"]
        threshold = artifacts["threshold"]

        memory_tracker.snapshot("after_artifact_load")

        with runtime_tracker.measure("load_test_split"):
            x_test, y_test = _load_test_split(
                config=config,
                feature_columns=feature_columns,
                encoding_manifest=encoding_manifest,
            )
        memory_tracker.snapshot("after_test_load")

        probabilities, test_metrics, learning_curve = _run_test(
            model=model,
            x_test=x_test,
            y_test=y_test,
            threshold=threshold,
            runtime_tracker=runtime_tracker,
        )

        with runtime_tracker.measure("build_metadata"):
            train_run_metadata = train_summary.get("run_metadata")
            seed = train_run_metadata.get("seed") if isinstance(train_run_metadata, dict) else train_summary.get("seed")
            data_profile = make_data_profile({"test": (config.test_path, x_test, y_test)}, feature_columns)
            score_profile = make_score_profile(y_test, probabilities, threshold)
            environment = collect_environment()

        with runtime_tracker.measure("save_outputs"):
            (
                metrics_path,
                confusion_matrix_path,
                prediction_scores_path,
                feature_assoc_path,
                prediction_scores_info,
            ) = _save_test_outputs(
                config=config,
                y_test=y_test,
                probabilities=probabilities,
                threshold=threshold,
                train_summary=train_summary,
                test_metrics=test_metrics,
                x_test=x_test,
                feature_columns=feature_columns,
                encoding_manifest=encoding_manifest,
                features_hash=features_hash,
                seed=seed,
            )
        memory_tracker.snapshot("end")
    finally:
        memory_profile = memory_tracker.finish()

    runtime_sec = runtime_tracker.as_dict()
    runtime_sec["total_test_xgb"] = float(time.perf_counter() - total_started)

    metrics_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "run_metadata": make_run_metadata(
            config.output_dir,
            seed=seed,
            artifact_file_name=config.metrics_file_name,
        ),
        "test_path": str(config.test_path),
        "threshold_source": str(threshold_path),
        "threshold_sha256": threshold_sha256,
        "threshold_strategy": threshold_payload.get("threshold_strategy"),
        "threshold": threshold,
        "sample_rows": config.sample_rows,
        "sampled": config.sample_rows is not None,
        "feature_count": len(feature_columns),
        "feature_columns_hash": features_hash,
        "encoding_manifest_path": None if encoding_manifest_path is None else str(encoding_manifest_path),
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
        "prediction_scores": prediction_scores_info,
        "learning_curve": {
            **learning_curve,
            "diagnostic_only": True,
            "not_used_for_training_or_threshold": True,
        },
        "feature_association_artifacts": {
            "test": str(feature_assoc_path),
        },
    }

    save_json(metrics_payload, metrics_path)

    return TestResult(
        output_dir=config.output_dir,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        feature_assoc_path=feature_assoc_path,
        test_metrics=test_metrics,
        prediction_scores_path=prediction_scores_path,
    )
