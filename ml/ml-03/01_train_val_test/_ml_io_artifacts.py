"""JSON, hash, and training artifact provenance helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _ml_io_features import (
    feature_columns_hash,
    load_encoding_manifest,
    load_saved_feature_columns,
)

def file_sha256(path: str | Path) -> str:
    """파일 내용의 SHA256 hash를 계산한다."""

    # 파일 자체가 변했는지 추적하기 위한 일반 hash 함수다.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found for sha256: {path}")
    digest = hashlib.sha256()

    # 큰 파일도 한 번에 메모리에 올리지 않고 1MB 단위로 읽는다.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



@dataclass(frozen=True)
class TrainingArtifactBundle:
    """Loaded training artifacts plus their provenance fingerprints."""

    model_path: Path
    feature_columns_path: Path
    train_summary_path: Path
    model: Any
    feature_columns: list[str]
    feature_columns_hash: str
    train_summary: dict[str, Any]
    model_sha256: str
    feature_columns_file_sha256: str
    train_summary_sha256: str
    encoding_manifest_path: Path | None
    encoding_manifest: dict[str, Any] | None


def _require_artifact_file(path: Path, label: str) -> Path:
    """Require an artifact file and return its resolved Path."""

    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{label} file not found: {artifact_path}")
    return artifact_path


def _resolve_encoding_manifest_path(
    configured_path: str | Path | None,
    train_summary: dict[str, Any],
) -> Path | None:
    """Use the explicit manifest path, or the path recorded in train_summary."""

    if configured_path is not None:
        return Path(configured_path).expanduser().resolve()
    summary_path = train_summary.get("encoding_manifest_path")
    if summary_path is None:
        return None
    return Path(str(summary_path)).expanduser().resolve()


def _verify_training_artifact_value(
    *,
    name: str,
    actual: Any,
    expected: Any,
    context: str = "Training artifact provenance check failed",
) -> None:
    """Raise a consistent provenance error when a fingerprint differs."""

    if actual != expected:
        raise ValueError(f"{context}: {name} mismatch. actual={actual!r}, expected={expected!r}")


def _load_training_artifact_bundle(
    *,
    model_path: str | Path,
    feature_columns_path: str | Path,
    train_summary_path: str | Path,
    current_val_path: str | Path | None = None,
    encoding_manifest_path: str | Path | None = None,
) -> TrainingArtifactBundle:
    """Load training artifacts and verify hashes recorded in train_summary.json.

    Validation and final-test code both need the same model/feature/train_summary
    provenance checks. Test-specific threshold checks stay in ml_test.py.
    """

    import joblib

    resolved_model_path = _require_artifact_file(Path(model_path), "model")
    resolved_feature_columns_path = _require_artifact_file(Path(feature_columns_path), "feature columns")
    resolved_train_summary_path = _require_artifact_file(Path(train_summary_path), "train summary")

    feature_columns = load_saved_feature_columns(resolved_feature_columns_path)
    features_hash = feature_columns_hash(feature_columns)
    train_summary = load_json(resolved_train_summary_path)

    train_run_id = train_summary.get("run_id")
    if not train_run_id:
        raise ValueError("train_summary.json is missing run_id. Rerun ml_train.train_xgb() with the updated module.")

    if current_val_path is not None:
        expected_val_path = train_summary.get("val_path")
        if not expected_val_path:
            raise ValueError("train_summary.json is missing val_path. Rerun ml_train.train_xgb() with the updated module.")
        if Path(expected_val_path).expanduser().resolve() != Path(current_val_path).expanduser().resolve():
            raise ValueError(
                "Validation artifact provenance check failed: val_path mismatch. "
                f"train_summary={expected_val_path!r}, current_val_path={str(current_val_path)!r}"
            )

    _verify_training_artifact_value(
        name="feature_columns_hash",
        actual=train_summary.get("feature_columns_hash"),
        expected=features_hash,
    )

    model_sha256_value = file_sha256(resolved_model_path)
    _verify_training_artifact_value(
        name="model_sha256",
        actual=train_summary.get("model_sha256"),
        expected=model_sha256_value,
    )

    feature_columns_file_sha256_value = file_sha256(resolved_feature_columns_path)
    _verify_training_artifact_value(
        name="feature_columns_file_sha256",
        actual=train_summary.get("feature_columns_file_sha256"),
        expected=feature_columns_file_sha256_value,
    )

    resolved_encoding_manifest_path = _resolve_encoding_manifest_path(encoding_manifest_path, train_summary)
    encoding_manifest = load_encoding_manifest(resolved_encoding_manifest_path)
    if resolved_encoding_manifest_path is not None and train_summary.get("encoding_manifest_sha256") is not None:
        encoding_manifest_sha256 = file_sha256(resolved_encoding_manifest_path)
        _verify_training_artifact_value(
            name="encoding_manifest_sha256",
            actual=train_summary.get("encoding_manifest_sha256"),
            expected=encoding_manifest_sha256,
        )

    train_summary_sha256_value = file_sha256(resolved_train_summary_path)
    model = joblib.load(resolved_model_path)

    return TrainingArtifactBundle(
        model_path=resolved_model_path,
        feature_columns_path=resolved_feature_columns_path,
        train_summary_path=resolved_train_summary_path,
        model=model,
        feature_columns=feature_columns,
        feature_columns_hash=features_hash,
        train_summary=train_summary,
        model_sha256=model_sha256_value,
        feature_columns_file_sha256=feature_columns_file_sha256_value,
        train_summary_sha256=train_summary_sha256_value,
        encoding_manifest_path=resolved_encoding_manifest_path,
        encoding_manifest=encoding_manifest,
    )

def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """
    dict payload를 UTF-8 JSON 파일로 저장
    사용 예
    - 실험 설정 저장
    - label_summary 결과 저장
    - 모델 평가 metric 저장
    """
    # 실험 설정, label 분포, 평가 metric 등 작은 메타데이터를 저장하는 공통 함수다.
    path = Path(path)

    # 결과 디렉터리가 없으면 생성한다.
    path.parent.mkdir(parents=True, exist_ok=True)

    # ensure_ascii=False로 한글 key/value도 읽기 쉬운 형태로 저장한다.
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """
    UTF-8 JSON 파일을 읽어 Python 객체로 반환합니다.
    주의
    - 타입 힌트는 dict[str, Any]이지만, JSON 파일 내용이 list이면 실제 반환값도 list가 됨
    - 반드시 dict만 허용해야 하는 상황이라면 isinstance(result, dict) 검사를 추가하는 것이 안전
    """
    # JSON 파일을 읽어 Python 객체로 역직렬화한다.
    # 확인 필요: 이 함수는 dict만 검증하지 않으므로, 호출부에서 타입 검증이 필요한지 확인해야 한다.
    return json.loads(Path(path).read_text(encoding="utf-8"))
