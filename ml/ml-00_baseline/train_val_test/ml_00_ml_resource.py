"""ML-00 실행 리소스와 진단 정보를 수집하는 저비용 helper.

이 모듈은 학습/검증/평가 로직을 바꾸지 않고, 이미 메모리에 올라온
DataFrame, prediction score, XGBoost booster에서 기록용 metadata만 만든다.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import resource
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:  # pragma: no cover - psutil은 requirements에 있지만 안전하게 fallback한다.
    psutil = None


THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
PACKAGE_VERSION_KEYS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "xgboost": "xgboost",
    "sklearn": "scikit-learn",
    "pyarrow": "pyarrow",
    "psutil": "psutil",
}
IMPORTANCE_TYPES = ("weight", "gain", "cover", "total_gain", "total_cover")
SCORE_QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


class RuntimeTracker:
    """여러 stage의 누적 실행 시간을 초 단위로 기록한다."""

    def __init__(self) -> None:
        self._runtime_sec: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._runtime_sec[name] = self._runtime_sec.get(name, 0.0) + elapsed

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self._runtime_sec.items()}


class MemoryTracker:
    """GNN 코드의 memory_mb 관례에 맞춰 run 구간 CPU RSS peak를 기록한다."""

    def __init__(self, scope: str, sample_interval_sec: float = 0.1) -> None:
        if sample_interval_sec <= 0:
            raise ValueError("sample_interval_sec must be positive.")
        self.scope = str(scope)
        self.sample_interval_sec = float(sample_interval_sec)
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._sample_count = 0
        self._rss_mb_start: float | None = None
        self._rss_mb_end: float | None = None
        self._rss_mb_peak_sampled: float | None = None
        self._ru_maxrss_mb_start: float | None = None
        self._ru_maxrss_mb_end: float | None = None
        self._snapshots: dict[str, dict[str, float | None]] = {}

    def _current_rss_mb(self) -> float | None:
        if self._process is None:
            return None
        return float(self._process.memory_info().rss / (1024 * 1024))

    def _record_sample(self) -> None:
        rss_mb = self._current_rss_mb()
        if rss_mb is None:
            return
        with self._lock:
            self._sample_count += 1
            if self._rss_mb_peak_sampled is None or rss_mb > self._rss_mb_peak_sampled:
                self._rss_mb_peak_sampled = rss_mb

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_sec):
            self._record_sample()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ru_maxrss_mb_start = current_ru_maxrss_mb()
        self._rss_mb_start = self._current_rss_mb()
        self._record_sample()
        self.snapshot("start")
        if self._process is not None:
            self._thread = threading.Thread(target=self._sample_loop, name="ml00-memory-sampler", daemon=True)
            self._thread.start()

    def snapshot(self, name: str) -> None:
        """현재 RSS와 ru_maxrss를 같은 stage 이름으로 기록한다."""

        self._record_sample()
        self._snapshots[str(name)] = {
            "rss_mb": self._current_rss_mb(),
            "ru_maxrss_mb": current_ru_maxrss_mb(),
        }

    def finish(self) -> dict[str, Any]:
        if self._running:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=max(1.0, self.sample_interval_sec * 2))
            self._record_sample()
            self._rss_mb_end = self._current_rss_mb()
            self._ru_maxrss_mb_end = current_ru_maxrss_mb()
            self._running = False

        peak_rss = self._rss_mb_peak_sampled
        if peak_rss is not None:
            memory_mb = peak_rss
            memory_mb_semantics = "cpu_rss_peak_sampled_mb"
        else:
            memory_mb = self._ru_maxrss_mb_end if self._ru_maxrss_mb_end is not None else current_ru_maxrss_mb()
            memory_mb_semantics = "process_peak_rss_mb"

        profile: dict[str, Any] = {
            "memory_mb": None if memory_mb is None else float(memory_mb),
            "memory_mb_semantics": memory_mb_semantics,
            "scope": self.scope,
            "device": "cpu",
            "sample_interval_sec": self.sample_interval_sec,
            "sample_count": int(self._sample_count),
            "cpu": {
                "rss_mb_start": self._rss_mb_start,
                "rss_mb_end": self._rss_mb_end,
                "rss_mb_peak_sampled": peak_rss,
                "ru_maxrss_mb_start": self._ru_maxrss_mb_start,
                "ru_maxrss_mb_end": self._ru_maxrss_mb_end,
            },
            "cuda": {
                "available": False,
                "allocated_peak_mb": None,
                "reserved_peak_mb": None,
            },
            "snapshots": self._snapshots,
        }

        # 기존 flat key 사용처를 위해 top-level snapshot도 유지한다.
        for name, snapshot in self._snapshots.items():
            profile[f"rss_mb_{name}"] = snapshot.get("rss_mb")
            profile[f"ru_maxrss_mb_{name}"] = snapshot.get("ru_maxrss_mb")
        return profile


def current_ru_maxrss_mb() -> float:
    """현재 프로세스의 peak RSS를 MB 단위로 반환한다."""

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(max_rss / (1024 * 1024))
    return float(max_rss / 1024)


def capture_memory_snapshot(memory_profile: dict[str, float], name: str) -> None:
    """memory_profile에 ru_maxrss 기반 snapshot을 추가한다."""

    memory_profile[f"ru_maxrss_mb_{name}"] = current_ru_maxrss_mb()


def make_run_metadata(output_dir: Path | str, seed: int | None = None) -> dict[str, Any]:
    """output_dir 구조에서 실험 식별자를 추론해 저장한다."""

    resolved = Path(output_dir).expanduser().resolve()
    metadata: dict[str, Any] = {
        "export_experiment_id": None,
        "input_run_id": None,
        "model_run_id": None,
        "run_kind": None,
        "output_dir": str(resolved),
        "seed": None if seed is None else int(seed),
    }

    parts = resolved.parts
    if "ml_outputs" not in parts:
        return metadata

    index = len(parts) - 1 - list(reversed(parts)).index("ml_outputs")
    tail = parts[index + 1 :]
    if len(tail) >= 4:
        metadata["export_experiment_id"] = tail[0]
        metadata["input_run_id"] = tail[1]
        metadata["run_kind"] = tail[2]
        metadata["model_run_id"] = tail[3]
    elif len(tail) == 3:
        metadata["export_experiment_id"] = tail[0]
        metadata["input_run_id"] = tail[1]
        metadata["run_kind"] = tail[2]
    elif len(tail) == 2:
        metadata["export_experiment_id"] = tail[0]
        metadata["run_kind"] = tail[1]
    elif len(tail) == 1:
        metadata["export_experiment_id"] = tail[0]
    return metadata


def collect_environment() -> dict[str, Any]:
    """재현성 확인에 필요한 실행 환경 정보를 수집한다."""

    package_versions: dict[str, str | None] = {}
    for output_name, distribution_name in PACKAGE_VERSION_KEYS.items():
        try:
            package_versions[output_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[output_name] = None

    return {
        "platform": sys.platform,
        "platform_detail": platform.platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "package_versions": package_versions,
        "thread_env": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
    }


def dataframe_memory_mb(frame: pd.DataFrame) -> float:
    """DataFrame의 pandas 기준 메모리 사용량을 MB 단위로 반환한다."""

    return float(frame.memory_usage(deep=True).sum() / (1024 * 1024))


def dtype_counts(frame: pd.DataFrame) -> dict[str, int]:
    """DataFrame dtype별 column 수를 JSON 저장 가능한 dict로 반환한다."""

    counts = frame.dtypes.astype(str).value_counts().sort_index().to_dict()
    return {str(dtype): int(count) for dtype, count in counts.items()}


def file_size_mb(path: Path | str) -> float:
    """파일 크기를 MB 단위로 반환한다."""

    return float(Path(path).stat().st_size / (1024 * 1024))


def make_data_profile(
    split_frames: Mapping[str, tuple[Path | str, pd.DataFrame, pd.Series]],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    """로드된 split별 파일/row/feature 메모리 정보를 만든다."""

    profile: dict[str, Any] = {
        "feature_count": int(len(feature_columns)),
    }
    first_dtype_counts: dict[str, int] | None = None

    for split_name, (path, x_frame, y_series) in split_frames.items():
        split = str(split_name)
        split_dtype_counts = dtype_counts(x_frame)
        if first_dtype_counts is None:
            first_dtype_counts = split_dtype_counts
        profile[f"{split}_file_size_mb"] = file_size_mb(path)
        profile[f"{split}_rows"] = int(len(y_series))
        profile[f"x_{split}_memory_mb"] = dataframe_memory_mb(x_frame)
        profile[f"x_{split}_dtype_counts"] = split_dtype_counts

    profile["feature_dtype_counts"] = first_dtype_counts or {}
    return profile


def _jsonable(value: Any) -> Any:
    """numpy/pandas scalar를 JSON 저장 가능한 기본 타입으로 변환한다."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if not isinstance(value, (list, tuple, Mapping, np.ndarray)) and pd.isna(value):
        return None
    return value


def make_xgboost_diagnostics(model: Any) -> dict[str, Any]:
    """학습된 XGBoost 모델에서 추가 학습 없이 진단 정보를 추출한다."""

    params = model.get_params() if hasattr(model, "get_params") else {}
    booster = model.get_booster()
    return {
        "eval_metric": params.get("eval_metric"),
        "evals_result": _jsonable(model.evals_result()),
        "num_boosted_rounds": int(booster.num_boosted_rounds()),
    }


def save_feature_importance(model: Any, feature_columns: Sequence[str], path: Path | str) -> pd.DataFrame:
    """XGBoost booster importance를 feature_importance.csv로 저장한다."""

    booster = model.get_booster()
    scores_by_type = {
        importance_type: booster.get_score(importance_type=importance_type)
        for importance_type in IMPORTANCE_TYPES
    }

    rows: list[dict[str, Any]] = []
    for index, feature_name in enumerate(feature_columns):
        row: dict[str, Any] = {
            "feature": str(feature_name),
            "feature_index": int(index),
        }
        fallback_name = f"f{index}"
        for importance_type, scores in scores_by_type.items():
            row[f"importance_{importance_type}"] = float(
                scores.get(str(feature_name), scores.get(fallback_name, 0.0))
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["rank_by_gain"] = frame["importance_gain"].rank(method="min", ascending=False).astype("int64")
        frame = frame.sort_values(["rank_by_gain", "feature_index"], ascending=[True, True])

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return frame


def _quantile_profile(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    quantiles = np.quantile(values, SCORE_QUANTILES)
    return {f"q{int(round(q * 100)):02d}": float(value) for q, value in zip(SCORE_QUANTILES, quantiles)}


def make_score_profile(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """validation/test probability score 분포와 예측 양성 비율을 요약한다."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype="float64")
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError(
            "score_profile requires 1-dimensional arrays with equal length. "
            f"y_shape={y.shape}, probability_shape={p.shape}"
        )

    predicted_positive_count: int | None = None
    predicted_positive_rate: float | None = None
    if threshold is not None:
        predictions = p >= float(threshold)
        predicted_positive_count = int(predictions.sum())
        predicted_positive_rate = float(predicted_positive_count / len(p)) if len(p) else None

    return {
        "probability_quantiles": _quantile_profile(p),
        "positive_score_quantiles": _quantile_profile(p[y == 1]),
        "negative_score_quantiles": _quantile_profile(p[y == 0]),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": predicted_positive_rate,
    }
