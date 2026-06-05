"""ML-00 실행 리소스와 진단 정보를 수집하는 저비용 helper.

이 모듈은 학습/검증/평가 로직을 바꾸지 않고, 이미 메모리에 올라온
DataFrame, prediction score, XGBoost booster에서 기록용 metadata만 만든다.
"""

from __future__ import annotations

import importlib.metadata
import math
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
NUMERIC_ASSOC_METHOD = "pearson"
CATEGORICAL_ASSOC_METHOD = "cramers_v"
MIXED_ASSOC_METHOD = "correlation_ratio_eta"


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


ML_ARTIFACT_SUFFIXES = (
    "_prediction_scores_val.parquet",
    "_prediction_scores_test.parquet",
    "_confusion_matrix_val.csv",
    "_confusion_matrix_test.csv",
    "_feature_assoc_mixed_train.json",
    "_feature_assoc_mixed_val.json",
    "_feature_assoc_mixed_test.json",
    "_scores_train_summary.json",
    "_feature_importance.csv",
    "_feature_columns.json",
    "_metrics_train.json",
    "_metrics_val.json",
    "_metrics_test.json",
    "_train_summary.json",
    "_threshold.json",
    "_model.pkl",
)


def make_run_metadata(
    output_dir: Path | str,
    seed: int | None = None,
    artifact_file_name: str | None = None,
) -> dict[str, Any]:
    """output_dir와 prefix 파일명에서 실험 식별자를 추론해 저장한다."""

    resolved = Path(output_dir).expanduser().resolve()
    metadata: dict[str, Any] = {
        "export_experiment_id": None,
        "input_run_id": None,
        "model_run_id": None,
        "run_kind": None,
        "output_dir": str(resolved),
        "artifact_file_name": artifact_file_name,
        "artifact_prefix": None,
        "artifact_name": None,
        "seed": None if seed is None else int(seed),
    }

    if artifact_file_name:
        name = Path(artifact_file_name).name
        parts_from_name = name.split("__", 2)
        if len(parts_from_name) == 3:
            experiment_id, run_id, model_and_artifact = parts_from_name
            model_run_id = None
            artifact_name = None

            for suffix in ML_ARTIFACT_SUFFIXES:
                if model_and_artifact.endswith(suffix):
                    model_run_id = model_and_artifact[: -len(suffix)]
                    artifact_name = suffix.removeprefix("_")
                    break

            if model_run_id is None and "_" in model_and_artifact:
                model_run_id, artifact_name = model_and_artifact.rsplit("_", 1)

            if model_run_id and artifact_name:
                metadata["export_experiment_id"] = experiment_id
                metadata["input_run_id"] = run_id
                metadata["model_run_id"] = model_run_id
                metadata["artifact_prefix"] = f"{experiment_id}__{run_id}__{model_run_id}"
                metadata["artifact_name"] = artifact_name

    parts = resolved.parts
    if "ml_outputs" not in parts:
        return metadata

    index = len(parts) - 1 - list(reversed(parts)).index("ml_outputs")
    tail = parts[index + 1 :]
    if len(tail) >= 4:
        metadata["export_experiment_id"] = metadata["export_experiment_id"] or tail[0]
        metadata["input_run_id"] = metadata["input_run_id"] or tail[1]
        metadata["run_kind"] = tail[2]
        metadata["model_run_id"] = metadata["model_run_id"] or tail[3]
    elif len(tail) == 3:
        metadata["export_experiment_id"] = metadata["export_experiment_id"] or tail[0]
        metadata["input_run_id"] = metadata["input_run_id"] or tail[1]
        metadata["run_kind"] = tail[2]
    elif len(tail) == 2:
        metadata["export_experiment_id"] = metadata["export_experiment_id"] or tail[0]
        metadata["run_kind"] = tail[1]
    elif len(tail) == 1:
        if metadata["input_run_id"] and metadata["input_run_id"] != tail[0]:
            raise ValueError(
                "artifact file RUN_ID does not match flat ml_outputs directory. "
                f"artifact_run_id={metadata['input_run_id']!r}, output_dir_run_id={tail[0]!r}"
            )
        metadata["input_run_id"] = metadata["input_run_id"] or tail[0]
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


def make_learning_curve_from_evals_result(
    model: Any,
    eval_set_aliases: Mapping[str, str],
    *,
    metric_name: str = "logloss",
) -> dict[str, Any]:
    """XGBoost evals_result에서 split별 logloss curve를 추출한다."""

    evals_result = model.evals_result()
    curves: dict[str, dict[str, list[float]]] = {}
    source_eval_sets: dict[str, str] = {}
    round_count: int | None = None

    for source_name, split_name in eval_set_aliases.items():
        split_metrics = evals_result.get(source_name)
        if split_metrics is None:
            raise ValueError(f"XGBoost evals_result is missing eval set: {source_name!r}")
        if metric_name not in split_metrics:
            raise ValueError(
                "XGBoost evals_result is missing required metric. "
                f"eval_set={source_name!r}, metric={metric_name!r}, available={sorted(split_metrics)}"
            )
        values = [float(value) for value in split_metrics[metric_name]]
        curves[str(split_name)] = {metric_name: values}
        source_eval_sets[str(split_name)] = str(source_name)
        round_count = len(values) if round_count is None else min(round_count, len(values))

    return {
        "curve_source": "xgboost_evals_result",
        "loss_name": metric_name,
        "metrics": [metric_name],
        "eval_set_aliases": {str(key): str(value) for key, value in eval_set_aliases.items()},
        "source_eval_sets": source_eval_sets,
        "curves": curves,
        "round_count": int(round_count or 0),
    }


def binary_logloss(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    """이진분류 logloss를 sklearn 없이 계산한다."""

    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(probabilities, dtype="float64")
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError(
            "binary_logloss requires 1-dimensional arrays with equal length. "
            f"y_shape={y.shape}, probability_shape={p.shape}"
        )
    if len(y) == 0:
        raise ValueError("binary_logloss requires at least one row.")
    eps = np.finfo("float64").eps
    clipped = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean())


def make_posthoc_logloss_learning_curve(
    model: Any,
    x_frame: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    *,
    split_name: str,
) -> dict[str, Any]:
    """학습 완료 모델로 split별 post-hoc logloss curve를 계산한다."""

    booster = model.get_booster()
    round_count = int(booster.num_boosted_rounds())
    values: list[float] = []
    for round_index in range(1, round_count + 1):
        probabilities = model.predict_proba(x_frame, iteration_range=(0, round_index))[:, 1]
        values.append(binary_logloss(y_true, probabilities))

    return {
        "curve_source": "xgboost_posthoc_predict_proba",
        "loss_name": "logloss",
        "metrics": ["logloss"],
        "curves": {
            str(split_name): {
                "logloss": values,
            }
        },
        "round_count": round_count,
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


def derive_feature_assoc_file_name(artifact_file_name: str, split: str) -> str:
    """기존 stage artifact 파일명과 같은 prefix로 split별 association JSON 파일명을 만든다."""

    name = Path(artifact_file_name).name
    suffixes = (
        "_train_summary.json",
        "_metrics_val.json",
        "_metrics_test.json",
    )
    for suffix in suffixes:
        if name.endswith(suffix):
            return f"{name.removesuffix(suffix)}_feature_assoc_mixed_{split}.json"
    if name in {"train_summary.json", "metrics_val.json", "metrics_test.json"}:
        return f"feature_assoc_mixed_{split}.json"
    return f"{Path(name).stem}_feature_assoc_mixed_{split}.json"


def infer_mixed_assoc_feature_types(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_feature_columns: Sequence[str] | None = None,
) -> dict[str, str]:
    """association 계산에 사용할 feature type을 numeric/categorical로 판별한다."""

    categorical_names = set(categorical_feature_columns or [])
    feature_types: dict[str, str] = {}
    for feature in feature_columns:
        if feature in categorical_names or isinstance(frame[feature].dtype, pd.CategoricalDtype):
            feature_types[str(feature)] = "categorical"
        elif pd.api.types.is_numeric_dtype(frame[feature]):
            feature_types[str(feature)] = "numeric"
        else:
            feature_types[str(feature)] = "categorical"
    return feature_types


def deterministic_evenly_spaced_sample(frame: pd.DataFrame, max_rows: int | None) -> tuple[pd.DataFrame, bool]:
    """DataFrame 전체 구간에서 균등 간격으로 최대 max_rows개 row를 선택한다."""

    if max_rows is None or len(frame) <= max_rows:
        return frame, False
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive or None. max_rows={max_rows!r}")

    positions = np.linspace(0, len(frame) - 1, num=max_rows, dtype=np.int64)
    sampled = frame.iloc[positions].copy()
    return sampled, True


def _to_jsonable_float(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _pearson_pair(left: pd.Series, right: pd.Series) -> float | None:
    value = left.astype("float64").corr(right.astype("float64"), method="pearson")
    return _to_jsonable_float(value)


def _cramers_v(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.DataFrame({"left": left.astype("string"), "right": right.astype("string")}).dropna()
    if frame.empty:
        return None
    total = float(len(frame))
    left_counts = frame["left"].value_counts()
    right_counts = frame["right"].value_counts()
    if len(left_counts) < 2 or len(right_counts) < 2:
        return 0.0

    observed = frame.groupby(["left", "right"], observed=True).size().rename("observed").reset_index()
    observed["left_total"] = observed["left"].map(left_counts).astype("float64")
    observed["right_total"] = observed["right"].map(right_counts).astype("float64")
    # Sparse chi-square formula: sum((obs-exp)^2/exp) == n*sum(obs^2/(row_total*col_total)) - n.
    chi2 = float(
        total
        * ((observed["observed"].astype("float64") ** 2) / (observed["left_total"] * observed["right_total"])).sum()
        - total
    )
    phi2 = chi2 / total
    rows, cols = len(left_counts), len(right_counts)
    denom = min(cols - 1, rows - 1)
    if denom <= 0:
        return 0.0
    return _to_jsonable_float(min(1.0, math.sqrt(max(0.0, phi2 / denom))))


def _correlation_ratio_eta(categories: pd.Series, measurements: pd.Series) -> float | None:
    values = pd.to_numeric(measurements, errors="coerce")
    valid = values.notna() & categories.notna()
    if not bool(valid.any()):
        return None

    grouped = pd.DataFrame({"category": categories[valid], "value": values[valid]}).groupby("category", observed=True)["value"]
    counts = grouped.count().to_numpy(dtype="float64")
    means = grouped.mean().to_numpy(dtype="float64")
    if counts.size <= 1:
        return 0.0

    all_values = values[valid].to_numpy(dtype="float64")
    grand_mean = float(all_values.mean())
    between = float((counts * (means - grand_mean) ** 2).sum())
    total = float(((all_values - grand_mean) ** 2).sum())
    if total <= 0:
        return 0.0
    return _to_jsonable_float(min(1.0, math.sqrt(max(0.0, between / total))))


def _mixed_association_pair(
    frame: pd.DataFrame,
    left_feature: str,
    right_feature: str,
    feature_types: Mapping[str, str],
) -> tuple[float | None, str]:
    left_type = feature_types[left_feature]
    right_type = feature_types[right_feature]

    if left_feature == right_feature:
        return 1.0, "self"
    if left_type == "numeric" and right_type == "numeric":
        return _pearson_pair(frame[left_feature], frame[right_feature]), NUMERIC_ASSOC_METHOD
    if left_type == "categorical" and right_type == "categorical":
        return _cramers_v(frame[left_feature], frame[right_feature]), CATEGORICAL_ASSOC_METHOD
    if left_type == "categorical":
        return _correlation_ratio_eta(frame[left_feature], frame[right_feature]), MIXED_ASSOC_METHOD
    return _correlation_ratio_eta(frame[right_feature], frame[left_feature]), MIXED_ASSOC_METHOD


def make_mixed_feature_association_payload(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    categorical_feature_columns: Sequence[str] | None = None,
    split: str,
    source_path: Path | str,
    feature_columns_hash_value: str,
    run_id: str | None,
    run_metadata: Mapping[str, Any],
    max_rows: int | None,
    sample_strategy: str,
    created_at: str,
) -> dict[str, Any]:
    """Streamlit heatmap용 split 단위 mixed-type association JSON payload를 만든다."""

    missing = [feature for feature in feature_columns if feature not in frame.columns]
    if missing:
        raise ValueError(
            "association frame is missing feature columns. "
            f"split={split!r}, missing={missing[:30]}, missing_count={len(missing)}"
        )

    row_count_total = int(len(frame))
    sampled_frame, sampled = deterministic_evenly_spaced_sample(frame[list(feature_columns)], max_rows)
    row_count_used = int(len(sampled_frame))
    feature_types = infer_mixed_assoc_feature_types(sampled_frame, feature_columns, categorical_feature_columns)

    feature_names = [str(feature) for feature in feature_columns]
    feature_count = len(feature_names)
    matrix: list[list[float | None]] = [[None for _ in feature_names] for _ in feature_names]
    metric_matrix: list[list[str]] = [["" for _ in feature_names] for _ in feature_names]
    for left_index, left in enumerate(feature_names):
        for right_index in range(left_index, feature_count):
            right = feature_names[right_index]
            value, method = _mixed_association_pair(sampled_frame, left, right, feature_types)
            matrix[left_index][right_index] = value
            matrix[right_index][left_index] = value
            metric_matrix[left_index][right_index] = method
            metric_matrix[right_index][left_index] = method

    return {
        "artifact_type": "feature_association_mixed",
        "schema_version": 1,
        "created_at": created_at,
        "split": str(split),
        "run_id": run_id,
        "run_metadata": dict(run_metadata),
        "source_path": str(source_path),
        "feature_columns_hash": str(feature_columns_hash_value),
        "association_methods": {
            "numeric_numeric": NUMERIC_ASSOC_METHOD,
            "categorical_categorical": CATEGORICAL_ASSOC_METHOD,
            "numeric_categorical": MIXED_ASSOC_METHOD,
        },
        "sample_policy": {
            "max_rows": None if max_rows is None else int(max_rows),
            "strategy": str(sample_strategy),
        },
        "row_count_total": row_count_total,
        "row_count_used": row_count_used,
        "sampled": bool(sampled),
        "features": [
            {"name": str(feature), "feature_type": feature_types[str(feature)]}
            for feature in feature_columns
        ],
        "association": {
            "features": feature_names,
            "matrix": matrix,
            "metric_matrix": metric_matrix,
        },
    }
