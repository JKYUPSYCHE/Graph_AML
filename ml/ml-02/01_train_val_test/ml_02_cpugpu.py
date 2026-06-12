"""CPU/GPU acceleration policy for ML-02 XGBoost runs.

This module keeps accelerator-specific XGBoost parameters in one place so the
training and tuning flows only pass through the requested mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Any

EXPECTED_XGBOOST_VERSION = "1.7.6"
ALLOWED_ACCELERATORS = {"auto", "cpu", "cuda"}


class XGBAccelerationError(RuntimeError):
    """Raised when the requested XGBoost accelerator cannot be used."""


@dataclass(frozen=True)
class XGBAccelerationConfig:
    """Resolved XGBoost acceleration settings."""

    requested_accelerator: str
    resolved_accelerator: str
    xgboost_version: str
    xgb_params: dict[str, Any]
    fallback_reason: str | None = None

    def as_summary(self) -> dict[str, Any]:
        """Return JSON-serializable metadata for train_summary.json."""

        return {
            "requested_accelerator": self.requested_accelerator,
            "resolved_accelerator": self.resolved_accelerator,
            "xgboost_version": self.xgboost_version,
            "xgboost_tree_method": self.xgb_params.get("tree_method"),
            "xgboost_predictor": self.xgb_params.get("predictor"),
            "gpu_fallback_reason": self.fallback_reason,
        }


def normalize_accelerator(requested: str) -> str:
    """Validate and normalize the user-facing accelerator mode."""

    value = str(requested).strip().lower()
    if value not in ALLOWED_ACCELERATORS:
        raise ValueError(
            "Unsupported XGBoost accelerator. "
            f"accelerator={requested!r}, allowed={sorted(ALLOWED_ACCELERATORS)}"
        )
    return value


def get_xgboost_version() -> str:
    """Return the installed XGBoost package version."""

    try:
        return metadata.version("xgboost")
    except metadata.PackageNotFoundError as exc:
        raise XGBAccelerationError(
            "xgboost is required for ML-02 training. Install requirements.txt first."
        ) from exc


def require_expected_xgboost_version(version: str) -> None:
    """Keep accelerator behavior pinned to the project requirements contract."""

    if version != EXPECTED_XGBOOST_VERSION:
        raise XGBAccelerationError(
            "Unsupported xgboost version for ML-02 accelerator policy. "
            f"installed={version!r}, expected={EXPECTED_XGBOOST_VERSION!r}. "
            "Install the project requirements.txt to avoid CPU/GPU parameter drift."
        )


@lru_cache(maxsize=1)
def _gpu_smoke_failure() -> str | None:
    """Return None when a tiny XGBoost GPU fit succeeds, otherwise the failure."""

    try:
        import numpy as np
        from xgboost import XGBClassifier

        x = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype="float32",
        )
        y = np.array([0, 0, 1, 1], dtype="int8")
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            random_state=0,
            n_jobs=1,
        )
        model.fit(x, y, verbose=False)
    except Exception as exc:  # noqa: BLE001 - the caller records the concrete smoke failure.
        return f"{type(exc).__name__}: {exc}"
    return None


def resolve_xgb_acceleration(requested: str = "auto") -> XGBAccelerationConfig:
    """Resolve XGBoost 1.7.6 CPU/GPU parameters for the requested mode."""

    accelerator = normalize_accelerator(requested)
    version = get_xgboost_version()
    require_expected_xgboost_version(version)

    if accelerator == "cpu":
        return XGBAccelerationConfig(
            requested_accelerator=accelerator,
            resolved_accelerator="cpu",
            xgboost_version=version,
            xgb_params={"tree_method": "hist"},
        )

    smoke_failure = _gpu_smoke_failure()
    if smoke_failure is None:
        return XGBAccelerationConfig(
            requested_accelerator=accelerator,
            resolved_accelerator="cuda",
            xgboost_version=version,
            xgb_params={"tree_method": "gpu_hist", "predictor": "gpu_predictor"},
        )

    if accelerator == "cuda":
        raise XGBAccelerationError(
            "XGBoost GPU acceleration was requested but the GPU smoke test failed. "
            f"reason={smoke_failure}"
        )

    return XGBAccelerationConfig(
        requested_accelerator=accelerator,
        resolved_accelerator="cpu",
        xgboost_version=version,
        xgb_params={"tree_method": "hist"},
        fallback_reason=f"XGBoost GPU smoke test failed: {smoke_failure}",
    )
