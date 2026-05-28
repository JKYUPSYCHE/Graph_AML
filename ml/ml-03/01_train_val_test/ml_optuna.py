"""ML common Optuna tuning pipeline for XGBoost.

The pipeline runs smoke tuning, full tuning, and final train/validation with the
best full-study parameters. It intentionally never reads or evaluates test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from ml_io import InputPaths, preflight_ml_inputs, resolve_project_path, save_json
from ml_optuna_spaces import suggest_xgb_params
from ml_train import XGBTrainConfig, XGBTrainResult, train_xgb
from ml_val import ValidationConfig, ValidationResult, validate_xgb

ALLOWED_SELECTION_METRICS = {"average_precision", "f1", "recall", "precision"}
SMOKE_SAMPLE_ROWS = 100_000
REQUIRED_XGB_PARAM_KEYS = {
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "reg_alpha",
    "gamma",
}


class MissingOptionalDependencyError(RuntimeError):
    """Raised when Optuna is not installed in the current environment."""


@dataclass(frozen=True)
class OptunaPipelineConfig:
    train_path: Union[Path, str]
    val_path: Union[Path, str]
    feature_columns_path: Union[Path, str]
    tuning_output_dir: Union[Path, str]
    final_output_dir: Union[Path, str]
    project_root: Optional[Union[Path, str]] = None
    encoding_manifest_path: Optional[Union[Path, str]] = None
    label_col: str = "label"
    sample_rows: Optional[int] = None
    allow_nan: bool = False
    export_feature_assoc: bool = False
    export_prediction_scores: bool = False
    model_file_name: str = "model.pkl"
    feature_columns_file_name: str = "feature_columns.json"
    train_summary_file_name: str = "train_summary.json"
    scores_train_summary_file_name: str = "scores_train_summary.json"
    feature_importance_file_name: str = "feature_importance.csv"
    threshold_file_name: str = "threshold.json"
    metrics_val_file_name: str = "metrics_val.json"
    confusion_matrix_val_file_name: str = "confusion_matrix_val.csv"
    smoke_trials: int = 3
    full_trials: int = 50
    selection_metric: str = "average_precision"
    study_name_prefix: str = "ml"
    seed: int = 42
    n_jobs: int = -1
    accelerator: str = "auto"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_path", resolve_project_path(self.train_path, self.project_root))
        object.__setattr__(self, "val_path", resolve_project_path(self.val_path, self.project_root))
        object.__setattr__(
            self,
            "feature_columns_path",
            resolve_project_path(self.feature_columns_path, self.project_root),
        )
        object.__setattr__(self, "tuning_output_dir", resolve_project_path(self.tuning_output_dir, self.project_root))
        object.__setattr__(self, "final_output_dir", resolve_project_path(self.final_output_dir, self.project_root))
        if self.encoding_manifest_path is not None:
            object.__setattr__(
                self,
                "encoding_manifest_path",
                resolve_project_path(self.encoding_manifest_path, self.project_root),
            )
        if self.sample_rows is not None:
            raise ValueError("Optuna full tuning must use sample_rows=None. Smoke sampling is fixed internally.")
        if self.smoke_trials <= 0:
            raise ValueError("smoke_trials must be a positive integer.")
        if self.full_trials <= 0:
            raise ValueError("full_trials must be a positive integer.")
        if self.selection_metric not in ALLOWED_SELECTION_METRICS:
            raise ValueError(
                "Unsupported selection_metric. "
                f"selection_metric={self.selection_metric!r}, allowed={sorted(ALLOWED_SELECTION_METRICS)}"
            )
        if not str(self.study_name_prefix).strip():
            raise ValueError("study_name_prefix must be a non-empty string.")


@dataclass(frozen=True)
class OptunaPipelineResult:
    tuning_output_dir: Path
    smoke_summary_path: Path
    full_summary_path: Path
    best_params_path: Path
    best_trial_summary_path: Path
    pipeline_summary_path: Path
    best_trial_number: int
    best_metric_value: float
    best_params: dict[str, Any]
    train_result: XGBTrainResult
    val_result: ValidationResult


def _require_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "Optuna is required for ML common tuning. Install dependencies from requirements.txt."
        ) from exc
    return optuna


def _prepare_tuning_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            "Existing Optuna artifacts found. Set overwrite=True or change tuning_output_dir. "
            f"output_dir={output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _metric_from_validation(result: ValidationResult, metric_name: str) -> float:
    metrics = result.val_metrics.get("summary", {})
    if metric_name not in metrics:
        raise KeyError(f"Validation metric not found: {metric_name!r}. available={sorted(metrics)}")
    return float(metrics[metric_name])


def _actual_trial_params(best_trial: Any, study_best_params: dict[str, Any]) -> dict[str, Any]:
    trial_params = best_trial.user_attrs.get("params")
    if trial_params is not None:
        return dict(trial_params)
    return dict(study_best_params)


def _run_train_val_trial(
    config: OptunaPipelineConfig,
    *,
    output_dir: Path,
    params: dict[str, Any],
    sample_rows: Optional[int],
) -> tuple[XGBTrainResult, ValidationResult]:
    train_result = train_xgb(
        XGBTrainConfig(
            train_path=config.train_path,
            val_path=config.val_path,
            feature_columns_path=config.feature_columns_path,
            output_dir=output_dir,
            label_col=config.label_col,
            sample_rows=sample_rows,
            allow_nan=config.allow_nan,
            overwrite=config.overwrite,
            seed=config.seed,
            n_jobs=config.n_jobs,
            accelerator=config.accelerator,
            encoding_manifest_path=config.encoding_manifest_path,
            **params,
        )
    )
    val_result = validate_xgb(
        ValidationConfig(
            val_path=config.val_path,
            output_dir=output_dir,
            label_col=config.label_col,
            sample_rows=sample_rows,
            allow_nan=config.allow_nan,
            overwrite=config.overwrite,
            encoding_manifest_path=config.encoding_manifest_path,
        )
    )
    return train_result, val_result


def _write_trials_summary(study: Any, output_dir: Path) -> tuple[Path, Path]:
    records = []
    for trial in study.trials:
        record = {
            "trial_number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
            **trial.user_attrs,
        }
        for name, value in trial.params.items():
            record[f"param_{name}"] = value
        records.append(record)

    trials_summary_path = output_dir / "optuna_trials_summary.csv"
    pd.DataFrame(records).to_csv(trials_summary_path, index=False)

    study_summary_path = output_dir / "optuna_study_summary.json"
    save_json(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "study_name": study.study_name,
            "direction": str(study.direction),
            "trial_count": len(study.trials),
            "best_trial_number": int(study.best_trial.number),
            "best_value": float(study.best_value),
            "best_params": _actual_trial_params(study.best_trial, dict(study.best_params)),
        },
        study_summary_path,
    )
    return trials_summary_path, study_summary_path


def _run_study(
    config: OptunaPipelineConfig,
    *,
    stage_name: str,
    trial_count: int,
    sample_rows: Optional[int],
) -> tuple[Any, Path, Path]:
    optuna = _require_optuna()
    stage_dir = config.tuning_output_dir / stage_name
    _prepare_tuning_dir(stage_dir, overwrite=config.overwrite)

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study_name = f"{config.study_name_prefix}_{stage_name}"
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)

    def objective(trial: Any) -> float:
        params = suggest_xgb_params(trial)
        trial_dir = stage_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "stage": stage_name,
                "trial_number": int(trial.number),
                "sample_rows": sample_rows,
                "sampled": sample_rows is not None,
                "selection_metric": config.selection_metric,
                "params": params,
                "final_test_evaluated": False,
            },
            trial_dir / "trial_params.json",
        )

        train_result, val_result = _run_train_val_trial(
            config,
            output_dir=trial_dir,
            params=params,
            sample_rows=sample_rows,
        )
        metric_value = _metric_from_validation(val_result, config.selection_metric)
        trial.set_user_attr("params", params)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("sample_rows", sample_rows)
        trial.set_user_attr("sampled", sample_rows is not None)
        trial.set_user_attr("train_summary_path", str(train_result.train_summary_path))
        trial.set_user_attr("metrics_path", str(val_result.metrics_path))
        trial.set_user_attr("threshold_path", str(val_result.threshold_path))
        trial.set_user_attr("selected_metric", config.selection_metric)
        trial.set_user_attr("selected_metric_value", metric_value)
        for metric_name in sorted(ALLOWED_SELECTION_METRICS):
            trial.set_user_attr(metric_name, _metric_from_validation(val_result, metric_name))
        return metric_value

    study.optimize(objective, n_trials=trial_count)
    trials_summary_path, study_summary_path = _write_trials_summary(study, stage_dir)
    return study, trials_summary_path, study_summary_path


def _run_final_train_val(
    config: OptunaPipelineConfig,
    best_params: dict[str, Any],
) -> tuple[XGBTrainResult, ValidationResult]:
    train_result = train_xgb(
        XGBTrainConfig(
            train_path=config.train_path,
            val_path=config.val_path,
            feature_columns_path=config.feature_columns_path,
            output_dir=config.final_output_dir,
            model_file_name=config.model_file_name,
            feature_columns_file_name=config.feature_columns_file_name,
            train_summary_file_name=config.train_summary_file_name,
            scores_train_summary_file_name=config.scores_train_summary_file_name,
            feature_importance_file_name=config.feature_importance_file_name,
            label_col=config.label_col,
            sample_rows=None,
            allow_nan=config.allow_nan,
            overwrite=config.overwrite,
            seed=config.seed,
            n_jobs=config.n_jobs,
            accelerator=config.accelerator,
            encoding_manifest_path=config.encoding_manifest_path,
            export_feature_assoc=True,
            **best_params,
        )
    )
    val_result = validate_xgb(
        ValidationConfig(
            val_path=config.val_path,
            output_dir=config.final_output_dir,
            model_file_name=config.model_file_name,
            feature_columns_file_name=config.feature_columns_file_name,
            train_summary_file_name=config.train_summary_file_name,
            scores_train_summary_file_name=config.scores_train_summary_file_name,
            threshold_file_name=config.threshold_file_name,
            metrics_file_name=config.metrics_val_file_name,
            confusion_matrix_file_name=config.confusion_matrix_val_file_name,
            label_col=config.label_col,
            sample_rows=None,
            allow_nan=config.allow_nan,
            overwrite=config.overwrite,
            encoding_manifest_path=config.encoding_manifest_path,
            export_feature_assoc=True,
            export_prediction_scores=True,
        )
    )
    return train_result, val_result


def _resolve_best_params(best_trial: Any, study_best_params: dict[str, Any]) -> dict[str, Any]:
    """Return actual trial params, including fixed search-space params."""

    best_params = _actual_trial_params(best_trial, study_best_params)
    best_params.setdefault("early_stopping_rounds", 40)
    missing = sorted(REQUIRED_XGB_PARAM_KEYS - best_params.keys())
    if missing:
        raise ValueError(
            "Missing required XGB best params. "
            "This usually means the Optuna search space returned fixed params "
            "without storing them on the best trial. "
            f"missing={missing}, best_trial_number={best_trial.number}"
        )
    return best_params


def run_ml_optuna_pipeline(config: OptunaPipelineConfig) -> OptunaPipelineResult:
    """Run smoke tuning, full tuning, and final train/validation for ML common."""

    preflight_ml_inputs(
        InputPaths(
            train_path=config.train_path,
            val_path=config.val_path,
            test_path=None,
            feature_columns_path=config.feature_columns_path,
            encoding_manifest_path=config.encoding_manifest_path,
        ),
        label_col=config.label_col,
        require_test=False,
    )
    _prepare_tuning_dir(config.tuning_output_dir, overwrite=config.overwrite)

    _, smoke_summary_path, _ = _run_study(
        config,
        stage_name="smoke",
        trial_count=config.smoke_trials,
        sample_rows=SMOKE_SAMPLE_ROWS,
    )
    full_study, full_summary_path, _ = _run_study(
        config,
        stage_name="full",
        trial_count=config.full_trials,
        sample_rows=None,
    )

    best_trial = full_study.best_trial
    best_params = _resolve_best_params(best_trial, dict(full_study.best_params))
    full_dir = config.tuning_output_dir / "full"
    best_params_path = full_dir / "best_params.json"
    best_trial_summary_path = full_dir / "best_trial_summary.json"
    save_json(best_params, best_params_path)
    save_json(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trial_number": int(best_trial.number),
            "selection_metric": config.selection_metric,
            "selection_metric_value": float(full_study.best_value),
            "params": best_params,
            "trial_attrs": dict(best_trial.user_attrs),
            "smoke_trials_excluded_from_selection": True,
            "final_test_evaluated": False,
        },
        best_trial_summary_path,
    )

    train_result, val_result = _run_final_train_val(config, best_params)
    pipeline_summary_path = config.tuning_output_dir / "optuna_pipeline_summary.json"
    save_json(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection_metric": config.selection_metric,
            "best_trial_number": int(best_trial.number),
            "best_metric_value": float(full_study.best_value),
            "best_params_path": str(best_params_path),
            "best_trial_summary_path": str(best_trial_summary_path),
            "final_train_summary_path": str(train_result.train_summary_path),
            "final_metrics_val_path": str(val_result.metrics_path),
            "final_threshold_path": str(val_result.threshold_path),
            "smoke_sample_rows": SMOKE_SAMPLE_ROWS,
            "smoke_trials": config.smoke_trials,
            "full_trials": config.full_trials,
            "final_test_evaluated": False,
        },
        pipeline_summary_path,
    )

    return OptunaPipelineResult(
        tuning_output_dir=config.tuning_output_dir,
        smoke_summary_path=smoke_summary_path,
        full_summary_path=full_summary_path,
        best_params_path=best_params_path,
        best_trial_summary_path=best_trial_summary_path,
        pipeline_summary_path=pipeline_summary_path,
        best_trial_number=int(best_trial.number),
        best_metric_value=float(full_study.best_value),
        best_params=best_params,
        train_result=train_result,
        val_result=val_result,
    )
