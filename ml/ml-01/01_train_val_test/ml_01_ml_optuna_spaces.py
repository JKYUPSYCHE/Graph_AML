"""Optuna search space for ML-01 XGBoost tuning.

This module only defines hyperparameter suggestions. It does not perform I/O,
training, validation, or final test evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import optuna

XGBParamValue = Union[int, float]


def suggest_xgb_params(trial: "optuna.Trial") -> dict[str, XGBParamValue]:
    """Suggest one XGBoost parameter set for ML-01 Optuna tuning."""

    return {
        "n_estimators": trial.suggest_int("n_estimators", 800, 4000, step=400),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.05),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "early_stopping_rounds": 50,
    }
