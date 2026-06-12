"""Optuna search space for ML common XGBoost tuning.

This module only defines hyperparameter suggestions. It does not perform I/O,
training, validation, or final test evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import optuna

XGBParamValue = Union[int, float]


def suggest_xgb_params(trial: "optuna.Trial") -> dict[str, XGBParamValue]:
    """Suggest one XGBoost parameter set for ML-06  feature set tuning."""

    return {
        "n_estimators": trial.suggest_int("n_estimators", 800, 1800, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.075, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 60.0, 180.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 0.85, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.70, step=0.05),
        "reg_lambda": trial.suggest_float("reg_lambda", 15.0, 150.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 10.0, 30.0),
        "early_stopping_rounds": trial.suggest_categorical(
            "early_stopping_rounds",
            [25, 35, 50],
        ),
    }
