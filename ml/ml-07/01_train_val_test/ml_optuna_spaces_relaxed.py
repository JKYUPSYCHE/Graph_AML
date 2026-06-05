"""Relaxed Optuna search space for ML-07 XGBoost tuning.

This module only defines hyperparameter suggestions. It does not perform I/O,
training, validation, or final test evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import optuna

XGBParamValue = Union[int, float]


def suggest_xgb_params(trial: "optuna.Trial") -> dict[str, XGBParamValue]:
    """Suggest one guarded relaxed XGBoost parameter set for ML-07 tuning."""

    max_depth = trial.suggest_categorical("max_depth", [4, 5])
    if max_depth == 4:
        min_child_weight = trial.suggest_float("min_child_weight", 80.0, 220.0, log=True)
    else:
        min_child_weight = trial.suggest_float("min_child_weight", 140.0, 320.0, log=True)

    return {
        "n_estimators": trial.suggest_int("n_estimators", 1600, 2600, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.055, log=True),
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": trial.suggest_float("subsample", 0.70, 0.85, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.70, step=0.05),
        "reg_lambda": trial.suggest_float("reg_lambda", 30.0, 180.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 10.0, 28.0),
        "early_stopping_rounds": trial.suggest_categorical(
            "early_stopping_rounds",
            [50, 75],
        ),
    }
