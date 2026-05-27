"""Optuna search space for ML-02 XGBoost tuning.

This module only defines hyperparameter suggestions. It does not perform I/O,
training, validation, or final test evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import optuna

XGBParamValue = Union[int, float]


def suggest_xgb_params(trial: "optuna.Trial") -> dict[str, XGBParamValue]:
    """Suggest one XGBoost parameter set for ML-02 Optuna tuning."""

    return {
  "n_estimators": 2000,
  "learning_rate": 0.044622354662795224,
  "max_depth": 7,
  "min_child_weight": 39.174758579268904,
  "subsample": 0.8,
  "colsample_bytree": 0.6,
  "reg_lambda": 3.1509230295928043,
  "reg_alpha": 0.0037788909156485253,
  "gamma": 7.870044357784006,
  "early_stopping_rounds": 40
}
