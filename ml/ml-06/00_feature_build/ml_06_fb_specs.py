"""ML-06 feature build specifications.

ML-06 keeps the ML-05 feature-build architecture, but this stage only applies
two conservative preprocessing policies:

- replace the two account recency ``-1`` sentinels with ``0``
- create pure ratio transform variants while skipping exact duplicates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


TARGET_RECENCY_COLUMNS: tuple[str, str] = (
    "recency__sender__out__seconds_since_last",
    "recency__receiver__in__seconds_since_last",
)

FIRST_FLAG_COLUMNS: tuple[str, str] = (
    "flag__sender__out__is_first_tx",
    "flag__receiver__in__is_first_tx",
)

AUDIT_RECENCY_COLUMNS: tuple[str, ...] = (
    "recency__sender__out__seconds_since_last",
    "recency__receiver__in__seconds_since_last",
    "pair__sender_receiver__forward__seconds_since_last_tx",
    "passflow__sender__in_then_out__seconds_since_last_in__w1h",
    "passflow__sender__in_then_out__seconds_since_last_in__w6h",
    "passflow__sender__in_then_out__seconds_since_last_in__w1d",
)

DERIVED_RATIO_MARKERS: tuple[str, ...] = (
    "log1p",
    "clip",
    "hist_lt2_flag",
)

RATIO_QUANTILE = 0.9999


@dataclass(frozen=True)
class RatioTransformSpec:
    """Ratio transform output names for one base ratio column."""

    base_column: str
    log1p_column: str
    clip_column: str


def is_ratio_feature(column: str) -> bool:
    """Return whether a column name is a ratio feature candidate."""

    return "ratio" in str(column)


def is_derived_ratio_feature(column: str) -> bool:
    """Return whether a ratio column is already a transform/helper output."""

    name = str(column)
    return any(marker in name for marker in DERIVED_RATIO_MARKERS)


def insert_transform_before_window(column: str, transform: str) -> str:
    """Insert a transform token before the terminal window token.

    Examples:
    - ``a__ratio__w1d`` -> ``a__ratio__log1p__w1d``
    - ``amount__paid_recv_ratio`` -> ``amount__paid_recv_ratio__log1p``
    """

    name = str(column).strip()
    token = str(transform).strip()
    if not name:
        raise ValueError("column name must not be empty")
    if not token:
        raise ValueError("transform token must not be empty")

    parts = name.split("__")
    if len(parts) >= 2 and parts[-1].startswith("w"):
        return "__".join([*parts[:-1], token, parts[-1]])
    return f"{name}__{token}"


def ratio_transform_spec(base_column: str) -> RatioTransformSpec:
    """Return the pure log1p and train-p99.99 clipping output names."""

    return RatioTransformSpec(
        base_column=base_column,
        log1p_column=insert_transform_before_window(base_column, "log1p"),
        clip_column=insert_transform_before_window(base_column, "clip_train_p9999"),
    )


def discover_base_ratio_columns(
    feature_columns: Sequence[str],
    available_columns: Iterable[str],
) -> list[str]:
    """Return base ratio columns from the ML-05 model feature list.

    Existing derived columns are intentionally not treated as bases. This avoids
    recursively transforming ``log1p``, ``clip`` or helper flag columns.
    """

    available = set(str(column) for column in available_columns)
    base_columns: list[str] = []
    for column in feature_columns:
        name = str(column)
        if name not in available:
            continue
        if not is_ratio_feature(name):
            continue
        if is_derived_ratio_feature(name):
            continue
        base_columns.append(name)
    return list(dict.fromkeys(base_columns))
