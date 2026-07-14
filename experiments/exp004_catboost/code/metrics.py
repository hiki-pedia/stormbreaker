"""Competition metrics and utility constants."""

from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}


def clip_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """Clip group predictions to valid hourly generation ranges."""
    clipped = preds.copy()
    for col, cap in GROUP_CAPACITY_KWH.items():
        if col in clipped.columns:
            clipped[col] = clipped[col].clip(lower=0.0, upper=cap)
    return clipped


def group_nmae(
    y_true: pd.Series,
    y_pred: pd.Series,
    capacity: float,
    min_actual_ratio: float | None = None,
) -> float:
    """Compute NMAE for one group, ignoring missing labels."""
    mask = y_true.notna() & y_pred.notna()
    if min_actual_ratio is not None:
        mask &= y_true >= capacity * min_actual_ratio
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / capacity))


def mean_nmae(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    min_actual_ratio: float | None = None,
) -> float:
    """Compute mean NMAE across available KPX group columns."""
    scores = []
    for col, cap in GROUP_CAPACITY_KWH.items():
        if col in y_true.columns and col in y_pred.columns:
            scores.append(group_nmae(y_true[col], y_pred[col], cap, min_actual_ratio=min_actual_ratio))
    scores = [score for score in scores if not np.isnan(score)]
    if not scores:
        return np.nan
    return float(np.mean(scores))


def one_minus_nmae(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    min_actual_ratio: float | None = None,
) -> float:
    """Compute 1-NMAE from mean NMAE."""
    score = mean_nmae(y_true, y_pred, min_actual_ratio=min_actual_ratio)
    if np.isnan(score):
        return np.nan
    return 1.0 - score
