from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def pcc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    if np.allclose(y_true.std(), 0.0) or np.allclose(y_pred.std(), 0.0):
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def scc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    y_true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float32)
    y_pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float32)
    if np.allclose(y_true_rank.std(), 0.0) or np.allclose(y_pred_rank.std(), 0.0):
        return 0.0
    corr = float(np.corrcoef(y_true_rank, y_pred_rank)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def p1mag_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((np.abs(y_true - y_pred) <= 1.0).mean() * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pcc": pcc_score(y_true, y_pred),
        "scc": scc_score(y_true, y_pred),
        "p1mag": p1mag_score(y_true, y_pred),
    }
