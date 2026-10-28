# -*- coding: utf-8 -*-
"""
fixed_baseline_ar.py

Train-once node-wise autoregressive baselines.

This module follows the same high-level protocol as MP-STGODE evaluation:
train/calibrate once on train, select hyperparameters on validation, and use
fixed coefficients for rolling-origin inference on test. No test-period refit or
rolling recalibration is performed.

Two modes are supported:
    1) Classical AR: raw-scale per-node AR(p), no log transform, no standardization,
       tiny ridge only for numerical stability.
    2) Legacy-compatible Ridge-AR: target log1p, optional standardization, optional
       recency weighting, optional train-only bias correction. This matches the
       behavior of the earlier baseline implementation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from fixed_baseline_utils import (
    compute_metrics,
    extract_split_arrays,
    metric_value,
    nonnegative,
    parse_float_grid,
    parse_int_grid,
    ypred_wnk_to_nwk,
    ytrue_wnk_to_nwk,
)


@dataclass
class FixedARConfig:
    lag_grid: str = "1,3,7"
    l2_grid: str = "1e-8"
    use_log1p: int = 0              # legacy behavior: transform target y only, not X
    standardize: int = 0            # standardize X columns and transformed y using train only
    recency_tau: float = 0.0        # >0 gives newer train windows larger weights
    bias_correction: str = "none"   # none/global/state, estimated from train predictions only
    tune_metric: str = "wMAPE%"


@dataclass
class ARNodeModel:
    w: np.ndarray
    b: float
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


@dataclass
class FixedARModel:
    models: List[ARNodeModel]
    lag: int
    l2: float
    use_log1p: int
    standardize: int
    recency_tau: float
    bias_correction: str
    train_bias: Optional[np.ndarray] = None


def _safe_expm1(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(np.clip(x, -60.0, 20.0)), 0.0)


def _make_recency_weights(n_samples: int, tau: float) -> Optional[np.ndarray]:
    if tau is None or float(tau) <= 0:
        return None
    idx = np.arange(n_samples, dtype=float)  # old -> new
    return np.exp((idx - (n_samples - 1)) / float(tau))


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float, sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    M, F = X.shape
    X_aug = np.concatenate([X, np.ones((M, 1), dtype=float)], axis=1)
    X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=1e12, neginf=-1e12)
    y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)

    if sample_weight is not None:
        sw = np.sqrt(np.asarray(sample_weight, dtype=float)).reshape(M, 1)
        X_aug = X_aug * sw
        y = y * sw.reshape(-1)

    A = X_aug.T @ X_aug
    b = X_aug.T @ y
    reg = float(l2) * np.eye(F + 1)
    reg[-1, -1] = 0.0
    theta = np.linalg.solve(A + reg + 1e-10 * np.eye(F + 1), b)
    return theta[:-1], float(theta[-1])


def fit_ar_once(train_view: Dict[str, object], lag: int, l2: float, use_log1p: int,
                standardize: int, recency_tau: float = 0.0,
                bias_correction: str = "none") -> FixedARModel:
    """Fit one AR(p) model per node using the training split only.

    Important legacy-compatible convention: if use_log1p=1, only the target y is
    transformed by log1p. The historical AR inputs X remain in the incidence scale,
    matching the earlier baseline implementation's behavior.
    """
    train = extract_split_arrays(train_view)
    hist = nonnegative(train["hist_incidence"])
    fut = nonnegative(train["future_incidence"])
    Wn, N, H = hist.shape
    lag = min(int(lag), H)
    models: List[ARNodeModel] = []
    sample_weight = _make_recency_weights(Wn, recency_tau)

    for i in range(N):
        X = hist[:, i, -lag:].astype(float)
        y = fut[:, i, 0].astype(float)
        X_fit = X
        y_fit = np.log1p(nonnegative(y)) if int(use_log1p) else y

        if int(standardize):
            x_mean = X_fit.mean(axis=0, keepdims=True)
            x_std = X_fit.std(axis=0, keepdims=True)
            x_std = np.where(x_std < 1e-8, 1.0, x_std)
            y_mean = float(y_fit.mean())
            y_std = float(y_fit.std()) if float(y_fit.std()) > 1e-8 else 1.0
            Xz = (X_fit - x_mean) / x_std
            yz = (y_fit - y_mean) / y_std
        else:
            x_mean = np.zeros((1, lag)); x_std = np.ones((1, lag)); y_mean = 0.0; y_std = 1.0
            Xz, yz = X_fit, y_fit

        wcoef, bcoef = _ridge_fit(Xz, yz, l2=l2, sample_weight=sample_weight)
        models.append(ARNodeModel(w=wcoef, b=bcoef, x_mean=x_mean.reshape(-1),
                                  x_std=x_std.reshape(-1), y_mean=y_mean, y_std=y_std))

    return FixedARModel(models=models, lag=lag, l2=float(l2), use_log1p=int(use_log1p),
                        standardize=int(standardize), recency_tau=float(recency_tau),
                        bias_correction=str(bias_correction), train_bias=None)


def predict_ar_raw_wnk(split_view: Dict[str, object], model: FixedARModel) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw predictions and targets as [W,N,K], before optional bias correction."""
    split = extract_split_arrays(split_view)
    hist = nonnegative(split["hist_incidence"])
    fut = split["future_incidence"]
    Wn, N, _ = hist.shape
    K = fut.shape[2]
    pred = np.zeros((Wn, N, K), dtype=float)

    for widx in range(Wn):
        buffers = [hist[widx, i, -model.lag:].astype(float).copy() for i in range(N)]
        for k in range(K):
            step_pred = np.zeros(N, dtype=float)
            for i, node_model in enumerate(model.models):
                x = buffers[i][-model.lag:].reshape(1, -1)
                if model.standardize:
                    x = (x - node_model.x_mean.reshape(1, -1)) / node_model.x_std.reshape(1, -1)
                yhat = float(x @ node_model.w.reshape(-1, 1) + node_model.b)
                if model.standardize:
                    yhat = yhat * node_model.y_std + node_model.y_mean
                if model.use_log1p:
                    yhat = float(_safe_expm1(np.array([yhat]))[0])
                step_pred[i] = max(0.0, yhat)
            pred[widx, :, k] = step_pred
            for i in range(N):
                buffers[i] = np.concatenate([buffers[i][1:], np.array([step_pred[i]])])
    return pred, fut


def _estimate_bias_nwk(y_true_nwk: np.ndarray, y_pred_nwk: np.ndarray, mode: str):
    if mode == "none":
        return None
    if mode == "global":
        return np.array(float((y_true_nwk - y_pred_nwk).mean()))
    if mode == "state":
        return (y_true_nwk - y_pred_nwk).mean(axis=(1, 2), keepdims=True)
    raise ValueError(f"Unknown bias_correction={mode}")


def _apply_bias_nwk(y_pred_nwk: np.ndarray, bias):
    if bias is None:
        return y_pred_nwk
    return nonnegative(y_pred_nwk + bias)


def predict_ar_once(split_view: Dict[str, object], model: FixedARModel) -> Tuple[np.ndarray, np.ndarray]:
    pred_wnk, fut_wnk = predict_ar_raw_wnk(split_view, model)
    y_pred = ypred_wnk_to_nwk(pred_wnk)
    y_true = ytrue_wnk_to_nwk(fut_wnk)
    y_pred = _apply_bias_nwk(y_pred, model.train_bias)
    return y_pred, y_true


def tune_and_fit_ar(data_loader: Dict[str, object], config: FixedARConfig) -> FixedARModel:
    bv = data_loader["baseline_views"]
    best_score = float("inf")
    best_setting = None
    for lag in parse_int_grid(config.lag_grid):
        for l2 in parse_float_grid(config.l2_grid):
            model = fit_ar_once(bv["train"], lag, l2, config.use_log1p, config.standardize,
                                config.recency_tau, config.bias_correction)
            # Bias correction is part of the fixed model and is estimated only from train.
            train_pred_raw, train_true_raw = predict_ar_raw_wnk(bv["train"], model)
            train_pred_nwk = ypred_wnk_to_nwk(train_pred_raw)
            train_true_nwk = ytrue_wnk_to_nwk(train_true_raw)
            model.train_bias = _estimate_bias_nwk(train_true_nwk, train_pred_nwk, config.bias_correction)

            y_pred_val, y_true_val = predict_ar_once(bv["validate"], model)
            score = metric_value(compute_metrics(y_true_val, y_pred_val), config.tune_metric)
            if score < best_score:
                best_score = score
                best_setting = (lag, l2)
    if best_setting is None:
        raise RuntimeError("AR validation tuning produced no candidate.")

    final_model = fit_ar_once(bv["train"], best_setting[0], best_setting[1], config.use_log1p,
                              config.standardize, config.recency_tau, config.bias_correction)
    train_pred_raw, train_true_raw = predict_ar_raw_wnk(bv["train"], final_model)
    final_model.train_bias = _estimate_bias_nwk(ytrue_wnk_to_nwk(train_true_raw),
                                                ypred_wnk_to_nwk(train_pred_raw),
                                                config.bias_correction)
    return final_model


def evaluate_fixed_ar(data_loader: Dict[str, object], config: FixedARConfig):
    bv = data_loader["baseline_views"]
    model = tune_and_fit_ar(data_loader, config)
    outputs = {}
    for split_name in ["train", "validate", "test"]:
        y_pred, y_true = predict_ar_once(bv[split_name], model)
        outputs[split_name] = {"y_true": y_true, "y_pred": y_pred, "metrics": compute_metrics(y_true, y_pred)}
    return model, outputs
