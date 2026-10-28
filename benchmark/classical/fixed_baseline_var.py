# -*- coding: utf-8 -*-
"""
fixed_baseline_var.py

Train-once Ridge-VAR baseline.

The model is ridge-regularized because a 51-node VAR can be ill-conditioned under
limited training samples. It supports two evaluation-compatible forms:
    - recursive: fit one-step VAR and roll it forward recursively;
    - direct: fit one multi-output ridge regression for each forecast lead.

All variants fit coefficients on train, select hyperparameters on validation, and
use fixed coefficients on test. No test-period refit or rolling recalibration is
performed.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

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
class FixedVARConfig:
    lag_grid: str = "1,3,7"
    l2_grid: str = "0.01,0.1,1,10,100"
    use_log1p: int = 0              # legacy v2 default: 0; if 1, transform target y only
    standardize: int = 1
    var_mode: str = "direct"        # direct or recursive
    recency_tau: float = 14.0
    bias_correction: str = "state"  # none/global/state, estimated from train only
    tune_metric: str = "wMAPE%"


@dataclass
class HorizonModel:
    W: np.ndarray
    b: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


@dataclass
class FixedVARModel:
    lag: int
    l2: float
    use_log1p: int
    standardize: int
    var_mode: str
    recency_tau: float
    bias_correction: str
    x_mean: np.ndarray
    x_std: np.ndarray
    num_nodes: int
    # recursive fields
    W: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    y_mean: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None
    # direct fields
    horizon_models: Optional[List[HorizonModel]] = None
    train_bias: Optional[np.ndarray] = None


def _safe_expm1(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(np.clip(x, -60.0, 20.0)), 0.0)


def _make_recency_weights(n_samples: int, tau: float) -> Optional[np.ndarray]:
    if tau is None or float(tau) <= 0:
        return None
    idx = np.arange(n_samples, dtype=float)
    return np.exp((idx - (n_samples - 1)) / float(tau))


def _build_X(split: Dict[str, np.ndarray], lag: int) -> np.ndarray:
    hist = nonnegative(split["hist_incidence"])
    Wn, N, H = hist.shape
    lag = min(int(lag), H)
    return np.transpose(hist[:, :, -lag:], (0, 2, 1)).reshape(Wn, lag * N).astype(float)


def _transform_y(Y: np.ndarray, use_log1p: int) -> np.ndarray:
    return np.log1p(nonnegative(Y)) if int(use_log1p) else np.asarray(Y, dtype=float)


def _inverse_y(Y: np.ndarray, use_log1p: int) -> np.ndarray:
    return _safe_expm1(Y) if int(use_log1p) else Y


def _ridge_fit_multi(X: np.ndarray, Y: np.ndarray, l2: float, sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    M, F = X.shape
    X_aug = np.concatenate([X, np.ones((M, 1), dtype=float)], axis=1)
    X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=1e12, neginf=-1e12)
    Y = np.nan_to_num(Y, nan=0.0, posinf=1e12, neginf=-1e12)
    if sample_weight is not None:
        sw = np.sqrt(np.asarray(sample_weight, dtype=float)).reshape(M, 1)
        X_aug = X_aug * sw
        Y = Y * sw
    A = X_aug.T @ X_aug
    B = X_aug.T @ Y
    reg = float(l2) * np.eye(F + 1)
    reg[-1, -1] = 0.0
    theta = np.linalg.solve(A + reg + 1e-10 * np.eye(F + 1), B)
    return theta[:-1], theta[-1]


def _standardize_X_train(X: np.ndarray, standardize: int):
    if not int(standardize):
        return X, np.zeros((1, X.shape[1])), np.ones((1, X.shape[1]))
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    return (X - x_mean) / x_std, x_mean, x_std


def _standardize_Y_train(Y: np.ndarray, standardize: int):
    if not int(standardize):
        return Y, np.zeros((1, Y.shape[1])), np.ones((1, Y.shape[1]))
    y_mean = Y.mean(axis=0, keepdims=True)
    y_std = Y.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    return (Y - y_mean) / y_std, y_mean, y_std


def fit_var_once(train_view: Dict[str, object], lag: int, l2: float, use_log1p: int,
                 standardize: int, var_mode: str = "direct", recency_tau: float = 0.0,
                 bias_correction: str = "none") -> FixedVARModel:
    train = extract_split_arrays(train_view)
    X = _build_X(train, lag)
    lag = min(int(lag), train["hist_incidence"].shape[2])
    fut = nonnegative(train["future_incidence"])
    Wn, N, K = fut.shape
    Xz, x_mean, x_std = _standardize_X_train(X, standardize)
    sample_weight = _make_recency_weights(Wn, recency_tau)
    mode = str(var_mode)
    if mode not in ("recursive", "direct"):
        raise ValueError("var_mode must be 'recursive' or 'direct'.")

    base = FixedVARModel(lag=lag, l2=float(l2), use_log1p=int(use_log1p),
                         standardize=int(standardize), var_mode=mode,
                         recency_tau=float(recency_tau), bias_correction=str(bias_correction),
                         x_mean=x_mean.reshape(-1), x_std=x_std.reshape(-1), num_nodes=N)

    if mode == "recursive":
        Y = _transform_y(fut[:, :, 0], use_log1p)
        Yz, y_mean, y_std = _standardize_Y_train(Y, standardize)
        Wmat, b = _ridge_fit_multi(Xz, Yz, l2, sample_weight)
        base.W = Wmat
        base.b = b
        base.y_mean = y_mean.reshape(-1)
        base.y_std = y_std.reshape(-1)
        return base

    horizon_models: List[HorizonModel] = []
    for k in range(K):
        Yk = _transform_y(fut[:, :, k], use_log1p)
        Ykz, y_mean, y_std = _standardize_Y_train(Yk, standardize)
        Wk, bk = _ridge_fit_multi(Xz, Ykz, l2, sample_weight)
        horizon_models.append(HorizonModel(W=Wk, b=bk, y_mean=y_mean.reshape(-1), y_std=y_std.reshape(-1)))
    base.horizon_models = horizon_models
    return base


def _prep_X_for_model(X: np.ndarray, model: FixedVARModel) -> np.ndarray:
    if model.standardize:
        return (X - model.x_mean.reshape(1, -1)) / model.x_std.reshape(1, -1)
    return X


def predict_var_raw_wnk(split_view: Dict[str, object], model: FixedVARModel) -> Tuple[np.ndarray, np.ndarray]:
    split = extract_split_arrays(split_view)
    hist = nonnegative(split["hist_incidence"])
    fut = split["future_incidence"]
    Wn, N, _ = hist.shape
    K = fut.shape[2]
    pred = np.zeros((Wn, N, K), dtype=float)

    if model.var_mode == "direct":
        X = _prep_X_for_model(_build_X(split, model.lag), model)
        for k, hm in enumerate(model.horizon_models or []):
            yhat = X @ hm.W + hm.b.reshape(1, -1)
            if model.standardize:
                yhat = yhat * hm.y_std.reshape(1, -1) + hm.y_mean.reshape(1, -1)
            pred[:, :, k] = nonnegative(_inverse_y(yhat, model.use_log1p))
        return pred, fut

    # recursive mode
    for widx in range(Wn):
        buffer = hist[widx, :, -model.lag:].astype(float).copy()
        for k in range(K):
            X = np.transpose(buffer.reshape(1, N, model.lag), (0, 2, 1)).reshape(1, model.lag * N)
            Xz = _prep_X_for_model(X, model)
            yhat = Xz @ model.W + model.b.reshape(1, -1)
            if model.standardize:
                yhat = yhat * model.y_std.reshape(1, -1) + model.y_mean.reshape(1, -1)
            yhat = nonnegative(_inverse_y(yhat, model.use_log1p).reshape(N))
            pred[widx, :, k] = yhat
            buffer = np.concatenate([buffer[:, 1:], yhat.reshape(N, 1)], axis=1)
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


def predict_var_once(split_view: Dict[str, object], model: FixedVARModel) -> Tuple[np.ndarray, np.ndarray]:
    pred_wnk, fut_wnk = predict_var_raw_wnk(split_view, model)
    y_pred = ypred_wnk_to_nwk(pred_wnk)
    y_true = ytrue_wnk_to_nwk(fut_wnk)
    y_pred = _apply_bias_nwk(y_pred, model.train_bias)
    return y_pred, y_true


def tune_and_fit_var(data_loader: Dict[str, object], config: FixedVARConfig) -> FixedVARModel:
    bv = data_loader["baseline_views"]
    best_score = float("inf")
    best_setting = None
    for lag in parse_int_grid(config.lag_grid):
        for l2 in parse_float_grid(config.l2_grid):
            model = fit_var_once(bv["train"], lag, l2, config.use_log1p, config.standardize,
                                 config.var_mode, config.recency_tau, config.bias_correction)
            train_pred_raw, train_true_raw = predict_var_raw_wnk(bv["train"], model)
            model.train_bias = _estimate_bias_nwk(ytrue_wnk_to_nwk(train_true_raw),
                                                  ypred_wnk_to_nwk(train_pred_raw),
                                                  config.bias_correction)
            y_pred_val, y_true_val = predict_var_once(bv["validate"], model)
            score = metric_value(compute_metrics(y_true_val, y_pred_val), config.tune_metric)
            if score < best_score:
                best_score = score
                best_setting = (lag, l2)
    if best_setting is None:
        raise RuntimeError("Ridge-VAR validation tuning produced no candidate.")

    final_model = fit_var_once(bv["train"], best_setting[0], best_setting[1], config.use_log1p,
                               config.standardize, config.var_mode, config.recency_tau,
                               config.bias_correction)
    train_pred_raw, train_true_raw = predict_var_raw_wnk(bv["train"], final_model)
    final_model.train_bias = _estimate_bias_nwk(ytrue_wnk_to_nwk(train_true_raw),
                                                ypred_wnk_to_nwk(train_pred_raw),
                                                config.bias_correction)
    return final_model


def evaluate_fixed_var(data_loader: Dict[str, object], config: FixedVARConfig):
    bv = data_loader["baseline_views"]
    model = tune_and_fit_var(data_loader, config)
    outputs = {}
    for split_name in ["train", "validate", "test"]:
        y_pred, y_true = predict_var_once(bv[split_name], model)
        outputs[split_name] = {"y_true": y_true, "y_pred": y_pred, "metrics": compute_metrics(y_true, y_pred)}
    return model, outputs
