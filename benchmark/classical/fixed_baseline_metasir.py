# -*- coding: utf-8 -*-
"""
fixed_baseline_metasir.py

Fixed-calibration Meta-SIR baseline using the same Lagrangian metapopulation
transmission structure as the mechanistic branch of MP-STGODE, but without the
neural encoder, Neural ODE representation learner, or residual correction.

Important protocol:
    - beta/gamma are calibrated only from the train split;
    - validation selects hyperparameters only;
    - test uses fixed parameters and historical-window mobility templates;
    - no true future mobility and no test target observations are used for refitting.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from fixed_baseline_utils import (
    compute_metrics,
    extract_split_arrays,
    nonnegative,
    reconstruct_state_at_forecast_origin,
    ypred_wnk_to_nwk,
    ytrue_wnk_to_nwk,
)


@dataclass
class FixedMetaSIRConfig:
    gamma_fixed: float = 0.14
    beta_mode: str = "global"  # "global" or "node"
    beta_min: float = 1e-8
    beta_max: float = 5.0
    beta_ridge: float = 1e-6
    mobility_scale: float = 3.0
    use_mobility_binary_mask: int = 1
    future_mobility_mode: str = "weekday_cycle"  # last_repeat/recent_mean/weekday_cycle/recent_cycle/mean_repeat
    future_recent_k: int = 3
    future_week_period: int = 7
    local_self_loop_weight: float = 0.5


@dataclass
class FixedMetaSIRParams:
    beta: np.ndarray  # [N]
    gamma: float
    beta_mode: str


def build_future_mobility_from_history(
    adj_hist: np.ndarray,
    K: int,
    mode: str = "weekday_cycle",
    recent_k: int = 3,
    week_period: int = 7,
) -> np.ndarray:
    """
    Build future mobility templates using historical mobility only.

    This mirrors the mechanism branch idea in MP-STGODE: future mobility forcing is
    generated from the input window rather than taken from unseen future mobility.
    """
    A = nonnegative(adj_hist)
    if A.ndim != 4:
        raise ValueError(f"adj_hist should be [W,H,N,N], got {A.shape}")
    W, H, N, _ = A.shape

    if mode == "last_repeat":
        template = A[:, -1:, :, :]
    elif mode == "recent_mean":
        k = min(max(int(recent_k), 1), H)
        template = A[:, -k:, :, :].mean(axis=1, keepdims=True)
    elif mode == "mean_repeat":
        template = A.mean(axis=1, keepdims=True)
    elif mode == "recent_cycle":
        k = min(max(int(recent_k), 1), H)
        template = A[:, -k:, :, :]
    elif mode == "weekday_cycle":
        p = min(max(int(week_period), 1), H)
        template = A[:, -p:, :, :]
    else:
        raise ValueError("future_mobility_mode must be last_repeat/recent_mean/weekday_cycle/recent_cycle/mean_repeat.")

    L = template.shape[1]
    reps = (K + L - 1) // L
    return np.tile(template, (1, reps, 1, 1))[:, :K, :, :]


def binary_and_normalized_adj(mob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return binary mask and row-normalized binary adjacency as in the main mechanism branch."""
    M = nonnegative(mob)
    N = M.shape[-1]
    eye = np.eye(N, dtype=float)
    # Broadcast eye to [...,N,N].
    bin_adj = (M > 0).astype(float) * (1.0 - eye.reshape((1,) * (M.ndim - 2) + (N, N)))
    row_sum = bin_adj.sum(axis=-1, keepdims=True) + 1e-12
    norm_adj = bin_adj / row_sum
    return bin_adj, norm_adj


def lagrangian_design_matrix(
    S: np.ndarray,
    I: np.ndarray,
    population: np.ndarray,
    mob: np.ndarray,
    use_binary_mask: int = 1,
    local_self_loop_weight: float = 0.5,
) -> np.ndarray:
    """
    Build C such that one-step Meta-SIR incidence is approximately C @ beta.

    The implementation follows the three-term structure in MP-STGODE MetaSIRRollout:
        term1: infectious visitors from m contribute to infection risk in n;
        term2: susceptible residents from n visit m and are exposed at destination m;
        term3: internal/local transmission.

    S/I: [W,N]
    mob: [W,N,N]
    return C: [W,N,N], where C[w,n,j] is the coefficient of beta_j for incidence at node n.
    """
    S = nonnegative(S)
    I = nonnegative(I)
    M = nonnegative(mob)
    W, N = S.shape
    pop = np.asarray(population, dtype=float).reshape(N)

    bin_adj, norm_adj = binary_and_normalized_adj(M)
    M_eff = M * bin_adj if int(use_binary_mask) else M

    C = np.zeros((W, N, N), dtype=float)

    # term1: beta_n * S_n/N_n * sum_m h_nm/N_m * I_m
    mobility_term1 = M_eff / (pop.reshape(1, 1, N) + 1e-12)
    sum_Im = np.einsum("wnm,wm->wn", mobility_term1, I)
    coeff_term1 = sum_Im * S / (pop.reshape(1, N) + 1e-12)
    for n in range(N):
        C[:, n, n] += coeff_term1[:, n]

    # term2: beta_m * I_m/N_m * sum_n h_mn/N_n * S_n.
    # In the main code beta_m is a row-normalized aggregation of beta_j through norm_adj[j,n].
    h_nm = np.transpose(M, (0, 2, 1))
    mobility_term2 = h_nm / (pop.reshape(1, N, 1) + 1e-12)
    sum_Sn = np.einsum("wnm,wn->wm", mobility_term2, S)
    coeff_term2_for_beta_m = sum_Sn * I / (pop.reshape(1, N) + 1e-12)  # [W,N]
    # beta_m[n] = sum_j norm_adj[j,n] * beta_j according to the main model einsum('btji,btj->bti').
    C += coeff_term2_for_beta_m[:, :, None] * np.transpose(norm_adj, (0, 2, 1))

    # term3: internal/local transmission with h_nn = 0.5 * self loop mobility.
    h_diag = np.diagonal(M, axis1=1, axis2=2) * float(local_self_loop_weight)
    sum_Sn_local = h_diag * S / (pop.reshape(1, N) + 1e-12)
    coeff_term3 = 2.0 * (sum_Sn_local * I) / (pop.reshape(1, N) + 1e-12)
    for n in range(N):
        C[:, n, n] += coeff_term3[:, n]

    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)


def _estimate_beta_meta_forecast_origin(
    split: Dict[str, np.ndarray],
    population: np.ndarray,
    config: FixedMetaSIRConfig,
) -> np.ndarray:
    """Estimate fixed beta by one-step least squares at forecast origins."""
    hist = nonnegative(split["hist_incidence"])
    fut = nonnegative(split["future_incidence"])
    S0 = split["S0_forecast_origin"]
    A_hist = nonnegative(split["adj_hist"])
    W, N, _ = hist.shape

    S, I, _ = reconstruct_state_at_forecast_origin(hist, S0, population, gamma=config.gamma_fixed)
    A_future = build_future_mobility_from_history(
        A_hist,
        K=1,
        mode=config.future_mobility_mode,
        recent_k=config.future_recent_k,
        week_period=config.future_week_period,
    )[:, 0] * float(config.mobility_scale)

    C = lagrangian_design_matrix(
        S=S,
        I=I,
        population=population,
        mob=A_future,
        use_binary_mask=config.use_mobility_binary_mask,
        local_self_loop_weight=config.local_self_loop_weight,
    )  # [W,N,N]
    y = fut[:, :, 0]  # [W,N]

    if config.beta_mode == "global":
        x = C.sum(axis=2)  # [W,N]
        b = float(np.sum(x * y) / (np.sum(x * x) + config.beta_ridge + 1e-12))
        beta = np.full(N, b, dtype=float)
    elif config.beta_mode == "node":
        X = C.reshape(W * N, N)
        Y = y.reshape(W * N)
        A = X.T @ X + float(config.beta_ridge) * np.eye(N)
        bvec = X.T @ Y
        beta = np.linalg.solve(A + 1e-10 * np.eye(N), bvec)
        if not np.all(np.isfinite(beta)):
            x = C.sum(axis=2)
            gb = float(np.sum(x * y) / (np.sum(x * x) + config.beta_ridge + 1e-12))
            beta = np.full(N, gb, dtype=float)
    else:
        raise ValueError("beta_mode must be 'global' or 'node'.")

    return np.clip(beta, config.beta_min, config.beta_max).astype(float)


def fit_fixed_metasir(train_view: Dict[str, object], population: np.ndarray, config: FixedMetaSIRConfig) -> FixedMetaSIRParams:
    train = extract_split_arrays(train_view)
    if "adj_hist" not in train:
        raise KeyError("Meta-SIR requires adj_hist in baseline_views.")
    beta = _estimate_beta_meta_forecast_origin(train, population, config)
    return FixedMetaSIRParams(beta=beta, gamma=float(config.gamma_fixed), beta_mode=config.beta_mode)


def predict_fixed_metasir_split(
    split_view: Dict[str, object],
    population: np.ndarray,
    params: FixedMetaSIRParams,
    config: FixedMetaSIRConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    split = extract_split_arrays(split_view)
    hist = nonnegative(split["hist_incidence"])
    fut = split["future_incidence"]
    S0 = split["S0_forecast_origin"]
    A_hist = nonnegative(split["adj_hist"])
    W, N, _ = hist.shape
    K = fut.shape[2]

    S, I, R = reconstruct_state_at_forecast_origin(hist, S0, population, gamma=params.gamma)
    A_future = build_future_mobility_from_history(
        A_hist,
        K=K,
        mode=config.future_mobility_mode,
        recent_k=config.future_recent_k,
        week_period=config.future_week_period,
    ) * float(config.mobility_scale)

    beta = params.beta.reshape(N)
    y_pred = np.zeros((W, N, K), dtype=float)
    for k in range(K):
        C = lagrangian_design_matrix(
            S=S,
            I=I,
            population=population,
            mob=A_future[:, k],
            use_binary_mask=config.use_mobility_binary_mask,
            local_self_loop_weight=config.local_self_loop_weight,
        )
        newI = np.einsum("wnj,j->wn", C, beta)
        newI = np.minimum(nonnegative(newI), np.maximum(S, 0.0))
        recov = np.minimum(params.gamma * I, np.maximum(I + newI, 0.0))
        y_pred[:, :, k] = newI
        S = np.maximum(S - newI, 0.0)
        I = np.maximum(I + newI - recov, 0.0)
        R = np.maximum(R + recov, 0.0)

    return ypred_wnk_to_nwk(y_pred), ytrue_wnk_to_nwk(fut)


def evaluate_fixed_metasir(data_loader: Dict[str, object], population: np.ndarray, config: FixedMetaSIRConfig):
    bv = data_loader["baseline_views"]
    model_params = fit_fixed_metasir(bv["train"], population, config)
    outputs = {}
    for split_name in ["train", "validate", "test"]:
        y_pred, y_true = predict_fixed_metasir_split(bv[split_name], population, model_params, config)
        outputs[split_name] = {"y_true": y_true, "y_pred": y_pred, "metrics": compute_metrics(y_true, y_pred)}
    return model_params, outputs
