# -*- coding: utf-8 -*-
"""
fixed_baseline_sir.py

Clean fixed-calibration independent SIR baseline for MP-STGODE comparison.

Main evaluation principle
-------------------------
1) N regions are modeled by N independent standard-incidence SIR systems.
   There is no graph, mobility, or inter-regional coupling in this baseline.
2) The recovery rate gamma is fixed in advance.
3) One transmission rate beta_i is calibrated for each region using TRAINING
   forecast origins only. beta_i is then kept fixed during validation/test.
4) At every forecast origin, only observations already available up to the
   LAST OBSERVED DAY are used to reconstruct the current S/I/R state.
5) No observation inside the future prediction horizon is used for state
   reconstruction or parameter recalibration.
6) The output keeps the common [N, W, K] shape and uses exactly the same
   compute_metrics() implementation as MP-STGODE.

This file is intentionally simple.  It does NOT implement rolling beta
recalibration.  A rolling fit-and-forecast SIR should be treated as a separate
operational sensitivity analysis rather than replacing this fixed-calibration
main benchmark.

Compatibility
-------------
This module is the fixed-parameter SIR implementation used by
run_fixed_baselines.py.  It expects baseline views produced by
run_fixed_baselines.build_baseline_views(), in particular:
    hist_incidence : [W, N, H]
    future_incidence: [W, N, K]
    S_hist         : [W, N, H]

Here H is the history length, K is the forecast horizon, and the last entry of
S_hist is the susceptible state on the last observed day immediately before
future_incidence[..., 0].
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from fixed_baseline_utils import (
    active_infected_from_history,
    compute_metrics,
    extract_split_arrays,
    nonnegative,
    ypred_wnk_to_nwk,
    ytrue_wnk_to_nwk,
)


@dataclass
class FixedSIRConfig:
    """Configuration of the fixed-calibration independent SIR baseline."""

    # Prespecified epidemiological recovery rate.
    gamma_fixed: float = 0.14

    # 'node': one beta_i per region (recommended main baseline).
    # 'global': one beta shared by all regions (optional diagnostic).
    beta_mode: str = "node"

    # Simple numerical bounds for the least-squares beta estimate.
    beta_min: float = 1e-8
    beta_max: float = 1.0

    # Very small ridge term used only to avoid division by zero.
    ridge: float = 1e-8


@dataclass
class FixedSIRParams:
    """Parameters calibrated from the training split."""

    beta: np.ndarray  # shape [N]
    gamma: float
    beta_mode: str


def _get_last_observed_susceptible(split: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Return S on the last observed day of every history window.

    S_hist has shape [W, N, H].  Its final time index is the last day that is
    available to the forecaster.  The first prediction target occurs one day
    later.  Using S_hist[..., -1] therefore avoids using susceptible values
    from the first future target day.
    """
    if "S_hist" not in split:
        raise KeyError(
            "Clean fixed SIR requires 'S_hist' in each baseline split. "
            "Please build the split with run_fixed_baselines.build_baseline_views()."
        )

    S_hist = np.asarray(split["S_hist"], dtype=float)
    if S_hist.ndim != 3:
        raise ValueError(f"S_hist must have shape [W,N,H], got {S_hist.shape}")

    return nonnegative(S_hist[:, :, -1])


def _reconstruct_state_at_last_observed_day(
    hist_incidence: np.ndarray,
    S_last: np.ndarray,
    population: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct S/I/R on the last observed day before the forecast horizon.

    I is a recovery-weighted sum of recent observed incident cases:

        I_i(t0) = sum_{ell=0}^{H-1} C_i(t0-ell) (1-gamma)^ell,

    where t0 is the last observed day.  R is obtained from population balance:

        R_i(t0) = N_i - S_i(t0) - I_i(t0).

    These are reconstructed/proxy compartment states, not independently
    observed epidemiological compartments.
    """
    hist = nonnegative(hist_incidence)
    S = nonnegative(S_last)
    pop = np.asarray(population, dtype=float).reshape(1, -1)

    if hist.ndim != 3:
        raise ValueError(f"hist_incidence must have shape [W,N,H], got {hist.shape}")
    if S.ndim != 2:
        raise ValueError(f"S_last must have shape [W,N], got {S.shape}")
    if hist.shape[:2] != S.shape:
        raise ValueError(
            f"hist_incidence and S_last disagree on [W,N]: {hist.shape[:2]} vs {S.shape}"
        )
    if pop.shape[1] != hist.shape[1]:
        raise ValueError(
            f"population length {pop.shape[1]} does not match number of nodes {hist.shape[1]}"
        )

    # Current active-infected proxy from the observed history only.
    I = active_infected_from_history(hist, gamma=float(gamma))

    # Enforce simple population-balance bounds.
    S = np.minimum(S, pop)
    I = np.minimum(nonnegative(I), np.maximum(pop - S, 0.0))
    R = np.maximum(pop - S - I, 0.0)

    return S, I, R


def _estimate_beta_from_train_origins(
    train: Dict[str, np.ndarray],
    population: np.ndarray,
    config: FixedSIRConfig,
) -> np.ndarray:
    """
    Calibrate beta using one causal one-step transition per TRAINING origin.

    For every training forecast origin, let t0 denote the last observed day.
    We reconstruct S_i(t0), I_i(t0) from the historical window and fit

        C_i(t0+1) ~= beta_i * S_i(t0) * I_i(t0) / N_i.

    Thus each training forecast origin contributes exactly one one-step
    transition.  Overlapping history windows do not cause the same within-window
    day to be repeatedly treated as an independent beta-fitting target.
    """
    hist = nonnegative(train["hist_incidence"])
    future = nonnegative(train["future_incidence"])
    S_last = _get_last_observed_susceptible(train)

    if future.ndim != 3 or future.shape[2] < 1:
        raise ValueError(f"future_incidence must have shape [W,N,K>=1], got {future.shape}")

    S, I, _ = _reconstruct_state_at_last_observed_day(
        hist_incidence=hist,
        S_last=S_last,
        population=population,
        gamma=config.gamma_fixed,
    )

    pop = np.asarray(population, dtype=float).reshape(1, -1)
    X = S * I / (pop + 1e-12)         # [W,N]
    y = future[:, :, 0]               # first future day [W,N]

    if config.beta_mode == "node":
        numer = np.sum(X * y, axis=0)             # [N]
        denom = np.sum(X * X, axis=0)             # [N]

        # Global estimate is used only as a numerical fallback for a node with
        # essentially no usable training signal.
        global_numer = float(np.sum(X * y))
        global_denom = float(np.sum(X * X))
        global_beta = global_numer / (global_denom + float(config.ridge) + 1e-12)

        beta = numer / (denom + float(config.ridge) + 1e-12)
        beta = np.where(np.isfinite(beta) & (denom > 1e-12), beta, global_beta)

    elif config.beta_mode == "global":
        numer = float(np.sum(X * y))
        denom = float(np.sum(X * X))
        global_beta = numer / (denom + float(config.ridge) + 1e-12)
        beta = np.full(X.shape[1], global_beta, dtype=float)

    else:
        raise ValueError("beta_mode must be 'node' or 'global'.")

    return np.clip(beta, float(config.beta_min), float(config.beta_max)).astype(float)


def fit_fixed_sir(
    train_view: Dict[str, object],
    population: np.ndarray,
    config: FixedSIRConfig,
) -> FixedSIRParams:
    """Calibrate beta once from the training split only."""
    train = extract_split_arrays(train_view)

    beta = _estimate_beta_from_train_origins(
        train=train,
        population=population,
        config=config,
    )

    return FixedSIRParams(
        beta=beta,
        gamma=float(config.gamma_fixed),
        beta_mode=str(config.beta_mode),
    )


def predict_fixed_sir_split(
    split_view: Dict[str, object],
    population: np.ndarray,
    params: FixedSIRParams,
    config: FixedSIRConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecast one split with fixed beta/gamma and rolling state initialization.

    For each forecast origin:
      - use only its observed history window to reconstruct S/I/R on the last
        observed day;
      - keep the TRAIN-calibrated beta_i and fixed gamma unchanged;
      - propagate the standard SIR equations for K future days.

    Returns
    -------
    y_pred_nwk, y_true_nwk : [N, W, K]
        Same layout used by MP-STGODE metric computation.
    """
    split = extract_split_arrays(split_view)
    hist = nonnegative(split["hist_incidence"])
    future = nonnegative(split["future_incidence"])
    S_last = _get_last_observed_susceptible(split)

    S, I, R = _reconstruct_state_at_last_observed_day(
        hist_incidence=hist,
        S_last=S_last,
        population=population,
        gamma=params.gamma,
    )

    W, N, _ = hist.shape
    K = future.shape[2]
    pop = np.asarray(population, dtype=float).reshape(1, N)
    beta = np.asarray(params.beta, dtype=float).reshape(1, N)

    if beta.shape[1] != N:
        raise ValueError(f"beta length {beta.shape[1]} does not match number of nodes {N}")

    y_pred = np.zeros((W, N, K), dtype=float)

    for k in range(K):
        # Standard-incidence SIR transition for the next day.
        new_infections = beta * S * I / (pop + 1e-12)
        new_infections = np.minimum(nonnegative(new_infections), np.maximum(S, 0.0))

        recoveries = float(params.gamma) * I
        recoveries = np.minimum(nonnegative(recoveries), np.maximum(I + new_infections, 0.0))

        y_pred[:, :, k] = new_infections

        S = np.maximum(S - new_infections, 0.0)
        I = np.maximum(I + new_infections - recoveries, 0.0)
        R = np.maximum(R + recoveries, 0.0)

    return ypred_wnk_to_nwk(y_pred), ytrue_wnk_to_nwk(future)


def evaluate_fixed_sir(
    data_loader: Dict[str, object],
    population: np.ndarray,
    config: FixedSIRConfig,
):
    """
    Calibrate once on TRAIN, then evaluate train/validate/test without refitting.

    The same calibrated beta vector is used for every split.  Validation/test
    histories are used only for current-state reconstruction at each forecast
    origin; their future targets never enter parameter calibration.
    """
    if "baseline_views" not in data_loader:
        raise KeyError("data_loader must contain 'baseline_views'.")

    baseline_views = data_loader["baseline_views"]
    params = fit_fixed_sir(baseline_views["train"], population, config)

    outputs = {}
    for split_name in ["train", "validate", "test"]:
        y_pred, y_true = predict_fixed_sir_split(
            split_view=baseline_views[split_name],
            population=population,
            params=params,
            config=config,
        )
        outputs[split_name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "metrics": compute_metrics(y_true, y_pred),
        }

    return params, outputs
