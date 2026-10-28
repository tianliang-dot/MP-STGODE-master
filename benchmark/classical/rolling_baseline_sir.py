# -*- coding: utf-8 -*-
"""
rolling_baseline_sir.py

Rolling fit-and-forecast SIR baseline for the additional analysis requested
by Reviewer Comment 2.

Design principles
-----------------
1. The spatial task is unchanged: N independent state-level SIR models.
2. The data split, forecast origins, history window, prediction horizon,
   target variable, and metric implementation are inherited from the same
   baseline_views used by the fixed-calibration baselines.
3. gamma is fixed, exactly as in the main fixed SIR baseline.
4. beta_i is re-estimated at every forecast origin using only observations
   that are already available before that forecast origin.
5. No observation inside the future prediction horizon is used to estimate
   beta_i or to initialize the current state.
6. The output shape is [N, W, K], matching the fixed SIR and MP-STGODE
   evaluation code.

The rolling procedure is intended as an additional operational analysis.
It does not replace the fixed-calibration SIR in the main benchmark table.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from fixed_baseline_utils import (
    compute_metrics,
    extract_split_arrays,
    nonnegative,
    ypred_wnk_to_nwk,
    ytrue_wnk_to_nwk,
)


@dataclass
class RollingSIRConfig:
    """Configuration for the rolling SIR baseline."""

    # Keep gamma identical to the main fixed-calibration SIR unless the paper
    # explicitly changes the common epidemiological assumption.
    gamma_fixed: float = 0.10

    # "node" gives one beta_i for each state; "global" gives one shared beta.
    beta_mode: str = "node"

    # Numerical bounds for the least-squares estimate.
    beta_min: float = 1e-8
    beta_max: float = 1.0
    ridge: float = 1e-8


@dataclass
class RollingSIRDiagnostics:
    """Quantities saved for checking the rolling experiment."""

    beta_by_origin: np.ndarray   # [W, N]
    target_start: np.ndarray     # [W]
    gamma: float
    beta_mode: str


def _write_day(
    array_nt: np.ndarray,
    day: int,
    values_n: np.ndarray,
) -> None:
    """
    Write one calendar day into an [N,T] array.

    The baseline views contain overlapping windows. Therefore the same calendar
    day can appear more than once. All duplicates originate from the same raw
    data, so later writes simply overwrite the same values.
    """
    if day < 0 or day >= array_nt.shape[1]:
        raise IndexError(f"day={day} outside reconstructed range [0,{array_nt.shape[1]})")
    array_nt[:, day] = np.asarray(values_n, dtype=float)


def _reconstruct_observed_series(
    data_loader: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the chronological incidence and susceptible series from the
    already-built baseline views.

    Returns
    -------
    incidence_nt : [N,T]
        Daily observed new infections.
    susceptible_nt : [N,T]
        Daily susceptible series whenever available.

    Notes
    -----
    This helper does not create new information. It only reorganizes the
    overlapping train/validation/test windows that were already constructed
    from the common input data by run_fixed_baselines.py.
    """
    views = data_loader["baseline_views"]

    first_view = next(iter(views.values()))
    first = extract_split_arrays(first_view)
    N = first["hist_incidence"].shape[1]

    max_day = 0
    for view in views.values():
        split = extract_split_arrays(view)
        H = split["hist_incidence"].shape[2]
        K = split["future_incidence"].shape[2]

        if "window_start" in split:
            max_day = max(max_day, int(np.max(split["window_start"])) + H)
        if "target_start" in split:
            max_day = max(max_day, int(np.max(split["target_start"])) + K)

    incidence_nt = np.full((N, max_day), np.nan, dtype=float)
    susceptible_nt = np.full((N, max_day), np.nan, dtype=float)

    for view in views.values():
        split = extract_split_arrays(view)

        hist = nonnegative(split["hist_incidence"])      # [W,N,H]
        fut = nonnegative(split["future_incidence"])     # [W,N,K]
        W, _, H = hist.shape
        K = fut.shape[2]

        if "window_start" not in split or "target_start" not in split:
            raise KeyError(
                "Rolling SIR requires window_start and target_start in each baseline view."
            )

        window_start = split["window_start"].astype(int).reshape(-1)
        target_start = split["target_start"].astype(int).reshape(-1)

        S_hist = split.get("S_hist", None)
        S_origin = split.get("S0_forecast_origin", None)

        for w in range(W):
            for h in range(H):
                day = int(window_start[w] + h)
                _write_day(incidence_nt, day, hist[w, :, h])
                if S_hist is not None:
                    _write_day(susceptible_nt, day, S_hist[w, :, h])

            for k in range(K):
                day = int(target_start[w] + k)
                _write_day(incidence_nt, day, fut[w, :, k])

            # S at target_start is kept only to reconstruct the chronological
            # S series. It is NOT used to initialize the rolling forecast,
            # because the first future day has not yet been observed.
            if S_origin is not None:
                _write_day(susceptible_nt, int(target_start[w]), S_origin[w])

    return incidence_nt, susceptible_nt


def _active_infected_at_day(
    incidence_nt: np.ndarray,
    day: int,
    history_window: int,
    gamma: float,
) -> np.ndarray:
    """
    Reconstruct I(day) from observed incidence ending on `day`.

    The same history length used by the forecasting experiment is used here:
        I_i(day) = sum_{ell=0}^{H-1}
                   C_i(day-ell) * (1-gamma)^ell
    """
    H = int(history_window)
    start = day - H + 1
    if start < 0:
        raise ValueError(
            f"Not enough history to reconstruct I at day={day} with H={H}."
        )

    recent = nonnegative(incidence_nt[:, start:day + 1])  # [N,H]
    if recent.shape[1] != H or np.any(~np.isfinite(recent)):
        raise ValueError(
            f"Incomplete incidence history for day={day}, H={H}."
        )

    # recent is ordered from oldest -> newest.
    weights = (1.0 - float(gamma)) ** np.arange(H - 1, -1, -1, dtype=float)
    I = np.sum(recent * weights[None, :], axis=1)
    return np.maximum(I, 1e-8)


def _fit_beta_expanding(
    incidence_nt: np.ndarray,
    susceptible_nt: np.ndarray,
    population: np.ndarray,
    forecast_start: int,
    history_window: int,
    config: RollingSIRConfig,
) -> np.ndarray:
    """
    Re-estimate beta at one forecast origin using all observations available
    before that origin.

    If the prediction horizon starts at calendar day T, the latest observable
    incidence is day T-1. Calibration pairs therefore satisfy

        state at day d  ->  incidence at day d+1,

    with d+1 <= T-1.

    Thus neither incidence at day T nor any later target is used.

    The least-squares relation is

        C_i(d+1) ~= beta_i * S_i(d) * I_i(d) / N_i.
    """
    pop = np.asarray(population, dtype=float).reshape(-1)
    N = pop.shape[0]
    H = int(history_window)
    T = int(forecast_start)

    # Earliest day d for which H days of incidence are available.
    first_state_day = H - 1

    # Latest calibration target is T-1, so latest state day is T-2.
    last_state_day = T - 2

    if last_state_day < first_state_day:
        raise ValueError(
            f"Not enough observations to fit rolling beta at forecast_start={T}."
        )

    numer = np.zeros(N, dtype=float)
    denom = np.zeros(N, dtype=float)

    global_numer = 0.0
    global_denom = 0.0
    num_pairs = 0

    for day in range(first_state_day, last_state_day + 1):
        next_day = day + 1

        S = susceptible_nt[:, day]
        y = incidence_nt[:, next_day]

        if np.any(~np.isfinite(S)) or np.any(~np.isfinite(y)):
            continue

        I = _active_infected_at_day(
            incidence_nt=incidence_nt,
            day=day,
            history_window=H,
            gamma=config.gamma_fixed,
        )

        S = np.clip(nonnegative(S), 0.0, pop)
        X = S * I / (pop + 1e-12)
        y = nonnegative(y)

        numer += X * y
        denom += X * X

        global_numer += float(np.sum(X * y))
        global_denom += float(np.sum(X * X))
        num_pairs += 1

    if num_pairs == 0:
        raise ValueError(
            f"No valid calibration pairs for forecast_start={T}."
        )

    if config.beta_mode == "global":
        b = global_numer / (global_denom + float(config.ridge) + 1e-12)
        beta = np.full(N, b, dtype=float)

    elif config.beta_mode == "node":
        global_b = global_numer / (global_denom + float(config.ridge) + 1e-12)

        beta = numer / (denom + float(config.ridge) + 1e-12)

        # If one state has no usable variation, fall back to the contemporaneous
        # global estimate rather than introducing another tuning rule.
        valid = np.isfinite(beta) & (denom > 1e-12)
        beta = np.where(valid, beta, global_b)

    else:
        raise ValueError("beta_mode must be 'global' or 'node'.")

    return np.clip(beta, config.beta_min, config.beta_max).astype(float)


def _state_before_forecast(
    incidence_nt: np.ndarray,
    susceptible_nt: np.ndarray,
    population: np.ndarray,
    forecast_start: int,
    history_window: int,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct S/I/R on the last observed day before the prediction horizon.

    If the first predicted day is T, the current state is constructed at T-1.
    This uses only information already available when the forecast is issued.
    """
    T = int(forecast_start)
    last_observed_day = T - 1

    pop = np.asarray(population, dtype=float).reshape(-1)
    S = susceptible_nt[:, last_observed_day]

    if np.any(~np.isfinite(S)):
        raise ValueError(
            f"Missing susceptible values at last observed day={last_observed_day}."
        )

    S = np.clip(nonnegative(S), 0.0, pop)

    I = _active_infected_at_day(
        incidence_nt=incidence_nt,
        day=last_observed_day,
        history_window=history_window,
        gamma=gamma,
    )
    I = np.minimum(I, np.maximum(pop - S, 0.0))

    R = np.maximum(pop - S - I, 0.0)
    return S, I, R


def predict_rolling_sir_split(
    split_view: Dict[str, object],
    data_loader: Dict[str, object],
    population: np.ndarray,
    config: RollingSIRConfig,
) -> Tuple[np.ndarray, np.ndarray, RollingSIRDiagnostics]:
    """
    Rolling fit-and-forecast prediction for one split.

    For each forecast origin:
      1) re-estimate beta using all observations available before that origin;
      2) reconstruct the current S/I/R state from the last observed day;
      3) recursively forecast the next K days with beta and gamma fixed over
         that short forecast horizon.

    Returns
    -------
    y_pred_nwk : [N,W,K]
    y_true_nwk : [N,W,K]
    diagnostics : RollingSIRDiagnostics
    """
    split = extract_split_arrays(split_view)

    hist = nonnegative(split["hist_incidence"])
    fut = nonnegative(split["future_incidence"])
    W, N, H = hist.shape
    K = fut.shape[2]

    if "target_start" not in split:
        raise KeyError("Rolling SIR requires target_start in split_view.")

    target_start = split["target_start"].astype(int).reshape(-1)

    pop = np.asarray(population, dtype=float).reshape(-1)
    if pop.shape[0] != N:
        raise ValueError(f"population has N={pop.shape[0]}, but split has N={N}.")

    incidence_nt, susceptible_nt = _reconstruct_observed_series(data_loader)

    y_pred = np.zeros((W, N, K), dtype=float)
    beta_by_origin = np.zeros((W, N), dtype=float)

    for w in range(W):
        T = int(target_start[w])

        beta = _fit_beta_expanding(
            incidence_nt=incidence_nt,
            susceptible_nt=susceptible_nt,
            population=pop,
            forecast_start=T,
            history_window=H,
            config=config,
        )
        beta_by_origin[w] = beta

        S, I, R = _state_before_forecast(
            incidence_nt=incidence_nt,
            susceptible_nt=susceptible_nt,
            population=pop,
            forecast_start=T,
            history_window=H,
            gamma=config.gamma_fixed,
        )

        # beta is updated once at the current forecast origin, then held fixed
        # over the requested 3-day or 7-day horizon.
        for k in range(K):
            newI = beta * S * I / (pop + 1e-12)
            newI = np.minimum(nonnegative(newI), np.maximum(S, 0.0))

            recov = config.gamma_fixed * I
            recov = np.minimum(nonnegative(recov), np.maximum(I + newI, 0.0))

            y_pred[w, :, k] = newI

            S = np.maximum(S - newI, 0.0)
            I = np.maximum(I + newI - recov, 0.0)
            R = np.maximum(R + recov, 0.0)

    diagnostics = RollingSIRDiagnostics(
        beta_by_origin=beta_by_origin,
        target_start=target_start.copy(),
        gamma=float(config.gamma_fixed),
        beta_mode=str(config.beta_mode),
    )

    return (
        ypred_wnk_to_nwk(y_pred),
        ytrue_wnk_to_nwk(fut),
        diagnostics,
    )


def evaluate_rolling_sir(
    data_loader: Dict[str, object],
    population: np.ndarray,
    config: RollingSIRConfig,
    split_name: str = "test",
):
    """
    Evaluate the rolling SIR on the requested split.

    The intended use for Reviewer Comment 2 is split_name="test".
    The returned prediction arrays and metric definitions are exactly aligned
    with the fixed SIR and MP-STGODE evaluation pipeline.
    """
    if split_name not in data_loader["baseline_views"]:
        raise KeyError(f"Unknown split_name={split_name!r}.")

    y_pred, y_true, diagnostics = predict_rolling_sir_split(
        split_view=data_loader["baseline_views"][split_name],
        data_loader=data_loader,
        population=population,
        config=config,
    )

    outputs = {
        split_name: {
            "y_true": y_true,
            "y_pred": y_pred,
            "metrics": compute_metrics(y_true, y_pred),
        }
    }

    return diagnostics, outputs
