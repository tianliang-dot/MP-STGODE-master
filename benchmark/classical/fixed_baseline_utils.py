# -*- coding: utf-8 -*-
"""
fixed_baseline_utils.py

Utilities for fixed-parameter epidemic forecasting baselines.

All baselines in this package follow the same evaluation principle as the
MP-STGODE test protocol:
    1) train split: estimate model parameters;
    2) validation split: select hyperparameters only;
    3) test split: fixed-parameter rolling-origin inference only;
    4) no test-period refitting, recalibration, or target leakage.
"""

import json
import os
import pickle
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Basic I/O
# -----------------------------------------------------------------------------

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    def convert(v: Any) -> Any:
        if is_dataclass(v):
            return asdict(v)
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): convert(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(val) for val in v]
        return v

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2, ensure_ascii=False)


def save_config(outdir: str, params: Dict[str, Any]) -> None:
    ensure_dir(outdir)
    save_json(os.path.join(outdir, "run_config.json"), params)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def get_population(data_loader: Dict[str, Any], input_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Read population vector from either the raw data dict or the data_loader."""
    sources = [data_loader, input_data or {}]

    for src in sources:
        for key in ["population", "Population", "pop", "N_pop"]:
            if key in src and src[key] is not None:
                pop = to_numpy(src[key]).astype(float)
                return pop.reshape(-1)

    for src in sources:
        for key in ["static_feat", "static_feats", "static_feature"]:
            if key in src and src[key] is not None:
                arr = to_numpy(src[key]).astype(float)
                if arr.ndim == 1:
                    return arr.reshape(-1)
                return arr[:, 0].reshape(-1)

    raise KeyError(
        "Cannot find population. Expected population/Population or static_feat[:,0]. "
        f"data_loader keys={list(data_loader.keys())}; input_data keys={list((input_data or {}).keys())}"
    )


def get_loc_list(data_loader: Dict[str, Any], input_data: Optional[Dict[str, Any]] = None, num_nodes: int = 0) -> List[str]:
    for src in [data_loader, input_data or {}]:
        for key in ["loc_list", "locations", "state_names", "states"]:
            if key in src and src[key] is not None:
                return [str(x) for x in list(src[key])]
    return [f"State_{i}" for i in range(num_nodes)]


def parse_float_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_int_grid(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_str_grid(s: str, allowed: Iterable[str], name: str) -> List[str]:
    allowed_set = set(allowed)
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    if not items:
        raise ValueError(f"{name} cannot be empty.")
    bad = [x for x in items if x not in allowed_set]
    if bad:
        raise ValueError(f"Invalid {name}: {bad}. Allowed: {sorted(allowed_set)}")
    return items


def nonnegative(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.asarray(a, dtype=float) / (np.asarray(b, dtype=float) + eps)


# -----------------------------------------------------------------------------
# State reconstruction
# -----------------------------------------------------------------------------

def active_infected_from_history(hist_incidence: np.ndarray, gamma: float, max_days: Optional[int] = None) -> np.ndarray:
    """
    Reconstruct active infected at the forecast origin from observed historical incidence.

    hist_incidence: [W,N,H] or [N,H]; last axis is time and the last element is the most recent observed day.
    I(t0) = sum_{ell=0}^{L-1} C(t0-1-ell) * (1-gamma)^ell.
    """
    hist = nonnegative(hist_incidence)
    H = hist.shape[-1]
    L = H if max_days is None else min(H, int(max_days))
    recent = hist[..., -L:]
    weights = (1.0 - float(gamma)) ** np.arange(L - 1, -1, -1, dtype=float)
    return np.sum(recent * weights, axis=-1)


def reconstruct_state_at_forecast_origin(
    hist_incidence: np.ndarray,
    S_forecast_origin: np.ndarray,
    population: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build S/I/R at the forecast origin using only the historical input window.

    This function intentionally uses S at the forecast origin. In the main MP-STGODE
    mechanism branch, the metapopulation rollout also starts from the forecast origin,
    so the fixed SIR and Meta-SIR baselines use the same state boundary.
    """
    S = nonnegative(S_forecast_origin)
    pop = np.asarray(population, dtype=float).reshape(-1)
    I = active_infected_from_history(hist_incidence, gamma=gamma)

    while pop.ndim < S.ndim:
        pop = pop[None, :]
    S = np.minimum(S, pop)
    I = np.minimum(nonnegative(I), np.maximum(pop - S, 0.0))
    R = np.maximum(pop - S - I, 0.0)
    return S, I, R


def extract_split_arrays(split_view: Dict[str, Any]) -> Dict[str, np.ndarray]:
    keys = [
        "hist_incidence",
        "future_incidence",
        "S0_forecast_origin",
        "S0_history_start",
        "S0_t0",
        "S_hist",
        "adj_hist",
        "window_start",
        "target_start",
    ]
    out = {}
    for k in keys:
        if k in split_view:
            out[k] = to_numpy(split_view[k]).astype(float)
    if "hist_incidence" not in out or "future_incidence" not in out:
        raise KeyError("split_view must contain hist_incidence and future_incidence.")
    if "S0_forecast_origin" not in out:
        if "S0_t0" in out:
            out["S0_forecast_origin"] = out["S0_t0"]
        else:
            raise KeyError("split_view must contain S0_forecast_origin or S0_t0.")
    return out


def ytrue_wnk_to_nwk(y_true_wnk: np.ndarray) -> np.ndarray:
    return np.asarray(y_true_wnk, dtype=float).transpose(1, 0, 2).copy()


def ypred_wnk_to_nwk(y_pred_wnk: np.ndarray) -> np.ndarray:
    return np.asarray(y_pred_wnk, dtype=float).transpose(1, 0, 2).copy()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    yt = np.nan_to_num(np.asarray(y_true, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    yp = np.nan_to_num(np.asarray(y_pred, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    diff = yp - yt
    abs_diff = np.abs(diff)

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(abs_diff))
    mask = np.abs(yt) > eps
    mape = float(np.mean(abs_diff[mask] / (np.abs(yt[mask]) + eps)) * 100.0) if np.any(mask) else float("nan")
    wmape = float(np.sum(abs_diff) / (np.sum(np.abs(yt)) + eps) * 100.0)
    smape = float(np.mean(2.0 * abs_diff / (np.abs(yt) + np.abs(yp) + eps)) * 100.0)
    rae = float(np.sum(abs_diff) / (np.sum(np.abs(yt - np.mean(yt))) + eps))
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "wMAPE%": wmape, "sMAPE%": smape, "RAE": rae}


def compute_stepwise_metrics(y_true_nwk: np.ndarray, y_pred_nwk: np.ndarray) -> pd.DataFrame:
    yt = np.asarray(y_true_nwk, dtype=float)
    yp = np.asarray(y_pred_nwk, dtype=float)
    K = yt.shape[2]
    rows = []
    for k in range(K):
        row = {"step": k + 1}
        row.update(compute_metrics(yt[:, :, k], yp[:, :, k]))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_state_metrics(y_true_nwk: np.ndarray, y_pred_nwk: np.ndarray, loc_list: Optional[List[str]] = None) -> pd.DataFrame:
    yt = np.asarray(y_true_nwk, dtype=float)
    yp = np.asarray(y_pred_nwk, dtype=float)
    N = yt.shape[0]
    rows = []
    for i in range(N):
        row = {"state_id": i, "state_name": loc_list[i] if loc_list and i < len(loc_list) else f"State_{i}"}
        row["mean_true_incidence"] = float(np.mean(yt[i]))
        row.update(compute_metrics(yt[i], yp[i]))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_incidence_stratified_metrics(
    y_true_nwk: np.ndarray,
    y_pred_nwk: np.ndarray,
    loc_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Group nodes into low/middle/high incidence tertiles using their mean true incidence."""
    yt = np.asarray(y_true_nwk, dtype=float)
    yp = np.asarray(y_pred_nwk, dtype=float)
    node_mean = yt.mean(axis=(1, 2))
    q1, q2 = np.quantile(node_mean, [1 / 3, 2 / 3])
    groups = {
        "low_incidence": np.where(node_mean <= q1)[0],
        "middle_incidence": np.where((node_mean > q1) & (node_mean <= q2))[0],
        "high_incidence": np.where(node_mean > q2)[0],
    }
    rows = []
    for name, idx in groups.items():
        if len(idx) == 0:
            continue
        row = {"group": name, "num_states": int(len(idx)), "mean_true_incidence": float(node_mean[idx].mean())}
        if loc_list:
            row["states"] = ";".join([loc_list[i] if i < len(loc_list) else f"State_{i}" for i in idx])
        row.update(compute_metrics(yt[idx], yp[idx]))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_metrics(
    y_true_nwk: np.ndarray,
    y_pred_nwk: np.ndarray,
    n_bootstrap: int = 30,
    seed: int = 42,
    unit: str = "window",
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true_nwk, dtype=float)
    yp = np.asarray(y_pred_nwk, dtype=float)
    N, W, K = yt.shape
    records = []
    if unit == "state":
        for _ in range(n_bootstrap):
            idx = rng.integers(0, N, size=N)
            records.append(compute_metrics(yt[idx], yp[idx]))
    else:
        yt_units = yt.reshape(N * W, K)
        yp_units = yp.reshape(N * W, K)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, N * W, size=N * W)
            records.append(compute_metrics(yt_units[idx], yp_units[idx]))
    df = pd.DataFrame(records)
    return df.mean().to_dict(), df.std(ddof=1).to_dict(), df


def metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Metric {metric_name!r} not found. Available: {list(metrics.keys())}")
    val = float(metrics[metric_name])
    return val if np.isfinite(val) else float("inf")


def save_prediction_bundle(
    outdir: str,
    model_name: str,
    split: str,
    y_true_nwk: np.ndarray,
    y_pred_nwk: np.ndarray,
    loc_list: Optional[List[str]] = None,
    n_bootstrap: int = 30,
) -> Dict[str, float]:
    ensure_dir(outdir)
    tag = f"{model_name}_{split}"
    np.save(os.path.join(outdir, f"{tag}_y_true_nwk.npy"), np.asarray(y_true_nwk, dtype=float))
    np.save(os.path.join(outdir, f"{tag}_y_pred_nwk.npy"), np.asarray(y_pred_nwk, dtype=float))

    metrics = compute_metrics(y_true_nwk, y_pred_nwk)
    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, f"{tag}_metrics.csv"), index=False)
    compute_stepwise_metrics(y_true_nwk, y_pred_nwk).to_csv(os.path.join(outdir, f"{tag}_stepwise_metrics.csv"), index=False)
    compute_state_metrics(y_true_nwk, y_pred_nwk, loc_list).to_csv(os.path.join(outdir, f"{tag}_state_metrics.csv"), index=False)
    compute_incidence_stratified_metrics(y_true_nwk, y_pred_nwk, loc_list).to_csv(
        os.path.join(outdir, f"{tag}_incidence_stratified_metrics.csv"), index=False
    )

    if n_bootstrap and n_bootstrap > 0:
        mean_m, std_m, df_boot = bootstrap_metrics(y_true_nwk, y_pred_nwk, n_bootstrap=n_bootstrap)
        df_boot.to_csv(os.path.join(outdir, f"{tag}_bootstrap_runs.csv"), index=False)
        boot_summary = {f"{k}_mean": v for k, v in mean_m.items()}
        boot_summary.update({f"{k}_std": v for k, v in std_m.items()})
        pd.DataFrame([boot_summary]).to_csv(os.path.join(outdir, f"{tag}_bootstrap_summary.csv"), index=False)
    return metrics
