# -*- coding: utf-8 -*-
"""
run_fixed_baselines.py

Entry script for the revised fixed-parameter baselines.

This script intentionally does NOT implement daily rolling recalibration. It uses
exactly the same high-level test principle as MP-STGODE:
    train once / calibrate once -> select hyperparameters on validation ->
    fixed-parameter rolling-origin inference on the held-out test windows.

Recommended main baselines:
    - SIR_fixed_calibration
    - MetaSIR_fixed_calibration
    - AR_train_once_classical
    - RidgeVAR_train_once
Optional stronger reference:
    - RidgeAR_train_once_optional
"""

import os
from pathlib import Path
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fixed_baseline_utils import (
    ensure_dir,
    get_loc_list,
    get_population,
    load_pickle,
    metric_value,
    parse_float_grid,
    parse_str_grid,
    save_config,
    save_prediction_bundle,
    to_numpy,
    compute_metrics,
)
from fixed_baseline_sir import (
    FixedSIRConfig,
    evaluate_fixed_sir,
    fit_fixed_sir,
    predict_fixed_sir_split
)
from fixed_baseline_metasir import (
    FixedMetaSIRConfig,
    evaluate_fixed_metasir,
    fit_fixed_metasir,
    predict_fixed_metasir_split,
)
from rolling_baseline_sir import (
    RollingSIRConfig,
    evaluate_rolling_sir,
)
from fixed_baseline_ar import FixedARConfig, evaluate_fixed_ar
from fixed_baseline_var import FixedVARConfig, evaluate_fixed_var


# -----------------------------------------------------------------------------
# Raw data parsing and split construction
# -----------------------------------------------------------------------------

def _normalize_input_keys(input_data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(input_data)
    if "input_feat" not in data:
        if "input_data" in data:
            data["input_feat"] = data["input_data"]
        else:
            raise KeyError("Cannot find input_feat or input_data in the data pickle.")

    if "adj_matrix" not in data:
        for key in ["adj", "od", "mobility", "M"]:
            if key in data:
                data["adj_matrix"] = data[key]
                break
    if "adj_matrix" not in data:
        raise KeyError("Cannot find adj_matrix / adj / od / mobility / M in the data pickle.")

    x = to_numpy(data["input_feat"])
    if x.ndim != 3:
        raise ValueError(f"input_feat should be [N,T,F], got {x.shape}")
    if x.shape[-1] < 4:
        raise ValueError(f"input_feat needs at least 4 features, got {x.shape[-1]}")
    data["input_feat"] = x[..., :4]

    A = to_numpy(data["adj_matrix"])
    if A.ndim == 4 and A.shape[-1] == 1:
        A = A[..., 0]
    if A.ndim != 3:
        raise ValueError(f"adj_matrix should be [T,N,N] or [T,N,N,1], got {A.shape}")
    data["adj_matrix"] = A
    return data


def _split_bounds(total_time: int, split_ratio: List[float], history_window: int) -> Dict[str, Tuple[int, int]]:
    r = list(split_ratio)
    train_len = int(r[0] / sum(r) * total_time)
    val_len = int(r[1] / sum(r) * total_time)
    val_end = train_len + val_len
    return {
        "train": (0, train_len),
        "validate": (train_len - history_window, val_end),
        "test": (val_end - history_window, total_time),
    }


def _build_split_view(
    x_ntf: np.ndarray,
    A_tnn: np.ndarray,
    split_start: int,
    split_end: int,
    history_window: int,
    pred_window: int,
    slide_step: int,
    newI_idx: int,
    S_idx: int,
) -> Dict[str, np.ndarray]:
    x_split = np.asarray(x_ntf[:, split_start:split_end, :], dtype=float)
    A_split = np.asarray(A_tnn[split_start:split_end], dtype=float)
    N, T_split, _ = x_split.shape
    H = int(history_window)
    K = int(pred_window)

    hist_list, fut_list, S_start_list, S_origin_list, S_hist_list = [], [], [], [], []
    adj_hist_list, window_start_list, target_start_list = [], [], []
    for t in range(0, T_split - H - K + 1, int(slide_step)):
        hist_list.append(x_split[:, t:t + H, newI_idx])
        fut_list.append(x_split[:, t + H:t + H + K, newI_idx])
        S_start_list.append(x_split[:, t, S_idx])
        S_origin_list.append(x_split[:, t + H, S_idx])
        S_hist_list.append(x_split[:, t:t + H, S_idx])
        adj_hist_list.append(A_split[t:t + H])
        window_start_list.append(split_start + t)
        target_start_list.append(split_start + t + H)

    if not hist_list:
        raise ValueError(
            f"No windows for split [{split_start},{split_end}). Need length >= history_window + pred_window."
        )

    return {
        "hist_incidence": np.stack(hist_list, axis=0),
        "future_incidence": np.stack(fut_list, axis=0),
        "S0_history_start": np.stack(S_start_list, axis=0),
        "S0_forecast_origin": np.stack(S_origin_list, axis=0),
        "S0_t0": np.stack(S_origin_list, axis=0),  # kept for backward compatibility; fixed to forecast origin
        "S_hist": np.stack(S_hist_list, axis=0),
        "adj_hist": np.stack(adj_hist_list, axis=0),
        "window_start": np.asarray(window_start_list, dtype=int),
        "target_start": np.asarray(target_start_list, dtype=int),
    }


def build_baseline_views(input_data: Dict[str, Any], params) -> Dict[str, Dict[str, np.ndarray]]:
    x = np.asarray(input_data["input_feat"], dtype=float)
    A = np.asarray(input_data["adj_matrix"], dtype=float)
    N, T, F = x.shape
    if A.shape[0] != T:
        raise ValueError(f"Time mismatch: input_feat T={T}, adj_matrix T={A.shape[0]}")
    if A.shape[1] != N or A.shape[2] != N:
        raise ValueError(f"Node mismatch: input_feat N={N}, adj_matrix shape={A.shape}")

    bounds = _split_bounds(T, params.split_ratio, params.history_window)
    views = {}
    for split, (a, b) in bounds.items():
        views[split] = _build_split_view(
            x_ntf=x,
            A_tnn=A,
            split_start=a,
            split_end=b,
            history_window=params.history_window,
            pred_window=params.pred_window,
            slide_step=params.slide_step,
            newI_idx=params.newI_idx,
            S_idx=params.S_idx,
        )
        h = views[split]["hist_incidence"]
        f = views[split]["future_incidence"]
        print(f"[baseline_views] {split}: raw_range=[{a},{b}), hist={h.shape}, future={f.shape}")
    return views


def build_data_loader(params):
    input_data = _normalize_input_keys(load_pickle(params.data_pkl))
    baseline_views = build_baseline_views(input_data, params)
    data_loader = {"baseline_views": baseline_views}
    for key in ["Population", "population", "static_feat", "static_feats", "loc_list", "locations", "state_names"]:
        if key in input_data:
            data_loader[key] = input_data[key]
    return input_data, data_loader


# -----------------------------------------------------------------------------
# Validation selection
# -----------------------------------------------------------------------------

def _small_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if k in ["RMSE", "MAE", "MAPE%", "wMAPE%", "sMAPE%", "RAE"]}


def select_sir_config(data_loader, population, params):
    if not int(params.sir_tune_validation):
        cfg = FixedSIRConfig(
            gamma_fixed=params.sir_gamma_fixed,
            beta_mode=params.sir_beta_mode,
            beta_max=params.sir_beta_max,
        )
        return cfg, {"enabled": False, "selected_config": asdict(cfg)}

    records = []
    best_score = float("inf")
    best_cfg = None
    best_record = None
    for gamma in parse_float_grid(params.sir_gamma_grid):
        for beta_mode in parse_str_grid(params.sir_beta_modes, ["global", "node"], "sir_beta_modes"):
            cfg = FixedSIRConfig(
                gamma_fixed=float(gamma),
                beta_mode=beta_mode,
                beta_max=params.sir_beta_max,
            )
            model_params = fit_fixed_sir(data_loader["baseline_views"]["train"], population, cfg)
            y_pred_val, y_true_val = predict_fixed_sir_split(data_loader["baseline_views"]["validate"], population, model_params, cfg)
            metrics = compute_metrics(y_true_val, y_pred_val)
            score = metric_value(metrics, params.sir_tune_metric)
            record = {
                "gamma_fixed": float(gamma),
                "beta_mode": beta_mode,
                "beta_mean": float(np.mean(model_params.beta)),
                "score_metric": params.sir_tune_metric,
                "score": float(score),
            }
            record.update({f"validate_{k}": v for k, v in _small_metrics(metrics).items()})
            records.append(record)
            if score < best_score:
                best_score = score
                best_cfg = cfg
                best_record = record
    print(f"[SIR tuning] selected {asdict(best_cfg)} with validation {params.sir_tune_metric}={best_score:.4f}")
    return best_cfg, {"enabled": True, "selected": best_record, "candidates": records}


def select_metasir_config(data_loader, population, params):
    if not int(params.metasir_tune_validation):
        cfg = FixedMetaSIRConfig(
            gamma_fixed=params.metasir_gamma_fixed,
            beta_mode=params.metasir_beta_mode,
            beta_max=params.metasir_beta_max,
            mobility_scale=params.metasir_mobility_scale,
            use_mobility_binary_mask=params.metasir_use_mobility_binary_mask,
            future_mobility_mode=params.metasir_future_mobility_mode,
            future_recent_k=params.metasir_future_recent_k,
            future_week_period=params.metasir_future_week_period,
        )
        return cfg, {"enabled": False, "selected_config": asdict(cfg)}

    gamma_grid = parse_float_grid(params.metasir_gamma_grid)
    beta_modes = parse_str_grid(params.metasir_beta_modes, ["global", "node"], "metasir_beta_modes")
    scale_grid = parse_float_grid(params.metasir_mobility_scale_grid)
    future_modes = parse_str_grid(
        params.metasir_future_mobility_modes,
        ["last_repeat", "recent_mean", "weekday_cycle", "recent_cycle", "mean_repeat"],
        "metasir_future_mobility_modes",
    )

    records = []
    best_score = float("inf")
    best_cfg = None
    best_record = None
    for gamma in gamma_grid:
        for beta_mode in beta_modes:
            for scale in scale_grid:
                for future_mode in future_modes:
                    cfg = FixedMetaSIRConfig(
                        gamma_fixed=float(gamma),
                        beta_mode=beta_mode,
                        beta_max=params.metasir_beta_max,
                        beta_ridge=params.metasir_beta_ridge,
                        mobility_scale=float(scale),
                        use_mobility_binary_mask=params.metasir_use_mobility_binary_mask,
                        future_mobility_mode=future_mode,
                        future_recent_k=params.metasir_future_recent_k,
                        future_week_period=params.metasir_future_week_period,
                    )
                    model_params = fit_fixed_metasir(data_loader["baseline_views"]["train"], population, cfg)
                    y_pred_val, y_true_val = predict_fixed_metasir_split(
                        data_loader["baseline_views"]["validate"], population, model_params, cfg
                    )
                    metrics = compute_metrics(y_true_val, y_pred_val)
                    score = metric_value(metrics, params.metasir_tune_metric)
                    record = {
                        "gamma_fixed": float(gamma),
                        "beta_mode": beta_mode,
                        "mobility_scale": float(scale),
                        "future_mobility_mode": future_mode,
                        "beta_mean": float(np.mean(model_params.beta)),
                        "score_metric": params.metasir_tune_metric,
                        "score": float(score),
                    }
                    record.update({f"validate_{k}": v for k, v in _small_metrics(metrics).items()})
                    records.append(record)
                    if score < best_score:
                        best_score = score
                        best_cfg = cfg
                        best_record = record
    print(f"[Meta-SIR tuning] selected {asdict(best_cfg)} with validation {params.metasir_tune_metric}={best_score:.4f}")
    return best_cfg, {"enabled": True, "selected": best_record, "candidates": records}


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_outputs(base_outdir, model_name, outputs, loc_list, n_bootstrap):
    model_dir = os.path.join(base_outdir, model_name)
    ensure_dir(model_dir)
    rows = []
    for split, obj in outputs.items():
        metrics = save_prediction_bundle(
            outdir=model_dir,
            model_name=model_name,
            split=split,
            y_true_nwk=obj["y_true"],
            y_pred_nwk=obj["y_pred"],
            loc_list=loc_list,
            n_bootstrap=n_bootstrap,
        )
        row = {"model": model_name, "split": split}
        row.update(metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(model_dir, f"{model_name}_summary.csv"), index=False)
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_pkl", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "Data_meta_stgode_causal7.pkl"))
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "outputs" / "classical"))
    parser.add_argument("--history_window", type=int, default=7)
    parser.add_argument("--pred_window", type=int, default=3)
    parser.add_argument("--slide_step", type=int, default=1)
    parser.add_argument("--split_ratio", type=float, nargs="+", default=[8, 1, 1])
    parser.add_argument("--newI_idx", type=int, default=3)
    parser.add_argument("--S_idx", type=int, default=2)
    parser.add_argument("--n_bootstrap", type=int, default=1000)

    # SIR: fixed-calibration, standard incidence.
    parser.add_argument("--run_sir", type=int, default=1)
    parser.add_argument("--sir_tune_validation", type=int, default=0)
    parser.add_argument("--sir_tune_metric", type=str, default="wMAPE%")
    parser.add_argument("--sir_gamma_fixed", type=float, default=0.10)
    parser.add_argument("--sir_gamma_grid", type=str, default="0.10")
    parser.add_argument("--sir_beta_mode", type=str, choices=["global", "node"], default="node")
    parser.add_argument("--sir_beta_modes", type=str, default="node")
    parser.add_argument("--sir_beta_max", type=float, default=1.0)

    # Rolling SIR: re-estimate beta at every test forecast origin using only
    # observations available before that origin. Disabled by default so the
    # established fixed-baseline runner remains unchanged unless requested.
    parser.add_argument("--run_rolling_sir", type=int, choices=[0, 1], default=1)
    parser.add_argument("--rolling_sir_beta_min", type=float, default=1e-8)
    parser.add_argument("--rolling_sir_ridge", type=float, default=1e-8)

    # Meta-SIR: same Lagrangian metapopulation equations as MP-STGODE mechanism branch.
    parser.add_argument("--run_metasir", type=int, default=0)
    parser.add_argument("--metasir_tune_validation", type=int, default=0)
    parser.add_argument("--metasir_tune_metric", type=str, default="wMAPE%")
    parser.add_argument("--metasir_gamma_fixed", type=float, default=0.19)
    parser.add_argument("--metasir_gamma_grid", type=str, default="0.19")
    parser.add_argument("--metasir_beta_mode", type=str, choices=["global", "node"], default="node")
    parser.add_argument("--metasir_beta_modes", type=str, default="node")
    parser.add_argument("--metasir_beta_max", type=float, default=1.0)
    parser.add_argument("--metasir_beta_ridge", type=float, default=1e-4)
    parser.add_argument("--metasir_mobility_scale", type=float, default=1.0)
    parser.add_argument("--metasir_mobility_scale_grid", type=str, default="1.0")
    parser.add_argument("--metasir_use_mobility_binary_mask", type=int, default=1)
    parser.add_argument("--metasir_future_mobility_mode", type=str, default="last_repeat")
    parser.add_argument("--metasir_future_mobility_modes", type=str, default="last_repeat, recent_mean, weekday_cycle")
    parser.add_argument("--metasir_future_recent_k", type=int, default=3)
    parser.add_argument("--metasir_future_week_period", type=int, default=7)

    # AR: main classical AR and optional stronger Ridge-AR.
    parser.add_argument("--run_ar", type=int, default=0)
    parser.add_argument("--ar_lag_grid", type=str, default="1,3,7")
    parser.add_argument("--ar_l2_grid", type=str, default="1e-8")
    parser.add_argument("--ar_use_log1p", type=int, default=0)
    parser.add_argument("--ar_standardize", type=int, default=0)
    parser.add_argument("--ar_recency_tau", type=float, default=0.0)
    parser.add_argument("--ar_bias_correction", type=str, choices=["none", "global", "state"], default="none")

    parser.add_argument("--run_ridge_ar", type=int, default=0)
    parser.add_argument("--ridge_ar_lag_grid", type=str, default="1")
    parser.add_argument("--ridge_ar_l2_grid", type=str, default="0.01,0.1,1,10,100")
    parser.add_argument("--ridge_ar_use_log1p", type=int, default=1)
    parser.add_argument("--ridge_ar_standardize", type=int, default=1)
    parser.add_argument("--ridge_ar_recency_tau", type=float, default=21.0)
    parser.add_argument("--ridge_ar_bias_correction", type=str, choices=["none", "global", "state"], default="state")

    # Ridge-VAR.
    parser.add_argument("--run_var", type=int, default=0)
    parser.add_argument("--var_lag_grid", type=str, default="1,3,7")
    parser.add_argument("--var_l2_grid", type=str, default="10,30,100,300,1000,3000")
    parser.add_argument("--var_use_log1p", type=int, default=1)
    parser.add_argument("--var_standardize", type=int, default=1)
    parser.add_argument("--var_mode", type=str, choices=["recursive", "direct"], default="direct")
    parser.add_argument("--var_recency_tau", type=float, default=30.0)
    parser.add_argument("--var_bias_correction", type=str, choices=["none", "global", "state"], default="state")

    params = parser.parse_args()
    ensure_dir(params.output_dir)
    save_config(params.output_dir, vars(params))

    input_data, data_loader = build_data_loader(params)
    population = get_population(data_loader, input_data)
    loc_list = get_loc_list(data_loader, input_data, num_nodes=len(population))

    all_rows = []

    if params.run_sir:
        cfg, tuning_info = select_sir_config(data_loader, population, params)
        model_params, outputs = evaluate_fixed_sir(data_loader, population, cfg)
        all_rows.extend(save_outputs(params.output_dir, "SIR_fixed_calibration", outputs, loc_list, params.n_bootstrap))
        save_config(
            os.path.join(params.output_dir, "SIR_fixed_calibration"),
            {"config": asdict(cfg), "beta_mean": float(model_params.beta.mean()), "validation_selection": tuning_info},
        )

    if params.run_rolling_sir:
        rolling_cfg = RollingSIRConfig(
            gamma_fixed=params.sir_gamma_fixed,
            beta_mode=params.sir_beta_mode,
            beta_min=params.rolling_sir_beta_min,
            beta_max=params.sir_beta_max,
            ridge=params.rolling_sir_ridge,
        )
        rolling_diag, rolling_outputs = evaluate_rolling_sir(
            data_loader=data_loader,
            population=population,
            config=rolling_cfg,
            split_name="test",
        )

        rolling_name = "SIR_rolling_fit_forecast"
        all_rows.extend(
            save_outputs(
                params.output_dir,
                rolling_name,
                rolling_outputs,
                loc_list,
                params.n_bootstrap,
            )
        )

        rolling_dir = os.path.join(params.output_dir, rolling_name)
        np.save(os.path.join(rolling_dir, "beta_by_origin.npy"), rolling_diag.beta_by_origin)
        np.save(os.path.join(rolling_dir, "target_start.npy"), rolling_diag.target_start)
        save_config(
            rolling_dir,
            {
                "config": asdict(rolling_cfg),
                "split_name": "test",
                "beta_by_origin_shape": list(rolling_diag.beta_by_origin.shape),
                "target_start_shape": list(rolling_diag.target_start.shape),
                "gamma": float(rolling_diag.gamma),
                "beta_mode": rolling_diag.beta_mode,
            },
        )

    if params.run_metasir:
        cfg, tuning_info = select_metasir_config(data_loader, population, params)
        model_params, outputs = evaluate_fixed_metasir(data_loader, population, cfg)
        all_rows.extend(save_outputs(params.output_dir, "MetaSIR_fixed_calibration", outputs, loc_list, params.n_bootstrap))
        save_config(
            os.path.join(params.output_dir, "MetaSIR_fixed_calibration"),
            {"config": asdict(cfg), "beta_mean": float(model_params.beta.mean()), "validation_selection": tuning_info},
        )

    if params.run_ar:
        cfg = FixedARConfig(
            lag_grid=params.ar_lag_grid,
            l2_grid=params.ar_l2_grid,
            use_log1p=params.ar_use_log1p,
            standardize=params.ar_standardize,
            recency_tau=params.ar_recency_tau,
            bias_correction=params.ar_bias_correction,
            tune_metric="wMAPE%",
        )
        model, outputs = evaluate_fixed_ar(data_loader, cfg)
        all_rows.extend(save_outputs(params.output_dir, "AR_train_once_classical", outputs, loc_list, params.n_bootstrap))
        save_config(
            os.path.join(params.output_dir, "AR_train_once_classical"),
            {"config": asdict(cfg), "selected_lag": model.lag, "selected_l2": model.l2},
        )

    if params.run_ridge_ar:
        cfg = FixedARConfig(
            lag_grid=params.ridge_ar_lag_grid,
            l2_grid=params.ridge_ar_l2_grid,
            use_log1p=params.ridge_ar_use_log1p,
            standardize=params.ridge_ar_standardize,
            recency_tau=params.ridge_ar_recency_tau,
            bias_correction=params.ridge_ar_bias_correction,
            tune_metric="wMAPE%",
        )
        model, outputs = evaluate_fixed_ar(data_loader, cfg)
        all_rows.extend(save_outputs(params.output_dir, "RidgeAR_train_once_optional", outputs, loc_list, params.n_bootstrap))
        save_config(
            os.path.join(params.output_dir, "RidgeAR_train_once_optional"),
            {"config": asdict(cfg), "selected_lag": model.lag, "selected_l2": model.l2},
        )

    if params.run_var:
        cfg = FixedVARConfig(
            lag_grid=params.var_lag_grid,
            l2_grid=params.var_l2_grid,
            use_log1p=params.var_use_log1p,
            standardize=params.var_standardize,
            var_mode=params.var_mode,
            recency_tau=params.var_recency_tau,
            bias_correction=params.var_bias_correction,
            tune_metric="wMAPE%",
        )
        model, outputs = evaluate_fixed_var(data_loader, cfg)
        all_rows.extend(save_outputs(params.output_dir, "RidgeVAR_train_once", outputs, loc_list, params.n_bootstrap))
        save_config(
            os.path.join(params.output_dir, "RidgeVAR_train_once"),
            {"config": asdict(cfg), "selected_lag": model.lag, "selected_l2": model.l2},
        )

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(params.output_dir, "fixed_baselines_all_summary.csv"), index=False)
    print(summary)
    print(f"Done. Results saved to: {params.output_dir}")


if __name__ == "__main__":
    main()
