"""Unified F2/F3/F4 benchmark launcher for MP-STGODE comparisons.

The launcher preserves every model-specific implementation.  It creates one
strictly truncated data file per requested fold, then calls the existing
baseline runners with the exact train/validation/test lengths for that fold.
No reference result values are imported or used here.

Examples
--------
Preview all commands without training (default)::

    python main_benchmark_walk_forward.py

Run all models on F2 and F3::

    python main_benchmark_walk_forward.py --action run --folds F2 F3

Run and tune one model on one fold/horizon::

    python main_benchmark_walk_forward.py --action run --folds F2 \
        --models stgcn --horizons 3 \
        --set F2.stgcn.3.hidden_dim=128 \
        --set F2.stgcn.3.learn_rate=0.001

Model settings default to each runner's own values. Override
syntax is ``FOLD.MODEL.HORIZON.ARG=VALUE``.  ``all`` can replace
FOLD or HORIZON, for example ``F3.stgcn.all.num_epochs=800``.  ``all`` may
also replace MODEL only when every selected model accepts that argument.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np



# -----------------------------------------------------------------------------
# PyCharm-friendly defaults
# -----------------------------------------------------------------------------
DEFAULT_ACTION = "plan"  # Change to "run" only after inspecting the plan.
DEFAULT_FOLDS = ["F2", "F3"]
DEFAULT_MODELS = [
    "sir", "rolling_sir", "metasir", "ar", "var", "lstm", "gru",
    "stgcn", "dcrnn", "stan", "mepognn",
]
DEFAULT_HORIZONS = [3, 7]
DEFAULT_BOOTSTRAP_SAMPLES = 1000


ROOT = Path(__file__).resolve().parent
CLASSICAL_RUNNER = ROOT / "classical" / "run_fixed_baselines.py"
RNN_RUNNER = ROOT / "neural" / "baseline_rnn.py"
GRAPH_RUNNER = ROOT / "neural" / "main_baseline.py"
STAN_RUNNER = ROOT / "neural" / "main_STAN.py"
MEPO_RUNNER = ROOT / "neural" / "main_MepoGNN.py"

DEFAULT_DATA = ROOT.parent / "data" / "Data_meta_stgode_causal7.pkl"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "walk_forward_benchmarks"
MODEL_NAMES = tuple(DEFAULT_MODELS)
FOLD_NAMES = ("F2", "F3", "F4")


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    label: str

    @property
    def split_lengths(self) -> Tuple[int, int, int]:
        return (
            self.train_end - self.train_start,
            self.val_end - self.val_start,
            self.test_end - self.test_start,
        )

    def validate(self, total_time: int) -> None:
        bounds = (
            self.train_start,
            self.train_end,
            self.val_start,
            self.val_end,
            self.test_start,
            self.test_end,
        )
        if self.train_start != 0:
            raise ValueError(f"{self.fold_id}: current baseline adapters require train_start=0")
        if not (0 <= bounds[0] < bounds[1] == bounds[2] < bounds[3] == bounds[4] < bounds[5] <= total_time):
            raise ValueError(f"Invalid contiguous boundaries for {self.fold_id}: {bounds}")


def make_fold_specs(f4_protocol: str) -> Dict[str, FoldSpec]:
    specs = {
        "F2": FoldSpec("F2", 0, 172, 172, 200, 200, 228, "growth/peak"),
        "F3": FoldSpec("F3", 0, 202, 202, 230, 230, 258, "peak/decline"),
    }
    if f4_protocol == "fixed_8_1_1":
        specs["F4"] = FoldSpec("F4", 0, 240, 240, 270, 270, 300, "fixed 8:1:1 split")
    else:
        specs["F4"] = FoldSpec("F4", 0, 244, 244, 272, 272, 300, "late held out")
    return specs



CLASSICAL_SWITCHES = {
    "sir": "run_sir",
    "rolling_sir": "run_rolling_sir",
    "metasir": "run_metasir",
    "ar": "run_ar",
    "var": "run_var",
}

CLASSICAL_RESULT_NAMES = {
    "sir": "SIR_fixed_calibration",
    "rolling_sir": "SIR_rolling_fit_forecast",
    "metasir": "MetaSIR_fixed_calibration",
    "ar": "AR_train_once_classical",
    "var": "RidgeVAR_train_once",
}

RESERVED_ARGS = {
    "data_pkl", "data_pickle", "output_dir", "expdir", "model",
    "pred_window", "split_ratio", "n_bootstrap", "bootstrap_eval", "seed", "device",
    "shuffle_test",
}

TIME_AXES = {
    "dates_time": 0,
    "dates": 0,
    "date": 0,
    "time": 0,
    "input_feat": 1,
    "seri_feat": 1,
    "edge_stack_feat": 1,
    "adj_matrix": 0,
    "normalized_adj_matrix": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_arg(command: List[str], key: str, value: Any) -> None:
    command.append(f"--{key}")
    if isinstance(value, (list, tuple)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def parse_overrides(items: Iterable[str]) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """Parse FOLD.MODEL.HORIZON.ARG=VALUE overrides."""
    result: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value {item!r}; missing '='")
        selector, value = item.split("=", 1)
        parts = selector.split(".")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid --set value {item!r}; expected FOLD.MODEL.HORIZON.ARG=VALUE"
            )
        fold, model, horizon, arg_name = parts
        fold = fold.upper()
        model = model.lower()
        horizon = horizon.lower()
        if fold != "ALL" and fold not in FOLD_NAMES:
            raise ValueError(f"Unknown fold in --set: {fold}")
        if model != "all" and model not in MODEL_NAMES:
            raise ValueError(f"Unknown model in --set: {model}")
        if horizon != "all" and horizon not in ("3", "7"):
            raise ValueError(f"Unknown horizon in --set: {horizon}")
        arg_name = arg_name.lstrip("-")
        if arg_name in RESERVED_ARGS:
            raise ValueError(f"{arg_name!r} is controlled by the walk-forward launcher")
        result.setdefault((fold, model, horizon), {})[arg_name] = value
    return result


def settings_for(
    fold: str,
    model: str,
    horizon: int,
    overrides: Mapping[Tuple[str, str, str], Mapping[str, str]],
) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    selectors = [
        ("ALL", "all", "all"),
        (fold, "all", "all"),
        (fold, "all", str(horizon)),
        ("ALL", model, "all"),
        ("ALL", model, str(horizon)),
        (fold, model, "all"),
        (fold, model, str(horizon)),
    ]
    for selector in selectors:
        settings.update(overrides.get(selector, {}))
    return settings


def slice_value(value: Any, axis: int, end: int) -> Any:
    if isinstance(value, list):
        return value[:end]
    if isinstance(value, tuple):
        return value[:end]
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(0, end)
    return value[tuple(slices)]


def create_fold_dataset(source: Path, destination: Path, spec: FoldSpec) -> Dict[str, Any]:
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in {source}, got {type(payload).__name__}")
    if "input_feat" not in payload or "adj_matrix" not in payload:
        raise KeyError("Source data must contain input_feat and adj_matrix")
    total_time = int(np.asarray(payload["input_feat"]).shape[1])
    spec.validate(total_time)

    sliced = copy.deepcopy(payload)
    for key, axis in TIME_AXES.items():
        if key not in sliced:
            continue
        value = sliced[key]
        length = len(value) if isinstance(value, (list, tuple)) else int(value.shape[axis])
        # Old files may contain a date vector longer than the actual 300-step
        # tensors.  The causal source used by default is already aligned.
        if length < spec.test_end:
            raise ValueError(f"{key} has length {length}, shorter than {spec.fold_id} end {spec.test_end}")
        sliced[key] = slice_value(value, axis, spec.test_end)

    if int(np.asarray(sliced["input_feat"]).shape[1]) != spec.test_end:
        raise AssertionError("Fold input_feat was not sliced correctly")
    if int(sliced["adj_matrix"].shape[0]) != spec.test_end:
        raise AssertionError("Fold adj_matrix was not sliced correctly")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(sliced, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dates = sliced.get("dates_time", [])
    date_at = lambda index: str(dates[index]) if len(dates) > index else None
    return {
        "path": str(destination),
        "sha256": sha256(destination),
        "time_steps": spec.test_end,
        "split_lengths": list(spec.split_lengths),
        "train_dates": [date_at(spec.train_start), date_at(spec.train_end - 1)],
        "validation_dates": [date_at(spec.val_start), date_at(spec.val_end - 1)],
        "test_dates": [date_at(spec.test_start), date_at(spec.test_end - 1)],
    }


def build_command(
    model: str,
    horizon: int,
    spec: FoldSpec,
    fold_data: Path,
    case_dir: Path,
    args: argparse.Namespace,
    overrides: Mapping[Tuple[str, str, str], Mapping[str, str]],
) -> Tuple[List[str], Dict[str, Any]]:
    settings = settings_for(spec.fold_id, model, horizon, overrides)
    resolved_settings = copy.deepcopy(settings)
    history_window = settings.pop("history_window", 7)
    split = spec.split_lengths

    if model in CLASSICAL_SWITCHES:
        command = [str(args.python_executable), "-B", str(CLASSICAL_RUNNER)]
        add_arg(command, "data_pkl", fold_data)
        add_arg(command, "output_dir", case_dir)
        add_arg(command, "history_window", history_window)
        add_arg(command, "pred_window", horizon)
        add_arg(command, "slide_step", 1)
        add_arg(command, "split_ratio", split)
        add_arg(command, "n_bootstrap", 0)
        for switch in CLASSICAL_SWITCHES.values():
            add_arg(command, switch, int(switch == CLASSICAL_SWITCHES[model]))
        add_arg(command, "run_ridge_ar", 0)
    elif model in ("lstm", "gru"):
        command = [str(args.python_executable), "-B", str(RNN_RUNNER)]
        add_arg(command, "data_pkl", fold_data)
        add_arg(command, "output_dir", case_dir)
        add_arg(command, "device", args.device)
        add_arg(command, "model", model.upper())
        add_arg(command, "history_window", history_window)
        add_arg(command, "pred_window", horizon)
        add_arg(command, "slide_step", 1)
        add_arg(command, "split_ratio", split)
        add_arg(command, "seed", args.seed)
        add_arg(command, "n_bootstrap", 0)
        add_arg(command, "shuffle_test", 0)
    elif model in ("stgcn", "dcrnn"):
        command = [str(args.python_executable), "-B", str(GRAPH_RUNNER)]
        add_arg(command, "data_pkl", fold_data)
        add_arg(command, "output_dir", case_dir)
        add_arg(command, "expdir", case_dir / "experiment")
        add_arg(command, "device", args.device)
        add_arg(command, "model", model.upper())
        add_arg(command, "history_window", history_window)
        add_arg(command, "pred_window", horizon)
        add_arg(command, "slide_step", 1)
        add_arg(command, "split_ratio", split)
        add_arg(command, "seed", args.seed)
        add_arg(command, "bootstrap_eval", 0)
    elif model == "stan":
        command = [str(args.python_executable), "-B", str(STAN_RUNNER)]
        add_arg(command, "data_pkl", fold_data)
        add_arg(command, "output_dir", case_dir)
        add_arg(command, "expdir", case_dir / "experiment")
        add_arg(command, "device", args.device)
        add_arg(command, "model", "STAN")
        add_arg(command, "history_window", history_window)
        add_arg(command, "pred_window", horizon)
        add_arg(command, "slide_step", 1)
        add_arg(command, "split_ratio", split)
        add_arg(command, "seed", args.seed)
        add_arg(command, "bootstrap_eval", 0)
    elif model == "mepognn":
        command = [str(args.python_executable), "-B", str(MEPO_RUNNER)]
        add_arg(command, "data_pickle", fold_data)
        add_arg(command, "output_dir", case_dir)
        add_arg(command, "expdir", case_dir / "checkpoints")
        add_arg(command, "device", args.device)
        add_arg(command, "history_window", history_window)
        add_arg(command, "pred_window", horizon)
        add_arg(command, "slide_step", 1)
        add_arg(command, "split_ratio", split)
        add_arg(command, "seed", args.seed)
        add_arg(command, "n_bootstrap", 0)
    else:
        raise ValueError(f"Unsupported model: {model}")

    for key, value in settings.items():
        add_arg(command, key, value)
    resolved_settings["history_window"] = history_window
    return command, resolved_settings


def command_text(command: List[str]) -> str:
    return subprocess.list2cmdline(command)


def run_child(command: List[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(command_text(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


METRIC_NAMES = ("RMSE", "MAE", "MAPE%", "wMAPE%", "RAE")


def canonical_wnk(array: Any, horizon: int, label: str) -> np.ndarray:
    """Convert saved predictions to [forecast_origin, state, horizon]."""
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError(f"{label}: expected a 3-D prediction array, got {value.shape}")
    if value.shape[-1] == horizon:
        return value
    if value.shape[1] == horizon:
        return value.transpose(0, 2, 1)
    raise ValueError(f"{label}: cannot identify horizon axis in {value.shape} for K={horizon}")


def load_prediction_arrays(model: str, case_dir: Path, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load each runner's raw test arrays without using its native bootstrap."""
    if model in CLASSICAL_SWITCHES:
        name = CLASSICAL_RESULT_NAMES[model]
        result_dir = case_dir / name
        true = np.load(result_dir / f"{name}_test_y_true_nwk.npy").transpose(1, 0, 2)
        pred = np.load(result_dir / f"{name}_test_y_pred_nwk.npy").transpose(1, 0, 2)
    elif model in ("lstm", "gru"):
        with np.load(case_dir / f"{model.upper()}_predictions.npz") as payload:
            pred, true = payload["pred"], payload["true"]
    elif model in ("stgcn", "dcrnn"):
        with np.load(case_dir / "experiment" / "dcrnn_dynamic_preds.npz") as payload:
            pred, true = payload["pred"], payload["true"]
    elif model == "stan":
        with np.load(case_dir / "experiment" / "stan_predictions.npz") as payload:
            pred, true = payload["pred"], payload["true"]
    elif model == "mepognn":
        candidates = sorted(case_dir.glob(f"mepognn_pred_H*_K{horizon}.npz"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one MepoGNN prediction file for K={horizon} under {case_dir}, got {candidates}"
            )
        with np.load(candidates[0]) as payload:
            pred, true = payload["y_pred"], payload["y_true"]
    else:
        raise ValueError(f"Unsupported model: {model}")

    pred_wnk = canonical_wnk(pred, horizon, f"{model} prediction")
    true_wnk = canonical_wnk(true, horizon, f"{model} truth")
    if pred_wnk.shape != true_wnk.shape:
        raise ValueError(f"{model}: prediction {pred_wnk.shape} != truth {true_wnk.shape}")
    if not np.isfinite(pred_wnk).all() or not np.isfinite(true_wnk).all():
        raise ValueError(f"{model}: predictions or truths contain non-finite values")
    return pred_wnk, true_wnk


def compute_metrics(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    prediction = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(true, dtype=np.float64).reshape(-1)
    error = prediction - target
    abs_error = np.abs(error)
    mask = np.abs(target) > eps
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(abs_error)),
        "MAPE%": float(np.mean(abs_error[mask] / np.abs(target[mask])) * 100.0) if mask.any() else float("nan"),
        "wMAPE%": float(np.sum(abs_error) / (np.sum(np.abs(target)) + eps) * 100.0),
        "RAE": float(np.sum(abs_error) / (np.sum(np.abs(target - np.mean(target))) + eps)),
    }


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def align_forecast_origins(
    pred_wnk: np.ndarray,
    true_wnk: np.ndarray,
    reference_true_wnk: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Restore reference forecast-origin order using strict one-to-one truth matching.

    This supports legacy LSTM/GRU outputs whose test DataLoader shuffled complete
    forecast windows.  Predictions and truths are moved together.  Alignment is
    accepted only if every candidate window matches exactly one reference window
    within float32/float64 tolerance and the mapping is a full permutation.
    """
    if true_wnk.shape != reference_true_wnk.shape:
        raise RuntimeError(
            f"truth shape {true_wnk.shape} differs from reference {reference_true_wnk.shape}"
        )

    candidate_to_reference: List[int] = []
    for candidate in true_wnk:
        matches = [
            index
            for index, reference in enumerate(reference_true_wnk)
            if np.allclose(candidate, reference, rtol=1e-6, atol=1e-4)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "ground truth cannot be uniquely aligned by forecast origin "
                f"(candidate matched {len(matches)} reference windows)"
            )
        candidate_to_reference.append(matches[0])

    if sorted(candidate_to_reference) != list(range(len(candidate_to_reference))):
        raise RuntimeError("ground-truth alignment is not a one-to-one permutation")

    reference_to_candidate = np.argsort(candidate_to_reference)
    return (
        pred_wnk[reference_to_candidate],
        true_wnk[reference_to_candidate],
        candidate_to_reference,
    )


def unified_bootstrap(
    pred_wnk: np.ndarray,
    true_wnk: np.ndarray,
    samples: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    """Paired cluster bootstrap over forecast origins, preserving states and horizon steps."""
    origins = int(true_wnk.shape[0])
    rng = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []
    for _ in range(samples):
        indices = rng.integers(0, origins, size=origins)
        records.append(compute_metrics(pred_wnk[indices], true_wnk[indices]))

    summary: Dict[str, Dict[str, float]] = {}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in records], dtype=np.float64)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
        }
    return summary, records


def save_unified_case_results(
    case_dir: Path,
    pred_wnk: np.ndarray,
    true_wnk: np.ndarray,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    point = compute_metrics(pred_wnk, true_wnk)
    bootstrap, records = unified_bootstrap(pred_wnk, true_wnk, samples=samples, seed=seed)
    payload: Dict[str, Any] = {
        "protocol": {
            "samples": samples,
            "seed": seed,
            "unit": "forecast_origin",
            "array_layout": "forecast_origin,state,horizon",
            "point_estimate_center": "full_test_set",
        },
        "shape": list(true_wnk.shape),
        "prediction_sha256": array_sha256(pred_wnk),
        "truth_sha256": array_sha256(true_wnk),
        "point": point,
        "bootstrap": bootstrap,
    }
    (case_dir / "unified_test_evaluation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (case_dir / "unified_bootstrap_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_NAMES))
        writer.writeheader()
        writer.writerows(records)

    step_rows = []
    for step in range(true_wnk.shape[2]):
        step_rows.append({"step": step + 1, **compute_metrics(pred_wnk[:, :, step], true_wnk[:, :, step])})
    with (case_dir / "unified_stepwise_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(step_rows)
    return payload


def summarize(run_dir: Path) -> Path:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bootstrap_config = manifest.get("bootstrap", {})
    samples = int(bootstrap_config.get("samples", DEFAULT_BOOTSTRAP_SAMPLES))
    base_seed = int(bootstrap_config.get("seed", manifest.get("seed", 42)))
    reference_truth: Dict[Tuple[str, int], np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    for case in manifest["cases"]:
        fold, model, horizon = case["fold"], case["model"], int(case["horizon"])
        row: Dict[str, Any] = {
            "fold": fold,
            "model": model,
            "horizon": horizon,
            "status": case.get("status", "unknown"),
        }
        try:
            pred, true = load_prediction_arrays(model, Path(case["output_dir"]), horizon)
            group = (fold, horizon)
            alignment_mode = "already_aligned"
            candidate_to_reference: Optional[List[int]] = None
            if group in reference_truth and not np.allclose(
                reference_truth[group], true, rtol=1e-6, atol=1e-4
            ):
                pred, true, candidate_to_reference = align_forecast_origins(
                    pred, true, reference_truth[group]
                )
                alignment_mode = "truth_matched_permutation"
            reference_truth.setdefault(group, true)

            alignment_audit = {
                "mode": alignment_mode,
                "candidate_to_reference": candidate_to_reference,
            }
            (Path(case["output_dir"]) / "unified_origin_alignment.json").write_text(
                json.dumps(alignment_audit, indent=2), encoding="utf-8"
            )

            # The same seed and number of origins give every model in a fold/horizon
            # exactly the same bootstrap index draws.
            bootstrap_seed = base_seed + int(fold[1:]) * 1000 + horizon
            evaluation = save_unified_case_results(
                Path(case["output_dir"]), pred, true, samples=samples, seed=bootstrap_seed
            )
            row["forecast_origins"] = int(true.shape[0])
            row["states"] = int(true.shape[1])
            row["bootstrap_samples"] = samples
            row["bootstrap_seed"] = bootstrap_seed
            row["origin_alignment"] = alignment_mode
            for metric in METRIC_NAMES:
                row[f"{metric}_point"] = evaluation["point"][metric]
                row[f"{metric}_bootstrap_mean"] = evaluation["bootstrap"][metric]["mean"]
                row[f"{metric}_bootstrap_std"] = evaluation["bootstrap"][metric]["std"]
                row[f"{metric}_ci_low"] = evaluation["bootstrap"][metric]["ci_low"]
                row[f"{metric}_ci_high"] = evaluation["bootstrap"][metric]["ci_high"]
        except Exception as exc:
            row["summary_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    output = run_dir / "walk_forward_benchmark_summary.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output


def validate_args(args: argparse.Namespace) -> None:
    unknown_models = sorted(set(args.models) - set(MODEL_NAMES))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    unknown_folds = sorted(set(args.folds) - set(FOLD_NAMES))
    if unknown_folds:
        raise ValueError(f"Unknown folds: {unknown_folds}")
    if not args.data_pkl.exists():
        raise FileNotFoundError(args.data_pkl)
    if args.n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2")
    for runner in (
        CLASSICAL_RUNNER, RNN_RUNNER, GRAPH_RUNNER,
        STAN_RUNNER, MEPO_RUNNER,
    ):
        if not runner.exists():
            raise FileNotFoundError(runner)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run F2/F3/F4 baselines with exact fold-specific chronological splits."
    )
    parser.add_argument("--action", choices=["plan", "run", "summarize"], default=DEFAULT_ACTION)
    parser.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--horizons", type=int, nargs="+", choices=[3, 7], default=DEFAULT_HORIZONS)
    parser.add_argument("--f4-protocol", choices=["fixed_8_1_1", "walk_forward_28d"], default="fixed_8_1_1")
    parser.add_argument("--data-pkl", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Unified test forecast-origin bootstrap samples (default: 1000).",
    )
    parser.add_argument("--set", action="append", default=[], metavar="FOLD.MODEL.HORIZON.ARG=VALUE")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.folds = [fold.upper() for fold in args.folds]
    args.models = [model.lower() for model in args.models]

    if args.action == "summarize":
        if args.run_dir is None:
            raise ValueError("--run-dir is required for --action summarize")
        print(f"Summary written to: {summarize(args.run_dir.resolve())}")
        return

    validate_args(args)
    overrides = parse_overrides(args.set)
    specs = make_fold_specs(args.f4_protocol)
    with args.data_pkl.open("rb") as handle:
        source_payload = pickle.load(handle)
    total_time = int(np.asarray(source_payload["input_feat"]).shape[1])
    for fold in args.folds:
        specs[fold].validate(total_time)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = args.run_tag.strip() or f"walk_forward_{timestamp}"
    run_dir = args.output_root.resolve() / run_tag
    dataset_paths = {fold: run_dir / "fold_datasets" / f"{fold}.pkl" for fold in args.folds}

    cases: List[Dict[str, Any]] = []
    for fold in args.folds:
        spec = specs[fold]
        for horizon in args.horizons:
            for model in args.models:
                case_dir = run_dir / fold / f"pred_{horizon}" / model
                command, resolved_settings = build_command(
                    model, horizon, spec, dataset_paths[fold], case_dir, args, overrides
                )
                cases.append({
                    "fold": fold,
                    "model": model,
                    "horizon": horizon,
                    "output_dir": str(case_dir),
                    "command": command,
                    "resolved_settings": resolved_settings,
                    "status": "planned",
                })

    print(f"Source data: {args.data_pkl}")
    print(f"Run directory: {run_dir}")
    print(
        f"Unified test bootstrap: {args.n_bootstrap} forecast-origin resamples; "
        "paired across models within each fold/horizon"
    )
    print("Fold definitions:")
    for fold in args.folds:
        spec = specs[fold]
        print(
            f"  {fold}: train=[{spec.train_start},{spec.train_end}) "
            f"val=[{spec.val_start},{spec.val_end}) test=[{spec.test_start},{spec.test_end}) "
            f"split_lengths={spec.split_lengths} ({spec.label})"
        )
    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] {case['fold']} | {case['model']} | {case['horizon']}-day")
        print(command_text(case["command"]))

    if args.action == "plan":
        print("\nPlan only: no fold dataset, output directory, or training process was created.")
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    dataset_records = {}
    for fold in args.folds:
        dataset_records[fold] = create_fold_dataset(args.data_pkl, dataset_paths[fold], specs[fold])

    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_data": str(args.data_pkl.resolve()),
        "source_sha256": sha256(args.data_pkl),
        "f4_protocol": args.f4_protocol,
        "seed": args.seed,
        "device": args.device,
        "bootstrap": {
            "samples": args.n_bootstrap,
            "seed": args.seed,
            "unit": "forecast_origin",
            "paired_across_models": True,
            "native_runner_bootstrap": False,
        },
        "folds": {fold: asdict(specs[fold]) for fold in args.folds},
        "fold_datasets": dataset_records,
        "cases": cases,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for index, case in enumerate(cases, start=1):
        case_dir = Path(case["output_dir"])
        case_dir.mkdir(parents=True, exist_ok=False)
        print(f"\n{'=' * 88}\n[{index}/{len(cases)}] RUN {case['fold']} | {case['model']} | {case['horizon']}-day\n{'=' * 88}")
        case["started_at"] = datetime.now().isoformat(timespec="seconds")
        return_code = run_child(case["command"], case_dir / "run.log")
        case["return_code"] = return_code
        case["finished_at"] = datetime.now().isoformat(timespec="seconds")
        case["status"] = "completed" if return_code == 0 else "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if return_code != 0 and not args.continue_on_error:
            raise RuntimeError(f"Failed case; see {case_dir / 'run.log'}")

    print(f"\nAll requested cases finished. Summary: {summarize(run_dir)}")


if __name__ == "__main__":
    main()
