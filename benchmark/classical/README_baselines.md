# Revised fixed-parameter baselines for MP-STGODE

This package implements revised SIR, Meta-SIR, AR, Ridge-AR, and Ridge-VAR baselines under a controlled fixed-parameter rolling-origin protocol.

## Protocol

All models follow the same evaluation boundary as MP-STGODE:

1. Estimate model parameters on the training split.
2. Select hyperparameters on the validation split.
3. Evaluate on the held-out test split with fixed parameters.
4. Use rolling-origin input windows for inference, but do not perform test-period refitting, recalibration, or daily updating.

This is intentionally different from an operational daily rolling-recalibration study.

## Models

### SIR_fixed_calibration

Standard-incidence SIR baseline:

`newI_i(t) = beta_i S_i(t) I_i(t) / N_i`

`beta` is estimated from training forecast-origin states using one-step least squares. `gamma` and `beta_mode` are selected on validation.

### MetaSIR_fixed_calibration

Mechanistic metapopulation baseline using the same Lagrangian transmission structure as the MP-STGODE mechanism branch. It removes the neural encoder, Neural ODE representation learning, and residual correction.

The future mobility forcing is constructed from each historical input window only. It does not use true future mobility.

### AR_train_once_classical

Main statistical baseline. Per-node raw-scale AR(p), with the lag selected on validation. A tiny ridge term is used only for numerical stability.

### RidgeAR_train_once_optional

A stronger supplementary baseline: per-node Ridge-AR with log1p transform and standardization. This should be reported as an enhanced autoregressive reference rather than a classical AR baseline.

### RidgeVAR_train_once

Multivariate ridge VAR over all nodes. Ridge regularization is used because ordinary VAR is ill-conditioned in a 51-node setting with limited samples.

## Run examples

3-day horizon:

```bash
python run_fixed_baselines.py \
  --data_pkl path/to/Data_meta_stgode_causal7.pkl \
  --output_dir outputs/Tout3 \
  --history_window 7 \
  --pred_window 3 \
  --split_ratio 8 1 1
```

7-day horizon:

```bash
python run_fixed_baselines.py \
  --data_pkl path/to/Data_meta_stgode_causal7.pkl \
  --output_dir outputs/Tout7 \
  --history_window 7 \
  --pred_window 7 \
  --split_ratio 8 1 1
```

For a stricter main-text table without the optional strong Ridge-AR:

```bash
python run_fixed_baselines.py --run_ridge_ar 0
```

## Outputs

For each model and split, the script saves:

- `*_y_true_nwk.npy`
- `*_y_pred_nwk.npy`
- `*_metrics.csv`
- `*_stepwise_metrics.csv`
- `*_state_metrics.csv`
- `*_incidence_stratified_metrics.csv`
- `*_bootstrap_summary.csv`

The top-level file `fixed_baselines_all_summary.csv` summarizes all model/split metrics.

## 2026-07-03 AR/VAR legacy-compatible update

This package now exposes legacy-compatible AR/VAR controls while keeping the fixed-parameter protocol.

### AR controls

- `--ar_use_log1p`: when set to 1, only the one-step target is log1p-transformed before fitting, matching the older `baseline_AR.py` behavior. Historical AR inputs remain in incidence scale.
- `--ar_standardize`: standardizes AR inputs and transformed target using train statistics only.
- `--ar_recency_tau`: exponential recency weighting on train windows. `0` disables it.
- `--ar_bias_correction`: `none`, `global`, or `state`. Bias is estimated from train predictions only and then reused on validation/test.

The optional Ridge-AR uses the same controls with the prefix `ridge_ar_`.

### VAR controls

- `--var_mode`: `direct` fits a separate fixed ridge regression for each forecast lead; `recursive` fits one-step VAR and rolls it forward.
- `--var_use_log1p`: when set to 1, transforms only the target incidence, not historical inputs.
- `--var_standardize`: standardizes X and transformed Y using train statistics only.
- `--var_recency_tau`: exponential recency weighting on train windows.
- `--var_bias_correction`: train-only bias correction reused on validation/test.

### Suggested legacy-like 3-day setting

```bash
python run_fixed_baselines.py ^
  --data_pkl path/to/Data_meta_stgode_7.pkl ^
  --output_dir outputs/legacy3 ^
  --history_window 7 --pred_window 3 --split_ratio 8 1 1 ^
  --run_sir 0 --run_metasir 0 ^
  --run_ar 1 --ar_lag_grid 1,3,7 --ar_l2_grid 0.01,0.1,1,10,100 ^
  --ar_use_log1p 1 --ar_standardize 1 --ar_recency_tau 21 --ar_bias_correction state ^
  --run_ridge_ar 0 ^
  --run_var 1 --var_mode direct --var_lag_grid 1,3,7 --var_l2_grid 0,0.1,1,10,100 ^
  --var_use_log1p 0 --var_standardize 1 --var_recency_tau 14 --var_bias_correction state
```

This remains a fixed-parameter baseline: all coefficients and train-bias corrections are estimated from the training split only; validation selects hyperparameters; test uses fixed coefficients only.
