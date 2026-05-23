from __future__ import division
from __future__ import print_function

import os


# =====================================================
# OpenMP / MKL runtime control
# Must be set before importing numpy / torch / scipy.
# =====================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import copy
import pickle
import shutil
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

# dirty hack: include top level folder to path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)

# 注意：这里保持和你当前工程一致。
from Utils import DataGenerator
from model import STGODE_SIR


def scalar_to_tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor(value, device=device, dtype=torch.float32)


def get_newI_scaler(data_loader, device):
    """读取训练集 scaler 中 new_I 对应的 mean/std。
    保留你原来的约定：new_I 是最后一个 feature，因此使用 mean[-1], std[-1]。
    """
    scalers = data_loader['scalers']
    mean = scalers['mean'][-1]
    std = scalers['std'][-1]
    return scalar_to_tensor(mean, device), torch.clamp(scalar_to_tensor(std, device), min=1e-6)


def apply_stage_preset(params):
    """把论文思路中的阶段转成可运行开关。

    - none: 不覆盖任何手动设置。
    - legacy_parallel: 旧结构对照，预测由 output_decoder 直接输出，机制只作辅助。
    - stage1_mech_fixed_gamma: 纯机制主预测，固定 gamma，无残差。
    - stage2_mech_residual_fixed_gamma: 机制主预测 + 小残差，固定 gamma。
    - stage3_mech_residual_weak_gamma: 机制主预测 + 小残差，gamma 弱学习。
    """
    preset = params.get('stage_preset', 'none')
    if preset == 'none':
        return params

    if preset == 'legacy_parallel':
        params['forecast_mode'] = 'legacy_parallel'
        params['gamma_mode'] = 'learn_free'
        params['mech_loss_weight'] = params.get('dyn_weight', 0.2)
        params['residual_penalty_weight'] = 0.0
        params['gamma_prior_weight'] = 0.0
        params['gamma_smooth_weight'] = 0.0
    elif preset == 'stage1_mech_fixed_gamma':
        params['forecast_mode'] = 'mech_only'
        params['gamma_mode'] = 'fixed'
        params['residual_scale'] = 0.0
        # pred == mech，因此不重复加入 mech_loss，避免实际权重翻倍。
        params['mech_loss_weight'] = 0.0
        params['residual_penalty_weight'] = 0.0
        params['gamma_prior_weight'] = 0.0
        params['gamma_smooth_weight'] = 0.0
    elif preset == 'stage2_mech_residual_fixed_gamma':
        params['forecast_mode'] = 'mech_residual'
        params['gamma_mode'] = 'fixed'
        params['residual_scale'] = params.get('residual_scale', 0.05)
        params['mech_loss_weight'] = params.get('mech_loss_weight', 0.5)
        params['residual_penalty_weight'] = params.get('residual_penalty_weight', 0.05)
        params['gamma_prior_weight'] = 0.0
        params['gamma_smooth_weight'] = 0.0
    elif preset == 'stage3_mech_residual_weak_gamma':
        params['forecast_mode'] = 'mech_residual'
        params['gamma_mode'] = 'weak'
        params['residual_scale'] = params.get('residual_scale', 0.05)
        params['mech_loss_weight'] = params.get('mech_loss_weight', 0.5)
        params['residual_penalty_weight'] = params.get('residual_penalty_weight', 0.05)
        params['gamma_prior_weight'] = params.get('gamma_prior_weight', 0.1)
        params['gamma_smooth_weight'] = params.get('gamma_smooth_weight', 0.1)
    else:
        raise ValueError(f'Unknown stage_preset: {preset}')

    print('[Stage preset applied]')
    print(f"  stage_preset={params['stage_preset']}")
    print(f"  forecast_mode={params['forecast_mode']}, gamma_mode={params['gamma_mode']}")
    print(f"  residual_scale={params['residual_scale']}, mech_loss_weight={params['mech_loss_weight']}")
    return params


class ModelTrainer(object):
    def __init__(self, params, data: dict):
        self.params = params
        self.data_container = data
        self.static_feats = data['static_feat']

        self.model = self.get_model().to(params['device'])
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_loss()

        self.clip = 5
        self.task_level = 1

    def get_model(self):
        if self.params['model'] == 'STGODE':
            model = STGODE_SIR(
                num_nodes=self.params['num_nodes'],
                num_features=self.params['input_features'],
                timesteps_input=self.params['history_window'],
                timesteps_output=self.params['pred_window'],
                forecast_mode=self.params['forecast_mode'],
                gamma_mode=self.params['gamma_mode'],
                gamma_fixed=self.params['gamma_fixed'],
                gamma_amp=self.params['gamma_amp'],
                gamma_min=self.params['gamma_min'],
                gamma_max=self.params['gamma_max'],
                beta_min=self.params['beta_min'],
                beta_max=self.params['beta_max'],
                residual_scale=self.params['residual_scale'],
                encoder_fusion=self.params['encoder_fusion'],
                od_scale_factor=self.params['od_scale_factor'],
                last_sir_format=self.params['last_sir_format'],
                clamp_states=bool(self.params['clamp_states']),
                use_mobility_binary_mask=bool(self.params['use_mobility_binary_mask']),
                i_init_mode=self.params['i_init_mode'],
                i0_blend_alpha=self.params['i0_blend_alpha'],
                hist_conv_len=self.params['hist_conv_len'],
                conv_gamma_mode=self.params['conv_gamma_mode'],
                remove_new_cases_same_day=bool(self.params['remove_new_cases_same_day']),
                history_newI_is_normalized=bool(self.params['history_newI_is_normalized']),
                future_mobility_mode=self.params['future_mobility_mode'],
                future_recent_k=self.params['future_recent_k'],
                future_week_period=self.params['future_week_period'],
                future_blend_alpha=self.params['future_blend_alpha'],
                detach_mech_mobility=bool(self.params['detach_mech_mobility'])
            )
            print(f'Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
            print(f"[Model config] forecast_mode={self.params['forecast_mode']}, gamma_mode={self.params['gamma_mode']}, encoder_fusion={self.params['encoder_fusion']}")
            print(f"[I0 config] i_init_mode={self.params['i_init_mode']}, hist_conv_len={self.params['hist_conv_len']}, conv_gamma_mode={self.params['conv_gamma_mode']}, i0_blend_alpha={self.params['i0_blend_alpha']}")
            print(f"[Mobility config] future_mobility_mode={self.params['future_mobility_mode']}, recent_k={self.params['future_recent_k']}, week_period={self.params['future_week_period']}, blend_alpha={self.params['future_blend_alpha']}, detach_mech_mobility={self.params['detach_mech_mobility']}")
        else:
            raise NotImplementedError('Invalid model name.')
        return model

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.params['learn_rate'],
                weight_decay=self.params['weight_decay']
            )
        elif self.params['optimizer'] == 'LBFGS':
            optimizer = torch.optim.LBFGS(
                params=self.model.parameters(),
                lr=self.params['learn_rate'],
                max_iter=20,
                history_size=100,
                line_search_fn='strong_wolfe'
            )
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer

    def get_loss(self):
        if self.params['loss'] == 'MSE':
            return torch.nn.MSELoss()
        elif self.params['loss'] == 'MAE':
            return torch.nn.L1Loss()
        else:
            raise NotImplementedError('Invalid loss function.')

    def forward_model(self, batch, data_loader):
        x_node, new_I, Sn, adj_matrix, last_SIR, y_true = batch
        newI_mean, newI_std = get_newI_scaler(data_loader, self.params['device'])
        param_b, param_g, pred_newI, d_SIR, aux = self.model(
            x_node, new_I, Sn, self.static_feats,
            adj_matrix, data_loader['max_od'], self.params['device'],
            last_SIR=last_SIR,
            newI_mean=newI_mean,
            newI_std=newI_std
        )
        return param_b, param_g, pred_newI, d_SIR, aux, y_true

    def combined_loss_function(self, param_b, param_g, pred_newI, y_true, aux):
        """新版 loss：预测主损失 + 机制自身预测损失 + 残差惩罚 + 参数正则。

        重点变化：不再对 y_dyn 做 batch 自标准化；机制输出使用训练集 scaler 进入 normalized scale。
        """
        device = pred_newI.device
        zero = torch.tensor(0.0, device=device)

        forecast_loss = self.criterion(pred_newI, y_true)

        newI_mech_norm = aux.get('newI_mech_norm', None)
        if newI_mech_norm is not None and self.params['mech_loss_weight'] > 0:
            mech_loss = self.criterion(newI_mech_norm, y_true)
        else:
            mech_loss = zero

        residual = aux.get('residual_norm', None)
        if residual is not None and self.params['residual_penalty_weight'] > 0 and self.params['forecast_mode'] == 'mech_residual':
            residual_loss = torch.mean(residual ** 2)
        else:
            residual_loss = zero

        if param_b.shape[2] > 1:
            beta_smooth = torch.mean((param_b[:, :, 1:, :] - param_b[:, :, :-1, :]) ** 2)
            gamma_smooth = torch.mean((param_g[:, :, 1:, :] - param_g[:, :, :-1, :]) ** 2)
        else:
            beta_smooth = zero
            gamma_smooth = zero

        if self.params['gamma_prior_weight'] > 0 and self.params['gamma_mode'] != 'fixed':
            gamma0 = torch.tensor(self.params['gamma_fixed'], device=device, dtype=param_g.dtype)
            gamma_prior = torch.mean((param_g - gamma0) ** 2)
        else:
            gamma_prior = zero

        total_loss = (
            forecast_loss
            + self.params['mech_loss_weight'] * mech_loss
            + self.params['residual_penalty_weight'] * residual_loss
            + self.params['beta_smooth_weight'] * beta_smooth
            + self.params['gamma_smooth_weight'] * gamma_smooth
            + self.params['gamma_prior_weight'] * gamma_prior
        )

        logs = {
            'total': total_loss,
            'forecast': forecast_loss,
            'mech': mech_loss,
            'residual': residual_loss,
            'beta_smooth': beta_smooth,
            'gamma_smooth': gamma_smooth,
            'gamma_prior': gamma_prior,
        }
        return total_loss, logs

    def train(self, data_loader: dict, modes: list, model_idx: int, early_stop_patience=60):
        bootstrap_dir = os.path.join(self.params['output_dir'], 'bootstrap_models')
        os.makedirs(bootstrap_dir, exist_ok=True)

        checkpoint = {'epoch': 0, 'state_dict': self.model.state_dict()}
        best_val_loss = np.inf
        patience_count = early_stop_patience
        best_model_path = None

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training begins:')

        for epoch in range(1, self.params['num_epochs'] + 1):
            epoch_start = datetime.now()
            running_loss = {mode: 0.0 for mode in modes}
            sample_counts = {mode: 0 for mode in modes}
            loss_logs = {mode: {} for mode in modes}

            for mode in modes:
                self.model.train() if mode == 'train' else self.model.eval()

                for batch in data_loader[mode]:
                    batch_size = batch[-1].shape[0]

                    with torch.set_grad_enabled(mode == 'train'):
                        if self.params['model'] == 'STGODE':
                            param_b, param_g, pred_newI, d_SIR, aux, y_true = self.forward_model(batch, data_loader)
                        else:
                            raise NotImplementedError('Invalid model name.')

                        total_loss, logs = self.combined_loss_function(param_b, param_g, pred_newI, y_true, aux)

                        if mode == 'train':
                            self.optimizer.zero_grad()
                            total_loss.backward()
                            if self.clip is not None:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                            self.optimizer.step()

                        running_loss[mode] += total_loss.item() * batch_size
                        sample_counts[mode] += batch_size
                        for k, v in logs.items():
                            loss_logs[mode][k] = loss_logs[mode].get(k, 0.0) + float(v.detach().cpu()) * batch_size

            epoch_loss = {
                mode: running_loss[mode] / max(sample_counts[mode], 1)
                for mode in modes
            }
            epoch_logs = {
                mode: {k: v / max(sample_counts[mode], 1) for k, v in loss_logs[mode].items()}
                for mode in modes
            }

            log_msg = f"Epoch {epoch} [{(datetime.now() - epoch_start).seconds}s] - "
            log_msg += ", ".join([f"{m} Loss: {epoch_loss[m]:.4f}" for m in modes])
            if epoch == 1 or epoch % self.params['print_every'] == 0:
                detail = []
                for m in modes:
                    if m in epoch_logs:
                        detail.append(
                            f"{m}: forecast={epoch_logs[m].get('forecast', 0):.4f}, mech={epoch_logs[m].get('mech', 0):.4f}, "
                            f"res={epoch_logs[m].get('residual', 0):.4f}, beta_sm={epoch_logs[m].get('beta_smooth', 0):.6f}, "
                            f"gamma_pr={epoch_logs[m].get('gamma_prior', 0):.6f}"
                        )
                log_msg += " | " + " ; ".join(detail)
            print(log_msg)

            if 'validate' in epoch_loss:
                val_loss = epoch_loss['validate']
                if val_loss < best_val_loss:
                    print(f"Epoch {epoch}: Validation loss improved {best_val_loss:.4f} -> {val_loss:.4f}")
                    best_val_loss = val_loss
                    checkpoint.update({
                        'epoch': epoch,
                        'state_dict': copy.deepcopy(self.model.state_dict()),
                        'val_loss': val_loss,
                        'params': copy.deepcopy(self.params)
                    })
                    patience_count = early_stop_patience

                    best_model_path = os.path.join(
                        self.params['output_dir'], 'bootstrap_models', f'model_{model_idx}_best.pth'
                    )
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(checkpoint, best_model_path)
                else:
                    patience_count -= 1
                    print(f"Epoch {epoch}: No improvement. Patience left: {patience_count}")

            if 'validate' in epoch_loss and patience_count <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_model_path is None:
            best_model_path = os.path.join(
                self.params['output_dir'], 'bootstrap_models', f'model_{model_idx}_best.pth'
            )
            torch.save(checkpoint, best_model_path)

        final_model_path = os.path.join(
            self.params['output_dir'], 'bootstrap_models', f'model_{model_idx}.pth'
        )
        shutil.copy(best_model_path, final_model_path)
        print(f"<-------Bootstrap Model {model_idx} saved. Best Val Loss: {checkpoint.get('val_loss', np.nan):.4f}----->")
        return checkpoint.get('val_loss', np.nan)

    def test(self, data_loader: dict, modes: list, model_idx: int):
        model_path = os.path.join(
            self.params['output_dir'], 'bootstrap_models', f'model_{model_idx}_best.pth'
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.params['device'])['state_dict'])
        self.model.eval()

        results = {
            mode: {
                'forecasts': [],
                'mech_forecasts': [],
                'residuals': [],
                'betas': [],
                'gammas': [],
                'd_SIR': [],
                'ground_truth': None
            }
            for mode in modes
        }

        with torch.no_grad():
            for mode in modes:
                param_beta, param_gamma = [], []
                sir_list, forecast, mech_forecast, residuals, ground_truth = [], [], [], [], []

                for batch in data_loader[mode]:
                    if self.params['model'] == 'STGODE':
                        param_b, param_g, pred_newI, d_SIR, aux, y_true = self.forward_model(batch, data_loader)
                    else:
                        raise NotImplementedError('Invalid model name.')

                    param_beta.append(param_b.cpu().detach())
                    param_gamma.append(param_g.cpu().detach())
                    sir_list.append(d_SIR.cpu().detach())
                    forecast.append(pred_newI.cpu().detach())
                    mech_forecast.append(aux['newI_mech_norm'].cpu().detach())
                    residuals.append(aux['effective_residual'].cpu().detach())
                    ground_truth.append(y_true.cpu().detach())

                results[mode]['forecasts'].append(torch.cat(forecast, dim=0).numpy())
                results[mode]['mech_forecasts'].append(torch.cat(mech_forecast, dim=0).numpy())
                results[mode]['residuals'].append(torch.cat(residuals, dim=0).numpy())
                results[mode]['betas'].append(torch.cat(param_beta, dim=0).numpy())
                results[mode]['gammas'].append(torch.cat(param_gamma, dim=0).numpy())
                results[mode]['d_SIR'].append(torch.cat(sir_list, dim=0).numpy())
                results[mode]['ground_truth'] = torch.cat(ground_truth, dim=0).numpy()

        for mode in modes:
            path = os.path.join(
                self.params['output_dir'], 'bootstrap_models', f'model_{model_idx}_results_{mode}.pkl'
            )
            with open(path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Bootstrap Model {model_idx} {mode} results saved to {path}")

    def agg_results_train(self, modes: list, model_idx: int):
        all_results = {
            mode: {
                'forecasts': [], 'mech_forecasts': [], 'residuals': [],
                'betas': [], 'gammas': [], 'd_SIR': [], 'ground_truth': None
            }
            for mode in modes
        }

        example_path = os.path.join(
            self.params['output_dir'], 'bootstrap_models', f'model_{model_idx}_results_{modes[0]}.pkl'
        )
        with open(example_path, 'rb') as f:
            model_results = pickle.load(f)

        for mode in modes:
            all_results[mode]['forecasts'].append(model_results[mode]['forecasts'])
            all_results[mode]['mech_forecasts'].append(model_results[mode].get('mech_forecasts', []))
            all_results[mode]['residuals'].append(model_results[mode].get('residuals', []))
            all_results[mode]['betas'].append(model_results[mode]['betas'])
            all_results[mode]['gammas'].append(model_results[mode]['gammas'])
            all_results[mode]['d_SIR'].append(model_results[mode]['d_SIR'])
            all_results[mode]['ground_truth'] = model_results[mode]['ground_truth']

        for mode in modes:
            all_forecasts = np.array(all_results[mode]['forecasts'][0])
            all_betas = np.array(all_results[mode]['betas'][0])
            all_gammas = np.array(all_results[mode]['gammas'][0])
            all_sir = np.array(all_results[mode]['d_SIR'][0])
            ground_truth = all_results[mode]['ground_truth']

            mean_forecast = np.mean(all_forecasts, axis=0)
            mean_beta = np.mean(all_betas, axis=0)
            mean_gamma = np.mean(all_gammas, axis=0)
            mean_sir = np.mean(all_sir, axis=0)

            agg_results_dir = os.path.join(self.params['output_dir'], 'bootstrap_models', 'agg_results')
            os.makedirs(agg_results_dir, exist_ok=True)

            np.save(os.path.join(agg_results_dir, f"{self.params['model']}_{mode}_mean_prediction.npy"), mean_forecast)
            np.save(os.path.join(agg_results_dir, f"{self.params['model']}_{mode}_mean_beta.npy"), mean_beta)
            np.save(os.path.join(agg_results_dir, f"{self.params['model']}_{mode}_mean_gamma.npy"), mean_gamma)
            np.save(os.path.join(agg_results_dir, f"{self.params['model']}_{mode}_mean_sir.npy"), mean_sir)
            np.save(os.path.join(agg_results_dir, f"{self.params['model']}_{mode}_groundtruth.npy"), ground_truth)

            if all_results[mode]['mech_forecasts'][0]:
                all_mech = np.array(all_results[mode]['mech_forecasts'][0])
                np.save(os.path.join(agg_results_dir, f"{self.params['model']}_{mode}_mean_mech_prediction.npy"), np.mean(all_mech, axis=0))
            if all_results[mode]['residuals'][0]:
                all_res = np.array(all_results[mode]['residuals'][0])
                np.save(os.path.join(agg_results_dir, f"{self.params['model']}_{mode}_mean_residual.npy"), np.mean(all_res, axis=0))

    def agg_results_test(self, data_loader: dict, modes: list):
        num_bootstrap_models = self.params['bootstrap_models']
        output_dir = os.path.join(self.params['output_dir'], 'bootstrap_models', 'agg_results')
        os.makedirs(output_dir, exist_ok=True)

        all_results = {
            mode: {'forecasts': [], 'mech_forecasts': [], 'residuals': [], 'betas': [], 'gammas': [], 'ground_truth': None}
            for mode in modes
        }
        all_metrics = {
            mode: {'MSE': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'RAE': []}
            for mode in modes
        }
        best_indices = {}
        best_betas = {}
        best_gammas = {}

        scalers = data_loader['scalers']
        mean, std = scalers['mean'][-1], scalers['std'][-1]

        for model_idx in range(num_bootstrap_models):
            for mode in modes:
                result_path = os.path.join(
                    self.params['output_dir'], 'bootstrap_models', f'model_{model_idx}_results_{mode}.pkl'
                )
                with open(result_path, 'rb') as f:
                    model_results = pickle.load(f)[mode]

                forecast_arr = model_results['forecasts'][0]
                beta_arr = model_results['betas'][0]
                gamma_arr = model_results['gammas'][0]
                mech_arr = model_results.get('mech_forecasts', [None])[0]
                res_arr = model_results.get('residuals', [None])[0]

                all_results[mode]['forecasts'].append(forecast_arr)
                all_results[mode]['betas'].append(beta_arr)
                all_results[mode]['gammas'].append(gamma_arr)
                if mech_arr is not None:
                    all_results[mode]['mech_forecasts'].append(mech_arr)
                if res_arr is not None:
                    all_results[mode]['residuals'].append(res_arr)

                if all_results[mode]['ground_truth'] is None:
                    all_results[mode]['ground_truth'] = model_results['ground_truth']

                forecast = torch.tensor(forecast_arr, dtype=torch.float32)
                ground_truth = torch.tensor(model_results['ground_truth'], dtype=torch.float32)

                forecast = forecast * std + mean
                ground_truth = ground_truth * std + mean

                MSE, RMSE, MAE, MAPE, RAE = Evaluator.evaluate(forecast, ground_truth)
                all_metrics[mode]['MSE'].append(MSE.item())
                all_metrics[mode]['RMSE'].append(RMSE.item())
                all_metrics[mode]['MAE'].append(MAE.item())
                all_metrics[mode]['MAPE'].append(MAPE.item())
                all_metrics[mode]['RAE'].append(RAE.item())

        for mode in modes:
            metrics_path = os.path.join(
                self.params['output_dir'], 'bootstrap_models', f"{self.params['model']}_bootstrap_metrics.txt"
            )
            with open(metrics_path, 'a') as f:
                f.write(f"\n[{mode.upper()}] Bootstrap Models Evaluation (mean±std):\n")
                for metric in ['MSE', 'RMSE', 'MAE', 'MAPE', 'RAE']:
                    values = np.array(all_metrics[mode][metric], dtype=np.float32)
                    f.write(f"{metric}: {values.mean():.4f} ± {values.std():.4f}\n")

            mse_array = np.array(all_metrics[mode]['MSE'])
            best_idx = int(np.argmin(mse_array))
            best_indices[mode] = best_idx
            best_betas[mode] = all_results[mode]['betas'][best_idx]
            best_gammas[mode] = all_results[mode]['gammas'][best_idx]

            forecasts_arr = np.stack(all_results[mode]['forecasts'], axis=0)
            ground_truth = all_results[mode]['ground_truth']

            quantile_levels = {'5%': (2.5, 97.5), '15%': (7.5, 92.5), '25%': (12.5, 87.5)}
            forecast_quantiles = {}
            for q_name, (q_low, q_high) in quantile_levels.items():
                forecast_quantiles[f'lower_{q_name}'] = np.percentile(forecasts_arr, q_low, axis=0)
                forecast_quantiles[f'upper_{q_name}'] = np.percentile(forecasts_arr, q_high, axis=0)

            save_items = {
                'mean_prediction': np.mean(forecasts_arr, axis=0),
                'groundtruth': ground_truth,
                'mean_beta': best_betas[mode],
                'mean_gamma': best_gammas[mode],
                **{f"{k}_prediction": v for k, v in forecast_quantiles.items()}
            }
            if len(all_results[mode]['mech_forecasts']) > 0:
                save_items['mean_mech_prediction'] = np.mean(np.stack(all_results[mode]['mech_forecasts'], axis=0), axis=0)
            if len(all_results[mode]['residuals']) > 0:
                save_items['mean_residual'] = np.mean(np.stack(all_results[mode]['residuals'], axis=0), axis=0)

            for suffix, variable in save_items.items():
                file_path = os.path.join(output_dir, f"{self.params['model']}_{mode}_{suffix}.npy")
                np.save(file_path, variable)

            mean_forecast = torch.tensor(save_items['mean_prediction'], dtype=torch.float32)
            ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)
            mean_forecast = mean_forecast * std + mean
            ground_truth_tensor = ground_truth_tensor * std + mean

            MSE, RMSE, MAE, MAPE, RAE = Evaluator.evaluate(mean_forecast, ground_truth_tensor)
            score_path = os.path.join(
                self.params['output_dir'], 'bootstrap_models', f"{self.params['model']}_prediction_scores.txt"
            )
            with open(score_path, 'a') as f:
                f.write("%s, MSE, RMSE, MAE, MAPE, RAE, %.10f, %.10f, %.10f, %.10f, %.10f\n" %
                        (mode, MSE, RMSE, MAE, MAPE, RAE))

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} bootstrap testing aggregation completed.')


class Evaluator:
    @staticmethod
    def evaluate(y_pred: torch.Tensor, y_true: torch.Tensor):
        mse = F.mse_loss(y_pred, y_true)
        rmse = torch.sqrt(mse)
        mae = F.l1_loss(y_pred, y_true)
        mape_val = Evaluator.mape(y_pred, y_true)
        rae_val = Evaluator.rae(y_pred, y_true)

        print('MSE:', round(mse.item(), 4))
        print('RMSE:', round(rmse.item(), 4))
        print('MAE:', round(mae.item(), 4))
        print('MAPE:', round((mape_val.item() * 100), 4), '%')
        print('RAE:', round(rae_val.item(), 4))

        return mse, rmse, mae, mape_val, rae_val

    @staticmethod
    def mape(preds: torch.Tensor, labels: torch.Tensor):
        eps = 1e-5
        mask = labels != 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device)
        return torch.mean(torch.abs((preds[mask] - labels[mask]) / (labels[mask] + eps)))

    @staticmethod
    def rae(preds: torch.Tensor, labels: torch.Tensor):
        numerator = torch.sum(torch.abs(preds - labels))
        denominator = torch.sum(torch.abs(labels - torch.mean(labels)))
        return numerator / (denominator + 1e-5)


def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='COVID-19')
    parser.add_argument('--graph', type=str, default='nn')
    parser.add_argument('--resroot', type=str, default='../MP_STGODE-master/artifacts')
    parser.add_argument('--output_dir', type=str, default='../MP_STGODE-master/output_stgode_mob')
    parser.add_argument('--data_pkl', type=str,
                        default='../MP_STGODE-master/datasets/Epi_data/Data_meta_stgode_causal7.pkl')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', type=str, default='STGODE')
    parser.add_argument('--num_nodes', type=int, default=51)
    parser.add_argument('--input_features', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--history_window', type=int, default=7)
    parser.add_argument('--pred_window', type=int, default=7)
    parser.add_argument('--slide_step', type=int, default=1)
    parser.add_argument('--k', type=float, default=2.5)
    parser.add_argument('--sigma_e', type=int, default=1)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_ratio', type=float, nargs='+', default=[8, 1, 1])
    parser.add_argument('--learn_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-6)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss', type=str, choices=['MSE', 'MAE'], default='MSE')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--ckptint', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--bootstrap_models', type=int, default=10)
    parser.add_argument('--specify_ckpt', type=str, default='')
    parser.add_argument('--expdir', type=str, default='')
    parser.add_argument('--extconf', type=str, default='')

    # ===== 新增：阶段预设与结构开关 =====
    parser.add_argument('--stage_preset', type=str, default='stage3_mech_residual_weak_gamma',
                        choices=['none', 'legacy_parallel', 'stage1_mech_fixed_gamma',
                                 'stage2_mech_residual_fixed_gamma', 'stage3_mech_residual_weak_gamma'])
    parser.add_argument('--forecast_mode', type=str, default='legacy_parallel',
                        choices=['legacy_parallel', 'mech_only', 'mech_residual'])
    parser.add_argument('--gamma_mode', type=str, default='weak',
                        choices=['fixed', 'weak', 'learn_free'])
    parser.add_argument('--encoder_fusion', type=str, default='max_parallel',
                        choices=['max_parallel', 'mean_parallel', 'sequential'])

    # ===== 新增：参数范围与机制设置 =====
    parser.add_argument('--beta_min', type=float, default=1e-5)
    parser.add_argument('--beta_max', type=float, default=1.5)
    parser.add_argument('--gamma_fixed', type=float, default=0.14)
    parser.add_argument('--gamma_amp', type=float, default=0.03)
    parser.add_argument('--gamma_min', type=float, default=0.05)
    parser.add_argument('--gamma_max', type=float, default=0.25)
    parser.add_argument('--residual_scale', type=float, default=0.1)
    parser.add_argument('--od_scale_factor', type=float, default=3.0)
    parser.add_argument('--last_sir_format', type=str, default='SIR',
                        choices=['SIR', 'newI_SIR', 'SIRD', 'ISRD'])
    parser.add_argument('--i_init_mode', type=str, default='hybrid',
                        choices=['last_sir', 'hist_conv', 'hybrid'],
                        help='I_T 初始化方式：last_sir 用 last_SIR；hist_conv 用历史 new_I 卷积；hybrid 融合二者。')
    parser.add_argument('--i0_blend_alpha', type=float, default=0.1,
                        help='i_init_mode=hybrid 时，I0 = alpha*I0_lastSIR + (1-alpha)*I0_conv。')
    parser.add_argument('--hist_conv_len', type=int, default=14,
                        help='历史卷积重构 I_T 时使用最近多少天 new_I。')
    parser.add_argument('--conv_gamma_mode', type=str, default='pred_first',
                        choices=['fixed', 'pred_mean', 'pred_first'],
                        help='历史卷积中的 gamma：fixed 用 gamma_fixed；pred_mean/pred_first 用预测 gamma 代理。')
    parser.add_argument('--remove_new_cases_same_day', type=int, default=0,
                        help='卷积重构时，新感染者是否当天也参与移除。一般建议 0。')
    parser.add_argument('--history_newI_is_normalized', type=int, default=0,
                        help='如果 batch 中 new_I 是标准化尺度，设为 1，模型内部会先反标准化再卷积。')
    parser.add_argument('--clamp_states', type=int, default=1)
    parser.add_argument('--use_mobility_binary_mask', type=int, default=1)

    # ===== 新增：机制分支 future mobility forcing 开关 =====
    # 这些开关只控制 Meta-SIR rollout 中未来 h_mn/h_nm/h_nn 的构造；
    # encoder/ODEG 内部的时空特征学习保持原结构，便于控制变量。
    parser.add_argument('--future_mobility_mode', type=str, default='learned_attn_detach',
                        choices=['learned_attn', 'learned_attn_detach',
                                 'last_repeat', 'recent_cycle', 'weekday_cycle', 'hybrid_cycle',
                                 'mean_repeat', 'recent_mean'],
                        help='机制分支未来 mobility 构造方式。learned_attn为旧逻辑；其他模式只用历史 mobility 模板。')
    parser.add_argument('--future_recent_k', type=int, default=3,
                        help='recent_cycle/recent_mean/hybrid_cycle 使用最近多少天历史 mobility。')
    parser.add_argument('--future_week_period', type=int, default=7,
                        help='weekday_cycle/hybrid_cycle 的周期长度。history_window=14,pred_window=7 时通常取 7。')
    parser.add_argument('--future_blend_alpha', type=float, default=0.7,
                        help='hybrid_cycle 中 weekday template 的权重 alpha。')
    parser.add_argument('--detach_mech_mobility', type=int, default=0,
                        help='learned_attn_detach 或设置为 1 时，机制分支不通过 mobility 反向更新 int_init。')

    # ===== 新增/调整：loss 权重 =====
    parser.add_argument('--beta_smooth_weight', type=float, default=0.0)
    parser.add_argument('--dyn_weight', type=float, default=0.2, help='legacy alias; 新版主要使用 mech_loss_weight')
    parser.add_argument('--mech_loss_weight', type=float, default=0.7)
    parser.add_argument('--residual_penalty_weight', type=float, default=0.05)
    parser.add_argument('--gamma_smooth_weight', type=float, default=0.0)
    parser.add_argument('--gamma_prior_weight', type=float, default=0.1)

    params = parser.parse_args().__dict__
    extconf, expdir, resroot = params['extconf'], params['expdir'], params['resroot']

    def load_extconf():
        try:
            with open(extconf, 'r') as extconffile:
                extconfdict = yaml.safe_load(extconffile)
                for k, v in extconfdict.items():
                    params[k] = v
            print('load external conf file from {}'.format(extconf))
        except Exception:
            print('no external conf file at {}'.format(extconf))

    def load_expconf():
        expconfpath = os.path.join(expdir, 'conf.yaml')
        try:
            with open(expconfpath, 'r') as expconf:
                expconfdict = yaml.safe_load(expconf)
                for k, v in expconfdict.items():
                    params[k] = v
            print('load existing conf file from {}'.format(expconfpath))
        except Exception:
            print('no existing conf file at {}'.format(expconfpath))

    def make_expdir():
        timestamp = datetime.now().isoformat().replace(':', '_')
        return os.path.join(resroot, params['dataset'], params['model'] + '_{}'.format(params['graph']), timestamp)

    if extconf == '' and expdir == '':
        params = apply_stage_preset(params)
        expdir = make_expdir()
    elif expdir == '':
        load_extconf()
        params = apply_stage_preset(params)
        expdir = make_expdir()
    else:
        load_expconf()
        params = apply_stage_preset(params)

    os.makedirs(expdir, exist_ok=True)
    os.makedirs(params['output_dir'], exist_ok=True)

    with open(os.path.join(expdir, 'conf.yaml'), 'w') as f:
        yaml.safe_dump(params, f)

    print(expdir)

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params['seed'])
        torch.cuda.manual_seed_all(params['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ===== Dataloader =====
    Input_Data = load_data_from_pickle(params['data_pkl'])
    print(f"Loaded data from: {params['data_pkl']}")
    print(f"Original input_feat shape: {Input_Data['input_feat'].shape}")
    print(f"Original adj_matrix shape: {Input_Data['adj_matrix'].shape}")
    if 'dates_time' in Input_Data:
        print(f"dates_time length: {len(Input_Data['dates_time'])}")

    Input_Data['input_feat'] = Input_Data['input_feat'][:, :, :4]
    print(f"Trimmed input_feat shape (first 4 features): {Input_Data['input_feat'].shape}")

    data_generator = DataGenerator(
        history_window=params['history_window'],
        pred_window=params['pred_window'],
        data_split_ratio=params['split_ratio']
    )
    data_loader = data_generator.get_data_loader(data=Input_Data, params=params)

    data_path = os.path.join(params['output_dir'], 'data_loader.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data_loader, f)

    all_val_losses = []
    for model_idx in range(params['bootstrap_models']):
        trainer = ModelTrainer(params=params, data=Input_Data)

        train_dataset = data_loader['train'].dataset
        n_train_samples = len(train_dataset)
        bootstrap_idx = np.random.choice(n_train_samples, size=n_train_samples, replace=True)
        bootstrap_train_dataset = Subset(train_dataset, bootstrap_idx)

        modified_data_loader = {
            'train': DataLoader(
                bootstrap_train_dataset,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=False
            ),
            'validate': data_loader['validate'],
            'test': data_loader['test'],
            'scalers': data_loader['scalers'],
            'max_od': data_loader['max_od']
        }

        print(f"\n=== Training Bootstrap Model {model_idx + 1}/{params['bootstrap_models']} ===")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_start = time.perf_counter()

        best_val_loss = trainer.train(
            data_loader=modified_data_loader,
            modes=['train', 'validate'],
            model_idx=model_idx
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time = time.perf_counter() - train_start

        all_val_losses.append(best_val_loss)
        print(f"[Runtime] Bootstrap Model {model_idx} training time: {train_time:.2f} s ({train_time / 60:.2f} min)")

        print(f"\n=== Testing Bootstrap Model {model_idx + 1}/{params['bootstrap_models']} ===")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        test_start = time.perf_counter()

        trainer.test(
            data_loader=modified_data_loader,
            modes=['test'],
            model_idx=model_idx
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        test_time = time.perf_counter() - test_start

        # =====================================================
        # Release resources after each bootstrap model
        # =====================================================
        del trainer
        del bootstrap_train_dataset
        del bootstrap_idx
        del modified_data_loader

        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print(f"[Runtime] Bootstrap Model {model_idx} test inference time: {test_time:.2f} s")

    best_model_idx = int(np.argmin(all_val_losses))
    print(f"全局最佳模型索引: {best_model_idx}, 验证损失: {all_val_losses[best_model_idx]:.4f}")

    best_trainer = ModelTrainer(params=params, data=Input_Data)
    best_trainer.test(
        data_loader=data_loader,
        modes=['train', 'validate'],
        model_idx=best_model_idx
    )

    best_trainer.agg_results_train(modes=['train', 'validate'], model_idx=best_model_idx)
    best_trainer.agg_results_test(data_loader=data_loader, modes=['test'])
