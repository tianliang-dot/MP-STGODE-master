# dcrnn_dynamic.py
# -*- coding: utf-8 -*-
"""
DCRNN with time-varying directed graphs (A_t):
- DCGRU + Seq2Seq + Scheduled Sampling
- Diffusion supports per time step and per sample: [B, S, N, N], S=2*k_hop
- Decoder supports from future A_t (if available) or persistence/moving-average

Inputs from the existing pipeline:
  baseline_views[split]:
    - hist_incidence: [W, N, H]  (history new cases)
    - future_incidence: [W, N, K]
And we will slice A_all[t] -> [T, N, N] to build per-window supports by win starts.
"""

import hashlib, os, yaml, time, pickle, random
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_STGCN import BaselineSTGCN, symmetrize_aggregate
from models.model_DCRNN import DCRNNModel, build_supports_sequence
from datasets.Utils_DCRNN import DataGenerator, extract_A_all, make_dataloaders_dynamic
from datasets.Utils_DCRNN import build_y_norm_from_loader, build_y_denorm_from_loader, Normalizer

# ===================== Utils & Metrics =====================

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # Compatibility with older PyTorch releases.
            torch.use_deterministic_algorithms(True)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _flatten(pred, true):
    return pred.reshape(-1), true.reshape(-1)

def rmse(p, y): p,y=_flatten(p,y); return float(np.sqrt(np.mean((p-y)**2)))
def mae (p, y): p,y=_flatten(p,y); return float(np.mean(np.abs(p-y)))
def mape_pct(p,y,eps=1e-8):
    p,y=_flatten(p,y); m=np.abs(y)>eps
    return float(np.mean(np.abs((p[m]-y[m])/(y[m]+1e-12)))*100.0) if m.any() else float("nan")
def wmape_pct(p,y,eps=1e-8):
    p,y=_flatten(p,y); denom=np.sum(np.abs(y))
    return float(np.sum(np.abs(p-y))/(denom+eps)*100.0)
def rae(p,y):
    p,y=_flatten(p,y); denom=np.sum(np.abs(y-np.mean(y)));
    return float(np.sum(np.abs(p-y))/(denom+1e-12))

def summarize_metrics(pred_w_n_k, true_w_n_k):
    return {"RMSE": rmse(pred_w_n_k,true_w_n_k),
            "MAE": mae(pred_w_n_k,true_w_n_k),
            "MAPE%": mape_pct(pred_w_n_k,true_w_n_k),
            "wMAPE%": wmape_pct(pred_w_n_k,true_w_n_k),
            "RAE": rae(pred_w_n_k,true_w_n_k)}


def get_optimizer(model, params):
    if params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=params['learn_rate'],
                                     weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'LBFGS':
        optimizer = torch.optim.LBFGS(params=model.parameters(),
                                      lr=params['learn_rate'],
                                      max_iter=20,  # 每个优化步的最大迭代次数
                                      history_size=100,  # 历史更新步长存储量
                                      line_search_fn='strong_wolfe')  # 线性搜索方法
    else:
        raise NotImplementedError('Invalid optimizer name.')
    return optimizer


# === Early Stopping ===
class EarlyStopping:
    """
    监控某个验证指标，当连续 patience 轮没有提升时触发早停。
    - monitor: 字符串，仅用于打印；真正比较的是传进 step() 的数值
    - mode: 'min' 或 'max'，决定越小越好还是越大越好
    - min_delta: 最小提升幅度（小于该幅度不算提升）
    - patience: 容忍多少个 epoch 没有提升
    - ckpt_path: 提升时自动保存模型参数
    """
    def __init__(self, monitor='val_loss', mode='min', min_delta=0.0, patience=30, ckpt_path=None):
        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.ckpt_path = ckpt_path

        self.best = None
        self.num_bad = 0
        self.should_stop = False
        self.best_epoch = -1

    def _is_better(self, current, best):
        if self.mode == 'min':
            return (best - current) > self.min_delta
        else:
            return (current - best) > self.min_delta

    def step(self, current_value, model, epoch):
        """
        返回 (should_stop, improved)
        - improved: 本轮是否提升（会自动保存 ckpt）
        - should_stop: 是否达到 patience 需要早停
        """
        improved = False
        if self.best is None:
            self.best = current_value
            self.best_epoch = epoch
            improved = True
            if self.ckpt_path:
                torch.save({"model": model.state_dict()}, self.ckpt_path)
            return self.should_stop, improved

        if self._is_better(current_value, self.best):
            self.best = current_value
            self.best_epoch = epoch
            self.num_bad = 0
            improved = True
            if self.ckpt_path:
                torch.save({"model": model.state_dict()}, self.ckpt_path)
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True

        return self.should_stop, improved


# ===== Bootstrap & 辅助指标 =====
def bootstrap_eval_windows(pred_w_n_k: np.ndarray, true_w_n_k: np.ndarray,
                           n_boot: int = 30, seed: int = 42):
    """
    针对“窗口维度 W”做自助法重采样（带放回），每次用 summarize_metrics 计算一次指标。
    返回：mean±std 以及每次采样的明细。
    """
    rng = np.random.default_rng(seed)
    W = pred_w_n_k.shape[0]
    metrics_list = []
    for _ in range(n_boot):
        idx = rng.integers(0, W, size=W)  # 采样窗口索引
        m = summarize_metrics(pred_w_n_k[idx], true_w_n_k[idx])
        metrics_list.append(m)
    keys = metrics_list[0].keys()
    mean_std = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=float)
        mean_std[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    return mean_std, metrics_list


def per_horizon_metrics(pred_w_n_k: np.ndarray, true_w_n_k: np.ndarray):
    """
    逐步长（K）指标：把 W,N 合并掉，对每个 step k 单独评估，便于分析短/中期。
    """
    W, N, K = pred_w_n_k.shape
    out = []
    for k in range(K):
        pk = pred_w_n_k[:, :, k].reshape(-1)
        yk = true_w_n_k[:, :, k].reshape(-1)
        out.append({"step": int(k+1),
                    "RMSE": float(np.sqrt(np.mean((pk-yk)**2))),
                    "MAE": float(np.mean(np.abs(pk-yk))),
                    "MAPE%": float(np.mean(np.abs((pk-yk)/(yk + 1e-12))) * 100.0),
                    "wMAPE%": float(np.sum(np.abs(pk-yk)) / (np.sum(np.abs(yk)) + 1e-8) * 100.0),
                    "RAE": float(np.sum(np.abs(pk-yk)) / (np.sum(np.abs(yk - yk.mean())) + 1e-12))})
    return out

def save_yaml(path, obj):
    import yaml, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_split_audit(input_data, params, data_loader):
    """Validate and record the absolute train/validation/test window boundaries."""
    total = int(input_data['input_feat'].shape[1])
    H, K, step = int(params['history_window']), int(params['pred_window']), int(params['slide_step'])
    ratios = [float(value) for value in params['split_ratio']]
    train_end = int(ratios[0] / sum(ratios) * total)
    val_len = int(ratios[1] / sum(ratios) * total)
    val_end = train_end + val_len
    bounds = {
        "train": (0, train_end),
        "validate": (train_end - H, val_end),
        "test": (val_end - H, total),
    }
    dates = input_data.get('dates_time', [])
    report = {
        "total_time_steps": total,
        "history_window": H,
        "prediction_window": K,
        "slide_step": step,
        "split_ratio": ratios,
        "splits": {},
    }
    for split, (start, end) in bounds.items():
        expected = np.arange(start, end - H - K + 1, step, dtype=np.int64)
        actual = np.asarray(data_loader['win_starts'][split], dtype=np.int64)
        if not np.array_equal(actual, expected):
            raise RuntimeError(
                f"{split} window starts do not match the fixed split: "
                f"expected {expected.tolist()}, got {actual.tolist()}"
            )
        target_start = int(actual[0] + H)
        target_end = int(actual[-1] + H + K - 1)
        report["splits"][split] = {
            "window_count": int(len(actual)),
            "history_start_index": int(actual[0]),
            "history_last_start_index": int(actual[-1]),
            "target_start_index": target_start,
            "target_end_index": target_end,
            "target_start_date": str(dates[target_start]) if len(dates) > target_start else None,
            "target_end_date": str(dates[target_end]) if len(dates) > target_end else None,
        }
    if report["splits"]["validate"]["target_start_index"] != train_end:
        raise RuntimeError("Validation targets do not start at the 80% boundary.")
    if report["splits"]["test"]["target_start_index"] != val_end:
        raise RuntimeError("Test targets do not start after the validation period.")
    if report["splits"]["test"]["target_end_index"] != total - 1:
        raise RuntimeError("Test targets do not end at the final observation.")
    return report


# ========= 统一的 Model Registry / Loader Registry / Step 适配 =========
def build_loaders(params, dl, A_all):
    """根据模型类型返回合适的 dataloader 三件套。"""
    model_name = params.get('model', 'DCRNN').upper()
    H, K = params['history_window'], params['pred_window']
    common_kwargs = dict(batch_size=params['batch_size'], slide_step=params['slide_step'],
                         H=H, K=K, split_ratio=tuple(int(x) for x in params['split_ratio']))
    if model_name in ['DCRNN', 'STGCN', 'GWNET', 'STAN', 'MEPO']:
        # 图模型：用包含 (xb,yb,Ah,Af) 的 loader
        tr = make_dataloaders_dynamic(dl, "train", A_all, future_mode=params['adj_future_mode'],
                                      ma_m=params['adj_ma_m'], **common_kwargs)
        va = make_dataloaders_dynamic(dl, "validate", A_all, future_mode=params['adj_future_mode'],
                                      ma_m=params['adj_ma_m'], **common_kwargs)
        te = make_dataloaders_dynamic(dl, "test", A_all, future_mode=params['adj_future_mode'],
                                      ma_m=params['adj_ma_m'], **common_kwargs)
        return tr, va, te
    elif model_name in ['AR', 'LSTM', 'GRU']:
        # 序列基线：不需要邻接。用 (xb,yb) 的简化 loader（来自 Utils_DCRNN 里可新增一个 make_dataloaders_sequence）
        from datasets.Utils_DCRNN import _get_dataset_from_dl_split
        def _seq_loader(split):
            ds = _get_dataset_from_dl_split(dl, split)
            X = ds.inputs['x_node']  # [W,N,H,F]
            Y = ds.output.permute(0,2,1,3).squeeze(-1)  # [W,K,N]
            xb = X.permute(0,2,1,3).contiguous().float()  # [W,H,N,F]
            yb = Y.contiguous().float()                   # [W,K,N]
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xb, yb),
                                               batch_size=params['batch_size'], shuffle=True, drop_last=False)
        return _seq_loader("train"), _seq_loader("validate"), _seq_loader("test")
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")

def build_model_by_name(params, A_static=None, device=None):
    """根据模型名构造模型。这里只把 DCRNN 先接上，其它留扩展位。"""
    name = params.get('model', 'DCRNN').upper()
    if name == 'DCRNN':
        return DCRNNModel(
            input_dim=params['input_dim'],
            output_dim=params['output_dim'],
            hidden_dim=params['hidden_dim'],
            enc_layers=params['enc_layers'],
            dec_layers=params['dec_layers'],
            k_hop=params['k_hop'],
            dropout=params['dropout']
        ).to(device)
    elif name == 'STGCN':
        return BaselineSTGCN(
            A = A_static.clone().detach().to(dtype=torch.float32, device=device),
            C_in=params['input_dim'],          # 若只用新增病例一个特征
            C_hidden=params['hidden_dim'],     # 可调
            H=params['history_window'],
            K=params['pred_window'],   # 3 或 7
            Kt=3, Ks=3,
            dropout=params['dropout']
        ).to(device)
    # elif name == 'STAN':
    #     return BaselineSTAN(
    #         in_dim=params['input_dim'],
    #         hidden_dim=params['hidden_dim'],
    #         pred_window=params['pred_window'],
    #         num_heads=params['num_heads'],
    #     ).to(device)
    elif name == 'AR':
        raise NotImplementedError("AR model not yet wired here.")
    # elif name == 'STGCN': ...
    else:
        raise NotImplementedError(f"Model {name} not implemented yet.")

# ===================== Train / Eval =====================

def train_epoch(model, loader, device, params, grad_clip=5.0, pop=None, optimizer=None, is_graph_model=True, y_norm=None):
    model.train()
    total = 0.0

    if is_graph_model:
        for batch in loader:  # ← 正确：遍历 DataLoader 得到 batch
            xb, yb, Ah, Af, nI_raw, last_S_raw = [t.to(device) for t in batch]  # ← 对 batch 里的每个 Tensor 调用 .to()
            if params['model'] =='DCRNN':
                sup_h = build_supports_sequence(Ah, params['k_hop'])  # list len H; each [B,S,N,N]
                sup_f = build_supports_sequence(Af, params['k_hop'])  # list len K
                preds = model(xb, sup_h, yb.shape[1], sup_f, y_teacher_btnc=yb.unsqueeze(-1),
                              teacher_forcing_ratio=params['tf_ratio']).squeeze(-1)  # [B,K,N]
                total_loss = F.l1_loss(preds, yb) if params['loss'] == "MAE" else F.mse_loss(preds, yb)
            elif params['model'] == 'STGCN':
                xb = xb.permute(0, 3, 1, 2).contiguous()
                preds = model(xb, Ah)
                total_loss = F.l1_loss(preds, yb) if params['loss'] == "MAE" else F.mse_loss(preds, yb)
            elif params['model'] == 'STAN':
                preds, aux = model(xb, Ah, nI_raw, last_S_raw, population=pop)  # xb:[B,H,N,C], Ah:[B,H,N,N]
                y_phys_raw = aux['y_phys']
                beta_bkn = aux['beta']
                y_phys_norm = y_norm(y_phys_raw)
                # (1) 主监督（对标签）：preds ≈ yb
                loss_sup = F.mse_loss(preds, yb)
                # (2) 物理拟合（对标签）：让物理外推也贴近标签（小权重稳尺度）
                loss_phys = F.mse_loss(y_phys_norm, yb)
                # (3) 一致性约束：让网络输出贴近物理外推
                loss_cons = F.mse_loss(preds, y_phys_norm)
                # (4) β 时间平滑：抑制逐步震荡（对时间维 L2）
                loss_smooth = ((beta_bkn[:, 1:, :] - beta_bkn[:, :-1, :]) ** 2).mean()
                # 推荐权重：先小后大地调 eta_cons；eta_phys 更小
                eta_cons, eta_phys, beta_smooth_coef = 0.3, 0.05, 1e-3
                total_loss = loss_sup + eta_cons * loss_cons + eta_phys * loss_phys + beta_smooth_coef * loss_smooth
            elif params['model'] == 'MEPO':
                xb = xb.permute(0, 3, 1, 2).contiguous()
                preds = model(xb, Ah, nI_raw, last_S_raw, population=pop)  # xb:[B,H,N,C], Ah:[B,H,N,N]
                total_loss = F.mse_loss(preds, yb) if params['loss'] == "MSE" else F.l1_loss(preds, yb)

            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip is not None: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(total_loss.item()) * xb.size(0)
    else:
        for batch in loader:
            xb, yb = [t.to(device) for t in batch]
            # 这里对 LSTM/GRU 的 forward 调用留口（未来补齐）
            preds = model(xb).squeeze(-1)  # 假设输出 [B,K,N]
            loss = F.l1_loss(preds, yb) if params['loss'] == "MAE" else F.mse_loss(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item()) * xb.size(0)

    return total / len(loader.dataset)

@torch.no_grad()
def evaluate_epoch(model, loader, device, params, pop=None, is_graph_model=True, y_norm=None, y_denorm=None):
    model.eval()

    losses, P, Y = [], [], []
    if is_graph_model:
        for batch in loader:  # ← 正确：遍历 DataLoader 得到 batch
            xb, yb, Ah, Af, nI_raw, last_S_raw = [t.to(device) for t in batch]
            if params['model'] == 'DCRNN':
                sup_h = build_supports_sequence(Ah, params['k_hop'])
                sup_f = build_supports_sequence(Af, params['k_hop'])
                preds = model(xb, sup_h, yb.shape[1], sup_f, y_teacher_btnc=None,
                             teacher_forcing_ratio=0.0).squeeze(-1)  # [B,K,N]
                total_loss = F.l1_loss(preds, yb) if params['loss'] == "MAE" else F.mse_loss(preds, yb)
            elif params['model'] == 'STGCN':
                xb = xb.permute(0, 3, 1, 2).contiguous()
                preds = model(xb, Ah)
                total_loss = F.l1_loss(preds, yb) if params['loss'] == "MAE" else F.mse_loss(preds, yb)
            elif params['model'] == 'STAN':
                preds, aux = model(xb, Ah, nI_raw, last_S_raw, population=pop)  # xb:[B,H,N,C], Ah:[B,H,N,N]
                y_phys_raw = aux['y_phys']
                beta_bkn = aux['beta']
                y_phys_norm = y_norm(y_phys_raw)  # 关键！
                # (1) 主监督（对标签）：preds ≈ yb
                loss_sup = F.mse_loss(preds, yb)
                # (2) 物理拟合（对标签）：让物理外推也贴近标签（小权重稳尺度）
                loss_phys = F.mse_loss(y_phys_norm, yb)
                # (3) 一致性约束：让网络输出贴近物理外推
                loss_cons = F.mse_loss(preds, y_phys_norm)
                # (4) β 时间平滑：抑制逐步震荡（对时间维 L2）
                loss_smooth = ((beta_bkn[:, 1:, :] - beta_bkn[:, :-1, :]) ** 2).mean()
                # 推荐权重：先小后大地调 eta_cons；eta_phys 更小
                eta_cons, eta_phys, beta_smooth_coef = 0.2, 0.05, 1e-3
                total_loss = loss_sup + eta_cons * loss_cons + eta_phys * loss_phys + beta_smooth_coef * loss_smooth
            elif params['model'] == 'MEPO':
                xb = xb.permute(0, 3, 1, 2).contiguous()
                preds = model(xb, Ah, nI_raw, last_S_raw, population=pop)  # xb:[B,H,N,C], Ah:[B,H,N,N]
                total_loss = F.mse_loss(preds, yb) if params['loss'] == "MSE" else F.l1_loss(preds, yb)

            losses.append(float(total_loss.item()))
            P.append(preds.detach().cpu().numpy())
            Y.append(yb.detach().cpu().numpy())
    else:
        for batch in loader:
            xb, yb = [t.to(device) for t in batch]
            preds = model(xb).squeeze(-1)
            loss = F.l1_loss(preds, yb) if params['loss'] == "MAE" else F.mse_loss(preds, yb)
            losses.append(float(loss.item()))
            P.append(preds.detach().cpu().numpy())
            Y.append(yb.detach().cpu().numpy())

    pred = np.concatenate(P, axis=0)  # [W,K,N]
    true = np.concatenate(Y, axis=0)  # [W,K,N]

    # ★ 先反归一化回原始尺度，再做指标 ★
    pred_dn = y_denorm(pred)  # [W,K,N]
    true_dn = y_denorm(true)

    # metrics expect [W,N,K]
    pred_w_n_k = np.transpose(pred_dn, (0,2,1))
    true_w_n_k = np.transpose(true_dn, (0,2,1))
    return float(np.mean(losses)), summarize_metrics(pred_w_n_k, true_w_n_k), pred_w_n_k, true_w_n_k


# ===================== Main =====================
def main():
    parser = ArgumentParser()
    parser.add_argument('--data_pkl', type=str, default=str(Path(__file__).resolve().parent.parent.parent / "data" / "Data_meta_stgode_7.pkl"))
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).resolve().parent.parent / "outputs" / "neural" / "spatiotemporal"))
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--bootstrap_eval', type=int, default=30, help='测试集Bootstrap重采样次数（按窗口W）')
    parser.add_argument('--model', type=str, default='DCRNN',
                        choices=['STGCN', 'DCRNN'])

    parser.add_argument('--num_nodes', type=int, default=51)
    parser.add_argument('--history_window', type=int, default=7)
    parser.add_argument('--pred_window', type=int, default=3)
    parser.add_argument('--slide_step', type=int, default=1)
    parser.add_argument('--split_ratio', type=float, nargs='+', default=[8, 1, 1])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--learn_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--loss', type=str, choices=['MSE','MAE'], default='MSE')
    parser.add_argument('--optimizer', type=str, default='Adam')

    # DCRNN hyper-params
    parser.add_argument('--input_dim', type=int, default=4)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--enc_layers', type=int, default=1)
    parser.add_argument('--dec_layers', type=int, default=1)
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--tf_ratio', type=float, default=0.5)

    # dynamic A options for decoder
    parser.add_argument('--adj_future_mode', type=str, choices=['persist','future','ma'], default='persist')
    parser.add_argument('--adj_ma_m', type=int, default=3)

    parser.add_argument('--expdir', type=str, default='')

    args = parser.parse_args()
    params = vars(args)

    set_seed(params['seed'])
    device = torch.device(params['device'])

    # expdir
    if not params['expdir']:
        ts = time.strftime("%Y%m%d_%H%M%S")
        params['expdir'] = os.path.join(params['output_dir'], f"~{params['history_window']}_K{params['pred_window']}_{ts}")
    ensure_dir(params['expdir'])
    with open(os.path.join(params['expdir'], 'conf.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(params, f, allow_unicode=True)

    # load data
    Input_Data = load_pickle(params['data_pkl'])
    Input_Data['input_feat'] = Input_Data['input_feat'][:, :, :4]  # cum, death, S, newI
    feat_idx = {"cum":0, "death":1, "S":2, "newI":3}
    A_all = extract_A_all(Input_Data)  # [T,N,N]

    if params['model'] =='STGCN':
        # 1) 静态对称邻接
        A_static = symmetrize_aggregate(A_all, agg=params.get('agg', 'mean')).to(device)

    # get sliding windows
    dg = DataGenerator(history_window=params['history_window'],
                       pred_window=params['pred_window'],
                       data_split_ratio=params['split_ratio'],
                       feat_idx=feat_idx)
    dl = dg.get_data_loader(Input_Data, params)
    Pop = dl['population'].transpose(0, 1).to(params['device'])

    split_audit = build_split_audit(Input_Data, params, dl)
    save_yaml(os.path.join(params['expdir'], "split_audit.yaml"), split_audit)

    # dataloaders (with dynamic A sequences)
    train_loader, val_loader, test_loader = build_loaders(params, dl, A_all)

    if params['model'] == 'DCRNN':
        # pass device as a keyword argument to avoid positional mismatch
        model = build_model_by_name(params, device=device)
    elif params['model'] == 'STGCN':
        model = build_model_by_name(params, A_static=A_static, device=device)
    elif params['model'] == 'STAN':
        model = build_model_by_name(params, device=device)
    elif params['model'] == 'MEPO':
        model = build_model_by_name(params, device=device)

    # 额外保险：确保模型确实移动到目标设备
    model = model.to(device)
    optimizer = get_optimizer(model, params)

    # train
    best_path = os.path.join(params['expdir'], 'best.pt')
    # 构造反归一化函数：用 data_loader 里存的 'input_feat' 的 scaler，并定位 'newI' 通道
    y_norm = build_y_norm_from_loader(dl, target_feat_key='input_feat', target_feat_name='newI')
    y_denorm = build_y_denorm_from_loader(dl, target_feat_key='input_feat', target_feat_name='newI')
    # EarlyStopping：监控指标与模式二选一
    es = EarlyStopping(monitor='val_RMSE', mode='min', min_delta=0.0, patience=params['patience'],
                       ckpt_path=best_path)
    for ep in range(1, params['num_epochs']+1):
        if params['model'] == 'STAN':
            tr_loss = train_epoch(model, train_loader, device, params, pop=Pop, optimizer=optimizer,
                                  y_norm=y_norm)
            val_loss, val_metrics, _, _ = evaluate_epoch(model, val_loader, device, params, pop=Pop,
                                                         y_norm=y_norm, y_denorm=y_denorm)
        else:
            tr_loss = train_epoch(model, train_loader, device, params, optimizer=optimizer)
            val_loss, val_metrics, _, _ = evaluate_epoch(model, val_loader, device, params,
                                                         y_denorm=y_denorm)
        print(f"[Epoch {ep:03d}] train={tr_loss:.4f} val={val_loss:.4f} ")
              # f"RMSE={val_metrics['RMSE']:.2f} MAE={val_metrics['MAE']:.2f} "
              # f"MAPE%={val_metrics['MAPE%']:.2f} RAE={val_metrics['RAE']:.4f}"

        # 选择一个监控量：val_loss 或 val_metrics['RMSE']
        monitor_value = val_metrics['RMSE']  # 若监控 val_loss 则改为：val_loss

        stop, improved = es.step(monitor_value, model, epoch=ep)
        if improved:
            print(f"  ↳ Improved {es.monitor}: {monitor_value:.6f} (epoch {ep})  [ckpt saved]")
        if stop:
            print(f"Early stopping at epoch {ep} (best {es.monitor}={es.best:.6f} @ epoch {es.best_epoch})")
            break

    # 恢复最佳权重
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])

    # test
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])

    if params['model'] == 'STAN':
        test_loss, test_metrics, pred_w_n_k, true_w_n_k = evaluate_epoch(model, test_loader, device, params,
                                                                         pop=Pop, y_norm=y_norm, y_denorm=y_denorm)
    else:
        test_loss, test_metrics, pred_w_n_k, true_w_n_k = evaluate_epoch(model, test_loader, device, params,
                                                                         y_denorm=y_denorm)
    print(f"[Test] loss={test_loss:.4f} RMSE={test_metrics['RMSE']:.2f} "
          f"MAE={test_metrics['MAE']:.2f} MAPE%={test_metrics['MAPE%']:.2f} "
          f"wMAPE%={test_metrics['wMAPE%']:.2f} RAE={test_metrics['RAE']:.4f}")

    split_audit["test_arrays"] = {
        "shape": list(true_w_n_k.shape),
        "true_sha256": array_sha256(true_w_n_k),
        "prediction_sha256": array_sha256(pred_w_n_k),
    }
    save_yaml(os.path.join(params['expdir'], "split_audit.yaml"), split_audit)

    np.savez_compressed(os.path.join(params['expdir'], "dcrnn_dynamic_preds.npz"),
                        pred=pred_w_n_k, true=true_w_n_k, A_all=A_all)

    # ---------- Bootstrap 评估 ----------
    n_boot = int(params.get('bootstrap_eval', 30))
    if n_boot > 0:
        boot_meanstd, boot_details = bootstrap_eval_windows(pred_w_n_k, true_w_n_k, n_boot=n_boot, seed=params['seed'])
        print("[Bootstrap @ Test] (mean±std)")
        for k, v in boot_meanstd.items():
            print(f"  {k}: {v['mean']:.6f} ± {v['std']:.6f}")
        save_yaml(os.path.join(params['expdir'], "bootstrap_summary.yaml"), boot_meanstd)
        np.savez_compressed(os.path.join(params['expdir'], "bootstrap_details.npz"), details=boot_details)

    # 逐步长指标
    perK = per_horizon_metrics(pred_w_n_k, true_w_n_k)
    save_yaml(os.path.join(params['expdir'], "per_horizon.yaml"), {"per_horizon": perK})


if __name__ == "__main__":
    main()
