# -*- coding: utf-8 -*-
"""
A clean main runner for MEPOGNN that mirrors main_STAN's style.

Dependencies:
- DataGenerator from datasets/Utils_baseline.py (existing preprocessing/scaling/windowing/DataLoader output)
- MEPOGNN from models/model_MepoGNN.py (model definition unchanged)

"""
import os
from pathlib import Path
import json
import math
import pickle
import argparse
import numpy as np
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# ====== project modules ======
from datasets.Utils_baseline import DataGenerator      # data preprocessing / windowing / scaling / loading
from models.model_MepoGNN import MEPOGNN                  # MEPOGNN model definition


# ----------------- 工具函数 -----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metrics_from_arrays(y_true, y_pred, eps=1e-8):
    """原始尺度指标（与其他 baseline 口径一致）"""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae  = float(np.mean(np.abs(yt - yp)))
    msk  = np.abs(yt) > eps
    mape = float(np.mean(np.abs((yp[msk] - yt[msk]) / (yt[msk] + 1e-12))) * 100.0) if msk.any() else float('nan')
    wmape = float(np.sum(np.abs(yt - yp)) / (np.sum(np.abs(yt)) + eps) * 100.0)
    rae  = float(np.sum(np.abs(yt - yp)) / (np.sum(np.abs(yt - yt.mean())) + 1e-12))
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "wMAPE%": wmape, "RAE": rae}

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

def bootstrap_pointwise(y_true, y_pred, B=0, seed=42):
    """与 RNN 口径一致的点级 bootstrap；B=0 不执行"""
    if B is None or B <= 0:
        return None
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true).reshape(-1); yp = np.asarray(y_pred).reshape(-1)
    n = yt.shape[0]
    rows = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        rows.append(metrics_from_arrays(yt[idx], yp[idx]))
    import pandas as pd
    df = pd.DataFrame(rows)
    mean_d = df.mean().to_dict()
    std_d  = df.std(ddof=1).to_dict() if B >= 2 else {k: float("nan") for k in df.columns}
    return mean_d, std_d, df


# ============ 用历史新增估计“活跃感染”I（指数留存） ============
def active_I_from_hist_newI(hist_newI_WNH: np.ndarray, gamma: float = 1/7) -> np.ndarray:
    """
    输入：hist_newI_WNH [W, N, H]（原始尺度）
    输出：I_last_WN     [W, N]    （历史窗末时刻的 I 估计）
    公式：I(t) ≈ sum_{k=0..H-1} newI(t-k) * (1 - gamma)^k
    """
    W, N, H = hist_newI_WNH.shape
    k = np.arange(H, dtype=float)                  # [0..H-1]
    w = (1.0 - gamma) ** k                         # 权重
    # 让最近一天权重对应 k=0（hist 的最后一维是时间从旧到新）
    # hist_newI_WNH[..., -1] 对应 t，-2 对应 t-1 ...
    # 构造 [H] 的权重向量按从新到旧排列（k=0 → 最后一天）
    w_rev = w.copy()[::-1]
    # 按时间维做带权和
    I_last = np.tensordot(hist_newI_WNH, w_rev, axes=([2], [0]))   # [W,N]
    return I_last


# ============ 从 data_loader 准备 MEPO 需要的“对齐序列” ============
def prepare_mepo_inputs_from_loader(dl: dict, split: str, gamma=1/7):
    """
    返回：
      y_norm_stats: dict，给模型把原始尺度预测映射到标准化空间用于计算损失（z-score）
      y_denorm:     callable，把标准化数据还原到原始尺度（画图/评估）
      od_hist:      [W, H, N, N]  历史邻接（已按 train 段 max 归一化）
      sir_hist_raw: [W, H, N, 3]  原始尺度 S/I/R（这里把历史窗末状态在 H 维“平铺”）
      pop:          [N]           常住人口（若没有则 None）
    以及：
      W, N, H, K   四个维度
    """
    # 1) 维度
    bv = dl['baseline_views'][split]
    hist_incidence = bv['hist_incidence'].cpu().numpy() if isinstance(bv['hist_incidence'], torch.Tensor) else np.asarray(bv['hist_incidence'])  # [W,N,H]
    S0_last        = bv['S0_last'].cpu().numpy()        if isinstance(bv['S0_last'], torch.Tensor)        else np.asarray(bv['S0_last'])         # [W,N]
    od_hist        = bv['adj_hist'].cpu().numpy().astype(np.float32)  # [W,H,N,N]  已按 max_od 归一化（在 DataGenerator 中完成）
    W, N, H = hist_incidence.shape
    K = dl['baseline_views'][split]['future_incidence'].shape[-1]

    # 2) I_last 估计（原始尺度）
    I_last = active_I_from_hist_newI(hist_incidence, gamma=gamma)   # [W,N]

    # 3) R_last（闭合人群）
    pop = dl.get('population', None)
    if pop is not None:
        pop = np.asarray(pop).reshape(-1)            # [N]
        R_last = np.clip(pop[None, :] - S0_last - I_last, a_min=0.0, a_max=None)
    else:
        R_last = np.zeros_like(S0_last)

    # 4) sir_hist_raw：只需要历史窗末状态，但模型接口是 [B,H,N,3]，直接在 H 上平铺
    SIR_last = np.stack([S0_last, I_last, R_last], axis=-1)         # [W,N,3]
    sir_hist_raw = np.repeat(SIR_last[:, None, :, :], H, axis=1)    # [W,H,N,3]

    # 5) standardization/denormalization of y (z-score of newI)
    #    DataGenerator 里 scalers['input_feat'] 是 StandardScaler(所有通道)
    scaler_info = dl.get('scalers', None)
    feat_idx_map = dl.get('feat_idx', {"cum":0,"death":1,"S":2,"newI":3})
    newI_idx = feat_idx_map.get("newI", 3)
    if scaler_info is None or 'scaler' not in scaler_info:
        y_norm_stats = None
        y_denorm = lambda x: x
    else:
        scaler = scaler_info['scaler']
        mean_arr  = getattr(scaler, 'mean_', None)
        scale_arr = getattr(scaler, 'scale_', None)
        if mean_arr is None or scale_arr is None:
            y_norm_stats = None
            y_denorm = lambda x: x
        else:
            y_mean = float(mean_arr[newI_idx])
            y_std  = float(scale_arr[newI_idx]) if float(scale_arr[newI_idx]) != 0 else 1.0
            y_norm_stats = {'type': 'zscore', 'mean': y_mean, 'std': y_std}
            def _denorm(arr: np.ndarray) -> np.ndarray:
                return arr * y_std + y_mean
            y_denorm = _denorm

    return y_norm_stats, y_denorm, od_hist, sir_hist_raw, pop, (W, N, H, K)


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

# ----------------- 训练 / 评估（仿 main_STAN 风格） -----------------
def train_epoch(model, loader, device, y_norm_stats, od_hist, sir_hist_raw, max_od, optimizer, print_nan=False):
    """
    loader: the existing DataLoader, yielding (x_node, new_I, Sn, adj_matrix, last_SIR, y_norm)
      - x_node: [B, N, H, F]（标准化）
      - y_norm: [B, N, K, 1]（标准化标签）
    od_hist, sir_hist_raw：按窗口顺序 [W, ...] 准备好；要求 loader shuffle=False 以保证对齐
    """
    model.train()
    total, n = 0.0, 0
    ptr = 0  # 窗口指针
    loss_fn = torch.nn.MSELoss()

    for batch in loader:
        x_node, _, _, _, _, y_norm = batch
        # 统一形状
        xb = x_node.to(device).permute(0, 2, 1, 3).contiguous().float()   # [B,H,N,F]
        yb = y_norm.to(device).squeeze(-1).permute(0, 2, 1).contiguous().float()  # [B,K,N]

        B = xb.size(0)
        Ah = torch.as_tensor(od_hist[ptr:ptr+B], device=device, dtype=xb.dtype)         # [B,H,N,N]（归一化）
        SIR = torch.as_tensor(sir_hist_raw[ptr:ptr+B], device=device, dtype=xb.dtype)   # [B,H,N,3]（原始尺度）
        ptr += B

        y_pred_raw, y_pred_norm, _ = model(xb, SIR, Ah, max_od=float(max_od), y_norm_stats=y_norm_stats)
        loss = loss_fn(y_pred_norm, yb)

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                if print_nan: print("[warn] NaN grad")
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step()

        total += loss.item() * B
        n += B

    return total / max(n, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, device, y_denorm, y_norm_stats, od_hist, sir_hist_raw, max_od):
    model.eval()
    total, n = 0.0, 0
    ptr = 0
    loss_fn = torch.nn.MSELoss()
    preds_raw, trues_raw = [], []

    for batch in loader:
        x_node, _, _, _, _, y_norm = batch
        xb = x_node.to(device).permute(0, 2, 1, 3).contiguous().float()              # [B,H,N,F]
        yb = y_norm.to(device).squeeze(-1).permute(0, 2, 1).contiguous().float()    # [B,K,N]

        B = xb.size(0)
        Ah = torch.as_tensor(od_hist[ptr:ptr+B], device=device, dtype=xb.dtype)       # [B,H,N,N]
        SIR = torch.as_tensor(sir_hist_raw[ptr:ptr+B], device=device, dtype=xb.dtype) # [B,H,N,3]
        ptr += B

        y_pred_raw, y_pred_norm, _ = model(xb, SIR, Ah, max_od=float(max_od), y_norm_stats=y_norm_stats)
        loss = loss_fn(y_pred_norm, yb)
        total += loss.item() * B
        n += B

        # 反标准化到原始尺度计算指标
        y_pred_np = y_pred_raw.detach().cpu().numpy()               # [B,K,N]
        y_true_np = y_denorm(yb.detach().cpu().numpy())             # [B,K,N]
        preds_raw.append(y_pred_np); trues_raw.append(y_true_np)

    y_pred = np.concatenate(preds_raw, axis=0)   # [W,K,N]
    y_true = np.concatenate(trues_raw, axis=0)   # [W,K,N]
    # reshape to [W,N,K] to match the other result files
    y_pred_WNK = np.transpose(y_pred, (0, 2, 1))
    y_true_WNK = np.transpose(y_true, (0, 2, 1))
    return total / max(n, 1), metrics_from_arrays(y_true_WNK, y_pred_WNK), y_pred_WNK, y_true_WNK


# ----------------- 主入口 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pickle', type=str, default=str(Path(__file__).resolve().parent.parent.parent / "data" / "Data_meta_stgode_7.pkl"))
    parser.add_argument('--expdir', type=str, default=str(Path(__file__).resolve().parent.parent / "outputs" / "neural" / "mepognn" / "checkpoints"))
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).resolve().parent.parent / "outputs" / "neural" / "mepognn"))
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--num_nodes', type=int, default=51)
    parser.add_argument('--history_window', type=int, default=7)
    parser.add_argument('--pred_window', type=int, default=7)
    parser.add_argument('--slide_step', type=int, default=1)
    parser.add_argument('--split_ratio', type=float, nargs='+', default=[8, 1, 1])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--learn_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--od_scale_factor', type=float, default=3.0)
    parser.add_argument('--loss', type=str, choices=['MSE', 'MAE'], default='MSE')
    parser.add_argument('--optimizer', type=str, default='Adam')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--shuffle_train', type=int, default=0)   # 很重要：MEPO 需要与窗口对齐，这里默认关掉

    # 新增：监控指标 & 早停轮数
    parser.add_argument('--monitor', type=str, default='val_rmse', choices=['val_rmse', 'val_loss'])
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--n_bootstrap', type=int, default=100)     # 点级 bootstrap；0=不做
    args = parser.parse_args()
    params = vars(args)

    os.makedirs(args.expdir, exist_ok=True)
    with open(os.path.join(args.expdir, "conf_MEPO.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = torch.device(args.device)


    # 1) 载入数据
    with open(args.data_pickle, "rb") as f:
        Input_Data = pickle.load(f)
    # 只保留前 4 维 (cum, death, S, newI)
    Input_Data['input_feat'] = Input_Data['input_feat'][:, :, :4]

    DG = DataGenerator(history_window=args.history_window,
                       pred_window=args.pred_window,
                       data_split_ratio=args.split_ratio,
                       feat_idx={"cum":0, "death":1, "S":2, "newI":3})

    data_loader = DG.get_data_loader(Input_Data, params)   # train/val/test loaders, baseline views, and scalers

    train_loader = data_loader['train']
    val_loader   = data_loader['validate']
    test_loader  = data_loader['test']

    # 3) 为 MEPO 准备“对齐的”邻接序列和原始尺度 SIR 初值（严格按照窗口顺序）
    y_stats_tr, y_denorm_tr, od_hist_tr, sir_hist_tr, pop, (Wtr, N, H, K) = prepare_mepo_inputs_from_loader(data_loader, "train", gamma=1/7)
    y_stats_va, y_denorm_va, od_hist_va, sir_hist_va, _,   (Wva, _, _, _) = prepare_mepo_inputs_from_loader(data_loader, "validate", gamma=1/7)
    y_stats_te, y_denorm_te, od_hist_te, sir_hist_te, _,   (Wte, _, _, _) = prepare_mepo_inputs_from_loader(data_loader, "test",     gamma=1/7)

    # 标签标准化统计：用“训练”的（与 STAN 一致）
    y_norm_stats = y_stats_tr
    y_denorm     = y_denorm_tr
    # 邻接缩放回原始尺度在模型内部通过 max_od * od_scale_factor 完成
    max_od = float(data_loader.get('max_od', 1.0))

    # 4) 构建模型/优化器
    model = MEPOGNN(
        num_nodes=N, in_dim=4, in_len=H, out_len=K,
        dropout=args.dropout, glm_type='Dynamic',
        residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=256,
        kernel_size=2, blocks=2, layers=3, od_scale_factor=args.od_scale_factor
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)

    # train
    best_path = os.path.join(params['expdir'], 'best.pt')
    es = EarlyStopping(monitor='val_rmse', mode='min', min_delta=0.0, patience=params['patience'],
                       ckpt_path=best_path)

    print(f"Starting MepoGNN training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6) 训练/验证
    for ep in range(1, args.num_epochs + 1):
        tr_loss = train_epoch(model, train_loader, device, y_norm_stats, od_hist_tr, sir_hist_tr, max_od, optimizer)
        va_loss, va_metrics, _, _ = evaluate_epoch(model, val_loader, device, y_denorm, y_norm_stats, od_hist_va,
                                                   sir_hist_va, max_od)

        if (ep % args.print_every == 0) or (ep == 1):
            print(f"[MEPOGNN|Epoch {ep:03d}] train={tr_loss:.4f}  val={va_loss:.4f}  {va_metrics}")

        monitor_value = va_metrics['RMSE'] if args.monitor == 'val_rmse' else va_loss
        stop, improved = es.step(monitor_value, model, epoch=ep)
        if improved:
            print(f"  ↳ Improved {args.monitor}: {monitor_value:.6f} (epoch {ep})  [ckpt saved]")
        if stop:
            print(f"Early stopping at epoch {ep} (best {args.monitor}={es.best:.6f} @ epoch {es.best_epoch})")
            break

    # test
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])

    te_loss, te_metrics, te_pred_WNK, te_true_WNK = evaluate_epoch(model, test_loader, device, y_denorm,
                                                                   y_norm_stats, od_hist_te, sir_hist_te, max_od)
    print(f"[MEPOGNN|Test] loss={te_loss:.4f} RMSE={te_metrics['RMSE']:.2f} MAE={te_metrics['MAE']:.2f} "
          f"MAPE%={te_metrics['MAPE%']:.2f} wMAPE%={te_metrics['wMAPE%']:.2f} "
          f"RAE={te_metrics['RAE']:.4f}")

    # 保存预测
    os.makedirs(args.output_dir, exist_ok=True)
    out_npz = os.path.join(args.output_dir, f"mepognn_pred_H{H}_K{K}.npz")
    np.savez_compressed(out_npz, y_pred=te_pred_WNK, y_true=te_true_WNK, metrics=te_metrics)
    print(f"[Saved] {out_npz}")

    # 7) （可选）bootstrap（点级）
    if args.n_bootstrap and args.n_bootstrap > 0:
        bs = bootstrap_pointwise(te_true_WNK, te_pred_WNK, B=args.n_bootstrap, seed=args.seed)
        if bs is not None:
            mean_d, std_d, df = bs
            print("[Bootstrap@Test] (mean±std)")
            for k in ['RMSE','MAE','MAPE%','wMAPE%','RAE']:
                print(f"  {k}: {mean_d[k]:.6f} ± {std_d[k]:.6f}")
            df.to_csv(os.path.join(args.output_dir, f"bootstrap_pointwise_MEPO_B{args.n_bootstrap}.csv"), index=False)


if __name__ == "__main__":
    main()
