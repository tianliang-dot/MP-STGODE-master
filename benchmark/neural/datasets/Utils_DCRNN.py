import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader



class ODDataset(Dataset):
    def __init__(self, inputs: dict, output: torch.Tensor, mode: str):
        self.mode = mode
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, item):
        return (self.inputs['x_node'][item],
                self.inputs['new_I'][item],
                self.inputs['Sn'][item],
                self.inputs['adj_matrix'][item],
                self.inputs['last_SIR'][item],
                self.output[item])

    def prepare_xy(self, inputs: dict, output: torch.Tensor):
        return inputs, output


class PaddedDataLoader(DataLoader):
    def __init__(self, *args, pad_value=0, **kwargs):
        super(PaddedDataLoader, self).__init__(*args, **kwargs)
        self.pad_value = pad_value

    def __iter__(self):
        for data in super(PaddedDataLoader, self).__iter__():
            if any(tensor.shape[0] < self.batch_size for tensor in data):
                data = self.pad_batch(data)
            yield data

    def pad_batch(self, data):
        padded_data = []
        for tensor in data:
            pad_size = self.batch_size - tensor.shape[0]
            if pad_size > 0:
                pad_left = pad_size // 2
                pad_right = pad_size - pad_left
                padding = (0, 0) * (tensor.dim() - 1) + (pad_left, pad_right)
                tensor = torch.nn.functional.pad(tensor, padding, value=self.pad_value)
            padded_data.append(tensor)
        return padded_data


class DataGenerator(object):
    def __init__(self, history_window: int, pred_window: int, data_split_ratio: tuple,
                 feat_idx: dict = None):
        self.history_window = history_window
        self.pred_window = pred_window
        self.data_split_ratio = data_split_ratio
        self.time_axis_map = {'input_feat': 1,
                              'adj_matrix': 0,
                              'seri_feat': 1,
                              'edge_stack_feat': 1,
                              'SIR_data': 1}
        # feature layout: cum=0, death=1, S=2, newI=3 (configurable)
        self.feat_idx = feat_idx or {"newI": -1, "S": 2, "cum": None, "death": None}

    # ---------- split helper ----------
    def split2len(self, data_len: int):
        mode_len = dict()
        mode_len['train'] = int(self.data_split_ratio[0] / sum(self.data_split_ratio) * data_len)
        mode_len['validate'] = int(self.data_split_ratio[1] / sum(self.data_split_ratio) * data_len)
        mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
        return mode_len

    def split_data(self, data: dict, history_window: int):
        """分割数据集为训练/验证/测试集（时间轴隔离；val/test 向前回退 H 天以保证首窗足够历史）"""
        total_time_steps = data['input_feat'].shape[self.time_axis_map['input_feat']]
        mode_len = self.split2len(total_time_steps)

        train_end = mode_len['train']
        val_start = train_end - history_window
        val_end = mode_len['train'] + mode_len['validate']
        test_start = val_end - history_window
        test_end = total_time_steps

        train_data, val_data, test_data = {}, {}, {}
        for k, v in data.items():
            if k in self.time_axis_map:
                time_dim = self.time_axis_map[k]
                train_slice = [slice(None) if i != time_dim else slice(0, mode_len['train']) for i in range(v.ndim)]
                val_slice = [slice(None) if i != time_dim else slice(val_start, val_end) for i in range(v.ndim)]
                test_slice = [slice(None) if i != time_dim else slice(test_start, test_end) for i in range(v.ndim)]
                train_data[k] = v[tuple(train_slice)]
                val_data[k] = v[tuple(val_slice)]
                test_data[k] = v[tuple(test_slice)]
            else:
                # 无时间维（人口等）广播
                if isinstance(v, torch.Tensor):
                    train_data[k] = val_data[k] = test_data[k] = v.clone()
                else:
                    train_data[k] = val_data[k] = test_data[k] = v
        return train_data, val_data, test_data

    # ---------- sliding windows ----------
    def create_sliding_windows(self, data, history_window, pred_window, slide_step, mode):
        """
        Generate sliding-window samples, preserving the existing tensor layout:
        返回 tuple:
          0: x_node         [W,N,H,F]  （这里直接用 data['input_feat'] 切出来）
          1: new_I_future   [W,N,K]    （原始尺度）
          2: S_future       [W,N,K]    （原始尺度）
          3: adj_hist       [W,H,N,N]  （只历史部分）
          4: last_SIR       [W,N,2]
          5: y_newI         [W,N,K]    （label，原始尺度）
        """
        nodes, time_steps, features = data['input_feat'].shape
        input_data = torch.from_numpy(data['input_feat']).to(dtype=torch.float32)
        # 邻接（T,N,N）→ edge_feat[T,N,N]
        edge_feat = data['adj_matrix'].to(dtype=torch.float32).squeeze(-1)

        idx_S = self.feat_idx.get("S", 2)
        idx_new = self.feat_idx.get("newI", -1)

        S = input_data[:, :, idx_S].unsqueeze(-1)    # [N,T,1]
        newI = input_data[:, :, idx_new].unsqueeze(-1)  # [N,T,1]
        SIR = torch.cat((S, newI), dim=-1)           # [N,T,2]

        x_batch, new_I, Sn, adj_matrix, last_SIR, y_newI = [], [], [], [], [], []

        for t in range(0, time_steps - history_window - pred_window + 1, slide_step):
            x_batch.append(input_data[:, t:t + history_window, :])             # [N,H,F]
            adj_matrix.append(edge_feat[t:t + history_window, :, :])           # [H,N,N]
            last_SIR.append(SIR[:, t + history_window, :])                     # [N,2]
            new_I.append(input_data[:, t + history_window:t + history_window + pred_window, 3])  # [N,K]
            Sn.append(input_data[:, t + history_window:t + history_window + pred_window, 2])     # [N,K]
            y_newI.append(input_data[:, t + history_window:t + history_window + pred_window, [-1]])  # [N,K,1]

        to_tensor = lambda x: torch.stack(x) if x else torch.empty(0)
        return (
            to_tensor(x_batch),
            to_tensor(new_I),
            to_tensor(Sn),
            to_tensor(adj_matrix),
            to_tensor(last_SIR),
            to_tensor(y_newI)
        )

    # ---------- contact windows for Meta-SIR ----------
    def make_contact_windows(self, A_norm, start, end, H, K, step, mode='hold_last'):
        """
        A_norm: (T,N,N), scaled by the training-set maximum
        返回:
          hist_contact:   (W,H,N,N)
          future_contact: (W,K,N,N)
          win_starts:     (W,)
        """
        hist_list, fut_list, starts = [], [], []
        t0 = start
        while t0 + H + K <= end:
            hist_list.append(A_norm[t0: t0 + H])  # (H,N,N)
            if mode == 'hold_last':               # 默认：冻结历史最后一天（无泄漏）
                last = A_norm[t0 + H - 1]
                fut = np.repeat(last[None, ...], K, axis=0)
            elif mode == 'history_mean':          # 历史均值（无泄漏）
                meanH = A_norm[t0: t0 + H].mean(axis=0, keepdims=True)
                fut = np.repeat(meanH, K, axis=0)
            elif mode == 'use_future':            # 仅上界/回放，会泄漏
                fut = A_norm[t0 + H: t0 + H + K]
            else:
                raise ValueError(f"Unknown forecast_H_mode: {mode}")
            fut_list.append(fut)
            starts.append(t0)
            t0 += step
        return (np.asarray(hist_list), np.asarray(fut_list), np.asarray(starts, dtype=int))

    # --------------------- 主入口 ---------------------
    def get_data_loader(self, data: dict, params: dict):
        """
        返回：
        - 'train'/'validate'/'test' DataLoaders
        - 'scalers': input_feat 标准化器
        - 'max_od': 邻接归一化因子（train 段 max）
        - 'baseline_views': 供基线直接使用的“原始尺度”视图（含 adj_hist/adj_future）
        - 'feat_idx': 特征映射
        - 'population': 若 data 中包含 Population，就带出来
        - 'win_starts': 各 split 的窗口起点（绝对索引）
        - 'series': 全序列的 newI/S（原始尺度）
        """
        H, K, step = params['history_window'], params['pred_window'], params['slide_step']
        device = params.get('device', 'cpu')
        forecast_H_mode = params.get('forecast_H_mode', 'hold_last')

        # === 计算 split 边界（前移，供后续邻接切窗使用）===
        T_total = data['input_feat'].shape[self.time_axis_map['input_feat']]
        mode_len = self.split2len(T_total)
        train_end = mode_len['train']
        val_start = train_end - H
        val_end = mode_len['train'] + mode_len['validate']
        test_start = val_end - H
        test_end = T_total
        bounds = {"train": (0, train_end), "validate": (val_start, val_end), "test": (test_start, test_end)}

        # === 时间切分（用于 GNN/特征标准化）===
        train_data, validate_data, test_data = self.split_data(data, H)

        # === 深拷贝原始数据（未标准化，用于 baseline）===
        original_train = copy.deepcopy(train_data)
        original_validate = copy.deepcopy(validate_data)
        original_test = copy.deepcopy(test_data)

        # === 标准化动态特征（不含 adj_matrix）===
        dynamic_features = ['input_feat', 'seri_feat', 'SIR_data', 'edge_stack_feat']
        scalers = {}
        for feat in dynamic_features:
            if feat not in train_data:
                continue

            is_tensor = isinstance(train_data[feat], torch.Tensor)
            time_dim = self.time_axis_map[feat]

            data_np = train_data[feat].numpy() if is_tensor else train_data[feat].copy()
            reshaped = np.moveaxis(data_np, time_dim, 1)              # -> (samples, time, features)
            n_samples, n_timesteps, n_features = reshaped.shape

            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(reshaped.reshape(-1, n_features))
            scalers[feat] = {"scaler": scaler, "is_tensor": is_tensor, "time_dim": time_dim}
            scaled_train = scaled_train.reshape(n_samples, n_timesteps, n_features)
            scaled_data = np.moveaxis(scaled_train, 1, time_dim)
            train_data[feat] = torch.from_numpy(scaled_data) if is_tensor else scaled_data

        for dataset in [validate_data, test_data]:
            for feat in dynamic_features:
                if feat not in dataset or feat not in scalers:
                    continue
                scaler_info = scalers[feat]
                scaler = scaler_info["scaler"]
                is_tensor = scaler_info["is_tensor"]
                time_dim = scaler_info["time_dim"]
                data_np = dataset[feat].numpy() if is_tensor else dataset[feat].copy()
                reshaped = np.moveaxis(data_np, time_dim, 1)
                n_samples, n_timesteps, num_features = reshaped.shape
                scaled = scaler.transform(reshaped.reshape(-1, num_features)).reshape(n_samples, n_timesteps, num_features)
                scaled = np.moveaxis(scaled, 1, time_dim)
                dataset[feat] = torch.from_numpy(scaled) if is_tensor else scaled

        # === 生成滑窗（原始 & 归一化）===
        orig_train_feats = self.create_sliding_windows(original_train, H, K, step, mode='train')
        orig_val_feats   = self.create_sliding_windows(original_validate, H, K, step, mode='validate')
        orig_test_feats  = self.create_sliding_windows(original_test, H, K, step, mode='test')

        norm_train_feats = self.create_sliding_windows(train_data, H, K, step, mode='train')
        norm_val_feats   = self.create_sliding_windows(validate_data, H, K, step, mode='validate')
        norm_test_feats  = self.create_sliding_windows(test_data, H, K, step, mode='test')

        # === adjacency scaling (by the training-set maximum) ===
        # 用原始全时段邻接计算同一个 max_od，避免形状误用
        Aall = data['adj_matrix']
        if isinstance(Aall, torch.Tensor):
            Aall_np = Aall.detach().cpu().numpy()
        else:
            Aall_np = np.asarray(Aall)
        max_od = float(np.max(Aall_np[:train_end]))  # train 段最大值
        # GNN 用的历史邻接（窗口级）：除以同一个 max_od
        train_adj = norm_train_feats[3] / (max_od + 1e-12)
        val_adj   = norm_val_feats[3]   / (max_od + 1e-12)
        test_adj  = norm_test_feats[3]  / (max_od + 1e-12)
        # Meta-SIR 用的“全时段邻接”（非窗口级）：同样除以 max_od，再切窗出历史/未来序列
        Aall_norm = Aall_np / (max_od + 1e-12)


        # === 三个 split：切出 (W,H,N,N)/(W,K,N,N) 和窗口起点 ===
        (tr_hist, tr_fut, tr_starts) = self.make_contact_windows(Aall_norm, *bounds['train'], H, K, step, mode=forecast_H_mode)
        (va_hist, va_fut, va_starts) = self.make_contact_windows(Aall_norm, *bounds['validate'], H, K, step, mode=forecast_H_mode)
        (te_hist, te_fut, te_starts) = self.make_contact_windows(Aall_norm, *bounds['test'], H, K, step, mode=forecast_H_mode)

        # === PyTorch DataLoaders for the neural models ===
        feat_dicts = {
            'train': {
                'x_node': norm_train_feats[0].float().to(device),
                'new_I': orig_train_feats[1].float().to(device),   # 原始尺度
                'Sn':     orig_train_feats[2].float().to(device),  # 原始尺度
                'adj_matrix': train_adj.unsqueeze(-1).float().to(device),  # [W,H,N,N,1]
                'last_SIR': norm_train_feats[4].float().to(device),
            },
            'validate': {
                'x_node': norm_val_feats[0].float().to(device),
                'new_I': orig_val_feats[1].float().to(device),
                'Sn':     orig_val_feats[2].float().to(device),
                'adj_matrix': val_adj.unsqueeze(-1).float().to(device),
                'last_SIR': norm_val_feats[4].float().to(device),
            },
            'test': {
                'x_node': norm_test_feats[0].float().to(device),
                'new_I': orig_test_feats[1].float().to(device),
                'Sn':     orig_test_feats[2].float().to(device),
                'adj_matrix': test_adj.unsqueeze(-1).float().to(device),
                'last_SIR': norm_test_feats[4].float().to(device),
            }
        }

        train_dataset = ODDataset(inputs=feat_dicts['train'],    output=norm_train_feats[-1].to(device), mode='train')
        validate_dataset = ODDataset(inputs=feat_dicts['validate'], output=norm_val_feats[-1].to(device), mode='validate')
        test_dataset = ODDataset(inputs=feat_dicts['test'],        output=norm_test_feats[-1].to(device), mode='test')

        make_torch_loaders = params.get('make_torch_loaders', True)
        if make_torch_loaders:
            bs_train = params.get('gnn_batch_size', params.get('batch_size', 64))
            bs_eval  = params.get('eval_batch_size', bs_train)
            data_loader_torch = {
                'train':    DataLoader(train_dataset, batch_size=bs_train, shuffle=params.get('shuffle_train', True),
                                       drop_last=params.get('drop_last_train', False)),
                'validate': DataLoader(validate_dataset, batch_size=bs_eval, shuffle=False, drop_last=False),
                'test':     DataLoader(test_dataset, batch_size=bs_eval, shuffle=False, drop_last=False),
            }
        else:
            data_loader_torch = {'train': None, 'validate': None, 'test': None}

        # === baseline 视图（原始尺度 + 接触序列）===
        def build_baseline_view(orig_feats, adj_hist, adj_fut):
            """
            返回:
              hist_incidence:   [W,N,H] 历史新增（原始尺度）
              future_incidence: [W,N,K] 未来新增（原始尺度）
              S_hist:           [W,N,H] 历史易感（原始尺度）
              S0_t0:            [W,N]   窗口起点 S(t0)
              S0_last:          [W,N]   历史窗最后一天 S(t0+H-1)
              adj_hist:         [W,H,N,N] 历史接触（归一化：/max_od）
              adj_future:       [W,K,N,N] 未来接触（同归一化）
            """
            x_node_raw = orig_feats[0]                      # [W,N,H,F] 原始
            idx_S  = self.feat_idx.get("S", 2)
            idx_nI = self.feat_idx.get("newI", -1)

            hist_incidence   = x_node_raw[..., idx_nI]      # [W,N,H]
            S_hist           = x_node_raw[..., idx_S]       # [W,N,H]
            S0_t0            = S_hist[..., 0]               # [W,N]
            S0_last          = S_hist[..., -1]              # [W,N]
            future_incidence = orig_feats[1]                # [W,N,K]

            return {
                "hist_incidence":   hist_incidence,
                "future_incidence": future_incidence,
                "S_hist":           S_hist,
                "S0_t0":            S0_t0,
                "S0_last":          S0_last,
                "adj_hist":         adj_hist,
                "adj_future":       adj_fut,
            }

        baseline_views = {
            "train":    build_baseline_view(orig_train_feats, tr_hist, tr_fut),
            "validate": build_baseline_view(orig_val_feats,   va_hist, va_fut),
            "test":     build_baseline_view(orig_test_feats,  te_hist, te_fut),
        }

        # === 汇总输出 ===
        # 全序列（原始尺度）供 baseline 辅助：newI/S
        arr = data['input_feat'].cpu().numpy() if isinstance(data['input_feat'], torch.Tensor) else data['input_feat']
        idx_new, idx_S = self.feat_idx.get("newI", 3), self.feat_idx.get("S", 2)

        data_loader = {
            'scalers': scalers.get('input_feat', None),
            'max_od': max_od,
            'baseline_views': baseline_views,
            'feat_idx': self.feat_idx,
            'population': data['static_feat'][:, [0]] if 'static_feat' in data else None,
            'win_starts': {"train": tr_starts, "validate": va_starts, "test": te_starts},
            'series': {"newI": arr[:, :, idx_new], "S": arr[:, :, idx_S]},
            **data_loader_torch
        }

        return data_loader


# ===================== Data glue: slice dynamic A_t per window =====================

def extract_A_all(Input_Data) -> np.ndarray:
    """
    Return A_all: [T,N,N] (float32). If not present, identity.
    """
    if 'adj_matrix' in Input_Data and Input_Data['adj_matrix'] is not None:
        A = Input_Data['adj_matrix']
        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().numpy()
        A = np.asarray(A)
        if A.ndim == 4:  # [T,N,N,1]
            A = A[..., 0]
        return A.astype(np.float32)  # [T,N,N]
    # fallback identity
    N = Input_Data['input_feat'].shape[0]
    T = Input_Data['input_feat'].shape[1]
    A = np.zeros((T, N, N), dtype=np.float32)
    for t in range(T):
        A[t] = np.eye(N, dtype=np.float32)
    return A


def _reconstruct_starts(W, split, slide_step, T_total, H, K, split_ratio):
    tr, va, te = [int(x) for x in split_ratio]
    tot = tr + va + te
    t_tr = int(np.floor(T_total * tr / tot))
    t_va = int(np.floor(T_total * va / tot))
    if split == "train":
        return np.arange(W) * slide_step
    elif split == "validate":
        return np.arange(W) * slide_step + max(t_tr - (H-1), 0)
    else:  # "test"
        return np.arange(W) * slide_step + max(t_tr + t_va - (H-1), 0)


def _build_window_adj_sequences_from_Aall(A_all: np.ndarray, starts: np.ndarray,
                                          H: int, K: int,
                                          future_mode: str = "persist", ma_m: int = 3):
    """
    A_all: [T,N,N]  全时序邻接（有向加权）
    starts: [W]     每个窗口的起点（全局时间索引）
    返回:
      Ahist: [W,H,N,N], Afut: [W,K,N,N]
    注：不在这里做归一化；扩散支撑处会做随机游走归一化（更稳）
    """
    W = len(starts)
    T_total, N, _ = A_all.shape
    Ahist = np.zeros((W, H, N, N), dtype=np.float32)
    Afut  = np.zeros((W, K, N, N), dtype=np.float32)
    for i, s in enumerate(starts):
        s = int(s)
        # 历史
        for t in range(H):
            Ahist[i, t] = A_all[np.clip(s + t, 0, T_total - 1)]
        # 未来
        if future_mode == "future":
            for t in range(K):
                Afut[i, t] = A_all[np.clip(s + H + t, 0, T_total - 1)]
        elif future_mode == "ma":
            m = max(1, ma_m)
            st = max(s + H - m, 0); ed = s + H
            Af = A_all[st:ed].mean(axis=0, keepdims=True)  # [1,N,N]
            Afut[i] = np.repeat(Af, K, axis=0)
        else:  # 'persist'
            Af = A_all[np.clip(s + H - 1, 0, T_total - 1)]
            Afut[i] = np.repeat(Af[None, ...], K, axis=0)
    return torch.from_numpy(Ahist), torch.from_numpy(Afut)


def _get_dataset_from_dl_split(dl, split: str):
    """兼容 dl[split] 为 DataLoader 或 dict 的两种写法"""
    obj = dl[split]
    if isinstance(obj, DataLoader):
        return obj.dataset
    if isinstance(obj, dict) and 'dataset' in obj:
        return obj['dataset']
    raise TypeError(f"dl['{split}'] 必须是 DataLoader 或包含 'dataset' 键的 dict，当前类型：{type(obj)}")


def make_dataloaders_dynamic(dl: dict, split: str, A_all: np.ndarray,
                             batch_size: int, slide_step: int,
                             H: int, K: int, split_ratio=(8,1,1),
                             future_mode: str = "persist", ma_m: int = 3,
                             use_all_features: bool = True, feature_index: int = 3):
    """
    Return DataLoader yielding:
      xb: [B,H,N,C_in], yb: [B,K,N], A_hist: [B,H,N,N], A_fut: [B,K,N,N]
    - features/labels come from the existing dataloader (dl[split])
    - 邻接序列由 A_all + win_starts 生成
    - C_in = 4 (use_all_features=True)；或 1 (选 feature_index 那一维)
    """
    ds = _get_dataset_from_dl_split(dl, split)
    X = ds.inputs['x_node']  # 期望 [W, N, H, F]
    Y = ds.output.permute(0, 2, 1, 3)  # 期望 [W, K, N] 或 [W, K, N, 1]

    # === 取原始尺度历史（与窗口严格对齐） ===
    bv = dl['baseline_views'][split]  # {'hist_incidence':[W,N,H], 'S_hist':[W,N,H], 'S0_last':[W,N], ...}
    newI_hist_raw = bv['hist_incidence'].permute(0, 2, 1).contiguous().float()  # -> [W,H,N]
    S0_last_raw = bv['S0_last'].contiguous().float()

    # 转为 torch，不改变原始数据
    if not isinstance(X, torch.Tensor): X = torch.as_tensor(X)
    if not isinstance(Y, torch.Tensor): Y = torch.as_tensor(Y)
    if Y.ndim == 4 and Y.shape[-1] == 1:  # [W,K,N,1] -> [W,K,N]
        Y = Y.squeeze(-1)

    W, N, H_ds, F = X.shape
    K_ds = Y.shape[1]
    assert H_ds == H, f"H mismatch: dataset H={H_ds}, arg={H}"
    assert K_ds == K, f"K mismatch: dataset K={K_ds}, arg={K}"

    # 取窗口起点
    if isinstance(dl, dict) and 'win_starts' in dl and split in dl['win_starts']:
        starts = np.asarray(dl['win_starts'][split], dtype=np.int64)
        assert len(starts) == W, f"win_starts length {len(starts)} != W {W}"
    else:
        starts = _reconstruct_starts(W, split, slide_step, A_all.shape[0], H, K, split_ratio)

    # 由 A_all + starts 生成 A_hist / A_fut
    Ahist, Afut = _build_window_adj_sequences_from_Aall(
        A_all=A_all, starts=starts, H=H, K=K,
        future_mode=future_mode, ma_m=ma_m
    )  # -> torch: [W,H,N,N], [W,K,N,N]

    # 组装输入（按需要保留 4 维或只取一维）
    if use_all_features:
        xb = X.permute(0, 2, 1, 3).contiguous().float()                     # [W,H,N,4]
    else:
        xb = X.permute(0, 2, 1, 3)[..., feature_index:feature_index+1].contiguous().float()  # [W,H,N,1]
    yb = Y.contiguous().float()                                            # [W,K,N]

    ds_torch = TensorDataset(xb, yb, Ahist, Afut, newI_hist_raw, S0_last_raw)
    # Training windows are shuffled; validation/test remain chronological so
    # saved predictions and window-bootstrap samples are reproducible.
    return DataLoader(ds_torch, batch_size=batch_size, shuffle=(split == "train"), drop_last=False)



def get_y_norm_stats_from_loader(data_loader, target_feat_key='input_feat', target_feat_name='newI'):
    """
    返回 {'type':'zscore','mean':float,'std':float}；若拿不到 scaler，就返回 None
    与 build_y_denorm_from_loader 使用同一套 mean_/scale_/var_。
    """
    scaler_info = data_loader.get('scalers', None)
    if scaler_info is None:
        return None
    scaler = scaler_info.get('scaler', None) if isinstance(scaler_info, dict) else None
    if scaler is None:
        return None

    feat_idx_map = data_loader.get('feat_idx', {})
    if target_feat_name not in feat_idx_map:
        return None
    idx = feat_idx_map[target_feat_name]

    mean_arr  = getattr(scaler, 'mean_', None)
    scale_arr = getattr(scaler, 'scale_', None)
    var_arr   = getattr(scaler, 'var_', None)
    if mean_arr is None:
        return None

    y_mean = float(mean_arr[idx])
    if scale_arr is not None:
        y_std = float(scale_arr[idx])
    elif var_arr is not None:
        y_std = float(np.sqrt(var_arr[idx]))
    else:
        y_std = 1.0

    return {'type': 'zscore', 'mean': y_mean, 'std': y_std}

def build_y_norm_from_loader(data_loader, target_feat_key='input_feat', target_feat_name='newI'):
    """
    返回一个可调用对象 y_norm(arr_np)：将“原始尺度”的 [W,K,N] 或 [B,K,N] numpy 数组
    Normalize to the same standardized scale used in training (z-score: (x-mean)/std).
    若取不到 scaler/索引，则返回恒等映射。
    """
    stats = get_y_norm_stats_from_loader(data_loader, target_feat_key, target_feat_name)
    if stats is None:
        return lambda x: x

    y_mean = stats['mean']
    y_std  = stats['std'] if stats['std'] != 0 else 1.0

    def _norm(arr_np: np.ndarray) -> np.ndarray:
        # arr_np: [W,K,N] 或 [B,K,N]，数值为“原始尺度”的 newI
        return (arr_np - y_mean) / y_std

    return _norm


def build_y_denorm_from_loader(data_loader, target_feat_key='input_feat', target_feat_name='newI'):
    """
    返回一个可调用对象 y_denorm(arr_np)：
      arr_np: [W, K, N] 的 numpy 数组（标准化后的 y）
      返回 : 反归一化到原始尺度的 [W, K, N]
    data_loader['scalers'] 里应包含针对 target_feat_key 的 StandardScaler；
    target_feat_name 的索引来自 data_loader['feat_idx']。
    """
    # 1) 取 scaler
    scaler_info = data_loader.get('scalers', None)
    if scaler_info is None:
        # 没存 scaler（或没做标准化），就原样返回
        return lambda x: x

    scaler = scaler_info.get('scaler', None) if isinstance(scaler_info, dict) else None
    if scaler is None:
        return lambda x: x

    # 2) 取目标通道索引（如 newI）
    feat_idx_map = data_loader.get('feat_idx', {})
    if target_feat_name not in feat_idx_map:
        # 没找到索引，保底不反归一化
        return lambda x: x
    idx = feat_idx_map[target_feat_name]

    # 3) 取该通道的 mean / std（StandardScaler）
    mean_arr  = getattr(scaler, 'mean_', None)
    scale_arr = getattr(scaler, 'scale_', None)
    var_arr   = getattr(scaler, 'var_', None)

    if mean_arr is None:
        return lambda x: x

    y_mean = float(mean_arr[idx])
    if scale_arr is not None:
        y_std = float(scale_arr[idx])
    elif var_arr is not None:
        y_std = float(np.sqrt(var_arr[idx]))
    else:
        y_std = 1.0

    # 4) 返回反归一化函数
    def _denorm(arr_np: np.ndarray) -> np.ndarray:
        # arr_np: [W, K, N]
        return arr_np * y_std + y_mean

    return _denorm


class Normalizer:
    """
    一个类，用于处理 y 的标准化和反标准化。
    支持多种标准化方法：Z-score 或 Min-Max 标准化。
    """

    def __init__(self, stats, norm_type='zscore'):
        """
        stats: dict，包含 'mean' 和 'std'（Z-score）或者 'min' 和 'max'（Min-Max）
        """
        self.norm_type = norm_type
        if norm_type == 'zscore':
            self.mean = stats['mean']
            self.std = stats['std']
        elif norm_type == 'minmax':
            self.min = stats['min']
            self.max = stats['max']
        elif norm_type == 'logz':
            self.mu = stats['mu']
            self.sigma = stats['sigma']
        else:
            raise ValueError("Invalid norm_type. Must be one of ['zscore', 'minmax', 'logz']")

    def norm(self, y_raw):
        """ 标准化操作 """
        if self.norm_type == 'zscore':
            return (y_raw - self.mean) / (self.std + 1e-6)
        elif self.norm_type == 'minmax':
            return (y_raw - self.min) / (self.max - self.min + 1e-6)
        elif self.norm_type == 'logz':
            y_log = np.log1p(np.clip(y_raw, 0, None))
            return (y_log - self.mu) / (self.sigma + 1e-6)

    def denorm(self, y_norm):
        """ 反标准化操作 """
        if self.norm_type == 'zscore':
            return y_norm * (self.std + 1e-6) + self.mean
        elif self.norm_type == 'minmax':
            return y_norm * (self.max - self.min + 1e-6) + self.min
        elif self.norm_type == 'logz':
            y_log = y_norm * (self.sigma + 1e-6) + self.mu
            return np.expm1(y_log)

    def get_norm_func(self):
        """ 获取标准化函数（用于批处理或其他场景） """
        return self.norm

    def get_denorm_func(self):
        """ 获取反标准化函数（用于评估/反归一化） """
        return self.denorm
