import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


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
                              'SIR_data': 1
                              }
        # 新增：特征索引映射（默认保持旧逻辑：S=2，新增=最后一列）
        # feature layout: cum=0, death=1, S=2, newI=3; pass this mapping explicitly
        self.feat_idx = feat_idx or {"newI": -1, "S": 2, "cum": None, "death": None}

    def split2len(self, data_len: int):
        mode_len = dict()
        mode_len['train'] = int(self.data_split_ratio[0] / sum(self.data_split_ratio) * data_len)
        mode_len['validate'] = int(self.data_split_ratio[1] / sum(self.data_split_ratio) * data_len)
        mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
        return mode_len

    def split_data(self, data: dict, history_window: int):
        """分割数据集为训练/验证/测试集（时间轴隔离）"""
        total_time_steps = data['input_feat'].shape[self.time_axis_map['input_feat']]
        mode_len = self.split2len(total_time_steps)

        train_end = mode_len['train']
        val_start = train_end - history_window  # 扩展 7 天
        val_end = mode_len['train'] + mode_len['validate']
        test_start = val_end - history_window  # 扩展 7 天
        test_end = total_time_steps

        train_data, val_data, test_data = {}, {}, {}
        for k, v in data.items():
            if k in self.time_axis_map:
                time_dim = self.time_axis_map[k]
                # 训练集切片
                train_slice = [slice(None) if i != time_dim else slice(0, mode_len['train']) for i in range(v.ndim)]
                # 验证集切片
                val_slice = [slice(None) if i != time_dim else slice(val_start, val_end) for i in range(v.ndim)]
                # 测试集切片
                test_slice = [slice(None) if i != time_dim else slice(test_start, test_end) for i in range(v.ndim)]

                train_data[k] = v[tuple(train_slice)]
                val_data[k] = v[tuple(val_slice)]
                test_data[k] = v[tuple(test_slice)]
            else:
                # 处理非常规数据类型（如图中的'mode_meta'）
                if isinstance(v, torch.Tensor):
                    # 无时间维度的张量直接广播
                    train_data[k] = val_data[k] = test_data[k] = v.clone()
                else:
                    # 列表/标量等保持原样
                    train_data[k] = val_data[k] = test_data[k] = v

        return train_data, val_data, test_data

    def create_sliding_windows(self, data, history_window, pred_window, slide_step, mode):
        """
        生成滑动窗口数据。
        """
        nodes, time_steps, features = data['input_feat'].shape
        input_data = torch.from_numpy(data['input_feat']).to(dtype=torch.float32)
        edge_feat = data['adj_matrix'].to(dtype=torch.float32).squeeze(-1)

        # 用映射取列（避免写死 2/3）
        idx_S = self.feat_idx.get("S", 2)
        idx_new = self.feat_idx.get("newI", -1)

        S = input_data[:, :, idx_S].unsqueeze(-1)  # [N,T,1]
        newI = input_data[:, :, idx_new].unsqueeze(-1)  # [N,T,1]
        SIR = torch.cat((S, newI), dim=-1)  # [N,T,2]

        x_batch, new_I, Sn, adj_matrix, last_SIR, y_newI = [], [], [], [], [], []

        for t in range(0, time_steps - history_window - pred_window + 1, slide_step):
            x_batch.append(input_data[:, t:t + history_window, :])  # 过去7天
            adj_matrix.append(edge_feat[t:t + history_window, :, :])
            last_SIR.append(SIR[:, t + history_window, :])

            new_I.append(input_data[:, t + history_window:t + history_window + pred_window, 3])
            Sn.append(input_data[:, t + history_window:t + history_window + pred_window, 2])
            y_newI.append(input_data[:, t + history_window:t + history_window + pred_window, [-1]])  # 未来7天
        print(f"实际生成样本数: {len(x_batch)}")

        to_tensor = lambda x: torch.stack(x) if x else torch.empty(0)
        return (
            to_tensor(x_batch),
            to_tensor(new_I),
            to_tensor(Sn),
            to_tensor(adj_matrix),
            to_tensor(last_SIR),
            to_tensor(y_newI)
        )

        # --------------------- 主入口 ---------------------
    def get_data_loader(self, data: dict, params: dict):
        """
        返回：
        - 'train'/'validate'/'test' 的 DataLoader（与原来一致）
        - 'scalers': input_feat 标准化器
        - 'max_od': 邻接归一化因子
        - 'baseline_views': 供基线直接使用的“原始尺度”视图
        - 'feat_idx': 实际使用的特征映射
        - （可选）'population': 若 data 中包含 Population，就带出来
        """
        H, K, step = params['history_window'], params['pred_window'], params['slide_step']

        # ---- temporal split (val/test start H days earlier) ----
        train_data, validate_data, test_data = self.split_data(data, H)

        # ---- 深拷贝原始数据（未归一化）----
        original_train = copy.deepcopy(train_data)
        original_validate = copy.deepcopy(validate_data)
        original_test = copy.deepcopy(test_data)

        # ---- 标准化动态特征 ----
        dynamic_features = ['input_feat', 'seri_feat', 'SIR_data', 'edge_stack_feat']
        scalers = {}

        for feat in dynamic_features:
            if feat not in train_data:  # 某些键可能不存在
                continue
            is_tensor = isinstance(train_data[feat], torch.Tensor)
            original_shape = train_data[feat].shape
            time_dim = self.time_axis_map[feat]

            data_np = train_data[feat].numpy() if is_tensor else train_data[feat].copy()
            reshaped = np.moveaxis(data_np, time_dim, 1)  # -> (samples, time, features)
            n_samples, n_timesteps, n_features = reshaped.shape
            reshaped_2d = reshaped.reshape(-1, n_features)

            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(reshaped_2d)

            scalers[feat] = {
                "scaler": scaler,
                "mean": scaler.mean_,
                "std": scaler.scale_,
                "is_tensor": is_tensor,
                "original_shape": original_shape,
                "time_dim": time_dim
            }

            scaled_train = scaled_train.reshape(n_samples, n_timesteps, n_features)
            scaled_data = np.moveaxis(scaled_train, 1, time_dim)
            train_data[feat] = torch.from_numpy(scaled_data) if is_tensor else scaled_data

        # ---- 用相同 scaler 变换 val/test ----
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

                scaled = scaler.transform(reshaped.reshape(-1, num_features)).reshape(n_samples, n_timesteps,
                                                                                      num_features)
                scaled = np.moveaxis(scaled, 1, time_dim)
                dataset[feat] = torch.from_numpy(scaled) if is_tensor else scaled

        # ---- 生成滑窗（原始 & 归一化）----
        orig_train_feats = self.create_sliding_windows(original_train, H, K, params['slide_step'], mode='train')
        orig_val_feats = self.create_sliding_windows(original_validate, H, K, params['slide_step'], mode='validate')
        orig_test_feats = self.create_sliding_windows(original_test, H, K, params['slide_step'], mode='test')

        norm_train_feats = self.create_sliding_windows(train_data, H, K, params['slide_step'], mode='train')
        norm_val_feats = self.create_sliding_windows(validate_data, H, K, params['slide_step'], mode='validate')
        norm_test_feats = self.create_sliding_windows(test_data, H, K, params['slide_step'], mode='test')

        # ---- adjacency scaling (divided by the training-set maximum) ----
        max_od = norm_train_feats[3][..., 0].max()
        train_adj = norm_train_feats[3] / max_od
        val_adj = norm_val_feats[3] / max_od
        test_adj = norm_test_feats[3] / max_od

        # ---- 组装用于原先 GNN 训练的数据字典（与原返回一致）----
        feat_dicts = {
            'train': {
                'x_node': norm_train_feats[0].float().to(params['device']),
                'new_I': orig_train_feats[1].float().to(params['device']),  # 用原始尺度
                'Sn': orig_train_feats[2].float().to(params['device']),  # 用原始尺度
                'adj_matrix': train_adj.unsqueeze(-1).float().to(params['device']),
                'last_SIR': norm_train_feats[4].float().to(params['device']),
            },
            'validate': {
                'x_node': norm_val_feats[0].float().to(params['device']),
                'new_I': orig_val_feats[1].float().to(params['device']),
                'Sn': orig_val_feats[2].float().to(params['device']),
                'adj_matrix': val_adj.unsqueeze(-1).float().to(params['device']),
                'last_SIR': norm_val_feats[4].float().to(params['device']),
            },
            'test': {
                'x_node': norm_test_feats[0].float().to(params['device']),
                'new_I': orig_test_feats[1].float().to(params['device']),
                'Sn': orig_test_feats[2].float().to(params['device']),
                'adj_matrix': test_adj.unsqueeze(-1).float().to(params['device']),
                'last_SIR': norm_test_feats[4].float().to(params['device']),
            }
        }

        train_dataset = ODDataset(inputs=feat_dicts['train'],
                                  output=norm_train_feats[-1].to(params['device']), mode='train')
        validate_dataset = ODDataset(inputs=feat_dicts['validate'],
                                     output=norm_val_feats[-1].to(params['device']), mode='validate')
        test_dataset = ODDataset(inputs=feat_dicts['test'],
                                 output=norm_test_feats[-1].to(params['device']), mode='test')

        # ----------------- baseline 友好视图（原始尺度）-----------------
        def build_baseline_view(orig_feats, norm_feats, adj_hist_norm):
            """
            返回:
              hist_incidence:   [W,N,H] 历史新增（原始尺度）
              future_incidence: [W,N,K] 未来新增（原始尺度）
              S_hist:           [W,N,H] 历史易感（原始尺度）
              S0_t0:            [W,N]   窗口起点 S(t0)
              S0_last:          [W,N]   历史窗最后一天 S(t0+H-1)
              adj_hist:         [W,H,N,N] 邻接（归一化）
            """
            x_node_raw = orig_feats[0]  # [W,N,H,F] 原始
            idx_S = self.feat_idx.get("S", 2)
            idx_new = self.feat_idx.get("newI", -1)

            hist_incidence = x_node_raw[..., idx_new]  # [W,N,H]
            S_hist = x_node_raw[..., idx_S]  # [W,N,H]
            S0_t0 = S_hist[..., 0]  # [W,N]
            S0_last = S_hist[..., -1]  # [W,N]
            future_incidence = orig_feats[1]  # [W,N,K]（create_sliding_windows里已取原始）
            adj_hist = adj_hist_norm  # [W,H,N,N]
            return {
                "hist_incidence": hist_incidence,
                "future_incidence": future_incidence,
                "S_hist": S_hist,
                "S0_t0": S0_t0,
                "S0_last": S0_last,
                "adj_hist": adj_hist,
            }

        baseline_views = {
            "train": build_baseline_view(orig_train_feats, norm_train_feats, train_adj),
            "validate": build_baseline_view(orig_val_feats, norm_val_feats, val_adj),
            "test": build_baseline_view(orig_test_feats, norm_test_feats, test_adj),
        }

        data_loader = {
            'scalers': scalers.get('input_feat', None),
            'max_od': max_od,
            'baseline_views': baseline_views,  # ← SIR / Meta-SIR / AR 直接用它，完全不需要 DataLoader
            'feat_idx': self.feat_idx,
        }

        data_loader['baseline_views'] = baseline_views
        data_loader['feat_idx'] = self.feat_idx

        if 'static_feat' in data:
            # 带出人口，供 SIR/Meta-SIR 基线使用
            data_loader['population'] = data['static_feat'][:, [0]]  # shape [N,1] 或 [N]

        # ---- 可选地创建 Pytorch DataLoader（只在训练神经网络时用）----
        make_torch_loaders = params.get('make_torch_loaders', True)
        if make_torch_loaders:
            bs_train = params.get('gnn_batch_size', params.get('batch_size', 64))
            bs_eval = params.get('eval_batch_size', bs_train)

            data_loader.update({
                'train': DataLoader(train_dataset, batch_size=bs_train, shuffle=params.get('shuffle_train', True),
                                    drop_last=params.get('drop_last_train', False)),
                'validate': DataLoader(validate_dataset, batch_size=bs_eval, shuffle=False, drop_last=False),
                'test': DataLoader(test_dataset, batch_size=bs_eval, shuffle=False, drop_last=False),
            })
        else:
            data_loader.update({'train': None, 'validate': None, 'test': None})

        # === 1) 计算各 split 在“整条时间轴”上的边界 + 每个窗口的绝对起点 ===
        T_total = data['input_feat'].shape[self.time_axis_map['input_feat']]
        H = params['history_window']
        K = params['pred_window']
        step = params['slide_step']

        # 复用 split_data 的同样逻辑
        mode_len = self.split2len(T_total)
        train_end = mode_len['train']
        val_start = train_end - H
        val_end = mode_len['train'] + mode_len['validate']
        test_start = val_end - H
        test_end = T_total

        bounds = {
            "train": (0, train_end),
            "validate": (val_start, val_end),
            "test": (test_start, test_end)
        }

        def _win_starts(seg):
            a, b = seg
            # 与 create_sliding_windows 的范围保持一致：0..(time_steps - H - K)
            return list(range(a, b - H - K + 1, step))

        data_loader['win_starts'] = {name: _win_starts(seg) for name, seg in bounds.items()}

        # === 2) 提供“全序列”的新增与易感（原始尺度） ===
        arr = data['input_feat'].cpu().numpy() if isinstance(data['input_feat'], torch.Tensor) else data['input_feat']
        idx_new, idx_S = 3, 2  # S=2 and newI=3 in the current feature layout
        data_loader['series'] = {
            "newI": arr[:, :, idx_new],  # [N, T]
            "S": arr[:, :, idx_S],  # [N, T]（可选）
        }

        return data_loader
