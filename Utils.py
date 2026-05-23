import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def rolling_average_3d(data, window):
    # data 的形状应该是 (time_steps, num_nodes, 1)
    time_steps, num_nodes, _ = data.shape
    # 初始化结果数组，为每个节点的时间序列数据应用滑动窗口平滑
    result = np.full((time_steps, num_nodes, 1), np.nan)

    for i in range(num_nodes):
        # 对每个节点的时间序列数据进行操作
        node_data = data[:, i, 0]  # 提取第 i 个节点的时间序列数据
        # 计算累积和
        cumsum = np.cumsum(np.insert(node_data, 0, 0))
        # 计算滚动平均值
        rolling_avg = (cumsum[window:] - cumsum[:-window]) / float(window)
        # 将计算结果赋值给结果数组的相应位置
        result[window - 1:, i, 0] = rolling_avg

    return result

class DataInput(object):
    def __init__(self, data_dir: str, data_split_ratio: tuple):
        self.data_dir = data_dir
        self.data_split_ratio = data_split_ratio

    def load_data(self):
        PATH = self.data_dir + '/jp20200401_20210921.npy'
        data = np.load(PATH, allow_pickle='TRUE').item()

        data_od = data['od']
        data_node_cases = data['node'][..., [0]]
        data_node_roll = rolling_average_3d(data_node_cases, 7)[6:, ...]
        data_node_header = data_node_cases[:6, ...]
        data_node_inf = np.concatenate((data_node_header, data_node_roll), axis=0)

        data_node_other = data['node'][...,[1,2,3]]  # (539, 47, 3)
        data_node = np.concatenate((data_node_inf, data_node_other), axis=-1)  # (539, 47, 4)
        data_SIR = data['SIR']  # (539, 47, 3)
        data_y = data['node'][...,[0]]  # (539, 47, 1) I_new
        commute = np.load(self.data_dir + '/commute_jp.npy')  # 加权邻接矩阵

        # return a dict
        dataset = dict()
        dataset['adj_matrix'] = data_od  # (539, 47, 47, 1)
        dataset['input_data'] = data_node  # (539, 47, 4)
        dataset['SIR'] = data_SIR  # (539, 47, 3)
        dataset['y'] = data_y  # (539, 47, 1)
        dataset['commute'] = commute  # (47, 47)
        dataset['Population'] = data_SIR[0, :, [0]].transpose(1, 0)
        return dataset



def safe_transform(y_tensor, scaler):
    original_shape = y_tensor.shape
    y_flat = y_tensor.detach().cpu().numpy().reshape(-1, 1)
    normalized = scaler.transform(y_flat)
    return torch.tensor(normalized.reshape(original_shape),
                      dtype=torch.float32,
                      device=y_tensor.device).clone()

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
    def __init__(self, history_window: int, pred_window: int, data_split_ratio: tuple):
        self.history_window = history_window
        self.pred_window = pred_window
        self.data_split_ratio = data_split_ratio
        self.time_axis_map = {'input_feat': 1,
                              'adj_matrix': 0,
                              'seri_feat': 1,
                              'edge_stack_feat': 1,
                              'SIR_data': 1
                              }

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

        S = input_data[:, :, 2].unsqueeze(-1)
        newI = input_data[:, :, 3].unsqueeze(-1)
        SIR = torch.cat((S, newI), dim=-1)
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


    def get_data_loader(self, data: dict, params: dict):
        # mode_len = self.split2len(data_len=data['input_feat'].shape[1])
        train_data, validate_data, test_data = self.split_data(data, params['history_window'])
        dynamic_features = ['input_feat', 'seri_feat', 'SIR_data', 'edge_stack_feat']

        # 深拷贝原始数据（未归一化）
        original_train = copy.deepcopy(train_data)
        original_validate = copy.deepcopy(validate_data)
        original_test = copy.deepcopy(test_data)

        scalers = {}

        for feat in dynamic_features:
            if feat not in train_data:
                continue

            is_tensor = isinstance(train_data[feat], torch.Tensor)
            original_shape = train_data[feat].shape
            time_dim = self.time_axis_map[feat]  # 从类属性获取时间轴位置

            # 重组数据为（样本*时间步，特征）,将时间维度移到最后以便标准化
            data_np = train_data[feat].numpy() if is_tensor else train_data[feat].copy()
            reshaped_data = np.moveaxis(data_np, time_dim, 1)  # 示例：time_dim=1 -> (samples, time, features)
            n_samples, n_timesteps, n_features = reshaped_data.shape
            reshaped_2d = reshaped_data.reshape(-1, n_features)

            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(reshaped_2d)

            scalers[feat] = {
                "scaler": scaler,
                "mean": scaler.mean_,  # 存储所有特征的均值数组
                "std": scaler.scale_,  # 存储所有特征的标准差数组
                "is_tensor": is_tensor,
                "original_shape": original_shape,
                "time_dim": time_dim
            }

            # 恢复原始形状
            scaled_train = scaled_train.reshape(n_samples, n_timesteps, n_features)
            scaled_data = np.moveaxis(scaled_train, 1, time_dim)
            train_data[feat] = torch.from_numpy(scaled_data) if is_tensor else scaled_data


        # 对验证/测试数据进行转换
        for dataset in [validate_data, test_data]:
            for feat in dynamic_features:
                if feat not in dataset:
                    continue

                # 获取预存的标准化参数
                scaler_info = scalers[feat]
                scaler = scaler_info["scaler"]
                is_tensor = scaler_info["is_tensor"]
                original_shape = scaler_info["original_shape"]
                time_dim = scaler_info["time_dim"]

                # 保持数据格式一致性
                data_np = dataset[feat].numpy() if is_tensor else dataset[feat].copy()
                reshaped_data = np.moveaxis(data_np, time_dim, 1)
                n_samples, n_timesteps, num_features = reshaped_data.shape

                # 执行标准化转换
                reshaped_2d = reshaped_data.reshape(-1, num_features)
                scaled_data = scaler.transform(reshaped_2d).reshape(n_samples, n_timesteps, num_features)

                # 恢复原始维度顺序
                scaled_data = np.moveaxis(scaled_data, 1, time_dim)
                dataset[feat] = torch.from_numpy(scaled_data) if is_tensor else scaled_data

        original_train_feats = self.create_sliding_windows(original_train, params['history_window'],
                                                           params['pred_window'], params['slide_step'], mode='train')
        original_validate_feats = self.create_sliding_windows(original_validate, params['history_window'],
                                                              params['pred_window'], params['slide_step'], mode='validate')
        original_test_feats = self.create_sliding_windows(original_test, params['history_window'],
                                                          params['pred_window'], params['slide_step'], mode='test')

        norm_train_feats = self.create_sliding_windows(train_data, params['history_window'],
                                                 params['pred_window'], params['slide_step'], mode='train')
        norm_validate_feats = self.create_sliding_windows(validate_data, params['history_window'],
                                                    params['pred_window'], params['slide_step'], mode='validate')
        norm_test_feats = self.create_sliding_windows(test_data, params['history_window'],
                                                params['pred_window'], params['slide_step'], mode='test')

        # MinMax_SCALER
        max_od = norm_train_feats[3][..., 0].max()
        train_adj = norm_train_feats[3] / max_od
        validate_adj = norm_validate_feats[3] / max_od  # 验证集应用相同缩放
        test_adj = norm_test_feats[3] / max_od


        feat_dicts = {
            'train': {
                'x_node': norm_train_feats[0].float().to(params['device']),
                'new_I': original_train_feats[1].float().to(params['device']),
                'Sn': original_train_feats[2].float().to(params['device']),
                'adj_matrix': train_adj.unsqueeze(-1).float().to(params['device']),
                'last_SIR': norm_train_feats[4].float().to(params['device']),
            },
            'validate': {
                'x_node': norm_validate_feats[0].float().to(params['device']),
                'new_I': original_validate_feats[1].float().to(params['device']),
                'Sn': original_validate_feats[2].float().to(params['device']),
                'adj_matrix': validate_adj.unsqueeze(-1).float().to(params['device']),
                'last_SIR': norm_validate_feats[4].float().to(params['device']),
            },
            'test': {
                'x_node': norm_test_feats[0].float().to(params['device']),
                'new_I': original_test_feats[1].float().to(params['device']),
                'Sn': original_test_feats[2].float().to(params['device']),
                'adj_matrix': test_adj.unsqueeze(-1).float().to(params['device']),
                'last_SIR': norm_test_feats[4].float().to(params['device']),
            }
        }

        # 构建数据集（无需传递mode_len）
        train_dataset = ODDataset(inputs=feat_dicts['train'],
                                  output=norm_train_feats[-1].to(params['device']), mode='train')
        validate_dataset = ODDataset(inputs=feat_dicts['validate'],
                                     output=norm_validate_feats[-1].to(params['device']), mode='validate')
        test_dataset = ODDataset(inputs=feat_dicts['test'],
                                 output=norm_test_feats[-1].to(params['device']), mode='test')

        # 创建DataLoader
        data_loader = {
            'train': DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=False),
            'validate': DataLoader(validate_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=False),
            'test': DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=False),
            'scalers': scalers['input_feat'],
            'max_od': max_od
        }
        return data_loader
