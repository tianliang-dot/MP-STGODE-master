import os
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import yaml
import copy
import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Tuple, List
from datasets.Utils import DataGenerator
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def build_y_denorm_from_loader(data_loader, target_feat_key='input_feat', target_feat_name='newI', feat_idx=None):
    """
    返回一个可调用对象 y_denorm(arr_np)：
      arr_np: [W, K, N] 的 numpy 数组（标准化后的 y）
      返回 : 反归一化到原始尺度的 [W, K, N]
    data_loader['scalers'] 里应包含针对 target_feat_key 的 StandardScaler；
    target_feat_name 的索引来自 feat_idx（需要外部传入，因为data_loader中可能没有）。
    """
    # 1) 取 scaler
    scaler_info = data_loader.get('scalers', None)
    if scaler_info is None:
        return lambda x: x

    # 如果scaler_info是字典，并且有'mean'和'std'键
    if not isinstance(scaler_info, dict) or 'mean' not in scaler_info or 'std' not in scaler_info:
        return lambda x: x

    # 2) 取目标通道索引（如 newI）
    if feat_idx is None:
        feat_idx = data_loader.get('feat_idx', {})
    if target_feat_name not in feat_idx:
        return lambda x: x
    idx = feat_idx[target_feat_name]

    # 3) 取该通道的 mean / std
    mean_arr = scaler_info['mean']  # 所有特征的均值数组
    std_arr = scaler_info['std']    # 所有特征的标准差数组

    if mean_arr is None or std_arr is None:
        return lambda x: x

    # 确保mean_arr和std_arr是numpy数组，并且长度足够
    if len(mean_arr) <= idx or len(std_arr) <= idx:
        return lambda x: x

    y_mean = float(mean_arr[idx])
    y_std = float(std_arr[idx])

    # 4) 返回反归一化函数
    def _denorm(arr_np: np.ndarray) -> np.ndarray:
        return arr_np * y_std + y_mean

    return _denorm

# ---------- LSTM Model ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, pred_window, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_window = pred_window

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size * pred_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, num_nodes, history_window, input_features)
        batch_size, num_nodes, history_window, input_features = x.shape

        # Reshape: treat each node as separate sequence
        x_reshaped = x.reshape(batch_size * num_nodes, history_window, input_features)

        # LSTM forward
        h0 = torch.zeros(self.num_layers, batch_size * num_nodes, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size * num_nodes, self.hidden_size).to(x.device)

        out, _ = self.lstm(x_reshaped, (h0, c0))

        # Use the last output for prediction
        last_output = out[:, -1, :]
        last_output = self.dropout(last_output)

        # Predict multiple time steps
        predictions = self.fc(last_output)
        predictions = predictions.reshape(batch_size, num_nodes, self.pred_window)

        return predictions


# ---------- GRU Model ----------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, pred_window, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_window = pred_window

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size * pred_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, num_nodes, history_window, input_features)
        batch_size, num_nodes, history_window, input_features = x.shape

        # Reshape: treat each node as separate sequence
        x_reshaped = x.reshape(batch_size * num_nodes, history_window, input_features)

        # GRU forward
        h0 = torch.zeros(self.num_layers, batch_size * num_nodes, self.hidden_size).to(x.device)

        out, _ = self.gru(x_reshaped, h0)

        # Use the last output for prediction
        last_output = out[:, -1, :]
        last_output = self.dropout(last_output)

        # Predict multiple time steps
        predictions = self.fc(last_output)
        predictions = predictions.reshape(batch_size, num_nodes, self.pred_window)

        return predictions


# ---------- Training Functions ----------
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        x_node, new_I, Sn, adj_matrix, last_SIR, y_true = batch

        # Move to device
        x_node = x_node.to(device)
        y_true = y_true.to(device)

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(x_node)

        # Calculate loss
        loss = criterion(y_pred, y_true.squeeze(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x_node, new_I, Sn, adj_matrix, last_SIR, y_true = batch

            # Move to device
            x_node = x_node.to(device)
            y_true = y_true.to(device)

            # Forward pass
            y_pred = model(x_node)

            # Calculate loss
            loss = criterion(y_pred, y_true.squeeze(-1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


# ---------- Evaluation Functions ----------
def evaluate_model(model, data_loader, split, device):
    """Evaluate model on given split and return predictions and true values"""
    model.eval()
    all_predictions = []
    all_targets = []

    # 创建反归一化函数
    y_denorm_fn = build_y_denorm_from_loader(
        data_loader,
        target_feat_key='input_feat',
        target_feat_name='newI',
        feat_idx={'cum': 0, 'death': 1, 'S': 2, 'newI': 3}  # 明确指定特征索引
    )

    with torch.no_grad():
        for batch in data_loader[split]:
            x_node, new_I, Sn, adj_matrix, last_SIR, y_true = batch

            x_node = x_node.to(device)
            y_pred = model(x_node)

            # 使用反归一化函数处理预测值和真实值
            y_pred_denorm = y_denorm_fn(y_pred.cpu().numpy())
            # 注意：y_true也需要反归一化，因为它在数据加载器中也被标准化了
            y_true_denorm = y_denorm_fn(y_true.cpu().numpy())

            all_predictions.append(y_pred_denorm)
            all_targets.append(y_true_denorm)

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets


def calculate_metrics(y_true, y_pred, eps=1e-8):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae  = float(np.mean(np.abs(yt - yp)))
    mask = np.abs(yt) > eps
    mape = float(np.mean(np.abs((yp[mask] - yt[mask]) / (yt[mask] + 1e-12))) * 100.0) if mask.any() else float('nan')
    wmape = float(np.sum(np.abs(yp - yt)) / (np.sum(np.abs(yt)) + eps) * 100.0)
    rae  = float(np.sum(np.abs(yp - yt)) / (np.sum(np.abs(yt - yt.mean())) + 1e-12))

    return {"RMSE": rmse,
            "MAE": mae,
            "MAPE%": mape,
            "wMAPE%": wmape,
            "RAE": rae}



def bootstrap_metrics(y_true, y_pred, n_bootstrap=30):
    """Calculate bootstrap confidence intervals for metrics"""
    if n_bootstrap <= 0:
        point = calculate_metrics(y_true, y_pred)
        return point, {key: float("nan") for key in point}, pd.DataFrame(columns=point.keys())
    n_samples = len(y_true.reshape(-1))
    metrics_list = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_sample = y_true.reshape(-1)[indices]
        y_pred_sample = y_pred.reshape(-1)[indices]

        # Calculate metrics for this sample
        metrics = calculate_metrics(y_true_sample, y_pred_sample)
        metrics_list.append(metrics)

    # Calculate mean and std across bootstrap samples
    metrics_df = pd.DataFrame(metrics_list)
    mean_metrics = metrics_df.mean().to_dict()
    std_metrics = metrics_df.std().to_dict()

    return mean_metrics, std_metrics, metrics_df


# ---------- Main Training Loop ----------
def main():
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='COVID-19')
    parser.add_argument('--data_pkl', type=str, default=str(Path(__file__).resolve().parent.parent.parent / "data" / "Data_meta_stgode_7.pkl"))
    parser.add_argument('--resroot', type=str, default=str(Path(__file__).resolve().parent.parent / "artifacts"))
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).resolve().parent.parent / "outputs" / "neural" / "rnn"))
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', type=str, default='GRU', choices=['LSTM', 'GRU'])
    parser.add_argument('--num_nodes', type=int, default=51)
    parser.add_argument('--input_features', type=int, default=4)
    parser.add_argument('--history_window', type=int, default=7)
    parser.add_argument('--pred_window', type=int, default=3)
    parser.add_argument('--slide_step', type=int, default=1)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_ratio', type=float, nargs='+', default=[8, 1, 1])
    parser.add_argument('--learn_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-7)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss', type=str, choices=['MSE', 'MAE'], default='MSE')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--ckptint', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--specify_ckpt', type=str, default='')
    parser.add_argument('--expdir', type=str, default='')
    parser.add_argument('--extconf', type=str, default='')
    parser.add_argument('--n_bootstrap', type=int, default=30, help='Bootstrap实验次数')
    parser.add_argument(
        '--shuffle_test', type=int, choices=[0, 1], default=1,
        help='Whether to shuffle test forecast windows. Keep 1 for legacy reproduction; use 0 for aligned walk-forward evaluation.'
    )

    # Model specific parameters
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    params = parser.parse_args().__dict__

    # Set random seeds for reproducibility
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    # Create output directory
    os.makedirs(params['output_dir'], exist_ok=True)

    # Save configuration
    with open(os.path.join(params['output_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(params, f)

    # Load data
    Input_Data = load_data_from_pickle(params['data_pkl'])
    loc_list = Input_Data['loc_list']
    Input_Data['input_feat'] = Input_Data['input_feat'][:, :, :4]

    # Dataset preparation
    data_generator = DataGenerator(
        history_window=params['history_window'],
        pred_window=params['pred_window'],
        data_split_ratio=params['split_ratio']
    )
    data_loader = data_generator.get_data_loader(data=Input_Data, params=params)

    # Initialize model
    if params['model'] == 'LSTM':
        model = LSTMModel(
            input_size=params['input_features'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            output_size=1,  # Predicting newI
            pred_window=params['pred_window'],
            dropout=params['dropout']
        )
    else:  # GRU
        model = GRUModel(
            input_size=params['input_features'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            output_size=1,  # Predicting newI
            pred_window=params['pred_window'],
            dropout=params['dropout']
        )

    model = model.to(params['device'])

    # Loss function and optimizer
    if params['loss'] == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(),
                           lr=params['learn_rate'],
                           weight_decay=params['weight_decay'])

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    print(f"Starting training for {params['model']} model...")
    for epoch in range(params['num_epochs']):
        # Train
        train_loss = train_epoch(model, data_loader['train'], optimizer, criterion, params['device'])
        train_losses.append(train_loss)

        # Validate
        val_loss = validate_epoch(model, data_loader['validate'], criterion, params['device'])
        val_losses.append(val_loss)

        # Print progress
        if (epoch + 1) % params['print_every'] == 0:
            print(f'Epoch [{epoch + 1}/{params["num_epochs"]}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

            # Save checkpoint
            if (epoch + 1) % params['ckptint'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(params['output_dir'], f'best_{params["model"]}_model.pth'))

    # Load best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_predictions, test_targets = evaluate_model(
        model, data_loader, 'test', params['device']
    )

    # 打印一些样本值来验证反归一化是否成功
    print("Sample denormalized predictions:", test_predictions[:5])
    print("Sample denormalized targets:", test_targets[:5])

    # Calculate metrics
    test_metrics = calculate_metrics(test_targets, test_predictions)
    print(f"Test Metrics: {test_metrics}")

    # Bootstrap analysis. The walk-forward launcher passes 0 because it applies
    # one shared forecast-origin bootstrap to every benchmark after inference.
    if params['n_bootstrap'] > 0:
        print("Performing bootstrap analysis...")
        bootstrap_mean, bootstrap_std, bootstrap_df = bootstrap_metrics(
            test_targets, test_predictions, params['n_bootstrap']
        )
        print("Bootstrap Results:")
        for metric in ['RMSE', 'MAE', 'MAPE%', 'wMAPE%', 'RAE']:
            print(f"{metric}: {bootstrap_mean[metric]:.6f}±{bootstrap_std[metric]:.6f}")
    else:
        print("Native bootstrap disabled; the walk-forward launcher will apply unified bootstrap.")
        bootstrap_mean, bootstrap_std = {}, {}
        bootstrap_df = pd.DataFrame(columns=['RMSE', 'MAE', 'MAPE%', 'wMAPE%', 'RAE'])

    # Save results
    results = {
        'test_metrics': test_metrics,
        'bootstrap_mean': bootstrap_mean,
        'bootstrap_std': bootstrap_std,
        'bootstrap_details': bootstrap_df.to_dict('list')
    }

    with open(os.path.join(params['output_dir'], f'{params["model"]}_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    np.savez_compressed(
        os.path.join(params['output_dir'], f'{params["model"]}_predictions.npz'),
        pred=np.squeeze(test_predictions),
        true=np.squeeze(test_targets),
    )

    # Save bootstrap details as CSV
    if params['n_bootstrap'] > 0:
        bootstrap_df.to_csv(os.path.join(params['output_dir'], f'{params["model"]}_bootstrap_details.csv'), index=False)

    # Plot training history
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(f'{params["model"]} Training History')
    # plt.legend()
    # plt.savefig(os.path.join(params['output_dir'], f'{params["model"]}_training_history.png'))
    # plt.close()
    #
    # print(f"Training completed. Results saved to {params['output_dir']}")


if __name__ == '__main__':
    main()
