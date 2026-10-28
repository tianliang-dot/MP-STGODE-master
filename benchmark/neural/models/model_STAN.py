# models/model_STAN_pure.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PureGATLayer(nn.Module):
    """纯PyTorch实现的图注意力层"""

    def __init__(self, in_features, out_features, num_heads=2, dropout=0.2):
        super(PureGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        # 线性变换
        self.linear = nn.Linear(in_features, out_features * num_heads, bias=False)

        # 注意力机制参数
        self.attn_src = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x, adj_matrix=None):
        """
        Args:
            x: [batch_size, num_nodes, in_features]
            adj_matrix: [batch_size, num_nodes, num_nodes] 邻接矩阵，可选
        Returns:
            [batch_size, num_nodes, out_features * num_heads]
        """
        batch_size, num_nodes, _ = x.shape

        # 线性变换
        h = self.linear(x)  # [B, N, out_features * num_heads]
        h = h.view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # 计算注意力分数
        # q = h * attn_dst, k = h * attn_src
        q = (h * self.attn_dst.view(1, 1, self.num_heads, self.head_dim)).sum(-1)  # [B, N, H]
        k = (h * self.attn_src.view(1, 1, self.num_heads, self.head_dim)).sum(-1)  # [B, N, H]

        # 计算注意力矩阵
        q_expanded = q.unsqueeze(2)  # [B, N, 1, H]
        k_expanded = k.unsqueeze(1)  # [B, 1, N, H]
        attention_scores = q_expanded + k_expanded  # [B, N, N, H]

        # 应用LeakyReLU
        attention_scores = self.leaky_relu(attention_scores)

        # 如果有邻接矩阵，用它作为注意力偏置
        if adj_matrix is not None:
            adj_bias = torch.log1p(torch.clamp(adj_matrix, min=0)).unsqueeze(-1)  # [B, N, N, 1]
            attention_scores = attention_scores + adj_bias

        # 沿邻居维度softmax
        attention_weights = F.softmax(attention_scores, dim=2)  # [B, N, N, H]
        attention_weights = self.dropout(attention_weights)

        # 消息聚合
        h_expanded = h.unsqueeze(1)  # [B, 1, N, H, D]
        attention_expanded = attention_weights.unsqueeze(-1)  # [B, N, N, H, 1]

        # 加权求和
        h_agg = (attention_expanded * h_expanded).sum(dim=2)  # [B, N, H, D]
        h_agg = h_agg.reshape(batch_size, num_nodes, -1)  # [B, N, H*D]

        return h_agg


class PureSTAN(nn.Module):
    """纯PyTorch实现的STAN模型"""

    def __init__(self, in_dim, hidden_dim, pred_window, num_heads=2, dropout=0.2, device='cuda'):
        super(PureSTAN, self).__init__()
        self.pred_window = pred_window
        self.device = device
        self.hidden_dim = hidden_dim

        # GAT layers
        self.gat1 = PureGATLayer(in_dim, hidden_dim, num_heads, dropout)
        self.gat2 = PureGATLayer(hidden_dim * num_heads, hidden_dim, 1, dropout)

        # GRU for temporal dynamics
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Output layers
        self.pred_fc = nn.Linear(hidden_dim, pred_window)
        self.sir_fc = nn.Linear(hidden_dim, 2)  # alpha, beta

        # Normalization and regularization
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Parameter initialization
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for layer in [self.pred_fc, self.sir_fc]:
            nn.init.xavier_normal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    # 在 PureSTAN 类里新增/替换：
    def _estimate_I0_from_hist(self, newI_hist_raw: torch.Tensor, beta: torch.Tensor):
        """
        newI_hist_raw: [B,H,N] (原始尺度)
        beta:          [B,N]   (恢复率，0~1)
        返回 I0:       [B,N]
        """
        B, H, N = newI_hist_raw.shape
        # 让“k=0”为最近一天
        y_rev = torch.flip(newI_hist_raw, dims=[1])  # [B,H,N]
        k = torch.arange(H, device=newI_hist_raw.device).view(1, H, 1)
        surv = torch.clamp(1.0 - beta, 0.0, 1.0).unsqueeze(1) ** k  # [B,H,N]
        I0 = torch.sum(y_rev * surv, dim=1)  # [B,N]
        return I0

    def compute_physics_prediction(self,
                                   S0_raw: torch.Tensor,  # [B,N] 原始尺度
                                   newI_hist_raw: torch.Tensor,  # [B,H,N] 原始尺度
                                   population: torch.Tensor,  # [N] 或 [1,N] 原始尺度
                                   alpha: torch.Tensor,  # [B,N]
                                   beta: torch.Tensor,  # [B,N]
                                   timesteps: int) -> torch.Tensor:
        """
        返回 (新增预测，原始尺度): [B,N,K]
        """
        # --- population 形状统一为 [1,N] ---
        if population.dim() == 2 and population.size(0) == 1:
            pop = population.squeeze(0)  # [N]
        else:
            pop = population
        pop = pop.view(1, -1)  # [1,N]

        B, N = S0_raw.shape
        # 估 I0，并做物理裁剪
        I0 = self._estimate_I0_from_hist(newI_hist_raw, beta)  # [B,N]
        I0 = I0.clamp_min(0.0)
        I0 = torch.minimum(I0, (pop - S0_raw).clamp_min(0.0))

        S = S0_raw.clone()  # [B,N]
        I = I0.clone()  # [B,N]

        alpha_c = torch.clamp(alpha, 0.0, 1.0)
        beta_c = torch.clamp(beta, 0.0, 1.0)

        preds = []  # 装 [B,N]，累计成 [B,N,K]
        for _ in range(timesteps):
            force = alpha_c * S * (I / (pop + 1e-12))  # [B,N] 新增
            recov = beta_c * I  # [B,N]

            # 状态推进（非负约束）
            newI = torch.clamp(force, min=0.0)
            S = torch.clamp(S - newI, min=0.0)
            I = torch.clamp(I + newI - recov, min=0.0)

            preds.append(newI)
        # -> [K,B,N] -> [B,N,K]
        return torch.stack(preds, dim=0).permute(1, 2, 0)

    def forward(self, xb_btnc, Ah_btnnn, nI_raw, last_S_raw, population=None, y_norm_stats=None):
        """
        xb_btnc:  [B,H,N,C]   （标准化后）
        Ah_btnnn: [B,H,N,N]
        nI_raw:   [B,H,N]     （原始尺度：历史新增）
        last_S_raw:[B,N]      （原始尺度：历史窗最后一天 S）
        population:[N] or [1,N] （原始尺度）
        """
        B, H, N, C = xb_btnc.shape
        h = torch.zeros(B, N, self.hidden_dim, device=self.device)

        all_predictions = []
        all_direct_preds = []
        all_physics_preds = []
        alpha_list, beta_list = [], []

        for t in range(H):
            x_t = xb_btnc[:, t, :, :]  # 注意：仅用于图特征，不给 physics 用
            A_t = Ah_btnnn[:, t, :, :]

            # GAT + GRU
            gat1_out = self.gat1(x_t, A_t)
            gat1_out_bn = self.bn1(gat1_out.permute(0, 2, 1)).permute(0, 2, 1)
            gat1_out = self.dropout(F.elu(gat1_out_bn))

            gat2_out = self.gat2(gat1_out, A_t)
            gat2_out_bn = self.bn2(gat2_out.permute(0, 2, 1)).permute(0, 2, 1)
            spatial_features = self.dropout(F.elu(gat2_out_bn))

            h = self.gru(spatial_features.reshape(B * N, -1), h.reshape(B * N, -1)).reshape(B, N, -1)

            # 直连分支（标准化尺度）
            direct_pred = self.pred_fc(h)  # [B,N,K]
            all_direct_preds.append(direct_pred)

            # 动力学参数
            sir_params = self.sir_fc(h)  # [B,N,2]
            alpha = torch.sigmoid(sir_params[:, :, 0]) * 0.3 + 0.01
            beta = torch.sigmoid(sir_params[:, :, 1]) * 0.2 + 0.01
            alpha_list.append(alpha);
            beta_list.append(beta)

            # 物理分支：原始尺度 → 标准化尺度
            if population is not None:
                phys_raw = self.compute_physics_prediction(
                    S0_raw=last_S_raw,  # [B,N]
                    newI_hist_raw=nI_raw,  # [B,H,N]
                    population=population,  # [N] or [1,N]
                    alpha=alpha, beta=beta,
                    timesteps=self.pred_window  # K
                )  # [B,N,K] 原始尺度

                if y_norm_stats is not None and y_norm_stats.get('type', 'zscore') == 'zscore':
                    mu = float(y_norm_stats['mean'])
                    std = float(max(y_norm_stats['std'], 1e-6))
                    physics_pred = (phys_raw - mu) / std  # [B,N,K] 标准化
                else:
                    physics_pred = phys_raw
            else:
                physics_pred = torch.zeros_like(direct_pred)

            all_physics_preds.append(physics_pred)

            # 融合（均在标准化尺度）
            physics_weight, direct_weight = 0.3, 0.7
            combined_pred = direct_weight * direct_pred + physics_weight * physics_pred  # [B,N,K]
            all_predictions.append(combined_pred)

        final_predictions = all_predictions[-1]  # [B,N,K]
        return final_predictions.permute(0, 2, 1), {  # -> [B,K,N]
            'direct_preds': torch.stack(all_direct_preds, dim=1),
            'physics_preds': torch.stack(all_physics_preds, dim=1),
            'alpha': torch.stack(alpha_list, dim=1),
            'beta': torch.stack(beta_list, dim=1)
        }
