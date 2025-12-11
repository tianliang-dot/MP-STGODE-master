import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict
from models.odegcn_meta import ODEG


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        if len(A.shape) == 2:
            x = torch.einsum('vw, ncwl->ncvl', A, x)
        else:
            x = torch.einsum('nvw, ncwl->ncvl', A, x)
        return x.contiguous()


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it  额外的维度将通过填充添加，删除它
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        """
        Args:
            input_dim: STGCNBlock 输出的最后一个通道数，例如 out_channels[-1]
            num_blocks: 堆叠了几个 block
        """
        super(AttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)

    def forward(self, block_outputs):
        """
        Args:
            block_outputs: List of Tensors，每个形状 [B, N, T, C]
        Return:
            pooled_output: [B, N, T, C]
        """
        x = torch.stack(block_outputs, dim=1)  # [B, Bk, N, T, C]
        B, Bk, N, T, C = x.shape

        # Flatten B, N, T for attention计算
        x_ = x.permute(0, 2, 3, 1, 4).reshape(B * N * T, Bk, C)
        query = self.query.expand(B * N * T, 1, C)  # [BNT, 1, C]

        key = self.key_proj(x_)
        value = self.value_proj(x_)
        attn = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / (C ** 0.5), dim=-1)  # [BNT, 1, Bk]

        out = torch.matmul(attn, value).squeeze(1)  # [BNT, C]
        return out.view(B, N, T, C)


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        # self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        return self.mlp(x)

def process_dynamic_adj(raw_adj: torch.Tensor,
                        threshold: float = 0.0,
                        device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    动态邻接矩阵处理函数
    Args:
        raw_adj: 原始邻接矩阵 (batch, time, nodes, nodes, 1)
        threshold: 二值化阈值，默认>0即视为连接
        device: 计算设备
    Returns:
        binary_adj: 二值化邻接矩阵 (batch, time, nodes, nodes)
        normalized_adj: 行归一化邻接矩阵 (batch, time, nodes, nodes)
    """
    # 维度压缩
    adj = raw_adj.squeeze(-1).to(device)  # (B,T,N,N)
    batch_size, time_steps, num_nodes, _ = adj.shape

    # 初始化输出
    binary_adj = torch.zeros_like(adj)
    normalized_adj = torch.zeros_like(adj)

    for b in range(batch_size):
        for t in range(time_steps):
            # 当前时间步邻接矩阵
            current_adj = adj[b, t]

            # 二值化处理
            bin_adj = (current_adj > threshold).float()
            bin_adj = bin_adj * (1 - torch.eye(num_nodes, device=device))  # 排除自连接
            binary_adj[b, t] = bin_adj

            # 行归一化
            row_sum = bin_adj.sum(dim=-1, keepdim=True) + 1e-7
            norm_adj = bin_adj / row_sum
            normalized_adj[b, t] = norm_adj

    return binary_adj, normalized_adj


# 字典版本（可选）
def process_dynamic_adj_dict(raw_adj: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
    bin_adj, norm_adj = process_dynamic_adj(raw_adj, **kwargs)
    return {
        "binary_adj": bin_adj,
        "normalized_adj": norm_adj,
        "mask": (bin_adj > 0).float()  # 附加掩膜矩阵
    }


class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.3):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, num_nodes, length)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)  # (batch_size, num_nodes, length, feature_dim)
        return y



class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, history_window, num_nodes, int_init):
        """
        Args:
            in_channels: Number of input features at each node in each time step.
            out_channels: a list of feature channels in timeblock, the last is output feature channel
            num_nodes: Number of nodes in the graph
            A_hat: the normalized adjacency matrix
        """
        super(STGCNBlock, self).__init__()
        self.int_init = int_init
        self.temporal1 = TemporalConvNet(num_inputs=in_channels, num_channels=out_channels)

        time_length = float(history_window - 1)
        time_length = torch.linspace(0, history_window - 1, history_window, device=self.int_init.device)
        self.odeg = ODEG(out_channels[-1], history_window, num_nodes, self.int_init, time=time_length)
        # time_length = torch.linspace(0, history_window-1, 70)  # You can change the number of time steps
        # self.odeg = ODEG(out_channels[-1], history_window, num_nodes, self.int_init, time=time_length)

        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1], num_channels=out_channels)
        # 融合层
        # self.fc = nn.Linear(out_channels[-1] * 2, out_channels[-1])  # 拼接STGODE和LSTM输出
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X, adj):
        output_t1 = self.temporal1(X)  # (16, 51, 7, 32)  (batch_size, num_nodes, length, feature_dim)
        output_ode = self.odeg(output_t1, adj)
        output_t2 = self.temporal2(F.relu(output_ode))  # (16, 51, 7, 32)
        # combined = torch.cat([output_ode, output_t2], dim=-1)  # [B, T, N, 2*hidden_dim]
        # output = self.fc(combined)  # [B, T, N, C]

        return self.batch_norm(output_t2)


class SIR_cell(nn.Module):
    def __init__(self):
        super(SIR_cell, self).__init__()


    def forward(self, Sn_t: torch.Tensor, new_I: torch.Tensor, static_feats: torch.Tensor, param_b: torch.Tensor,
                param_g: torch.Tensor, adj_dict:torch.Tensor, mob: torch.Tensor, device):

        num_node = static_feats.shape[0]
        length = param_b.shape[2]
        New_I_original = new_I.unsqueeze(-1)  # (16,51,7,1)
        N = static_feats[:, 0].to(device)  # (1,51)
        pop_m = N.unsqueeze(0).unsqueeze(1)  # (1,1,51)
        pop_n = N.unsqueeze(0).unsqueeze(-1)  # (1,51,1)

        In_t = torch.zeros(Sn_t.shape[0], num_node, length, 1).to(device)
        for T in range(1, length + 1):
            v_T = torch.zeros(Sn_t.shape[0], num_node, 1).to(device)  # 存储当前T的累加结果
            for t in range(T):
                weight = torch.pow((1 - param_g[:, :, t, :]), T - t)  # 计算权重
                v_T += weight * New_I_original[:, :, t, :]  # 累加结果
            In_t[:, :, T-1, :] = v_T  # 将当前 T 的结果记录到 v 中


        beta_n = param_b.permute(0, 2, 1, 3)  # (16,3/7/14,51,1)
        beta_m = torch.einsum('btji,btj->bti',
                              adj_dict['normalized_adj'],  # (16,3/7,51,51)
                              beta_n.squeeze(-1))  # (16,51,3/7,1) → (16,51,3/7,1)
        # 恢复原维度顺序
        beta_m = beta_m.unsqueeze(-1)
        beta_n = beta_n.permute(0, 2, 1, 3)  # (16,51,3/7,1)
        beta_m = beta_m.permute(0, 2, 1, 3)  # (16,51,3/7,1)

        d_St, d_It, d_Rt = [], [], []
        for i in range(length):
            bn_t = beta_n[:, :, i, :]  # (16,51,1)
            bm_t = beta_m[:, :, i, :]  # (16,51,1)
            h_mn_t = mob[:, i, :, :]  # (16,51,51)
            h_nm_t = mob[:, i, :, :].transpose(1, 2)
            h_nn_t = mob[:, i, :, :] * torch.eye(mob.shape[-1], device=device)
            h_nn_t = h_nn_t * 0.5  # 对角线元素减半（即 Pn / 2）

            # 应用邻接矩阵过滤非邻居
            masked_hmn = h_mn_t * adj_dict['binary_adj'][:, i, ...]  # (16,51,51) 过滤非邻居
            # Term1: 计算 h_mn(t)/P_m * I_m(t)
            mobility_term1 = masked_hmn / (pop_m + 1e-7) # (16,51,51)
            sum_Im = torch.einsum('bnm,bm->bn',mobility_term1, In_t[..., i, :].squeeze(-1))  # (16,51)
            # 乘以 β_n(t) * S_n(t)/P_n
            term1 = bn_t.squeeze(-1) * (sum_Im * Sn_t[..., i, :].squeeze(-1) / (pop_n.squeeze(-1) + 1e-7))

            # Term2: 计算 h_nm(t)/P_n * S_n(t)
            mobility_term2 = h_nm_t / (pop_n + 1e-7)  # h_nm维度 (16,51,51)
            Sn = Sn_t[..., i, :].squeeze(-1)  # S_n(t)维度 (16,51)
            sum_Sn = torch.einsum('bnm,bn->bm', mobility_term2, Sn)
            # 乘以 β_m(t) * I_m(t)/P_m, β_m(t)维度 (16,51)
            term2 = bm_t.squeeze(-1) * (sum_Sn * In_t[..., i, :].squeeze(-1) / (pop_m.squeeze(1) + 1e-7))

            # Term3: 计算内部传播
            mobility_term3 = h_nn_t / (pop_n + 1e-7)
            Sn = Sn_t[..., i, :].squeeze(-1)  # (batch, node)
            sum_Sn = torch.einsum('bnm,bn->bm', mobility_term3, Sn)
            internal_term = 2 * bn_t.squeeze(-1) * (sum_Sn * In_t[..., i, :].squeeze(-1)) / (pop_n.squeeze(-1) + 1e-7)

            dS = -(term1 + term2 + internal_term).unsqueeze(-1)
            dI = (term1 + term2 + internal_term).unsqueeze(-1) - param_g[:, :, i, :] * In_t[:, :, i, :]
            dR = param_g[:, :, i, :] * In_t[:, :, i, :]

            d_St.append(dS)
            d_It.append(dI)
            d_Rt.append(dR)

        d_St = torch.stack(d_St, dim=2)  # (t+history_window : t+history_window+pred_window)  (16, 51, 7, 1)
        d_It = torch.stack(d_It, dim=2)  # (t+history_window : t+history_window+pred_window)
        d_Rt = torch.stack(d_Rt, dim=2)  # (t+history_window : t+history_window+pred_window)

        d_SIR = torch.cat((In_t, -d_St, d_It, d_Rt), dim=-1)
        return d_SIR


class STGODE_SIR(nn.Module):
    """ the overall network framework """
    def __init__(self, num_nodes, num_features, timesteps_input, timesteps_output):
        super(STGODE_SIR, self).__init__()

        self.int_init = nn.Parameter(torch.empty(timesteps_output, timesteps_input), requires_grad=True)
        nn.init.normal_(self.int_init, 1, 0.01)
        self.od_scale_factor = 3
        # self.num_ctrl_pts = num_ctrl_pts

        # self.input_proj = nn.Linear(num_features, hidden_dim)
        self.ST_blocks = nn.ModuleList(
            [STGCNBlock(in_channels=num_features, out_channels=[32, 16, 32], history_window=timesteps_input,
                        num_nodes=num_nodes, int_init=self.int_init) for _ in range(3)
             ])

        # self.output_decoder = nn.Sequential(
        #     nn.Linear(timesteps_input * 32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, timesteps_output * 3),
        # )

        self.output_decoder = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, timesteps_output * 2),
        )

        self.output_newI = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, timesteps_output * 1),
        )

        self.SIR_cell = SIR_cell()

        self.AttentionPooling = AttentionPooling(input_dim=32)

        # self.beta_spline = BSplineBasis(num_ctrl_pts=num_ctrl_pts, degree=3, pred_len=timesteps_output)
        # self.gamma_spline = BSplineBasis(num_ctrl_pts=num_ctrl_pts, degree=3, pred_len=timesteps_output)
        # self.newI_spline = BSplineBasis(num_ctrl_pts=num_ctrl_pts, degree=3, pred_len=timesteps_output)

    def forward(self, x_input, newI_original, Sn, static_feats, adj, max_od, device):
        """
        Args:
            x : input data of shape (batch_size, num_nodes, num_timesteps, num_features) == (B, N, T, F)
        Returns:
            prediction for future of shape (batch_size, num_nodes, num_timesteps_output)
        """

        # x = self.input_proj(x_input)
        B, N, T, C = x_input.shape
        outs = []
        # spatial graph
        for blk in self.ST_blocks:
            outs.append(blk(x_input, adj))

        # outs = torch.stack(outs)
        # x_out = torch.max(outs, dim=0)[0]
        x_out = self.AttentionPooling(outs)
        x_last = x_out[:, :, -1, :]

        # x_out = x_out.reshape((B, N, -1))  # (32,51,224)
        output = self.output_decoder(x_last).reshape(B, N, -1, 2)
        param_b = torch.sigmoid(output[...,[0]])
        param_g = torch.sigmoid(output[...,[1]])
        # output_newI = output[...,[2]]
        output_newI = self.output_newI(x_last).reshape(B, N, -1, 1)

        # x_last = self.output_decoder(x_last)
        # param_b = torch.sigmoid(self.beta_spline(x_last[..., 0])).unsqueeze(-1)  # [B, N, T, 1]
        # param_g = torch.sigmoid(self.gamma_spline(x_last[..., 1])).unsqueeze(-1)
        # output_newI = self.newI_spline(x_last[..., 2]).unsqueeze(-1)

        # x_last = self.output_decoder(x_last).reshape(B, N, -1, 3) # (B, N, T, 1)
        # param_b = torch.sigmoid(x_last[...,[0]])
        # param_g = torch.sigmoid(x_last[...,[1]])
        # output_newI = x_last[..., [2]]

        # SIR cell
        Snt = Sn.unsqueeze(-1)
        incidence = torch.softmax(self.int_init, dim=1)
        mob = torch.einsum('kl,blnmc->bknmc', incidence, adj).squeeze(-1) # (batch_size, length, nodes, nodes)
        adj_dict = process_dynamic_adj_dict(mob, device="cuda:0")
        d_SIR = self.SIR_cell(Snt, newI_original, static_feats, param_b, param_g, adj_dict,
                              mob * max_od * self.od_scale_factor, device)  # (batch_size, nodes, length, feature)

        return param_b, param_g, output_newI, d_SIR




class BSplineBasis(nn.Module):
    def __init__(self, num_ctrl_pts, degree, pred_len):
        super(BSplineBasis, self).__init__()
        self.num_ctrl_pts = num_ctrl_pts
        self.degree = degree
        self.pred_len = pred_len

        # 构造节点向量（uniform B-spline）
        self.register_buffer('knot_vector', self.create_knot_vector())
        self.register_buffer('x_query', torch.linspace(0, 1, pred_len))  # 归一化时间坐标

    def create_knot_vector(self):
        n = self.num_ctrl_pts
        d = self.degree
        total_knots = n + d + 1
        kv = np.concatenate([
            np.zeros(d),
            np.linspace(0, 1, total_knots - 2 * d),
            np.ones(d)
        ])
        return torch.tensor(kv, dtype=torch.float32)

    def basis_function(self, i, d, x, knot):
        if d == 0:
            return ((x >= knot[i]) & (x < knot[i + 1])).float()
        else:
            denom1 = knot[i + d] - knot[i]
            denom2 = knot[i + d + 1] - knot[i + 1]

            term1 = ((x - knot[i]) / denom1 * self.basis_function(i, d - 1, x, knot)) if denom1 > 0 else 0.0
            term2 = ((knot[i + d + 1] - x) / denom2 * self.basis_function(i + 1, d - 1, x, knot)) if denom2 > 0 else 0.0

            return term1 + term2

    def forward(self, ctrl_pts):  # ctrl_pts: (B, N, K)
        B, N, K = ctrl_pts.shape
        D = self.degree
        T = self.pred_len

        # 构造 (K, T) 的基函数矩阵
        basis_all = []
        for i in range(K):
            b_i = self.basis_function(i, D, self.x_query, self.knot_vector)  # [T]
            basis_all.append(b_i)
        basis = torch.stack(basis_all, dim=0).to(ctrl_pts.device)  # (K, T)

        # 插值 (B, N, K) × (K, T) → (B, N, T)
        out = torch.einsum('bnk,kt->bnt', ctrl_pts, basis)
        return out