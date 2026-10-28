# model_MepoGNN.py
# MEPOGNN implementation aligned with the benchmark runner in interface, shape, and scale.
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== 核心子模块（保留原思路，统一形状） ======

class NConv(nn.Module):
    def forward(self, x, A):
        """
        x: [B, C, N, L]
        A: [B, N, N] (dynamic) or [N, N] (static)
        out: [B, C, N, L]
        """
        if A.dim() == 2:
            # static
            return torch.einsum('vw, bcwl -> bcvl', A, x).contiguous()
        else:
            # dynamic with batch
            return torch.einsum('bnvw, bcwl -> bcvl', A.unsqueeze(1), x).contiguous()

class Linear1x1(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear1x1, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1,1), padding=0, bias=True)
    def forward(self, x): return self.mlp(x)

class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super(GCN, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear1x1(c_in * (order * support_len), c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, supports: List[torch.Tensor]):
        outs = []
        for A in supports:
            x1 = self.nconv(x, A)        # [B,C,N,L]
            outs.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, A)
                outs.append(x1)
        h = torch.cat(outs, dim=1)       # [B, C*order*len, N, L]
        h = self.mlp(h)
        return F.dropout(h, self.dropout, self.training)

class STBlock(nn.Module):
    """
    GraphWaveNet 风格的时空模块，输出 K 步的 (beta, gamma) 参数场。
    期望输入 x 为 [B, C_in, N, H]。
    """
    def __init__(self, num_nodes, dropout, in_dim, out_len,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=2, layers=3):
        super(STBlock, self).__init__()
        self.blocks = blocks; self.layers = layers
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1,1))
        self.supports_len = 2
        self.dropout = dropout

        self.filter_convs, self.gate_convs = nn.ModuleList(), nn.ModuleList()
        self.residual_convs, self.skip_convs = nn.ModuleList(), nn.ModuleList()
        self.ln, self.gconv = nn.ModuleList(), nn.ModuleList()

        receptive_field = 1
        for _b in range(blocks):
            add_scope, dil = 1, 1
            for _i in range(layers):
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=(1,dil)))
                self.gate_convs.append(  nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=(1,dil)))
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1,1)))
                self.skip_convs.append(    nn.Conv2d(dilation_channels, skip_channels,     kernel_size=(1,1)))
                self.ln.append(nn.LayerNorm([residual_channels, num_nodes, 1]))
                self.gconv.append(GCN(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
                dil *= 2; receptive_field += add_scope; add_scope *= 2

        self.end_conv_b1 = nn.Conv2d(skip_channels*blocks*layers, end_channels, kernel_size=(1,1))
        self.end_conv_b2 = nn.Conv2d(end_channels, out_len, kernel_size=(1,1))
        self.end_conv_g1 = nn.Conv2d(skip_channels*blocks*layers, end_channels, kernel_size=(1,1))
        self.end_conv_g2 = nn.Conv2d(end_channels, out_len, kernel_size=(1,1))
        self.receptive_field = receptive_field

    def forward(self, x_cnh, supports: List[torch.Tensor]):
        """
        x_cnh: [B, C_in, N, H]
        supports: length-2 list of graphs （正向/转置归一化）
          - 动态图场景：每个都是 [B,N,N]（已按时间聚合到一个图）
        返回:
          param_b, param_g: [B, K, N, 1]  （β_t(n), γ_t(n)）
        """
        B, C, N, H = x_cnh.shape
        # ensure temporal length >= receptive field
        if H < self.receptive_field:
            x = F.pad(x_cnh, (self.receptive_field - H, 0, 0, 0))  # pad time-left
        else:
            x = x_cnh
        x = self.start_conv(x)    # [B, R, N, L]
        skip = None

        for i in range(self.blocks * self.layers):
            res = x
            filt = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filt * gate

            s = self.skip_convs[i](x)
            skip = s if skip is None else torch.cat((s, skip[..., -s.size(3):]), dim=1)

            x = self.gconv[i](x, supports)
            dense = res[..., -x.size(3):]
            g = torch.sigmoid(x)
            x = x * g + dense * (1 - g)
            # LayerNorm 按照 [B, C, N, L] 的 C,N,L 归一（把 L 当 1 处理以稳定）
            x = self.ln[i](x.mean(dim=3, keepdim=True)).expand_as(x)

        pb = F.relu(skip)
        pb = F.relu(self.end_conv_b1(pb))
        pb = torch.sigmoid(self.end_conv_b2(pb))  # [B, K, N, 1]

        pg = F.relu(skip)
        pg = F.relu(self.end_conv_g1(pg))
        pg = torch.sigmoid(self.end_conv_g2(pg))  # [B, K, N, 1]

        return pb, pg

class SIRStep(nn.Module):
    """
    单步集合种群 SIR 通勤更新，返回 (I_new, S_next, I_next, R_next)，全部原始尺度。
    """
    def forward(self, beta_t: torch.Tensor, gamma_t: torch.Tensor,
                mob_t: torch.Tensor, SIR_t: torch.Tensor):
        """
        beta_t, gamma_t: [B, N]  （β_t(n), γ_t(n) ∈ (0,1)）
        mob_t:           [B, N, N] 原始尺度通勤流量
        SIR_t:           [B, N, 3] 依次 (S,I,R) 原始尺度
        """
        B, N = beta_t.shape
        S = SIR_t[..., 0]; I = SIR_t[..., 1]; R = SIR_t[..., 2]    # [B,N]
        pop = (S + I + R).clamp_min(1.0)                           # [B,N]

        # exposure propagation from bidirectional flows
        # 计算每个目的地的“外来感染压力”
        # inbound: sum_m mob(m->n) * I(m)/pop(m)
        inbound = torch.einsum('bmn,bm->bn', mob_t, (I / pop))
        # outbound 对称项：sum_m mob(n->m) * I(m)/pop(m)
        outbound = torch.einsum('bnm,bm->bn', mob_t, (I / pop))
        propagation = inbound + outbound                           # [B,N]

        I_new = beta_t * propagation                               # [B,N]
        I_new = torch.clamp_min(I_new, 0.0)
        I_new = torch.minimum(I_new, S)                            # 不超过易感

        recov = gamma_t * I
        S_next = S - I_new
        I_next = (I + I_new - recov).clamp_min(0.0)
        R_next = R + recov
        return torch.stack([I_new, S_next, I_next, R_next], dim=-1)  # [B,N,4]

class MEPOGNN(nn.Module):
    """
    选用 Dynamic 图构造：用 softmax(K,H) 把 H 段历史 OD 聚成 K 段未来迁移强度。
    输出 K 步原始尺度新增 [B,K,N]。
    """
    def __init__(self, num_nodes: int, in_dim: int, in_len: int, out_len: int,
                 dropout: float = 0.3, glm_type: str = 'Dynamic',
                 residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=256,
                 kernel_size=2, blocks=2, layers=3, od_scale_factor: float = 3.0):
        super(MEPOGNN, self).__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.in_len = in_len
        self.out_len = out_len
        self.dropout = dropout
        self.glm_type = glm_type
        self.od_scale_factor = od_scale_factor

        self.stblock = STBlock(num_nodes, dropout, in_dim, out_len,
                               residual_channels, dilation_channels, skip_channels,
                               end_channels, kernel_size, blocks, layers)
        self.sir = SIRStep()

        if self.glm_type == 'Adaptive':
            self.register_parameter('log_g_rescaled', None)  # 预留
        elif self.glm_type == 'Dynamic':
            self.incidence = nn.Parameter(torch.empty(out_len, in_len))
            nn.init.normal_(self.incidence, mean=1.0, std=0.01)
        else:
            raise ValueError("glm_type must be 'Dynamic' or 'Adaptive'")

    def forward(self,
                x_btnc: torch.Tensor,          # [B,H,N,C] 标准化
                sir_hist_raw: torch.Tensor,    # [B,H,N,3] 原始尺度 (S,I,R)
                od_hist_norm: torch.Tensor,    # [B,H,N,N] 归一化
                max_od: float,                 # 标量
                y_norm_stats: Optional[Dict] = None
                ) -> Tuple[torch.Tensor, Dict]:
        B, H, N, C = x_btnc.shape
        K = self.out_len

        # 1) 图聚合 (Dynamic)：用 softmax(K,H) 把历史 H 段 OD 聚出未来 K 段的图
        inc = torch.softmax(self.incidence, dim=1)                 # [K,H]
        # od_hist_norm -> [B,H,N,N]；聚合成 [B,K,N,N]，再放大回原始尺度
        gk = torch.einsum('kh, bhmn -> bkmn', inc, od_hist_norm)   # [B,K,N,N]
        mob_k = gk * (max_od * self.od_scale_factor)               # [B,K,N,N] 原始尺度

        # 2) 支持图给 ST block：对每个样本构造一个“汇总图”给 GCN
        #    这里取 K 个图在时间维度做平均当作 supports（也可只取最近一步）
        g_agg = mob_k.mean(dim=1)                                  # [B,N,N]
        g_t = g_agg.transpose(1,2).contiguous()                    # [B,N,N]
        supports = [g_agg, g_t]

        # 3) ST block 产出 K 步 β, γ 参数场（每步每节点）
        x_cnh = x_btnc.permute(0, 3, 2, 1).contiguous()               # [B,C,N,H]
        param_b, param_g = self.stblock(x_cnh, supports)           # [B,K,N,1]×2
        beta = torch.clamp(param_b.squeeze(-1), 0.0, 1.0)          # [B,K,N]
        gamma = torch.clamp(param_g.squeeze(-1), 0.0, 1.0)         # [B,K,N]

        # 4) 以历史窗最后时刻作为初值，在原始尺度前推 K 步
        SIR_t = sir_hist_raw[:, -1, :, :]                          # [B,N,3]
        outs = []
        for k in range(K):
            # mob_k[b,k]: [B,N,N]； beta/gamma 第 k 步：[B,N]
            step_out = self.sir(beta[:,k,:], gamma[:,k,:], mob_k[:,k,:,:], SIR_t)  # [B,N,4]
            I_new, S_next, I_next, R_next = step_out.split([1,1,1,1], dim=-1)
            outs.append(I_new.squeeze(-1))                         # [B,N]
            SIR_t = torch.cat([S_next, I_next, R_next], dim=-1)    # 下一步初值

        y_pred_raw = torch.stack(outs, dim=1)                      # [B,K,N]

        # 5) （可选）标准化到标签空间（供训练损失）
        if y_norm_stats is not None and y_norm_stats.get('type','zscore') == 'zscore':
            mu = torch.as_tensor(y_norm_stats['mean'], device=y_pred_raw.device, dtype=y_pred_raw.dtype).view(1, 1, 1)
            std = torch.as_tensor(y_norm_stats['std'], device=y_pred_raw.device, dtype=y_pred_raw.dtype).clamp_min(
                1e-6).view(1, 1, 1)
            y_pred_norm = (y_pred_raw - mu) / std
        else:
            y_pred_norm = y_pred_raw

        aux = {"beta": beta, "gamma": gamma, "mob_k": mob_k, "g_agg": g_agg}
        return y_pred_raw, y_pred_norm, aux



