import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, Optional
from odegcn import ODEG


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        if len(A.shape) == 2:
            x = torch.einsum('vw, ncwl->ncvl', A, x)
        else:
            x = torch.einsum('nvw, ncwl->ncvl', A, x)
        return x.contiguous()


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it  额外的维度将通过 padding 添加，这里删除它。
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        return self.mlp(x)


def _as_tensor(x, device, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _clamp_nonnegative_with_max(x: torch.Tensor, max_tensor: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本兼容的 clamp：先做下界 0，再用 torch.minimum 做张量上界。

    老版本 PyTorch 不支持 torch.clamp(x, min=float, max=Tensor)，因此不能混用
    Number min 和 Tensor max。
    """
    x = torch.clamp(x, min=0.0)
    max_tensor = max_tensor.to(device=x.device, dtype=x.dtype)
    if max_tensor.shape != x.shape:
        max_tensor = max_tensor.expand_as(x)
    return torch.min(x, max_tensor)




def reconstruct_I0_by_convolution(newI_hist: torch.Tensor,
                                  gamma_hist: torch.Tensor,
                                  hist_conv_len: int = 14,
                                  remove_new_cases_same_day: bool = False,
                                  eps: float = 1e-8) -> torch.Tensor:
    """用历史新增感染序列卷积重构预测起点活跃感染者 I_T。

    数学含义：
        I_T = sum_u newI_u * prod_{s=u+1}^{T-1}(1-gamma_s)
    如果 remove_new_cases_same_day=True，则使用 prod_{s=u}^{T-1}(1-gamma_s)。

    Args:
        newI_hist: [B,N,T] 或 [B,N,T,1]，应为原始病例尺度。
        gamma_hist: 标量、[B,N,1,1]、[B,N,T,1] 或可广播张量。
        hist_conv_len: 只使用最近 L 天历史新增感染，避免过长历史引入噪声。
        remove_new_cases_same_day: 新增感染者是否在感染当天也参与移除。

    Returns:
        I0_conv: [B,N]，预测起点的活跃感染者代理。
    """
    if newI_hist.dim() == 3:
        newI_hist = newI_hist.unsqueeze(-1)
    if newI_hist.dim() != 4 or newI_hist.size(-1) != 1:
        raise ValueError(f"newI_hist should have shape [B,N,T] or [B,N,T,1], got {tuple(newI_hist.shape)}")

    B, N, T, _ = newI_hist.shape
    device = newI_hist.device
    dtype = newI_hist.dtype
    L = min(int(hist_conv_len), T)
    if L <= 0:
        raise ValueError(f"hist_conv_len must be positive, got {hist_conv_len}")

    newI_hist = torch.clamp(torch.nan_to_num(newI_hist[:, :, -L:, :], nan=0.0, posinf=0.0, neginf=0.0), min=0.0)

    if not torch.is_tensor(gamma_hist):
        gamma_hist = torch.tensor(gamma_hist, device=device, dtype=dtype)
    gamma_hist = gamma_hist.to(device=device, dtype=dtype)

    if gamma_hist.dim() == 0:
        gamma_hist = gamma_hist.view(1, 1, 1, 1).expand(B, N, L, 1)
    elif gamma_hist.dim() == 2:
        gamma_hist = gamma_hist.unsqueeze(-1).unsqueeze(-1).expand(B, N, L, 1)
    elif gamma_hist.dim() == 3:
        gamma_hist = gamma_hist.unsqueeze(-1)

    if gamma_hist.dim() != 4:
        raise ValueError(f"gamma_hist should be broadcastable to [B,N,T,1], got {tuple(gamma_hist.shape)}")

    if gamma_hist.size(2) == 1:
        gamma_hist = gamma_hist.expand(B, N, L, 1)
    else:
        gamma_hist = gamma_hist[:, :, -L:, :]
        if gamma_hist.size(2) != L:
            raise ValueError(f"gamma_hist time length {gamma_hist.size(2)} cannot match L={L}")

    gamma_hist = torch.clamp(torch.nan_to_num(gamma_hist, nan=0.0, posinf=0.99, neginf=0.0), 0.0, 0.99)
    survival = 1.0 - gamma_hist

    I0 = torch.zeros(B, N, 1, device=device, dtype=dtype)
    for u in range(L):
        start = u if remove_new_cases_same_day else u + 1
        if start <= L - 1:
            weight = torch.prod(survival[:, :, start:L, :], dim=2)
        else:
            weight = torch.ones(B, N, 1, device=device, dtype=dtype)
        I0 = I0 + newI_hist[:, :, u, :] * weight

    return torch.clamp(I0.squeeze(-1), min=eps)

def process_dynamic_adj(raw_adj: torch.Tensor,
                        threshold: float = 0.0,
                        device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    动态邻接矩阵处理函数。
    保留原始逻辑：mobility > threshold 得到 binary_adj，再做行归一化。
    """
    adj = raw_adj.squeeze(-1).to(device) if raw_adj.dim() == 5 else raw_adj.to(device)
    batch_size, time_steps, num_nodes, _ = adj.shape

    binary_adj = torch.zeros_like(adj)
    normalized_adj = torch.zeros_like(adj)
    eye = torch.eye(num_nodes, device=device)

    for b in range(batch_size):
        for t in range(time_steps):
            current_adj = adj[b, t]
            bin_adj = (current_adj > threshold).float()
            bin_adj = bin_adj * (1 - eye)
            binary_adj[b, t] = bin_adj

            row_sum = bin_adj.sum(dim=-1, keepdim=True) + 1e-7
            norm_adj = bin_adj / row_sum
            normalized_adj[b, t] = norm_adj

    return binary_adj, normalized_adj


def process_dynamic_adj_dict(raw_adj: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
    bin_adj, norm_adj = process_dynamic_adj(raw_adj, **kwargs)
    return {
        "binary_adj": bin_adj,
        "normalized_adj": norm_adj,
        "mask": (bin_adj > 0).float()
    }


class TemporalConvNet(nn.Module):
    """time dilation convolution"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.3):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=(1, kernel_size),
                             dilation=(1, dilation_size), padding=(0, padding))
            conv.weight.data.normal_(0, 0.01)
            chomp = Chomp1d(padding)
            relu = nn.ReLU()
            dropout_layer = nn.Dropout(dropout)
            layers += [nn.Sequential(conv, chomp, relu, dropout_layer)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)  # (B, F, N, T)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)  # (B, N, T, F)
        return y


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, history_window, num_nodes, int_init):
        super(STGCNBlock, self).__init__()
        self.int_init = int_init
        self.temporal1 = TemporalConvNet(num_inputs=in_channels, num_channels=out_channels)

        time_length = float(history_window - 1)
        self.odeg = ODEG(out_channels[-1], history_window, num_nodes, self.int_init, time=time_length)

        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1], num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X, adj):
        output_t1 = self.temporal1(X)
        output_ode = self.odeg(output_t1, adj)
        output_t2 = self.temporal2(F.relu(output_ode))
        return self.batch_norm(output_t2)


class MetaSIRRollout(nn.Module):
    """
    机制主生成模块。

    与原 SIR_cell 的区别：
    1) 原 SIR_cell 用历史 new_I 与 gamma 卷积构造未来 In_t；
    2) 新版从 last_SIR 或 Sn/new_I fallback 得到 (S0, I0, R0)，逐步递推未来状态；
    3) 输出 newI_mech_raw 作为机制主预测，并保留 d_SIR[..., 1] = newI_mech_raw。

    注意：这里尽量保留你原来的三项传播项 term1/term2/internal_term，只改变“状态递推方式”。
    """
    def __init__(self,
                 last_sir_format: str = "SIR",
                 clamp_states: bool = True,
                 use_mobility_binary_mask: bool = True,
                 i_init_mode: str = "last_sir",
                 i0_blend_alpha: float = 0.5,
                 hist_conv_len: int = 14,
                 conv_gamma_mode: str = "fixed",
                 gamma_fixed: float = 0.14,
                 remove_new_cases_same_day: bool = False):
        super(MetaSIRRollout, self).__init__()
        self.last_sir_format = last_sir_format
        self.clamp_states = clamp_states
        self.use_mobility_binary_mask = use_mobility_binary_mask
        self.i_init_mode = i_init_mode
        self.i0_blend_alpha = float(i0_blend_alpha)
        self.hist_conv_len = int(hist_conv_len)
        self.conv_gamma_mode = conv_gamma_mode
        self.gamma_fixed = float(gamma_fixed)
        self.remove_new_cases_same_day = bool(remove_new_cases_same_day)

    def _parse_last_or_fallback_state(self,
                                      last_SIR: Optional[torch.Tensor],
                                      Sn: torch.Tensor,
                                      new_I: torch.Tensor,
                                      static_feats: torch.Tensor,
                                      gamma: torch.Tensor,
                                      device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """解析 last_SIR 得到基础 S0, I0_last, R0；如果不可用则 fallback。"""
        N_pop = _as_tensor(static_feats, device)[:, 0].view(1, -1)
        B = Sn.shape[0]
        N = Sn.shape[1]

        if last_SIR is not None:
            z = _as_tensor(last_SIR, device)
            if z.dim() == 4:
                z = z[:, :, -1, :]
            elif z.dim() == 2:
                z = z.unsqueeze(0).expand(B, -1, -1)

            if z.dim() == 3 and z.shape[-1] >= 3:
                fmt = self.last_sir_format
                if fmt == "SIR":
                    S0, I0, R0 = z[..., 0], z[..., 1], z[..., 2]
                elif fmt == "newI_SIR" and z.shape[-1] >= 4:
                    S0, I0, R0 = z[..., 1], z[..., 2], z[..., 3]
                elif fmt == "SIRD":
                    S0, I0, R0 = z[..., 0], z[..., 1], z[..., 2]
                elif fmt == "ISRD" and z.shape[-1] >= 3:
                    I0, S0, R0 = z[..., 0], z[..., 1], z[..., 2]
                else:
                    S0, I0, R0 = z[..., 0], z[..., 1], z[..., 2]

                S0 = torch.nan_to_num(S0, nan=0.0, posinf=0.0, neginf=0.0)
                I0 = torch.nan_to_num(I0, nan=0.0, posinf=0.0, neginf=0.0)
                R0 = torch.nan_to_num(R0, nan=0.0, posinf=0.0, neginf=0.0)
                return S0, I0, R0

        Sn_t = _as_tensor(Sn, device)
        if Sn_t.dim() == 4:
            Sn_t = Sn_t[..., 0]
        if Sn_t.dim() == 3:
            S0 = Sn_t[:, :, 0]
        else:
            S0 = Sn_t

        new_I_t = _as_tensor(new_I, device)
        if new_I_t.dim() == 4:
            new_I_t = new_I_t[..., 0]
        if new_I_t.dim() == 3:
            recent_newI = torch.clamp(new_I_t[:, :, -1], min=0.0)
        else:
            recent_newI = torch.clamp(new_I_t, min=0.0)

        gamma0 = torch.clamp(gamma[:, :, 0, 0], min=1e-3)
        I0 = recent_newI / gamma0
        R0 = torch.clamp(N_pop.expand(B, N) - S0 - I0, min=0.0)
        return S0, I0, R0

    def _build_gamma_hist_for_convolution(self,
                                          new_I: torch.Tensor,
                                          param_g: torch.Tensor,
                                          device) -> torch.Tensor:
        """为历史卷积构造 gamma_hist。"""
        if new_I.dim() == 3:
            B, N, T = new_I.shape
        elif new_I.dim() == 4:
            B, N, T = new_I.shape[:3]
        else:
            raise ValueError(f"new_I should have [B,N,T] or [B,N,T,1], got {tuple(new_I.shape)}")

        dtype = new_I.dtype
        if self.conv_gamma_mode == "fixed":
            gamma_hist = torch.ones(B, N, T, 1, device=device, dtype=dtype) * self.gamma_fixed
        elif self.conv_gamma_mode == "pred_mean":
            gamma_mean = param_g.mean(dim=2, keepdim=True)
            gamma_hist = gamma_mean.expand(B, N, T, 1)
        elif self.conv_gamma_mode == "pred_first":
            gamma_first = param_g[:, :, :1, :]
            gamma_hist = gamma_first.expand(B, N, T, 1)
        else:
            raise ValueError(f"Unknown conv_gamma_mode: {self.conv_gamma_mode}")
        return gamma_hist

    def _extract_initial_state(self,
                               last_SIR: Optional[torch.Tensor],
                               Sn: torch.Tensor,
                               new_I: torch.Tensor,
                               static_feats: torch.Tensor,
                               gamma: torch.Tensor,
                               device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 S0, I0, R0，形状均为 [B,N]。

        i_init_mode:
        - last_sir: 使用 last_SIR / fallback 中的 I0；
        - hist_conv: 使用历史 new_I 通过 survival convolution 重构 I0；
        - hybrid: I0 = alpha * I0_last + (1-alpha) * I0_conv。
        """
        S0, I0_last, R0 = self._parse_last_or_fallback_state(last_SIR, Sn, new_I, static_feats, gamma, device)
        mode = self.i_init_mode
        if mode == "last_sir":
            return S0, I0_last, R0
        if mode not in ("hist_conv", "hybrid"):
            raise ValueError(f"Unknown i_init_mode: {mode}")

        new_I_t = _as_tensor(new_I, device)
        if new_I_t.dim() == 3:
            new_I_t = new_I_t.unsqueeze(-1)
        gamma_hist = self._build_gamma_hist_for_convolution(new_I_t, gamma, device)
        I0_conv = reconstruct_I0_by_convolution(
            newI_hist=new_I_t,
            gamma_hist=gamma_hist,
            hist_conv_len=self.hist_conv_len,
            remove_new_cases_same_day=self.remove_new_cases_same_day
        )

        if mode == "hist_conv":
            I0 = I0_conv
        else:
            alpha = min(max(float(self.i0_blend_alpha), 0.0), 1.0)
            I0 = alpha * I0_last + (1.0 - alpha) * I0_conv
        return S0, I0, R0

    def forward(self,
                Sn: torch.Tensor,
                new_I: torch.Tensor,
                static_feats: torch.Tensor,
                param_b: torch.Tensor,
                param_g: torch.Tensor,
                adj_dict: Dict[str, torch.Tensor],
                mob: torch.Tensor,
                device,
                last_SIR: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_node = static_feats.shape[0]
        horizon = param_b.shape[2]
        B = param_b.shape[0]

        N_pop = _as_tensor(static_feats, device)[:, 0]
        pop_m = N_pop.unsqueeze(0).unsqueeze(1)      # (1, 1, N)
        pop_n = N_pop.unsqueeze(0).unsqueeze(-1)     # (1, N, 1)
        pop_vec = N_pop.unsqueeze(0)                 # (1, N)

        S_t, I_t, R_t = self._extract_initial_state(last_SIR, Sn, new_I, static_feats, param_g, device)
        S_t = S_t.to(device)
        I_t = I_t.to(device)
        R_t = R_t.to(device)

        if self.clamp_states:
            S_t = _clamp_nonnegative_with_max(S_t, pop_vec)
            I_t = _clamp_nonnegative_with_max(I_t, pop_vec)
            R_t = torch.clamp(R_t, min=0.0)

        beta_n_all = param_b.permute(0, 2, 1, 3)  # (B,H,N,1)
        beta_m_all = torch.einsum('btji,btj->bti',
                                  adj_dict['normalized_adj'],
                                  beta_n_all.squeeze(-1))
        beta_m_all = beta_m_all.unsqueeze(-1).permute(0, 2, 1, 3)  # (B,N,H,1)
        beta_n_all = beta_n_all.permute(0, 2, 1, 3)                # (B,N,H,1)

        I_states, newI_list, dI_list, dR_list = [], [], [], []
        S_states, R_states = [], []

        eye = torch.eye(num_node, device=device).unsqueeze(0)

        for k in range(horizon):
            bn_t = beta_n_all[:, :, k, :].squeeze(-1)  # (B,N)
            bm_t = beta_m_all[:, :, k, :].squeeze(-1)  # (B,N)
            gamma_t = param_g[:, :, k, :].squeeze(-1)  # (B,N)

            h_mn_t = mob[:, k, :, :]                   # (B,N,N)
            h_nm_t = mob[:, k, :, :].transpose(1, 2)
            h_nn_t = h_mn_t * eye * 0.5

            if self.use_mobility_binary_mask:
                h_mn_eff = h_mn_t * adj_dict['binary_adj'][:, k, ...]
            else:
                h_mn_eff = h_mn_t

            # Term1: infectious visitors from m contribute to infection risk in n.
            mobility_term1 = h_mn_eff / (pop_m + 1e-7)  # (B,N,N)
            sum_Im = torch.einsum('bnm,bm->bn', mobility_term1, I_t)
            term1 = bn_t * (sum_Im * S_t / (pop_n.squeeze(-1) + 1e-7))

            # Term2: susceptible residents from n visit m and are exposed at destination m.
            mobility_term2 = h_nm_t / (pop_n + 1e-7)
            sum_Sn = torch.einsum('bnm,bn->bm', mobility_term2, S_t)
            term2 = bm_t * (sum_Sn * I_t / (pop_m.squeeze(1) + 1e-7))

            # Term3: internal/local transmission.
            mobility_term3 = h_nn_t / (pop_n + 1e-7)
            sum_Sn_local = torch.einsum('bnm,bn->bm', mobility_term3, S_t)
            internal_term = 2.0 * bn_t * (sum_Sn_local * I_t) / (pop_n.squeeze(-1) + 1e-7)

            new_inf = term1 + term2 + internal_term
            new_inf = torch.nan_to_num(new_inf, nan=0.0, posinf=0.0, neginf=0.0)
            new_inf = torch.clamp(new_inf, min=0.0)
            if self.clamp_states:
                new_inf = torch.minimum(new_inf, torch.clamp(S_t, min=0.0))

            removal = gamma_t * I_t
            removal = torch.nan_to_num(removal, nan=0.0, posinf=0.0, neginf=0.0)
            removal = torch.clamp(removal, min=0.0)
            if self.clamp_states:
                removal = torch.minimum(removal, torch.clamp(I_t + new_inf, min=0.0))

            dI = new_inf - removal
            dR = removal

            I_states.append(I_t.unsqueeze(-1))
            S_states.append(S_t.unsqueeze(-1))
            R_states.append(R_t.unsqueeze(-1))
            newI_list.append(new_inf.unsqueeze(-1))
            dI_list.append(dI.unsqueeze(-1))
            dR_list.append(dR.unsqueeze(-1))

            S_t = S_t - new_inf
            I_t = I_t + dI
            R_t = R_t + dR
            if self.clamp_states:
                S_t = _clamp_nonnegative_with_max(S_t, pop_vec)
                I_t = _clamp_nonnegative_with_max(I_t, pop_vec)
                R_t = torch.clamp(R_t, min=0.0)

        I_stack = torch.stack(I_states, dim=2)       # (B,N,H,1)
        newI_stack = torch.stack(newI_list, dim=2)   # (B,N,H,1)
        dI_stack = torch.stack(dI_list, dim=2)       # (B,N,H,1)
        dR_stack = torch.stack(dR_list, dim=2)       # (B,N,H,1)
        S_stack = torch.stack(S_states, dim=2)
        R_stack = torch.stack(R_states, dim=2)

        # 保持旧接口语义：d_SIR[..., 1] 是 new infections，也就是 -dS。
        d_SIR = torch.cat((I_stack, newI_stack, dI_stack, dR_stack), dim=-1)
        return d_SIR, newI_stack, S_stack, I_stack, R_stack


# 向后兼容：保留旧类名。如果你想完全回到旧机制辅助分支，可用 forecast_mode=legacy_parallel。
class SIR_cell(MetaSIRRollout):
    def __init__(self):
        super(SIR_cell, self).__init__()


class STGODE_SIR(nn.Module):
    """The overall network framework.

    新增核心开关：
    - forecast_mode:
        legacy_parallel: 原始并行预测头 pred_newI = output_decoder(h)，机制只作辅助。
        mech_only:       pred_newI = Meta-SIR rollout(beta, gamma)；不使用残差。
        mech_residual:   pred_newI = Meta-SIR rollout(beta, gamma) + residual_scale * residual.
    - gamma_mode:
        learn_free: 原始自由学习 gamma = 0.01 + 0.29 sigmoid(raw)。
        fixed:      gamma 固定为 gamma_fixed。
        weak:       gamma = gamma_fixed + gamma_amp * tanh(raw)，再 clamp。
    - encoder_fusion:
        max_parallel: 保留原始三个 block 并行取 max。
        mean_parallel: 三个 block 并行取 mean。
        sequential: 真的串联 3 个 STGCNBlock（新增，但可能改变原模型表达）。
    - i_init_mode:
        last_sir:  使用 last_SIR 中的 I_T。
        hist_conv: 用历史 new_I 卷积重构 I_T，再进入未来机制递推。
        hybrid:    融合 last_SIR 的 I_T 与卷积重构 I_T。
    """
    def __init__(self,
                 num_nodes,
                 num_features,
                 timesteps_input,
                 timesteps_output,
                 forecast_mode: str = "mech_residual",
                 gamma_mode: str = "fixed",
                 gamma_fixed: float = 0.14,
                 gamma_amp: float = 0.03,
                 gamma_min: float = 0.05,
                 gamma_max: float = 0.25,
                 beta_min: float = 1e-5,
                 beta_max: float = 1.5,
                 residual_scale: float = 0.05,
                 encoder_fusion: str = "max_parallel",
                 od_scale_factor: float = 3.0,
                 last_sir_format: str = "SIR",
                 clamp_states: bool = True,
                 use_mobility_binary_mask: bool = True,
                 i_init_mode: str = "last_sir",
                 i0_blend_alpha: float = 0.5,
                 hist_conv_len: int = 14,
                 conv_gamma_mode: str = "fixed",
                 remove_new_cases_same_day: bool = False,
                 history_newI_is_normalized: bool = False,
                 future_mobility_mode: str = "learned_attn",
                 future_recent_k: int = 7,
                 future_week_period: int = 7,
                 future_blend_alpha: float = 0.7,
                 detach_mech_mobility: bool = False):
        super(STGODE_SIR, self).__init__()

        self.num_nodes = num_nodes
        self.timesteps_input = timesteps_input
        self.timesteps_output = timesteps_output
        self.forecast_mode = forecast_mode
        self.gamma_mode = gamma_mode
        self.gamma_fixed = gamma_fixed
        self.gamma_amp = gamma_amp
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.residual_scale = residual_scale
        self.encoder_fusion = encoder_fusion
        self.od_scale_factor = od_scale_factor
        self.i_init_mode = i_init_mode
        self.i0_blend_alpha = i0_blend_alpha
        self.hist_conv_len = hist_conv_len
        self.conv_gamma_mode = conv_gamma_mode
        self.remove_new_cases_same_day = bool(remove_new_cases_same_day)
        self.history_newI_is_normalized = bool(history_newI_is_normalized)
        # ===== 机制分支未来 mobility 构造开关 =====
        # 重要：这些开关只控制 Meta-SIR rollout 中的未来 mobility forcing，
        # 不改变 encoder / ODEG 内部对历史 mobility 的使用。
        self.future_mobility_mode = future_mobility_mode
        self.future_recent_k = int(future_recent_k)
        self.future_week_period = int(future_week_period)
        self.future_blend_alpha = float(future_blend_alpha)
        self.detach_mech_mobility = bool(detach_mech_mobility)

        self.int_init = nn.Parameter(torch.empty(timesteps_output, timesteps_input), requires_grad=True)
        nn.init.normal_(self.int_init, 1, 0.01)

        if encoder_fusion == "sequential":
            self.ST_blocks = nn.ModuleList([
                STGCNBlock(in_channels=num_features, out_channels=[32, 16, 32],
                           history_window=timesteps_input, num_nodes=num_nodes, int_init=self.int_init),
                STGCNBlock(in_channels=32, out_channels=[32, 16, 32],
                           history_window=timesteps_input, num_nodes=num_nodes, int_init=self.int_init),
                STGCNBlock(in_channels=32, out_channels=[32, 16, 32],
                           history_window=timesteps_input, num_nodes=num_nodes, int_init=self.int_init),
            ])
        else:
            # 保留原始结构：3 个 block 接收同一个输入，后续 max/mean fusion。
            self.ST_blocks = nn.ModuleList(
                [STGCNBlock(in_channels=num_features, out_channels=[32, 16, 32],
                            history_window=timesteps_input, num_nodes=num_nodes, int_init=self.int_init)
                 for _ in range(3)]
            )

        self.output_param = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, timesteps_output * 2),
        )

        # 原来的 output_decoder 保留，但在新结构中它只作为 residual decoder。
        self.output_decoder = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, timesteps_output * 1),
        )

        self.SIR_cell = MetaSIRRollout(last_sir_format=last_sir_format,
                                       clamp_states=clamp_states,
                                       use_mobility_binary_mask=use_mobility_binary_mask,
                                       i_init_mode=i_init_mode,
                                       i0_blend_alpha=i0_blend_alpha,
                                       hist_conv_len=hist_conv_len,
                                       conv_gamma_mode=conv_gamma_mode,
                                       gamma_fixed=gamma_fixed,
                                       remove_new_cases_same_day=remove_new_cases_same_day)

    def encode(self, x_input, adj):
        if self.encoder_fusion == "sequential":
            h = x_input
            for blk in self.ST_blocks:
                h = blk(h, adj)
            return h[:, :, -1, :]

        outs = []
        for blk in self.ST_blocks:
            outs.append(blk(x_input, adj))
        outs = torch.stack(outs, dim=0)
        if self.encoder_fusion == "mean_parallel":
            x_out = torch.mean(outs, dim=0)
        else:
            x_out = torch.max(outs, dim=0)[0]
        return x_out[:, :, -1, :]

    def decode_params(self, x_out):
        B, N, _ = x_out.shape
        raw = self.output_param(x_out).reshape(B, N, self.timesteps_output, 2)
        beta_raw = raw[..., [0]]
        gamma_raw = raw[..., [1]]

        param_b = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(beta_raw)

        if self.gamma_mode == "fixed":
            param_g = torch.ones_like(param_b) * self.gamma_fixed
        elif self.gamma_mode == "weak":
            param_g = self.gamma_fixed + self.gamma_amp * torch.tanh(gamma_raw)
            param_g = torch.clamp(param_g, min=self.gamma_min, max=self.gamma_max)
        elif self.gamma_mode == "learn_free":
            param_g = 0.01 + 0.29 * torch.sigmoid(gamma_raw)
        else:
            raise ValueError(f"Unknown gamma_mode: {self.gamma_mode}")
        return param_b, param_g

    def _cycle_to_horizon(self, template: torch.Tensor) -> torch.Tensor:
        """把历史 mobility 模板循环/重复到预测长度 H。

        Args:
            template: [B,L,N,N]，L 可以是 1、recent_k、week_period 等。
        Returns:
            future_mob: [B,H,N,N]
        """
        H = self.timesteps_output
        if template.dim() != 4:
            raise ValueError(f"template should be [B,L,N,N], got {tuple(template.shape)}")
        L = template.size(1)
        if L <= 0:
            raise ValueError("future mobility template length must be positive")
        reps = (H + L - 1) // L
        return template.repeat(1, reps, 1, 1)[:, :H, :, :]

    def _build_future_mobility_template(self, M_hist: torch.Tensor) -> torch.Tensor:
        """从历史 mobility 构造未来机制分支使用的 mobility 模板。

        该函数只使用历史窗口内的 mobility，不使用未来真实 mobility。
        Modes:
            - last_repeat:     使用最后一天 mobility，重复到未来 H 步。
            - recent_cycle:    使用最近 future_recent_k 天 mobility，并循环/截断到未来 H 步。
                                当 future_recent_k=pred_window 且 history_window 足够时，等价于“最近 H 天对齐”。
            - weekday_cycle:   使用最近 future_week_period 天 mobility，并循环/截断到未来 H 步。
                                对 history_window=14,pred_window=7，通常就是上一周同星期模式。
            - hybrid_cycle:    alpha * weekday_cycle + (1-alpha) * recent_cycle。
            - mean_repeat:     历史窗口均值，重复到未来 H 步。
            - recent_mean:     最近 future_recent_k 天均值，重复到未来 H 步。
        """
        if M_hist.dim() == 5:
            M_hist = M_hist.squeeze(-1)
        if M_hist.dim() != 4:
            raise ValueError(f"M_hist should be [B,Th,N,N] or [B,Th,N,N,1], got {tuple(M_hist.shape)}")

        B, Th, N, _ = M_hist.shape
        mode = self.future_mobility_mode

        if mode == "last_repeat":
            return M_hist[:, -1:, :, :]

        if mode == "recent_cycle":
            k = min(max(int(self.future_recent_k), 1), Th)
            return M_hist[:, -k:, :, :]

        if mode == "weekday_cycle":
            p = min(max(int(self.future_week_period), 1), Th)
            return M_hist[:, -p:, :, :]

        if mode == "hybrid_cycle":
            p = min(max(int(self.future_week_period), 1), Th)
            week_template = M_hist[:, -p:, :, :]
            k = min(max(int(self.future_recent_k), 1), Th)
            recent = M_hist[:, -k:, :, :]
            reps = (p + k - 1) // k
            recent_p = recent.repeat(1, reps, 1, 1)[:, :p, :, :]
            alpha = min(max(float(self.future_blend_alpha), 0.0), 1.0)
            return alpha * week_template + (1.0 - alpha) * recent_p

        if mode == "mean_repeat":
            return M_hist.mean(dim=1, keepdim=True)

        if mode == "recent_mean":
            k = min(max(int(self.future_recent_k), 1), Th)
            return M_hist[:, -k:, :, :].mean(dim=1, keepdim=True)

        raise ValueError(f"Unknown future_mobility_mode={mode}")

    def build_future_mobility(self, adj, max_od, device):
        """构造机制分支使用的未来 mobility forcing。

        注意：这里控制的是 Meta-SIR rollout 中 h_mn/h_nm/h_nn 的未来外部强迫。
        encoder/ODEG 仍然可以使用原来的历史 mobility 表征学习。
        """
        mode = self.future_mobility_mode

        if mode in ("learned_attn", "learned_attn_detach"):
            # 旧逻辑：可学习 temporal attention 聚合历史 mobility 到未来 H 步。
            # detach 版本只阻断机制分支对 int_init 的梯度，避免 mobility 与 beta 共同补偿。
            attn_param = self.int_init.detach() if (mode == "learned_attn_detach" or self.detach_mech_mobility) else self.int_init
            incidence = torch.softmax(attn_param, dim=1)
            mob = torch.einsum('kl,blnmc->bknmc', incidence, adj).squeeze(-1)
        else:
            # 显式历史模板：只使用历史 mobility，不用可学习 attention。
            M_hist = adj.squeeze(-1) if adj.dim() == 5 else adj
            template = self._build_future_mobility_template(M_hist)
            mob = self._cycle_to_horizon(template)

        mob = mob.to(device=device, dtype=torch.float32)
        return mob * max_od * self.od_scale_factor

    def forward(self,
                x_input,
                newI_original,
                Sn,
                static_feats,
                adj,
                max_od,
                device,
                last_SIR: Optional[torch.Tensor] = None,
                newI_mean: Optional[torch.Tensor] = None,
                newI_std: Optional[torch.Tensor] = None):
        """
        Returns:
            param_b:    (B,N,H,1)
            param_g:    (B,N,H,1)
            pred_newI:  (B,N,H,1), normalized scale, for loss/evaluation
            d_SIR:      (B,N,H,4), raw scale; d_SIR[...,1] = newI_mech_raw
            aux:        dict with mechanism/residual outputs
        """
        device = torch.device(device) if isinstance(device, str) else device
        B, N, T, C = x_input.shape
        x_out = self.encode(x_input, adj)

        param_b, param_g = self.decode_params(x_out)
        residual_norm = self.output_decoder(x_out).reshape(B, N, self.timesteps_output, 1)

        if newI_mean is None:
            newI_mean = torch.tensor(0.0, device=device)
        if newI_std is None:
            newI_std = torch.tensor(1.0, device=device)
        newI_mean = _as_tensor(newI_mean, device).view(1, 1, 1, 1)
        newI_std = torch.clamp(_as_tensor(newI_std, device).view(1, 1, 1, 1), min=1e-6)

        # 历史卷积重构 I_T 必须使用原始病例尺度。
        # 如果 DataLoader 给出的 newI_original 已经是 normalized scale，则用开关反标准化。
        if self.history_newI_is_normalized:
            if newI_original.dim() == 3:
                newI_for_state = newI_original * newI_std.view(1, 1, 1) + newI_mean.view(1, 1, 1)
            else:
                newI_for_state = newI_original * newI_std + newI_mean
        else:
            newI_for_state = newI_original

        mob = self.build_future_mobility(adj, max_od, device)
        adj_dict = process_dynamic_adj_dict(mob, device=device)
        d_SIR, newI_mech_raw, S_roll, I_roll, R_roll = self.SIR_cell(
            Sn, newI_for_state, static_feats, param_b, param_g, adj_dict, mob, device, last_SIR=last_SIR
        )

        newI_mech_norm = (newI_mech_raw - newI_mean) / newI_std

        if self.forecast_mode == "legacy_parallel":
            pred_newI = residual_norm
            # legacy 下 output_decoder 仍是最终预测，因此 residual 不应被惩罚。
            effective_residual = torch.zeros_like(residual_norm)
        elif self.forecast_mode == "mech_only":
            pred_newI = newI_mech_norm
            effective_residual = torch.zeros_like(residual_norm)
        elif self.forecast_mode == "mech_residual":
            effective_residual = self.residual_scale * residual_norm
            pred_newI = newI_mech_norm + effective_residual
        else:
            raise ValueError(f"Unknown forecast_mode: {self.forecast_mode}")

        aux = {
            "newI_mech_raw": newI_mech_raw,
            "newI_mech_norm": newI_mech_norm,
            "residual_norm": residual_norm,
            "effective_residual": effective_residual,
            "S_roll": S_roll,
            "I_roll": I_roll,
            "R_roll": R_roll,
            "mob": mob,
            "i_init_mode": self.i_init_mode,
            "history_newI_is_normalized": self.history_newI_is_normalized,
        }
        return param_b, param_g, pred_newI, d_SIR, aux


class BSplineBasis(nn.Module):
    def __init__(self, num_ctrl_pts, degree, pred_len):
        super(BSplineBasis, self).__init__()
        self.num_ctrl_pts = num_ctrl_pts
        self.degree = degree
        self.pred_len = pred_len

        self.register_buffer('knot_vector', self.create_knot_vector())
        self.register_buffer('x_query', torch.linspace(0, 1, pred_len))

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
            # 兼容 x=1 的最后一个基函数。
            if i == self.num_ctrl_pts - 1:
                return ((x >= knot[i]) & (x <= knot[i + 1])).float()
            return ((x >= knot[i]) & (x < knot[i + 1])).float()
        else:
            denom1 = knot[i + d] - knot[i]
            denom2 = knot[i + d + 1] - knot[i + 1]
            term1 = ((x - knot[i]) / denom1 * self.basis_function(i, d - 1, x, knot)) if denom1 > 0 else 0.0
            term2 = ((knot[i + d + 1] - x) / denom2 * self.basis_function(i + 1, d - 1, x, knot)) if denom2 > 0 else 0.0
            return term1 + term2

    def forward(self, ctrl_pts):
        B, N, K = ctrl_pts.shape
        D = self.degree
        basis_all = []
        for i in range(K):
            b_i = self.basis_function(i, D, self.x_query, self.knot_vector)
            basis_all.append(b_i)
        basis = torch.stack(basis_all, dim=0).to(ctrl_pts.device)
        out = torch.einsum('bnk,kt->bnt', ctrl_pts, basis)
        return out
