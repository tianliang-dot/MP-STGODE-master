import random
import numpy as np
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== Diffusion supports (batched, time-varying) =====================

def _row_normalize(A_bnn: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    # A: [B,N,N], row-normalized (random walk)
    d = A_bnn.sum(dim=2, keepdim=True)  # out-degree
    return A_bnn / (d + eps)

def build_diffusion_supports_batch(A_bnn: torch.Tensor, k_hop: int, add_self_loop: bool=True) -> torch.Tensor:
    """
    Build 2*k_hop supports for a batch of directed graphs:
      [I, P_f, P_f^2, ..., I, P_b, P_b^2, ...]
    A_bnn: [B,N,N]
    return: supports [B, 2*k_hop, N, N]
    """
    B,N,_ = A_bnn.shape
    device = A_bnn.device
    dtype = A_bnn.dtype
    A = A_bnn
    if add_self_loop:
        I = torch.eye(N, device=device, dtype=dtype)
        A = A + I.unsqueeze(0)

    # forward random walk  P_f
    P_f = _row_normalize(A)                     # [B,N,N]
    # backward random walk P_b on A^T
    P_b = _row_normalize(A.transpose(1,2))      # [B,N,N]

    I_b = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B,-1,-1)
    supp_f = [I_b]
    supp_b = [I_b]
    # iterative powers
    Pf_pow = I_b
    Pb_pow = I_b
    for _ in range(1, k_hop):
        Pf_pow = torch.bmm(Pf_pow, P_f)        # [B,N,N]
        Pb_pow = torch.bmm(Pb_pow, P_b)
        supp_f.append(Pf_pow)
        supp_b.append(Pb_pow)
    # [B,k_hop,N,N] concat -> [B,2*k_hop,N,N]
    supports = torch.stack(supp_f + supp_b, dim=1)
    return supports


def build_supports_sequence(A_btnn: torch.Tensor, k_hop: int) -> List[torch.Tensor]:
    """
    A_btnn: [B, T, N, N]
    Return list length T; each item is supports [B, 2*k_hop, N, N] for that time step.
    """
    B,T,N,_ = A_btnn.shape
    device = A_btnn.device  # 确保使用正确的设备
    seq = []
    for t in range(T):
        supports = build_diffusion_supports_batch(A_btnn[:, t], k_hop, add_self_loop=False)
        # 确保返回的张量在正确的设备上
        seq.append(supports.to(device))
    return seq  # list of [B,S,N,N]


# ===================== DiffusionConv with batched supports =====================

class DiffusionConv(nn.Module):
    """
    y = sum_s S_s x Θ_s, with batched supports.
    x: [B, N, Cin], supports: [B, S, N, N], theta: [S, Cin, Cout]
    return y: [B, N, Cout]
    """
    def __init__(self, in_dim: int, out_dim: int, num_supports: int, bias: bool=True):
        super(DiffusionConv, self).__init__()
        self.in_dim = in_dim; self.out_dim = out_dim; self.S = num_supports
        self.theta = nn.Parameter(torch.empty(self.S, in_dim, out_dim))
        self.bias  = nn.Parameter(torch.zeros(out_dim)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        for s in range(self.S):
            nn.init.xavier_uniform_(self.theta[s])

    def forward(self, x_bnc: torch.Tensor, supports_bsn: torch.Tensor) -> torch.Tensor:
        # supports_bsn: [B,S,N,N]
        # Step1: S_s x -> [B,S,N,Cin]
        Sx = torch.einsum('bsij,bjc->bsic', supports_bsn, x_bnc)
        # Step2: multiply per-support weights Θ_s and sum over S
        y_bsnc = torch.einsum('bsic,scd->bsid', Sx, self.theta)  # [B,S,N,Cout]
        y = y_bsnc.sum(dim=1)  # [B,N,Cout]
        if self.bias is not None:
            y = y + self.bias
        return y


# ===================== DCGRU cell (time-varying supports) =====================

class DCGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k_hop: int=2):
        super(DCGRUCell, self).__init__()
        self.input_dim = input_dim; self.hidden_dim = hidden_dim
        self.S = 2 * k_hop
        # gates
        self.conv_x_z = DiffusionConv(input_dim, hidden_dim, self.S, bias=True)
        self.conv_h_z = DiffusionConv(hidden_dim, hidden_dim, self.S, bias=False)
        self.conv_x_r = DiffusionConv(input_dim, hidden_dim, self.S, bias=True)
        self.conv_h_r = DiffusionConv(hidden_dim, hidden_dim, self.S, bias=False)
        # candidate
        self.conv_x_h = DiffusionConv(input_dim, hidden_dim, self.S, bias=True)
        self.conv_h_h = DiffusionConv(hidden_dim, hidden_dim, self.S, bias=False)

    def forward(self, x_bnc: torch.Tensor, h_prev_bnc: torch.Tensor, supports_bsn: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.conv_x_z(x_bnc, supports_bsn) + self.conv_h_z(h_prev_bnc, supports_bsn))
        r = torch.sigmoid(self.conv_x_r(x_bnc, supports_bsn) + self.conv_h_r(h_prev_bnc, supports_bsn))
        h_tilde = torch.tanh(self.conv_x_h(x_bnc, supports_bsn) + self.conv_h_h(r * h_prev_bnc, supports_bsn))
        return (1.0 - z) * h_prev_bnc + z * h_tilde


# ===================== Encoder / Decoder (time-varying supports) =====================

class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, k_hop: int=2, dropout: float=0.0):
        super(DCRNNEncoder, self).__init__()
        self.layers = nn.ModuleList([
            DCGRUCell(input_dim if l==0 else hidden_dim, hidden_dim, k_hop) for l in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()

    def forward(self, x_btnc: torch.Tensor, supports_seq: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        x_btnc: [B,T_in,N,C_in]
        supports_seq: list length T_in; each [B,S,N,N]
        """
        B,T,N,C = x_btnc.shape
        h = [torch.zeros(B, N, self.layers[l].hidden_dim, device=x_btnc.device) for l in range(len(self.layers))]
        for t in range(T):
            sup_t = supports_seq[t]  # [B,S,N,N]
            for l, cell in enumerate(self.layers):
                inp = x_btnc[:, t] if l==0 else h[l-1]
                h[l] = cell(inp, h[l], sup_t)
                h[l] = self.drop(h[l])
        return h  # list of [B,N,H]


class DCRNNDecoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, num_layers: int, k_hop: int=2, dropout: float=0.0):
        super(DCRNNDecoder, self).__init__()
        self.layers = nn.ModuleList([
            DCGRUCell(output_dim if l==0 else hidden_dim, hidden_dim, k_hop) for l in range(num_layers)
        ])
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()

    def forward(self, T_out: int, h0: List[torch.Tensor],
                supports_seq: List[torch.Tensor],
                y_teacher_btnc: Optional[torch.Tensor]=None,
                teacher_forcing_ratio: float=0.0) -> torch.Tensor:
        """
        supports_seq: list length T_out; each [B,S,N,N] (may be persistence/future)
        y_teacher_btnc: [B,T_out,N,C_out] or None
        """
        B,N,H = h0[-1].shape
        device = h0[-1].device
        h = [hi.clone() for hi in h0]
        y_t = torch.zeros(B, N, 1, device=device)  # GO token (output_dim=1)

        preds = []
        for t in range(T_out):
            sup_t = supports_seq[t]
            for l, cell in enumerate(self.layers):
                inp = y_t if l==0 else h[l-1]
                h[l] = cell(inp, h[l], sup_t)
                h[l] = self.drop(h[l])
            y_hat = self.proj(h[-1])  # [B,N,1]
            preds.append(y_hat)
            use_teacher = (y_teacher_btnc is not None) and (random.random() < teacher_forcing_ratio)
            y_t = y_teacher_btnc[:, t] if use_teacher else y_hat.detach()
        return torch.stack(preds, dim=1)  # [B,T_out,N,1]


class DCRNNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 enc_layers: int, dec_layers: int, k_hop: int=2, dropout: float=0.0):
        super(DCRNNModel, self).__init__()
        self.encoder = DCRNNEncoder(input_dim, hidden_dim, enc_layers, k_hop, dropout)
        self.decoder = DCRNNDecoder(output_dim, hidden_dim, dec_layers, k_hop, dropout)

    def forward(self, x_hist_btnc: torch.Tensor, supports_hist_seq: List[torch.Tensor],
                T_out: int, supports_fut_seq: List[torch.Tensor],
                y_teacher_btnc: Optional[torch.Tensor]=None, teacher_forcing_ratio: float=0.0) -> torch.Tensor:

        h = self.encoder(x_hist_btnc, supports_hist_seq)
        preds = self.decoder(T_out, h, supports_fut_seq, y_teacher_btnc, teacher_forcing_ratio)
        return preds  # [B,K,N,1]