import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def _sym_norm_laplacian_bt(A_btnn: torch.Tensor, eps: float = 1e-8):
    """
    A_btnn: [B,H,N,N]（可能是有向、有权）
    返回 L_tilde: [B,H,N,N]，其中 L_tilde ≈ L - I，谱范围约 [-1,1]
    做法：A_t 对称化 + 加自环 → 对称归一化 → L = I - D^-1/2 A D^-1/2 → L_tilde = L - I
    """
    B,H,N,_ = A_btnn.shape
    # 对称化 + 自环
    A = 0.5 * (A_btnn + A_btnn.transpose(-1,-2))
    I = torch.eye(N, device=A.device, dtype=A.dtype).view(1,1,N,N)
    A_hat = A + I
    # 对称归一化
    d = A_hat.sum(-1)                          # [B,H,N]
    d_inv_sqrt = (d + eps).pow(-0.5)           # [B,H,N]
    D_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # [B,H,N,N]
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt   # [B,H,N,N]
    L = I.expand(B,H,N,N) - A_norm             # 规范拉普拉斯
    L_tilde = L - I                            # 约束谱到 [-1,1]（λ_max≈2 时成立）
    return L_tilde

def build_cheb_polynomials_bt(A_btnn: torch.Tensor, Ks: int):
    """
    批量-时刻版 Chebyshev 多项式。
    输入：A_btnn [B,H,N,N]
    返回：list 长度 Ks，每个元素形状 [B,H,N,N]，对应 T0..T{Ks-1}
    """
    assert Ks >= 1
    B,H,N,_ = A_btnn.shape
    L_tilde = _sym_norm_laplacian_bt(A_btnn)     # [B,H,N,N]
    T_k = []
    T0 = torch.eye(N, device=A_btnn.device, dtype=A_btnn.dtype).view(1,1,N,N).expand(B,H,N,N).contiguous()
    T_k.append(T0)
    if Ks >= 2:
        T1 = L_tilde
        T_k.append(T1)
        for _ in range(2, Ks):
            T_k.append(2.0 * L_tilde @ T_k[-1] - T_k[-2])   # 递推
    return T_k  # list[Ks] of [B,H,N,N]


# ---------- 图工具：对称化 + Chebyshev 多项式 ----------
def symmetrize_aggregate(A_all: np.ndarray, agg: str = "mean", eps: float = 1e-8) -> torch.Tensor:
    """
    A_all: [T, N, N] 时变有向加权邻接（人口流 / 接触）
    先时间聚合 (mean/median/max)，再对称化：A_sym = 0.5*(A + A^T)
    返回 torch.Tensor [N,N]（float32）
    """
    assert A_all.ndim == 3, "A_all must be [T,N,N]"
    if agg == "median":
        A = np.median(A_all, axis=0)
    elif agg == "max":
        A = A_all.max(axis=0)
    else:
        A = A_all.mean(axis=0)
    A = 0.5 * (A + A.T)
    # 非负&去 NaN/Inf
    A = np.clip(A, 0.0, np.inf)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(A, dtype=torch.float32)


def normalize_adj_sym(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # A: [N,N]，自动加自环
    N = A.size(0)
    A_hat = A + torch.eye(N, device=A.device, dtype=A.dtype)
    d = A_hat.sum(dim=1)
    d_inv_sqrt = (d + eps).pow(-0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt  # D^{-1/2} A D^{-1/2}

def scaled_laplacian(A: torch.Tensor) -> torch.Tensor:
    # 归一化拉普拉斯 L = I - D^{-1/2} A D^{-1/2}，再缩放到 [-1,1]
    N = A.size(0)
    A_norm = normalize_adj_sym(A)
    L = torch.eye(N, device=A.device, dtype=A.dtype) - A_norm
    # 对归一化拉普拉斯，lambda_max ∈ [0,2]，近似取 2
    L_tilde = 2.0 * L / 2.0 - torch.eye(N, device=A.device, dtype=A.dtype)
    return L_tilde

def build_cheb_polynomials(A: torch.Tensor, Ks: int):
    """
    返回 [Ks] 个支撑矩阵 T_k(L~)：T0=I, T1=L~, T_k = 2 L~ T_{k-1} - T_{k-2}
    """
    assert Ks >= 1
    N = A.size(0)
    L_tilde = scaled_laplacian(A)
    T_k = []
    T0 = torch.eye(N, device=A.device, dtype=A.dtype)
    T_k.append(T0)
    if Ks >= 2:
        T1 = L_tilde
        T_k.append(T1)
        for k in range(2, Ks):
            T_k.append(2.0 * L_tilde @ T_k[-1] - T_k[-2])
    return T_k  # list of [N,N]

# ---------- Temporal GLU ----------
class TemporalConvGLU(nn.Module):
    def __init__(self, c_in, c_out, Kt=3, dropout=0.1):
        super(TemporalConvGLU, self).__init__()
        pad = Kt // 2  # same padding on time
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(Kt, 1), padding=(pad, 0))
        self.ln = nn.LayerNorm(2 * c_out)
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):  # x: [B, C, T, N]
        h = self.conv(x)  # [B, 2*C_out, T, N]
        h = h.permute(0, 2, 3, 1)  # [B, T, N, 2*C_out]
        h = self.ln(h)
        h = h.permute(0, 3, 1, 2).contiguous()  # [B, 2*C_out, T, N]
        P, Q = torch.chunk(h, 2, dim=1)
        y = torch.tanh(P) * torch.sigmoid(Q)  # GLU
        return self.drop(y)

# ---------- Chebyshev 图卷积（无时间循环，向量化实现） ----------
class ChebGraphConv(nn.Module):
    """
    实现  sum_k  T_k(L~) X Θ_k
    把每个支持矩阵的聚合结果在通道维拼接，再用 1×1 卷积实现通道线性组合。
    """
    def __init__(self, c_in, c_out, Ks):
        super(ChebGraphConv, self).__init__()
        self.Ks = Ks
        self.theta = nn.Conv2d(c_in * Ks, c_out, kernel_size=(1, 1), bias=True)
        # self.bn = nn.BatchNorm2d(c_out)
        self.ln = nn.LayerNorm(c_out)

    # def forward(self, x, cheb_polys):
    #     # x: [B, C_in, T, N], cheb_polys: list[Ks] of [N, N]
    #     B, C, T, N = x.shape
    #     supports_out = []
    #     # 用 einsum 在“节点维”做矩阵乘法： (N,N) × (B,C,T,N) -> (B,C,T,N)
    #     for Tk in cheb_polys:
    #         supports_out.append(torch.einsum('ij,bctj->bcti', Tk, x))
    #     x_cat = torch.cat(supports_out, dim=1)        # [B, C*Ks, T, N]
    #     x_lin = self.theta(x_cat)                      # [B, C_out, T, N]
    #
    #     y = x_lin.permute(0, 2, 3, 1)  # [B, T, N, C_out]
    #     y = self.ln(y)  # 按最后一维 C_out 归一化
    #     y = y.permute(0, 3, 1, 2).contiguous()  # [B, C_out, T, N]
    #     return y

    def forward(self, x, A_hist):
        # 构建每个时刻的 Cheb 支持：list[Ks] of [B,H,N,N]
        polys = build_cheb_polynomials_bt(A_hist, self.Ks)
        # 对每个阶 k： (B,H,N,N) × (B,C,H,N) → (B,C,H,N)
        outs = [torch.einsum('bhnm,bchm->bchn', Tk, x) for Tk in polys]
        x_cat = torch.cat(outs, dim=1)  # [B, C_in*Ks, H, N]
        x_lin = self.theta(x_cat)  # [B, C_out, H, N]
        # LN on channel dim: [B,C,H,N] -> [B,H,N,C] -> LN -> back
        y = x_lin.permute(0, 2, 3, 1)  # [B, H, N, C_out]
        y = self.ln(y)
        y = y.permute(0, 3, 1, 2).contiguous()  # [B, C_out, H, N]
        return y

# ---------- ST-Conv Block： Temporal(GLU) → Spatial(Cheb) → Temporal(GLU) + Res ----------
class STConvBlock(nn.Module):
    def __init__(self, c_in, c_hidden, Kt=3, Ks=3, dropout=0.1):
        super(STConvBlock, self).__init__()
        self.t1 = TemporalConvGLU(c_in, c_hidden, Kt, dropout=dropout)
        self.s  = ChebGraphConv(c_hidden, c_hidden, Ks)
        self.t2 = TemporalConvGLU(c_hidden, c_hidden, Kt, dropout=dropout)
        self.res = nn.Conv2d(c_in, c_hidden, kernel_size=1) if c_in != c_hidden else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x, A_hist):  # A_hist: [B,H,N,N]
        y = self.t1(x)
        y = self.s(y, A_hist)  # ← 这里传时变图
        y = self.t2(y)
        y = y + self.res(x)
        return self.act(y)

    # def forward(self, x, cheb_polys):
    #     # x: [B, C_in, T, N]
    #     y = self.t1(x.permute(0, 3, 1, 2))                      # 时域门控 （B, W, N, F）
    #     y = self.s(y, cheb_polys)           # 空间图卷积
    #     y = self.t2(y)                      # 再一次时域门控
    #     y = y + self.res(x.permute(0, 3, 1, 2))                 # 残差
    #     return self.act(y)

# ---------- Baseline STGCN（两层 ST-Conv + 输出层） ----------
class BaselineSTGCN(nn.Module):
    """
    输入:  x: [B, C_in(=1或多特征), T=H, N]
    输出:  y_hat: [B, K, N]  （多步预测）
    """
    def __init__(self, A: torch.Tensor, C_in: int, C_hidden: int, H: int, K: int,
                 Kt: int = 3, Ks: int = 3, dropout: float = 0.1):
        super(BaselineSTGCN, self).__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.H, self.K = H, K
        # 预计算 Chebyshev 支持（静态图）
        polys = build_cheb_polynomials(A, Ks)
        # 注册为 buffer（随模型走，不算参数）
        for i, Tk in enumerate(polys):
            self.register_buffer(f"Tk_{i}", Tk)
        self.Ks = Ks

        self.block1 = STConvBlock(C_in, C_hidden, Kt=Kt, Ks=Ks, dropout=dropout)
        self.block2 = STConvBlock(C_hidden, C_hidden, Kt=Kt, Ks=Ks, dropout=dropout)

        # 输出层：先保留时间长度（same padding 已保证），再用 (H,1) 卷积聚合时间到 1，
        # 最后通道数设为 K，实现多步输出
        self.out_time = nn.Conv2d(C_hidden, C_hidden, kernel_size=(H, 1))
        self.out_proj = nn.Conv2d(C_hidden, K, kernel_size=(1, 1))

    def _get_polys(self):
        return [getattr(self, f"Tk_{i}") for i in range(self.Ks)]

    def forward(self, x, A_hist):  # x: [B, C_in, H, N]
        # polys = self._get_polys()
        y = self.block1(x, A_hist)
        y = self.block2(y, A_hist)
        y = self.out_time(y)                # [B, C_hidden, 1, N]
        y = self.out_proj(y)                # [B, K, 1, N]
        return y.squeeze(2)                 # [B, K, N]
