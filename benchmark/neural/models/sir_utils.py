# sir_utils.py
import torch

def active_I_from_newI(newI_hist_raw: torch.Tensor, gamma: float = 1/7) -> torch.Tensor:
    """
    newI_hist_raw: [B,H,N] 原始尺度
    返回 I_hist:    [B,H,N]  逐日活跃感染近似（指数留存）
    """
    B, H, N = newI_hist_raw.shape
    k = torch.arange(H, device=newI_hist_raw.device).float()  # 0..H-1
    w = (1.0 - gamma) ** k                                    # [H]
    # 卷积式：每个 t 的 I_t = sum_{j=0..t} newI_{t-j} * w[j]
    # 用逐日循环更直观（H=7 很小，开销可忽略）
    I_hist = []
    for t in range(H):
        kk = torch.arange(t+1, device=newI_hist_raw.device).float()
        wt = (1.0 - gamma) ** kk                              # [t+1]
        contrib = newI_hist_raw[:, :t+1, :] * wt.flip(0).view(1,-1,1)
        I_hist.append(contrib.sum(dim=1))
    return torch.stack(I_hist, dim=1)                         # [B,H,N]

def build_SIR_hist_from_feats(x_raw_btnc: torch.Tensor, feat_idx: dict, gamma: float = 1/7):
    """
    x_raw_btnc: [B,H,N,C=4] 原始尺度
    feat_idx:   {"cum":0, "death":1, "S":2, "newI":3}
    返回 sir_hist_raw: [B,H,N,3]  (S,I,R)
    """
    cum   = x_raw_btnc[..., feat_idx["cum"]]    # [B,H,N]
    death = x_raw_btnc[..., feat_idx["death"]]  # [B,H,N]
    S     = x_raw_btnc[..., feat_idx["S"]]      # [B,H,N]
    newI  = x_raw_btnc[..., feat_idx["newI"]]   # [B,H,N]

    I_hist = active_I_from_newI(newI, gamma=gamma)            # [B,H,N]
    R_hist = (cum - death - I_hist).clamp_min(0.0)            # [B,H,N]
    sir = torch.stack([S, I_hist, R_hist], dim=-1)            # [B,H,N,3]  (S,I,R)
    return sir


