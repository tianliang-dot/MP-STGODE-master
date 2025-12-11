import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):

    def __init__(self, feature_dim, temporal_dim, num_nodes, int_init):
        super(ODEFunc, self).__init__()
        self.inc_init = int_init
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(num_nodes))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)


    def forward(self, x, adj):
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        incidence = torch.softmax(self.inc_init, dim=1)
        mob = torch.einsum('kl,blnmc->bknmc', incidence, adj).squeeze(-1)
        g = mob.mean(1)
        g_dyn = g / g.sum(2, True)
        xa = torch.einsum('ncv, ncwl->nvwl', g_dyn, x)  # (batch_size, num_nodes, length, features)

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)  # (batch_size, num_nodes, length, features)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)  # (batch_size, num_nodes, length, features)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f


class ODEblock(nn.Module):
    # def __init__(self, odefunc, t=torch.tensor([0,1])):
    #     super(ODEblock, self).__init__()
    #     self.t = t
    #     self.odefunc = odefunc

    def __init__(self, odefunc, t):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x, adj):
        t = self.t.type_as(x)  # 将self.t的类型转换为与 x 相同的类型
        # 在这里将adj传递给odefunc
        ode_func_with_adj = lambda t, x: self.odefunc(x, adj)
        z = odeint(ode_func_with_adj, x, t, method='euler')[-1]  # 取求解结果的第二个元素，通常是最终的解

        return z


# Define the ODEGCN model.
class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, num_nodes, int_init, time):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, num_nodes, int_init), t=time)
        # self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, num_nodes, int_init), t=torch.tensor([0, time]))

    def forward(self, x, adj):
        self.odeblock.set_x0(x)  # (batch_size, num_nodes, length, hidden_size【32)
        z = self.odeblock(x, adj)
        return F.relu(z)
