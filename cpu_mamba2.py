"""CPU reference for the exact Mamba2 parameters used by SGMA-Net (inference only)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mamba2_cpu_forward(self, u):
    # Mamba2(32, d_state=32, expand=2, headdim=64, ngroups=1).
    zxbcdt = self.in_proj(u)
    d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm
             - 2 * self.ngroups * self.d_state - self.nheads) // 2
    z0, x0, z, xbc, dt = torch.split(
        zxbcdt, [d_mlp, d_mlp, self.d_ssm,
                 self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads], -1
    )
    xbc = F.silu(self.conv1d(xbc.transpose(1, 2)).transpose(1, 2)[:, :u.shape[1]])
    x, b, c = torch.split(xbc, [self.d_ssm, self.d_state, self.d_state], -1)
    dt = F.softplus(dt + self.dt_bias).clamp(*self.dt_limit)
    a = -torch.exp(self.A_log.float())
    decay = torch.exp(dt * a).unsqueeze(-1)
    state = x.new_zeros(u.shape[0], self.d_ssm, self.d_state)
    values = []
    for index in range(u.shape[1]):
        state = decay[:, index, :, :] * state + (
            dt[:, index, :, None] * x[:, index, :, None] * b[:, index, None, :]
        )
        y = (state * c[:, index, None, :]).sum(-1) + self.D * x[:, index]
        values.append(y)
    y = torch.stack(values, dim=1)
    if self.norm_before_gate:
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.norm.eps)
        y = y * self.norm.weight * F.silu(z)
    else:
        y = y * F.silu(z)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.norm.eps)
        y = y * self.norm.weight
    if d_mlp:
        y = torch.cat((F.silu(z0) * x0, y), dim=-1)
    return self.out_proj(y)
