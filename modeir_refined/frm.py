from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.norm = _make_gn(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualDenseBlock5C(nn.Module):
    def __init__(self, nf: int, gc: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = float(res_scale)
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + self.res_scale * x5


class RRDB(nn.Module):
    def __init__(self, nf: int, gc: int = 32, rdb_res_scale: float = 0.2, rrdb_res_scale: float = 0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock5C(nf, gc, rdb_res_scale)
        self.rdb2 = ResidualDenseBlock5C(nf, gc, rdb_res_scale)
        self.rdb3 = ResidualDenseBlock5C(nf, gc, rdb_res_scale)
        self.rrdb_res_scale = float(rrdb_res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + self.rrdb_res_scale * (out - x)


class FRMInjectBlock(nn.Module):
    """Decoder feature refinement block compatible with the Stage 2 checkpoint."""

    def __init__(
        self,
        enc_ch: int,
        dec_ch: int,
        rrdb_nf: Optional[int] = None,
        rrdb_gc: int = 32,
        num_rrdb: int = 2,
        use_t_gate: bool = True,
        t_gate_hid: int = 32,
        t_max: float = 1000.0,
        max_gate_scale: float = 0.10,
        init_gate: float = 0.1,
        init_delta_std: float = 1e-4,
        use_window_attn: bool = False,
        attn_heads: int = 4,
        attn_window: int = 8,
    ):
        super().__init__()
        if use_window_attn:
            raise ValueError("Legacy FRM checkpoint uses use_window_attn=False")
        rrdb_nf = int(rrdb_nf) if rrdb_nf is not None else int(dec_ch)
        self.use_t_gate = bool(use_t_gate)
        self.t_max = float(t_max)
        self.max_gate_scale = float(max_gate_scale)
        self.align = nn.Sequential(ConvGNAct(enc_ch, dec_ch), nn.Conv2d(dec_ch, dec_ch, 3, 1, 1))
        self.fuse_in = nn.Sequential(
            ConvGNAct(dec_ch * 2, rrdb_nf),
            nn.Conv2d(rrdb_nf, rrdb_nf, 3, 1, 1),
            _make_gn(rrdb_nf),
            nn.SiLU(),
        )
        self.trunk = nn.ModuleList([RRDB(rrdb_nf, gc=rrdb_gc) for _ in range(int(num_rrdb))])
        self.attn = None
        self.fuse_out = nn.Conv2d(rrdb_nf, dec_ch, 3, 1, 1)
        nn.init.normal_(self.fuse_out.weight, mean=0.0, std=float(init_delta_std))
        nn.init.zeros_(self.fuse_out.bias)
        init_gate = max(-0.95 * self.max_gate_scale, min(0.95 * self.max_gate_scale, float(init_gate)))
        self.raw_w = nn.Parameter(torch.tensor(math.atanh(init_gate / max(self.max_gate_scale, 1e-8))))
        self.t_mlp = nn.Sequential(nn.Linear(1, int(t_gate_hid)), nn.SiLU(), nn.Linear(int(t_gate_hid), 1))
        nn.init.zeros_(self.t_mlp[-1].weight)
        nn.init.zeros_(self.t_mlp[-1].bias)
        self.capture_fdec = False
        self.last_f_dec: Optional[torch.Tensor] = None

    def _gate(self, f_dec: torch.Tensor, t_val: Optional[torch.Tensor]) -> torch.Tensor:
        base = self.max_gate_scale * torch.tanh(self.raw_w).view(1, 1, 1, 1)
        if t_val is None or not self.use_t_gate:
            return base.to(device=f_dec.device, dtype=f_dec.dtype)
        t = torch.as_tensor(t_val, device=f_dec.device).float().view(-1, 1)
        t = (t / self.t_max).clamp(0.0, 1.0)
        return (base * (1.0 + 0.1 * torch.tanh(self.t_mlp(t))).view(-1, 1, 1, 1)).to(f_dec.dtype)

    def forward(self, f_dec: torch.Tensor, f_enc: Optional[torch.Tensor], t_val: Optional[torch.Tensor] = None):
        if self.capture_fdec:
            self.last_f_dec = f_dec.detach().float()
        if f_enc is None:
            return f_dec
        dtype = self.fuse_out.weight.dtype
        dec = f_dec.to(dtype)
        enc = f_enc.to(dtype)
        if enc.shape[-2:] != dec.shape[-2:]:
            enc = F.interpolate(enc, size=dec.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_in(torch.cat([self.align(enc), dec], dim=1))
        for block in self.trunk:
            x = block(x)
        delta = self.fuse_out(x).to(f_dec.dtype)
        return f_dec + delta * self._gate(f_dec, t_val)
