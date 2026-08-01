from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import window_sdpa


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / max(half, 1))
    emb = torch.cat([torch.cos(t.float()[:, None] * freqs), torch.sin(t.float()[:, None] * freqs)], dim=-1)
    return torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1) if dim % 2 else emb


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, act: str = "gelu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.norm = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.act = nn.GELU() if act == "gelu" else nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LightResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(ch), ch)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(_num_groups(ch), ch)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act2(self.norm2(self.conv1(self.act1(self.norm1(x))))))


class SobelMag(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1, keepdim=True) if x.shape[1] == 3 else x
        gx = F.conv2d(x, self.kx.to(x), padding=1)
        gy = F.conv2d(x, self.ky.to(x), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)


class LocalVariance(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.ks = int(kernel_size)
        self.pad = self.ks // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(x, self.ks, 1, self.pad)
        return (F.avg_pool2d(x * x, self.ks, 1, self.pad) - mean * mean).clamp_min(0)


class TimestepFiLM(nn.Module):
    def __init__(self, feat_ch: int, t_dim: int = 128):
        super().__init__()
        self.t_dim = int(t_dim)
        self.mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.GELU(), nn.Linear(t_dim, feat_ch * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor, t_int: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(timestep_embedding(t_int, self.t_dim)).chunk(2, dim=1)
        return feat * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class WindowSelfAttention2d(nn.Module):
    """State-compatible local replacement for the legacy global attention."""

    def __init__(self, ch: int, heads: int = 4, dim_head: int = 32, window: int = 8):
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.window = int(window)
        inner = heads * dim_head
        self.norm = nn.GroupNorm(_num_groups(ch), ch)
        self.to_q = nn.Conv2d(ch, inner, 1)
        self.to_k = nn.Conv2d(ch, inner, 1)
        self.to_v = nn.Conv2d(ch, inner, 1)
        self.proj = nn.Conv2d(inner, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out = window_sdpa(
            self.to_q(h.float()), self.to_k(h.float()), self.to_v(h.float()),
            heads=self.heads, dim_head=self.dim_head, window=self.window,
        )
        return x + self.proj(out).to(x.dtype)


class ProgressiveDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window: int = 8):
        super().__init__()
        self.conv = ConvGNAct(in_ch, out_ch, 3, 2, 1)
        self.rb = LightResBlock(out_ch)
        self.attn = WindowSelfAttention2d(out_ch, 4, 32, window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.rb(self.conv(x)))


class StructureCueExtractor(nn.Module):
    def __init__(self, base_ch: int = 64, t_dim: int = 128, out_ch: int = 320):
        super().__init__()
        self.rgb_stem = nn.Sequential(ConvGNAct(3, base_ch), LightResBlock(base_ch), LightResBlock(base_ch))
        self.sobel = SobelMag()
        self.edge_proj = nn.Sequential(ConvGNAct(1, base_ch), LightResBlock(base_ch))
        self.local_var = LocalVariance(3)
        self.var_proj = nn.Sequential(ConvGNAct(base_ch, base_ch), LightResBlock(base_ch))
        self.fuse = nn.Sequential(ConvGNAct(base_ch * 3, base_ch), LightResBlock(base_ch))
        self.film = TimestepFiLM(base_ch, t_dim)
        self.to_out = nn.Sequential(ConvGNAct(base_ch, out_ch), LightResBlock(out_ch))

    def forward(self, x_deg: torch.Tensor, size_hw: tuple[int, int], t_int: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x_deg, size=size_hw, mode="bilinear", align_corners=False)
        rgb = self.rgb_stem(x)
        feat = self.fuse(torch.cat([rgb, self.edge_proj(self.sobel(x)), self.var_proj(self.local_var(rgb))], dim=1))
        return self.to_out(self.film(feat, t_int))


class LatentPerturbHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 4):
        super().__init__()
        self.net = nn.Sequential(ConvGNAct(in_ch, in_ch), LightResBlock(in_ch), nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class TSCM(nn.Module):
    """Structure compensation pyramid with explicit latent perturbation policy."""

    def __init__(
        self,
        t_list: list[int],
        alphas_cumprod: torch.Tensor,
        *,
        t_dim: int = 128,
        struct_base_ch: int = 64,
        ch1: int = 320,
        ch2: int = 640,
        ch3: int = 1280,
        window: int = 8,
    ):
        super().__init__()
        self.register_buffer("t_list", torch.tensor(t_list, dtype=torch.long))
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.struct_extractor = StructureCueExtractor(struct_base_ch, t_dim, ch1)
        self.delta_z_head = LatentPerturbHead(ch1, 4)
        self.down12 = ProgressiveDown(ch1, ch2, window)
        self.down23 = ProgressiveDown(ch2, ch3, window)

    def _latent_perturb(self, z: torch.Tensor, delta: torch.Tensor, t_int: torch.Tensor, eps: torch.Tensor):
        alpha = self.alphas_cumprod[t_int].view(-1, 1, 1, 1).to(z).clamp(1e-12, 1)
        return torch.sqrt(alpha) * (z + delta) + torch.sqrt((1 - alpha).clamp_min(1e-12)) * eps

    def forward(
        self,
        z_base: torch.Tensor,
        x_deg: torch.Tensor,
        t_idx: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        *,
        latent_mode: str = "strict_delta",
    ):
        t_int = self.t_list[t_idx.long()]
        f1 = self.struct_extractor(x_deg, z_base.shape[-2:], t_int)
        f2 = self.down12(f1)
        f3 = self.down23(f2)
        delta = self.delta_z_head(f1) if latent_mode == "strict_delta" else None
        if latent_mode not in {"legacy_no_delta", "strict_delta"}:
            raise ValueError(f"Unknown latent_mode: {latent_mode}")
        if eps is None:
            eps = torch.randn_like(z_base)
        z_unet = self._latent_perturb(z_base, delta, t_int, eps) if delta is not None else z_base
        return z_unet, [f1, f2, f3], {"t_int": t_int, "delta_z": delta, "f_str": f1}
