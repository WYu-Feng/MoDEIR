from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import window_sdpa


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ZeroConv1x1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WindowCrossAttention2d(nn.Module):
    def __init__(self, q_ch: int, kv_ch: int, heads: int = 4, dim_head: int = 32, window: int = 8):
        super().__init__()
        self.heads, self.dim_head, self.window = int(heads), int(dim_head), int(window)
        inner = heads * dim_head
        self.norm_q = nn.GroupNorm(_num_groups(q_ch), q_ch)
        self.norm_kv = nn.GroupNorm(_num_groups(kv_ch), kv_ch)
        self.to_q = nn.Conv2d(q_ch, inner, 1)
        self.to_k = nn.Conv2d(kv_ch, inner, 1)
        self.to_v = nn.Conv2d(kv_ch, inner, 1)
        self.proj = nn.Conv2d(inner, q_ch, 1)

    def forward(self, feat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if delta.shape[-2:] != feat.shape[-2:]:
            delta = F.interpolate(delta, feat.shape[-2:], mode="bilinear", align_corners=False)
        kv = self.norm_kv(delta)
        out = window_sdpa(
            self.to_q(self.norm_q(feat).float()), self.to_k(kv.float()), self.to_v(kv.float()),
            heads=self.heads, dim_head=self.dim_head, window=self.window,
        )
        return self.proj(out)


class WindowSelfAttention2d(nn.Module):
    def __init__(self, ch: int, heads: int = 4, dim_head: int = 32, window: int = 8):
        super().__init__()
        self.heads, self.dim_head, self.window = int(heads), int(dim_head), int(window)
        inner = heads * dim_head
        self.norm = nn.GroupNorm(_num_groups(ch), ch)
        self.to_q = nn.Conv2d(ch, inner, 1)
        self.to_k = nn.Conv2d(ch, inner, 1)
        self.to_v = nn.Conv2d(ch, inner, 1)
        self.proj = nn.Conv2d(inner, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out = window_sdpa(
            self.to_q(h.float()), self.to_k(h.float()), self.to_v(h.float()),
            heads=self.heads, dim_head=self.dim_head, window=self.window,
        )
        return self.proj(out)


class FeatureCompensationBlock(nn.Module):
    def __init__(self, feat_ch: int, delta_ch: int, window: int = 8):
        super().__init__()
        self.cross = WindowCrossAttention2d(feat_ch, delta_ch, window=window)
        self.self_attn = WindowSelfAttention2d(feat_ch, window=window)
        self.zero = ZeroConv1x1(feat_ch, feat_ch)

    def forward(self, feat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return feat + self.zero(self.self_attn(self.cross(feat, delta)))


class StaticUNetFeatureAdapter(nn.Module):
    """UNet wrapper with three fixed, resolution-independent TSCM injectors."""

    DEFAULT_SPECS = ((320, 320), (320, 640), (640, 1280))

    def __init__(self, unet: nn.Module, window: int = 8, specs=None):
        super().__init__()
        self.unet = unet
        self.specs = tuple(specs or self.DEFAULT_SPECS)
        self.blocks = nn.ModuleList([FeatureCompensationBlock(*spec, window=window) for spec in self.specs])
        self.last_stats: dict[str, Any] = {}

    def forward(self, x, timesteps=None, context=None, y=None, *, tscm_feats=None, **kwargs):
        from ldm.modules.diffusionmodules.openaimodel import timestep_embedding

        if tscm_feats is None:
            return self.unet(x, timesteps=timesteps, context=context, y=y, **kwargs)
        hs, slot = [], 0
        emb = self.unet.time_embed(timestep_embedding(timesteps, self.unet.model_channels, repeat_only=False))
        if self.unet.num_classes is not None:
            emb = emb + self.unet.label_emb(y)
        h = x.type(self.unet.dtype)
        injected = []
        for module in self.unet.input_blocks:
            h = module(h, emb, context)
            if slot < len(self.blocks):
                expected_ch, _ = self.specs[slot]
                delta = tscm_feats[slot]
                if h.shape[1] == expected_ch and h.shape[-2:] == delta.shape[-2:]:
                    h = self.blocks[slot](h.float(), delta.float()).to(h.dtype)
                    injected.append((slot, tuple(h.shape)))
                    slot += 1
            hs.append(h)
        h = self.unet.middle_block(h, emb, context)
        for module in self.unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        self.last_stats = {"injected": injected, "count": slot}
        return self.unet.id_predictor(h) if self.unet.predict_codebook_ids else self.unet.out(h)


def migrate_legacy_injectors(
    adapter: StaticUNetFeatureAdapter,
    legacy_unet_state: dict[str, torch.Tensor],
    *,
    preferred_latent_hw: int,
) -> list[str]:
    """Map H-dependent legacy injection blocks to fixed channel-signature blocks."""
    grouped: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    prefix = "tscm_inject_blocks."
    for key in legacy_unet_state:
        if not key.startswith(prefix):
            continue
        block_name = key[len(prefix):].split(".", 1)[0]
        h, feat_ch, delta_ch = (int(part) for part in block_name.split("_"))
        grouped[(feat_ch, delta_ch)].append((h, block_name))

    chosen = []
    for block, signature in zip(adapter.blocks, adapter.specs):
        candidates = grouped.get(signature, [])
        if not candidates:
            raise RuntimeError(f"No legacy TSCM injector found for signature={signature}")
        h, block_name = min(candidates, key=lambda item: abs(item[0] - preferred_latent_hw))
        source_prefix = f"{prefix}{block_name}."
        state = {key[len(source_prefix):]: value for key, value in legacy_unet_state.items() if key.startswith(source_prefix)}
        block.load_state_dict(state, strict=True)
        chosen.append(f"slot={len(chosen)} <- {block_name} (H={h}, signature={signature})")
        preferred_latent_hw //= 2
    return chosen
