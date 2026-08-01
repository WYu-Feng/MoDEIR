from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .bootstrap import PROJECT_ROOT, bootstrap_legacy_backend
from .checkpoints import load_compatible_state, load_decoder_refine, load_payload, load_state
from .daclip_encoder import DACLIPDegEncoder, DEFAULT_DEGRADATION_LABELS
from .injection import StaticUNetFeatureAdapter, migrate_legacy_injectors
from .lora import cache_expert_layers, inject_encoder_lora, inject_unet_expert_lora, set_active_expert
from .losses import NLayerPatchDiscriminator
from .router import TARRouter
from .tscm import TSCM


bootstrap_legacy_backend()
from ldm.util import instantiate_from_config  # noqa: E402


def freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def disable_unet_checkpointing(unet: nn.Module) -> None:
    for module in unet.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False


def patch_ldm_checkpoint_direct() -> None:
    """Avoid the legacy custom checkpoint backward on frozen backbone inputs."""
    import ldm.modules.diffusionmodules.util as diffusion_util

    def direct_forward(function, inputs, params, flag):
        return function(*inputs)

    diffusion_util.checkpoint = direct_forward


def load_ldm(config_path: str | Path, checkpoint_path: str | Path, device: torch.device):
    config = OmegaConf.load(str(config_path))
    patch_ldm_checkpoint_direct()
    previous_cwd = Path.cwd()
    try:
        # The checked-in OpenCLIP embedder resolves its local safetensors file
        # relative to the original project root during construction.
        os.chdir(PROJECT_ROOT)
        ldm = instantiate_from_config(config.model)
    finally:
        os.chdir(previous_cwd)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    missing, unexpected = ldm.load_state_dict(state, strict=False)
    print(f"[LOAD] base LDM: missing={len(missing)} unexpected={len(unexpected)}")
    ldm.eval().to(device)
    freeze(ldm)
    disable_unet_checkpointing(ldm.model.diffusion_model)
    return ldm


def _as_hw(image_size) -> tuple[int, int]:
    if isinstance(image_size, (tuple, list)):
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def encoder_pyramid(vae, x: torch.Tensor, scale_factor: float, image_size: int | tuple[int, int]):
    encoder = vae.encoder
    if not hasattr(encoder, "cfw_feature_context"):
        posterior = vae.encode(x)
        return posterior.mode().float() * scale_factor, []
    image_h, image_w = _as_hw(image_size)
    sizes = [(max(1, image_h // div), max(1, image_w // div)) for div in (8, 4, 2, 1)]
    with encoder.cfw_feature_context(max_feats=4, skip_highest_hw=False, detach_feats=True, clear_on_exit=True):
        posterior = vae.encode(x)
        pyramid = encoder.get_cfw_pyramid()
    by_size = {
        (int(feat.shape[-2]), int(feat.shape[-1])): feat
        for feat in pyramid
        if torch.is_tensor(feat) and feat.ndim == 4
    }
    available = sorted(by_size)
    if not available:
        return posterior.mode().float() * scale_factor, []
    features = []
    for size in sizes:
        nearest = min(available, key=lambda candidate: abs(candidate[0] - size[0]) + abs(candidate[1] - size[1]))
        feat = by_size[nearest]
        if feat.shape[-2:] != size:
            feat = F.interpolate(feat, size, mode="bilinear", align_corners=False)
        features.append(feat)
    return posterior.mode().float() * scale_factor, features


def unet_to_z0(out: torch.Tensor, z_t: torch.Tensor, t_int: torch.Tensor, alpha: torch.Tensor, parameterization: str):
    a = alpha[t_int].view(-1, 1, 1, 1).to(out).clamp(1e-12, 1)
    if parameterization in {"eps", "epsilon"}:
        return (z_t - torch.sqrt(1 - a) * out) / torch.sqrt(a)
    if parameterization in {"v", "velocity"}:
        return torch.sqrt(a) * z_t - torch.sqrt(1 - a) * out
    raise ValueError(f"Unknown diffusion parameterization: {parameterization}")


class LatentFusionResBlock(nn.Module):
    """Post-routing latent fusion block used after weighted expert aggregation."""

    def __init__(self, channels: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, 1, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


@dataclass
class RefinedMoDEIR:
    ldm: nn.Module
    vae: nn.Module
    unet: nn.Module
    adapter: StaticUNetFeatureAdapter
    tscm: TSCM
    router: TARRouter
    latent_fusion: LatentFusionResBlock
    critic: NLayerPatchDiscriminator
    expert_layers: list[nn.Module]
    t_list: list[int]
    scale_factor: float
    parameterization: str
    critic_in_channels: int
    device: torch.device

    def autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def conditioning(self, batch_size: int):
        return self.ldm.get_learned_conditioning([""] * batch_size)

    def encode_degraded(self, x: torch.Tensor, image_size: int | tuple[int, int]):
        return encoder_pyramid(self.vae, x, self.scale_factor, image_size)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor):
        return self.vae.encode(x).mode().float() * self.scale_factor

    def decode(self, z: torch.Tensor, enc_feats: list[torch.Tensor], t_val: Optional[torch.Tensor] = None):
        return self.vae.decode(z.float() / self.scale_factor, enc_feats=enc_feats, t_val=t_val).float()

    def _run_group(self, x: torch.Tensor, z_base: torch.Tensor, ctx: torch.Tensor, expert_id: int, latent_mode: str):
        ids = torch.full((x.shape[0],), expert_id, device=x.device, dtype=torch.long)
        z_unet, feats, aux = self.tscm(z_base, x, ids, latent_mode=latent_mode)
        set_active_expert(self.expert_layers, expert_id)
        with self.autocast():
            out = self.adapter(z_unet, timesteps=aux["t_int"], context=ctx, tscm_feats=feats)
        z0 = unet_to_z0(out.float(), z_unet.float(), aux["t_int"], self.tscm.alphas_cumprod, self.parameterization)
        return z0, aux

    def restore_selected(self, x: torch.Tensor, z_base: torch.Tensor, ctx: torch.Tensor, selected: torch.Tensor, latent_mode: str):
        result = torch.empty_like(z_base)
        t_values = torch.empty(x.shape[0], device=x.device, dtype=torch.long)
        for expert_tensor in selected.unique(sorted=True):
            expert_id = int(expert_tensor.item())
            indices = torch.nonzero(selected == expert_id, as_tuple=False).squeeze(1)
            z0, aux = self._run_group(x.index_select(0, indices), z_base.index_select(0, indices), ctx.index_select(0, indices), expert_id, latent_mode)
            result.index_copy_(0, indices, z0)
            t_values.index_copy_(0, indices, aux["t_int"])
        return result, t_values

    def restore_weighted(
        self,
        x: torch.Tensor,
        z_base: torch.Tensor,
        ctx: torch.Tensor,
        selected: torch.Tensor,
        weights: torch.Tensor,
        latent_mode: str,
    ):
        if selected.ndim != 2 or weights.ndim != 2:
            raise ValueError("selected and weights must be [batch, top_s] tensors")
        result = torch.zeros_like(z_base)
        max_t_values = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        weights = weights.to(device=x.device, dtype=z_base.dtype)
        for slot in range(selected.shape[1]):
            z0, t_val = self.restore_selected(x, z_base, ctx, selected[:, slot].long(), latent_mode)
            w = weights[:, slot].view(-1, 1, 1, 1)
            result = result + z0 * w
            max_t_values = torch.maximum(max_t_values, t_val.float())
        return self.latent_fusion(result), max_t_values

    @torch.no_grad()
    def oracle_targets(self, x: torch.Tensor, z_base: torch.Tensor, z_gt: torch.Tensor, ctx: torch.Tensor, latent_mode: str):
        scores = []
        for expert_id in range(len(self.t_list)):
            z0, _ = self._run_group(x, z_base, ctx, expert_id, latent_mode)
            scores.append((z0 - z_gt).abs().flatten(1).mean(1))
        return torch.stack(scores, dim=1).argmin(dim=1)


def build_refined_model(
    *,
    config_path: str | Path,
    base_checkpoint: str | Path,
    stage1_checkpoint: str | Path,
    router_checkpoint: str | Path,
    stage2_checkpoint: str | Path,
    device: torch.device,
    preferred_latent_hw: int = 32,
    window: int = 8,
    router_backbone: str = "daclip",
    daclip_checkpoint: str | Path | None = None,
    top_s: int = 2,
) -> RefinedMoDEIR:
    stage1 = load_payload(stage1_checkpoint, mmap=True)
    router_payload = load_payload(router_checkpoint)
    stage2 = load_payload(stage2_checkpoint)
    t_list = [int(value) for value in stage1["t_list"]]
    legacy_args = stage1.get("args", {}) if isinstance(stage1.get("args"), dict) else {}
    ldm = load_ldm(config_path, base_checkpoint, device)
    vae, unet = ldm.first_stage_model, ldm.model.diffusion_model
    scale_factor = float(getattr(ldm, "scale_factor", stage1.get("scale_factor", 0.18215)))
    parameterization = str(getattr(ldm, "parameterization", stage1.get("parameterization", "eps"))).lower()

    inject_encoder_lora(
        vae.encoder,
        r_lin=int(legacy_args.get("lora_lin_r", 8)),
        a_lin=int(legacy_args.get("lora_lin_alpha", 8)),
        r_conv=int(legacy_args.get("lora_conv_r", 8)),
        a_conv=int(legacy_args.get("lora_conv_alpha", 8)),
    )
    load_state(vae.encoder, stage1["lora_vae_encoder"], "VAE encoder LoRA", strict=False)
    freeze(vae.encoder)

    ch1 = int(legacy_args.get("tscm_ch1", 320))
    ch2 = int(legacy_args.get("tscm_ch2", 640))
    ch3 = int(legacy_args.get("tscm_ch3", 1280))
    tscm = TSCM(
        t_list,
        ldm.alphas_cumprod,
        t_dim=int(legacy_args.get("tscm_t_dim", 128)),
        struct_base_ch=int(legacy_args.get("tscm_struct_base_ch", 64)),
        ch1=ch1,
        ch2=ch2,
        ch3=ch3,
        window=window,
    ).to(device)
    load_state(tscm, stage1["tscm"], "TSCM", strict=True)

    inject_unet_expert_lora(
        unet,
        len(t_list),
        r=int(legacy_args.get("lora_r", 8)),
        alpha=int(legacy_args.get("lora_alpha", 16)),
    )
    legacy_unet = stage1["unet_state"]
    stable_unet = {key: value for key, value in legacy_unet.items() if not key.startswith("tscm_inject_blocks.")}
    _, unexpected = load_state(unet, stable_unet, "UNet expert LoRA", strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected UNet expert keys: {unexpected[:5]}")
    adapter = StaticUNetFeatureAdapter(unet, window=window, specs=((320, ch1), (320, ch2), (640, ch3))).to(device)
    for line in migrate_legacy_injectors(adapter, legacy_unet, preferred_latent_hw=preferred_latent_hw):
        print(f"[MIGRATE] {line}")
    expert_layers = cache_expert_layers(unet)
    print(f"[INFO] expert LoRA layers={len(expert_layers)}")

    router_backbone = str(router_backbone).lower()
    if router_backbone == "daclip":
        daclip_checkpoint = daclip_checkpoint or (PROJECT_ROOT / "pretrained" / "daclip_ViT-B-32.pt")
        deg_encoder = DACLIPDegEncoder(daclip_checkpoint)
        router = TARRouter(
            len(t_list),
            deg_feat_dim=512,
            deg_prob_dim=len(DEFAULT_DEGRADATION_LABELS),
            proj_dim=512,
            top_s=top_s,
            encoder=deg_encoder,
        ).to(device)
        load_compatible_state(router, router_payload["router"], "TAR router", skip_prefixes=("encoder.",))
    elif router_backbone == "simple":
        router = TARRouter(len(t_list), deg_feat_dim=512, deg_prob_dim=16, proj_dim=256, top_s=top_s).to(device)
        load_compatible_state(router, router_payload["router"], "TAR router")
    else:
        raise ValueError(f"Unknown router_backbone: {router_backbone}")
    latent_fusion = LatentFusionResBlock(channels=4, hidden=32).to(device)
    load_decoder_refine(vae.decoder, stage2["decoder_refine"])

    first_disc_weight = next(value for key, value in stage2["disc"].items() if key.endswith("weight_orig"))
    critic_in_channels = int(first_disc_weight.shape[1])
    critic = NLayerPatchDiscriminator(in_ch=critic_in_channels).to(device)
    load_state(critic, stage2["disc"], "fixed Stage 2 critic", strict=True)
    freeze(critic)
    critic.eval()
    return RefinedMoDEIR(
        ldm, vae, unet, adapter, tscm, router, latent_fusion, critic, expert_layers,
        t_list, scale_factor, parameterization, critic_in_channels, device,
    )
