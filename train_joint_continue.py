#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continue training from the three pretrained checkpoints in ./outputs.

Default joint fine-tuning updates:
  - Stage 1 UNet expert LoRA and TSCM injection blocks
  - Stage 1 TSCM
  - Stage 2 TAR router
  - Stage 2 decoder FRM blocks

Frozen:
  - Stable Diffusion backbone
  - VAE encoder and its pretrained LoRA
  - VAE decoder backbone outside FRM
  - Stage 2 high-frequency discriminator (used as a fixed critic)

The main restoration path is deterministic top-1 routing. The router is trained
by a separate periodic oracle branch because the discrete top-1 decision cannot
provide a reliable gradient to the router.
"""

import argparse
import contextlib
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from universal_dataset import AlignedDataset
from our_modules.fape_adv import NLayerPatchDiscriminator, gan_g_loss, set_requires_grad
from our_modules.hint_module import (
    materialize_tscm_inject_blocks_from_state_dict,
    patch_unet_forward_for_fdi,
)
from our_modules.lora_module import (
    inject_lora_into_vae_encoder,
    lora_state_dict,
    set_lora_enabled as set_lora_enabled_vae,
)
from our_modules.stage2_module import MultiLoRALinear
from our_modules.tar_router import TARRouter
from our_modules.tscm_module import TSCM
from train_stage2_frm import (
    adv_hf_feat,
    build_lpips_module,
    denorm01,
    disable_sd_unet_checkpoint_flags,
    encode_deg_once_get_z_and_encfeats,
    get_decoder_refine_blocks,
    get_decoder_refine_state,
    get_parameterization,
    lhffid_l1,
    load_decoder_refine_state,
    load_full_ldm_from_ckpt,
    monkey_patch_sd_checkpoint,
    patch_vae_decode_for_tms,
    setup_torch_flags,
    unet_out_to_z0,
)


def load_checkpoint(path: str, *, mmap: bool = False) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    try:
        payload = torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a dict: {path}")
    return payload


def unique_params(params: Iterable[nn.Parameter]) -> List[nn.Parameter]:
    seen, out = set(), []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def require_keys(payload: Dict[str, Any], keys: Sequence[str], label: str):
    missing = [key for key in keys if key not in payload]
    if missing:
        raise RuntimeError(f"{label} checkpoint missing keys: {missing}")


def as_int_list(value: Any) -> List[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return [int(x) for x in list(value)]


def _set_module(root: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora_multi_into_unet_crossattn(
    unet: nn.Module,
    num_experts: int,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
):
    target_keys = ("to_q", "to_k", "to_v", "to_out")
    replacements = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in target_keys):
            replacements.append((name, module))
    for name, module in replacements:
        _set_module(
            unet,
            name,
            MultiLoRALinear(
                module,
                num_experts=int(num_experts),
                r=int(r),
                alpha=int(alpha),
                dropout=float(dropout),
            ),
        )


def cache_multilora_layers(unet: nn.Module) -> List[MultiLoRALinear]:
    layers = [module for module in unet.modules() if isinstance(module, MultiLoRALinear)]
    print(f"[INFO] cached MultiLoRALinear layers: {len(layers)}")
    return layers


def set_active_expert_cached(lora_layers: Sequence[MultiLoRALinear], expert_id: int):
    for module in lora_layers:
        module.set_active(int(expert_id))


def is_unet_expert_key(name: str) -> bool:
    lower = name.lower()
    return ("lora_" in name) or any(
        key in lower for key in ("fdi", "inject", "control", "hint", "tscm")
    )


def compact_unet_state_dict(unet: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in unet.state_dict().items()
        if is_unet_expert_key(name)
    }


def run_tscm(
    tscm: TSCM,
    z_base: torch.Tensor,
    x_deg: torch.Tensor,
    t_idx: torch.Tensor,
    eps: torch.Tensor,
    *,
    use_delta_z: bool,
):
    """
    Make the latent-perturbation choice explicit.

    The checked-in TSCM.forward() currently returns z_base and therefore keeps
    delta-z disabled. --use-delta-z restores the strict paper Eq.(2) path.
    """
    _, _, h, w = z_base.shape
    t_idx = t_idx.long()
    t_int = tscm.t_list[t_idx]

    f1 = tscm.struct_extractor(x_deg, size_hw=(h, w), t_int=t_int)
    f2 = tscm.down12(f1)
    f3 = tscm.down23(f2)

    delta_z = None
    if use_delta_z:
        delta_z = tscm.delta_z_head(f1)
        z_unet = tscm._latent_perturb(z_base, delta_z, t_int, eps)
    else:
        z_unet = z_base

    feats = {
        int(f1.shape[-2]): f1,
        int(f2.shape[-2]): f2,
        int(f3.shape[-2]): f3,
    }
    aux = {"t_int": t_int, "f_str": f1, "delta_z": delta_z}
    return z_unet, feats, aux


def run_selected_experts(
    *,
    x_deg: torch.Tensor,
    z_base: torch.Tensor,
    ctx: torch.Tensor,
    selected: torch.Tensor,
    t_shift: int,
    tscm: TSCM,
    unet: nn.Module,
    lora_layers: Sequence[MultiLoRALinear],
    alphas_cumprod: torch.Tensor,
    parameterization: str,
    autocast_ctx,
    use_delta_z: bool,
) -> torch.Tensor:
    """
    Run only selected experts while preserving gradients to expert LoRA/TSCM.
    """
    parts, positions = [], []
    common_eps = torch.randn_like(z_base)

    for expert_tensor in selected.unique(sorted=True):
        expert_id = int(expert_tensor.item())
        ids = torch.nonzero(selected == expert_id, as_tuple=False).squeeze(1)
        x_part = x_deg.index_select(0, ids)
        z_part = z_base.index_select(0, ids)
        ctx_part = ctx.index_select(0, ids)
        eps_part = common_eps.index_select(0, ids)
        t_idx = torch.full(
            (ids.numel(),),
            expert_id,
            device=x_deg.device,
            dtype=torch.long,
        )

        z_unet, feats, aux = run_tscm(
            tscm,
            z_part,
            x_part,
            t_idx,
            eps_part,
            use_delta_z=bool(use_delta_z),
        )
        set_active_expert_cached(lora_layers, expert_id)
        t_int = aux["t_int"].long()
        t_feed = (t_int + int(t_shift)).clamp(
            0,
            int(alphas_cumprod.shape[0]) - 1,
        )

        with autocast_ctx(True):
            unet_out = unet(
                z_unet,
                timesteps=t_feed,
                context=ctx_part,
                fdi_feats=feats,
            )
        z0_part = unet_out_to_z0(
            unet_out.float(),
            z_unet.float(),
            t_int,
            alphas_cumprod,
            parameterization,
        )
        parts.append(z0_part)
        positions.append(ids)

    packed = torch.cat(parts, dim=0)
    packed_positions = torch.cat(positions, dim=0)
    restore_order = torch.argsort(packed_positions)
    return packed.index_select(0, restore_order)


@torch.no_grad()
def run_all_experts(
    *,
    x_deg: torch.Tensor,
    z_base: torch.Tensor,
    ctx: torch.Tensor,
    num_experts: int,
    t_shift: int,
    tscm: TSCM,
    unet: nn.Module,
    lora_layers: Sequence[MultiLoRALinear],
    alphas_cumprod: torch.Tensor,
    parameterization: str,
    autocast_ctx,
    use_delta_z: bool,
) -> torch.Tensor:
    outputs = []
    common_eps = torch.randn_like(z_base)
    for expert_id in range(int(num_experts)):
        t_idx = torch.full(
            (x_deg.shape[0],),
            expert_id,
            device=x_deg.device,
            dtype=torch.long,
        )
        z_unet, feats, aux = run_tscm(
            tscm,
            z_base,
            x_deg,
            t_idx,
            common_eps,
            use_delta_z=bool(use_delta_z),
        )
        set_active_expert_cached(lora_layers, expert_id)
        t_int = aux["t_int"].long()
        t_feed = (t_int + int(t_shift)).clamp(
            0,
            int(alphas_cumprod.shape[0]) - 1,
        )
        with autocast_ctx(True):
            unet_out = unet(
                z_unet,
                timesteps=t_feed,
                context=ctx,
                fdi_feats=feats,
            )
        outputs.append(
            unet_out_to_z0(
                unet_out.float(),
                z_unet.float(),
                t_int,
                alphas_cumprod,
                parameterization,
            )
        )
    return torch.stack(outputs, dim=1)


def router_soft_target(oracle_losses: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(
        -oracle_losses / max(float(temperature), 1e-6),
        dim=1,
    ).clamp_min(1e-8)


def router_balance_loss(probs: torch.Tensor) -> torch.Tensor:
    mean_probs = probs.mean(dim=0)
    target = torch.full_like(mean_probs, 1.0 / float(mean_probs.numel()))
    return ((mean_probs - target) ** 2).mean()


def router_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()


@contextlib.contextmanager
def temporarily_freeze(params: Sequence[nn.Parameter]):
    old_flags = [param.requires_grad for param in params]
    try:
        for param in params:
            param.requires_grad_(False)
        yield
    finally:
        for param, old_flag in zip(params, old_flags):
            param.requires_grad_(old_flag)


@torch.no_grad()
def warmup_current_resolution(
    *,
    train_size: int,
    tscm: TSCM,
    unet: nn.Module,
    lora_layers: Sequence[MultiLoRALinear],
    ctx: torch.Tensor,
    autocast_ctx,
    use_delta_z: bool,
):
    latent_hw = int(train_size) // 8
    if latent_hw <= 0 or int(train_size) % 8 != 0:
        raise ValueError("--train-size must be a positive multiple of 8")

    before = set(unet.tscm_inject_blocks.keys())
    z_base = torch.randn(1, 4, latent_hw, latent_hw, device=ctx.device)
    x_deg = torch.randn(1, 3, int(train_size), int(train_size), device=ctx.device)
    t_idx = torch.zeros(1, dtype=torch.long, device=ctx.device)
    eps = torch.randn_like(z_base)
    z_unet, feats, aux = run_tscm(
        tscm,
        z_base,
        x_deg,
        t_idx,
        eps,
        use_delta_z=bool(use_delta_z),
    )
    set_active_expert_cached(lora_layers, 0)
    with autocast_ctx(True):
        _ = unet(
            z_unet,
            timesteps=aux["t_int"].long(),
            context=ctx[:1],
            fdi_feats=feats,
        )
    created = sorted(set(unet.tscm_inject_blocks.keys()) - before)
    if created:
        print(f"[INFO] warm-up created current-resolution inject blocks: {created}")


def add_group(groups: List[Dict[str, Any]], params: Sequence[nn.Parameter], lr: float, name: str):
    params = unique_params(params)
    if params:
        groups.append({"params": params, "lr": float(lr), "name": str(name)})
        count = sum(param.numel() for param in params)
        print(f"[TRAIN] {name}: tensors={len(params)} params={count:,} lr={float(lr):.3e}")


def build_compact_payload(
    *,
    args,
    epoch: int,
    global_step: int,
    opt_step: int,
    micro_step: int,
    t_list: List[int],
    scale_factor: float,
    parameterization: str,
    t_shift: int,
    vae,
    tscm: TSCM,
    unet: nn.Module,
    router: TARRouter,
    disc: nn.Module,
    opt: torch.optim.Optimizer,
    scaler,
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "opt_step": int(opt_step),
        "micro_step": int(micro_step),
        "t_list": list(t_list),
        "scale_factor": float(scale_factor),
        "parameterization": str(parameterization),
        "t_shift": int(t_shift),
        "lora_vae_encoder": lora_state_dict(vae.encoder),
        "tscm": cpu_state_dict(tscm),
        "unet_trainable": compact_unet_state_dict(unet),
        "router": cpu_state_dict(router),
        "decoder_refine": cpu_tree(get_decoder_refine_state(vae.decoder)),
        "disc": cpu_state_dict(disc),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "args": vars(args),
    }


def export_compatible_checkpoints(
    *,
    args,
    outdir: str,
    t_list: List[int],
    scale_factor: float,
    parameterization: str,
    t_shift: int,
    vae,
    tscm: TSCM,
    unet: nn.Module,
    router: TARRouter,
    disc: nn.Module,
):
    """
    Export three files accepted by test_all.py's existing CLI.
    The Stage 1 export is intentionally large because test_all.py expects the
    complete UNet state under 'unet_state'.
    """
    export_dir = os.path.join(outdir, "compatible")
    os.makedirs(export_dir, exist_ok=True)
    stage1_path = os.path.join(export_dir, "ckpt_stage1_joint.pt")
    router_path = os.path.join(export_dir, "ckpt_router_joint.pt")
    stage2_path = os.path.join(export_dir, "ckpt_stage2_joint.pt")

    common = {
        "t_list": list(t_list),
        "scale_factor": float(scale_factor),
        "parameterization": str(parameterization),
        "t_shift": int(t_shift),
        "args": vars(args),
    }
    stage1_payload = {
        **common,
        "lora_vae_encoder": lora_state_dict(vae.encoder),
        "tscm": cpu_state_dict(tscm),
        "unet_state": cpu_state_dict(unet),
    }
    torch.save(stage1_payload, stage1_path)
    del stage1_payload

    torch.save(
        {
            **common,
            "stage1_ckpt": stage1_path,
            "router": cpu_state_dict(router),
        },
        router_path,
    )
    torch.save(
        {
            **common,
            "stage1_ckpt": stage1_path,
            "router_ckpt": router_path,
            "decoder_refine": cpu_tree(get_decoder_refine_state(vae.decoder)),
            "disc": cpu_state_dict(disc),
        },
        stage2_path,
    )
    print("[EXPORT]", stage1_path)
    print("[EXPORT]", router_path)
    print("[EXPORT]", stage2_path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./checkpoint/v2-inference.yaml")
    ap.add_argument("--ckpt", default="./checkpoint/512-base-ema.ckpt")
    ap.add_argument("--stage1-ckpt", default="./outputs/ckpt_stage1.pt")
    ap.add_argument("--router-ckpt", default="./outputs/ckpt_router.pt")
    ap.add_argument("--stage2-ckpt", default="./outputs/ckpt_stage2.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--outdir", default="./outputs/joint_continue")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--train-experts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-unet-inject", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-tscm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-router", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-frm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-frm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-delta-z", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--lr-expert", type=float, default=5e-7)
    ap.add_argument("--lr-inject", type=float, default=5e-7)
    ap.add_argument("--lr-tscm", type=float, default=1e-6)
    ap.add_argument("--lr-router", type=float, default=2e-6)
    ap.add_argument("--lr-frm", type=float, default=5e-7)
    ap.add_argument("--wd", type=float, default=0.0)

    ap.add_argument("--w-latent", type=float, default=1.0)
    ap.add_argument("--w-img", type=float, default=1.0)
    ap.add_argument("--w-hf", type=float, default=0.5)
    ap.add_argument("--hf-kind", default="wavelet", choices=["laplacian", "wavelet", "dct"])
    ap.add_argument("--hf-lap-mode", default="4", choices=["4", "8"])
    ap.add_argument("--hf-dct-cutoff", type=float, default=0.10)
    ap.add_argument("--hf-dct-mode", default="radial", choices=["radial", "square"])
    ap.add_argument("--w-lpips", type=float, default=0.0)
    ap.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--w-adv", type=float, default=0.002)
    ap.add_argument("--adv-hf-kind", default="laplacian", choices=["laplacian", "wavelet", "dct"])
    ap.add_argument("--adv-gan-mode", default="nsgan", choices=["nsgan", "hinge"])

    ap.add_argument("--router-every", type=int, default=4)
    ap.add_argument("--router-soft-tau", type=float, default=0.20)
    ap.add_argument("--router-mix-temperature", type=float, default=0.75)
    ap.add_argument("--w-router-ce", type=float, default=0.5)
    ap.add_argument("--w-router-kl", type=float, default=0.5)
    ap.add_argument("--w-router-balance", type=float, default=0.1)
    ap.add_argument("--w-router-entropy", type=float, default=0.02)
    ap.add_argument("--w-router-img", type=float, default=0.1)

    ap.add_argument("--tar-deg-feat-dim", type=int, default=512)
    ap.add_argument("--tar-deg-prob-dim", type=int, default=16)
    ap.add_argument("--tar-proj-dim", type=int, default=256)
    ap.add_argument("--tar-temperature", type=float, default=1.0)
    ap.add_argument("--enc-ms-max-feats", type=int, default=4)
    ap.add_argument("--enc-ms-include-full-res", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--enc-ms-detach", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--vis-every", type=int, default=400)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N train steps; 0 means unlimited")
    ap.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--export-compatible-final", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true", help="build the model and optimizer, then exit")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_dir = os.path.join(args.outdir, "ckpt")
    vis_dir = os.path.join(args.outdir, "vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    setup_torch_flags()
    monkey_patch_sd_checkpoint()
    use_amp = bool(args.use_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = bool(use_amp and args.amp_dtype == "fp16")

    def autocast_ctx(enabled: bool = True):
        if device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=bool(use_amp and enabled),
        )

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    stage1 = load_checkpoint(args.stage1_ckpt, mmap=True)
    require_keys(
        stage1,
        ["t_list", "lora_vae_encoder", "tscm", "unet_state"],
        "Stage 1",
    )
    router_init = load_checkpoint(args.router_ckpt)
    require_keys(router_init, ["router"], "Router")
    stage2 = load_checkpoint(args.stage2_ckpt)
    require_keys(stage2, ["decoder_refine", "disc"], "Stage 2")

    t_list = as_int_list(stage1["t_list"])
    num_experts = len(t_list)
    scale_factor_src = float(stage1.get("scale_factor", 1.0))
    parameterization_src = str(stage1.get("parameterization", "eps")).lower()
    t_shift = int(stage1.get("t_shift", 0))
    stage1_args = stage1.get("args", {}) if isinstance(stage1.get("args"), dict) else {}
    print(f"[STAGE1] num_experts={num_experts}, t_list={t_list}")

    cfg = OmegaConf.load(args.config)
    ldm = instantiate_from_config(cfg.model)
    load_full_ldm_from_ckpt(ldm, args.ckpt)
    ldm.to(device)
    disable_sd_unet_checkpoint_flags(ldm)
    ldm.eval()
    for param in ldm.parameters():
        param.requires_grad_(False)

    vae = ldm.first_stage_model
    unet = ldm.model.diffusion_model
    patch_unet_forward_for_fdi(
        unet,
        inject_in_input=True,
        inject_in_middle=False,
        inject_in_output=False,
    )
    patch_vae_decode_for_tms(vae)

    alphas_cumprod = getattr(ldm, "alphas_cumprod", None)
    if alphas_cumprod is None:
        alphas_cumprod = getattr(ldm.model, "alphas_cumprod")
    alphas_cumprod = alphas_cumprod.to(device=device, dtype=torch.float32)
    scale_factor = float(scale_factor_src or getattr(ldm, "scale_factor", 1.0))
    parameterization = parameterization_src or get_parameterization(ldm)

    inject_lora_into_vae_encoder(
        vae.encoder,
        r_lin=int(stage1_args.get("lora_lin_r", 8)),
        a_lin=int(stage1_args.get("lora_lin_alpha", 8)),
        r_conv=int(stage1_args.get("lora_conv_r", 8)),
        a_conv=int(stage1_args.get("lora_conv_alpha", 8)),
        dropout=float(stage1_args.get("lora_dropout", 0.0)),
    )
    miss, unexp = vae.encoder.load_state_dict(stage1["lora_vae_encoder"], strict=False)
    print(f"[STAGE1] vae.encoder loaded. missing={len(miss)} unexpected={len(unexp)}")

    tscm = TSCM(
        t_list=t_list,
        alphas_cumprod=alphas_cumprod,
        t_dim=int(stage1_args.get("tscm_t_dim", 128)),
        struct_base_ch=int(stage1_args.get("tscm_struct_base_ch", 64)),
        ch1=int(stage1_args.get("tscm_ch1", 320)),
        ch2=int(stage1_args.get("tscm_ch2", 640)),
        ch3=int(stage1_args.get("tscm_ch3", 1280)),
    ).to(device)
    miss, unexp = tscm.load_state_dict(stage1["tscm"], strict=False)
    print(f"[STAGE1] tscm loaded. missing={len(miss)} unexpected={len(unexp)}")

    inject_lora_multi_into_unet_crossattn(
        unet,
        num_experts=num_experts,
        r=int(stage1_args.get("lora_r", 8)),
        alpha=int(stage1_args.get("lora_alpha", 16)),
        dropout=float(stage1_args.get("lora_dropout", 0.0)),
    )
    materialize_tscm_inject_blocks_from_state_dict(unet, stage1["unet_state"])
    miss, unexp = unet.load_state_dict(stage1["unet_state"], strict=False)
    print(f"[STAGE1] unet loaded. missing={len(miss)} unexpected={len(unexp)}")
    lora_layers = cache_multilora_layers(unet)
    del stage1

    router = TARRouter(
        num_experts=num_experts,
        deg_feat_dim=int(args.tar_deg_feat_dim),
        deg_prob_dim=int(args.tar_deg_prob_dim),
        proj_dim=int(args.tar_proj_dim),
        top_s=1,
        encoder=None,
    ).to(device)
    miss, unexp = router.load_state_dict(router_init["router"], strict=False)
    print(f"[ROUTER] loaded. missing={len(miss)} unexpected={len(unexp)}")
    del router_init

    refine_blocks = get_decoder_refine_blocks(vae.decoder)
    if not refine_blocks:
        raise RuntimeError("No decoder FRM blocks found.")
    load_decoder_refine_state(vae.decoder, stage2["decoder_refine"])
    print(f"[STAGE2] loaded FRM blocks: {len(refine_blocks)}")

    adv_in_ch = 9 if str(args.adv_hf_kind) == "wavelet" else 3
    stage2_args = stage2.get("args", {}) if isinstance(stage2.get("args"), dict) else {}
    disc = NLayerPatchDiscriminator(
        in_ch=int(adv_in_ch),
        base_ch=int(stage2_args.get("disc_base_ch", 64)),
        n_layers=int(stage2_args.get("disc_layers", 3)),
        norm="instance",
        use_sn=bool(stage2_args.get("disc_use_sn", True)),
    ).to(device)
    miss, unexp = disc.load_state_dict(stage2["disc"], strict=False)
    print(f"[STAGE2] fixed critic loaded. missing={len(miss)} unexpected={len(unexp)}")
    disc.eval()
    set_requires_grad(disc, False)
    del stage2

    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    for param in unet.parameters():
        param.requires_grad_(False)
    for param in tscm.parameters():
        param.requires_grad_(False)
    for param in router.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        ldm.cond_stage_model.eval()
        uc_cache = ldm.get_learned_conditioning([""] * int(args.bs)).to(device)
    if use_amp:
        uc_cache = uc_cache.to(dtype=amp_dtype)

    def get_uc(batch_size: int):
        if uc_cache.shape[0] == batch_size:
            return uc_cache
        return uc_cache[:1].repeat(batch_size, 1, 1)

    try:
        ldm.cond_stage_model.to("cpu")
    except Exception:
        pass

    warmup_current_resolution(
        train_size=int(args.train_size),
        tscm=tscm,
        unet=unet,
        lora_layers=lora_layers,
        ctx=get_uc(1),
        autocast_ctx=autocast_ctx,
        use_delta_z=bool(args.use_delta_z),
    )

    resume_payload: Optional[Dict[str, Any]] = None
    if args.resume:
        resume_payload = load_checkpoint(args.resume)
        require_keys(
            resume_payload,
            ["tscm", "unet_trainable", "router", "decoder_refine"],
            "Joint resume",
        )
        if as_int_list(resume_payload.get("t_list", t_list)) != t_list:
            raise RuntimeError("Resume checkpoint t_list does not match Stage 1 checkpoint.")
        materialize_tscm_inject_blocks_from_state_dict(unet, resume_payload["unet_trainable"])
        tscm.load_state_dict(resume_payload["tscm"], strict=False)
        unet.load_state_dict(resume_payload["unet_trainable"], strict=False)
        router.load_state_dict(resume_payload["router"], strict=False)
        load_decoder_refine_state(vae.decoder, resume_payload["decoder_refine"])
        if "disc" in resume_payload:
            disc.load_state_dict(resume_payload["disc"], strict=False)
        print(f"[RESUME] loaded weights from {args.resume}")

    unet_lora_params, unet_inject_params = [], []
    for name, param in unet.named_parameters():
        if "lora_" in name:
            param.requires_grad_(bool(args.train_experts))
            if param.requires_grad:
                unet_lora_params.append(param)
        elif is_unet_expert_key(name):
            param.requires_grad_(bool(args.train_unet_inject))
            if param.requires_grad:
                unet_inject_params.append(param)

    tscm_params = []
    for param in tscm.parameters():
        param.requires_grad_(bool(args.train_tscm))
        if param.requires_grad:
            tscm_params.append(param)

    frm_params = []
    for block in refine_blocks:
        for param in block.parameters():
            param.requires_grad_(bool(args.train_frm))
            if param.requires_grad:
                frm_params.append(param)
    frm_params = unique_params(frm_params)

    router_params = []
    for param in router.parameters():
        param.requires_grad_(bool(args.train_router))
        if param.requires_grad:
            router_params.append(param)

    groups: List[Dict[str, Any]] = []
    add_group(groups, unet_lora_params, args.lr_expert, "unet_expert_lora")
    add_group(groups, unet_inject_params, args.lr_inject, "unet_tscm_inject")
    add_group(groups, tscm_params, args.lr_tscm, "tscm")
    add_group(groups, frm_params, args.lr_frm, "frm")
    add_group(groups, router_params, args.lr_router, "router")
    if not groups:
        raise RuntimeError("No trainable parameters selected.")
    trainable_params = unique_params(param for group in groups for param in group["params"])
    opt = torch.optim.AdamW(groups, weight_decay=float(args.wd))

    start_epoch = 0
    global_step = 0
    opt_step = 0
    micro_step = 0
    if resume_payload is not None:
        if "opt" in resume_payload:
            opt.load_state_dict(resume_payload["opt"])
        if use_scaler and resume_payload.get("scaler") is not None:
            scaler.load_state_dict(resume_payload["scaler"])
        start_epoch = int(resume_payload.get("epoch", 0))
        global_step = int(resume_payload.get("global_step", 0))
        opt_step = int(resume_payload.get("opt_step", 0))
        micro_step = int(resume_payload.get("micro_step", 0))
        print(
            f"[RESUME] counters epoch={start_epoch} step={global_step} "
            f"opt_step={opt_step} micro_step={micro_step}"
        )
        del resume_payload

    if args.dry_run:
        print("[DRY-RUN] model, pretrained weights, dynamic inject blocks, and optimizer are ready.")
        return

    lpips_mod = None
    if float(args.w_lpips) > 0.0:
        lpips_mod = build_lpips_module(net=str(args.lpips_net)).to(device).eval()
        for param in lpips_mod.parameters():
            param.requires_grad_(False)
        print(f"[LPIPS] enabled net={args.lpips_net}")

    dataset = AlignedDataset(train_size=int(args.train_size), mode="train")
    loader = DataLoader(
        dataset,
        batch_size=int(args.bs),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=4 if int(args.num_workers) > 0 else None,
    )
    grad_accum = max(1, int(args.grad_accum))
    t_list_tensor = torch.tensor(t_list, device=device, dtype=torch.long)
    opt.zero_grad(set_to_none=True)
    stop_training = False

    for epoch in range(start_epoch, int(args.epochs)):
        if args.train_experts or args.train_unet_inject:
            unet.train()
        else:
            unet.eval()
        tscm.train(bool(args.train_tscm))
        router.train(bool(args.train_router))
        vae.decoder.eval()
        for block in refine_blocks:
            block.train(bool(args.train_frm))

        for iteration, (x_deg, x_gt) in enumerate(loader):
            x_deg = x_deg.to(device, non_blocking=True).float()
            x_gt = x_gt.to(device, non_blocking=True).float()
            batch_size = x_deg.shape[0]
            ctx = get_uc(batch_size)
            if (micro_step % grad_accum) == 0:
                opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                set_lora_enabled_vae(vae.encoder, False)
                z_gt = vae.encode(x_gt).mode().float() * float(scale_factor)
                set_lora_enabled_vae(vae.encoder, True)
                z_base, enc_feats = encode_deg_once_get_z_and_encfeats(
                    vae=vae,
                    x_deg=x_deg,
                    scale_factor=float(scale_factor),
                    train_size=int(args.train_size),
                    max_feats=int(args.enc_ms_max_feats),
                    include_full_res=bool(args.enc_ms_include_full_res),
                    detach_feats=bool(args.enc_ms_detach),
                )

            tar_out = router(x_deg, temperature=float(args.tar_temperature))
            logits = tar_out["logits"].float()
            probs = tar_out["probs"].float()
            selected = logits.detach().argmax(dim=1)
            t_selected = t_list_tensor[selected]

            z0_selected = run_selected_experts(
                x_deg=x_deg,
                z_base=z_base,
                ctx=ctx,
                selected=selected,
                t_shift=t_shift,
                tscm=tscm,
                unet=unet,
                lora_layers=lora_layers,
                alphas_cumprod=alphas_cumprod,
                parameterization=parameterization,
                autocast_ctx=autocast_ctx,
                use_delta_z=bool(args.use_delta_z),
            )
            with autocast_ctx(True):
                x_hat = vae.decode(
                    (z0_selected / float(scale_factor)).float(),
                    enc_feats=enc_feats if args.use_frm else None,
                    t_val=t_selected if args.use_frm else None,
                )

            l_latent = (z0_selected - z_gt).abs().mean()
            l_img = (x_hat - x_gt).abs().mean()
            l_hf = lhffid_l1(
                x_hat,
                x_gt,
                kind=str(args.hf_kind),
                lap_mode=str(args.hf_lap_mode),
                dct_cutoff=float(args.hf_dct_cutoff),
                dct_mode=str(args.hf_dct_mode),
            )
            l_lpips = torch.tensor(0.0, device=device)
            if lpips_mod is not None:
                with torch.autocast(device_type=device.type, enabled=False):
                    l_lpips = lpips_mod(
                        x_hat.float().clamp(-1.0, 1.0),
                        x_gt.float().clamp(-1.0, 1.0),
                    ).mean()

            l_adv = torch.tensor(0.0, device=device)
            if float(args.w_adv) > 0.0:
                with autocast_ctx(True):
                    hf_fake = adv_hf_feat(
                        x_hat,
                        kind=str(args.adv_hf_kind),
                        lap_mode=str(args.hf_lap_mode),
                        dct_cutoff=float(args.hf_dct_cutoff),
                        dct_mode=str(args.hf_dct_mode),
                    )
                    l_adv = gan_g_loss(disc(hf_fake), mode=str(args.adv_gan_mode))

            main_loss = (
                float(args.w_latent) * l_latent
                + float(args.w_img) * l_img
                + float(args.w_hf) * l_hf
                + float(args.w_lpips) * l_lpips
                + float(args.w_adv) * l_adv
            )

            router_loss = torch.tensor(0.0, device=device)
            l_router_ce = torch.tensor(0.0, device=device)
            l_router_kl = torch.tensor(0.0, device=device)
            l_router_balance = torch.tensor(0.0, device=device)
            l_router_entropy = torch.tensor(0.0, device=device)
            l_router_img = torch.tensor(0.0, device=device)
            router_due = (
                bool(args.train_router)
                and int(args.router_every) > 0
                and (global_step % int(args.router_every) == 0)
            )
            if router_due:
                z0_all = run_all_experts(
                    x_deg=x_deg,
                    z_base=z_base,
                    ctx=ctx,
                    num_experts=num_experts,
                    t_shift=t_shift,
                    tscm=tscm,
                    unet=unet,
                    lora_layers=lora_layers,
                    alphas_cumprod=alphas_cumprod,
                    parameterization=parameterization,
                    autocast_ctx=autocast_ctx,
                    use_delta_z=bool(args.use_delta_z),
                )
                oracle_losses = (z0_all - z_gt[:, None]).abs().mean(dim=(2, 3, 4))
                oracle_idx = oracle_losses.argmin(dim=1)
                target_probs = router_soft_target(
                    oracle_losses,
                    temperature=float(args.router_soft_tau),
                )
                l_router_ce = F.cross_entropy(logits, oracle_idx)
                l_router_kl = F.kl_div(
                    F.log_softmax(logits, dim=1),
                    target_probs,
                    reduction="batchmean",
                )
                l_router_balance = router_balance_loss(probs)
                l_router_entropy = router_entropy(probs)

                if float(args.w_router_img) > 0.0:
                    mix_weights = torch.softmax(
                        logits / max(float(args.router_mix_temperature), 1e-6),
                        dim=1,
                    )
                    z0_mix = (z0_all * mix_weights[:, :, None, None, None]).sum(dim=1)
                    t_mix = (mix_weights * t_list_tensor[None].float()).sum(dim=1)
                    with temporarily_freeze(frm_params):
                        with autocast_ctx(True):
                            x_mix = vae.decode(
                                (z0_mix / float(scale_factor)).float(),
                                enc_feats=enc_feats if args.use_frm else None,
                                t_val=t_mix if args.use_frm else None,
                            )
                    l_router_img = (x_mix - x_gt).abs().mean()

                router_loss = (
                    float(args.w_router_ce) * l_router_ce
                    + float(args.w_router_kl) * l_router_kl
                    + float(args.w_router_balance) * l_router_balance
                    - float(args.w_router_entropy) * l_router_entropy
                    + float(args.w_router_img) * l_router_img
                )

            loss = main_loss + router_loss
            micro_step += 1
            scaled_loss = loss / float(grad_accum)
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            global_step += 1

            if (micro_step % grad_accum) == 0:
                if float(args.grad_clip) > 0.0:
                    if use_scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

            if global_step % int(args.log_every) == 0:
                usage = selected.bincount(minlength=num_experts).tolist()
                print(
                    f"[JOINT ep{epoch} it{iteration}] step={global_step} opt={opt_step} "
                    f"loss={loss.item():.4f} latent={l_latent.item():.4f} "
                    f"img={l_img.item():.4f} hf={l_hf.item():.4f} "
                    f"lpips={l_lpips.item():.4f} adv={l_adv.item():.4f} "
                    f"r_ce={l_router_ce.item():.4f} r_kl={l_router_kl.item():.4f} "
                    f"r_img={l_router_img.item():.4f} sel={usage}"
                )

            if global_step % int(args.vis_every) == 0:
                with torch.no_grad():
                    count = min(4, batch_size)
                    x_base = vae.decode((z_base[:count] / float(scale_factor)).float())
                    panels = []
                    for idx in range(count):
                        images = [
                            denorm01(x_deg[idx]).clamp(0, 1),
                            denorm01(x_base[idx]).clamp(0, 1),
                            denorm01(x_hat[idx]).clamp(0, 1),
                            denorm01(x_gt[idx]).clamp(0, 1),
                        ]
                        height = images[0].shape[-2]
                        gap = torch.ones(3, height, 6, device=device)
                        row = []
                        for image in images:
                            if row:
                                row.append(gap)
                            row.append(image)
                        panels.append(torch.cat(row, dim=2))
                    vutils.save_image(
                        torch.stack(panels),
                        os.path.join(vis_dir, f"joint_step_{global_step:08d}.png"),
                        nrow=1,
                    )

            if global_step % int(args.save_every) == 0:
                payload = build_compact_payload(
                    args=args,
                    epoch=epoch,
                    global_step=global_step,
                    opt_step=opt_step,
                    micro_step=micro_step,
                    t_list=t_list,
                    scale_factor=scale_factor,
                    parameterization=parameterization,
                    t_shift=t_shift,
                    vae=vae,
                    tscm=tscm,
                    unet=unet,
                    router=router,
                    disc=disc,
                    opt=opt,
                    scaler=scaler,
                )
                path = os.path.join(ckpt_dir, "ckpt_joint_continue_latest.pt")
                torch.save(payload, path)
                print("[SAVE]", path)

            if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
                stop_training = True
                break

        if (micro_step % grad_accum) != 0:
            if float(args.grad_clip) > 0.0:
                if use_scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
            if use_scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            print(f"[FLUSH] epoch={epoch} opt_step={opt_step}")

        if stop_training:
            print(f"[STOP] reached --max-steps={int(args.max_steps)}")
            break

    if not args.save_final:
        print("[DONE] skipped final checkpoint because --no-save-final was set.")
        return

    final_payload = build_compact_payload(
        args=args,
        epoch=int(args.epochs),
        global_step=global_step,
        opt_step=opt_step,
        micro_step=micro_step,
        t_list=t_list,
        scale_factor=scale_factor,
        parameterization=parameterization,
        t_shift=t_shift,
        vae=vae,
        tscm=tscm,
        unet=unet,
        router=router,
        disc=disc,
        opt=opt,
        scaler=scaler,
    )
    final_path = os.path.join(ckpt_dir, "ckpt_joint_continue_final.pt")
    torch.save(final_payload, final_path)
    print("[DONE] saved compact resume checkpoint:", final_path)

    if args.export_compatible_final:
        export_compatible_checkpoints(
            args=args,
            outdir=args.outdir,
            t_list=t_list,
            scale_factor=scale_factor,
            parameterization=parameterization,
            t_shift=t_shift,
            vae=vae,
            tscm=tscm,
            unet=unet,
            router=router,
            disc=disc,
        )


if __name__ == "__main__":
    main()
