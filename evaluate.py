#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from modeir_refined.bootstrap import bootstrap_legacy_backend

bootstrap_legacy_backend()
from modeir_refined.checkpoints import load_compatible_state, load_decoder_refine, load_payload  # noqa: E402
from modeir_refined.pipeline import build_refined_model  # noqa: E402


WEIGHTS = HERE / "weights" / "legacy"


def parse_args():
    parser = argparse.ArgumentParser(description="Run refined MoDE-IR restoration")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--config", default=str(HERE / "configs" / "v2-inference.yaml"))
    parser.add_argument("--base-checkpoint", default=str(WEIGHTS / "512-base-ema.ckpt"))
    parser.add_argument("--stage1-checkpoint", default=str(WEIGHTS / "ckpt_stage1.pt"))
    parser.add_argument("--router-checkpoint", default=str(WEIGHTS / "ckpt_router.pt"))
    parser.add_argument("--stage2-checkpoint", default=str(WEIGHTS / "ckpt_stage2.pt"))
    parser.add_argument("--daclip-checkpoint", default=str(WEIGHTS / "daclip_ViT-B-32.pt"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--router-backbone", choices=["daclip", "simple"], default="daclip")
    parser.add_argument("--top-s", type=int, default=2)
    parser.add_argument("--latent-mode", choices=["legacy_no_delta", "strict_delta"], default="strict_delta")
    return parser.parse_args()


def overlay_resume(model, path: str):
    if not path:
        return
    payload = load_payload(path)
    if "vae_encoder_lora" in payload:
        model.vae.encoder.load_state_dict(payload["vae_encoder_lora"], strict=False)
    model.unet.load_state_dict(payload["unet_lora"], strict=False)
    model.tscm.load_state_dict(payload["tscm"], strict=True)
    model.adapter.blocks.load_state_dict(payload["static_injectors"], strict=True)
    load_compatible_state(
        model.router,
        payload["router"],
        "resume TAR router",
        skip_prefixes=("encoder.",) if getattr(model.router.encoder, "skip_checkpoint_state", False) else (),
    )
    if "latent_fusion" in payload:
        model.latent_fusion.load_state_dict(payload["latent_fusion"], strict=True)
    else:
        print("[RESUME] checkpoint has no latent_fusion state; keep initialized fusion ResBlock")
    load_decoder_refine(model.vae.decoder, payload["decoder_refine"])


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stable Diffusion backend")
    device = torch.device("cuda")
    model = build_refined_model(
        config_path=args.config,
        base_checkpoint=args.base_checkpoint,
        stage1_checkpoint=args.stage1_checkpoint,
        router_checkpoint=args.router_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        device=device,
        preferred_latent_hw=args.image_size // 8,
        window=args.window,
        router_backbone=args.router_backbone,
        daclip_checkpoint=args.daclip_checkpoint,
        top_s=args.top_s,
    )
    overlay_resume(model, args.resume)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in Path(args.input_dir).iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    for path in paths:
        image = Image.open(path).convert("RGB").resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        degraded = (TF.to_tensor(image).unsqueeze(0).to(device) * 2 - 1)
        with model.autocast():
            z_base, enc_feats = model.encode_degraded(degraded, args.image_size)
            ctx = model.conditioning(1)
            route = model.router(degraded)
            weights, selected = model.router.route_topk(route["logits"], route["probs"])
            z0, t_val = model.restore_weighted(degraded, z_base, ctx, selected, weights, args.latent_mode)
            restored = model.decode(z0, enc_feats, t_val).clamp(-1, 1)
        TF.to_pil_image(((restored[0] + 1) * 0.5).cpu()).save(output_dir / f"{path.stem}.png")
        experts = "|".join(str(int(value)) for value in selected[0].tolist())
        expert_weights = "|".join(f"{float(value):.6f}" for value in weights[0].tolist())
        print(f"[DONE] {path.name}: experts={experts} weights={expert_weights}")


if __name__ == "__main__":
    main()
