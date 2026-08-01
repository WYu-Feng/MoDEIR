#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from modeir_refined.bootstrap import bootstrap_legacy_backend

bootstrap_legacy_backend()
from modeir_refined.checkpoints import load_compatible_state, load_decoder_refine, load_payload  # noqa: E402
from modeir_refined.pipeline import build_refined_model  # noqa: E402


WEIGHTS = HERE / "weights" / "legacy"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Paired folder test for refined MoDE-IR")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", default=str(HERE / "test_outputs" / "deblur_baryir_current"))
    parser.add_argument("--resume", default="")
    parser.add_argument("--config", default=str(HERE / "configs" / "v2-inference.yaml"))
    parser.add_argument("--base-checkpoint", default=str(WEIGHTS / "512-base-ema.ckpt"))
    parser.add_argument("--stage1-checkpoint", default=str(WEIGHTS / "ckpt_stage1.pt"))
    parser.add_argument("--router-checkpoint", default=str(WEIGHTS / "ckpt_router.pt"))
    parser.add_argument("--stage2-checkpoint", default=str(WEIGHTS / "ckpt_stage2.pt"))
    parser.add_argument("--daclip-checkpoint", default=str(WEIGHTS / "daclip_ViT-B-32.pt"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--preserve-resolution",
        action="store_true",
        help="Use each image's original resolution. Requires paired input/GT images in the same batch to share size.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--router-backbone", choices=["daclip", "simple"], default="daclip")
    parser.add_argument("--top-s", type=int, default=2)
    parser.add_argument("--latent-mode", choices=["legacy_no_delta", "strict_delta"], default="strict_delta")
    parser.add_argument("--save-side-by-side", action="store_true")
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
    print(f"[RESUME] loaded refined checkpoint: {path}")


def paired_paths(input_dir: Path, gt_dir: Path):
    inputs = {path.name: path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS}
    gts = {path.name: path for path in gt_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS}
    names = sorted(set(inputs) & set(gts), key=lambda name: (Path(name).stem.zfill(12), name))
    return [(name, inputs[name], gts[name]) for name in names]


def load_tensor(path: Path, size: int, device: torch.device, preserve_resolution: bool):
    image = Image.open(path).convert("RGB")
    if not preserve_resolution:
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return TF.to_tensor(image).to(device)


def pad_to_multiple(x: torch.Tensor, multiple: int):
    if multiple <= 1:
        return x, (int(x.shape[-2]), int(x.shape[-1]))
    h, w = int(x.shape[-2]), int(x.shape[-1])
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    padded = F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    return padded, (h, w)


def psnr(pred: np.ndarray, target: np.ndarray):
    mse = float(np.mean((pred - target) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def ssim(pred: np.ndarray, target: np.ndarray):
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(target, pred, channel_axis=2, data_range=1.0))
    except Exception:
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_x = pred.mean()
        mu_y = target.mean()
        var_x = pred.var()
        var_y = target.var()
        cov = ((pred - mu_x) * (target - mu_y)).mean()
        return float(((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)))


def save_image(tensor: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.clamp(0, 1).cpu()).save(path)


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stable Diffusion backend")
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    restored_dir = output_dir / "restored"
    side_dir = output_dir / "side_by_side"
    output_dir.mkdir(parents=True, exist_ok=True)
    restored_dir.mkdir(parents=True, exist_ok=True)

    pairs = paired_paths(Path(args.input_dir), Path(args.gt_dir))
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No paired image names found.")
    print(f"[DATA] paired images: {len(pairs)}")
    first_size = Image.open(pairs[0][1]).size
    if args.preserve_resolution:
        first_gt_size = Image.open(pairs[0][2]).size
        if first_size != first_gt_size:
            raise RuntimeError(f"Input/GT size mismatch for {pairs[0][0]}: {first_size} vs {first_gt_size}")
        preferred_latent_hw = max(1, int(first_size[1] // 8))
        print(
            f"[DATA] preserve original resolution: first_size={first_size}, "
            f"preferred_latent_hw={preferred_latent_hw}, pad_multiple={args.pad_multiple}"
        )
    else:
        preferred_latent_hw = args.image_size // 8

    model = build_refined_model(
        config_path=args.config,
        base_checkpoint=args.base_checkpoint,
        stage1_checkpoint=args.stage1_checkpoint,
        router_checkpoint=args.router_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        device=device,
        preferred_latent_hw=preferred_latent_hw,
        window=args.window,
        router_backbone=args.router_backbone,
        daclip_checkpoint=args.daclip_checkpoint,
        top_s=args.top_s,
    )
    overlay_resume(model, args.resume)

    rows = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        names = [item[0] for item in batch]
        original_hw = None
        if args.preserve_resolution:
            sizes = [(Image.open(inp).size, Image.open(gt_path).size) for _, inp, gt_path in batch]
            for name, (inp_size, gt_size) in zip(names, sizes):
                if inp_size != gt_size:
                    raise RuntimeError(f"Input/GT size mismatch for {name}: {inp_size} vs {gt_size}")
                if inp_size != sizes[0][0]:
                    raise RuntimeError(
                        "All images in a preserve-resolution batch must share size. "
                        "Use --batch-size 1 for mixed-resolution folders."
                    )
            original_hw = (int(sizes[0][0][1]), int(sizes[0][0][0]))
        else:
            image_size_for_model = (args.image_size, args.image_size)
        deg_items = []
        for _, path, _ in batch:
            tensor = load_tensor(path, args.image_size, device, args.preserve_resolution)
            if args.preserve_resolution:
                tensor, original_hw = pad_to_multiple(tensor, args.pad_multiple)
            deg_items.append(tensor)
        deg = torch.stack(deg_items, dim=0)
        gt = torch.stack(
            [load_tensor(path, args.image_size, device, args.preserve_resolution) for _, _, path in batch],
            dim=0,
        )
        if args.preserve_resolution:
            image_size_for_model = (int(deg.shape[-2]), int(deg.shape[-1]))
        degraded = deg * 2 - 1

        with model.autocast():
            z_base, enc_feats = model.encode_degraded(degraded, image_size_for_model)
            ctx = model.conditioning(degraded.shape[0])
            route = model.router(degraded)
            weights, selected = model.router.route_topk(route["logits"], route["probs"])
            z0, t_val = model.restore_weighted(degraded, z_base, ctx, selected, weights, args.latent_mode)
            restored = ((model.decode(z0, enc_feats, t_val).clamp(-1, 1) + 1) * 0.5).clamp(0, 1)
        if args.preserve_resolution and original_hw is not None:
            restored = restored[:, :, : original_hw[0], : original_hw[1]]
            deg_for_save = deg[:, :, : original_hw[0], : original_hw[1]]
        else:
            deg_for_save = deg

        for idx, name in enumerate(names):
            pred_np = restored[idx].detach().cpu().permute(1, 2, 0).numpy().astype(np.float64)
            gt_np = gt[idx].detach().cpu().permute(1, 2, 0).numpy().astype(np.float64)
            row = {
                "name": name,
                "expert": "|".join(str(int(value)) for value in selected[idx].tolist()),
                "weights": "|".join(f"{float(value):.6f}" for value in weights[idx].tolist()),
                "psnr": psnr(pred_np, gt_np),
                "ssim": ssim(pred_np, gt_np),
            }
            rows.append(row)
            save_image(restored[idx], restored_dir / Path(name).with_suffix(".png").name)
            if args.save_side_by_side:
                comparison = torch.cat([deg_for_save[idx].cpu(), restored[idx].cpu(), gt[idx].cpu()], dim=2)
                save_image(comparison, side_dir / Path(name).with_suffix(".png").name)
            print(
                f"[{len(rows):04d}/{len(pairs):04d}] {name} experts={row['expert']} "
                f"weights={row['weights']} psnr={row['psnr']:.3f} ssim={row['ssim']:.4f}"
            )

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "expert", "weights", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "gt_dir": str(Path(args.gt_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_images": len(rows),
        "image_size": "original" if args.preserve_resolution else args.image_size,
        "preserve_resolution": bool(args.preserve_resolution),
        "pad_multiple": int(args.pad_multiple),
        "latent_mode": args.latent_mode,
        "router_backbone": args.router_backbone,
        "top_s": args.top_s,
        "resume": args.resume,
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "top1_expert_counts": {
            str(idx): int(sum(int(str(row["expert"]).split("|")[0]) == idx for row in rows))
            for idx in range(len(model.t_list))
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[SUMMARY] PSNR={summary['mean_psnr']:.4f} SSIM={summary['mean_ssim']:.5f}")
    print(f"[SUMMARY] wrote: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
