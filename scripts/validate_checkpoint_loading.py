#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from modeir_refined.bootstrap import bootstrap_legacy_backend

bootstrap_legacy_backend()

from modeir_refined.checkpoints import load_payload
from modeir_refined.pipeline import build_refined_model


def shape(value):
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def compare_state(module, state, *, skip_prefixes=()):
    own_state = module.state_dict()
    compatible = []
    unexpected = []
    skipped = []
    for key, value in state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped.append((key, "skip_prefix", shape(value), None))
            continue
        if key not in own_state:
            unexpected.append((key, shape(value)))
            continue
        if tuple(own_state[key].shape) != tuple(value.shape):
            skipped.append((key, "shape", shape(value), shape(own_state[key])))
            continue
        compatible.append(key)
    missing = [key for key in own_state if key not in compatible]
    return compatible, missing, unexpected, skipped


def print_list(title, values, limit):
    print(title + " count=" + str(len(values)))
    for item in values[:limit]:
        print("  " + repr(item))
    if len(values) > limit:
        print("  ...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(HERE / "weights" / "legacy"))
    parser.add_argument("--config", default=str(HERE / "configs" / "v2-inference.yaml"))
    parser.add_argument("--train-size", type=int, default=192)
    parser.add_argument("--router-backbone", choices=["daclip", "simple"], default="daclip")
    parser.add_argument("--top-s", type=int, default=2)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    weights = Path(args.weights)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("[CHECK] weights=" + str(weights.resolve()))
    print("[CHECK] device=" + str(device))

    print("\n[BUILD] instantiate current model and load checkpoints")
    model = build_refined_model(
        config_path=args.config,
        base_checkpoint=weights / "512-base-ema.ckpt",
        stage1_checkpoint=weights / "ckpt_stage1.pt",
        router_checkpoint=weights / "ckpt_router.pt",
        stage2_checkpoint=weights / "ckpt_stage2.pt",
        daclip_checkpoint=weights / "daclip_ViT-B-32.pt",
        device=device,
        preferred_latent_hw=args.train_size // 8,
        router_backbone=args.router_backbone,
        top_s=args.top_s,
    )
    print("[BUILD] OK")

    print("\n[DETAIL] ckpt_stage1.pt")
    stage1 = load_payload(weights / "ckpt_stage1.pt", mmap=True)
    print("top keys=" + repr(sorted(stage1.keys())))
    print("t_list=" + repr(stage1.get("t_list")))
    print("lora_vae_encoder keys=" + str(len(stage1["lora_vae_encoder"])))
    print("tscm keys=" + str(len(stage1["tscm"])))
    print("unet_state keys=" + str(len(stage1["unet_state"])))

    print("\n[DETAIL] ckpt_router.pt")
    router_payload = load_payload(weights / "ckpt_router.pt")
    compatible, missing, unexpected, skipped = compare_state(
        model.router,
        router_payload["router"],
        skip_prefixes=("encoder.",) if args.router_backbone == "daclip" else (),
    )
    print("router payload keys=" + str(len(router_payload["router"])))
    print("compatible=" + str(len(compatible)))
    print_list("missing", missing, args.limit)
    print_list("unexpected", unexpected, args.limit)
    print_list("skipped", skipped, args.limit)

    print("\n[DETAIL] ckpt_stage2.pt")
    stage2 = load_payload(weights / "ckpt_stage2.pt")
    print("top keys=" + repr(sorted(stage2.keys())))
    decoder_refine = stage2["decoder_refine"]
    print("frm_mid32 payload keys=" + str(len(decoder_refine["frm_mid32"])))
    print("frm_up payload block keys=" + repr([len(block) for block in decoder_refine["frm_up"]]))
    for index, block_state in enumerate(decoder_refine["frm_up"]):
        compatible, missing, unexpected, skipped = compare_state(model.vae.decoder.frm_up[index], block_state)
        print("frm_up[" + str(index) + "] compatible=" + str(len(compatible)))
        print_list("frm_up[" + str(index) + "] missing", missing, args.limit)
        print_list("frm_up[" + str(index) + "] unexpected", unexpected, args.limit)
        print_list("frm_up[" + str(index) + "] skipped", skipped, args.limit)
    compatible, missing, unexpected, skipped = compare_state(model.critic, stage2["disc"])
    print("disc compatible=" + str(len(compatible)))
    print_list("disc missing", missing, args.limit)
    print_list("disc unexpected", unexpected, args.limit)
    print_list("disc skipped", skipped, args.limit)


if __name__ == "__main__":
    main()
