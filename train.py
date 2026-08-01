#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import random
import shutil
import shlex
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))

from modeir_refined.bootstrap import bootstrap_legacy_backend

bootstrap_legacy_backend()
from universal_dataset import AlignedDataset  # noqa: E402

from modeir_refined.checkpoints import (  # noqa: E402
    cpu_state,
    decoder_refine_state,
    load_compatible_state,
    load_decoder_refine,
    load_payload,
)
from modeir_refined.losses import critic_features, gan_generator_loss, high_frequency_l1  # noqa: E402
from modeir_refined.pipeline import build_refined_model, freeze  # noqa: E402


WEIGHTS = HERE / "weights" / "legacy"
DEFAULT_DATASET_ROOT = HERE / "datasets" if (HERE / "datasets").is_dir() else HERE.parent / "datasets"


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, int(os.environ["RANK"]), local_rank, int(os.environ["WORLD_SIZE"])


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def unwrap_module(module):
    return module.module if isinstance(module, DDP) else module


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-aligned MoDEIR Stage 2 training")
    parser.add_argument("--config", default=str(HERE / "configs" / "v2-inference.yaml"))
    parser.add_argument("--base-checkpoint", default=str(WEIGHTS / "512-base-ema.ckpt"))
    parser.add_argument("--stage1-checkpoint", default=str(WEIGHTS / "ckpt_stage1.pt"))
    parser.add_argument("--router-checkpoint", default=str(WEIGHTS / "ckpt_router.pt"))
    parser.add_argument("--stage2-checkpoint", default=str(WEIGHTS / "ckpt_stage2.pt"))
    parser.add_argument("--daclip-checkpoint", default=str(WEIGHTS / "daclip_ViT-B-32.pt"))
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument(
        "--self-contained-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a self-contained run folder with code snapshot, configs, manifest, and hard-linked weights.",
    )
    parser.add_argument(
        "--weight-materialization",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to place immutable pretrained weights inside the self-contained output folder.",
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    parser.add_argument("--train-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-per-task", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--vis-every", type=int, default=0, help="Save stage visualization every N steps; 0 disables it")
    parser.add_argument("--vis-num", type=int, default=5, help="Number of random samples to save in each visualization")
    parser.add_argument("--vis-dir", default="", help="Visualization directory; defaults to <output-dir>/visualizations")
    parser.add_argument("--oracle-every", type=int, default=10)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--router-backbone", choices=["daclip", "simple"], default="daclip")
    parser.add_argument("--top-s", type=int, default=2)
    parser.add_argument("--router-gumbel-topk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp-find-unused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-mode",
        choices=[
            "encoder_lora",
            "router_classifier",
            "tscm_unet_lora_router_time",
            "fusion_s1",
            "paper_stage2",
            "router",
            "experts",
            "tscm_decoder",
            "router_tscm_decoder",
            "joint",
        ],
        default="paper_stage2",
        help=(
            "encoder_lora: train only VAE encoder LoRA against clean latents; "
            "router_classifier: train TAR as a timestep classifier from latent-difference bins; "
            "tscm_unet_lora_router_time: freeze router and train TSCM+UNet expert LoRA using router top1 time; "
            "fusion_s1: freeze previous stages and train latent fusion+FRM with s=1; "
            "paper_stage2: freeze the expert pool and train TAR, latent fusion, and FRMs; "
            "router: freeze encoder/experts/TSCM/decoder and train TAR only; "
            "experts: only continue expert LoRA with fixed Router/TSCM/decoder; "
            "tscm_decoder: freeze Router+experts and train TSCM+static injectors+decoder FRM; "
            "router_tscm_decoder: optional combined stage for Router+TSCM+static injectors+decoder FRM; "
            "joint: train all trainable refined components."
        ),
    )
    parser.add_argument(
        "--expert-sampling",
        choices=["cycle", "random"],
        default="cycle",
        help="Expert selection policy used only by --train-mode experts.",
    )
    parser.add_argument("--latent-mode", choices=["legacy_no_delta", "strict_delta"], default="strict_delta")
    parser.add_argument("--lr-experts", type=float, default=5e-6)
    parser.add_argument("--lr-tscm", type=float, default=5e-6)
    parser.add_argument("--lr-injectors", type=float, default=5e-6)
    parser.add_argument("--lr-router", type=float, default=1e-4)
    parser.add_argument("--lr-frm", type=float, default=2e-5)
    parser.add_argument("--lr-latent-fusion", type=float, default=2e-5)
    parser.add_argument("--w-latent", type=float, default=0.0)
    parser.add_argument("--w-hf", type=float, default=0.0)
    parser.add_argument("--w-adv", type=float, default=0.2)
    parser.add_argument("--w-router", type=float, default=0.0)
    parser.add_argument("--w-severity", type=float, default=0.1)
    parser.add_argument("--w-balance", type=float, default=0.0)
    parser.add_argument("--w-encoder", type=float, default=1.0)
    parser.add_argument("--w-router-cls", type=float, default=1.0)
    parser.add_argument("--freeze-experts", action="store_true")
    parser.add_argument("--freeze-encoder-lora", action="store_true")
    parser.add_argument("--freeze-tscm", action="store_true")
    parser.add_argument("--freeze-injectors", action="store_true")
    parser.add_argument("--freeze-router", action="store_true")
    parser.add_argument("--freeze-latent-fusion", action="store_true")
    parser.add_argument("--freeze-frm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-final-save", action="store_true", help="Do not write refined_last.pt at process exit")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def unique_params(parameters):
    seen, result = set(), []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def _copy_code_snapshot(destination: Path) -> None:
    code_root = destination / "code" / "refined_modeir"
    if HERE.resolve() == code_root.resolve():
        return
    code_root.mkdir(parents=True, exist_ok=True)
    for filename in ("train.py", "evaluate.py", "test_paired.py", "README.md"):
        source = HERE / filename
        if source.is_file():
            shutil.copy2(source, code_root / filename)

    def ignore(_dir, names):
        return {
            name
            for name in names
            if name == "__pycache__"
            or name.endswith(".pyc")
            or name in {"weights", "outputs", "test_outputs", "analysis_outputs"}
        }

    for dirname in ("modeir_refined", "our_modules", "scripts", "configs"):
        source = HERE / dirname
        if source.is_dir():
            shutil.copytree(source, code_root / dirname, dirs_exist_ok=True, ignore=ignore)


def _materialize_file(source: str | Path, destination: Path, mode: str) -> Path:
    source = Path(source).expanduser().resolve()
    destination = destination.expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return destination.resolve()
        except OSError:
            pass
        if destination.stat().st_size == source.stat().st_size:
            print(f"[SELF-CONTAINED] keep existing: {destination}")
            return destination.resolve()
        raise RuntimeError(f"Existing self-contained file has different size: {destination}")
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        os.symlink(source, destination)
    else:
        raise ValueError(f"Unknown weight materialization mode: {mode}")
    print(f"[SELF-CONTAINED] {mode}: {destination}")
    return destination.resolve()


def _write_resume_script(destination: Path, args) -> None:
    script_path = destination / "resume_same_stage.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "${RUN_DIR}/code/refined_modeir"',
        'source ~/.venvs/dsdm_np1/bin/activate',
        "python train.py \\",
        '  --config "${RUN_DIR}/configs/v2-inference.yaml" \\',
        '  --base-checkpoint "${RUN_DIR}/weights/512-base-ema.ckpt" \\',
        '  --stage1-checkpoint "${RUN_DIR}/weights/ckpt_stage1.pt" \\',
        '  --router-checkpoint "${RUN_DIR}/weights/ckpt_router.pt" \\',
        '  --stage2-checkpoint "${RUN_DIR}/weights/ckpt_stage2.pt" \\',
        '  --daclip-checkpoint "${RUN_DIR}/weights/daclip_ViT-B-32.pt" \\',
        '  --resume "${RUN_DIR}/refined_last.pt" \\',
        '  --resume-optimizer \\',
        '  --output-dir "${RUN_DIR}" \\',
        f"  --dataset-root {shlex.quote(str(args.dataset_root))} \\",
        f"  --train-mode {shlex.quote(str(args.train_mode))} \\",
        f"  --router-backbone {shlex.quote(str(args.router_backbone))} \\",
        f"  --top-s {int(args.top_s)} \\",
        f"  --train-size {int(args.train_size)} \\",
        f"  --batch-size {int(args.batch_size)} \\",
        f"  --num-workers {int(args.num_workers)} \\",
        f"  --target-per-task {int(args.target_per_task)} \\",
        f"  --epochs {int(args.epochs)} \\",
        f"  --save-every {int(args.save_every)} \\",
        f"  --vis-every {int(args.vis_every)} \\",
        f"  --vis-num {int(args.vis_num)} \\",
        f"  --oracle-every {int(args.oracle_every)} \\",
        f"  --window {int(args.window)} \\",
        f"  --expert-sampling {shlex.quote(str(args.expert_sampling))} \\",
        f"  --latent-mode {shlex.quote(str(args.latent_mode))} \\",
        f"  --lr-experts {float(args.lr_experts)} \\",
        f"  --lr-tscm {float(args.lr_tscm)} \\",
        f"  --lr-injectors {float(args.lr_injectors)} \\",
        f"  --lr-router {float(args.lr_router)} \\",
        f"  --lr-frm {float(args.lr_frm)} \\",
        f"  --lr-latent-fusion {float(args.lr_latent_fusion)} \\",
        f"  --w-latent {float(args.w_latent)} \\",
        f"  --w-hf {float(args.w_hf)} \\",
        f"  --w-adv {float(args.w_adv)} \\",
        f"  --w-router {float(args.w_router)} \\",
        f"  --w-severity {float(args.w_severity)} \\",
        f"  --w-balance {float(args.w_balance)} \\",
        f"  --w-encoder {float(args.w_encoder)} \\",
        f"  --w-router-cls {float(args.w_router_cls)} \\",
        "  --self-contained-output",
    ]
    for flag in (
        "freeze_experts",
        "freeze_encoder_lora",
        "freeze_tscm",
        "freeze_injectors",
        "freeze_router",
        "freeze_latent_fusion",
        "freeze_frm",
        "router_gumbel_topk",
    ):
        if getattr(args, flag):
            lines[-1] += " \\"
            lines.append(f"  --{flag.replace('_', '-')}")
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def prepare_self_contained_output(args) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.self_contained_output:
        args.output_dir = str(output_dir)
        return output_dir

    config_path = _materialize_file(args.config, output_dir / "configs" / Path(args.config).name, "copy")
    weight_paths = {
        "base_checkpoint": _materialize_file(args.base_checkpoint, output_dir / "weights" / "512-base-ema.ckpt", args.weight_materialization),
        "stage1_checkpoint": _materialize_file(args.stage1_checkpoint, output_dir / "weights" / "ckpt_stage1.pt", args.weight_materialization),
        "router_checkpoint": _materialize_file(args.router_checkpoint, output_dir / "weights" / "ckpt_router.pt", args.weight_materialization),
        "stage2_checkpoint": _materialize_file(args.stage2_checkpoint, output_dir / "weights" / "ckpt_stage2.pt", args.weight_materialization),
        "daclip_checkpoint": _materialize_file(args.daclip_checkpoint, output_dir / "weights" / "daclip_ViT-B-32.pt", args.weight_materialization),
    }
    _copy_code_snapshot(output_dir)

    args.output_dir = str(output_dir)
    args.config = str(config_path)
    args.base_checkpoint = str(weight_paths["base_checkpoint"])
    args.stage1_checkpoint = str(weight_paths["stage1_checkpoint"])
    args.router_checkpoint = str(weight_paths["router_checkpoint"])
    args.stage2_checkpoint = str(weight_paths["stage2_checkpoint"])
    args.daclip_checkpoint = str(weight_paths["daclip_checkpoint"])

    manifest = {
        "format": "refined_modeir_self_contained_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "argv": sys.argv,
        "cwd": str(Path.cwd().resolve()),
        "code_snapshot": str((output_dir / "code" / "refined_modeir").resolve()),
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "note": "Training images are not copied; dataset_root remains an external dataset path.",
        "args": vars(args),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_resume_script(output_dir, args)
    return output_dir


def configure_optimizer(model, args):
    encoder_lora_params = [p for name, p in model.vae.encoder.named_parameters() if "lora_" in name]
    expert_params = [p for name, p in model.unet.named_parameters() if "lora_" in name]
    router_params = [
        p
        for name, p in model.router.named_parameters()
        if not (name.startswith("encoder.") and getattr(model.router.encoder, "frozen_backbone", False))
    ]
    if getattr(model.router.encoder, "frozen_backbone", False):
        freeze(model.router.encoder)
    groups = []

    mode_frozen = {
        "encoder_lora": {
            "encoder_lora": False,
            "experts": True,
            "tscm": True,
            "injectors": True,
            "router": True,
            "latent_fusion": True,
            "frm": True,
        },
        "router_classifier": {
            "encoder_lora": True,
            "experts": True,
            "tscm": True,
            "injectors": True,
            "router": False,
            "latent_fusion": True,
            "frm": True,
        },
        "tscm_unet_lora_router_time": {
            "encoder_lora": True,
            "experts": False,
            "tscm": False,
            "injectors": False,
            "router": True,
            "latent_fusion": True,
            "frm": True,
        },
        "fusion_s1": {
            "encoder_lora": True,
            "experts": True,
            "tscm": True,
            "injectors": True,
            "router": True,
            "latent_fusion": False,
            "frm": False,
        },
        "paper_stage2": {
            "encoder_lora": True,
            "experts": True,
            "tscm": True,
            "injectors": True,
            "router": False,
            "latent_fusion": False,
            "frm": False,
        },
        "router": {
            "encoder_lora": True,
            "experts": True,
            "tscm": True,
            "injectors": True,
            "router": False,
            "latent_fusion": True,
            "frm": True,
        },
        "experts": {
            "encoder_lora": True,
            "experts": False,
            "tscm": True,
            "injectors": True,
            "router": True,
            "latent_fusion": True,
            "frm": True,
        },
        "tscm_decoder": {
            "encoder_lora": True,
            "experts": True,
            "tscm": False,
            "injectors": False,
            "router": True,
            "latent_fusion": False,
            "frm": False,
        },
        "router_tscm_decoder": {
            "encoder_lora": True,
            "experts": True,
            "tscm": False,
            "injectors": False,
            "router": False,
            "latent_fusion": False,
            "frm": False,
        },
        "joint": {
            "encoder_lora": True,
            "experts": False,
            "tscm": False,
            "injectors": False,
            "router": False,
            "latent_fusion": False,
            "frm": False,
        },
    }[args.train_mode]

    def frozen(component: str) -> bool:
        return bool(mode_frozen[component]) or bool(getattr(args, f"freeze_{component}"))

    def add(label, params, lr, frozen):
        params = unique_params(params)
        for parameter in params:
            parameter.requires_grad_(not frozen)
        params = [parameter for parameter in params if parameter.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "name": label})
            print(f"[TRAIN] {label}: tensors={len(params)} params={sum(p.numel() for p in params):,} lr={lr:g}")

    print(f"[TRAIN] mode={args.train_mode}")
    add("encoder_lora", encoder_lora_params, args.lr_experts, frozen("encoder_lora"))
    add("expert_lora", expert_params, args.lr_experts, frozen("experts"))
    add("tscm", list(model.tscm.parameters()), args.lr_tscm, frozen("tscm"))
    add("static_injectors", list(model.adapter.blocks.parameters()), args.lr_injectors, frozen("injectors"))
    add("router", router_params, args.lr_router, frozen("router"))
    add("latent_fusion", list(model.latent_fusion.parameters()), args.lr_latent_fusion, frozen("latent_fusion"))
    frm_up = list(model.vae.decoder.frm_up.parameters())
    add("frm_up", frm_up, args.lr_frm, frozen("frm"))
    freeze(model.vae.decoder.frm_mid32)
    if not groups:
        raise RuntimeError("All trainable components are frozen")
    return torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=1e-4)


def wrap_model_for_ddp(model, args, local_rank: int):
    if not dist.is_available() or not dist.is_initialized():
        return model

    kwargs = {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "find_unused_parameters": bool(args.ddp_find_unused),
    }

    def wrap_if_trainable(module):
        if any(parameter.requires_grad for parameter in module.parameters()):
            return DDP(module, **kwargs)
        return module

    if any(parameter.requires_grad for parameter in model.vae.encoder.parameters()):
        model.vae.encoder = wrap_if_trainable(model.vae.encoder)
    model.tscm = wrap_if_trainable(model.tscm)
    model.adapter = wrap_if_trainable(model.adapter)
    model.router = wrap_if_trainable(model.router)
    model.latent_fusion = wrap_if_trainable(model.latent_fusion)
    for index, block in enumerate(model.vae.decoder.frm_up):
        model.vae.decoder.frm_up[index] = wrap_if_trainable(block)
    rank0_print(f"[DDP] wrapped trainable modules on local_rank={local_rank}")
    return model


def compact_unet_lora(model):
    return {key: value.detach().cpu() for key, value in unwrap_module(model.unet).state_dict().items() if "lora_" in key}


def compact_encoder_lora(model):
    return {key: value.detach().cpu() for key, value in unwrap_module(model.vae.encoder).state_dict().items() if "lora_" in key}


def compact_router_state(model):
    router = unwrap_module(model.router)
    state = router.state_dict()
    if getattr(router.encoder, "skip_checkpoint_state", False):
        state = {key: value for key, value in state.items() if not key.startswith("encoder.")}
    return {key: value.detach().cpu() for key, value in state.items()}


def decoder_refine_state_unwrapped(decoder):
    return {
        "frm_mid32": cpu_state(unwrap_module(decoder.frm_mid32)),
        "frm_up": [cpu_state(unwrap_module(block)) for block in decoder.frm_up],
    }


def save_checkpoint(path: Path, model, optimizer, args, step: int, epoch: int):
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "refined_modeir_v1",
            "step": step,
            "epoch": epoch,
            "args": vars(args),
            "t_list": model.t_list,
            "vae_encoder_lora": compact_encoder_lora(model),
            "unet_lora": compact_unet_lora(model),
            "tscm": cpu_state(unwrap_module(model.tscm)),
            "static_injectors": cpu_state(unwrap_module(model.adapter).blocks),
            "router": compact_router_state(model),
            "latent_fusion": cpu_state(unwrap_module(model.latent_fusion)),
            "decoder_refine": decoder_refine_state_unwrapped(model.vae.decoder),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"[SAVE] {path}")


def _image01(x: torch.Tensor) -> torch.Tensor:
    return ((x.detach().float().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)


def _gather_for_rank0(x: torch.Tensor) -> torch.Tensor | None:
    x = x.detach().contiguous()
    if not dist.is_available() or not dist.is_initialized():
        return x
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x)
    if dist.get_rank() == 0:
        return torch.cat(gathered, dim=0)
    return None


@torch.no_grad()
def save_stage1_visualization(model, degraded: torch.Tensor, target: torch.Tensor, args, step: int, output_dir: Path) -> None:
    if int(args.vis_every) <= 0 or step % int(args.vis_every) != 0:
        return
    with model.autocast():
        z_vis, _ = model.encode_degraded(degraded, args.train_size)
        decoded = model.vae.decode(z_vis.float() / model.scale_factor, enc_feats=None, t_val=None).float()

    degraded_all = _gather_for_rank0(_image01(degraded))
    decoded_all = _gather_for_rank0(_image01(decoded))
    target_all = _gather_for_rank0(_image01(target))
    if not is_main_process():
        return
    if degraded_all is None or decoded_all is None or target_all is None:
        return

    total = degraded_all.shape[0]
    count = min(int(args.vis_num), total)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + int(step))
    indices = torch.randperm(total, generator=generator)[:count]
    rows = []
    for index in indices.tolist():
        error = (decoded_all[index] - target_all[index]).abs().clamp(0, 1)
        rows.extend([degraded_all[index], decoded_all[index], target_all[index], error])

    vis_dir = Path(args.vis_dir).expanduser() if args.vis_dir else output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    path = vis_dir / f"stage1_step_{step:08d}.png"
    vutils.save_image(torch.stack(rows, dim=0), path, nrow=4, padding=2)
    print(f"[VIS] {path} columns=degraded,decoded_z,target,error samples={count}")


def overlay_resume(model, optimizer, path: str, args):
    if not path:
        return 0, 0
    payload = load_payload(path)
    if "vae_encoder_lora" in payload:
        model.vae.encoder.load_state_dict(payload["vae_encoder_lora"], strict=False)
    else:
        print("[RESUME] checkpoint has no vae_encoder_lora state; keep initialized encoder LoRA")
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
    payload_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    payload_mode = payload_args.get("train_mode")
    stage_switch = bool(payload_mode and payload_mode != args.train_mode)
    if args.resume_optimizer and not stage_switch and optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    elif args.resume_optimizer and stage_switch:
        print(f"[RESUME] skip optimizer because train mode changes: {payload_mode} -> {args.train_mode}")
    print(f"[RESUME] loaded refined checkpoint: {path}")
    if stage_switch:
        return 0, 0
    return int(payload.get("step", 0)), int(payload.get("epoch", 0))


def choose_experts_for_expert_training(args, batch_size: int, step: int, num_experts: int, device: torch.device):
    if args.expert_sampling == "random":
        return torch.randint(num_experts, (batch_size,), device=device)
    start = int(step) % int(num_experts)
    return (torch.arange(batch_size, device=device) + start).remainder(num_experts).long()


def latent_difference_timestep_labels(z_base: torch.Tensor, z_gt: torch.Tensor, num_experts: int) -> torch.Tensor:
    diff = (z_base.detach() - z_gt.detach()).abs().flatten(1).mean(1)
    if diff.numel() == 1:
        return torch.zeros(1, device=diff.device, dtype=torch.long)
    order = diff.argsort()
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(diff.numel(), device=diff.device)
    labels = torch.div(ranks * int(num_experts), diff.numel(), rounding_mode="floor")
    return labels.clamp_(0, int(num_experts) - 1).long()


@torch.no_grad()
def smoke_test(model, image_size: int, latent_mode: str):
    x = torch.randn(1, 3, image_size, image_size, device=model.device).clamp(-1, 1)
    with model.autocast():
        z_base, enc_feats = model.encode_degraded(x, image_size)
        ctx = model.conditioning(1)
        route = model.router(x)
        weights, selected = model.router.route_topk(route["logits"], route["probs"])
        z0, t_val = model.restore_weighted(x, z_base, ctx, selected, weights, latent_mode)
        restored = model.decode(z0, enc_feats, t_val)
    print(f"[DRY-RUN] selected={selected.tolist()} weights={weights.tolist()} z={tuple(z0.shape)} image={tuple(restored.shape)}")
    print(f"[DRY-RUN] static injection stats={model.adapter.last_stats}")


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size = init_distributed()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stable Diffusion backend")
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    output_dir = prepare_self_contained_output(args)
    model = build_refined_model(
        config_path=args.config,
        base_checkpoint=args.base_checkpoint,
        stage1_checkpoint=args.stage1_checkpoint,
        router_checkpoint=args.router_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        device=device,
        preferred_latent_hw=args.train_size // 8,
        window=args.window,
        router_backbone=args.router_backbone,
        daclip_checkpoint=args.daclip_checkpoint,
        top_s=args.top_s,
    )
    optimizer = configure_optimizer(model, args)
    start_step, start_epoch = overlay_resume(model, optimizer, args.resume, args)
    model = wrap_model_for_ddp(model, args, local_rank)
    if args.dry_run:
        smoke_test(model, args.train_size, args.latent_mode)
        if distributed:
            dist.destroy_process_group()
        return

    dataset = AlignedDataset(
        root=args.dataset_root, mode="train", train_size=args.train_size,
        target_per_task=args.target_per_task,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    loader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    ctx_cache = model.conditioning(args.batch_size)
    step = start_step

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for degraded, target in loader:
            degraded = degraded.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.train_mode == "encoder_lora":
                with model.autocast():
                    z_base, enc_feats = model.encode_degraded(degraded, args.train_size)
                with torch.no_grad(), model.autocast():
                    z_gt = model.encode_target(target)
            else:
                with torch.no_grad(), model.autocast():
                    z_base, enc_feats = model.encode_degraded(degraded, args.train_size)
                    z_gt = model.encode_target(target)

            zero = torch.zeros((), device=device)
            loss_image = zero
            loss_latent = zero
            loss_hf = zero
            loss_adv = zero
            loss_router = zero
            loss_severity = zero
            loss_balance = zero

            route = None
            selected = None
            weights = None
            train_router = args.train_mode in {"paper_stage2", "router", "router_tscm_decoder", "joint"} and not args.freeze_router

            if args.train_mode == "encoder_lora":
                loss_latent = F.l1_loss(z_base, z_gt)
                loss = float(args.w_encoder) * loss_latent
                selected = torch.empty(degraded.shape[0], 0, device=device, dtype=torch.long)
            elif args.train_mode == "router_classifier":
                route = model.router(degraded)
                oracle = latent_difference_timestep_labels(z_base, z_gt, len(model.t_list))
                loss_router = F.cross_entropy(route["logits"], oracle)
                loss_severity = model.router.severity_loss(route, degraded, target)
                if float(args.w_balance) > 0.0:
                    loss_balance = model.router.load_balance_loss(route["probs"])
                loss = (
                    float(args.w_router_cls) * loss_router
                    + float(args.w_severity) * loss_severity
                    + float(args.w_balance) * loss_balance
                )
                selected = oracle.view(-1, 1)
            elif args.train_mode == "experts":
                selected_1d = choose_experts_for_expert_training(
                    args,
                    batch_size=degraded.shape[0],
                    step=step,
                    num_experts=len(model.t_list),
                    device=device,
                )
                z0, t_val = model.restore_selected(degraded, z_base, ctx_cache, selected_1d, args.latent_mode)
                selected = selected_1d.view(-1, 1)
            elif args.train_mode == "router":
                route = model.router(degraded)
                weights, selected = model.router.route_topk(
                    route["logits"],
                    route["probs"],
                    training=True,
                    gumbel_noise=args.router_gumbel_topk,
                )
                oracle = model.oracle_targets(degraded, z_base, z_gt, ctx_cache, args.latent_mode)
                loss_router = F.cross_entropy(route["logits"], oracle)
                loss_balance = model.router.load_balance_loss(route["probs"])
                loss_severity = model.router.severity_loss(route, degraded, target)
                loss = args.w_router * loss_router + args.w_severity * loss_severity + args.w_balance * loss_balance
            elif args.train_mode == "tscm_unet_lora_router_time":
                with torch.no_grad():
                    route = model.router(degraded)
                    selected_1d = route["logits"].argmax(dim=1)
                z0, t_val = model.restore_selected(degraded, z_base, ctx_cache, selected_1d, args.latent_mode)
                selected = selected_1d.view(-1, 1)
            elif args.train_mode == "fusion_s1":
                with torch.no_grad():
                    route = model.router(degraded)
                    selected = route["logits"].argmax(dim=1, keepdim=True)
                    weights = torch.ones(selected.shape, device=device, dtype=z_base.dtype)
                    z0, t_val = model.restore_weighted(degraded, z_base, ctx_cache, selected, weights, args.latent_mode)
            else:
                if train_router:
                    route = model.router(degraded)
                    weights, selected = model.router.route_topk(
                        route["logits"],
                        route["probs"],
                        training=True,
                        gumbel_noise=args.router_gumbel_topk,
                    )
                else:
                    with torch.no_grad():
                        route = model.router(degraded)
                        weights, selected = model.router.route_topk(route["logits"], route["probs"])
                z0, t_val = model.restore_weighted(degraded, z_base, ctx_cache, selected, weights, args.latent_mode)

            if args.train_mode not in {"encoder_lora", "router", "router_classifier"}:
                with model.autocast():
                    restored = model.decode(z0, enc_feats, t_val)
                    loss_image = F.l1_loss(restored, target)
                    loss_latent = F.l1_loss(z0, z_gt)
                    loss_hf = high_frequency_l1(restored, target)
                    loss_adv = gan_generator_loss(model.critic(critic_features(restored, model.critic_in_channels)))
                    if route is not None and train_router:
                        if float(args.w_balance) > 0.0:
                            loss_balance = model.router.load_balance_loss(route["probs"])
                        if float(args.w_router) > 0.0 and args.oracle_every > 0 and step % args.oracle_every == 0:
                            oracle = model.oracle_targets(degraded, z_base, z_gt, ctx_cache, args.latent_mode)
                            loss_router = F.cross_entropy(route["logits"], oracle)
                        loss_severity = model.router.severity_loss(route, degraded, target)
                    loss = (
                        loss_image + args.w_latent * loss_latent + args.w_hf * loss_hf + args.w_adv * loss_adv +
                        args.w_router * loss_router + args.w_severity * loss_severity + args.w_balance * loss_balance
                    )
            loss.backward()
            optimizer.step()
            step += 1
            if args.train_mode == "encoder_lora":
                save_stage1_visualization(model, degraded, target, args, step, output_dir)
            if is_main_process() and (step == 1 or step % 10 == 0):
                print(
                    f"[STEP {step}] total={loss.item():.5f} img={loss_image.item():.5f} "
                    f"latent={loss_latent.item():.5f} hf={loss_hf.item():.5f} "
                    f"router={loss_router.item():.5f} sev={loss_severity.item():.5f} "
                    f"selected={selected.tolist()}"
                )
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(output_dir / f"refined_step_{step:08d}.pt", model, optimizer, args, step, epoch)
            if args.max_steps > 0 and step >= args.max_steps:
                if not args.skip_final_save:
                    save_checkpoint(output_dir / "refined_last.pt", model, optimizer, args, step, epoch)
                if distributed:
                    dist.barrier()
                    dist.destroy_process_group()
                return
    if not args.skip_final_save:
        save_checkpoint(output_dir / "refined_last.pt", model, optimizer, args, step, args.epochs)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
