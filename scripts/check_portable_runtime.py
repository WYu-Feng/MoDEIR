#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeir_refined.bootstrap import PROJECT_ROOT, VENDOR_ROOT, bootstrap_legacy_backend  # noqa: E402


def check_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    print(f"[OK] {label}: {path}")


def main() -> None:
    print(f"[INFO] refined root: {ROOT}")
    print(f"[INFO] backend root: {PROJECT_ROOT}")
    print(f"[INFO] vendor root: {VENDOR_ROOT}")
    bootstrap_legacy_backend()

    check_path(ROOT / "configs" / "v2-inference.yaml", "config")
    check_path(ROOT / "weights" / "legacy" / "512-base-ema.ckpt", "base checkpoint")
    check_path(ROOT / "weights" / "legacy" / "ckpt_stage1.pt", "stage1 checkpoint")
    check_path(ROOT / "weights" / "legacy" / "ckpt_router.pt", "router checkpoint")
    check_path(ROOT / "weights" / "legacy" / "ckpt_stage2.pt", "stage2 checkpoint")
    check_path(ROOT / "weights" / "legacy" / "daclip_ViT-B-32.pt", "DA-CLIP checkpoint")
    check_path(PROJECT_ROOT / "ldm", "ldm backend")
    check_path(PROJECT_ROOT / "universal_dataset.py", "universal_dataset.py")
    check_path(PROJECT_ROOT / "da-clip" / "src" / "open_clip", "DA-CLIP open_clip package")
    check_path(PROJECT_ROOT / "CLIP-ViT-H-14-laion2B-s32B-b79K", "OpenCLIP text encoder assets")

    importlib.import_module("ldm.util")
    importlib.import_module("universal_dataset")
    importlib.import_module("modeir_refined.pipeline")
    print("[OK] python imports resolved")


if __name__ == "__main__":
    main()
