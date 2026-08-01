from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Create a one-pair overfit dataset from the Baseline dataset tree.")
    parser.add_argument("--dataset-root", default="/home/Baseline/datasets")
    parser.add_argument("--output-root", default="validation_datasets_baseline_overfit")
    parser.add_argument("--image-size", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    src_deg_dir = dataset_root / "Image dehazing" / "SOTS" / "outdoor" / "hazy"
    src_gt_dir = dataset_root / "Image dehazing" / "SOTS" / "outdoor" / "gt"
    src_deg = next(src_deg_dir.glob("*.png"))
    src_gt = src_gt_dir / src_deg.name
    if not src_gt.is_file():
        raise FileNotFoundError(src_gt)

    project_root = Path(__file__).resolve().parents[1]
    out = project_root / args.output_root / "Image dehazing" / "SOTS" / "outdoor"
    (out / "hazy").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)

    for src, dst in (
        (src_deg, out / "hazy" / "000001.png"),
        (src_gt, out / "gt" / "000001.png"),
    ):
        image = Image.open(src).convert("RGB").resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        image.save(dst)
        print(dst)


if __name__ == "__main__":
    main()
