from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "modeir_paper_aligned" / "validation_datasets_overfit" / "Image dehazing" / "SOTS" / "outdoor"


def main() -> None:
    (OUT / "hazy").mkdir(parents=True, exist_ok=True)
    (OUT / "gt").mkdir(parents=True, exist_ok=True)
    src_deg = next((ROOT / "dehaz" / "deg").glob("*.png"))
    src_gt = ROOT / "dehaz" / "gt" / src_deg.name
    for src, dst in (
        (src_deg, OUT / "hazy" / "000001.png"),
        (src_gt, OUT / "gt" / "000001.png"),
    ):
        image = Image.open(src).convert("RGB").resize((192, 192), Image.Resampling.BICUBIC)
        image.save(dst)
        print(dst)


if __name__ == "__main__":
    main()
