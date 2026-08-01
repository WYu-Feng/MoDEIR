#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source ~/.venvs/dsdm_np1/bin/activate

if [[ -z "${DATASET_ROOT:-}" ]]; then
  for candidate in \
    "${ROOT}/ceshi/datasets" \
    "/home/Baseline/ceshi/datasets" \
    "/home/ceshi/datasets" \
    "/root/ceshi/datasets" \
    "/home/Baseline/datasets"; do
    if [[ -d "${candidate}" ]]; then
      DATASET_ROOT="${candidate}"
      break
    fi
  done
fi
DATASET_ROOT="${DATASET_ROOT:-/home/Baseline/ceshi/datasets}"
export DATASET_ROOT
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/extreme_staged}"
TRAIN_SIZE="${TRAIN_SIZE:-192}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TARGET_PER_TASK="${TARGET_PER_TASK:-10000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
VIS_EVERY="${VIS_EVERY:-2000}"
VIS_NUM="${VIS_NUM:-5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29577}"

ENCODER_STEPS="${ENCODER_STEPS:-20000}"
ROUTER_STEPS="${ROUTER_STEPS:-20000}"
TSCM_UNET_STEPS="${TSCM_UNET_STEPS:-50000}"
FUSION_STEPS="${FUSION_STEPS:-30000}"

mkdir -p "${RUN_ROOT}"

common_args=(
  --no-self-contained-output
  --dataset-root "${DATASET_ROOT}"
  --train-size "${TRAIN_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --target-per-task "${TARGET_PER_TASK}"
  --save-every "${SAVE_EVERY}"
  --vis-every "${VIS_EVERY}"
  --vis-num "${VIS_NUM}"
  --latent-mode strict_delta
  --router-backbone daclip
  --top-s 2
)

run_ddp=(
  torchrun
  --standalone
  --nnodes 1
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  train.py
)

echo "[STAGE 1] VAE encoder LoRA -> high-quality degraded latent Z"
echo "[DATA] root=${DATASET_ROOT}"
python - <<'PY'
from pathlib import Path
import glob
import os

root = Path(os.environ["DATASET_ROOT"])
required = [
    "Image deblurring/GoPro/test",
    "Image deblurring/GoPro/train",
    "Image dehazing/SOTS/outdoor",
    "Image denoise/BSD400",
    "Image denoise/BSD68",
    "Image deraining/RainTrainL",
    "Image deraining/Rain100L",
    "Low-light enhancement/LOL/eval",
    "Low-light enhancement/LOL/train",
]
missing = [p for p in required if not (root / p).is_dir()]
print(f"[DATA] root exists={root.is_dir()} path={root}")
for rel in required:
    count = len(glob.glob(str(root / rel / "**" / "*.png"), recursive=True))
    print(f"[DATA] {rel}: png={count}")
if missing:
    raise SystemExit(f"Missing required dataset folders: {missing}")
PY

"${run_ddp[@]}" \
  "${common_args[@]}" \
  --train-mode encoder_lora \
  --output-dir "${RUN_ROOT}/stage1_encoder_lora" \
  --max-steps "${ENCODER_STEPS}" \
  --lr-experts 5e-6 \
  --w-encoder 1.0 \
  --w-latent 0.0 \
  --w-hf 0.0 \
  --w-adv 0.0

echo "[STAGE 2] TAR classifier from |Z_deg - Z_gt| timestep bins"
"${run_ddp[@]}" \
  "${common_args[@]}" \
  --train-mode router_classifier \
  --resume "${RUN_ROOT}/stage1_encoder_lora/refined_last.pt" \
  --output-dir "${RUN_ROOT}/stage2_router_classifier" \
  --max-steps "${ROUTER_STEPS}" \
  --lr-router 1e-4 \
  --w-router-cls 1.0 \
  --w-severity 0.05 \
  --w-balance 0.01

echo "[STAGE 3] TSCM + UNet expert LoRA with frozen router top1 time"
"${run_ddp[@]}" \
  "${common_args[@]}" \
  --train-mode tscm_unet_lora_router_time \
  --resume "${RUN_ROOT}/stage2_router_classifier/refined_last.pt" \
  --output-dir "${RUN_ROOT}/stage3_tscm_unet_lora" \
  --max-steps "${TSCM_UNET_STEPS}" \
  --lr-experts 5e-6 \
  --lr-tscm 5e-6 \
  --lr-injectors 5e-6 \
  --w-latent 0.5 \
  --w-hf 0.05 \
  --w-adv 0.05

echo "[STAGE 4] final latent fusion + FRM with s=1"
"${run_ddp[@]}" \
  "${common_args[@]}" \
  --train-mode fusion_s1 \
  --resume "${RUN_ROOT}/stage3_tscm_unet_lora/refined_last.pt" \
  --output-dir "${RUN_ROOT}/stage4_fusion_s1" \
  --max-steps "${FUSION_STEPS}" \
  --top-s 1 \
  --lr-latent-fusion 2e-5 \
  --lr-frm 2e-5 \
  --w-latent 0.0 \
  --w-hf 0.05 \
  --w-adv 0.2

echo "[DONE] Final checkpoint: ${RUN_ROOT}/stage4_fusion_s1/refined_last.pt"
