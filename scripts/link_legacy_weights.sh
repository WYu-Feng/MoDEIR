#!/usr/bin/env bash
set -euo pipefail

new_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "${new_root}/.." && pwd)"
target="${new_root}/weights/legacy"
mkdir -p "${target}"

link_weight() {
  local source="$1"
  local name="$2"
  if [[ ! -f "${source}" ]]; then
    echo "missing source weight: ${source}" >&2
    exit 1
  fi
  if [[ -e "${target}/${name}" ]]; then
    echo "keep existing: ${target}/${name}"
    return
  fi
  ln "${source}" "${target}/${name}"
  echo "hard-linked: ${target}/${name}"
}

link_weight "${project_root}/checkpoint/512-base-ema.ckpt" "512-base-ema.ckpt"
link_weight "${project_root}/outputs/ckpt_stage1.pt" "ckpt_stage1.pt"
link_weight "${project_root}/outputs/ckpt_router.pt" "ckpt_router.pt"
link_weight "${project_root}/outputs/ckpt_stage2.pt" "ckpt_stage2.pt"
link_weight "${project_root}/pretrained/daclip_ViT-B-32.pt" "daclip_ViT-B-32.pt"
