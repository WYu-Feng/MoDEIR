#!/usr/bin/env bash
set -euo pipefail

new_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "${new_root}/.." && pwd)"
vendor_root="${new_root}/vendor"
mode="${1:-hardlink}"

if [[ "${mode}" != "hardlink" && "${mode}" != "copy" && "${mode}" != "symlink" ]]; then
  echo "usage: bash scripts/prepare_portable_vendor.sh [hardlink|copy|symlink]" >&2
  exit 2
fi

mkdir -p "${vendor_root}"

place_file() {
  local source="$1"
  local target="$2"
  if [[ ! -f "${source}" ]]; then
    echo "missing file: ${source}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target}")"
  if [[ -e "${target}" ]]; then
    echo "keep existing: ${target}"
    return
  fi
  case "${mode}" in
    hardlink) ln "${source}" "${target}" 2>/dev/null || cp -a "${source}" "${target}" ;;
    copy) cp -a "${source}" "${target}" ;;
    symlink) ln -s "${source}" "${target}" ;;
  esac
  echo "${mode}: ${target}"
}

place_dir() {
  local source="$1"
  local target="$2"
  if [[ ! -d "${source}" ]]; then
    echo "missing directory: ${source}" >&2
    exit 1
  fi
  if [[ -e "${target}" ]]; then
    echo "keep existing: ${target}"
    return
  fi
  mkdir -p "$(dirname "${target}")"
  case "${mode}" in
    hardlink) cp -al "${source}" "${target}" 2>/dev/null || cp -a "${source}" "${target}" ;;
    copy) cp -a "${source}" "${target}" ;;
    symlink) ln -s "${source}" "${target}" ;;
  esac
  echo "${mode}: ${target}"
}

place_dir "${project_root}/ldm" "${vendor_root}/ldm"
place_file "${project_root}/universal_dataset.py" "${vendor_root}/universal_dataset.py"
mkdir -p "${vendor_root}/da-clip/src"
place_dir "${project_root}/da-clip/src/open_clip" "${vendor_root}/da-clip/src/open_clip"
place_dir "${project_root}/CLIP-ViT-H-14-laion2B-s32B-b79K" "${vendor_root}/CLIP-ViT-H-14-laion2B-s32B-b79K"

cat > "${vendor_root}/PORTABLE_VENDOR_MANIFEST.txt" <<EOF
This directory contains external runtime assets needed after copying refined_modeir
to a new server:

- ldm/
- universal_dataset.py
- da-clip/src/open_clip/
- CLIP-ViT-H-14-laion2B-s32B-b79K/

Created by scripts/prepare_portable_vendor.sh using mode=${mode}.
EOF

echo "portable vendor is ready: ${vendor_root}"
