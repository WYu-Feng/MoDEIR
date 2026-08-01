#!/usr/bin/env bash
set -euo pipefail

new_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parent_root="$(cd "${new_root}/.." && pwd)"
archive_path="${1:-${parent_root}/refined_modeir_portable.tar.gz}"
include_datasets="${INCLUDE_DATASETS:-0}"

if [[ ! -d "${new_root}/vendor/ldm" ]]; then
  echo "vendor is not prepared yet; running scripts/prepare_portable_vendor.sh" >&2
  bash "${new_root}/scripts/prepare_portable_vendor.sh" hardlink
fi

excludes=(
  "--exclude=refined_modeir/__pycache__"
  "--exclude=refined_modeir/*/__pycache__"
  "--exclude=refined_modeir/*/*/__pycache__"
  "--exclude=refined_modeir/test_outputs"
  "--exclude=refined_modeir/analysis_outputs"
  "--exclude=refined_modeir/outputs"
)

if [[ "${include_datasets}" != "1" ]]; then
  excludes+=("--exclude=refined_modeir/datasets")
fi

mkdir -p "$(dirname "${archive_path}")"
tar "${excludes[@]}" -czf "${archive_path}" -C "${parent_root}" refined_modeir
echo "wrote portable archive: ${archive_path}"
if [[ "${include_datasets}" != "1" ]]; then
  echo "datasets were excluded. On the new server, pass --dataset-root /path/to/datasets for training."
fi
