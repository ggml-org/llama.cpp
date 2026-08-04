#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build-moe-mma-probe}"

mkdir -p "${OUT_DIR}"

"${NVCC}" \
    -std=c++17 \
    -O3 \
    -lineinfo \
    --generate-code=arch=compute_120a,code=[sm_120a] \
    "${ROOT_DIR}/scripts/blackwell-moe-prefill/moe_mma_probe.cu" \
    -o "${OUT_DIR}/moe-mma-probe"

for blocks in 256 512 1024; do
    "${OUT_DIR}/moe-mma-probe" "${blocks}" 128 9
done

"${NVCC%/nvcc}/cuobjdump" --dump-resource-usage "${OUT_DIR}/moe-mma-probe"
