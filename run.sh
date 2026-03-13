#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build-rel/bin/llama-cli"
MODEL="${ROOT}/models/Fairy-plus-minus-i-700M/ifairy.gguf"

COMMON=( -m "${MODEL}" --gpu-layers 0 -t 4 -b 1 --seed 1 -p "I believe life is" -n 512 -no-cnv )

run_case() {
  local name="$1"
  shift
  echo
  echo "==== ${name} ===="
  "$@" "${COMMON[@]}" 2>&1 | tee "/tmp/ifairy_lut_${name}.log" | grep -E "tok/s|eval time|prompt eval time|sampling time" || true
}

run_case "lut0" env GGML_IFAIRY_LUT=0 "${BIN}"
run_case "lut1" env GGML_IFAIRY_LUT=1 "${BIN}"
run_case "lut1_fullacc" env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_FULLACC=1 "${BIN}"
run_case "lut1_bk2_fullacc" env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=2 GGML_IFAIRY_LUT_FULLACC=1 "${BIN}"