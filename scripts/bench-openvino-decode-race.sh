#!/usr/bin/env bash
# Compare -pg 512,128: all CPU, all iGPU, phase split, decode race.
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 model.gguf" >&2
  exit 1
fi

# Caller must: source "\$HOME/intel/openvino/setupvars.sh"

BENCH="${BENCH:-$HOME/src/llama/build_oneapi_master_openvino_RelWithDebInfo/bin/llama-bench}"
MODEL="$1"
export GGML_OPENVINO_CACHE_DIR="${CACHE_ROOT:-/tmp/ov_decode_race_bench}"

if [[ ! -x "$BENCH" ]]; then
  echo "error: llama-bench not found: $BENCH" >&2
  exit 1
fi

run() {
  local name=$1
  shift
  echo "--- $name ---"
  env "$@" "$BENCH" -m "$MODEL" -r 3 --no-warmup -pg 512,128 2>&1 | awk -F'|' '/pp512\+tg128/ {print $NF}'
}

run all_cpu GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_DECODE_RACE=0 GGML_OPENVINO_PHASE_SPLIT=0
run all_igpu GGML_OPENVINO_DEVICE=GPU.0 GGML_OPENVINO_STATEFUL_EXECUTION=1 GGML_OPENVINO_DECODE_RACE=0 GGML_OPENVINO_PHASE_SPLIT=0
run split GGML_OPENVINO_PHASE_SPLIT=1 GGML_OPENVINO_PREFILL_DEVICE=CPU GGML_OPENVINO_DECODE_DEVICE=GPU.0 GGML_OPENVINO_STATEFUL_EXECUTION=1 GGML_OPENVINO_DECODE_RACE=0
run decode_race GGML_OPENVINO_DECODE_RACE=1 GGML_OPENVINO_PREFILL_DEVICE=CPU GGML_OPENVINO_RACE_CPU_DEVICE=CPU GGML_OPENVINO_RACE_GPU_DEVICE=GPU.0 GGML_OPENVINO_STATEFUL_EXECUTION=0
