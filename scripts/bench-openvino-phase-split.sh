#!/usr/bin/env bash
# End-to-end -pg 512,128 for OpenVINO phase split vs single device.
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 model.gguf" >&2
  exit 1
fi

# Caller must: source "$HOME/intel/openvino/setupvars.sh"

BENCH="${BENCH:-$HOME/src/llama/build_oneapi_master_openvino_RelWithDebInfo/bin/llama-bench}"
MODEL="$1"
TAG="$(basename "$MODEL" .gguf)"
CACHE_ROOT="${CACHE_ROOT:-/tmp/ov_e2e_bench}"
OUT="${OUT:-/tmp/ov_e2e_${TAG}.jsonl}"

if [[ ! -x "$BENCH" ]]; then
  echo "error: llama-bench not found: $BENCH" >&2
  exit 1
fi

bench_pg() {
  local name="$1"
  shift
  export GGML_OPENVINO_CACHE_DIR="${CACHE_ROOT}/${TAG}/${name}"
  mkdir -p "$GGML_OPENVINO_CACHE_DIR"
  env "$@" "$BENCH" -m "$MODEL" -r 3 -o jsonl -pg 512,128 | tail -1
}

echo "{\"model\":\"$TAG\"}"
while read -r name vars; do
  # shellcheck disable=SC2086
  line=$(bench_pg "$name" $vars)
  echo "$line" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({'config':'$name','avg_ts':d['avg_ts'],'avg_ns':d['avg_ns']}))"
done <<'EOF'
all_cpu GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=0
all_igpu GGML_OPENVINO_DEVICE=GPU.0 GGML_OPENVINO_STATEFUL_EXECUTION=1
split_cpu_pp_igpu_tg GGML_OPENVINO_PHASE_SPLIT=1 GGML_OPENVINO_PREFILL_DEVICE=CPU GGML_OPENVINO_DECODE_DEVICE=GPU.0 GGML_OPENVINO_STATEFUL_EXECUTION=1
split_igpu_pp_cpu_tg GGML_OPENVINO_PHASE_SPLIT=1 GGML_OPENVINO_PREFILL_DEVICE=GPU.0 GGML_OPENVINO_DECODE_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1
EOF
