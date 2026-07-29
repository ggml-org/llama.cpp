#!/usr/bin/env bash
set -eo pipefail
HOST_LABEL="${1:?host label}"
shift
set +u
source "${OPENVINO_SETUP:-$HOME/intel/openvino/setupvars.sh}"
set -e
BENCH="${BENCH:-$HOME/src/llama/build_oneapi_master_openvino_RelWithDebInfo/bin/llama-bench}"
MODEL="${MODEL:-$HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
export GGML_OPENVINO_CACHE_DIR="${GGML_OPENVINO_CACHE_DIR:-/tmp/ov_phase0_xcheck_${HOST_LABEL}}"
export GGML_OPENVINO_PHASE_SPLIT=0
unset GGML_OPENVINO_PREFILL_DEVICE GGML_OPENVINO_DECODE_DEVICE 2>/dev/null || true

parse_ts() {
  awk -F'|' '
    /pp512/ && $0 !~ /test/ { gsub(/ /, "", $NF); split($NF, a, "+/-"); print a[1]; exit }
    /tg128/ && $0 !~ /test/ { gsub(/ /, "", $NF); split($NF, a, "+/-"); print a[1]; exit }
  '
}

run_dev() {
  local dev=$1 role=$2
  if [ "$dev" = CPU ]; then
    export GGML_OPENVINO_STATEFUL_EXECUTION=0
  else
    export GGML_OPENVINO_STATEFUL_EXECUTION=1
  fi
  export GGML_OPENVINO_DEVICE="$dev"
  echo "OpenVINO device $dev ($role) ..." >&2
  pp=$("$BENCH" -m "$MODEL" -r 3 -p 512 -n 0 2>&1 | parse_ts)
  tg=$("$BENCH" -m "$MODEL" -r 3 -p 0 -n 128 2>&1 | parse_ts)
  echo "${HOST_LABEL},${role},${dev},${pp},${tg}"
}

echo "host,role,ov_device,prefill_pp512_tps,decode_tg128_tps"
while [ $# -ge 2 ]; do
  run_dev "$1" "$2"
  shift 2
done
