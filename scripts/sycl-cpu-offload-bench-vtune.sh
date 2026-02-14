#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/sycl-cpu-offload-bench-vtune.sh --model /path/model.gguf [options]

Options:
  --model PATH                 GGUF model path (required)
  --profile NAME               Profile preset (default: nonstream-cpuoffload)
  --bin-dir PATH               Binary directory (default: ./build-sycl/bin if present, else ./build/bin)
  --pp TOKENS                  PP prompt tokens (default: 512)
  --tg TOKENS                  TG generation tokens (default: 128)
  --tg-batch N                 TG batch size (default: 16)
  --repeat N                   Benchmark repeats (default: 3)
  --vram-pct N                 GGML_SYCL_VRAM_BUDGET_PCT (profile default: 25)
  --selector STR               ONEAPI_DEVICE_SELECTOR (profile default: level_zero:0;opencl:cpu)
  --device N                   GGML_SYCL_DEVICE (profile default: 0)
  --result-root PATH           VTune result root (default: /tmp/vtune_sycl_cpu_offload_YYYYmmdd_HHMMSS)
  --max-memalloc-host-calls N  Fail if zeMemAllocHost calls > N (0 disables)
  --max-event-sync-calls N     Fail if zeEventHostSynchronize calls > N (0 disables)
  --max-set-tensor-cpu-ms N    Fail if set_tensor CPU time (ms) > N (0 disables)
  --no-vtune                   Skip VTune collection
  -h, --help                   Show this help

Notes:
  - Runs PP and TG separately.
  - Profile `nonstream-cpuoffload` keeps `-ngl` policy untouched and exercises unified-cache offload.
USAGE
}

MODEL=""
PROFILE="nonstream-cpuoffload"
BIN_DIR=""
PP_TOKENS=512
TG_TOKENS=128
TG_BATCH=16
REPEAT=3
VRAM_PCT=""
SELECTOR=""
SYCL_DEVICE=""
RUN_VTUNE=1
RESULT_ROOT=""

MAX_MEMALLOC_HOST_CALLS=0
MAX_EVENT_SYNC_CALLS=0
MAX_SET_TENSOR_CPU_MS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --bin-dir) BIN_DIR="${2:-}"; shift 2 ;;
    --pp) PP_TOKENS="${2:-}"; shift 2 ;;
    --tg) TG_TOKENS="${2:-}"; shift 2 ;;
    --tg-batch) TG_BATCH="${2:-}"; shift 2 ;;
    --repeat) REPEAT="${2:-}"; shift 2 ;;
    --vram-pct) VRAM_PCT="${2:-}"; shift 2 ;;
    --selector) SELECTOR="${2:-}"; shift 2 ;;
    --device) SYCL_DEVICE="${2:-}"; shift 2 ;;
    --result-root) RESULT_ROOT="${2:-}"; shift 2 ;;
    --max-memalloc-host-calls) MAX_MEMALLOC_HOST_CALLS="${2:-}"; shift 2 ;;
    --max-event-sync-calls) MAX_EVENT_SYNC_CALLS="${2:-}"; shift 2 ;;
    --max-set-tensor-cpu-ms) MAX_SET_TENSOR_CPU_MS="${2:-}"; shift 2 ;;
    --no-vtune) RUN_VTUNE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  usage
  exit 2
fi

if [[ "$PROFILE" == "nonstream-cpuoffload" ]]; then
  : "${SELECTOR:=level_zero:0;opencl:cpu}"
  : "${SYCL_DEVICE:=0}"
  : "${VRAM_PCT:=25}"
else
  : "${SELECTOR:=level_zero:0;opencl:cpu}"
  : "${SYCL_DEVICE:=0}"
  : "${VRAM_PCT:=25}"
fi

if [[ -z "$BIN_DIR" ]]; then
  if [[ -x ./build-sycl/bin/llama-bench ]]; then
    BIN_DIR=./build-sycl/bin
  else
    BIN_DIR=./build/bin
  fi
fi

if [[ ! -x "$BIN_DIR/llama-bench" ]]; then
  echo "llama-bench not found in $BIN_DIR" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model file not found: $MODEL" >&2
  exit 1
fi

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  tmp_env="$(mktemp)"
  if /bin/bash -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && env -0" >"$tmp_env" 2>/dev/null; then
    while IFS= read -r -d '' kv; do
      case "$kv" in
        PATH=*|LD_LIBRARY_PATH=*|LIBRARY_PATH=*|CPATH=*|PKG_CONFIG_PATH=*|CMAKE_PREFIX_PATH=*)
          export "$kv"
          ;;
      esac
    done <"$tmp_env"
  fi
  rm -f "$tmp_env"
fi

export ONEAPI_DEVICE_SELECTOR="$SELECTOR"
export GGML_SYCL_DEVICE="$SYCL_DEVICE"
export GGML_SYCL_UNIFIED_CACHE=1
export GGML_SYCL_CPU_OFFLOAD=1
export GGML_SYCL_CPU_OFFLOAD_ASYNC=1
export GGML_SYCL_CPU_BATCH_THRESHOLD_PP=4
export GGML_SYCL_CPU_BATCH_THRESHOLD_TG=16
export GGML_SYCL_VRAM_BUDGET_PCT="$VRAM_PCT"
export GGML_SYCL_OFFLOAD_STATS=1
export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"

STAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "$RESULT_ROOT" ]]; then
  RESULT_ROOT="/tmp/vtune_sycl_cpu_offload_${STAMP}"
fi
mkdir -p "$RESULT_ROOT"

PP_CMD=("$BIN_DIR/llama-bench" -m "$MODEL" -p "$PP_TOKENS" -n 0 --tg-batch "$TG_BATCH" -ngl 99 -fa 1 -r "$REPEAT")
TG_CMD=("$BIN_DIR/llama-bench" -m "$MODEL" -p 0 -n "$TG_TOKENS" --tg-batch "$TG_BATCH" -ngl 99 -fa 1 -r "$REPEAT")

pp_log="$RESULT_ROOT/pp.log"
tg_log="$RESULT_ROOT/tg.log"

echo "[cpu-offload] profile=$PROFILE selector=$ONEAPI_DEVICE_SELECTOR device=$GGML_SYCL_DEVICE vram_pct=$GGML_SYCL_VRAM_BUDGET_PCT"
echo "[cpu-offload] cmd(pp): ${PP_CMD[*]}"
"${PP_CMD[@]}" 2>&1 | tee "$pp_log"

echo "[cpu-offload] cmd(tg): ${TG_CMD[*]}"
"${TG_CMD[@]}" 2>&1 | tee "$tg_log"

parse_tps() {
  local log_file="$1"
  awk -F'|' '/\|[[:space:]]*(pp|tg)[0-9]+[[:space:]]*\|/ { gsub(/ /, "", $4); v=$4 } END { if (v != "") print v; }' "$log_file"
}

pp_tps="$(parse_tps "$pp_log" || true)"
tg_tps="$(parse_tps "$tg_log" || true)"
echo "[cpu-offload] pp_tps=${pp_tps:-na} tg_tps=${tg_tps:-na}"

memalloc_calls=""
event_sync_calls=""
set_tensor_cpu_ms=""

extract_first_int() {
  local s="$1"
  echo "$s" | grep -Eo '[0-9]+' | head -n1
}

if [[ "$RUN_VTUNE" -eq 1 ]]; then
  if ! command -v vtune >/dev/null 2>&1; then
    echo "vtune not found in PATH; rerun with --no-vtune or source oneAPI VTune env" >&2
    exit 1
  fi

  pp_hot="$RESULT_ROOT/pp_hotspots"
  tg_hot="$RESULT_ROOT/tg_hotspots"
  tg_offload="$RESULT_ROOT/tg_gpu_offload"

  echo "[cpu-offload] vtune hotspots (pp): $pp_hot"
  vtune -collect hotspots -result-dir "$pp_hot" -- "${PP_CMD[@]}" >/dev/null

  echo "[cpu-offload] vtune hotspots (tg): $tg_hot"
  vtune -collect hotspots -result-dir "$tg_hot" -- "${TG_CMD[@]}" >/dev/null

  echo "[cpu-offload] vtune gpu-offload (tg): $tg_offload"
  vtune -collect gpu-offload -result-dir "$tg_offload" -- "${TG_CMD[@]}" >/dev/null

  hotspots_csv="$RESULT_ROOT/tg_hotspots.csv"
  offload_csv="$RESULT_ROOT/tg_gpu_offload.csv"
  vtune -report hotspots -r "$tg_hot" -format csv > "$hotspots_csv" || true
  vtune -report gpu-offload -r "$tg_offload" -format csv > "$offload_csv" || true

  if grep -q "ggml_backend_sycl_buffer_set_tensor" "$hotspots_csv"; then
    line="$(grep -m1 "ggml_backend_sycl_buffer_set_tensor" "$hotspots_csv")"
    set_tensor_cpu_ms="$(extract_first_int "$line")"
  fi
  if grep -q "zeMemAllocHost" "$offload_csv"; then
    line="$(grep -m1 "zeMemAllocHost" "$offload_csv")"
    memalloc_calls="$(extract_first_int "$line")"
  fi
  if grep -q "zeEventHostSynchronize" "$offload_csv"; then
    line="$(grep -m1 "zeEventHostSynchronize" "$offload_csv")"
    event_sync_calls="$(extract_first_int "$line")"
  fi

  echo "[cpu-offload][vtune] zeMemAllocHost_calls=${memalloc_calls:-na}"
  echo "[cpu-offload][vtune] zeEventHostSynchronize_calls=${event_sync_calls:-na}"
  echo "[cpu-offload][vtune] ggml_backend_sycl_buffer_set_tensor_cpu_ms=${set_tensor_cpu_ms:-na}"
fi

fail=0
if [[ "$MAX_MEMALLOC_HOST_CALLS" -gt 0 && -n "$memalloc_calls" && "$memalloc_calls" -gt "$MAX_MEMALLOC_HOST_CALLS" ]]; then
  echo "[cpu-offload][gate] FAIL zeMemAllocHost calls $memalloc_calls > $MAX_MEMALLOC_HOST_CALLS" >&2
  fail=1
fi
if [[ "$MAX_EVENT_SYNC_CALLS" -gt 0 && -n "$event_sync_calls" && "$event_sync_calls" -gt "$MAX_EVENT_SYNC_CALLS" ]]; then
  echo "[cpu-offload][gate] FAIL zeEventHostSynchronize calls $event_sync_calls > $MAX_EVENT_SYNC_CALLS" >&2
  fail=1
fi
if [[ "$MAX_SET_TENSOR_CPU_MS" -gt 0 && -n "$set_tensor_cpu_ms" && "$set_tensor_cpu_ms" -gt "$MAX_SET_TENSOR_CPU_MS" ]]; then
  echo "[cpu-offload][gate] FAIL set_tensor cpu ms $set_tensor_cpu_ms > $MAX_SET_TENSOR_CPU_MS" >&2
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "[cpu-offload] results_root=$RESULT_ROOT"
