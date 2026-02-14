#!/usr/bin/env bash
set -euo pipefail

required_fields=(
  "wait_count="
  "alloc_count_host="
  "alloc_count_device="
  "alloc_count_shared="
  "pool_hit_count="
  "pool_miss_count="
  "cross_domain_transfer_count="
  "transfer_bytes_h2d="
  "transfer_bytes_d2h="
  "host_alloc_call_count="
  "host_alloc_bytes="
)

require_fields() {
  local line="$1"
  for f in "${required_fields[@]}"; do
    if [[ "$line" != *"$f"* ]]; then
      echo "missing required field '$f' in stats line: $line" >&2
      return 1
    fi
  done
}

if [[ "${1:-}" == "--self-check" ]]; then
  sample="[SYCL-OFFLOAD-STATS] tag=graph_compute device=0 wait_count=1 alloc_count_host=2 alloc_count_device=0 alloc_count_shared=0 pool_hit_count=3 pool_miss_count=4 cross_domain_transfer_count=5 transfer_bytes_h2d=6 transfer_bytes_d2h=7 host_alloc_call_count=8 host_alloc_bytes=9"
  require_fields "$sample"
  echo "self-check passed"
  exit 0
fi

if [[ -z "${SYCL_OFFLOAD_BENCH_CMD:-}" ]]; then
  cat >&2 <<'EOF'
SYCL_OFFLOAD_BENCH_CMD is required for live run.
Example:
  SYCL_OFFLOAD_BENCH_CMD="ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:cpu' GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_OFFLOAD_STATS=1 GGML_SYCL_VRAM_BUDGET_PCT=35 ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 64 -n 32 --tg-batch 4 -ngl 99 -fa 0"

Optional:
  SYCL_OFFLOAD_BENCH_ITERATIONS=2      # repeat run to catch intermittent crashes
  SYCL_OFFLOAD_BENCH_TIMEOUT_SEC=180   # timeout per iteration
  SYCL_OFFLOAD_REQUIRE_TG_LINE=1       # require tg* result row in bench output
EOF
  exit 2
fi

iterations="${SYCL_OFFLOAD_BENCH_ITERATIONS:-1}"
if ! [[ "$iterations" =~ ^[0-9]+$ ]] || (( iterations < 1 )); then
  echo "SYCL_OFFLOAD_BENCH_ITERATIONS must be a positive integer" >&2
  exit 2
fi

timeout_sec="${SYCL_OFFLOAD_BENCH_TIMEOUT_SEC:-0}"
if ! [[ "$timeout_sec" =~ ^[0-9]+$ ]]; then
  echo "SYCL_OFFLOAD_BENCH_TIMEOUT_SEC must be a non-negative integer" >&2
  exit 2
fi

tmp_log="$(mktemp)"
trap 'rm -f "$tmp_log"' EXIT

for ((iter = 1; iter <= iterations; iter++)); do
  echo "running [$iter/$iterations]: $SYCL_OFFLOAD_BENCH_CMD"
  if (( timeout_sec > 0 )); then
    if ! timeout --preserve-status "${timeout_sec}s" bash -lc "$SYCL_OFFLOAD_BENCH_CMD" 2>&1 | tee -a "$tmp_log"; then
      echo "bench command failed on iteration $iter" >&2
      exit 1
    fi
  else
    if ! bash -lc "$SYCL_OFFLOAD_BENCH_CMD" 2>&1 | tee -a "$tmp_log"; then
      echo "bench command failed on iteration $iter" >&2
      exit 1
    fi
  fi
done

stats_line="$(grep -m1 '\[SYCL-OFFLOAD-STATS\]' "$tmp_log" || true)"
if [[ -z "$stats_line" ]]; then
  echo "no [SYCL-OFFLOAD-STATS] line found" >&2
  exit 1
fi

require_fields "$stats_line"

if [[ "${SYCL_OFFLOAD_REQUIRE_TG_LINE:-0}" != "0" ]]; then
  if ! grep -Eq '\|\s*tg[0-9]+' "$tmp_log"; then
    echo "missing tg* benchmark row in output log" >&2
    exit 1
  fi
fi

echo "offload regression check passed"
