#!/usr/bin/env bash
# Production-placement rerun (--moe-cache off, as the service pins it): GRF knob off/on interleaved x2,
# then plain decode NEW vs OLD build. Host not quiet (Hindsight postgres); conditions logged per run.
set -uo pipefail
readonly UNIT=llama-gpu@Ornith-1.5-35B-Q4_K_M.service OUT=/mnt/nvme1/oneapi-ab/grfbench-moeoff
readonly MODEL=/mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf OLD=/mnt/nvme1/oneapi-ab/decode/old-4e7400c3a
readonly DEADLINE=1350
trap 'ssh -o BatchMode=yes vinbonesjr "systemctl start $UNIT"; echo "service: $(systemctl is-active $UNIT) $(date +%T)"' EXIT
bench() { # tag build knob reps tests...
  local tag=$1 build=$2 knob=$3 reps=$4; shift 4
  if (( SECONDS > DEADLINE )); then echo "== $tag SKIPPED (deadline)"; return; fi
  mkdir -p "$OUT/$tag"
  echo "== $tag build=$build knob=$knob start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) idle=$(vmstat 3 2 | tail -1 | awk '{print $15}')%"
  local common=(ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300)
  local args=(-m "$MODEL" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 -t 12 --moe-cache off -r "$reps" -o md "$@")
  if [[ $build == new ]]; then
    env "${common[@]}" GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 GGML_SYCL_FA_LARGE_GRF=$knob timeout 900 /usr/bin/llama-bench "${args[@]}" > "$OUT/$tag/bench.md" 2> "$OUT/$tag/bench.err"
  else
    env "${common[@]}" LD_LIBRARY_PATH=$OLD/usr/lib timeout 900 $OLD/usr/bin/llama-bench "${args[@]}" > "$OUT/$tag/bench.md" 2> "$OUT/$tag/bench.err"
  fi
  echo "   exit=$? end $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) idle=$(vmstat 3 2 | tail -1 | awk '{print $15}')% :: $(rg -o 'FA_LARGE_GRF: [0-9]+ \([^)]*\)' "$OUT/$tag/bench.err" | head -1) $(rg -o 'moe-cache[^|]*' "$OUT/$tag/bench.md" | head -1)"
}
ssh -o BatchMode=yes vinbonesjr "systemctl stop $UNIT"; sleep 5
echo "gpu users after stop: $(fuser /dev/dri/renderD129 2>/dev/null | wc -w)"
bench warmup new 0 1 -p 512 -n 0 -d 0
bench off1 new 0 5 -p 128,512 -n 64 -d 0,8192
bench on1  new 1 5 -p 128,512 -n 64 -d 0,8192
bench off2 new 0 5 -p 128,512 -n 64 -d 0,8192
bench on2  new 1 5 -p 128,512 -n 64 -d 0,8192
bench dec-new new 1 5 -p 0 -n 64 -d 0
bench dec-old old 0 5 -p 0 -n 64 -d 0
echo "done $(date +%T) elapsed ${SECONDS}s"
