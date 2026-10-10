#!/usr/bin/env bash
# Quiet-host llama-bench of the installed package with GGML_SYCL_FA_LARGE_GRF off/on, interleaved
# off,on,off,on. Waits for load < 4 for 3 consecutive minutes (max 2 h) BEFORE stopping the service,
# warms the model up once, then runs all four back to back and restarts the service.
set -uo pipefail
readonly UNIT=llama-gpu@Ornith-1.5-35B-Q4_K_M.service OUT=/mnt/nvme1/oneapi-ab/grfbench
readonly MODEL=/mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf
quiet=0
for ((m = 0; m < 120; m++)); do
  if awk -v l="$(cut -d' ' -f1 /proc/loadavg)" 'BEGIN{exit !(l < 4)}'; then quiet=$((quiet + 1)); else quiet=0; fi
  [[ $quiet -ge 3 ]] && break
  sleep 60
done
echo "quiet wait done $(date +%T) quiet=$quiet load=$(cut -d' ' -f1-3 /proc/loadavg)"
trap 'ssh -o BatchMode=yes vinbonesjr "systemctl start $UNIT"; echo "service: $(systemctl is-active $UNIT) $(date +%T)"' EXIT
ssh -o BatchMode=yes vinbonesjr "systemctl stop $UNIT"
sleep 5
echo "gpu users after stop: $(fuser /dev/dri/renderD129 2>/dev/null | wc -w)"
bench() { # tag knob reps tests...
  local tag=$1 knob=$2 reps=$3; shift 3
  mkdir -p "$OUT/$tag"
  echo "== $tag knob=$knob start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) top: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,3p | tr -s ' ' | tr '\n' ';')"
  ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300 \
  GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 GGML_SYCL_FA_LARGE_GRF=$knob \
    timeout 3600 /usr/bin/llama-bench -m "$MODEL" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 -t 12 -r "$reps" -o md "$@" \
    > "$OUT/$tag/bench.md" 2> "$OUT/$tag/bench.err"
  echo "   exit=$? end $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) :: $(rg -o 'FA_LARGE_GRF: [0-9]+ \([^)]*\)' "$OUT/$tag/bench.err" | head -1)"
}
bench warmup 0 1 -p 512 -n 0 -d 0
for r in 1 2; do
  bench "off$r" 0 5 -p 128,512 -n 64 -d 0,8192
  bench "on$r"  1 5 -p 128,512 -n 64 -d 0,8192
done
